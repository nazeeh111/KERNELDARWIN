"""Inference-only, stable L2 normalization and offline Metal plan installation.

No tuning, answer cache, model call or network access is performed at runtime.
Importing this module does not import MLX, so ordinary CPU installations work.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Literal

EPSILON = 1e-12
MAX_ELEMENTS = 16_777_216
CANDIDATES = ("simd", "simd4", "vector4", "group128")
SOURCE_FILES = ("normalize_simd.metal", "normalize_vector.metal", "normalize_group.metal", "vector_load.metal")
COMPILE_OPTIONS = {"math_mode": "safe"}
_LOADED_SOURCE = tuple((name, (Path(__file__).parent / "metal" / name).read_bytes()) for name in SOURCE_FILES)
_LOADED_IDENTITY = hashlib.sha256(
    Path(__file__).read_bytes() + b"".join(name.encode() + source for name, source in _LOADED_SOURCE)
).hexdigest()


def source_identity() -> str:
    """Bind decisions to the actual operation, adapter and readable MSL bytes."""
    # Compiled libraries and new shapes use this same immutable source snapshot.
    # Hot file replacement cannot relabel cached code. Restart to install changes.
    return _LOADED_IDENTITY


def environment_identity() -> dict[str, Any]:
    import mlx.core as mx
    info = mx.device_info()
    try:
        displays=json.loads(subprocess.check_output(['/usr/sbin/system_profiler','SPDisplaysDataType','-json'],text=True,timeout=10))
        os_build=subprocess.check_output(['/usr/bin/sw_vers','-buildVersion'],text=True,timeout=5).strip()
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f'OS/GPU identity probe unavailable: {exc}') from exc
    matches=[g for g in displays['SPDisplaysDataType'] if g.get('sppci_model')==info['device_name']]
    if len(matches)!=1 or not str(matches[0].get('sppci_cores','')).isdigit():
        raise RuntimeError('Cannot identify GPU core count; use the library fallback')
    return {
        "mlx": importlib.metadata.version("mlx"),
        "mlx_metal": importlib.metadata.version("mlx-metal"),
        "system": platform.system(), "os_version": platform.mac_ver()[0],
        "os_build": os_build,
        "kernel_release": platform.release(), "machine": platform.machine(),
        "python": platform.python_version(),
        "device": {k: info[k] for k in ("device_name", "architecture", "memory_size")},
        "gpu_cores":int(matches[0]['sppci_cores']),
        "compiler": "MLX runtime Metal JIT, bound to MLX and OS build",
        "compile_options": COMPILE_OPTIONS,
    }


def _check(x: Any) -> None:
    import mlx.core as mx
    if not isinstance(x, mx.array):
        raise TypeError("Expected an mlx.core.array; convert explicitly before calling")
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional row matrix; broadcasting is not inferred")
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise TypeError("Only float16, bfloat16 and float32 inputs are supported")


def library_normalize(x: Any) -> Any:
    """Stable float32 accumulation, clamped L2 denominator, original output dtype.

    Float32 input subnormals follow native Metal flush-to-zero behavior.
    All strides supported by MLX are supported. Zero-width/zero-row matrices
    return an empty array. Any NaN/Inf makes its entire output row NaN.
    Inputs are never mutated. No differentiation contract is provided.
    """
    import mlx.core as mx
    _check(x)
    if not x.size:
        return mx.zeros(x.shape, dtype=x.dtype)
    return _compiled_library()(x)


@lru_cache(maxsize=1)
def _compiled_library() -> Any:
    import mlx.core as mx

    def stable(x: Any) -> Any:
        z = x.astype(mx.float32)
        scale = mx.maximum(mx.max(mx.abs(z), axis=1, keepdims=True), EPSILON)
        z = z / scale
        denominator = mx.maximum(mx.sqrt(mx.sum(z * z, axis=1, keepdims=True)), EPSILON / scale)
        return (z / denominator).astype(x.dtype)

    return mx.compile(stable)


@lru_cache(maxsize=4)
def _kernel(name: str) -> Any:
    import mlx.core as mx
    if name not in CANDIDATES:
        raise ValueError("Unknown authored candidate")
    file = "normalize_group.metal" if name == "group128" else "normalize_vector.metal" if name == "vector4" else "normalize_simd.metal"
    return mx.fast.metal_kernel(
        name="kd_l2_" + name, input_names=["inp"], output_names=["out"],
        source=dict(_LOADED_SOURCE)[file].decode(),
        header=dict(_LOADED_SOURCE)["vector_load.metal"].decode() if name == "vector4" else "",
        ensure_row_contiguous=False, compile_options=COMPILE_OPTIONS,
    )


def candidate_normalize(x: Any, candidate: str) -> Any:
    """Bounded authored candidates for offline investigation; not a tuned API."""
    import mlx.core as mx
    _check(x)
    if candidate not in CANDIDATES:
        raise ValueError("Unknown authored candidate")
    if x.dtype != mx.float32 or not x.size or x.shape[1] > 2048 or x.size > MAX_ELEMENTS:
        raise ValueError("Candidate requires nonempty float32 rows <=2048 columns and <=16M elements")
    rows, width = x.shape
    threads = 128 if candidate in ("simd4", "vector4", "group128") else 32
    grid = rows * (128 if candidate == "group128" else 32)
    # A rounded grid guarantees whole SIMD groups. Guard rows are unnecessary:
    # MLX dispatchThreads launches exactly grid threads, including a partial group.
    with mx.stream(mx.gpu):
        return _kernel(candidate)(
            inputs=[x], template=[("D", width)], grid=(grid, 1, 1),
            threadgroup=(min(grid, threads), 1, 1),
            output_shapes=[x.shape], output_dtypes=[mx.float32],
        )[0]


@dataclass(frozen=True)
class Decision:
    implementation: str
    reason: str


class Normalizer:
    """Install an explicit static plan, or use the verified compiled MLX fallback.

    Unknown inputs/environments and malformed/stale plans use the fallback.
    Plans are data. They cannot name an executable, source path or import.
    Each shape match covers all MLX layouts; kernels index actual device strides.
    """

    def __init__(self, artifact: str | Path | None = None) -> None:
        self._rules: dict[tuple[int, int], str] = {}
        self.status = "library_only"
        self.calls = 0
        self.fallback_calls = 0
        self.last_decision = Decision("library", "no installed plan")
        if artifact is None:
            return
        try:
            path = Path(artifact)
            if path.is_dir():
                path = path / "manifest.json"
            if path.stat().st_size > 1_000_000:
                raise ValueError("plan exceeds 1 MB")
            plan = json.loads(path.read_text())
            if plan["schema"] != 1 or plan["operation"] != "stable_l2_rows_v1" or plan["epsilon"] != EPSILON:
                raise ValueError("incompatible operation")
            if plan["source_sha256"] != source_identity() or plan["environment"] != environment_identity():
                raise ValueError("source or environment changed")
            from .study_identity import implementation_identity
            if plan["study_identity"] != implementation_identity():
                raise ValueError("application, library baseline or study implementation changed")
            if plan["evidence"]["numerical_passed"] is not True or plan["evidence"]["confirmation_passed"] is not True:
                raise ValueError("unconfirmed plan")
            expected_sources = {name: hashlib.sha256(content).hexdigest() for name, content in _LOADED_SOURCE}
            if plan["sources"] != expected_sources:
                raise ValueError("manifest source differs from the installed authored kernel")
            for name, content in _LOADED_SOURCE:
                if (path.parent / name).read_bytes() != content:
                    raise ValueError("artifact source differs from the installed authored kernel")
            if len(plan["rules"]) > 128:
                raise ValueError("too many dispatch rules")
            for rule in plan["rules"]:
                rows, width = rule["shape"]
                if type(rows) is not int or type(width) is not int or not 0 < rows * width <= MAX_ELEMENTS or not 0 < width <= 2048:
                    raise ValueError("invalid dispatch shape")
                if rule["candidate"] not in CANDIDATES or rule["dtype"] != "float32" or rule["layout"] != "any_mlx_strides":
                    raise ValueError("unsupported dispatch contract")
                if (rows, width) in self._rules:
                    raise ValueError("duplicate dispatch shape")
                self._rules[(rows, width)] = rule["candidate"]
            if len(set(self._rules.values())) > 2:
                raise ValueError("portfolio exceeds two kernels")
            self.status = "installed"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, importlib.metadata.PackageNotFoundError) as exc:
            self._rules.clear()
            self.status = "fallback: " + str(exc)

    def __call__(self, x: Any, *, mode: Literal["inference"] = "inference") -> Any:
        import mlx.core as mx
        _check(x)
        if mode != "inference":
            raise ValueError("Only inference is supported; use MLX library operations for autodiff")
        self.calls += 1
        choice = self._rules.get(x.shape) if x.dtype == mx.float32 and mx.default_device() == mx.gpu else None
        if choice is None:
            self.fallback_calls += 1
            self.last_decision = Decision("library", "unfamiliar shape/dtype/device or no compatible plan")
            return library_normalize(x)
        self.last_decision = Decision(choice, "frozen exact-shape rule; all strides supported")
        return candidate_normalize(x, choice)
