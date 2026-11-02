"""Guarded inference-only stable top-five selection for compiled cosine search.

Only ranking changes: stable normalization, vendor matrix multiplication, query
validation, refresh and failed-refresh rollback retain the existing library path.
Absent, stale or unconfirmed plans use the complete compiled stable query.
No tuning, result cache, autodiff contract or performance claim is provided here.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Literal

from .search import CompiledCosineIndex, _compiled_search, _search_inputs, _search_results
from .workload import EPSILON, Decision, _check, environment_identity

FIXED_K = 5
KEY_POLICY = "native_ftz_stable_index_original_bits_v1"
RANK_CANDIDATES = ("simd32", "group128")
RANK_SOURCE_FILES = ("rank_key.metal", "rank_simd32.metal", "rank_group128.metal")
MAX_MSL_BYTES = 12 * 1024
_VARIANT_FILES = {"simd32": "rank_simd32.metal", "group128": "rank_group128.metal"}


def _read_msl(path: Path, expected_size: int | None = None) -> bytes:
    """Bound reads before opening, then reject concurrent size changes."""
    size = path.stat().st_size
    if size > MAX_MSL_BYTES or (expected_size is not None and size != expected_size):
        raise ValueError("Rank source size differs from the installed bounded source")
    with path.open("rb") as source:
        content = source.read(MAX_MSL_BYTES + 1)
    if len(content) != size:
        raise ValueError("Rank source changed while reading")
    return content


_SOURCES = tuple((name, _read_msl(Path(__file__).parent / "metal" / name))
                 for name in RANK_SOURCE_FILES)
if sum(len(content) for _, content in _SOURCES) > MAX_MSL_BYTES:
    raise ValueError("Authored rank MSL exceeds its 12 KiB budget")
_IDENTITY = hashlib.sha256(Path(__file__).read_bytes() + b"".join(
    name.encode() + content for name, content in _SOURCES)).hexdigest()


def rank_source_identity() -> str:
    """Identify the loaded adapter and immutable three-file MSL snapshot."""
    return _IDENTITY


def rank_implementation_identity() -> str:
    """Bind ranking to the existing application/baselines and strict study pin."""
    from .study_identity import implementation_identity
    pin = json.loads((Path(__file__).parent / "rank-study-source.json").read_text())
    script = pin["measurement_script_sha256"]
    if not isinstance(script, str) or len(script) != 64 or any(c not in "0123456789abcdef" for c in script):
        raise ValueError("Invalid rank study source identity")
    return hashlib.sha256(json.dumps({"base": implementation_identity(),
        "rank": rank_source_identity(), "study": script}, sort_keys=True).encode()).hexdigest()


def _bounded(shape: tuple[int, int, int, int]) -> bool:
    q, n, d, k = shape
    return (1 <= q <= 128 and 5 <= n <= 2048 and 1 <= d <= 2048
            and k == FIXED_K and q * n * d <= 16_777_216)


@lru_cache(maxsize=2)
def _kernel(variant: str) -> Any:
    import mlx.core as mx
    if variant not in RANK_CANDIDATES:
        raise ValueError("Unknown authored rank candidate")
    sources = dict(_SOURCES)
    return mx.fast.metal_kernel(name="kd_top5_" + variant,
        input_names=["scores"], output_names=["ids", "values"],
        source=sources[_VARIANT_FILES[variant]].decode(),
        header=sources["rank_key.metal"].decode(), ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"})


def candidate_rank(scores: Any, variant: str, *,
                   mode: Literal["inference"] = "inference") -> tuple[Any, Any]:
    """Lazy fixed top-five pairs for float32 scores[Q,N], Q1..128,N5..2048.

    Ties select unique original indices in ascending order. Native MLX float32
    comparison flushes subnormal keys to zero; signed zeros therefore also tie.
    Selected values retain their original bits, including zeros and subnormals.
    Actual signed MLX strides and offsets are used without mutating inputs.
    Any NaN/Inf makes that entire row UINT32_MAX IDs and quiet-NaN values, with
    no reads through sentinel IDs. High-level query rejection stays unchanged.
    """
    import mlx.core as mx
    if mode != "inference":
        raise ValueError("Only inference is supported")
    if variant not in RANK_CANDIDATES:
        raise ValueError("Unknown authored rank candidate")
    _check(scores)
    if scores.dtype != mx.float32:
        raise TypeError("Rank candidates require float32 inputs")
    q, n = scores.shape
    if not 1 <= q <= 128 or not 5 <= n <= 2048:
        raise ValueError("Rank candidate requires Q1..128,N5..2048,K5")
    threads = 32 if variant == "simd32" else 128
    group = min(q * 32, 128) if variant == "simd32" else 128
    with mx.stream(mx.gpu):
        ids, values = _kernel(variant)(inputs=[scores],
            template=[("N", n), ("K", FIXED_K)], grid=(q * threads, 1, 1),
            threadgroup=(group, 1, 1), output_shapes=[(q, FIXED_K), (q, FIXED_K)],
            output_dtypes=[mx.uint32, mx.float32])
    return ids, values


@lru_cache(maxsize=2)
def compiled_rank(variant: str) -> Any:
    """Cache a compiled rank graph; current scores are supplied every call."""
    import mlx.core as mx
    if variant not in RANK_CANDIDATES:
        raise ValueError("Unknown authored rank candidate")
    return mx.compile(lambda scores: candidate_rank(scores, variant))


@lru_cache(maxsize=2)
def compiled_query(variant: str) -> Any:
    """Compile the complete stable query, capturing only the rank variant."""
    import mlx.core as mx
    from .library_baselines import eager_normalize
    if variant not in RANK_CANDIDATES:
        raise ValueError("Unknown authored rank candidate")

    def graph(queries: Any, docs: Any) -> tuple[Any, Any, Any]:
        normalized = eager_normalize(queries)
        finite = mx.all(mx.isfinite(normalized))
        scores = normalized @ docs.T
        ids, values = candidate_rank(scores, variant)
        return ids, values, finite

    return mx.compile(graph)


@lru_cache(maxsize=1)
def library_rank() -> Any:
    """Compiled negative/full-sort/slice/gather baseline for finite scores."""
    import mlx.core as mx

    def graph(scores: Any) -> tuple[Any, Any]:
        ids = mx.argsort(-scores, axis=-1)[:, :FIXED_K]
        return ids, mx.take_along_axis(scores, ids, axis=-1)

    return mx.compile(graph)


def reduction_rank_formula(scores: Any) -> tuple[Any, Any]:
    """Library-only fixed-five reduction baseline for finite float32 scores."""
    import mlx.core as mx
    current = mx.where(mx.abs(scores) < 1.1754943508222875e-38, 0., scores)
    cols = mx.arange(scores.shape[1])[None, :]
    ids, values = [], []
    for _ in range(FIXED_K):
        chosen = mx.argmax(current, axis=-1)[:, None]
        ids.append(chosen)
        values.append(mx.take_along_axis(scores, chosen, axis=-1))
        current = mx.where(cols == chosen, -float("inf"), current)
    return mx.concatenate(ids, axis=1), mx.concatenate(values, axis=1)


@lru_cache(maxsize=1)
def library_reduction_rank() -> Any:
    import mlx.core as mx
    return mx.compile(reduction_rank_formula)


@lru_cache(maxsize=2)
def compiled_reduction_query(kind: str) -> Any:
    """Complete stable/fast normalization, vendor matmul and reduction query."""
    import mlx.core as mx
    from .library_baselines import eager_normalize, fast_formula
    if kind not in ("stable", "fast"):
        raise ValueError("Unknown library graph kind")
    formula = eager_normalize if kind == "stable" else fast_formula

    def graph(queries: Any, docs: Any) -> tuple[Any, Any, Any]:
        normalized = formula(queries)
        finite = mx.all(mx.isfinite(normalized))
        scores = normalized @ docs.T
        ids, values = reduction_rank_formula(scores)
        return ids, values, finite

    return mx.compile(graph)


class ReductionIndex(CompiledCosineIndex):
    """Matched full-query library alternative; inherited stable/fast refresh."""

    def search(self, queries: Any, k: int = FIXED_K) -> tuple[list[list[int]], list[list[float]]]:
        import mlx.core as mx
        _search_inputs(queries, self.vectors, k)
        if queries.shape[0] == 0:
            return [], []
        if k == FIXED_K and queries.dtype == self.vectors.dtype == mx.float32:
            return _search_results(compiled_reduction_query(self.kind)(queries, self.vectors))
        return _search_results(_compiled_search(self.kind, k)(queries, self.vectors))


class RankIndex(CompiledCosineIndex):
    """Stable compiled index with guarded optional top-five selection."""

    def __init__(self, vectors: Any, plan: RankPlan | None = None) -> None:
        self.plan = plan if plan is not None else RankPlan()
        super().__init__(vectors, "stable")

    def search(self, queries: Any, k: int = FIXED_K) -> tuple[list[list[int]], list[list[float]]]:
        _search_inputs(queries, self.vectors, k)
        if queries.shape[0] == 0:
            return [], []
        # Route/counters once per request, outside all compiled graphs.
        choice = self.plan._route(queries, self.vectors, k)
        graph = compiled_query(choice) if choice is not None else _compiled_search("stable", k)
        return _search_results(graph(queries, self.vectors))


class RankPlan:
    """Data-only, source/study/environment-bound exact [Q,N,D,K] portfolio.

    Public loading requires both numerical and complete-application confirmation.
    Artifact MSL can only equal the installed immutable authored snapshot; it is
    never compiled directly. Unknown tuples/dtypes/devices use the whole stable
    library query. There is no live tuning or cross-request result reuse.
    """

    def __init__(self, artifact: str | Path | None = None) -> None:
        self._rules: dict[tuple[int, int, int, int], str] = {}
        self.status = "library_only"
        self.calls = 0
        self.fallback_calls = 0
        self.last_decision = Decision("library", "no installed rank plan")
        if artifact is None:
            return
        try:
            path = Path(artifact)
            if path.is_dir():
                path = path / "manifest.json"
            if path.stat().st_size > 1_000_000:
                raise ValueError("Plan exceeds 1 MB")
            with path.open("rb") as manifest:
                data = manifest.read(1_000_001)
            if len(data) > 1_000_000:
                raise ValueError("Plan exceeds 1 MB")
            self._load(json.loads(data), confirmed=True)
            for name, content in _SOURCES:
                if _read_msl(path.parent / name, len(content)) != content:
                    raise ValueError("Artifact source differs from the installed authored kernel")
            self.status = "installed"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                importlib.metadata.PackageNotFoundError) as exc:
            self._rules.clear()
            self.status = "fallback: " + str(exc)

    def _load(self, plan: dict[str, Any], *, confirmed: bool) -> None:
        if not isinstance(plan, dict):
            raise ValueError("Rank plan must be an object")
        if (type(plan["schema"]) is not int or plan["schema"] != 1
                or plan["operation"] != "stable_top5_pairs_v1" or plan["epsilon"] != EPSILON
                or type(plan["fixed_k"]) is not int or plan["fixed_k"] != FIXED_K
                or plan["key_policy"] != KEY_POLICY
                or plan["compile_options"] != {"math_mode": "safe"}):
            raise ValueError("Incompatible rank operation")
        if (plan["source_sha256"] != rank_source_identity()
                or plan["study_identity"] != rank_implementation_identity()):
            raise ValueError("Rank source, application, baselines or study changed")
        if plan["environment"] != environment_identity():
            raise ValueError("Rank plan environment changed")
        evidence = plan["evidence"]
        if (not isinstance(evidence, dict) or evidence.get("numerical_passed") is not True
                or (confirmed and evidence.get("confirmation_passed") is not True)):
            raise ValueError("Unconfirmed rank plan")
        if plan["sources"] != {name: hashlib.sha256(content).hexdigest() for name, content in _SOURCES}:
            raise ValueError("Rank source hash mismatch")
        rules = plan["rules"]
        if not isinstance(rules, list) or len(rules) > 128:
            raise ValueError("Invalid rank dispatch rule count")
        loaded: dict[tuple[int, int, int, int], str] = {}
        for rule in rules:
            if not isinstance(rule, dict) or not isinstance(rule["shape"], list):
                raise ValueError("Invalid rank dispatch rule")
            shape = tuple(rule["shape"])
            if len(shape) != 4 or any(type(n) is not int for n in shape) or not _bounded(shape):
                raise ValueError("Invalid rank dispatch tuple")
            if (rule["candidate"] not in RANK_CANDIDATES or rule["dtype"] != "float32"
                    or rule["layout"] != "any_mlx_strides"):
                raise ValueError("Unsupported rank dispatch contract")
            if shape in loaded:
                raise ValueError("Duplicate rank dispatch tuple")
            loaded[shape] = rule["candidate"]
        if len(set(loaded.values())) > 2:
            raise ValueError("Rank portfolio exceeds two kernels")
        self._rules = loaded

    @classmethod
    def _trial(cls, manifest: dict[str, Any]) -> RankPlan:
        """Private numerical trial; never persists or labels a confirmed plan."""
        result = cls()
        result._load(manifest, confirmed=False)
        result.status = "numerical_trial"
        return result

    def _route(self, queries: Any, docs: Any, k: int) -> str | None:
        import mlx.core as mx
        _check(queries)
        _check(docs)
        if queries.shape[1] != docs.shape[1]:
            raise ValueError("Query and document widths must match")
        self.calls += 1
        shape = (queries.shape[0], docs.shape[0], queries.shape[1], k)
        choice = (self._rules.get(shape) if type(k) is int
                  and queries.dtype == docs.dtype == mx.float32
                  and mx.default_device() == mx.gpu else None)
        if choice is None:
            self.fallback_calls += 1
            self.last_decision = Decision("library", "Unfamiliar tuple/k/dtype/device or no compatible rank plan")
        else:
            self.last_decision = Decision(choice, "Frozen exact-tuple rule, actual MLX strides")
        return choice

    def index(self, vectors: Any, *, mode: Literal["inference"] = "inference") -> RankIndex:
        """Create an index retaining stable refresh and complete query fallback."""
        if mode != "inference":
            raise ValueError("Only inference is supported")
        return RankIndex(vectors, self)
