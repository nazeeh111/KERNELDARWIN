"""Bounded offline query-normalization/scoring fusion and static installation.

Document refresh uses the compiled stable MLX implementation. Unfamiliar query
tuples use the complete compiled library query, including ranking and validation.
There is no online search, answer cache, input mutation or autodiff contract.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Literal

from .workload import COMPILE_OPTIONS, EPSILON, Decision, _check, environment_identity
from .search import CompiledCosineIndex, _compiled_search, _search_inputs, _search_results

SCORE_CANDIDATES = ("score_simd", "score_tile4")
SCORE_SOURCE_FILES = tuple(name + ".metal" for name in SCORE_CANDIDATES)
_SOURCES = tuple((name, (Path(__file__).parent / "metal" / name).read_bytes()) for name in SCORE_SOURCE_FILES)
_IDENTITY = hashlib.sha256(Path(__file__).read_bytes() + b"".join(name.encode() + source for name, source in _SOURCES)).hexdigest()


def score_source_identity() -> str:
    """Identify exactly the loaded adapter and immutable authored MSL snapshot."""
    return _IDENTITY


def fused_implementation_identity() -> str:
    from .study_identity import implementation_identity
    pin = json.loads((Path(__file__).parent / "fused-study-source.json").read_text())
    script = pin["measurement_script_sha256"]
    if not isinstance(script, str) or len(script) != 64 or any(c not in "0123456789abcdef" for c in script):
        raise ValueError("Invalid fused study source identity")
    return hashlib.sha256(json.dumps({"base": implementation_identity(), "scores": score_source_identity(), "study": script}, sort_keys=True).encode()).hexdigest()


def _bounded(shape: tuple[int, int, int]) -> bool:
    q, n, d = shape
    return 1 <= q <= 128 and 1 <= n <= 2048 and 1 <= d <= 2048 and q * n * d <= 16_777_216


@lru_cache(maxsize=2)
def _kernel(variant: str) -> Any:
    import mlx.core as mx
    if variant not in SCORE_CANDIDATES:
        raise ValueError("Unknown authored score candidate")
    return mx.fast.metal_kernel(name="kd_" + variant, input_names=["queries", "docs"], output_names=["scores"],
        source=dict(_SOURCES)[variant + ".metal"].decode(), ensure_row_contiguous=False, compile_options=COMPILE_OPTIONS)


def candidate_scores(queries: Any, docs: Any, variant: str) -> Any:
    """Lazy float32 scores for raw queries and an already normalized index.

    Precondition: docs are finite, normalized document rows with abs(component)
    <=1, as produced and checked by FusedCosineIndex.refresh. This lower-level
    investigation function does not scan docs. Both inputs are nonempty rank-two
    float32 MLX arrays with equal width. All MLX strides and offsets are read
    directly; inputs are never changed. Zero queries are handled by the index.
    Query NaN/Inf yields an all-NaN score row. Native float32 subnormal flushing
    applies. Accumulation is float32 with safe Metal math and epsilon=1e-12.
    """
    import mlx.core as mx
    if variant not in SCORE_CANDIDATES:
        raise ValueError("Unknown authored score candidate")
    for value in (queries, docs):
        _check(value)
        if value.dtype != mx.float32:
            raise TypeError("Score candidates require float32 inputs")
    if queries.shape[1] != docs.shape[1]:
        raise ValueError("Query and document widths must match")
    q, d = queries.shape
    n = docs.shape[0]
    if not _bounded((q, n, d)):
        raise ValueError("Score candidate exceeds its bounded nonempty input contract")
    grid = (n * 32, q, 1) if variant == "score_simd" else (((n + 3) // 4) * 128, q, 1)
    with mx.stream(mx.gpu):
        return _kernel(variant)(inputs=[queries, docs], template=[("D", d)], grid=grid,
            threadgroup=(32 if variant == "score_simd" else 128, 1, 1),
            output_shapes=[(q, n)], output_dtypes=[mx.float32])[0]


@lru_cache(maxsize=2)
def _library_scores(kind: str) -> Any:
    import mlx.core as mx
    from .library_baselines import eager_normalize, fast_formula
    if kind not in ("stable", "fast"):
        raise ValueError("Unknown score library graph")
    formula = eager_normalize if kind == "stable" else fast_formula
    return mx.compile(lambda q, docs: formula(q) @ docs.T)


def library_scores(queries: Any, docs: Any, kind: str = "stable") -> Any:
    """Optimized compiled normalization plus matrix multiplication baseline."""
    import mlx.core as mx
    _check(queries); _check(docs)
    if queries.shape[1] != docs.shape[1]:
        raise ValueError("Query and document widths must match")
    if not queries.size or not docs.size:
        return mx.zeros((queries.shape[0], docs.shape[0]), dtype=mx.float32)
    return _library_scores(kind)(queries, docs)


@lru_cache(maxsize=32)
def _rank_scores(k: int) -> Any:
    import mlx.core as mx
    def graph(scores: Any) -> tuple[Any, Any, Any]:
        finite = mx.all(mx.isfinite(scores))
        ids = mx.argsort(-scores, axis=-1)[:, :k]
        return ids, mx.take_along_axis(scores, ids, axis=-1), finite
    return mx.compile(graph)


class FusedCosineIndex(CompiledCosineIndex):
    """Same installed search API, with optional fused query scoring.

    Refresh and failed-refresh rollback are inherited from the compiled stable
    library. A static plan's fallback executes the whole compiled stable query.
    """
    def __init__(self, vectors: Any, scorer: Any) -> None:
        self.scorer = scorer
        super().__init__(vectors, "stable")

    def search(self, queries: Any, k: int = 5) -> tuple[list[list[int]], list[list[float]]]:
        _search_inputs(queries, self.vectors, k)
        if queries.shape[0] == 0:
            return [], []
        if isinstance(self.scorer, FusedPlan):
            candidate = self.scorer._route(queries, self.vectors)
            if candidate is None:
                return _search_results(_compiled_search("stable", k)(queries, self.vectors))
            scores = candidate_scores(queries, self.vectors, candidate)
        else:
            scores = self.scorer(queries, self.vectors)
        return _search_results(_rank_scores(k)(scores))


class FusedPlan:
    """Data-only, source/device-bound exact-tuple score portfolio.

    Public installation requires numerical and full-application confirmation.
    Invalid/stale/absent artifacts retain the complete compiled MLX fallback.
    """
    def __init__(self, artifact: str | Path | None = None) -> None:
        self._rules: dict[tuple[int, int, int], str] = {}
        self.status = "library_only"
        self.calls = 0
        self.fallback_calls = 0
        self.last_decision = Decision("library", "no installed fused plan")
        if artifact is None:
            return
        try:
            path = Path(artifact)
            if path.is_dir():
                path = path / "manifest.json"
            if path.stat().st_size > 1_000_000:
                raise ValueError("Plan exceeds 1 MB")
            plan = json.loads(path.read_text())
            self._load(plan, confirmed=True)
            for name, content in _SOURCES:
                if (path.parent / name).read_bytes() != content:
                    raise ValueError("Artifact source differs from the installed authored kernel")
            self.status = "installed"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, importlib.metadata.PackageNotFoundError) as exc:
            self._rules.clear()
            self.status = "fallback: " + str(exc)

    def _load(self, plan: dict[str, Any], *, confirmed: bool) -> None:
        if plan["schema"] != 1 or plan["operation"] != "stable_l2_scores_v1" or plan["epsilon"] != EPSILON:
            raise ValueError("Incompatible fused operation")
        if plan["source_sha256"] != score_source_identity() or plan["study_identity"] != fused_implementation_identity():
            raise ValueError("Fused source, application, baselines or study changed")
        if plan["environment"] != environment_identity():
            raise ValueError("Fused plan environment changed")
        if plan["evidence"]["numerical_passed"] is not True or (confirmed and plan["evidence"]["confirmation_passed"] is not True):
            raise ValueError("Unconfirmed fused plan")
        if plan["sources"] != {name: hashlib.sha256(content).hexdigest() for name, content in _SOURCES}:
            raise ValueError("Fused source hash mismatch")
        if len(plan["rules"]) > 128:
            raise ValueError("Too many fused dispatch rules")
        for rule in plan["rules"]:
            shape = tuple(rule["shape"])
            if len(shape) != 3 or any(type(n) is not int for n in shape) or not _bounded(shape):
                raise ValueError("Invalid fused dispatch tuple")
            if rule["candidate"] not in SCORE_CANDIDATES or rule["dtype"] != "float32" or rule["layout"] != "any_mlx_strides":
                raise ValueError("Unsupported fused dispatch contract")
            if shape in self._rules:
                raise ValueError("Duplicate fused dispatch tuple")
            self._rules[shape] = rule["candidate"]
        if len(set(self._rules.values())) > 2:
            raise ValueError("Fused portfolio exceeds two kernels")

    @classmethod
    def _trial(cls, manifest: dict[str, Any]) -> FusedPlan:
        """Private offline measurement adapter. Never persists a confirmed plan."""
        result = cls()
        result._load(manifest, confirmed=False)
        result.status = "numerical_trial"
        return result

    def _route(self, queries: Any, docs: Any) -> str | None:
        import mlx.core as mx
        _check(queries); _check(docs)
        if queries.shape[1] != docs.shape[1]:
            raise ValueError("Query and document widths must match")
        self.calls += 1
        shape = (queries.shape[0], docs.shape[0], queries.shape[1])
        choice = self._rules.get(shape) if queries.dtype == docs.dtype == mx.float32 and mx.default_device() == mx.gpu else None
        if choice is None:
            self.fallback_calls += 1
            self.last_decision = Decision("library", "Unfamiliar tuple/dtype/device or no compatible fused plan")
        else:
            self.last_decision = Decision(choice, "Frozen exact-tuple rule, actual MLX strides")
        return choice

    def score(self, queries: Any, docs: Any, *, mode: Literal["inference"] = "inference") -> Any:
        """Lower-level scoring with the same normalized-document precondition."""
        if mode != "inference":
            raise ValueError("Only inference is supported")
        choice = self._route(queries, docs)
        return candidate_scores(queries, docs, choice) if choice else library_scores(queries, docs)

    def index(self, vectors: Any) -> FusedCosineIndex:
        """Create an application index with guarded static dispatch and fallback."""
        return FusedCosineIndex(vectors, self)
