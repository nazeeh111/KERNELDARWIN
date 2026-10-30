"""Local vector-search application, with an injectable row-normalization call.

The corpus demo uses token-count hashing, not a pretrained semantic model.
The same index accepts caller-provided embeddings without loading any weights.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
import hashlib
import re
from pathlib import Path
from typing import Any

from .workload import _check, library_normalize
_LOADED_APPLICATION_IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def text_vectors(texts: Sequence[str], width: int = 384) -> Any:
    """Deterministic, unsigned-count float32 token features. No downloaded data."""
    import mlx.core as mx
    if not 16 <= width <= 2048:
        raise ValueError("Feature width must be 16..2048")
    values = [[0.0] * width for _ in texts]
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            values[row][int.from_bytes(digest, "little") % width] += 1.0
    return mx.array(values, dtype=mx.float32) if texts else mx.zeros((0, width))


def _rank(queries: Any, vectors: Any, k: int) -> tuple[Any, Any, Any]:
    import mlx.core as mx
    finite=mx.all(mx.isfinite(queries))
    scores=queries @ vectors.T
    # Keep the same MLX sorting and tie policy for every implementation.
    ids=mx.argsort(-scores,axis=-1)[:, :k]
    values=mx.take_along_axis(scores,ids,axis=-1)
    return ids,values,finite


@lru_cache(maxsize=32)
def _compiled_search(kind: str, k: int) -> Any:
    """Cache graphs, never query contents or results."""
    import mlx.core as mx
    from .library_baselines import eager_normalize, fast_formula
    if kind not in ("normalized","stable","fast"):
        raise ValueError("Unknown library graph kind")
    def graph(queries: Any, vectors: Any) -> tuple[Any, Any, Any]:
        if kind=="stable":
            queries=eager_normalize(queries)
        elif kind=="fast":
            queries=fast_formula(queries)
        return _rank(queries,vectors,k)
    return mx.compile(graph)


def _search_inputs(queries: Any, vectors: Any, k: int) -> None:
    _check(queries)
    if queries.shape[1]!=vectors.shape[1]:
        raise ValueError("Query width must match index width")
    if type(k) is not int or not 1<=k<=vectors.shape[0]:
        raise ValueError("k must be between one and the index row count")


@lru_cache(maxsize=2)
def _compiled_refresh(kind: str) -> Any:
    import mlx.core as mx
    from .library_baselines import eager_normalize,fast_formula
    formula=eager_normalize if kind=='stable' else fast_formula
    def graph(vectors: Any) -> tuple[Any,Any]:
        normalized=formula(vectors)
        return normalized,mx.all(mx.isfinite(normalized))
    return mx.compile(graph)


def _search_results(result: tuple[Any, Any, Any]) -> tuple[list[list[int]], list[list[float]]]:
    import mlx.core as mx
    ids,values,finite=result
    mx.eval(ids,values,finite)
    if not finite.item():
        raise ValueError("Search query vectors must be finite")
    return ids.tolist(),values.tolist()


class CosineIndex:
    """Refresh an embedding index and return exact top-k cosine matches.

    Normalization, matrix multiplication, selection and host result materialization
    form the measured compute API. File I/O and text encoding are separate stages.
    """

    def __init__(self, vectors: Any, normalize: Callable[[Any], Any] = library_normalize) -> None:
        self.normalize = normalize
        self.refresh(vectors)

    def refresh(self, vectors: Any) -> None:
        import mlx.core as mx
        _check(vectors)
        if not vectors.size:
            raise ValueError("An index needs nonempty two-dimensional vectors")
        normalized = self.normalize(vectors)
        if not mx.all(mx.isfinite(normalized)).item():
            raise ValueError("Search index vectors must be finite")
        self.vectors = normalized

    def search(self, queries: Any, k: int = 5) -> tuple[list[list[int]], list[list[float]]]:
        _search_inputs(queries,self.vectors,k)
        if queries.shape[0]==0:return [],[]
        # Stay outside the compiled ranking graph so dispatch guards and call
        # counters run once for each request, including invalid finite values.
        q = self.normalize(queries)
        return _search_results(_compiled_search("normalized",k)(q,self.vectors))


class CompiledCosineIndex(CosineIndex):
    """Persist a normalized index and compile each complete query computation.

    A request includes normalization, finite validation, scoring and selection in
    one graph. Requests remain separate; index refresh and host results retain
    the same behavior as CosineIndex.
    """

    def __init__(self, vectors: Any, kind: str = "stable") -> None:
        from .library_baselines import eager_normalize, fast_formula
        if kind not in ("stable","fast"):
            raise ValueError("Library graph kind must be 'stable' or 'fast'")
        self.kind=kind
        super().__init__(vectors,eager_normalize if kind=="stable" else fast_formula)

    def search(self, queries: Any, k: int = 5) -> tuple[list[list[int]], list[list[float]]]:
        _search_inputs(queries,self.vectors,k)
        if queries.shape[0]==0:return [],[]
        return _search_results(_compiled_search(self.kind,k)(queries,self.vectors))

    def refresh(self, vectors: Any) -> None:
        import mlx.core as mx
        _check(vectors)
        if not vectors.size:raise ValueError('An index needs nonempty two-dimensional vectors')
        normalized,finite=_compiled_refresh(self.kind)(vectors)
        mx.eval(normalized,finite)
        if not finite.item():raise ValueError('Search index vectors must be finite')
        self.vectors=normalized
