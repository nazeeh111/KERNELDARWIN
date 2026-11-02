"""Stateless, compiled index refresh and batched cosine searches.

This public API uses optimized MLX libraries. Private authored trials are only
for numerical/research work; no accepted transaction artifact exists yet.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from .search import CompiledCosineIndex, _search_inputs
from .workload import _check

SearchResult = tuple[list[list[int]], list[list[float]]]
LIBRARIES = ('stable_library', 'fast_library', 'stable_reduction', 'fast_reduction')
CANDIDATES = ('simd32', 'group128')
MAX_BATCHES = 16
_LOADED_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def transaction_source_identity() -> str:
    """Bind the loaded transaction and its existing application/kernel callees."""
    from . import library_baselines, rank_search, search, study_identity, workload
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _LOADED_CODE_SHA256:
        raise ValueError('Transaction source changed after import')
    for module, loaded in ((library_baselines, library_baselines._LOADED_BASELINE_IDENTITY),
                           (search, search._LOADED_APPLICATION_IDENTITY),
                           (study_identity, study_identity._LOADED_BINDING_IDENTITY)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != loaded:
            raise ValueError('Transaction callee changed after import')
    for module, names, loaded in ((rank_search, rank_search.RANK_SOURCE_FILES, rank_search.rank_source_identity()),
                                  (workload, workload.SOURCE_FILES, workload.source_identity())):
        content = Path(module.__file__).read_bytes() + b''.join(
            name.encode() + (Path(module.__file__).parent/'metal'/name).read_bytes() for name in names)
        if hashlib.sha256(content).hexdigest() != loaded:
            raise ValueError('Transaction runtime or Metal source changed after import')
    return hashlib.sha256(json.dumps({'transaction':_LOADED_CODE_SHA256,
        'rank_application':rank_search.rank_implementation_identity()},sort_keys=True).encode()).hexdigest()


def _supported_custom(rows: Sequence[int], n: int, d: int, k: int) -> bool:
    """Static numerical bounds; performance confirmation is separate."""
    return (type(k) is int and k == 5 and type(n) is int and type(d) is int
            and 1 <= len(rows) <= MAX_BATCHES
            and all(type(q) is int and q > 0 for q in rows)
            and 1 <= sum(rows) <= 128 and 5 <= n <= 2048 and 1 <= d <= 2048
            and sum(rows) * n * d <= 16_777_216)


@lru_cache(maxsize=32)
def _compiled_transaction(method: str, k: int, count: int) -> Any:
    import mlx.core as mx
    from .library_baselines import eager_normalize, fast_formula
    from .rank_search import candidate_rank, reduction_rank_formula
    if method not in (*LIBRARIES, *CANDIDATES) or not 1 <= count <= MAX_BATCHES:
        raise ValueError('Unknown bounded transaction graph')
    if method in CANDIDATES and k != 5:
        raise ValueError('Authored transaction trials require k=5')
    normalize = fast_formula if method.startswith('fast') else eager_normalize

    def graph(documents: Any, *queries: Any) -> tuple[Any, Any, Any, Any]:
        normalized_docs = normalize(documents)
        combined = queries[0] if count == 1 else mx.concatenate(queries, axis=0)
        normalized_queries = normalize(combined)
        doc_finite = mx.all(mx.isfinite(normalized_docs))
        query_finite = mx.all(mx.isfinite(normalized_queries))
        scores = normalized_queries @ normalized_docs.T
        if method in CANDIDATES:
            ids, values = candidate_rank(scores, method)
        elif method.endswith('reduction') and k == 5:
            ids, values = reduction_rank_formula(scores)
        else:
            ids = mx.argsort(-scores, axis=-1)[:, :k]
            values = mx.take_along_axis(scores, ids, axis=-1)
        return ids, values, doc_finite, query_finite

    return mx.compile(graph)


def _execute(method: str, documents: Any, query_batches: Sequence[Any], *,
             k: int = 5, mode: Literal['inference'] = 'inference') -> list[SearchResult]:
    """Research entry; unsupported custom inputs use the complete library path."""
    if mode != 'inference':
        raise ValueError('Only inference is supported')
    if not isinstance(query_batches, (list, tuple)) or not 1 <= len(query_batches) <= MAX_BATCHES:
        raise ValueError('Supply one to sixteen query batches')
    if method not in (*LIBRARIES, *CANDIDATES):
        raise ValueError('Unknown transaction method')
    import mlx.core as mx
    _check(documents)
    if not documents.size:
        raise ValueError('An index needs nonempty two-dimensional vectors')
    for queries in query_batches:
        _search_inputs(queries, documents, k)
    rows = [q.shape[0] for q in query_batches]
    # Concatenating different query dtypes would change normalization precision.
    # Preserve each batch's original dtype through the sequential library path.
    mixed = len({q.dtype for q in query_batches}) > 1
    if mixed or sum(rows) == 0:
        kind = 'fast' if method.startswith('fast') else 'stable'
        index = CompiledCosineIndex(documents, kind)
        return [index.search(q, k=k) for q in query_batches]
    choice = method
    if method in CANDIDATES and not (
            _supported_custom(rows, documents.shape[0], documents.shape[1], k)
            and documents.dtype == mx.float32
            and all(q.dtype == mx.float32 for q in query_batches)
            and mx.default_device() == mx.gpu):
        choice = 'stable_library'
    ids, values, doc_finite, query_finite = _compiled_transaction(choice, k, len(rows))(
        documents, *query_batches)
    mx.eval(ids, values, doc_finite, query_finite)
    if not doc_finite.item():
        raise ValueError('Search index vectors must be finite')
    if not query_finite.item():
        raise ValueError('Search query vectors must be finite')
    host_ids, host_values = ids.tolist(), values.tolist()
    result, offset = [], 0
    for count in rows:
        result.append((host_ids[offset:offset+count], host_values[offset:offset+count]))
        offset += count
    return result


def search_batches(documents: Any, query_batches: Sequence[Any], *, k: int = 5,
                   mode: Literal['inference'] = 'inference') -> list[SearchResult]:
    """Refresh once and search 1..16 query batches in one compiled transaction.

    Inputs are two-dimensional MLX floating arrays with a common feature width.
    The result preserves batch boundaries, including empty query batches. The
    call does not mutate inputs, persist an index or cache answers. Mixed query
    dtypes retain the sequential compiled library path. This function does not
    install or select authored kernels, and supports inference only.
    """
    return _execute('stable_library', documents, query_batches, k=k, mode=mode)
