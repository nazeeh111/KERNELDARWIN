"""Equivalent optimized-library alternatives used by the workload investigation."""
from __future__ import annotations
from functools import lru_cache
from typing import Any
import hashlib
from pathlib import Path

_LOADED_BASELINE_IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def eager_normalize(x: Any) -> Any:
    import mlx.core as mx
    if not x.size:
        return mx.zeros(x.shape,dtype=x.dtype)
    z=x.astype(mx.float32)
    scale=mx.maximum(mx.max(mx.abs(z),axis=1,keepdims=True),1e-12)
    z=z/scale
    return (z/mx.maximum(mx.sqrt(mx.sum(z*z,axis=1,keepdims=True)),1e-12/scale)).astype(x.dtype)


def fast_library_normalize(x: Any) -> Any:
    import mlx.core as mx
    from .workload import _check
    _check(x)
    if not x.size:
        return mx.zeros(x.shape,dtype=x.dtype)
    return _compiled_fast()(x)


def fast_formula(x: Any) -> Any:
    """Pure stable, clamped RMS formula, available to a larger compiled graph."""
    import mlx.core as mx
    if not x.size:
        return mx.zeros(x.shape,dtype=x.dtype)
    z=x.astype(mx.float32)
    scale=mx.maximum(mx.max(mx.abs(z),axis=1,keepdims=True),1e-12)
    z=z/scale
    squares=mx.sum(z*z,axis=1,keepdims=True)
    threshold=1e-12/scale
    # Optimized RMSNorm supplies the unit-length case. Near-zero rows use
    # the exact clamped denominator; NaNs are explicitly whole-row NaNs.
    unit=mx.fast.rms_norm(z,None,0)*(1/x.shape[1]**0.5)
    value=mx.where(squares>=threshold*threshold,unit,z/threshold)
    return mx.where(mx.isnan(squares),float('nan'),value).astype(x.dtype)


@lru_cache(maxsize=1)
def _compiled_fast() -> Any:
    import mlx.core as mx
    return mx.compile(fast_formula)


class FullGraphLibrary:
    """Use the same formula for operation and complete query-graph baselines."""

    def __init__(self, kind: str = "stable") -> None:
        if kind not in ("stable","fast"):
            raise ValueError("Library graph kind must be 'stable' or 'fast'")
        self.kind=kind

    def __call__(self, x: Any) -> Any:
        from .workload import library_normalize
        return library_normalize(x) if self.kind=="stable" else fast_library_normalize(x)

    def index(self, vectors: Any) -> Any:
        from .search import CompiledCosineIndex
        return CompiledCosineIndex(vectors,self.kind)
