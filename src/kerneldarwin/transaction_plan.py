"""Data-only static dispatch for host-origin batch search transactions."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Literal, Sequence

from . import rank_search as rank
from . import transaction_search as tx
from .workload import Decision, EPSILON, environment_identity as _base_environment

OPERATION = 'batch_refresh_top5_v1'
LAYOUTS = ('host_contiguous', 'host_column_stride2', 'host_transpose')
_LOADED_CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _read_regular(path: Path, limit: int) -> bytes:
    if path.is_symlink():raise ValueError('Expected a regular file, not a symlink')
    descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_NONBLOCK',0)|getattr(os,'O_NOFOLLOW',0))
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size>limit:
            raise ValueError('Expected a bounded regular file')
        file=os.fdopen(descriptor,'rb')
    except BaseException:
        os.close(descriptor)
        raise
    with file:
        content=file.read(limit+1)
    if len(content)!=info.st_size or len(content)>limit:
        raise ValueError('File changed during bounded read')
    return content


def _decode(content: bytes) -> Any:
    def invalid(value):raise ValueError('Nonfinite JSON number: '+value)
    def floating(value):
        number=float(value)
        if not math.isfinite(number):invalid(value)
        return number
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(content,parse_constant=invalid,parse_float=floating,object_pairs_hook=unique)


def source_identity() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _LOADED_CODE:
        raise ValueError('Batch dispatcher source changed after import')
    return hashlib.sha256(json.dumps({'dispatcher':_LOADED_CODE,
        'executor':tx.transaction_source_identity()},sort_keys=True).encode()).hexdigest()


def study_identity() -> str:
    path = Path(__file__).parent/'transaction-study-source.json'
    content=_read_regular(path,4096)
    pin=_decode(content)
    keys={'schema','measurement_script_sha256','measurement_helper_sha256','verifier_sha256',
          'numerical_producer_sha256','numerical_tests_sha256'}
    if (not isinstance(pin,dict) or set(pin)!=keys or type(pin['schema']) is not int or pin['schema']!=1
            or not isinstance(pin['numerical_tests_sha256'],dict)
            or set(pin['numerical_tests_sha256'])!={'test_transaction_search.py','test_transaction_plan.py'}):
        raise ValueError('Invalid batch measurement source pin')
    hashes=[pin[key] for key in keys if key.endswith('_sha256') and key!='numerical_tests_sha256']
    hashes+=list(pin['numerical_tests_sha256'].values())
    if any(not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in hashes):
        raise ValueError('Invalid batch measurement source pin')
    return hashlib.sha256(json.dumps({'source':source_identity(),'evidence_code':pin},sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def _environment_snapshot() -> str:
    """Version/device identity plus the actual loaded runtime's installed files."""
    import mlx.core as mx
    import numpy as np
    try:
        from numpy._core import _multiarray_umath
    except ImportError:
        from numpy.core import _multiarray_umath
    base=_base_environment()
    directory=Path(mx.__file__).parent
    files={'core':Path(mx.__file__),'library':directory/'lib/libmlx.dylib',
           'metallib':directory/'lib/mlx.metallib','numpy_core':Path(_multiarray_umath.__file__)}
    return json.dumps({**base,'numpy':np.__version__,'runtime_file_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest()
                                         for name,path in files.items()}},sort_keys=True)


def environment_identity() -> dict[str, Any]:
    return json.loads(_environment_snapshot())


def layout_strides(layout: str, n: int, d: int) -> tuple[int,int]:
    if layout=='host_contiguous':return d*4,4
    if layout=='host_column_stride2':return d*8,8
    if layout=='host_transpose':return 4,n*4
    raise ValueError('Unknown profiled host layout')


def rule_key(rows: Sequence[int], n: int, d: int, k: int, layout: str) -> tuple:
    return tuple(rows),n,d,k,layout


def host_signature(documents: Any, batches: Sequence[Any], k: int) -> tuple | None:
    """Read host metadata only; never evaluate GPU arrays to discover strides."""
    import numpy as np
    if not isinstance(documents,np.ndarray) or not isinstance(batches,(list,tuple)):
        return None
    if (documents.ndim!=2 or documents.dtype!=np.float32 or not documents.flags.aligned or not documents.flags.writeable
            or not all(isinstance(q,np.ndarray) and q.ndim==2 and q.dtype==np.float32
                       and q.flags.aligned and q.flags.writeable for q in batches)):
        return None
    n,d=documents.shape
    rows=[q.shape[0] for q in batches]
    if not tx._supported_custom(rows,n,d,k) or any(q.shape[1]!=d or q.strides!=(d*4,4) for q in batches):
        return None
    layout=next((name for name in LAYOUTS if documents.strides==layout_strides(name,n,d)),None)
    return rule_key(rows,n,d,k,layout) if layout else None


def prepare(documents: Any, batches: Sequence[Any], k: int = 5,
            mode: Literal['inference'] = 'inference') -> tuple[Any,list[Any]]:
    """Validate metadata, then explicitly stage host matrices without downcasting."""
    if mode!='inference':raise ValueError('Only inference is supported')
    if not isinstance(batches,(list,tuple)) or not 1<=len(batches)<=tx.MAX_BATCHES:
        raise ValueError('Supply one to sixteen query batches')
    import mlx.core as mx
    import numpy as np
    from .workload import _check
    def check(value):
        if isinstance(value,np.ndarray):
            if value.ndim!=2:raise ValueError('Expected a two-dimensional row matrix')
            if value.dtype not in (np.float16,np.float32):
                raise TypeError('Host matrices require explicit float16 or float32 dtype')
        else:_check(value)
    check(documents)
    if not documents.size:raise ValueError('An index needs nonempty two-dimensional vectors')
    if type(k) is not int or not 1<=k<=documents.shape[0]:
        raise ValueError('k must be between one and the index row count')
    for q in batches:
        check(q)
        if q.shape[1]!=documents.shape[1]:raise ValueError('Query width must match index width')
    def stage(value):
        if not isinstance(value,np.ndarray):return value
        # MLX's DLPack import rejects negative strides and read-only storage.
        # Explicitly copy these unprofiled requests before library fallback.
        if any(s<0 for s in value.strides) or not value.flags.aligned or not value.flags.writeable:
            value=value.copy(order='C')
        return mx.array(value)
    return stage(documents),[stage(q) for q in batches]


def execute_host(method: str, documents: Any, batches: Sequence[Any], *, k: int = 5,
                 mode: Literal['inference'] = 'inference') -> list[tx.SearchResult]:
    """Private fixed-method comparator with all staging and materialization."""
    docs,queries=prepare(documents,batches,k,mode)
    return tx._execute(method,docs,queries,k=k,mode=mode)


class TransactionPlan:
    """Install a confirmed exact host shape/stride portfolio, or use the library.

    Inputs may be NumPy or MLX matrices. Confirmed custom rules apply only to
    aligned float32 NumPy requests matching known shapes and strides. Other
    valid requests use the complete compiled library transaction. Inputs must
    not be concurrently modified during the call. No index or answer persists.
    """
    def __init__(self, artifact: str | Path | None = None) -> None:
        self._rules: dict[tuple,str]={}
        self.status='library_only'
        self.calls=self.fallback_calls=0
        self.last_decision=Decision('library','No installed batch plan')
        if artifact is None:return
        try:
            path=Path(artifact)
            if path.is_dir():path=path/'manifest.json'
            content=_read_regular(path,1_000_000)
            self._load(_decode(content),confirmed=True)
            for name,expected in rank._SOURCES:
                source=path.parent/name
                if _read_regular(source,rank.MAX_MSL_BYTES)!=expected:
                    raise ValueError('Readable batch kernel differs from installed source')
            self.status='installed'
        except (OSError,ValueError,TypeError,KeyError,RuntimeError,ImportError) as error:
            self._rules.clear();self.status='fallback: '+str(error)

    def _load(self, manifest: dict[str,Any], *, confirmed: bool) -> None:
        if not isinstance(manifest,dict):raise ValueError('Batch plan must be an object')
        if (type(manifest['schema']) is not int or manifest['schema']!=1
                or manifest['operation']!=OPERATION or manifest['epsilon']!=EPSILON
                or manifest['key_policy']!=rank.KEY_POLICY
                or manifest['compile_options']!={'math_mode':'safe'}):
            raise ValueError('Incompatible batch operation')
        if manifest['source_sha256']!=source_identity() or manifest['study_identity']!=study_identity():
            raise ValueError('Batch source or measurement changed')
        if manifest['environment']!=environment_identity():raise ValueError('Batch runtime or device changed')
        evidence=manifest['evidence']
        if (not isinstance(evidence,dict) or evidence.get('numerical_passed') is not True
                or (confirmed and evidence.get('confirmation_passed') is not True)):
            raise ValueError('Unconfirmed batch plan')
        if manifest['sources']!={name:hashlib.sha256(content).hexdigest() for name,content in rank._SOURCES}:
            raise ValueError('Batch Metal source hash mismatch')
        rules=manifest['rules']
        if not isinstance(rules,list) or len(rules)>128 or (confirmed and not rules):
            raise ValueError('Invalid or empty confirmed batch rules')
        loaded={}
        for row in rules:
            shape=row['shape'];layout=row['document_layout']
            if (not isinstance(shape,list) or not 4<=len(shape)<=19
                    or any(type(v) is not int for v in shape)
                    or not tx._supported_custom(shape[:-3],*shape[-3:])
                    or layout not in LAYOUTS or row['dtype']!='float32'
                    or row['query_layout']!='host_contiguous' or row['candidate'] not in tx.CANDIDATES):
                raise ValueError('Unsupported batch dispatch contract')
            key=rule_key(shape[:-3],*shape[-3:],layout)
            if key in loaded:raise ValueError('Duplicate batch dispatch rule')
            loaded[key]=row['candidate']
        if len(set(loaded.values()))>2:raise ValueError('Batch portfolio exceeds two kernels')
        self._rules=loaded

    @classmethod
    def _trial(cls, manifest: dict[str,Any]) -> TransactionPlan:
        result=cls();result._load(manifest,confirmed=False);result.status='numerical_trial'
        return result

    def _choose(self, signature: tuple | None) -> str | None:
        import mlx.core as mx
        self.calls+=1
        choice=self._rules.get(signature) if signature is not None and mx.default_device()==mx.gpu else None
        if choice is None:
            self.fallback_calls+=1
            self.last_decision=Decision('library','Unfamiliar request or no compatible batch plan')
        else:self.last_decision=Decision(choice,'Confirmed host shape and stride rule')
        return choice

    def search(self, documents: Any, query_batches: Sequence[Any], *, k: int = 5,
               mode: Literal['inference'] = 'inference') -> list[tx.SearchResult]:
        docs,queries=prepare(documents,query_batches,k,mode)
        choice=self._choose(host_signature(documents,query_batches,k))
        return tx._execute(choice or 'stable_library',docs,queries,k=k,mode=mode)

    def _rank_trial(self, signature: tuple | None, scores: Any) -> tuple[Any,Any]:
        """Study-only ranking on evaluated scores and prevalidated metadata.

        Complete host validation/staging belongs to search and its application
        timing. This private adapter isolates bounded dispatch and ranking.
        """
        choice=self._choose(signature)
        return (rank.compiled_rank(choice) if choice else rank.library_rank())(scores)
