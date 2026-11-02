"""Numerical-only fused batch premise. No accepted plan, export or tuning."""
from functools import lru_cache
import hashlib
from pathlib import Path

from kerneldarwin import transaction_plan as host
from kerneldarwin import transaction_search as baseline
from kerneldarwin import rank_search as rank

HERE=Path(__file__).resolve().parent
SOURCE=(HERE/'normalize_pack.metal').read_bytes()
LOADED=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def identity():
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=LOADED or (HERE/'normalize_pack.metal').read_bytes()!=SOURCE:
        raise ValueError('Prototype source changed after import')
    return hashlib.sha256(LOADED.encode()+SOURCE+host.source_identity().encode()).hexdigest()


@lru_cache(maxsize=1)
def kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='kd_normalize_pack3',input_names=['docs','q0','q1','q2'],
        output_names=['normalized_docs','normalized_queries','invalid_rows'],source=SOURCE.decode(),
        ensure_row_contiguous=False,compile_options={'math_mode':'safe'})


def normalize(docs,queries):
    import mlx.core as mx
    n,d=docs.shape;rows=[q.shape[0] for q in queries];total=n+sum(rows)
    if (len(rows)!=3 or not baseline._supported_custom(rows,n,d,5) or docs.dtype!=mx.float32
            or any(q.dtype!=mx.float32 or q.ndim!=2 or q.shape[1]!=d for q in queries)):
        raise ValueError('Private normalization needs three supported float32 query groups')
    with mx.stream(mx.gpu):
        return kernel()(inputs=[docs,*queries],template=[('N',n),('D',d),*[(f'Q{i}',r) for i,r in enumerate(rows)]],
            grid=(total*32,1,1),threadgroup=(min(total*32,128),1,1),
            output_shapes=[(n,d),(sum(rows),d),(total,)],output_dtypes=[mx.float32,mx.float32,mx.uint32])


@lru_cache(maxsize=2)
def graph(variant):
    import mlx.core as mx
    if variant not in baseline.CANDIDATES:raise ValueError('Unknown bounded rank configuration')
    def run(docs,*queries):
        nd,nq,flags=normalize(docs,queries)
        scores=nq@nd.T
        ids,values=rank.candidate_rank(scores,variant)
        valid=mx.all(flags==0)
        return ids,values,valid
    return mx.compile(run)


class Trial:
    def __init__(self,variant):
        if variant not in baseline.CANDIDATES:raise ValueError('Unknown bounded rank configuration')
        self.variant=variant;self.last='not_called';self.calls=self.fallback_calls=0

    def search(self,documents,queries,*,k=5,mode='inference'):
        docs,groups=host.prepare(documents,queries,k,mode)
        import mlx.core as mx
        self.calls+=1
        if len(groups)!=3 or host.host_signature(documents,queries,k) is None or mx.default_device()!=mx.gpu:
            self.fallback_calls+=1;self.last='library'
            return baseline.search_batches(docs,groups,k=k,mode=mode)
        ids,values,valid=graph(self.variant)(docs,*groups);mx.eval(ids,values,valid)
        self.last=self.variant
        if not valid.item():raise ValueError('Search vectors must be finite')
        ids,values=ids.tolist(),values.tolist();result=[];offset=0
        for group in groups:
            count=group.shape[0];result.append((ids[offset:offset+count],values[offset:offset+count]));offset+=count
        return result
