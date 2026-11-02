"""Bounded local NPZ input for the batch transaction consumer."""
import io
import math
from pathlib import Path
import struct
import time
import zipfile

from .transaction_plan import _read_regular

MAX_BYTES=256*1024**2


def load_batches(path: Path):
    import numpy as np
    payload=_read_regular(Path(path),MAX_BYTES)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries=archive.infolist();names=[item.filename for item in entries]
            if len(names)!=len(set(names)) or not names or sum(item.file_size for item in entries)>MAX_BYTES:
                raise ValueError('Invalid or oversized NPZ entries')
            for item in entries:
                if '/' in item.filename or '\\' in item.filename or not item.filename.endswith('.npy'):
                    raise ValueError('NPZ requires flat numeric array entries')
                with archive.open(item) as stream:
                    version=np.lib.format.read_magic(stream)
                    if version not in ((1,0),(2,0)):raise ValueError('Only NPY1/2 numeric arrays are supported')
                    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                    shape,_,dtype=reader(stream,max_header_size=10_000)
                    if (dtype not in (np.dtype('float16'),np.dtype('float32')) or not 2<=len(shape)<=3
                            or any(type(n) is not int or not 0<=n<=100_000_000 for n in shape)):
                        raise ValueError('NPZ matrices require explicit float16/float32 and bounded2D/3D shapes')
                    size=math.prod(shape)*dtype.itemsize
                    if size>MAX_BYTES or stream.tell()+size!=item.file_size:
                        raise ValueError('NPY payload does not match its bounded declared shape')
        with np.load(io.BytesIO(payload),allow_pickle=False,max_header_size=10_000) as arrays:
            keys=set(arrays.files)
            if 'documents' not in keys:raise ValueError('NPZ needs documents')
            if 'queries' in keys:
                if keys!={'documents','queries'}:raise ValueError('Choose queries or numbered query groups, not both')
                query=arrays['queries']
                if query.ndim==2:batches=[query]
                else:batches=[query[i] for i in range(query.shape[0])]
            else:
                count=len(keys)-1
                expected={'documents',*(f'queries_{i}' for i in range(count))}
                if keys!=expected:raise ValueError('Query group keys must be consecutive queries_0..queries_N')
                batches=[arrays[f'queries_{i}'] for i in range(count)]
            if not 1<=len(batches)<=16 or any(batch.ndim!=2 for batch in batches):
                raise ValueError('Supply one to sixteen2D query groups')
            documents=arrays['documents']
            if documents.ndim!=2:raise ValueError('Documents must be a2D matrix')
            return documents,batches
    except (zipfile.BadZipFile,EOFError,OverflowError,struct.error) as error:
        raise ValueError('Input must be a valid bounded NPZ archive') from error


def run(args):
    from dataclasses import asdict
    from .transaction_plan import TransactionPlan
    import mlx.core as mx
    mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
    start=time.perf_counter();documents,queries=load_batches(args.vectors)
    preparation=(time.perf_counter()-start)*1000
    plan=TransactionPlan(args.artifact);mx.synchronize();start=time.perf_counter()
    result=plan.search(documents,queries,k=args.k);mx.synchronize()
    return {'application':'one index refresh and batched searches','artifact_status':plan.status,
        'dispatch_scope':'batch_refresh_top5_v1','input_prepare_ms':preparation,
        'compute_ms':(time.perf_counter()-start)*1000,'calls':plan.calls,'fallback_calls':plan.fallback_calls,
        'last_decision':asdict(plan.last_decision),'batches':[{'ids':ids,'scores':scores} for ids,scores in result],
        'timing_note':'Single invocation including cold work and all host staging; not an acceleration claim'}
