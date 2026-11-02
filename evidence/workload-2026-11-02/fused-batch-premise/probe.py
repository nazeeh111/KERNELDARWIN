"""One numerical/resource premise test of fused batch normalization, not timing."""
from pathlib import Path
import collections
import ctypes as C
import hashlib
import json
import os
import random
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BINARY=Path('/private/tmp/kd-fused-premise-counters-20261102.dylib')
CAPACITY=4096


class Event(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('tag','kind','a','b','c','d','e','f')]+[('name',C.c_char*512)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    start=time.perf_counter();mx=None;observer=None;installed=False
    pin=json.loads((HERE/'preregistered.json').read_text())
    for name,digest in pin['files'].items():
        if sha(HERE/name)!=digest:raise ValueError('Prospective source changed: '+name)
    if sha(BINARY)!=pin['binary_sha256']:raise ValueError('Observer binary changed')
    state={'kind':'fused_batch_normalization_numerical_resource_premise','checks':[],'samples':[],
           'candidate_performance_timed':False,'acceptance_study':False,'heldout_executed':False}
    save('started.json',{'pid':os.getpid(),'unix_ns':time.time_ns(),'script_sha256':sha(__file__)})
    try:
        observer=C.CDLL(str(BINARY));observer.kd_begin.argtypes=[C.c_uint64]
        observer.kd_read.argtypes=[C.POINTER(Event),C.c_size_t];observer.kd_read.restype=C.c_size_t
        for name in ('kd_pending','kd_dropped','kd_event_size'):getattr(observer,name).restype=C.c_uint64
        if observer.kd_event_size()!=C.sizeof(Event):raise ValueError('Counter ABI mismatch')
        if observer.kd_install()!=0:raise RuntimeError('Own-process observer unavailable')
        installed=True
        def budget():
            if time.perf_counter()-start>120:raise TimeoutError('120-second premise cap; no retry')
            if mx is not None and mx.get_peak_memory()>2*1024**3:raise RuntimeError('2GiB cap')
        def collect(tag,operation):
            budget()
            if observer.kd_begin(tag)!=0:raise RuntimeError('Counter not ready')
            try:
                output=operation();observer.kd_api_return()
                if mx is not None:mx.synchronize()
            finally:observer.kd_end()
            deadline=time.perf_counter()+1
            while observer.kd_pending():
                if time.perf_counter()>deadline:raise RuntimeError('Callbacks did not drain')
                time.sleep(.0001)
            buffer=(Event*CAPACITY)();count=observer.kd_read(buffer,CAPACITY)
            if count>CAPACITY or observer.kd_dropped():raise RuntimeError('Counter overflow')
            events=[{**{k:getattr(item,k) for k,_ in Event._fields_ if k!='name'},'name':item.name.decode()} for item in buffer[:count]]
            if any(e['tag']!=tag for e in events) or sum(e['kind']==9 for e in events)!=1:raise RuntimeError('Missing call boundary')
            if sum(e['kind']==7 for e in events)!=sum(e['kind']==8 for e in events) or any(e['a']!=4 for e in events if e['kind']==8):
                raise RuntimeError('Missing/failed GPU completions')
            return output,events
        code,control=collect(1,observer.kd_copy_control)
        if code or [e['a'] for e in control if e['kind']==4]!=[16]:raise RuntimeError('Blit positive control failed')
        if [(e['kind'],e['name']) for e in control if e['kind'] in (5,6)]!=[(5,'kd_counter_control'),(6,'kd_counter_control')]:
            raise RuntimeError('Named dispatch positive controls failed')
        if not all(any(e['kind']==kind for e in control) for kind in (1,2,3,7,8,9)):raise RuntimeError('Resource controls missing')
        state['positive_control']=control
        import mlx.core as mlx
        import numpy as np
        mx=mlx
        mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
        import prototype
        from kerneldarwin import transaction_plan as host
        from kerneldarwin import transaction_search as baseline
        from kerneldarwin.search import text_vectors
        if host.source_identity()!=pin['installed_source_sha256']:raise ValueError('Installed runtime changed')
        if '/site-packages/' not in host.__file__:raise ValueError('Use the actual independent installation')
        source_bytes=len(prototype.SOURCE)+sum(len(data) for _,data in prototype.rank._SOURCES)
        if source_bytes>pin['max_combined_msl_bytes']:raise ValueError('Readable MSL source cap exceeded')
        state.update(prototype_identity=prototype.identity(),installed_source_sha256=host.source_identity(),
                     actual_installed_import=host.__file__,environment=host.environment_identity(),
                     authored_source_bytes=source_bytes)
        def normalized(x):
            value=np.asarray(x,np.float64).copy();value[np.abs(value)<np.finfo(np.float32).tiny]=0.
            norm=np.sqrt(np.sum(value*value,axis=1,keepdims=True))
            output=value/np.maximum(norm,1e-12);output[np.abs(output)<np.finfo(np.float32).tiny]=0.
            return output
        def check(output,docs,queries,tolerance=2e-5):
            nd=normalized(docs)
            for (ids,values),q in zip(output,queries):
                if len(q)==0:
                    assert ids==values==[];continue
                ids=np.asarray(ids);values=np.asarray(values);ref=normalized(q)@nd.T
                assert ids.shape==values.shape==(len(q),5)
                assert all(len(set(row))==5 for row in ids.tolist()) and np.all((ids>=0)&(ids<len(docs)))
                np.testing.assert_allclose(values,np.take_along_axis(ref,ids,axis=1),rtol=tolerance,atol=tolerance/10)
                np.testing.assert_allclose(values,-np.sort(-ref,axis=1)[:,:5],rtol=tolerance,atol=tolerance/10)
        def layout(value,kind):
            if kind=='transpose':return value.T.copy().T
            if kind=='column_stride2':
                storage=np.zeros((len(value),value.shape[1]*2),np.float32);storage[:,::2]=value;return storage[:,::2]
            return value.copy()
        trials={v:prototype.Trial(v) for v in baseline.CANDIDATES}
        shapes=[(5,1,(1,1,1)),(7,31,(1,2,3)),(37,97,(1,3,7)),
                (64,384,(1,1,1)),(161,768,(8,8,8)),(161,1536,(17,17,17)),(2048,2048,(1,1,1))]
        rng=np.random.default_rng(50401)
        for n,d,rows in shapes:
            docs=rng.normal(size=(n,d)).astype(np.float32);queries=[rng.normal(size=(r,d)).astype(np.float32) for r in rows]
            for kind in ('contiguous','column_stride2','transpose'):
                a=layout(docs,kind);before=a.copy()
                for name,trial in trials.items():
                    check(trial.search(a,queries),a,queries);assert trial.last==name
                for method in baseline.LIBRARIES:check(host.execute_host(method,a,queries),a,queries)
                md,mq=host.prepare(a,queries);nd,nq,flags=prototype.normalize(md,mq);mx.eval(nd,nq,flags)
                np.testing.assert_allclose(np.array(nd),normalized(a),rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(np.array(nq),normalized(np.concatenate(queries)),rtol=2e-5,atol=2e-6)
                assert not np.any(np.array(flags));np.testing.assert_array_equal(a,before)
                state['checks'].append({'kind':'random_shapes_layouts','n':n,'d':d,'query_rows':list(rows),'layout':kind,'authored_paths':2,'library_paths':4})
                budget()
        docs=np.zeros((37,97),np.float32);queries=[np.zeros((r,97),np.float32) for r in (1,3,7)]
        for name,trial in trials.items():
            output=trial.search(docs,queries);check(output,docs,queries)
            for (ids,values),q in zip(output,queries):
                np.testing.assert_array_equal(ids,np.tile(np.arange(5),(len(q),1)));assert not np.any(values)
        state['checks'].append({'kind':'zero_stable_ties','authored_paths':2})
        docs=rng.normal(size=(37,97)).astype(np.float32)
        docs[0]=np.finfo(np.float32).tiny;docs[1]=1e-20;docs[2]=0.;docs[3]=np.finfo(np.float32).max*.25
        docs[4]=np.resize(np.array([1e30,-1e30,1e-30],np.float32),97);docs[5]=np.nextafter(np.float32(0),np.float32(1))
        queries=[docs[:1].copy(),docs[1:4].copy(),docs[:7].copy()]
        md,mq=host.prepare(docs,queries);nd,nq,flags=prototype.normalize(md,mq);mx.eval(nd,nq,flags)
        np.testing.assert_allclose(np.array(nd),normalized(docs),rtol=2e-5,atol=0)
        np.testing.assert_allclose(np.array(nq),normalized(np.concatenate(queries)),rtol=2e-5,atol=0)
        assert not np.any(np.array(flags))
        for trial in trials.values():check(trial.search(docs,queries),docs,queries)
        state['checks'].append({'kind':'adversarial_magnitude_cancellation_subnormal_and_tiny_atol0','authored_paths':2})
        for nonfinite in (np.nan,np.inf,-np.inf):
            for target in range(4):
                arrays=[docs.copy(),*[q.copy() for q in queries]];arrays[target][0,0]=nonfinite
                staged=host.prepare(arrays[0],arrays[1:]);nd,nq,flags=prototype.normalize(*staged);mx.eval(nd,nq,flags)
                expected=np.concatenate([np.any(~np.isfinite(x),axis=1) for x in arrays]).astype(np.uint32)
                np.testing.assert_array_equal(np.array(flags),expected)
                packed=np.concatenate([np.array(nd),np.array(nq)])
                assert np.all(np.isnan(packed[expected==1]))
                for trial in trials.values():
                    try:trial.search(arrays[0],arrays[1:])
                    except ValueError:pass
                    else:raise AssertionError('Nonfinite input was not rejected')
                state['checks'].append({'kind':'nonfinite_rejection_flags_and_nan_rows','target':target,'value':str(nonfinite),'authored_paths':2})
                budget()
        docs=rng.normal(size=(37,97)).astype(np.float32);queries=[docs[:1],docs[1:4],docs[:7]];saved=docs.copy()
        for trial in trials.values():check(trial.search(docs,queries),docs,queries)
        np.testing.assert_array_equal(docs,saved);state['checks'].append({'kind':'aliasing_and_nonfinite_recovery','authored_paths':2})
        # Unsupported host layouts/dtypes/boundaries retain the whole library.
        readonly=docs.copy();readonly.flags.writeable=False
        requests=[(docs[:,::-1],[q[:,::-1] for q in queries]),(readonly,queries),
                  (docs.astype(np.float16),[q.astype(np.float16) for q in queries]),
                  (mx.array(docs),[mx.array(q) for q in queries]),(docs,queries[:2]),
                  (docs,[np.zeros((0,97),np.float32),*queries[:2]])]
        for a,b in requests:
            for trial in trials.values():
                check(trial.search(a,b),np.array(a),[np.array(q) for q in b],tolerance=.03 if isinstance(a,np.ndarray) and a.dtype==np.float16 else 2e-5)
                assert trial.last=='library'
            state['checks'].append({'kind':'unsupported_request_library_fallback','authored_paths':2})
        # Resource counts use only the existing three development profiles.
        corpus_path=ROOT/'evidence/workload-2026-10-29/corpus.json'
        if sha(corpus_path)!=pin['corpus_sha256']:raise ValueError('Licensed corpus changed')
        texts=json.loads(corpus_path.read_text())['chunks'];order=list(range(len(texts)));random.Random(50411).shuffle(order)
        profiles=[('notes',64,384,1,'contiguous'),('manuals',161,768,8,'column_stride2'),('archive',161,1536,17,'transpose')]
        tag=1
        for name,n,d,q,kind in profiles:
            a=layout(np.array(text_vectors([texts[order[i]]+' revision50411' for i in range(n)],d)),kind)
            base=np.array(text_vectors([texts[order[(i*7+3)%len(order)]]+' query50411' for i in range(q)],d))
            b=[np.ascontiguousarray(base+j*.001) for j in range(3)]
            methods={**{m:(lambda m=m:host.execute_host(m,a,b)) for m in baseline.LIBRARIES},
                     **{m:(lambda t=t:t.search(a,b)) for m,t in trials.items()}}
            for operation in methods.values():check(operation(),a,b)
            mx.synchronize()
            for method,operation in methods.items():
                tag+=1;output,events=collect(tag,operation);check(output,a,b)
                kernels=[e for e in events if e['kind'] in (5,6)]
                if not kernels or any(e['name']=='unknown' for e in kernels):raise RuntimeError('Unresolved compute entry point')
                boundary=next(i for i,e in enumerate(events) if e['kind']==9)
                state['samples'].append({'profile':name,'method':method,'seed':50411,'n':n,'d':d,'query_rows':[q]*3,
                    'compute_dispatches':len(kernels),'kernel_counts':dict(collections.Counter(e['name'] for e in kernels)),
                    'api_command_buffers':sum(e['kind']==7 for e in events[:boundary]),
                    'diagnostic_sync_command_buffers':sum(e['kind']==7 for e in events[boundary+1:]),
                    'events':events,'float64_reference_passed':True})
                budget()
        if prototype.identity()!=state['prototype_identity']:raise ValueError('Prototype changed during execution')
        for name,digest in pin['files'].items():
            if sha(HERE/name)!=digest:raise ValueError('Registered source changed')
        state['completed']=True
    except Exception as error:
        state.update(completed=False,error=type(error).__name__+': '+str(error));raise
    finally:
        if observer is not None:
            observer.kd_end()
            if mx is not None:mx.synchronize()
            deadline=time.perf_counter()+1
            while observer.kd_pending() and time.perf_counter()<deadline:time.sleep(.0001)
            state.update(dropped_records=observer.kd_dropped(),pending_callbacks=observer.kd_pending())
            if installed:state['observer_uninstall_result']=observer.kd_uninstall()
        if mx is not None:state['peak_memory_bytes']=mx.get_peak_memory()
        state['wall_seconds']=time.perf_counter()-start;save('result.json',state)
        print(json.dumps({k:v for k,v in state.items() if k not in ('checks','samples','positive_control','environment')},indent=2))


if __name__=='__main__':main()
