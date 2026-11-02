"""One fixed, bounded installed-library call-accounting diagnostic, not timing."""
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
LIBRARY=Path('/private/tmp/kd-call-counters-20261102.dylib')
CAPACITY=4096


class Event(C.Structure):
    _fields_=[(name,C.c_uint64) for name in ('tag','kind','a','b','c','d','e','f')]+[('name',C.c_char*160)]


def exclusive(name,data):
    with (HERE/name).open('x') as file:json.dump(data,file,indent=2,allow_nan=False);file.write('\n')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start=time.perf_counter();observer=None;installed=False;mx=None
    pin=json.loads((HERE/'preregistered.json').read_text())
    if pin['script_sha256']!=digest(__file__) or pin['helper_sha256']!=digest(HERE/'call_counters.mm') or pin['binary_sha256']!=digest(LIBRARY):
        raise ValueError('Diagnostic source/binary differs from its registration')
    state={'kind':'installed_batch_call_resource_accounting','source_sha256':pin['script_sha256'],
           'candidate_performance_timed':False,'acceptance_study':False,'samples':[]}
    exclusive('started.json',{'pid':os.getpid(),'unix_ns':time.time_ns(),'source_sha256':pin['script_sha256']})
    try:
        observer=C.CDLL(str(LIBRARY));observer.kd_begin.argtypes=[C.c_uint64]
        observer.kd_read.argtypes=[C.POINTER(Event),C.c_size_t];observer.kd_read.restype=C.c_size_t
        observer.kd_pending.restype=C.c_uint64;observer.kd_dropped.restype=C.c_uint64
        if observer.kd_install()!=0:raise RuntimeError('Own-process Metal observer unavailable')
        installed=True
        def collect(tag,operation):
            if time.perf_counter()-start>60:raise TimeoutError('60-second diagnostic cap')
            if observer.kd_begin(tag)!=0:raise RuntimeError('Observer not ready')
            try:
                output=operation()
                if mx is not None:mx.synchronize()
            finally:observer.kd_end()
            deadline=time.perf_counter()+1
            while observer.kd_pending():
                if time.perf_counter()>deadline:raise RuntimeError('Own completion callbacks did not drain')
                time.sleep(.0001)
            buffer=(Event*CAPACITY)();size=observer.kd_read(buffer,CAPACITY)
            if size>CAPACITY or observer.kd_dropped():raise RuntimeError('Counter overflow; no inferred counts')
            events=[{**{key:getattr(item,key) for key,_ in Event._fields_ if key!='name'},
                     'name':item.name.decode()} for item in buffer[:size]]
            if any(e['tag']!=tag for e in events):raise RuntimeError('Mixed observation scopes')
            commits=[e for e in events if e['kind']==7];done=[e for e in events if e['kind']==8]
            if len(commits)!=len(done) or not done or any(e['a']!=4 for e in done):raise RuntimeError('Missing or failed actual GPU completion')
            return output,events
        code,events=collect(1,observer.kd_copy_control)
        if code!=0:raise RuntimeError('GPU copy/dispatch positive control failed')
        if [e['a'] for e in events if e['kind']==4]!=[16]:raise RuntimeError('Exact16byte blit control not captured')
        if [(e['kind'],e['name']) for e in events if e['kind'] in (5,6)]!=[(5,'kd_counter_control'),(6,'kd_counter_control')]:
            raise RuntimeError('Both named compute entry points must be captured')
        if not any(e['kind']==1 and e['c']==2 and e['a']==64 for e in events):raise RuntimeError('Heap allocation control missing')
        if not any(e['kind']==2 and e['a']==32 for e in events):raise RuntimeError('Byte-copy allocation control missing')
        if not any(e['kind']==3 for e in events):raise RuntimeError('No-copy allocation control missing')
        if sum(e['a']==128 and e['kind']==1 and e['c']==1 for e in events)!=2:raise RuntimeError('Device allocation control missing')
        state['positive_control']={'passed':True,'events':events}
        import mlx.core as mlx
        mx=mlx
        import numpy as np
        from kerneldarwin import transaction_plan as plan
        from kerneldarwin.search import text_vectors
        if not mx.metal.is_available():raise RuntimeError('Actual Metal unavailable')
        mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
        if plan.source_identity()!=pin['operation_source_sha256'] or plan.study_identity()!=pin['study_identity']:
            raise ValueError('Installed operation source changed')
        if '/site-packages/' not in plan.__file__:raise ValueError('Use the independent installed runtime')
        state.update(operation_source_sha256=plan.source_identity(),study_identity=plan.study_identity(),
                     actual_import=plan.__file__,environment=plan.environment_identity())
        corpus_path=ROOT/'evidence/workload-2026-10-29/corpus.json'
        if digest(corpus_path)!=pin['corpus_sha256']:raise ValueError('Licensed corpus changed')
        texts=json.loads(corpus_path.read_text())['chunks']
        cases=[('notes',64,384,1,'contiguous'),('manuals',161,768,8,'column_stride2'),
               ('archive',161,1536,17,'transpose'),('notes_negative',64,384,1,'negative')]
        def request(case,seed):
            name,n,d,q,layout=case;order=list(range(len(texts)));random.Random(seed).shuffle(order)
            docs=np.array(text_vectors([texts[order[i]]+f' revision{seed%17}' for i in range(n)],d))
            query=np.array(text_vectors([texts[order[(i*7+3)%len(order)]]+f' query{seed%23}' for i in range(q)],d))
            queries=[np.ascontiguousarray(query+j*.001) for j in range(3)]
            if layout=='column_stride2':
                backing=np.zeros((n,d*2),np.float32);backing[:,::2]=docs;docs=backing[:,::2]
            elif layout=='transpose':docs=docs.T.copy().T
            elif layout=='negative':docs=docs[:,::-1];queries=[x[:,::-1] for x in queries]
            return docs,queries
        def reference(output,docs,queries):
            def normalize(x):
                x=x.astype(np.float64);scale=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-12)
                y=x/scale;return y/np.maximum(np.sqrt(np.sum(y*y,axis=1,keepdims=True)),1e-12/scale)
            for (ids,values),q in zip(output,queries):
                ids=np.asarray(ids);scores=np.asarray(values);ref=normalize(q)@normalize(docs).T
                assert ids.shape==scores.shape==(len(q),5)
                assert all(len(set(row))==5 for row in ids.tolist()) and np.all((ids>=0)&(ids<len(docs)))
                np.testing.assert_allclose(scores,np.take_along_axis(ref,ids,axis=1),rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(scores,-np.sort(-ref,axis=1)[:,:5],rtol=2e-5,atol=2e-6)
        # Install before import/warmup so cached pipelines keep their real names.
        policy=plan.TransactionPlan()
        for case in cases:
            docs,queries=request(case,50310);reference(policy.search(docs,queries),docs,queries)
        mx.synchronize()
        tag=1
        for seed in (50311,50312):
            for case in cases:
                docs,queries=request(case,seed);before=[x.copy() for x in (docs,*queries)];mx.synchronize()
                tag+=1;memory_before={'active':mx.get_active_memory(),'cache':mx.get_cache_memory()}
                output,events=collect(tag,lambda:policy.search(docs,queries))
                memory_after={'active':mx.get_active_memory(),'cache':mx.get_cache_memory()}
                reference(output,docs,queries)
                for actual,saved in zip((docs,*queries),before):np.testing.assert_array_equal(actual,saved)
                kernels=[e for e in events if e['kind'] in (5,6)]
                if not kernels or any(e['name']=='unknown' for e in kernels):raise RuntimeError('Unresolved compute entry point')
                arrays=[docs,*queries]
                explicit=[x for x in arrays if any(s<0 for s in x.strides) or not x.flags.aligned or not x.flags.writeable]
                def span(x):
                    if any(s<0 for s in x.strides) or not x.flags.aligned or not x.flags.writeable:return x.nbytes
                    return x.itemsize+sum((n-1)*s for n,s in zip(x.shape,x.strides))
                state['samples'].append({'case':case[0],'seed':seed,'rows':case[1],'width':case[2],
                    'query_rows':[case[3]]*3,'document_strides':list(docs.strides),
                    'host_input_logical_bytes':sum(x.nbytes for x in arrays),
                    'host_adapter_copy_calls':len(explicit),'host_adapter_copy_logical_bytes':sum(x.nbytes for x in explicit),
                    'mlx_host_constructor_calls':4,'mlx_host_constructor_source_span_bytes':sum(span(x) for x in arrays),
                    'output_numeric_payload_bytes':3*case[3]*5*8,'finite_scalar_payload_bytes':2,
                    'memory_before':memory_before,'memory_after':memory_after,'events':events,
                    'compute_dispatches':len(kernels),'kernel_counts':dict(collections.Counter(e['name'] for e in kernels)),
                    'command_buffers':sum(e['kind']==7 for e in events),'blit_copy_calls':sum(e['kind']==4 for e in events),
                    'blit_copy_bytes':sum(e['a'] for e in events if e['kind']==4),
                    'numerical_float64_and_id_association':True,'inputs_preserved':True,'decision':policy.last_decision.implementation})
                if mx.get_peak_memory()>2*1024**3:raise RuntimeError('2GiB diagnostic cap')
        if observer.kd_dropped():raise RuntimeError('Counter records dropped')
        if digest(__file__)!=pin['script_sha256'] or plan.source_identity()!=pin['operation_source_sha256']:
            raise ValueError('Source changed during diagnostic')
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
        state['wall_seconds']=time.perf_counter()-start
        exclusive('result.json',state)
        print(json.dumps({k:v for k,v in state.items() if k not in ('samples','positive_control','environment')},indent=2))


if __name__=='__main__':main()
