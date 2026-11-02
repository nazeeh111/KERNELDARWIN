"""Owned-buffer library composition diagnostic, not custom selection/acceptance."""
from pathlib import Path
import ctypes as C
import hashlib
import importlib.util
import json
import random
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
composition=module('composition',HERE/'probe.py')
study=composition.study
timer=module('owned_timer',ROOT/'.work/rank-device-timing-20261101/probe.py')

def main():
    start=time.perf_counter()
    state={'kind':'owned_buffer_library_transaction_composition_diagnostic',
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'composition_sha256':hashlib.sha256((HERE/'probe.py').read_bytes()).hexdigest(),
           'candidate_kernel_timing':False,'acceptance_study':False,'samples':[],'checks':[]}
    timer.exclusive(HERE/'owned-buffers-started.json',state)
    mx=None;observer=None;installed=False;tag=0
    try:
        import numpy as np
        mx=study.setup()
        observer=C.CDLL('/private/tmp/kd-buffer-timer-20261101.dylib')
        observer.kd_begin.argtypes=[C.c_uint64]
        observer.kd_read.argtypes=[C.POINTER(timer.Record),C.c_size_t]
        observer.kd_read.restype=C.c_size_t
        for name in ('kd_pending','kd_committed','kd_dropped'):getattr(observer,name).restype=C.c_uint64
        observer.kd_class.restype=C.c_char_p
        state['telemetry_before']=study.telemetry()
        texts=study.corpus()['chunks']
        tasks=[{'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},
               {'id':'manuals','rows':min(256,len(texts)),'width':768,'queries':8,'layout':'column_stride2'},
               {'id':'archive','rows':len(texts),'width':1536,'queries':17,'layout':'transpose'}]
        methods=list(study.LIBRARIES)
        baselines={name:study.StudyMethod(name) for name in methods}
        prepared=[study.arrays(task,texts,42200) for task in tasks]
        def execute(method,which):
            return [composition.execute(method,task,*prepared[i]) if which=='transaction'
                    else study.application(baselines[method],task,*prepared[i]) for i,task in enumerate(tasks)]
        # Compilation and warmup precede observation for both compositions.
        for method in methods:
            for which in ('sequential','transaction'):
                for _ in range(5):execute(method,which)
        assert observer.kd_install()==0;installed=True
        state['observed_command_buffer_class']=observer.kd_class().decode()
        rng=random.Random(42317)
        for repeat in range(40):
            prepared=[study.arrays(task,texts,42201+repeat) for task in tasks]
            order_methods=methods.copy();rng.shuffle(order_methods)
            for method in order_methods:
                order=['sequential','transaction'];rng.shuffle(order);order+=order[::-1]
                outputs={}
                for which in order:
                    if time.perf_counter()-start>60:raise TimeoutError('60-second cap; no retry')
                    if mx.get_peak_memory()>2*1024**3:raise RuntimeError('2GiB cap exceeded')
                    mx.synchronize();tag+=1;before=observer.kd_committed()
                    assert observer.kd_begin(tag)==0
                    begin=time.perf_counter_ns()
                    output=execute(method,which)
                    mx.synchronize()
                    wall=(time.perf_counter_ns()-begin)/1000
                    observer.kd_end()
                    deadline=time.perf_counter()+1
                    while observer.kd_pending():
                        if time.perf_counter()>deadline:raise RuntimeError('Observer did not drain')
                        time.sleep(.0001)
                    buf=(timer.Record*8192)();n=observer.kd_read(buf,8192)
                    assert n==observer.kd_committed()-before and n<=8192 and observer.kd_dropped()==0
                    records=[{name:getattr(r,name) for name,_ in timer.Record._fields_} for r in buf[:n]]
                    assert all(r['tag']==tag and r['status']==4 and r['gpu_start']>0 and r['gpu_end']>=r['gpu_start'] for r in records)
                    state['samples'].append({'repeat':repeat,'input_seed':42201+repeat,'method':method,
                        'method_order':order_methods,'composition':which,'composition_order':order,
                        'wall_us':wall,'command_buffers':n,'gpu_buffer_union_us':timer.union_us(records),'records':records})
                    outputs[which]=output
                for i,task in enumerate(tasks):
                    for (_,a),(_,b) in zip(outputs['sequential'][i],outputs['transaction'][i]):
                        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
                    if repeat in (0,39):
                        for which in ('sequential','transaction'):
                            study.check_application(outputs[which][i],*prepared[i])
                            state['checks'].append({'repeat':repeat,'method':method,'task':task['id'],'composition':which,'float64_and_id_association':True})
        state['completed']=True
    except Exception as error:
        state.update(completed=False,error=type(error).__name__+': '+str(error));raise
    finally:
        if observer is not None:observer.kd_end()
        if mx is not None:mx.synchronize();state['peak_memory_bytes']=mx.get_peak_memory()
        if installed:
            deadline=time.perf_counter()+1
            while observer.kd_pending() and time.perf_counter()<deadline:time.sleep(.0001)
            state.update(observer_uninstall_result=observer.kd_uninstall(),pending_callbacks=observer.kd_pending(),dropped_records=observer.kd_dropped())
        state.update(wall_seconds=time.perf_counter()-start,telemetry_after=study.telemetry())
        timer.exclusive(HERE/'owned-buffers-result.json',state)
        print(json.dumps({k:v for k,v in state.items() if k not in ('samples','checks','telemetry_before','telemetry_after')},indent=2))

if __name__=='__main__':main()
