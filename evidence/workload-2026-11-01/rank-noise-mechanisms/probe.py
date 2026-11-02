"""One bounded library-only experiment; no candidate timing or acceptance."""
from pathlib import Path
import ctypes as C
import hashlib
import importlib.util
import json
import math
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'result.json'
LIB = Path('/private/tmp/kd-buffer-timer-20261101.dylib')
spec = importlib.util.spec_from_file_location('rank_study', ROOT / 'scripts/rank_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

class Record(C.Structure):
    _fields_ = [('tag', C.c_uint64), ('sequence', C.c_uint64), ('commit_ns', C.c_uint64),
                ('gpu_start', C.c_double), ('gpu_end', C.c_double), ('status', C.c_uint64)]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def exclusive(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def union_us(rows):
    intervals = sorted((r['gpu_start'], r['gpu_end']) for r in rows
                       if r['gpu_start'] > 0 and r['gpu_end'] > r['gpu_start'])
    if not intervals:
        raise RuntimeError('No positive completed GPU execution timestamps')
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end-start for start,end in merged)*1e6

def main():
    started = time.perf_counter()
    state = {'kind': 'library_only_owned_command_buffer_timing_diagnostic', 'schema': 1,
             'candidate_timing': False, 'acceptance_study': False, 'samples': [],
             'source_sha256': {p.name: sha(p) for p in (Path(__file__), HERE/'buffer_timer.mm', LIB)},
             'measurement_sha256': sha(ROOT/'scripts/rank_study.py'),
             'limits': {'wall_seconds': 120, 'memory_bytes': 2*1024**3, 'records_per_sample': 8192},
             'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    exclusive(HERE/'started.json', state)
    mx = None
    observer = None
    installed = False
    tag = 0
    def budget():
        if time.perf_counter()-started > 120:
            raise TimeoutError('Diagnostic 120-second cap reached; no retry')
        if mx is not None and mx.get_peak_memory() > 2*1024**3:
            raise RuntimeError('Diagnostic memory cap reached')
    try:
        mx = study.setup()
        state['study_identity'] = study.current_implementation()
        state['telemetry_before'] = study.telemetry()
        observer = C.CDLL(str(LIB))
        observer.kd_begin.argtypes = [C.c_uint64]
        observer.kd_read.argtypes = [C.POINTER(Record), C.c_size_t]
        observer.kd_read.restype = C.c_size_t
        observer.kd_class.restype = C.c_char_p
        for name in ('kd_pending','kd_dropped','kd_committed'):
            getattr(observer,name).restype = C.c_uint64
        assert observer.kd_install() == 0
        installed = True
        state['observed_command_buffer_class'] = observer.kd_class().decode()
        data = study.corpus()['chunks']
        tasks = [
            {'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},
            {'id':'manuals','rows':min(256,len(data)),'width':768,'queries':8,'layout':'column_stride2'},
            {'id':'archive','rows':len(data),'width':1536,'queries':17,'layout':'transpose'}]
        baseline = study.StudyMethod('stable_library')
        def sample(function, label, capture=True, delay=0.0):
            nonlocal tag
            budget()
            mx.synchronize()
            tag += 1
            before = observer.kd_committed()
            def call():
                if capture and observer.kd_begin(tag) != 0:
                    raise RuntimeError('Overlapping capture or pending callbacks')
                if delay:
                    time.sleep(delay)
                return function()
            cpu = time.thread_time_ns()
            wall = study.measure(call)
            cpu = (time.thread_time_ns()-cpu)/1000
            observer.kd_end()
            deadline = time.perf_counter()+1
            while observer.kd_pending():
                if time.perf_counter()>deadline:
                    raise RuntimeError('Completion observer failed to drain')
                time.sleep(.0001)
            buf = (Record*8192)()
            count = observer.kd_read(buf,8192) if capture else 0
            committed = observer.kd_committed()-before
            if capture and (count != committed or count > 8192 or observer.kd_dropped()):
                raise RuntimeError('Incomplete, overflowing or dropped command-buffer records')
            rows = [{name:getattr(r,name) for name,_ in Record._fields_} for r in buf[:count]]
            if any(r['tag']!=tag or r['status']!=4 or not all(math.isfinite(r[k]) for k in ('gpu_start','gpu_end'))
                   or r['gpu_start']<0 or r['gpu_end']<r['gpu_start'] for r in rows):
                raise RuntimeError('Bad tag/status/timestamp')
            if len({r['sequence'] for r in rows}) != count:
                raise RuntimeError('Duplicate command-buffer sequence')
            row = {'label':label,'wall_us':wall,'current_thread_cpu_us':cpu,'capture':capture,
                   'injected_host_sleep_seconds':delay,'records':rows,'committed':committed,
                   'zero_timestamp_buffers':sum(r['gpu_start']==0 or r['gpu_end']==0 for r in rows)}
            if capture:
                row['gpu_buffer_union_us'] = union_us(rows)
                row['gpu_buffer_sum_us'] = sum(r['gpu_end']-r['gpu_start'] for r in rows)*1e6
                positive=[r for r in rows if r['gpu_start']>0 and r['gpu_end']>r['gpu_start']]
                row['gpu_buffer_extent_us']=(max(r['gpu_end'] for r in positive)-min(r['gpu_start'] for r in positive))*1e6
            state['samples'].append(row)
            return row
        rng=random.Random(39217)
        for task in tasks:
            docs,query=study.arrays(task,data,39000)
            staged=study.rank_inputs(task,docs,query)
            study.check_rank(study.rank_call(baseline,*staged[0]),staged[0][2])
            for _ in range(5):study.measure_burst(baseline,staged)
            fn=lambda:[study.rank_call(baseline,*inputs) for inputs in staged]
            for repeat in range(40):
                order=['a','b'];rng.shuffle(order)
                for arm in order:sample(fn,{'phase':'rank_control','task':task['id'],'repeat':repeat,'arm':arm,'order':order})
            if task['id']=='notes':
                for repeat in range(10):
                    order=[0.0,.005];rng.shuffle(order)
                    for delay in order:sample(fn,{'phase':'host_delay_check','repeat':repeat,'order':order},delay=delay)
                for repeat in range(20):
                    order=[False,True];rng.shuffle(order)
                    for capture in order:sample(fn,{'phase':'observer_overhead','repeat':repeat,'order':order},capture=capture)
        for repeat in range(20):
            prepared=[study.arrays(task,data,39100+repeat) for task in tasks]
            order=['a','b'];rng.shuffle(order)
            for arm in order:
                sample(lambda:[[study.application(baseline,tasks[i],*prepared[i]) for i in range(3)] for _ in range(16)],
                       {'phase':'application_control','repeat':repeat,'arm':arm,'order':order})
        budget()
        state['completed']=True
    except Exception as error:
        state.update(completed=False,error=type(error).__name__+': '+str(error))
        raise
    finally:
        if observer is not None:
            observer.kd_end()
        if mx is not None:
            mx.synchronize()
            state['peak_memory_bytes']=mx.get_peak_memory()
        if installed:
            deadline=time.perf_counter()+1
            while observer.kd_pending() and time.perf_counter()<deadline:time.sleep(.0001)
            state['observer_uninstall_result']=observer.kd_uninstall()
            state['dropped_records']=observer.kd_dropped()
            state['pending_callbacks']=observer.kd_pending()
        state['wall_seconds']=time.perf_counter()-started
        state['telemetry_after']=study.telemetry()
        exclusive(OUT,state)
        print(json.dumps({k:v for k,v in state.items() if k not in ('samples','telemetry_before','telemetry_after')},indent=2))

if __name__=='__main__':
    main()
