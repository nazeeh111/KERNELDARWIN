"""One library-only long-window ABBA diagnostic, never an acceptance study."""
from pathlib import Path
import hashlib
import importlib.util
import json
import random
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rank_study',ROOT/'scripts/rank_study.py')
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

def save(name,value):
    with (HERE/name).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    start=time.perf_counter()
    state={'kind':'library_only_long_window_counterbalance_diagnostic','schema':1,
           'candidate_timing':False,'acceptance_study':False,
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'measurement_sha256':hashlib.sha256((ROOT/'scripts/rank_study.py').read_bytes()).hexdigest(),
           'rank_bursts_per_long_window':32,'application_blocks16_per_long_window':2,
           'rounds':40,'rows':[],'stage':'setup'}
    save('counterbalanced-started.json',state)
    mx=None
    def budget(stage):
        state['stage']=stage
        if time.perf_counter()-start>120:raise TimeoutError('120-second cap; no retry')
        if mx is not None and mx.get_peak_memory()>2*1024**3:raise RuntimeError('2GiB cap exceeded')
    try:
        mx=study.setup()
        state['study_identity']=study.current_implementation()
        state['telemetry_before']=study.telemetry()
        texts=study.corpus()['chunks']
        tasks=[{'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},
               {'id':'manuals','rows':min(256,len(texts)),'width':768,'queries':8,'layout':'column_stride2'},
               {'id':'archive','rows':len(texts),'width':1536,'queries':17,'layout':'transpose'}]
        baseline=study.StudyMethod('stable_library')
        rng=random.Random(40731)
        for task in tasks:
            docs,query=study.arrays(task,texts,40500)
            staged=study.rank_inputs(task,docs,query)
            study.check_rank(study.rank_call(baseline,*staged[0]),staged[0][2])
            for _ in range(5):study.measure_burst(baseline,staged)
            calls_before=baseline.calls
            for repeat in range(40):
                designs=['short_ab','long_abba'];rng.shuffle(designs)
                for design in designs:
                    order=['a','b'];rng.shuffle(order)
                    if design=='long_abba':order+=order[::-1]
                    count=32 if design=='long_abba' else 1
                    row={'phase':'rank','task':task['id'],'repeat':repeat,'design':design,
                         'design_order':designs,'arm_order':order,'windows':[]}
                    state['rows'].append(row)
                    for arm in order:
                        inner=[];begin=time.perf_counter_ns()
                        for _ in range(count):
                            inner.append(study.measure_burst(baseline,staged))
                            budget('rank')
                        elapsed=(time.perf_counter_ns()-begin)/1000
                        row['windows'].append({'arm':arm,'outer_wall_us':elapsed,
                            'per_operation_us':elapsed/(count*128),'inner_per_operation_us':inner})
            expected=40*(2+4*32)*128
            if baseline.calls-calls_before!=expected:raise RuntimeError('Rank call count mismatch')
        for repeat in range(40):
            prepared=[study.arrays(task,texts,40600+repeat) for task in tasks]
            # Independent F64 association checks outside the timed windows.
            if repeat in (0,39):
                for i in range(3):study.check_application(study.application(baseline,tasks[i],*prepared[i]),*prepared[i])
            designs=['short_ab','long_abba'];rng.shuffle(designs)
            for design in designs:
                order=['a','b'];rng.shuffle(order)
                if design=='long_abba':order+=order[::-1]
                count=2 if design=='long_abba' else 1
                row={'phase':'application','repeat':repeat,'design':design,'design_order':designs,
                     'input_seed':40600+repeat,'arm_order':order,'windows':[]}
                state['rows'].append(row)
                for arm in order:
                    inner=[];begin=time.perf_counter_ns()
                    for _ in range(count):
                        inner.append(study.measure(lambda:[[study.application(baseline,tasks[i],*prepared[i])
                                      for i in range(3)] for _ in range(16)])/16)
                        budget('application')
                    elapsed=(time.perf_counter_ns()-begin)/1000
                    row['windows'].append({'arm':arm,'outer_wall_us':elapsed,
                        'per_sequence_us':elapsed/(count*16),'inner_per_sequence_us':inner})
        state['completed']=True
        budget('complete')
    except Exception as error:
        state.update(completed=False,error=type(error).__name__+': '+str(error))
        raise
    finally:
        state['wall_seconds']=time.perf_counter()-start
        state['peak_memory_bytes']=mx.get_peak_memory() if mx is not None else None
        state['telemetry_after']=study.telemetry()
        save('counterbalanced-result.json',state)
        print(json.dumps({k:v for k,v in state.items() if k not in ('rows','telemetry_before','telemetry_after')},indent=2))

if __name__=='__main__':main()
