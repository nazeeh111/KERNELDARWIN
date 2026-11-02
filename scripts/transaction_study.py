#!/usr/bin/env python3
"""One bounded host-origin batch-transaction study; no stage runs on import."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
HELPER_SHA='4470b506443feea5cf6a4f22b5fea01338e8a199621b02dd0f45f20f114f5b3a'
CORPUS_SHA='c066057478c35531b5fbe71dc5716d2868c85556ddd36b6da6c9439e1c1d70a9'
helper_path=ROOT/'scripts/rank_study.py'
if hashlib.sha256(helper_path.read_bytes()).hexdigest()!=HELPER_SHA:
    raise ValueError('Legacy numerical/timing helper changed; preserve the frozen dependency')
spec=importlib.util.spec_from_file_location('transaction_legacy_helpers',helper_path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from kerneldarwin import transaction_plan as engine
from kerneldarwin import transaction_search as executor
from kerneldarwin import rank_search as rank

EVIDENCE=ROOT/'evidence/workload-2026-11-02/transaction-v1'
MEASUREMENT_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DESIGN='host_batch_palindromic_long_windows_v1'
LIBRARIES=executor.LIBRARIES
METHODS=(*LIBRARIES,'best_fixed','per_shape','dispatcher','sequential_stable')
CANONICAL=('stable_library','stable_reduction')
GLOBAL_NAMES=(*LIBRARIES,*executor.CANDIDATES)
RANK_ARMS=(*GLOBAL_NAMES,*('plan:'+name for name in executor.CANDIDATES))
APP_ARMS=(*('global:'+name for name in GLOBAL_NAMES),*('routed:'+name for name in GLOBAL_NAMES),
          *('plan:'+name for name in executor.CANDIDATES),'plan_library')
UNCERTAINTY=dict(base.UNCERTAINTY)
LIMITS={'freeze_wall_seconds':180,'development_wall_seconds':900,'confirmation_wall_seconds':600,
        'cold_study_wall_seconds':180,'memory_bytes':2*1024**3,'cache_bytes':256*1024**2,
        'source_bytes':12*1024,'portfolio_kernels':2,'max_rules':128}
WINDOWS={'rank_bursts':32,'rank_inputs_per_burst':128,'application_blocks':2,'sequences_per_block':16}
CONFIRM={'sessions':3,'repeats':40,'seeds':[46101,47203,48307],
         'sequence_orders':[[1,0,3,2],[2,3,0,1],[0,2,1,3]],'minimum_separation_seconds':120,
         'unseen_task':{'id':'unseen','rows':113,'width':1153,'queries':11,'layout':'host_column_stride2'}}
CALIBRATION={'seed':44100,'application_seeds':list(range(44101,44141)),
             'rank_order_seed':44211,'application_order_seed':44213,
             'application_orders':[[0,1,2],[1,2,0],[2,0,1]]*13+[[0,1,2]],
             'equality_rank_labels':[n for n in METHODS if n!='dispatcher'],
             'equality_application_labels':[n for n in METHODS if n not in ('dispatcher','sequential_stable')]}
DEVELOPMENT={'seed':45104,'repeats':40,'order_seed':45207,'rank_arms':list(RANK_ARMS),'application_arms':list(APP_ARMS)}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((EVIDENCE/name).read_text())
def save(name,value):base.save(EVIDENCE/name,value)


def reserve(stage,outcomes):
    for name in outcomes:
        if (EVIDENCE/name).exists():raise FileExistsError('Preserve prior study outcome: '+name)
    save(stage+'-started.json',{'stage':stage,'script_sha256':MEASUREMENT_SHA,'pid':os.getpid(),
         'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'unix_ns':time.time_ns(),
         'monotonic_ns':time.monotonic_ns()})


def implementation():
    pin=json.loads((Path(engine.__file__).parent/'transaction-study-source.json').read_text())
    if pin.get('measurement_script_sha256')!=MEASUREMENT_SHA or digest(__file__)!=MEASUREMENT_SHA:
        raise ValueError('Batch measurement source differs from installed pin')
    if digest(helper_path)!=HELPER_SHA:raise ValueError('Frozen helper changed')
    if pin.get('measurement_helper_sha256')!=HELPER_SHA:raise ValueError('Measurement helper pin changed')
    return engine.study_identity()


def numerical():
    data=read('numerical-validation.json')
    pin=json.loads((Path(engine.__file__).parent/'transaction-study-source.json').read_text())
    if (data.get('kind')!='transaction_plan_production_numerical_validation'
            or data.get('passed') is not True or data.get('gpu_executed') is not True
            or data.get('source_sha256')!=engine.source_identity() or data.get('study_identity')!=implementation()
            or data.get('environment')!=engine.environment_identity()
            or any(data.get(k)!=0 for k in ('failures','errors','skips'))
            or type(data.get('tests_run')) is not int or data['tests_run']<28
            or not isinstance(data.get('native_metal_methods_completed'),list)
            or len(set(data['native_metal_methods_completed']))<14
            or data.get('producer_sha256')!=pin['numerical_producer_sha256']
            or data.get('test_sha256')!=pin['numerical_tests_sha256']
            or data.get('validation_root_src_exists') is not False
            or not isinstance(data.get('actual_imports'),dict) or len(data['actual_imports'])<7
            or any('/site-packages/' not in v.get('path','') for v in data['actual_imports'].values())
            or any(str(data.get('validation_flags',{}).get(k)).lower() not in ('1','true')
                   for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'))):
        raise ValueError('Current independent installed native batch-plan numerical report required')
    log=engine._read_regular(EVIDENCE/'native-validation.log',2_000_000).decode(errors='replace')
    if 'Metal API Validation Enabled' not in log or 'Metal GPU Validation Enabled' not in log:
        raise ValueError('Actual native API/GPU validation log required')
    return data


def source_metadata():
    sources={name:hashlib.sha256(data).hexdigest() for name,data in rank._SOURCES}
    if any(digest(Path(rank.__file__).parent/'metal'/name)!=sha for name,sha in sources.items()):
        raise ValueError('Loaded readable rank source changed')
    size=sum(len(data) for _,data in rank._SOURCES)
    if size>LIMITS['source_bytes'] or executor.CANDIDATES!=('simd32','group128'):
        raise ValueError('Two-kernel/source budget changed')
    return sources,size


def tasks_for(corpus):
    n=len(corpus['chunks'])
    if n<113:raise ValueError('Licensed corpus needs at least113 chunks')
    return [{'id':'notes','rows':64,'width':384,'queries':1,'layout':'host_contiguous'},
            {'id':'manuals','rows':min(256,n),'width':768,'queries':8,'layout':'host_column_stride2'},
            {'id':'archive','rows':n,'width':1536,'queries':17,'layout':'host_transpose'}]


def key_of(task):return engine.rule_key([task['queries']]*3,task['rows'],task['width'],5,task['layout'])


def rule(key,name):
    rows,n,d,k,layout=key
    return {'shape':[*rows,n,d,k],'document_layout':layout,'query_layout':'host_contiguous',
            'dtype':'float32','candidate':name}


def host_layout(array,layout):
    import numpy as np
    if layout=='host_contiguous':return np.ascontiguousarray(array)
    if layout=='host_transpose':return array.T.copy().T
    if layout=='host_column_stride2':
        backing=np.zeros((array.shape[0],array.shape[1]*2),array.dtype);backing[:,::2]=array
        return backing[:,::2]
    raise ValueError('Unknown host layout')


def request(task,texts,seed):
    import numpy as np
    documents,queries=base.arrays(task,texts,seed)
    return host_layout(documents,task['layout']),[np.ascontiguousarray(queries+j*.001) for j in range(3)]


def check_application(output,prepared):
    documents,queries=prepared
    # The numerical helper expects the first query and original three offsets.
    base.check_application(output,documents,queries[0])


def staged_rank(task,texts,seed):
    import mlx.core as mx
    import numpy as np
    from kerneldarwin.library_baselines import eager_normalize
    docs,query=base.arrays(task,texts,seed);staged=[]
    for i in range(128):
        hd=host_layout(docs+(i+1)*.0007,task['layout'])
        hq=[np.ascontiguousarray(query+j*.001+(i+1)*.0011) for j in range(3)]
        signature=engine.host_signature(hd,hq,5)
        if signature!=key_of(task):raise ValueError('Staging layout differs from declared profile')
        d,q=engine.prepare(hd,hq)
        scores=eager_normalize(mx.concatenate(q,axis=0))@eager_normalize(d).T
        mx.eval(scores)
        staged.append((signature,scores))
    return staged


class StudyMethod:
    def __init__(self,name='stable_library',rules=None):
        self.name,self.rules=name,rules;self.calls=self.fallback_calls=0
    def choose(self,signature):
        self.calls+=1
        name=self.rules.get(signature,'stable_library') if self.rules is not None else self.name
        if name not in executor.CANDIDATES:self.fallback_calls+=1
        return name
    def rank(self,signature,scores):
        name=self.choose(signature)
        function=(rank.compiled_rank(name) if name in executor.CANDIDATES else
                  rank.library_reduction_rank() if name.endswith('reduction') else rank.library_rank())
        return function(scores)
    def search(self,documents,queries):
        signature=engine.host_signature(documents,queries,5) if self.rules is not None else None
        name=self.choose(signature)
        if name=='sequential_stable':
            from kerneldarwin.search import CompiledCosineIndex
            d,q=engine.prepare(documents,queries);index=CompiledCosineIndex(d,'stable')
            return [index.search(batch,k=5) for batch in q]
        return engine.execute_host(name,documents,queries)


def rank_call(method,signature,scores):
    return method._rank_trial(signature,scores) if isinstance(method,engine.TransactionPlan) else method.rank(signature,scores)


def window_rank(method,staged):
    if len(staged)!=128:raise ValueError('Exactly128 evaluated changed rank inputs required')
    start=time.perf_counter_ns();inner=[]
    for _ in range(32):inner.append(base.measure(lambda:[rank_call(method,*item) for item in staged])/128)
    outer=(time.perf_counter_ns()-start)/1000
    return {'outer_us':outer,'inner_us':inner,'cost_us':outer/(32*128)}


def window_application(method,prepared,order):
    start=time.perf_counter_ns();inner=[]
    for _ in range(2):
        inner.append(base.measure(lambda:[[method.search(*prepared[i]) for i in order] for _ in range(16)])/16)
    outer=(time.perf_counter_ns()-start)/1000
    return {'outer_us':outer,'inner_us':inner,'cost_us':outer/32}


def group(methods,function,rng):
    order=list(methods);rng.shuffle(order);order+=order[::-1]
    raw={name:[] for name in methods}
    for name in order:raw[name].append(function(methods[name]))
    return order,raw


def folded(records):return [statistics.mean(w['cost_us'] for w in row) for row in records]


def manifest(protocol,rules):
    return {'schema':1,'operation':engine.OPERATION,'epsilon':engine.EPSILON,'key_policy':rank.KEY_POLICY,
        'compile_options':{'math_mode':'safe'},'source_sha256':protocol['source_sha256'],
        'study_identity':protocol['study_identity'],'environment':protocol['environment'],
        'sources':protocol['sources'],'rules':rules,
        'evidence':{'numerical_passed':True,'confirmation_passed':False}}


def trial(protocol,rules):return engine.TransactionPlan._trial(manifest(protocol,rules))


def verify_frozen(policy=None):
    if (EVIDENCE/'rejected-baseline.json').exists() or (policy is not None and (EVIDENCE/'development-failed.json').exists()):
        raise ValueError('Preserve failed stage; no continuation or export')
    protocol=read('protocol.json')
    if (protocol.get('schema')!=1 or protocol.get('target')!=engine.OPERATION or protocol.get('design')!=DESIGN
            or protocol['limits']!=LIMITS or protocol['windows']!=WINDOWS or protocol['uncertainty']!=UNCERTAINTY
            or protocol['confirmation']!=CONFIRM or protocol['calibration']!=CALIBRATION or protocol['development']!=DEVELOPMENT
            or protocol['source_sha256']!=engine.source_identity() or protocol['study_identity']!=implementation()
            or protocol['environment']!=engine.environment_identity() or protocol['script_sha256']!=MEASUREMENT_SHA
            or protocol['helper_sha256']!=HELPER_SHA or protocol['sources']!=source_metadata()[0]
            or protocol['corpus_sha256']!=digest(EVIDENCE/'corpus.json')
            or protocol['corpus_sha256']!=digest(ROOT/'evidence/workload-2026-10-29/corpus.json')
            or protocol['corpus_sha256']!=CORPUS_SHA
            or protocol['tasks']!=tasks_for(read('corpus.json'))
            or protocol['numerical_report_sha256']!=digest(EVIDENCE/'numerical-validation.json')):
        raise ValueError('Frozen batch protocol/source/environment/numerical evidence changed')
    if protocol['native_validation_log_sha256']!=digest(EVIDENCE/'native-validation.log'):
        raise ValueError('Native numerical validation log changed')
    numerical()
    completion=read('freeze-completed.json')
    if completion['protocol_sha256']!=digest(EVIDENCE/'protocol.json') or not 0<completion['wall_seconds']<=180:
        raise ValueError('Calibration completion checkpoint changed or exceeded budget')
    expected_rank=base.symmetric_noise_floor([c for row in protocol['rank_controls']
        for c in row['comparisons'].values()],1.10)
    expected_app=base.symmetric_noise_floor(list(protocol['application_control']['comparisons'].values()),1.03)
    if protocol['criterion']!={'rank_ci95_lower_speedup':expected_rank,
            'application_ci95_lower_speedup':expected_app,
            'batching_vs_sequential_ci95_lower_speedup':expected_app,'second_kernel_gain':.02}:
        raise ValueError('Frozen noise-derived criterion changed')
    if policy is not None and (policy['study_identity']!=protocol['study_identity']
            or policy['source_sha256']!=protocol['source_sha256'] or policy['environment']!=protocol['environment']
            or any(policy['evidence'][key]!=digest(EVIDENCE/name) for key,name in
                   (('protocol_sha256','protocol.json'),('development_sha256','development.json'),
                    ('corpus_sha256','corpus.json'),('numerical_report_sha256','numerical-validation.json')))):
        raise ValueError('Frozen batch policy changed')
    if policy is not None:
        checkpoint=read('policy-checkpoint.json')
        completion=read('tune-completed.json')
        if (checkpoint['policy_sha256']!=digest(EVIDENCE/'policy-frozen.json')
                or checkpoint['protocol_sha256']!=digest(EVIDENCE/'protocol.json')
                or checkpoint['development_sha256']!=digest(EVIDENCE/'development.json')
                or completion['policy_sha256']!=digest(EVIDENCE/'policy-frozen.json')
                or not 0<completion['wall_seconds']<=900):
            raise ValueError('Frozen policy checkpoint changed')
    return protocol


def freeze():
    start=time.perf_counter();reserve('freeze',('protocol.json','rejected-baseline.json'))
    state={'schema':1,'design':DESIGN,'script_sha256':MEASUREMENT_SHA,'rank_controls':[],'stage':'setup'}
    mx=None
    def budget(stage):
        state['stage']=stage
        if time.perf_counter()-start>180:raise TimeoutError('180-second calibration cap; no retry')
        if mx is not None and mx.get_peak_memory()>LIMITS['memory_bytes']:raise RuntimeError('Calibration memory cap')
    try:
        mx=base.setup();identity=implementation();numerical();sources,size=source_metadata()
        state['telemetry_before']=base.telemetry();corpus=base.corpus();tasks=tasks_for(corpus)
        if digest(ROOT/'evidence/workload-2026-10-29/corpus.json')!=CORPUS_SHA:
            raise ValueError('Canonical licensed corpus changed')
        baseline=StudyMethod();rng=random.Random(CALIBRATION['rank_order_seed'])
        controls={name:baseline for name in METHODS}
        for task in tasks:
            staged=staged_rank(task,corpus['chunks'],44100);budget('rank_staging')
            first_ms=base.measure(lambda:rank_call(baseline,*staged[0]))/1000
            base.check_rank(rank_call(baseline,*staged[0]),staged[0][1])
            for _ in range(5):base.measure(lambda:[rank_call(baseline,*item) for item in staged])
            row={'task':task,'windows':{n:[] for n in METHODS},'orders':[],'first_ms':first_ms}
            state['rank_controls'].append(row)
            for _ in range(40):
                order,raw=group(controls,lambda m:window_rank(m,staged),rng);row['orders'].append(order)
                for name,windows in raw.items():row['windows'][name].append(windows)
                budget('rank_controls')
            samples={name:folded(windows) for name,windows in row['windows'].items()}
            row['comparisons']={name:base.intervals(samples[name],samples['dispatcher'])
                                for name in CALIBRATION['equality_rank_labels']}
        controls={name:baseline for name in METHODS};controls['sequential_stable']=StudyMethod('sequential_stable')
        prepared=[request(task,corpus['chunks'],44100) for task in tasks]
        for method in (baseline,controls['sequential_stable']):
            for i in range(3):check_application(method.search(*prepared[i]),prepared[i])
            for _ in range(5):base.measure(lambda:[[method.search(*prepared[i]) for i in range(3)] for _ in range(16)])
        row={'windows':{n:[] for n in METHODS},'orders':[]};state['application_control']=row
        rng=random.Random(CALIBRATION['application_order_seed'])
        for seed,sequence in zip(CALIBRATION['application_seeds'],CALIBRATION['application_orders']):
            prepared=[request(task,corpus['chunks'],seed) for task in tasks]
            order,raw=group(controls,lambda m:window_application(m,prepared,sequence),rng);row['orders'].append(order)
            for name,windows in raw.items():row['windows'][name].append(windows)
            budget('application_controls')
        samples={name:folded(windows) for name,windows in row['windows'].items()}
        row['comparisons']={name:base.intervals(samples[name],samples['dispatcher'])
                            for name in CALIBRATION['equality_application_labels']}
        rank_comparisons=[c for task in state['rank_controls'] for c in task['comparisons'].values()]
        app_comparisons=list(row['comparisons'].values())
        rank_floor=base.symmetric_noise_floor(rank_comparisons,1.10)
        app_floor=base.symmetric_noise_floor(app_comparisons,1.03)
        if any(not .9<=bound<=1.1 for c in app_comparisons for bound in c['ci95']):
            raise RuntimeError('Batched application equality control outside[.9,1.1]; no candidate timing')
        if implementation()!=identity:raise ValueError('Source changed during calibration')
        numerical();budget('final_checks')
        save('corpus.json',corpus)
        protocol={'schema':1,'target':engine.OPERATION,'design':DESIGN,'frozen_at':base.telemetry()['time_utc'],
            'tasks':tasks,'methods':list(METHODS),'libraries':list(LIBRARIES),'canonical_rank_libraries':list(CANONICAL),
            'limits':dict(LIMITS),'windows':dict(WINDOWS),'uncertainty':dict(UNCERTAINTY),
            'confirmation':CONFIRM,'calibration':CALIBRATION,'development':DEVELOPMENT,
            'criterion':{'rank_ci95_lower_speedup':rank_floor,'application_ci95_lower_speedup':app_floor,
                         'batching_vs_sequential_ci95_lower_speedup':app_floor,'second_kernel_gain':.02},
            'source_sha256':engine.source_identity(),'study_identity':identity,'environment':engine.environment_identity(),
            'sources':sources,'source_bytes':size,'script_sha256':MEASUREMENT_SHA,'helper_sha256':HELPER_SHA,
            'corpus_sha256':digest(EVIDENCE/'corpus.json'),'numerical_report_sha256':digest(EVIDENCE/'numerical-validation.json'),
            'native_validation_log_sha256':digest(EVIDENCE/'native-validation.log'),
            'rank_controls':state['rank_controls'],'application_control':row,'peak_memory_bytes':mx.get_peak_memory(),
            'telemetry_before':state['telemetry_before'],'telemetry_after':base.telemetry(),
            'timing_validation_flags':base.timing_flags(),'calibration_wall_seconds':time.perf_counter()-start,
            'scope':'Host NumPy requests at API boundary; includes validation/staging/copies/refresh/threequeries/finite/staticdispatch/hostresults. Input encoding/layout construction separate. Rank-only isolates lookup/rank on evaluated scores and prevalidatedmetadata; not full host validation.',
            'limitations':['Palindrome balances nominal positions; not proven independent environments or drift cancellation.',
                          'IID/block envelopes are sensitivity rules, not guaranteed formal coverage.',
                          'Sequential stable is a batching comparator only; custom acceptance uses all4equallybatched libraries.']}
        budget('save_protocol');save('protocol.json',protocol)
        budget('protocol_serialized')
        save('freeze-completed.json',{'wall_seconds':time.perf_counter()-start,'protocol_sha256':digest(EVIDENCE/'protocol.json')})
        budget('freeze_complete')
        print(json.dumps(protocol['criterion']))
    except Exception as error:
        state.update(reason=type(error).__name__+': '+str(error),wall_seconds=time.perf_counter()-start,
                     peak_memory_bytes=mx.get_peak_memory() if mx is not None else None,telemetry_after=base.telemetry())
        save('rejected-baseline.json',state);raise


def select_policy(protocol,rows):
    cost=lambda row,label:statistics.mean(folded(row['application_windows'][label]))
    eligible=lambda row,name:all(row['comparisons'][name][base]['ci95'][0]
        >protocol['criterion']['rank_ci95_lower_speedup'] for base in CANONICAL)
    fixed=min(GLOBAL_NAMES,key=lambda name:sum(cost(row,'global:'+name) for row in rows))
    per_shape={key_of(row['task']):min(GLOBAL_NAMES,key=lambda name:cost(row,'routed:'+name)) for row in rows}
    possibilities=[]
    for count in (0,1,2):
        for subset in itertools.combinations(executor.CANDIDATES,count):
            chosen,total={},0.
            for row in rows:
                options=['plan_library',*('plan:'+name for name in subset if eligible(row,name))]
                best=min(options,key=lambda label:cost(row,label));total+=cost(row,best)
                if best!='plan_library':chosen[key_of(row['task'])]=best.removeprefix('plan:')
            possibilities.append({'subset':subset,'total_us':total,'rules':chosen})
    one=min((p for p in possibilities if len(p['subset'])<=1),key=lambda p:p['total_us'])
    joint=min(possibilities,key=lambda p:p['total_us'])
    if joint['total_us']>one['total_us']*(1-protocol['criterion']['second_kernel_gain']):joint=one
    return joint['rules'],{'best_fixed':fixed,'joint_subset':sorted(set(joint['rules'].values())),
        'development_application_cost_us':joint['total_us'],'best_one_cost_us':one['total_us'],
        'best_fixed_cost_us':sum(cost(row,'global:'+fixed) for row in rows),
        'per_shape_cost_us':sum(cost(row,'routed:'+per_shape[key_of(row['task'])]) for row in rows),
        'per_shape':[rule(key,name) for key,name in sorted(per_shape.items())]}


def development_methods(protocol,task):
    globals_={name:StudyMethod(name) for name in GLOBAL_NAMES}
    routed={name:StudyMethod(rules={key_of(task):name}) for name in GLOBAL_NAMES}
    plans={name:trial(protocol,[rule(key_of(task),name)]) for name in executor.CANDIDATES}
    return ({**globals_,**{'plan:'+n:p for n,p in plans.items()}},
        {**{'global:'+n:m for n,m in globals_.items()},**{'routed:'+n:m for n,m in routed.items()},
         **{'plan:'+n:p for n,p in plans.items()},'plan_library':trial(protocol,[])})


def tune():
    start=time.perf_counter();reserve('tune',('development.json','development-failed.json','policy-frozen.json','policy-checkpoint.json'))
    mx=base.setup();protocol=verify_frozen();texts=read('corpus.json')['chunks'];rows=[];active={}
    def budget(stage):
        if time.perf_counter()-start>900:raise TimeoutError('900-second development cap; no retry')
        if mx.get_peak_memory()>LIMITS['memory_bytes']:raise RuntimeError('Development memory cap')
    try:
        rng=random.Random(DEVELOPMENT['order_seed'])
        for task in protocol['tasks']:
            ranks,apps=development_methods(protocol,task)
            staged=staged_rank(task,texts,DEVELOPMENT['seed']);prepared=request(task,texts,DEVELOPMENT['seed'])
            active={'task':task,'rank_windows':{n:[] for n in ranks},'application_windows':{n:[] for n in apps},
                    'rank_orders':[],'application_orders':[],'first_rank_ms':{},'first_application_ms':{},'checks':[]}
            for name,method in ranks.items():
                active['first_rank_ms'][name]=base.measure(lambda m=method:rank_call(m,*staged[0]))/1000
                base.check_rank(rank_call(method,*staged[0]),staged[0][1])
                for _ in range(5):base.measure(lambda m=method:[rank_call(m,*item) for item in staged])
                active['checks'].append({'phase':'rank','method':name,'ids_and_original_bits':True})
                budget('rank_warmup')
            for name,method in apps.items():
                active['first_application_ms'][name]=base.measure(lambda m=method:m.search(*prepared))/1000
                check_application(method.search(*prepared),prepared)
                for _ in range(5):base.measure(lambda m=method:m.search(*prepared))
                active['checks'].append({'phase':'application','method':name,'scores_and_id_association':True})
                budget('application_warmup')
            for repeat in range(40):
                seed=DEVELOPMENT['seed']+repeat+1
                staged=staged_rank(task,texts,seed);prepared=request(task,texts,seed)
                order,raw=group(ranks,lambda m:window_rank(m,staged),rng);active['rank_orders'].append(order)
                for name,windows in raw.items():active['rank_windows'][name].append(windows)
                order,raw=group(apps,lambda m:window_application(m,[prepared],[0]),rng);active['application_orders'].append(order)
                for name,windows in raw.items():active['application_windows'][name].append(windows)
                budget('development_measurement')
            samples={name:folded(windows) for name,windows in active['rank_windows'].items()}
            active['comparisons']={name:{canonical:base.intervals(samples[canonical],samples['plan:'+name])
                for canonical in CANONICAL} for name in executor.CANDIDATES}
            rows.append(active);active={};budget('row_analysis')
            print('development',task['id'],'complete',flush=True)
        rules,selection=select_policy(protocol,rows)
        result={'schema':1,'source_sha256':protocol['source_sha256'],'study_identity':implementation(),
                'environment':protocol['environment'],'protocol_sha256':digest(EVIDENCE/'protocol.json'),
                'corpus_sha256':protocol['corpus_sha256'],'numerical_report_sha256':protocol['numerical_report_sha256'],
                'rows':rows,'search_wall_seconds':time.perf_counter()-start,'peak_memory_bytes':mx.get_peak_memory(),
                'telemetry':base.telemetry(),'timing_validation_flags':base.timing_flags()}
        budget('final_development_checks');save('development.json',result)
        policy=manifest(protocol,[rule(key,name) for key,name in sorted(rules.items())]);policy['policy_selection']=selection
        policy['evidence'].update({k:digest(EVIDENCE/n) for k,n in
            (('protocol_sha256','protocol.json'),('development_sha256','development.json'),
             ('corpus_sha256','corpus.json'),('numerical_report_sha256','numerical-validation.json'))})
        budget('final_policy_save');save('policy-frozen.json',policy)
        save('policy-checkpoint.json',{'policy_sha256':digest(EVIDENCE/'policy-frozen.json'),
             'protocol_sha256':digest(EVIDENCE/'protocol.json'),'development_sha256':digest(EVIDENCE/'development.json')})
        budget('policy_serialized')
        save('tune-completed.json',{'wall_seconds':time.perf_counter()-start,'policy_sha256':digest(EVIDENCE/'policy-frozen.json')})
        budget('tune_complete')
        print(json.dumps({'rules':policy['rules'],'selection':selection},indent=2))
    except Exception as error:
        save('development-failed.json',{'reason':type(error).__name__+': '+str(error),'completed_rows':rows,
             'partial':active,'wall_seconds':time.perf_counter()-start,'peak_memory_bytes':mx.get_peak_memory()})
        raise


def policies(protocol,policy,requested=None):
    result={}
    for name in METHODS if requested is None else (requested,):
        if name=='dispatcher':result[name]=engine.TransactionPlan._trial(policy)
        elif name=='best_fixed':result[name]=StudyMethod(policy['policy_selection']['best_fixed'])
        elif name=='per_shape':
            rules={engine.rule_key(row['shape'][:-3],*row['shape'][-3:],row['document_layout']):row['candidate']
                   for row in policy['policy_selection']['per_shape']}
            result[name]=StudyMethod(rules=rules)
        else:result[name]=StudyMethod(name)
    return result


def session_preflight(session,protocol,policy,start_unix,start_monotonic):
    if session not in (0,1,2):raise ValueError('Session must be0,1or2')
    if session==0:return
    previous=read(f'confirmation-{session-1}.json');completion=read(f'confirmation-{session-1}-completed.json')
    if (previous['session']!=session-1 or previous['policy_sha256']!=digest(EVIDENCE/'policy-frozen.json')
            or previous['protocol_sha256']!=digest(EVIDENCE/'protocol.json')
            or completion['result_sha256']!=digest(EVIDENCE/f'confirmation-{session-1}.json')
            or (EVIDENCE/f'confirmation-{session-1}-failed.json').exists()):
        raise ValueError('Previous confirmation belongs to another frozen study')
    for now,key in ((start_unix,'session_end_unix_ns'),(start_monotonic,'session_end_monotonic_ns')):
        if type(completion.get(key)) is not int or now-completion[key]<120*1_000_000_000:
            raise RuntimeError('Confirmation requires120s separation on both clocks; no marker created')


def confirm(session):
    start=time.perf_counter();unix,monotonic=time.time_ns(),time.monotonic_ns()
    policy=read('policy-frozen.json');protocol=read('protocol.json')
    session_preflight(session,protocol,policy,unix,monotonic)
    verify_frozen(policy);reserve(f'confirmation-{session}',(f'confirmation-{session}.json',f'confirmation-{session}-failed.json'))
    mx=base.setup();seed=CONFIRM['seeds'][session];sequence=CONFIRM['sequence_orders'][session]
    tasks=[*protocol['tasks'],CONFIRM['unseen_task']];texts=read('corpus.json')['chunks']
    before=base.telemetry();load=time.perf_counter();methods=policies(protocol,policy)
    load_ms=(time.perf_counter()-load)*1000
    ranks={name:[] for name in methods};apps={name:[] for name in methods}
    rank_orders=[];app_orders=[];first_rank={};first_app={};checks=[]
    def budget():
        if time.perf_counter()-start>600:raise TimeoutError('600-second confirmation cap; no retry')
        if mx.get_peak_memory()>LIMITS['memory_bytes']:raise RuntimeError('Confirmation memory cap')
    try:
        prepared=[request(task,texts,seed) for task in tasks]
        staged=[staged_rank(task,texts,seed) for task in tasks]
        for name,method in methods.items():
            first_rank[name]=[];first_app[name]=[]
            for i,task in enumerate(tasks):
                first_rank[name].append(base.measure(lambda m=method:rank_call(m,*staged[i][0]))/1000)
                base.check_rank(rank_call(method,*staged[i][0]),staged[i][0][1])
                first_app[name].append(base.measure(lambda m=method:m.search(*prepared[i]))/1000)
                check_application(method.search(*prepared[i]),prepared[i])
                checks.append({'task':task['id'],'method':name,'rank_ids_bits':True,'scores_id_association':True})
                budget()
        rng=random.Random(seed)
        for repeat in range(40):
            prepared=[request(task,texts,seed+repeat+1) for task in tasks]
            staged=[staged_rank(task,texts,seed+repeat+1) for task in tasks]
            def rank_sequence(method):
                # Keep per-task raw windows; sequence cost is their sum.
                windows=[window_rank(method,staged[i]) for i in sequence]
                return {'task_windows':windows,'cost_us':sum(w['cost_us'] for w in windows)}
            order,raw=group(methods,rank_sequence,rng);rank_orders.append(order)
            for name,windows in raw.items():ranks[name].append(windows)
            order,raw=group(methods,lambda m:window_application(m,prepared,sequence),rng);app_orders.append(order)
            for name,windows in raw.items():apps[name].append(windows)
            budget()
            if (repeat+1)%10==0:print('confirmation',session,'group',repeat+1,flush=True)
        rank_samples={n:folded(v) for n,v in ranks.items()};app_samples={n:folded(v) for n,v in apps.items()}
        comparisons={name:{canonical:{'rank':base.intervals(rank_samples[canonical],rank_samples[name],seed),
            'application':base.intervals(app_samples[canonical],app_samples[name],seed)} for canonical in LIBRARIES} for name in METHODS}
        gates={'rank':all(comparisons['dispatcher'][c]['rank']['ci95'][0]>protocol['criterion']['rank_ci95_lower_speedup'] for c in CANONICAL),
               'application':all(comparisons['dispatcher'][c]['application']['ci95'][0]>protocol['criterion']['application_ci95_lower_speedup'] for c in LIBRARIES)}
        portfolio={name:base.intervals(app_samples[name],app_samples['dispatcher'],seed) for name in ('best_fixed','per_shape')}
        batching=base.intervals(app_samples['sequential_stable'],app_samples['stable_library'],seed)
        claims={'over_best_fixed':portfolio['best_fixed']['ci95'][0]>1.02,'over_per_shape':portfolio['per_shape']['ci95'][0]>1,
                'two_authored_configs':len(set(policy['policy_selection']['joint_subset']))==2,
                'batching_over_sequential':batching['ci95'][0]>protocol['criterion']['batching_vs_sequential_ci95_lower_speedup']}
        verify_frozen(policy);budget();after=base.telemetry();budget()
        result={'schema':1,'session':session,'seed':seed,'sequence_order':sequence,'process_id':os.getpid(),
            'source_sha256':protocol['source_sha256'],'study_identity':implementation(),'environment':protocol['environment'],
            'protocol_sha256':digest(EVIDENCE/'protocol.json'),'policy_sha256':digest(EVIDENCE/'policy-frozen.json'),
            'development_sha256':digest(EVIDENCE/'development.json'),'corpus_sha256':protocol['corpus_sha256'],
            'numerical_report_sha256':protocol['numerical_report_sha256'],'session_start_unix_ns':unix,
            'session_start_monotonic_ns':monotonic,'session_end_unix_ns':time.time_ns(),'session_end_monotonic_ns':time.monotonic_ns(),
            'wall_seconds':time.perf_counter()-start,'telemetry_before':before,'telemetry_after':after,
            'timing_validation_flags':base.timing_flags(),'peak_memory_bytes':mx.get_peak_memory(),
            'policy_setup_ms':load_ms,'first_rank_ms':first_rank,'first_application_ms':first_app,'checks':checks,
            'rank_windows':ranks,'application_windows':apps,'rank_orders':rank_orders,'application_orders':app_orders,
            'comparisons':comparisons,'session_gate':gates,'portfolio':portfolio,'batching':batching,'claim_flags':claims,
            'fallback_counts':{n:{'calls':m.calls,'fallback_calls':m.fallback_calls} for n,m in methods.items()},
            'counter_scope':'Private rank adapter and whole transaction calls, including validation and window repetitions; not user request counts.',
            'cold_note':'First calls can share preceding caches; fresh processes measured separately.'}
        save(f'confirmation-{session}.json',result);budget()
        completion={'result_sha256':digest(EVIDENCE/f'confirmation-{session}.json')}
        completion.update(session_end_unix_ns=time.time_ns(),session_end_monotonic_ns=time.monotonic_ns(),
                          wall_seconds=time.perf_counter()-start)
        save(f'confirmation-{session}-completed.json',completion);budget()
        print(json.dumps({'session_gate':gates,'claims':claims}))
    except Exception as error:
        save(f'confirmation-{session}-failed.json',{'reason':type(error).__name__+': '+str(error),
            'rank_windows':ranks,'application_windows':apps,'wall_seconds':time.perf_counter()-start,
            'session_end_unix_ns':time.time_ns(),'session_end_monotonic_ns':time.monotonic_ns()});raise


def require_confirmations():
    if any(not (EVIDENCE/f'confirmation-{i}-completed.json').is_file()
           or (EVIDENCE/f'confirmation-{i}-failed.json').exists() for i in range(3)):
        raise ValueError('Complete all3planned confirmations before cold work; do not expose heldout inputs early')


def cold(name):
    require_confirmations();start=time.perf_counter();base.setup();policy=read('policy-frozen.json');protocol=verify_frozen(policy)
    setup_ms=(time.perf_counter()-start)*1000
    before=time.perf_counter();method=policies(protocol,policy,name)[name];plan_ms=(time.perf_counter()-before)*1000
    tasks=[*protocol['tasks'],CONFIRM['unseen_task']];before=time.perf_counter()
    prepared=[request(task,read('corpus.json')['chunks'],49441) for task in tasks]
    input_ms=(time.perf_counter()-before)*1000
    first=base.measure(lambda:[method.search(*p) for p in prepared])/1000
    print(json.dumps({'method':name,'source_sha256':protocol['source_sha256'],'study_identity':implementation(),
        'environment':protocol['environment'],'policy_sha256':digest(EVIDENCE/'policy-frozen.json'),
        'runtime_and_frozen_checks_ms':setup_ms,'policy_setup_ms':plan_ms,'input_preparation_ms':input_ms,
        'first_application_sequence_ms':first,'timing_validation_flags':base.timing_flags(),
        'peak_memory_bytes':sys.modules['mlx.core'].get_peak_memory(),
        'scope':'FreshPython/MLX;OS/drivercachesretained;purecompiler time not isolated'}))


def cold_study():
    require_confirmations();start=time.perf_counter();reserve('cold-study',('cold-study.json','cold-study-failed.json'))
    policy=read('policy-frozen.json');protocol=verify_frozen(policy);rows=[];orders=[];rng=random.Random(49317)
    try:
        for repeat in range(3):
            order=list(METHODS);rng.shuffle(order);orders.append(order)
            for name in order:
                remaining=180-(time.perf_counter()-start)
                if remaining<=0:raise TimeoutError('180-second cold study cap; no retry')
                before=time.perf_counter()
                child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'cold','--method',name,'--directory',str(EVIDENCE)],
                                     cwd=ROOT,text=True,capture_output=True,timeout=min(60,remaining))
                if child.returncode:raise RuntimeError('Cold child failed: '+child.stderr[-2000:])
                row=json.loads(child.stdout);row.update(repeat=repeat,process_wall_ms=(time.perf_counter()-before)*1000);rows.append(row)
                if time.perf_counter()-start>180:raise TimeoutError('180-second cold study cap; no retry')
                print('cold',repeat,name,flush=True)
        save('cold-study.json',{'schema':1,'source_sha256':protocol['source_sha256'],'study_identity':implementation(),
             'environment':protocol['environment'],'policy_sha256':digest(EVIDENCE/'policy-frozen.json'),
             'rows':rows,'orders':orders,'wall_seconds':time.perf_counter()-start,'telemetry':base.telemetry()})
    except Exception as error:
        save('cold-study-failed.json',{'reason':type(error).__name__+': '+str(error),'rows':rows,'orders':orders,
             'wall_seconds':time.perf_counter()-start});raise


def main():
    global EVIDENCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','tune','confirm','cold','cold-study'))
    parser.add_argument('--directory',type=Path,default=EVIDENCE)
    parser.add_argument('--session',type=int,default=0)
    parser.add_argument('--method',choices=METHODS,default='dispatcher')
    args=parser.parse_args();EVIDENCE=args.directory.resolve()
    if args.stage=='freeze':freeze()
    elif args.stage=='tune':tune()
    elif args.stage=='confirm':confirm(args.session)
    elif args.stage=='cold':cold(args.method)
    else:cold_study()


if __name__=='__main__':main()
