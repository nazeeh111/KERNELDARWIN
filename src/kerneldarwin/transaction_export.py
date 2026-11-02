"""Independent raw-evidence replay and exclusive batch-artifact export.

No builder functions, tuning, GPU timing or arbitrary artifact code executes.
Local evidence remains provenance rather than cryptographic GPU attestation.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any

from . import rank_export as audit
from . import rank_search as rank
from . import transaction_plan as engine

AUDIT_HELPER_SHA='1d4ad81ec880226e4d60a735bf0f38acbf3a768c48bf074f5442ca3b13c2d1fd'
HELPER_SHA='4470b506443feea5cf6a4f22b5fea01338e8a199621b02dd0f45f20f114f5b3a'
CORPUS_SHA='c066057478c35531b5fbe71dc5716d2868c85556ddd36b6da6c9439e1c1d70a9'
LIBRARIES=('stable_library','fast_library','stable_reduction','fast_reduction')
CANDIDATES=('simd32','group128')
METHODS=(*LIBRARIES,'best_fixed','per_shape','dispatcher','sequential_stable')
GLOBALS=(*LIBRARIES,*CANDIDATES)
RANK_ARMS=(*GLOBALS,*('plan:'+name for name in CANDIDATES))
APP_ARMS=(*('global:'+name for name in GLOBALS),*('routed:'+name for name in GLOBALS),
          *('plan:'+name for name in CANDIDATES),'plan_library')
CANONICAL=('stable_library','stable_reduction')
LIMITS={'freeze_wall_seconds':180,'development_wall_seconds':900,'confirmation_wall_seconds':600,
        'cold_study_wall_seconds':180,'memory_bytes':2147483648,'cache_bytes':268435456,
        'source_bytes':12288,'portfolio_kernels':2,'max_rules':128}
WINDOWS={'rank_bursts':32,'rank_inputs_per_burst':128,'application_blocks':2,'sequences_per_block':16}
UNCERTAINTY={'resamples':4000,'circular_block_lengths':[4,8],'block_seed_stride':100003,
             'primary_interval':'envelope_iid_block4_block8'}
TASKS=[{'id':'notes','rows':64,'width':384,'queries':1,'layout':'host_contiguous'},
       {'id':'manuals','rows':161,'width':768,'queries':8,'layout':'host_column_stride2'},
       {'id':'archive','rows':161,'width':1536,'queries':17,'layout':'host_transpose'}]
CONFIRM={'sessions':3,'repeats':40,'seeds':[46101,47203,48307],
         'sequence_orders':[[1,0,3,2],[2,3,0,1],[0,2,1,3]],'minimum_separation_seconds':120,
         'unseen_task':{'id':'unseen','rows':113,'width':1153,'queries':11,'layout':'host_column_stride2'}}
CALIBRATION={'seed':44100,'application_seeds':list(range(44101,44141)),
    'rank_order_seed':44211,'application_order_seed':44213,
    'application_orders':[[(r+i)%3 for i in range(3)] for r in range(40)],
    'equality_rank_labels':[n for n in METHODS if n!='dispatcher'],
    'equality_application_labels':[n for n in METHODS if n not in ('dispatcher','sequential_stable')]}
DEVELOPMENT={'seed':45104,'repeats':40,'order_seed':45207,'rank_arms':list(RANK_ARMS),'application_arms':list(APP_ARMS)}
DATA_FILES=('protocol.json','corpus.json','numerical-validation.json','native-validation.log',
    'freeze-completed.json','development.json','policy-frozen.json','policy-checkpoint.json','tune-completed.json',
    *tuple(f'confirmation-{i}.json' for i in range(3)),
    *tuple(f'confirmation-{i}-completed.json' for i in range(3)),'cold-study.json')
REQUIRED_METHODS={
    'TransactionSearchTests.'+name for name in (
        'test_all_library_and_authored_paths_match_batch_reference','test_changed_contents_and_changed_documents',
        'test_empty_batches_still_validate_document_refresh','test_invalid_shapes_types_and_k',
        'test_lower_precision_and_mixed_query_dtypes','test_nonfinite_rejection_does_not_mutate_inputs',
        'test_public_variable_batch_boundaries','test_strides_offsets_broadcasts_and_input_preservation',
        'test_unsupported_custom_shapes_and_k_use_library_results','test_zero_ties_and_adversarial_magnitudes')}
REQUIRED_METHODS|={'TransactionPlanMetalTests.'+name for name in (
    'test_known_host_layouts_execute_both_authored_trials','test_unknown_layout_dtype_boundary_and_mlx_inputs_fall_back',
    'test_changed_aliasing_inputs_and_nonfinite_recovery','test_rank_adapter_uses_evaluated_scores_and_bounded_dispatch')}


def _bytes(path):return engine._read_regular(Path(path),16_000_000)
def _sha(path):return hashlib.sha256(_bytes(path)).hexdigest()
def _equal(a,b,message):audit._require(a,b,message)


def _key(task):
    return (tuple([task['queries']]*3),task['rows'],task['width'],5,task['layout'])


def _rule(key,name):
    rows,n,d,k,layout=key
    return {'shape':[*rows,n,d,k],'document_layout':layout,'query_layout':'host_contiguous','dtype':'float32','candidate':name}


def _criterion(rank_comparisons,application_comparisons):
    def floor(comparisons,minimum):
        envelope=1.
        for comparison in comparisons:
            lower,upper=comparison['ci95'];envelope=max(envelope,upper,1/lower)
        return max(minimum,envelope*envelope)
    a=floor(application_comparisons,1.03)
    return {'rank_ci95_lower_speedup':floor(rank_comparisons,1.10),'application_ci95_lower_speedup':a,
            'batching_vs_sequential_ci95_lower_speedup':a,'second_kernel_gain':.02}


def _window(record,kind):
    if not isinstance(record,dict):raise ValueError('Missing raw timing window')
    if kind=='sequence_rank':
        _equal(sorted(record),['cost_us','task_windows'],'Unexpected sequence window fields')
        if not isinstance(record['task_windows'],list) or len(record['task_windows'])!=4:
            raise ValueError('Confirmation rank sequence must contain4task windows')
        cost=sum(_window(row,'rank') for row in record['task_windows'])
        _equal(record['cost_us'],cost,'Wrong rank sequence sum')
        return cost
    _equal(sorted(record),['cost_us','inner_us','outer_us'],'Unexpected raw window fields')
    count,batch=(32,128) if kind=='rank' else (2,16)
    audit._number(record['outer_us'],600_000_000)
    if not isinstance(record['inner_us'],list) or len(record['inner_us'])!=count:
        raise ValueError('Wrong inner repetition count')
    for value in record['inner_us']:audit._number(value,600_000_000)
    if math.fsum(record['inner_us'])*batch>record['outer_us']+1e-6:
        raise ValueError('Inner timing costs exceed complete outer window')
    cost=record['outer_us']/(count*batch)
    _equal(record['cost_us'],cost,'Raw timing window normalization changed')
    return cost


def _samples(raw,names,kind):
    if not isinstance(raw,dict) or set(raw)!=set(names):raise ValueError('Changed timing arm domain')
    result={}
    for name in names:
        groups=raw[name]
        if not isinstance(groups,list) or len(groups)!=40:raise ValueError('Need40paired groups')
        values=[]
        for group in groups:
            if not isinstance(group,list) or len(group)!=2:raise ValueError('Need2windows folded into1group')
            values.append(statistics.mean([_window(w,kind) for w in group]))
        result[name]=values
    return result


def _order(actual,names,rng):
    expected=list(names);rng.shuffle(expected);expected+=expected[::-1]
    _equal(actual,expected,'Changed palindromic method order')


def _outer_seconds(raw,kind):
    def amount(record):
        return sum(t['outer_us'] for t in record['task_windows']) if kind=='sequence_rank' else record['outer_us']
    return math.fsum(amount(w) for groups in raw.values() for group in groups for w in group)/1e6


def _stage(record,wall,maximum,measured=0):
    audit._number(record[wall],maximum)
    if measured>record[wall]+1e-6:raise ValueError('Recorded timings exceed reported stage wall')
    audit._count(record['peak_memory_bytes'],LIMITS['memory_bytes'])
    audit._flags(record)


def _selection(rows,criterion):
    def cost(row,label):return statistics.mean(row['application'][label])
    def allowed(row,name):
        return all(row['comparisons'][name][c]['ci95'][0]>criterion['rank_ci95_lower_speedup'] for c in CANONICAL)
    fixed_cost={name:sum(cost(row,'global:'+name) for row in rows) for name in GLOBALS}
    best_fixed=min(fixed_cost,key=fixed_cost.get)
    routed={_key(row['task']):min(GLOBALS,key=lambda n:cost(row,'routed:'+n)) for row in rows}
    candidates=[]
    for size in range(3):
        for subset in itertools.combinations(CANDIDATES,size):
            chosen={};total=0.
            for row in rows:
                labels=['plan_library']+['plan:'+name for name in subset if allowed(row,name)]
                best=min(labels,key=lambda label:cost(row,label));total+=cost(row,best)
                if best!='plan_library':chosen[_key(row['task'])]=best.split(':',1)[1]
            candidates.append((subset,total,chosen))
    one=min((item for item in candidates if len(item[0])<=1),key=lambda item:item[1])
    best=min(candidates,key=lambda item:item[1])
    if best[1]>one[1]*(1-criterion['second_kernel_gain']):best=one
    selection={'best_fixed':best_fixed,'joint_subset':sorted(set(best[2].values())),
        'development_application_cost_us':best[1],'best_one_cost_us':one[1],
        'best_fixed_cost_us':fixed_cost[best_fixed],
        'per_shape_cost_us':sum(cost(row,'routed:'+routed[_key(row['task'])]) for row in rows),
        'per_shape':[_rule(key,name) for key,name in sorted(routed.items())]}
    return [_rule(key,name) for key,name in sorted(best[2].items())],selection


def _verify(directory):
    if _sha(audit.__file__)!=AUDIT_HELPER_SHA:raise ValueError('Independent replay helper changed')
    pin=audit._decode(engine._read_regular(Path(engine.__file__).parent/'transaction-study-source.json',4096))
    if _sha(__file__)!=pin['verifier_sha256']:raise ValueError('Batch verifier differs from installed pin')
    if pin['measurement_helper_sha256']!=HELPER_SHA:raise ValueError('Measurement helper pin changed')
    source,identity,environment=engine.source_identity(),engine.study_identity(),engine.environment_identity()
    sources,source_bytes=audit._source_snapshot(rank);audit._loaded_application_sources()
    for failed in ('rejected-baseline.json','development-failed.json','cold-study-failed.json',
                   *tuple(f'confirmation-{i}-failed.json' for i in range(3))):
        if (directory/failed).exists():raise ValueError('Preserve failed stage; no artifact')
    content={};total_bytes=0
    for name in DATA_FILES:
        raw=_bytes(directory/name);total_bytes+=len(raw)
        if total_bytes>96_000_000:raise ValueError('Study exceeds total bounded evidence size')
        content[name]=raw
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in content.items()}
    data={name:audit._decode(raw) for name,raw in content.items() if name.endswith('.json')}
    p=data['protocol.json'];n=data['numerical-validation.json'];development=data['development.json'];policy=data['policy-frozen.json']
    if hashes['corpus.json']!=CORPUS_SHA:raise ValueError('Canonical licensed corpus changed')
    _equal(len(data['corpus.json']['chunks']),161,'Changed corpus size')
    for key,value in {'schema':1,'target':engine.OPERATION,'design':'host_batch_palindromic_long_windows_v1',
        'tasks':TASKS,'methods':list(METHODS),'libraries':list(LIBRARIES),'canonical_rank_libraries':list(CANONICAL),
        'limits':LIMITS,'windows':WINDOWS,'uncertainty':UNCERTAINTY,'confirmation':CONFIRM,'calibration':CALIBRATION,
        'development':DEVELOPMENT,'source_sha256':source,'study_identity':identity,'environment':environment,
        'sources':sources,'source_bytes':source_bytes,'script_sha256':pin['measurement_script_sha256'],
        'helper_sha256':HELPER_SHA,'corpus_sha256':CORPUS_SHA,
        'numerical_report_sha256':hashes['numerical-validation.json'],'native_validation_log_sha256':hashes['native-validation.log']}.items():
        _equal(p.get(key),value,'Changed protocol '+key)
    if ('Metal API Validation Enabled' not in content['native-validation.log'].decode(errors='replace')
            or 'Metal GPU Validation Enabled' not in content['native-validation.log'].decode(errors='replace')):
        raise ValueError('Native validation activation log missing')
    for key,value in {'kind':'transaction_plan_production_numerical_validation','schema':2,'passed':True,'gpu_executed':True,
        'source_sha256':source,'study_identity':identity,'environment':environment,'failures':0,'errors':0,'skips':0,
        'validation_root_src_exists':False,'producer_sha256':pin['numerical_producer_sha256'],
        'test_sha256':pin['numerical_tests_sha256']}.items():_equal(n.get(key),value,'Changed native numerical prerequisite '+key)
    if (type(n['tests_run']) is not int or n['tests_run']<28 or not isinstance(n['native_metal_methods_completed'],list)
            or not REQUIRED_METHODS<=set(n['native_metal_methods_completed']) or n.get('candidate_timing') is not False):
        raise ValueError('Incomplete native numerical coverage')
    audit._number(n['elapsed_seconds'],600);audit._count(n['peak_memory_bytes'],LIMITS['memory_bytes'])
    for flag in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):
        if str(n['validation_flags'].get(flag)).lower() not in ('1','true'):raise ValueError('Native validation not requested')
    modules={'kerneldarwin.transaction_plan':engine,'kerneldarwin.transaction_search':engine.tx,
             'kerneldarwin.rank_search':rank}
    from . import search,library_baselines,workload,study_identity
    modules.update({m.__name__:m for m in (search,library_baselines,workload,study_identity)})
    if set(n['actual_imports'])!=set(modules):raise ValueError('Missing actual installed module paths')
    for name,module in modules.items():
        item=n['actual_imports'][name]
        if '/site-packages/' not in item['path']:raise ValueError('Numerical execution was not source-free installed')
        _equal(item['sha256'],_sha(module.__file__),'Numerical module bytes changed')
    audit._telemetry(p['telemetry_before']);audit._telemetry(p['telemetry_after'])
    rng=random.Random(44211);rank_controls=[];measured_calibration=0.
    if not isinstance(p['rank_controls'],list) or len(p['rank_controls'])!=3:raise ValueError('Missing rank controls')
    for task,row in zip(TASKS,p['rank_controls']):
        _equal(row['task'],task,'Changed calibration task')
        values=_samples(row['windows'],METHODS,'rank')
        if len(row['orders'])!=40:raise ValueError('Missing control orders')
        for order in row['orders']:_order(order,METHODS,rng)
        expected={name:audit._interval(values[name],values['dispatcher']) for name in CALIBRATION['equality_rank_labels']}
        _equal(row['comparisons'],expected,'Rank controls do not match raw samples');rank_controls+=list(expected.values())
        audit._number(row['first_ms'],180000);measured_calibration+=_outer_seconds(row['windows'],'rank')
    row=p['application_control'];values=_samples(row['windows'],METHODS,'application');rng=random.Random(44213)
    if len(row['orders'])!=40:raise ValueError('Missing app control orders')
    for order in row['orders']:_order(order,METHODS,rng)
    app_controls={name:audit._interval(values[name],values['dispatcher']) for name in CALIBRATION['equality_application_labels']}
    _equal(row['comparisons'],app_controls,'App controls do not match raw samples')
    if any(not .9<=bound<=1.1 for c in app_controls.values() for bound in c['ci95']):raise ValueError('App equality prerequisite failed')
    criterion=_criterion(rank_controls,list(app_controls.values()));_equal(p['criterion'],criterion,'Noise thresholds weakened/changed')
    measured_calibration+=_outer_seconds(row['windows'],'application')
    _stage(p,'calibration_wall_seconds',180,measured_calibration)
    complete=data['freeze-completed.json'];_equal(complete['protocol_sha256'],hashes['protocol.json'],'Freeze checkpoint changed')
    audit._number(complete['wall_seconds'],180)
    if complete['wall_seconds']<p['calibration_wall_seconds']:raise ValueError('Bad calibration completion wall')

    def bindings(record,items):
        for key,value in {'source_sha256':source,'study_identity':identity,'environment':environment}.items():
            _equal(record[key],value,'Changed source/runtime binding')
        for key,name in items:_equal(record[key],hashes[name],'Changed evidence hash '+name)
    bindings(development,[('protocol_sha256','protocol.json'),('corpus_sha256','corpus.json'),('numerical_report_sha256','numerical-validation.json')])
    rows=development['rows'];_equal(len(rows),3,'Missing development tasks');replayed=[];measured_development=0.;rng=random.Random(45207)
    for task,row in zip(TASKS,rows):
        _equal(row['task'],task,'Changed development task')
        ranks=_samples(row['rank_windows'],RANK_ARMS,'rank');apps=_samples(row['application_windows'],APP_ARMS,'application')
        _equal(len(row['rank_orders']),40,'Changed development rank groups');_equal(len(row['application_orders']),40,'Changed app groups')
        for ro,ao in zip(row['rank_orders'],row['application_orders']):_order(ro,RANK_ARMS,rng);_order(ao,APP_ARMS,rng)
        comparisons={name:{c:audit._interval(ranks[c],ranks['plan:'+name]) for c in CANONICAL} for name in CANDIDATES}
        _equal(row['comparisons'],comparisons,'Changed development rank eligibility')
        expected_checks=[{'phase':'rank','method':name,'ids_and_original_bits':True} for name in RANK_ARMS]
        expected_checks +=[{'phase':'application','method':name,'scores_and_id_association':True} for name in APP_ARMS]
        _equal(row['checks'],expected_checks,'Missing development numerical checks')
        for key,names in (('first_rank_ms',RANK_ARMS),('first_application_ms',APP_ARMS)):
            if set(row[key])!=set(names):raise ValueError('Missing first-call data')
            for value in row[key].values():audit._number(value,900000)
        replayed.append({'task':task,'application':apps,'comparisons':comparisons})
        measured_development+=_outer_seconds(row['rank_windows'],'rank')+_outer_seconds(row['application_windows'],'application')
    _stage(development,'search_wall_seconds',900,measured_development);audit._telemetry(development['telemetry'])
    expected_rules,selection=_selection(replayed,criterion)
    expected_policy={'schema':1,'operation':engine.OPERATION,'epsilon':engine.EPSILON,'key_policy':rank.KEY_POLICY,
        'compile_options':{'math_mode':'safe'},'source_sha256':source,'study_identity':identity,'environment':environment,
        'sources':sources,'rules':expected_rules,'evidence':{'numerical_passed':True,'confirmation_passed':False,
            'protocol_sha256':hashes['protocol.json'],'development_sha256':hashes['development.json'],
            'corpus_sha256':CORPUS_SHA,'numerical_report_sha256':hashes['numerical-validation.json']},'policy_selection':selection}
    _equal(policy,expected_policy,'Policy differs from independent matched-budget selection')
    _equal(data['policy-checkpoint.json'],{'policy_sha256':hashes['policy-frozen.json'],
        'protocol_sha256':hashes['protocol.json'],'development_sha256':hashes['development.json']},'Policy freeze checkpoint changed')
    tune_done=data['tune-completed.json'];_equal(tune_done['policy_sha256'],hashes['policy-frozen.json'],'Tuning completion changed')
    audit._number(tune_done['wall_seconds'],900)
    if tune_done['wall_seconds']<development['search_wall_seconds']:raise ValueError('Tuning wall excludes measured work')

    sessions=[];spans=[];per_session=[];all_app={name:[] for name in METHODS};all_rank={name:[] for name in METHODS}
    rank_passed=application_passed=over_fixed=over_shape=batching_passed=True
    pids=set();peaks=[p['peak_memory_bytes'],development['peak_memory_bytes']]
    previous_completion=None
    for session in range(3):
        record=data[f'confirmation-{session}.json'];done=data[f'confirmation-{session}-completed.json']
        bindings(record,[('protocol_sha256','protocol.json'),('policy_sha256','policy-frozen.json'),
            ('development_sha256','development.json'),('corpus_sha256','corpus.json'),
            ('numerical_report_sha256','numerical-validation.json')])
        _equal(record['session'],session,'Changed confirmation index');_equal(record['seed'],CONFIRM['seeds'][session],'Changed untouched seed')
        _equal(record['sequence_order'],CONFIRM['sequence_orders'][session],'Changed untouched sequence')
        pid=record['process_id'];audit._count(pid,2**63-1)
        if pid<=0 or pid in pids:raise ValueError('Confirmation processes are not distinct')
        pids.add(pid)
        _equal(done['result_sha256'],hashes[f'confirmation-{session}.json'],'Confirmation completion hash changed')
        clocks={}
        for clock in ('unix','monotonic'):
            start=record[f'session_start_{clock}_ns'];end=record[f'session_end_{clock}_ns'];completed=done[f'session_end_{clock}_ns']
            if any(type(v) is not int or v<=0 for v in (start,end,completed)) or not start<end<=completed:
                raise ValueError('Invalid confirmation clock span')
            if previous_completion is not None and start-previous_completion[f'session_end_{clock}_ns']<120*1_000_000_000:
                raise ValueError('Confirmation separation below120seconds')
            clocks[clock]=(completed-start)/1e9
        audit._number(done['wall_seconds'],600)
        if done['wall_seconds']+1e-3<clocks['monotonic']:raise ValueError('Stage wall excludes confirmation work')
        previous_completion=done
        ranks=_samples(record['rank_windows'],METHODS,'sequence_rank')
        apps=_samples(record['application_windows'],METHODS,'application')
        _equal(len(record['rank_orders']),40,'Missing confirmation rank groups')
        _equal(len(record['application_orders']),40,'Missing confirmation app groups')
        rng=random.Random(CONFIRM['seeds'][session])
        for ro,ao in zip(record['rank_orders'],record['application_orders']):_order(ro,METHODS,rng);_order(ao,METHODS,rng)
        measured=_outer_seconds(record['rank_windows'],'sequence_rank')+_outer_seconds(record['application_windows'],'application')
        _stage(record,'wall_seconds',600,measured)
        if done['wall_seconds']<record['wall_seconds']:raise ValueError('Completion omits session wall')
        audit._telemetry(record['telemetry_before']);audit._telemetry(record['telemetry_after'])
        tasks=[*TASKS,CONFIRM['unseen_task']]
        checks=[{'task':t['id'],'method':name,'rank_ids_bits':True,'scores_id_association':True} for name in METHODS for t in tasks]
        _equal(record['checks'],checks,'Missing actual confirmation numerical checks')
        for key in ('first_rank_ms','first_application_ms'):
            if set(record[key])!=set(METHODS):raise ValueError('Missing session first calls')
            for values in record[key].values():
                if not isinstance(values,list) or len(values)!=4:raise ValueError('Missing per-task first calls')
                for value in values:audit._number(value,600000)
        audit._number(record['policy_setup_ms'],600000,zero=True)
        seed=CONFIRM['seeds'][session]
        comparisons={name:{c:{'rank':audit._interval(ranks[c],ranks[name],seed),
            'application':audit._interval(apps[c],apps[name],seed)} for c in LIBRARIES} for name in METHODS}
        _equal(record['comparisons'],comparisons,'Confirmation comparisons differ from raw window folds')
        gates={'rank':all(comparisons['dispatcher'][c]['rank']['ci95'][0]>criterion['rank_ci95_lower_speedup'] for c in CANONICAL),
               'application':all(comparisons['dispatcher'][c]['application']['ci95'][0]>criterion['application_ci95_lower_speedup'] for c in LIBRARIES)}
        portfolio={name:audit._interval(apps[name],apps['dispatcher'],seed) for name in ('best_fixed','per_shape')}
        batching=audit._interval(apps['sequential_stable'],apps['stable_library'],seed)
        claims={'over_best_fixed':portfolio['best_fixed']['ci95'][0]>1.02,'over_per_shape':portfolio['per_shape']['ci95'][0]>1,
            'two_authored_configs':len(selection['joint_subset'])==2,
            'batching_over_sequential':batching['ci95'][0]>criterion['batching_vs_sequential_ci95_lower_speedup']}
        _equal(record['session_gate'],gates,'Saved session gates differ from raw evidence')
        _equal(record['portfolio'],portfolio,'Portfolio comparisons changed');_equal(record['batching'],batching,'Batching comparison changed')
        _equal(record['claim_flags'],claims,'Unsupported claim flags')
        per_task=4+40*(2*32*128+2*2*16)
        per_shape={engine.rule_key(row['shape'][:-3],*row['shape'][-3:],row['document_layout']):row['candidate'] for row in selection['per_shape']}
        rules={engine.rule_key(row['shape'][:-3],*row['shape'][-3:],row['document_layout']):row['candidate'] for row in expected_rules}
        counts={}
        for name in METHODS:
            fallback=0
            for task in tasks:
                if name=='dispatcher':choice=rules.get(_key(task),'stable_library')
                elif name=='per_shape':choice=per_shape.get(_key(task),'stable_library')
                elif name=='best_fixed':choice=selection['best_fixed']
                else:choice=name
                if choice not in CANDIDATES:fallback+=per_task
            counts[name]={'calls':per_task*4,'fallback_calls':fallback}
        _equal(record['fallback_counts'],counts,'Recorded adapter call/fallback counts changed')
        rank_passed &= gates['rank'];application_passed &= gates['application']
        over_fixed &= claims['over_best_fixed'];over_shape &= claims['over_per_shape'];batching_passed &= claims['batching_over_sequential']
        per_session.append({'gates':gates,'claims':claims,'dispatcher':comparisons['dispatcher'],'portfolio':portfolio,'batching':batching})
        spans.append({'process_id':pid,'clock_spans_seconds':clocks,'wall_seconds':done['wall_seconds']})
        peaks.append(record['peak_memory_bytes']);sessions.append(record)
        for name in METHODS:all_app[name]+=apps[name];all_rank[name]+=ranks[name]

    cold=data['cold-study.json'];bindings(cold,[('policy_sha256','policy-frozen.json')]);audit._number(cold['wall_seconds'],180)
    rows=cold['rows'];_equal(len(rows),24,'Need24cold processes');rng=random.Random(49317);index=0;orders=[]
    for repeat in range(3):
        order=list(METHODS);rng.shuffle(order);orders.append(order)
        for name in order:
            row=rows[index];index+=1;_equal(row['repeat'],repeat,'Cold repeat changed');_equal(row['method'],name,'Cold method order changed')
            bindings(row,[('policy_sha256','policy-frozen.json')]);audit._flags(row)
            costs=('runtime_and_frozen_checks_ms','policy_setup_ms','input_preparation_ms','first_application_sequence_ms')
            for key in (*costs,'process_wall_ms'):audit._number(row[key],180000,zero=key!='process_wall_ms')
            if math.fsum(row[key] for key in costs)>row['process_wall_ms']+1e-3:raise ValueError('Cold components exceed process wall')
            audit._count(row['peak_memory_bytes'],LIMITS['memory_bytes']);peaks.append(row['peak_memory_bytes'])
    _equal(cold['orders'],orders,'Cold randomized roster changed');audit._telemetry(cold['telemetry'])
    if math.fsum(r['process_wall_ms'] for r in rows)>cold['wall_seconds']*1000+1e-3:raise ValueError('Cold costs exceed total budget')
    cold_summary={name:{'median_process_wall_ms':statistics.median(r['process_wall_ms'] for r in rows if r['method']==name),
        'median_first_sequence_ms':statistics.median(r['first_application_sequence_ms'] for r in rows if r['method']==name)} for name in METHODS}
    # Exclusive stage markers bind their script and confirmation process IDs.
    for stage in ('freeze','tune',*(f'confirmation-{i}' for i in range(3)),'cold-study'):
        marker=audit._decode(_bytes(directory/(stage+'-started.json')))
        _equal(marker['stage'],stage,'Changed stage reservation');_equal(marker['script_sha256'],pin['measurement_script_sha256'],'Stage source changed')
        if stage.startswith('confirmation-'):
            _equal(marker['pid'],sessions[int(stage[-1])]['process_id'],'Confirmation reservation PID differs')
    passed=bool(expected_rules) and rank_passed and application_passed
    mean_dispatch=statistics.mean(all_app['dispatcher']);baseline_means={c:statistics.mean(all_app[c]) for c in LIBRARIES}
    savings=min(baseline_means.values())-mean_dispatch
    extra_cold=max(0.,cold_summary['dispatcher']['median_process_wall_ms']-min(cold_summary[c]['median_process_wall_ms'] for c in LIBRARIES))
    payback=math.ceil((tune_done['wall_seconds']*1e6+extra_cold*1000)/savings) if passed and savings>0 else None
    return {'schema':1,'operation':engine.OPERATION,'numerical_passed':True,'resource_passed':True,
        'rank_passed':rank_passed,'application_passed':application_passed,'confirmation_passed':passed,
        'source_sha256':source,'study_identity':identity,'environment':environment,'criterion':criterion,
        'evidence_sha256':hashes,'authored_source_bytes':source_bytes,'selection':selection,'per_session':per_session,
        'portfolio':{'over_best_fixed':over_fixed,'over_per_shape':over_shape,'two_authored_configs':len(selection['joint_subset'])==2,
                     'two_kernel_advantage_over_fixed':len(selection['joint_subset'])==2 and over_fixed,
                     'two_kernel_advantage_over_per_shape':len(selection['joint_subset'])==2 and over_shape},
        'batching_over_sequential_passed':batching_passed,'sessions':spans,'cold_start':cold_summary,
        'search_cost':{'wall_seconds':tune_done['wall_seconds'],'recorded_window_seconds':measured_development,
                       'authored_configurations':2,'rank_arms':8,'application_arms':15},
        'calibration':{'wall_seconds':complete['wall_seconds'],'recorded_window_seconds':measured_calibration},
        'memory':{'mlx_limit_bytes':LIMITS['memory_bytes'],'mlx_cache_bytes':LIMITS['cache_bytes'],'maximum_observed_mlx_peak_bytes':max(peaks)},
        'break_even':{'full_four_task_sequences':payback,'guaranteed':False,'reliable_payback':False,
            'included_costs':'Development wall including serialization plus positive incremental median cold-process cost',
            'excluded_costs':'Calibration, numerical validation, confirmation/gaps, earlier studies, profiling, engineering, installation and packaging',
            'baseline_mean_sequence_us':baseline_means,'dispatcher_mean_sequence_us':mean_dispatch},
        'limitations':['Local evidence is provenance, not cryptographic GPU attestation.',
            'Every session must pass; no pooled rescue. Paired block envelopes do not guarantee formal95%coverage or independent environments.',
            'Rank isolates lookup and ranking on evaluated scores; application includes all host staging, guards and materialization.',
            'Cold processes retain OS/driver caches; pure compiler and pure kernel times are not isolated.',
            'MLX peak memory and Metal resources have distinct scopes; CPU/system peak is not inferred.',
            'Batching comparison is separate from custom/portfolio acceptance; source/device/runtime guards limit coverage.']}


def verify_study(directory: Path) -> dict[str,Any]:
    try:return _verify(Path(directory))
    except (KeyError,TypeError,IndexError,AttributeError,OverflowError,ZeroDivisionError) as error:
        raise ValueError('Malformed or incomplete batch study') from error


def export_artifact(study: Path, output: Path) -> dict[str,Any]:
    study,output=Path(study),Path(output)
    if os.path.lexists(output):raise FileExistsError('Choose a new artifact directory')
    verification=verify_study(study)
    if verification['confirmation_passed'] is not True:
        raise ValueError('Frozen batch confirmation failed; preserve the null result')
    for name,sha in verification['evidence_sha256'].items():
        if _sha(study/name)!=sha:raise ValueError('Study changed after verification')
    policy=audit._decode(_bytes(study/'policy-frozen.json'))
    if not policy['rules']:raise ValueError('No confirmed custom batch rules')
    policy['evidence']={**policy['evidence'],'confirmation_passed':True,'verification':verification['evidence_sha256']}
    audit._source_snapshot(rank);engine.source_identity()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kd-batch-export-',dir=output.parent) as temp:
        staging=Path(temp)/'artifact';staging.mkdir()
        for name,content in rank._SOURCES:(staging/name).write_bytes(content)
        (staging/'manifest.json').write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
        (staging/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
        if engine.TransactionPlan(staging).status!='installed':raise ValueError('Staged batch artifact failed installed guards')
        audit._rename_new(staging,output)
    return {'artifact':str(output),'confirmation_passed':True,'source_sha256':verification['source_sha256'],
            'bytes':sum(path.stat().st_size for path in output.iterdir())}
