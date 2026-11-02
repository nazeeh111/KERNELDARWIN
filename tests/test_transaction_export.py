"""Manufactured CPU timing fixtures exercise replay; never GPU evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin import transaction_export as verify
from kerneldarwin import transaction_plan as engine

ROOT=Path(__file__).resolve().parents[1]
ENV={'synthetic_fixture':True,'warning':'CPU guard test; not executed Metal evidence'}
FLAGS={name:None for name in verify.audit.DEBUG_FLAGS}
TELEMETRY={'time_utc':'2000-01-01T00:00:00Z','load_average':[0.,0.,0.],
           'power':{'unavailable':'synthetic'},'thermal':{'unavailable':'synthetic'},'thermal_limit':'synthetic fixture'}


def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return json.dumps(value,separators=(',',':'),allow_nan=False).encode()


def window(cost,kind):
    count,batch=(32,128) if kind=='rank' else (2,16)
    return {'outer_us':float(cost*count*batch),'inner_us':[float(cost/2)]*count,'cost_us':float(cost)}


def windows(names,kind,costs):
    return {name:[[window(costs[name],kind),window(costs[name],kind)] for _ in range(40)] for name in names}


def orders(names,rng):
    result=[]
    for _ in range(40):
        first=list(names);rng.shuffle(first);result.append(first+first[::-1])
    return result


def rule(task,name):
    return {'shape':[task['queries']]*3+[task['rows'],task['width'],5],
            'document_layout':task['layout'],'query_layout':'host_contiguous','dtype':'float32','candidate':name}


class Fixture:
    def __init__(self,pin):
        self.source=engine.source_identity();self.identity=engine.study_identity();self.pin=pin
        self.raw_corpus=(ROOT/'evidence/workload-2026-10-29/corpus.json').read_bytes()
        modules=[engine,engine.tx,verify.rank]
        from kerneldarwin import search,library_baselines,workload,study_identity
        modules +=[search,library_baselines,workload,study_identity]
        self.numeric={'kind':'transaction_plan_production_numerical_validation','schema':2,'passed':True,'gpu_executed':True,
            'tests_run':28,'native_metal_methods_completed':sorted(verify.REQUIRED_METHODS),'candidate_timing':False,
            'source_sha256':self.source,'study_identity':self.identity,'environment':copy.deepcopy(ENV),
            'failures':0,'errors':0,'skips':0,'elapsed_seconds':1.,'peak_memory_bytes':1,
            'validation_root_src_exists':False,'producer_sha256':pin['numerical_producer_sha256'],
            'test_sha256':pin['numerical_tests_sha256'],'validation_flags':{'MTL_DEBUG_LAYER':'1','MTL_SHADER_VALIDATION':'1'},
            'actual_imports':{m.__name__:{'path':'/synthetic/site-packages/'+Path(m.__file__).name,'sha256':sha(Path(m.__file__).read_bytes())} for m in modules}}
        controls=[];rng=random.Random(44211)
        for task in verify.TASKS:
            raw=windows(verify.METHODS,'rank',{name:1. for name in verify.METHODS})
            controls.append({'task':copy.deepcopy(task),'windows':raw,'orders':orders(verify.METHODS,rng),'first_ms':1.,
                'comparisons':{name:verify.audit._interval([1.]*40,[1.]*40) for name in verify.CALIBRATION['equality_rank_labels']}})
        appraw=windows(verify.METHODS,'application',{name:2. if name=='sequential_stable' else 1. for name in verify.METHODS})
        app={'windows':appraw,'orders':orders(verify.METHODS,random.Random(44213)),
             'comparisons':{name:verify.audit._interval([1.]*40,[1.]*40) for name in verify.CALIBRATION['equality_application_labels']}}
        self.protocol={'schema':1,'target':engine.OPERATION,'design':'host_batch_palindromic_long_windows_v1',
            'tasks':copy.deepcopy(verify.TASKS),'methods':list(verify.METHODS),'libraries':list(verify.LIBRARIES),
            'canonical_rank_libraries':list(verify.CANONICAL),'limits':copy.deepcopy(verify.LIMITS),
            'windows':copy.deepcopy(verify.WINDOWS),'uncertainty':copy.deepcopy(verify.UNCERTAINTY),
            'confirmation':copy.deepcopy(verify.CONFIRM),'calibration':copy.deepcopy(verify.CALIBRATION),
            'development':copy.deepcopy(verify.DEVELOPMENT),'source_sha256':self.source,'study_identity':self.identity,
            'environment':copy.deepcopy(ENV),'sources':{name:sha(data) for name,data in verify.rank._SOURCES},
            'source_bytes':sum(len(data) for _,data in verify.rank._SOURCES),'script_sha256':pin['measurement_script_sha256'],
            'helper_sha256':verify.HELPER_SHA,'rank_controls':controls,'application_control':app,
            'criterion':{'rank_ci95_lower_speedup':1.1,'application_ci95_lower_speedup':1.03,
                         'batching_vs_sequential_ci95_lower_speedup':1.03,'second_kernel_gain':.02},
            'telemetry_before':copy.deepcopy(TELEMETRY),'telemetry_after':copy.deepcopy(TELEMETRY),
            'timing_validation_flags':copy.deepcopy(FLAGS),'peak_memory_bytes':1,'calibration_wall_seconds':20.}
        rows=[];rng=random.Random(45207)
        for index,task in enumerate(verify.TASKS):
            rank_cost={name:(1. if name.startswith('plan:') or name in verify.CANDIDATES else
                             4. if name.endswith('reduction') else 2.) for name in verify.RANK_ARMS}
            app_cost={name:100. for name in verify.APP_ARMS}
            app_cost.update({'global:simd32':50.,'global:group128':60.,'routed:simd32':45.,'routed:group128':55.,
                             'plan:simd32':80. if index==1 else 40.,'plan:group128':40. if index==1 else 80.})
            ro=[];ao=[]
            for _ in range(40):
                a=list(verify.RANK_ARMS);rng.shuffle(a);ro.append(a+a[::-1])
                a=list(verify.APP_ARMS);rng.shuffle(a);ao.append(a+a[::-1])
            rows.append({'task':copy.deepcopy(task),'rank_windows':windows(verify.RANK_ARMS,'rank',rank_cost),
                'application_windows':windows(verify.APP_ARMS,'application',app_cost),'rank_orders':ro,'application_orders':ao,
                'comparisons':{name:{c:verify.audit._interval([rank_cost[c]]*40,[1.]*40) for c in verify.CANONICAL} for name in verify.CANDIDATES},
                'first_rank_ms':{name:1. for name in verify.RANK_ARMS},'first_application_ms':{name:1. for name in verify.APP_ARMS},
                'checks':[{'phase':'rank','method':name,'ids_and_original_bits':True} for name in verify.RANK_ARMS]+
                         [{'phase':'application','method':name,'scores_and_id_association':True} for name in verify.APP_ARMS]})
        self.development={'schema':1,'source_sha256':self.source,'study_identity':self.identity,'environment':copy.deepcopy(ENV),
            'rows':rows,'search_wall_seconds':100.,'peak_memory_bytes':1,'telemetry':copy.deepcopy(TELEMETRY),'timing_validation_flags':copy.deepcopy(FLAGS)}
        self.policy={'schema':1,'operation':engine.OPERATION,'epsilon':engine.EPSILON,'key_policy':verify.rank.KEY_POLICY,
            'compile_options':{'math_mode':'safe'},'source_sha256':self.source,'study_identity':self.identity,'environment':copy.deepcopy(ENV),
            'sources':copy.deepcopy(self.protocol['sources']),'rules':[rule(t,'group128' if i==1 else 'simd32') for i,t in enumerate(verify.TASKS)],
            'evidence':{'numerical_passed':True,'confirmation_passed':False},
            'policy_selection':{'best_fixed':'simd32','joint_subset':['group128','simd32'],'development_application_cost_us':120.,
                'best_one_cost_us':160.,'best_fixed_cost_us':150.,'per_shape_cost_us':135.,
                'per_shape':[rule(t,'simd32') for t in verify.TASKS]}}
        self.sessions=[];self.completions=[]
        task_calls=330244
        for index,seed in enumerate(verify.CONFIRM['seeds']):
            rank_raw={}
            for name in verify.METHODS:
                costs=([1.,1.,1.,2.] if name in ('per_shape','dispatcher') else [1.]*4 if name=='best_fixed'
                       else [4.]*4 if name.endswith('reduction') else [2.]*4)
                rank_raw[name]=[[{'task_windows':[window(cost,'rank') for cost in costs],'cost_us':sum(costs)} for _ in range(2)] for _ in range(40)]
            app_cost=dict(zip(verify.METHODS,(100.,90.,110.,105.,60.,70.,50.,200.)))
            app_raw=windows(verify.METHODS,'application',app_cost);rng=random.Random(seed);ro=[];ao=[]
            for _ in range(40):
                a=list(verify.METHODS);rng.shuffle(a);ro.append(a+a[::-1])
                a=list(verify.METHODS);rng.shuffle(a);ao.append(a+a[::-1])
            samples={name:[rank_raw[name][0][0]['cost_us']]*40 for name in verify.METHODS}
            comparisons={name:{c:{'rank':verify.audit._interval(samples[c],samples[name],seed),
                'application':verify.audit._interval([app_cost[c]]*40,[app_cost[name]]*40,seed)} for c in verify.LIBRARIES} for name in verify.METHODS}
            portfolio={name:verify.audit._interval([app_cost[name]]*40,[50.]*40,seed) for name in ('best_fixed','per_shape')}
            start=1_000_000_000_000+index*250_000_000_000
            counts={name:{'calls':task_calls*4,'fallback_calls':task_calls if name in ('per_shape','dispatcher') else
                         0 if name=='best_fixed' else task_calls*4} for name in verify.METHODS}
            self.sessions.append({'schema':1,'session':index,'seed':seed,'sequence_order':verify.CONFIRM['sequence_orders'][index],
                'process_id':10+index,'source_sha256':self.source,'study_identity':self.identity,'environment':copy.deepcopy(ENV),
                'session_start_unix_ns':start,'session_start_monotonic_ns':start,
                'session_end_unix_ns':start+100_000_000_000,'session_end_monotonic_ns':start+100_000_000_000,
                'wall_seconds':100.,'peak_memory_bytes':1,'telemetry_before':copy.deepcopy(TELEMETRY),'telemetry_after':copy.deepcopy(TELEMETRY),
                'timing_validation_flags':copy.deepcopy(FLAGS),'policy_setup_ms':1.,'rank_windows':rank_raw,'application_windows':app_raw,
                'rank_orders':ro,'application_orders':ao,'comparisons':comparisons,'portfolio':portfolio,
                'batching':verify.audit._interval([200.]*40,[100.]*40,seed),'session_gate':{'rank':True,'application':True},
                'claim_flags':{'over_best_fixed':True,'over_per_shape':True,'two_authored_configs':True,'batching_over_sequential':True},
                'first_rank_ms':{name:[1.]*4 for name in verify.METHODS},'first_application_ms':{name:[1.]*4 for name in verify.METHODS},
                'checks':[{'task':t['id'],'method':name,'rank_ids_bits':True,'scores_id_association':True} for name in verify.METHODS for t in [*verify.TASKS,verify.CONFIRM['unseen_task']]],
                'fallback_counts':counts})
            self.completions.append({'session_end_unix_ns':start+101_000_000_000,
                'session_end_monotonic_ns':start+101_000_000_000,'wall_seconds':101.})
        rng=random.Random(49317);rows=[];cold_orders=[]
        for repeat in range(3):
            order=list(verify.METHODS);rng.shuffle(order);cold_orders.append(order)
            for name in order:rows.append({'method':name,'repeat':repeat,'source_sha256':self.source,'study_identity':self.identity,
                'environment':copy.deepcopy(ENV),'runtime_and_frozen_checks_ms':1.,'policy_setup_ms':1.,'input_preparation_ms':1.,
                'first_application_sequence_ms':1.,'process_wall_ms':5.,'peak_memory_bytes':1,'timing_validation_flags':copy.deepcopy(FLAGS)})
        self.cold={'schema':1,'source_sha256':self.source,'study_identity':self.identity,'environment':copy.deepcopy(ENV),
                   'rows':rows,'orders':cold_orders,'wall_seconds':1.,'telemetry':copy.deepcopy(TELEMETRY)}

    def write(self,directory):
        root=Path(directory);files={}
        def add(name,value,raw=False):files[name]=value if raw else encode(value)
        def digest(name):return sha(files[name])
        add('corpus.json',self.raw_corpus,True);add('numerical-validation.json',self.numeric)
        add('native-validation.log',b'Metal API Validation Enabled\nMetal GPU Validation Enabled\n',True)
        self.protocol.update(corpus_sha256=digest('corpus.json'),numerical_report_sha256=digest('numerical-validation.json'),native_validation_log_sha256=digest('native-validation.log'))
        add('protocol.json',self.protocol)
        add('freeze-completed.json',{'protocol_sha256':digest('protocol.json'),'wall_seconds':21.})
        common={'protocol_sha256':digest('protocol.json'),'corpus_sha256':digest('corpus.json'),'numerical_report_sha256':digest('numerical-validation.json')}
        self.development.update(common);add('development.json',self.development)
        common['development_sha256']=digest('development.json');self.policy['evidence'].update(common);add('policy-frozen.json',self.policy)
        common['policy_sha256']=digest('policy-frozen.json')
        add('policy-checkpoint.json',{key:common[key] for key in ('protocol_sha256','development_sha256','policy_sha256')})
        add('tune-completed.json',{'policy_sha256':common['policy_sha256'],'wall_seconds':101.})
        for i,record in enumerate(self.sessions):
            record.update(common);name=f'confirmation-{i}.json';add(name,record)
            self.completions[i]['result_sha256']=digest(name);add(f'confirmation-{i}-completed.json',self.completions[i])
        self.cold['policy_sha256']=common['policy_sha256']
        for row in self.cold['rows']:row['policy_sha256']=common['policy_sha256']
        add('cold-study.json',self.cold)
        for stage in ('freeze','tune',*(f'confirmation-{i}' for i in range(3)),'cold-study'):
            add(stage+'-started.json',{'stage':stage,'script_sha256':self.pin['measurement_script_sha256'],
                'pid':10+int(stage[-1]) if stage.startswith('confirmation-') else 100})
        for name,content in files.items():(root/name).write_bytes(content)
        return root


class TransactionExportTests(unittest.TestCase):
    def setUp(self):
        self.pin={'schema':1,'measurement_script_sha256':sha((ROOT/'scripts/transaction_study.py').read_bytes()),
            'measurement_helper_sha256':verify.HELPER_SHA,'verifier_sha256':sha(Path(verify.__file__).read_bytes()),
            'numerical_producer_sha256':sha((ROOT/'scripts/validate_transaction.py').read_bytes()),
            'numerical_tests_sha256':{n:sha((ROOT/'tests'/n).read_bytes()) for n in ('test_transaction_search.py','test_transaction_plan.py')}}
        original=engine._read_regular;pin_path=Path(engine.__file__).parent/'transaction-study-source.json'
        def read(path,limit):return encode(self.pin) if Path(path)==pin_path else original(Path(path),limit)
        self.reader=patch.object(engine,'_read_regular',side_effect=read);self.env=patch.object(engine,'environment_identity',return_value=copy.deepcopy(ENV))
        self.reader.start();self.env.start();self.addCleanup(self.reader.stop);self.addCleanup(self.env.stop)
        self.fixture=Fixture(self.pin)

    def test_complete_manufactured_study_replays_and_exports_exclusively(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture.write(Path(temp));verified=verify.verify_study(root)
            self.assertTrue(verified['confirmation_passed']);self.assertTrue(verified['batching_over_sequential_passed'])
            target=root/'artifact';verify.export_artifact(root,target)
            self.assertEqual(engine.TransactionPlan(target).status,'installed')
            before=(target/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):verify.export_artifact(root,target)
            self.assertEqual((target/'manifest.json').read_bytes(),before)

    def rejected(self,mutate,message):
        data=copy.deepcopy(self.fixture);mutate(data)
        with tempfile.TemporaryDirectory() as temp:
            root=data.write(Path(temp))
            with self.assertRaisesRegex(ValueError,message):verify.verify_study(root)

    def test_criterion_cannot_be_weakened(self):
        self.rejected(lambda f:f.protocol['criterion'].update(application_ci95_lower_speedup=1.0),'Noise thresholds')

    def test_source_protocol_and_runtime_are_bound(self):
        for key,value in (('helper_sha256','f'*64),('design','different'),('environment',{})):
            with self.subTest(key=key):self.rejected(lambda f,k=key,v=value:f.protocol.update({k:v}),'Changed protocol')

    def test_inner_repetition_count_and_normalization_are_replayed(self):
        def missing(f):f.development['rows'][0]['rank_windows']['simd32'][0][0]['inner_us'].pop()
        self.rejected(missing,'inner repetition count')
        def changed(f):f.development['rows'][0]['rank_windows']['simd32'][0][0]['cost_us']=99.
        self.rejected(changed,'normalization changed')

    def test_no_missing_or_extra_group_can_be_hidden(self):
        self.rejected(lambda f:f.sessions[0]['application_windows']['dispatcher'].pop(),'40paired groups')

    def test_method_order_is_not_an_unchecked_annotation(self):
        def changed(f):
            order=f.sessions[0]['rank_orders'][0];order[0],order[1]=order[1],order[0]
        self.rejected(changed,'palindromic method order')

    def test_raw_confidence_intervals_are_recomputed(self):
        def changed(f):f.sessions[0]['comparisons']['dispatcher']['stable_library']['application']['ci95'][0]=100.
        self.rejected(changed,'comparisons differ')

    def test_app_equality_control_cannot_be_rescued_by_edited_summary(self):
        def changed(f):
            f.protocol['application_control']['windows']['dispatcher']=[
                [window(.5,'application'),window(.5,'application')] for _ in range(40)]
            f.protocol['application_control']['comparisons']={name:verify.audit._interval([1.]*40,[.5]*40)
                for name in verify.CALIBRATION['equality_application_labels']}
        self.rejected(changed,'App equality prerequisite')

    def test_numerical_report_requires_named_native_coverage(self):
        def changed(f):f.numeric['native_metal_methods_completed'][0]='unrelated method'
        self.rejected(changed,'Incomplete native numerical coverage')

    def test_numerical_report_must_be_independent_installed_execution(self):
        self.rejected(lambda f:f.numeric.update(validation_root_src_exists=True),'native numerical prerequisite')

    def test_debug_timing_is_rejected(self):
        self.rejected(lambda f:f.sessions[1]['timing_validation_flags'].update(MTL_DEBUG_LAYER='1'),'Timing under native')

    def test_resource_wall_and_measured_costs_are_checked(self):
        self.rejected(lambda f:f.development.update(search_wall_seconds=901.),'over-budget')
        self.rejected(lambda f:f.development.update(search_wall_seconds=1.),'timings exceed')

    def test_both_clocks_and_distinct_processes_are_required(self):
        def short(f):f.sessions[1]['session_start_unix_ns']=f.completions[0]['session_end_unix_ns']+119_000_000_000
        self.rejected(short,'separation below')
        self.rejected(lambda f:f.sessions[1].update(process_id=10),'processes are not distinct')

    def test_adapter_counts_cannot_skip_work_or_hide_fallbacks(self):
        def changed(f):f.sessions[0]['fallback_counts']['dispatcher']['fallback_calls']-=1
        self.rejected(changed,'call/fallback counts changed')

    def test_confirmed_policy_must_equal_independent_budgeted_selection(self):
        def changed(f):f.policy['rules'][0]['candidate']='group128'
        self.rejected(changed,'Policy differs from independent')

    def test_cold_process_count_and_order_are_replayed(self):
        self.rejected(lambda f:f.cold['rows'].pop(),'24cold processes')
        def changed(f):f.cold['orders'][0][0],f.cold['orders'][0][1]=f.cold['orders'][0][1],f.cold['orders'][0][0]
        self.rejected(changed,'Cold randomized roster')

    def test_failed_or_incomplete_stage_never_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture.write(Path(temp));(root/'confirmation-0-failed.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'failed stage'):verify.export_artifact(root,root/'absent')
            self.assertFalse((root/'absent').exists())

    def test_one_losing_session_refuses_export_despite_pooled_gain(self):
        f=self.fixture;s=f.sessions[0];seed=s['seed'];s['application_windows']['dispatcher']=[[window(120.,'application')]*2 for _ in range(40)]
        costs=dict(zip(verify.METHODS,(100.,90.,110.,105.,60.,70.,120.,200.)))
        for method in verify.METHODS:
            for base in verify.LIBRARIES:s['comparisons'][method][base]['application']=verify.audit._interval([costs[base]]*40,[costs[method]]*40,seed)
        s['portfolio']={name:verify.audit._interval([costs[name]]*40,[120.]*40,seed) for name in ('best_fixed','per_shape')}
        s['session_gate']['application']=False;s['claim_flags'].update(over_best_fixed=False,over_per_shape=False)
        with tempfile.TemporaryDirectory() as temp:
            root=f.write(Path(temp));result=verify.verify_study(root)
            self.assertFalse(result['confirmation_passed']);self.assertFalse(result['application_passed'])
            self.assertTrue(result['numerical_passed']);self.assertTrue(result['resource_passed'])
            with self.assertRaisesRegex(ValueError,'preserve the null'):verify.export_artifact(root,root/'absent')
            self.assertFalse((root/'absent').exists())

    def test_atomic_export_race_preserves_new_owner_files(self):
        original=verify.audit._rename_new
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture.write(Path(temp));target=root/'artifact'
            def raced(source,destination):
                destination.mkdir();(destination/'owner.txt').write_text('preserve')
                return original(source,destination)
            with patch.object(verify.audit,'_rename_new',side_effect=raced):
                with self.assertRaises(FileExistsError):verify.export_artifact(root,target)
            self.assertEqual((target/'owner.txt').read_text(),'preserve')
            self.assertFalse((target/'manifest.json').exists())

    def test_builder_and_independent_selection_match_manual_fixture(self):
        path=ROOT/'scripts/transaction_study.py';spec=importlib.util.spec_from_file_location('batch_builder',path)
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        rules,selection=builder.select_policy(self.fixture.protocol,self.fixture.development['rows'])
        self.assertEqual(selection,self.fixture.policy['policy_selection'])
        self.assertEqual([builder.rule(k,n) for k,n in sorted(rules.items())],self.fixture.policy['rules'])
