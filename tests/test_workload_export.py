"""Synthetic CPU fixtures test promotion guards; never hardware evidence."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin.workload_export import export_artifact,verify_study,METHODS,DEVELOPMENT_METHODS,LIMITS,SEEDS,SEQUENCES
from kerneldarwin.workload import source_identity,_LOADED_SOURCE
from kerneldarwin.study_identity import implementation_identity


def write(path,value):path.write_text(json.dumps(value))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def repin_fixture(root):
    """Rebind synthetic bytes so tests reach guards beyond hash mismatches."""
    development=json.loads((root/'development.json').read_text())
    development['protocol_sha256']=sha(root/'protocol.json');write(root/'development.json',development)
    policy=json.loads((root/'policy-frozen.json').read_text())
    policy['evidence'].update(protocol_sha256=sha(root/'protocol.json'),development_sha256=sha(root/'development.json'))
    write(root/'policy-frozen.json',policy)
    for i in range(3):
        record=json.loads((root/f'confirmation-{i}.json').read_text())
        record.update(protocol_sha256=sha(root/'protocol.json'),policy_sha256=sha(root/'policy-frozen.json'))
        write(root/f'confirmation-{i}.json',record)
    cold=json.loads((root/'cold-study.json').read_text());cold['policy_sha256']=sha(root/'policy-frozen.json');write(root/'cold-study.json',cold)


def fixture(root):
    environment={'unit_fixture':'synthetic; no GPU execution'}
    write(root/'corpus.json',{'chunks':['synthetic unit test corpus']*64})
    tasks=[{'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},
           {'id':'manuals','rows':64,'width':768,'queries':8,'layout':'column_stride2'},
           {'id':'archive','rows':64,'width':1536,'queries':17,'layout':'transpose'}]
    from kerneldarwin import study_identity
    script=json.loads((Path(study_identity.__file__).parent/'study-source.json').read_text())['measurement_script_sha256']
    protocol={'schema':1,'environment':environment,'corpus_chunks':64,'script_sha256':script,
              'corpus_sha256':sha(root/'corpus.json'),'study_identity':implementation_identity(),
              'baseline_noise':[{'task':task,'baseline_a_us':[10.]*40,'baseline_b_us':[10.]*40} for task in tasks],
              'tasks':tasks,'development':{'seed':101,'repeats':40},'measurement_batch':32,'application_measurement_batch':4,
              'confirmation':{'sessions':3,'seeds':SEEDS,'unseen_task':{'id':'unseen','rows':97,'width':513,'queries':3,'layout':'column_stride2'},'sequence_orders':SEQUENCES,'repeats_per_session':40},
              'criterion':{'normalization_ci95_lower_speedup':1.1,'application_claim_ci95_lower_speedup':1.03,
                           'portfolio_over_fixed_minimum_total_cost_gain':.02,'numerical_rtol':2e-5,'numerical_atol':2e-6},
              'limits':LIMITS}
    write(root/'protocol.json',protocol)
    numerical={'study_identity':implementation_identity(),'source_sha256':source_identity(),'environment':environment,'success':True,'gpu_executed':True,'tests_run':12,'failures':0,'errors':0,'skipped':0,'fixture':True}
    write(root/'numerical-validation.json',numerical)
    policy={'study_identity':implementation_identity(),'schema':1,'operation':'stable_l2_rows_v1','epsilon':1e-12,'sources':{name:hashlib.sha256(source).hexdigest() for name,source in _LOADED_SOURCE},'source_sha256':source_identity(),'environment':environment,'rules':[{'shape':[64,384],'candidate':'simd','dtype':'float32','layout':'any_mlx_strides'}],
            'evidence':{'corpus_sha256':sha(root/'corpus.json'),'numerical_passed':True,'confirmation_passed':False,'protocol_sha256':sha(root/'protocol.json'),'numerical_report_sha256':sha(root/'numerical-validation.json')}}
    write(root/'policy-frozen.json',policy)
    rows=[];rng=random.Random(207)
    for task in tasks:
        for label,key,weight in (('index','rows',1),('query','queries',3)):
            orders=[]
            for _ in range(40):
                order=list(DEVELOPMENT_METHODS);rng.shuffle(order);orders.append(order)
            rows.append({'id':task['id']+'-'+label,'shape':[task[key],task['width']],
                         'layout':task['layout'] if label=='index' else 'contiguous','weight':weight,
                         'numerical_check':{'passed':True,'rtol':2e-5,'atol':2e-6},'orders':orders,
                         'samples_us':{method:[10.]*40 for method in DEVELOPMENT_METHODS}})
    development={'corpus_sha256':sha(root/'corpus.json'),'study_identity':implementation_identity(),'source_sha256':source_identity(),'protocol_sha256':sha(root/'protocol.json'),'rows':rows,'search_wall_seconds':1,'peak_memory_bytes':1024}
    write(root/'development.json',development)
    policy['evidence']['development_sha256']=sha(root/'development.json');write(root/'policy-frozen.json',policy)
    samples={name:[float(value)]*40 for name,value in (('library',10),('eager_library',12),('fast_library',9),('compiled_application',9),('fast_compiled_application',8),('best_fixed',2),('per_shape',1.1),('dispatcher',1))}
    for i in range(3):
        rng=random.Random(SEEDS[i]);orders=[]
        for _ in range(40):
            order=list(METHODS);rng.shuffle(order);orders.append(order)
        record={'corpus_sha256':sha(root/'corpus.json'),'study_identity':implementation_identity(),'session':i,'seed':protocol['confirmation']['seeds'][i],'sequence_order':protocol['confirmation']['sequence_orders'][i],'source_sha256':source_identity(),'environment':environment,
                'protocol_sha256':sha(root/'protocol.json'),'policy_sha256':sha(root/'policy-frozen.json'),
                'checks':[{'task':task,'method':method,'scores_passed':True,'ids_association_passed':True} for task in ('notes','manuals','archive','unseen') for method in samples], 'fallback_counts':{'dispatcher':{'fallback_calls':1}},'orders':orders,
                'normalization_samples_us':samples,'application_samples_us':samples,'peak_memory_bytes':2048}
        write(root/f'confirmation-{i}.json',record)
    cold={'source_sha256':source_identity(),'study_identity':implementation_identity(),'corpus_sha256':sha(root/'corpus.json'),'policy_sha256':sha(root/'policy-frozen.json'),'wall_seconds':1,
          'rows':[{'repeat':i,'method':method,'source_sha256':source_identity(),'study_identity':implementation_identity(),'environment':environment,'process_wall_ms':100.,'first_application_sequence_ms':10.,'policy_setup_ms':1.,'peak_memory_bytes':2048} for i in range(3) for method in samples]}
    write(root/'cold-study.json',cold)
    return environment


class ExportTests(unittest.TestCase):
    def test_one_independent_fast_library_loss_cannot_hide_in_pooled_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);environment=fixture(root)
            p=json.loads((root/'confirmation-2.json').read_text())
            for stage in ('normalization_samples_us','application_samples_us'):p[stage]['fast_library']=[.5]*40
            write(root/'confirmation-2.json',p)
            with patch('kerneldarwin.workload.environment_identity',return_value=environment):
                self.assertFalse(verify_study(root)['confirmation_passed'])
                with self.assertRaises(ValueError):export_artifact(root,root/'artifact')
            self.assertFalse((root/'artifact').exists())
    def test_complete_synthetic_study_can_export_and_missing_session_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);environment=fixture(root)
            with patch('kerneldarwin.workload.environment_identity',return_value=environment):
                result=export_artifact(root,root/'artifact')
                self.assertTrue(result['confirmation_passed'])
                self.assertTrue(json.loads((root/'artifact/manifest.json').read_text())['evidence']['confirmation_passed'])
                (root/'confirmation-2.json').unlink()
                with self.assertRaises(FileNotFoundError):verify_study(root)

    def test_false_numerical_policy_and_report_tampering_refuse(self):
        for mutation in ('false_policy','changed_report','failed_report'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);environment=fixture(root)
                if mutation=='false_policy':
                    p=json.loads((root/'policy-frozen.json').read_text());p['evidence']['numerical_passed']=False;write(root/'policy-frozen.json',p)
                else:
                    p=json.loads((root/'numerical-validation.json').read_text());p['success']=False if mutation=='failed_report' else True;p['changed']=True;write(root/'numerical-validation.json',p)
                with patch('kerneldarwin.workload.environment_identity',return_value=environment),self.assertRaises(ValueError):verify_study(root)

    def test_stale_empty_failed_or_over_budget_development_cannot_promote(self):
        for mutation in ('source','protocol','empty','numerical','memory','corpus'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);environment=fixture(root)
                p=json.loads((root/'development.json').read_text())
                if mutation=='source':p['source_sha256']='stale'
                elif mutation=='protocol':p['protocol_sha256']='stale'
                elif mutation=='empty':p['rows']=[]
                elif mutation=='numerical':p['rows'][0]['numerical_check']['passed']=False
                elif mutation=='corpus':write(root/'corpus.json',{'chunks':['changed corpus']})
                else:p['peak_memory_bytes']=3*1024**3
                write(root/'development.json',p)
                with patch('kerneldarwin.workload.environment_identity',return_value=environment):
                    with self.assertRaises(ValueError):export_artifact(root,root/'artifact')
                self.assertFalse((root/'artifact').exists())

    def test_missing_or_false_id_association_flag_refuses(self):
        for flag in ('missing',False):
            with self.subTest(flag=flag),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);environment=fixture(root)
                record=json.loads((root/'confirmation-0.json').read_text())
                if flag=='missing':record['checks'][0].pop('ids_association_passed')
                else:record['checks'][0]['ids_association_passed']=False
                write(root/'confirmation-0.json',record)
                with patch('kerneldarwin.workload.environment_identity',return_value=environment):
                    with self.assertRaisesRegex(ValueError,'ID association'):verify_study(root)

    def test_lowered_criterion_cannot_promote_half_percent_gain(self):
        for lowered in (False,True):
            with self.subTest(lowered=lowered),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);environment=fixture(root)
                for i in range(3):
                    record=json.loads((root/f'confirmation-{i}.json').read_text())
                    for key in ('normalization_samples_us','application_samples_us'):
                        for baseline in ('library','fast_library','compiled_application','fast_compiled_application'):record[key][baseline]=[100.5]*40
                        record[key]['dispatcher']=[100.]*40
                    write(root/f'confirmation-{i}.json',record)
                if lowered:
                    protocol=json.loads((root/'protocol.json').read_text())
                    protocol['criterion'].update(normalization_ci95_lower_speedup=1.,application_claim_ci95_lower_speedup=1.)
                    write(root/'protocol.json',protocol);repin_fixture(root)
                with patch('kerneldarwin.workload.environment_identity',return_value=environment):
                    if lowered:
                        with self.assertRaisesRegex(ValueError,'acceptance criterion'):verify_study(root)
                    else:
                        self.assertFalse(verify_study(root)['confirmation_passed'])
                    with self.assertRaises(ValueError):export_artifact(root,root/'artifact')
                self.assertFalse((root/'artifact').exists())

    def test_baseline_derived_threshold_cannot_be_lowered_to_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);environment=fixture(root)
            protocol=json.loads((root/'protocol.json').read_text())
            for row in protocol['baseline_noise']:row['baseline_a_us']=[10.7]*40
            # 7% observed imbalance demands >14%, not the ordinary 10% floor.
            write(root/'protocol.json',protocol);repin_fixture(root)
            with patch('kerneldarwin.workload.environment_identity',return_value=environment):
                with self.assertRaisesRegex(ValueError,'acceptance criterion'):verify_study(root)

    def test_repinned_invalid_limits_and_session_protocol_refuse(self):
        changes=[('limits','candidates_per_shape',5),('limits','portfolio_kernels',3),('limits','max_rules',129),
                 ('limits','source_bytes',24000),('limits','development_wall_seconds',901),('limits','memory_bytes',3*1024**3),
                 ('limits','cache_bytes',512*1024**2),('confirmation','sessions',2),('confirmation','repeats_per_session',41),
                 ('development','repeats',41)]
        for section,key,value in changes:
            with self.subTest(section=section,key=key),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);environment=fixture(root)
                protocol=json.loads((root/'protocol.json').read_text());protocol[section][key]=value
                write(root/'protocol.json',protocol);repin_fixture(root)
                with patch('kerneldarwin.workload.environment_identity',return_value=environment),self.assertRaises(ValueError):verify_study(root)

    def test_repinned_invalid_rules_or_manifest_sources_refuse(self):
        changes=[lambda p:p['rules'][0].update(shape=[0,384]),lambda p:p['rules'][0].update(shape=[64,2049]),
                 lambda p:p['rules'][0].update(shape=[True,384]),lambda p:p['rules'][0].update(candidate='unrecognized'),
                 lambda p:p['rules'].append(copy.deepcopy(p['rules'][0])),lambda p:p['sources'].update({_LOADED_SOURCE[0][0]:'0'*64}),
                 lambda p:p.update(rules=[{'shape':shape,'candidate':name,'dtype':'float32','layout':'any_mlx_strides'} for shape,name in (([64,384],'simd'),([1,384],'simd4'),([64,768],'vector4'))])]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);environment=fixture(root)
                policy=json.loads((root/'policy-frozen.json').read_text());change(policy);write(root/'policy-frozen.json',policy);repin_fixture(root)
                with patch('kerneldarwin.workload.environment_identity',return_value=environment),self.assertRaises(ValueError):verify_study(root)


if __name__=='__main__':unittest.main()
