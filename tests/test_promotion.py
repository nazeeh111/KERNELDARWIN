"""Synthetic timing evidence checks, never hardware measurements."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_core import confirmed_record
from kerneldarwin.promotion import promotion_decision,evidence_identity
from kerneldarwin.records import validate_record,ExperimentStore
from kerneldarwin.statistics import compare_paired
from kerneldarwin.cli import export_record


def fixture():
    r=confirmed_record();r.update(evidence_kind='cuda_run',device={'kind':'cuda','name':'synthetic'},environment={'threads':1},provenance={'gpu_execution':True,'source_sha256':'a'*64})
    from kerneldarwin.promotion import authored_source_pair
    sources=authored_source_pair();r['experiments'][0]['source']=sources
    import hashlib
    r['provenance']['source_sha256']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    r['baseline']={'name':'pytorch_eager','samples_us':[10.]*20}
    r['compiled_baseline']={'status':'measured','correctness_passed':True,'samples_us':[7.]*20}
    e=r['experiments'][0];e['config']={'block_size':256,'num_warps':4}
    for phase in (e['timing'],e['confirmation']):
        a=[9.]*20;b=[3.]*20
        phase['compiled']={'baseline_us':a,'candidate_us':b,'comparison':compare_paired(a,b,seed=42 if phase is e['timing'] else 7042),'correctness_passed':True}
        phase['evidence_identity']=evidence_identity(r,e)
    return r


class PromotionTests(unittest.TestCase):
    def test_all_comparators_and_confirmation_required(self):
        r=fixture();validate_record(r);self.assertTrue(promotion_decision(r,'candidate')['eligible'])
        for phase in ('timing','confirmation'):
            for change in ('lose','missing','incorrect','corrupt'):
                x=copy.deepcopy(r);p=x['experiments'][0][phase]
                if change=='missing':p.pop('compiled')
                elif change=='incorrect':p['compiled']['correctness_passed']=False
                elif change=='corrupt':p['compiled']['comparison']['ci_low']=99
                else:
                    p['compiled']['baseline_us']=[1.]*20;p['compiled']['candidate_us']=[3.]*20
                    p['compiled']['comparison']=compare_paired([1.]*20,[3.]*20,seed=42)
                with self.subTest(phase=phase,change=change):
                    self.assertFalse(promotion_decision(x,'candidate')['eligible'])
                    with self.assertRaises(ValueError):validate_record(x)
        for field in ('correctness_passed','status'):
            x=copy.deepcopy(r)
            if field=='status':x['compiled_baseline'][field]='unavailable'
            else:x['compiled_baseline'][field]=False
            self.assertFalse(promotion_decision(x,'candidate')['eligible'])
        x=copy.deepcopy(r);x['experiments'][0].pop('confirmation')
        self.assertTrue(promotion_decision(x,'candidate',confirmation=False)['eligible'])
        self.assertFalse(promotion_decision(x,'candidate')['eligible'])

    def test_exact_bindings_and_no_mutation(self):
        r=fixture();before=copy.deepcopy(r)
        self.assertTrue(promotion_decision(r,'candidate')['eligible']);self.assertEqual(r,before)
        with patch('kerneldarwin.promotion.EVALUATOR_IDENTITY','changed'):
            self.assertFalse(promotion_decision(r,'candidate')['eligible'])
        for owner,key,value in [('environment','threads',True),('environment','threads',1.0),('provenance','source_sha256','b'*64),('workload','stride',True),('device','name','changed')]:
            x=copy.deepcopy(r);x[owner][key]=value
            self.assertFalse(promotion_decision(x,'candidate')['eligible'])
        x=copy.deepcopy(r);x['experiments'][0]['config']['num_warps']=True
        self.assertFalse(promotion_decision(x,'candidate')['eligible'])
        x=copy.deepcopy(r);x['experiments'][0]['confirmation']['compiled']['comparison']['seed']=42
        self.assertFalse(promotion_decision(x,'candidate')['eligible'])
        x=copy.deepcopy(r);x['experiments'][0]['status']='failed'
        self.assertFalse(promotion_decision(x,'candidate')['eligible'])

    def test_direct_export_revalidates_before_any_write_or_backend(self):
        r=fixture();r['experiments'][0]['confirmation']['compiled']['correctness_passed']=False
        with tempfile.TemporaryDirectory() as d,patch('kerneldarwin.gpu.source_pair') as source:
            with self.assertRaises(ValueError):export_record(r,Path(d)/'out.py')
            source.assert_not_called();self.assertEqual(list(Path(d).iterdir()),[])

    def test_sources_and_named_reference_are_required(self):
        r=fixture()
        for source in (None,{}, {'malicious':'unrelated'}, {**r['experiments'][0]['source'],'candidate':'changed'}):
            x=copy.deepcopy(r);x['experiments'][0]['source']=source
            self.assertFalse(promotion_decision(x,'candidate')['eligible'])
            with self.assertRaises(ValueError):validate_record(x)
        cpu=confirmed_record()
        for owner,key,value in [('baseline','name','unknown'),('provenance','gpu_execution',True),('provenance','source_sha256','0'*64),('provenance','measurement','invented')]:
            x=copy.deepcopy(cpu);x[owner][key]=value
            self.assertFalse(promotion_decision(x,'candidate')['eligible'])
            with self.assertRaises(ValueError):validate_record(x)
        x=copy.deepcopy(cpu);x['baseline']={};x['provenance']={}
        self.assertFalse(promotion_decision(x,'candidate')['eligible'])
        with self.assertRaises(ValueError):validate_record(x)

    def test_cpu_scope_and_legacy_eager_only_refusal(self):
        cpu=confirmed_record();self.assertEqual(promotion_decision(cpu,'candidate')['scope'],'cpu_named_reference')
        r=fixture()
        for phase in ('timing','confirmation'):r['experiments'][0][phase].pop('compiled')
        self.assertFalse(promotion_decision(r,'candidate')['eligible'])
