"""CPU-only manufactured manifests verify guards, never performance evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin import transaction_plan as plan

ENV={'device':'synthetic CPU fixture; not a GPU validation'}


def fixture():
    return {'schema':1,'operation':plan.OPERATION,'epsilon':plan.EPSILON,
        'key_policy':plan.rank.KEY_POLICY,'compile_options':{'math_mode':'safe'},
        'source_sha256':plan.source_identity(),'study_identity':'c'*64,'environment':copy.deepcopy(ENV),
        'evidence':{'numerical_passed':True,'confirmation_passed':True},
        'sources':{name:hashlib.sha256(data).hexdigest() for name,data in plan.rank._SOURCES},
        'rules':[{'shape':[1,3,7,37,97,5],'document_layout':'host_contiguous',
                  'query_layout':'host_contiguous','dtype':'float32','candidate':'simd32'}]}


class TransactionPlanContractTests(unittest.TestCase):
    def setUp(self):
        self.environment=patch.object(plan,'environment_identity',return_value=copy.deepcopy(ENV))
        self.identity=patch.object(plan,'study_identity',return_value='c'*64)
        self.environment.start();self.identity.start()
        self.addCleanup(self.environment.stop);self.addCleanup(self.identity.stop)

    def artifact(self,root,manifest):
        root=Path(root);(root/'manifest.json').write_text(json.dumps(manifest))
        for name,data in plan.rank._SOURCES:(root/name).write_bytes(data)
        return root

    def test_default_missing_and_valid_manufactured_artifact(self):
        self.assertEqual(plan.TransactionPlan().status,'library_only')
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(plan.TransactionPlan(Path(temp)/'absent').status.startswith('fallback:'))
            root=self.artifact(temp,fixture());loaded=plan.TransactionPlan(root)
            self.assertEqual(loaded.status,'installed')
            self.assertEqual(len(loaded._rules),1)

    def test_source_study_device_and_operation_fail_closed(self):
        changes=[('schema',True),('schema',2),('operation','stable_top5_pairs_v1'),
                 ('source_sha256','a'*64),('study_identity','a'*64),('environment',{}),
                 ('compile_options',{'math_mode':'fast'}),('key_policy','unstable'),('sources',{})]
        with tempfile.TemporaryDirectory() as temp:
            for key,value in changes:
                data=fixture();data[key]=value
                loaded=plan.TransactionPlan(self.artifact(temp,data))
                self.assertTrue(loaded.status.startswith('fallback:'),(key,loaded.status))
                self.assertFalse(loaded._rules)

    def test_numerical_and_confirmation_flags_are_exact_booleans(self):
        for key in ('numerical_passed','confirmation_passed'):
            for value in (False,1,'true',None):
                data=fixture();data['evidence'][key]=value
                with self.assertRaises(ValueError):plan.TransactionPlan()._load(data,confirmed=True)
        data=fixture();data['evidence']['confirmation_passed']=False
        self.assertEqual(plan.TransactionPlan._trial(data).status,'numerical_trial')
        data['evidence']['numerical_passed']=False
        with self.assertRaises(ValueError):plan.TransactionPlan._trial(data)

    def test_malformed_data_and_unavailable_backend_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            for data in ([],None,{'rules':[]},{**fixture(),'rules':[None]}):
                self.assertTrue(plan.TransactionPlan(self.artifact(temp,data)).status.startswith('fallback:'))
            root=self.artifact(temp,fixture())
            with patch.object(plan,'environment_identity',side_effect=ImportError('Metal unavailable')):
                self.assertTrue(plan.TransactionPlan(root).status.startswith('fallback:'))

    def test_empty_duplicate_and_excessive_rules_are_rejected(self):
        data=fixture()
        for rules in ([],data['rules']*2,data['rules']*129):
            value=copy.deepcopy(data);value['rules']=rules
            with self.assertRaises(ValueError):plan.TransactionPlan()._load(value,confirmed=True)
        data['rules']=[]
        self.assertFalse(plan.TransactionPlan._trial(data)._rules)

    def test_unsupported_rule_semantics_and_resource_domains(self):
        changes=[('shape',[0,3,7,37,97,5]),('shape',[64,65,37,97,5]),
                 ('shape',[1,3,7,2049,97,5]),('shape',[1,3,7,37,2049,5]),
                 ('shape',[128,2048,2048,5]),('shape',[1,3,7,37,97,3]),
                 ('shape',[True,3,7,37,97,5]),('shape',[1,3,7,37,97.0,5]),
                 ('shape',[1,2,3]),('document_layout','unknown'),
                 ('query_layout','any_mlx_strides'),('dtype','float16'),('candidate','arbitrary_shader')]
        for key,value in changes:
            data=fixture();data['rules'][0][key]=value
            with self.assertRaises(ValueError,msg=str((key,value))):plan.TransactionPlan()._load(data,confirmed=True)

    def test_readable_source_must_match_and_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.artifact(temp,fixture());name,data=plan.rank._SOURCES[0]
            (root/name).write_bytes(data[:-1]+b'x')
            self.assertTrue(plan.TransactionPlan(root).status.startswith('fallback:'))
            (root/name).unlink();(root/'separate').write_bytes(data);(root/name).symlink_to(root/'separate')
            self.assertTrue(plan.TransactionPlan(root).status.startswith('fallback:'))
            (root/'manifest.json').rename(root/'original.json');(root/'manifest.json').symlink_to(root/'original.json')
            self.assertTrue(plan.TransactionPlan(root).status.startswith('fallback:'))

    def test_bounded_regular_reads_reject_oversize_and_fifo(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'manifest.json';file.write_bytes(b' '*1_000_001)
            self.assertTrue(plan.TransactionPlan(file).status.startswith('fallback:'))
            file.unlink()
            if hasattr(os,'mkfifo'):
                os.mkfifo(file)
                self.assertTrue(plan.TransactionPlan(file).status.startswith('fallback:'))

    def test_duplicate_and_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.artifact(temp,fixture());text=json.dumps(fixture())
            bad=[text.replace('"schema": 1','"schema": 0, "schema": 1',1),
                 text[:-1]+', "extra": NaN}',text[:-1]+', "extra": 1e309}']
            for value in bad:
                (root/'manifest.json').write_text(value)
                self.assertTrue(plan.TransactionPlan(root).status.startswith('fallback:'))

    def test_profiled_host_strides_and_unknown_requests(self):
        import numpy as np
        n,d=37,97;queries=[np.zeros((q,d),np.float32) for q in (1,3,7)]
        arrays={'host_contiguous':np.zeros((n,d),np.float32),
                'host_column_stride2':np.zeros((n,d*2),np.float32)[:,::2],
                'host_transpose':np.zeros((d,n),np.float32).T}
        for layout,docs in arrays.items():
            self.assertEqual(plan.host_signature(docs,queries,5),plan.rule_key((1,3,7),n,d,5,layout))
        docs=arrays['host_contiguous']
        for value in (docs[:,::-1],docs[::-1],docs.astype(np.float16),object()):
            self.assertIsNone(plan.host_signature(value,queries,5))
        self.assertIsNone(plan.host_signature(docs,[q[:,::-1] for q in queries],5))
        self.assertIsNone(plan.host_signature(docs,queries,3))
        self.assertIsNone(plan.host_signature(docs,[np.zeros((0,d),np.float32)],5))

    def test_changed_dispatcher_source_is_rejected(self):
        original=Path.read_bytes
        def read(path):
            return b'changed source' if path==Path(plan.__file__) else original(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaisesRegex(ValueError,'changed after import'):plan.source_identity()

    def test_cached_environment_cannot_be_mutated_by_callers(self):
        self.environment.stop()
        with patch.object(plan,'_environment_snapshot',return_value=json.dumps(ENV)):
            first=plan.environment_identity();first['device']='changed'
            self.assertEqual(plan.environment_identity(),ENV)
        self.environment.start()


@unittest.skipUnless(os.getenv('KD_RUN_METAL')=='1','Actual Metal execution is opt-in')
class TransactionPlanMetalTests(unittest.TestCase):
    completed_methods=[]
    gpu_executed=False

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        if not mx.metal.is_available():raise unittest.SkipTest('Native Metal unavailable')
        mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
        x=mx.arange(128,dtype=mx.float32);y=mx.sum(x*x);mx.eval(y)
        if y.item()!=690880:raise AssertionError('Actual GPU arithmetic failed')
        cls.mx,cls.np=mx,np
        cls.completed_methods=[];cls.gpu_executed=True

    def tearDown(self):
        self.mx.synchronize();type(self).completed_methods.append(self._testMethodName)

    def request(self,seed=49011,rows=(1,3,7),layout='host_contiguous'):
        np=self.np;rng=np.random.default_rng(seed)
        dense=rng.normal(size=(37,97)).astype(np.float32)
        if layout=='host_column_stride2':
            backing=np.zeros((37,194),np.float32);backing[:,::2]=dense;docs=backing[:,::2]
        elif layout=='host_transpose':docs=dense.T.copy().T
        else:docs=dense
        return docs,[rng.normal(size=(q,97)).astype(np.float32) for q in rows]

    def trial(self,variant='simd32',rows=(1,3,7),layouts=plan.LAYOUTS):
        data=fixture();data['study_identity']=plan.study_identity();data['environment']=plan.environment_identity()
        data['evidence']['confirmation_passed']=False
        data['rules']=[{'shape':[*rows,37,97,5],'document_layout':layout,'query_layout':'host_contiguous',
                       'dtype':'float32','candidate':variant} for layout in layouts]
        return plan.TransactionPlan._trial(data)

    def check(self,output,docs,queries,tolerance=2e-5):
        np,mx=self.np,self.mx
        def normalize(x):
            if isinstance(x,mx.array):x=np.array(x.astype(mx.float32))
            x=np.asarray(x,np.float64);peak=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-12)
            scaled=x/peak
            return scaled/np.maximum(np.sqrt(np.sum(scaled*scaled,axis=1,keepdims=True)),1e-12/peak)
        self.assertEqual(len(output),len(queries))
        for (ids,values),query in zip(output,queries):
            ids,values=np.asarray(ids),np.asarray(values);ref=normalize(query)@normalize(docs).T
            self.assertEqual(ids.shape,(query.shape[0],5))
            self.assertTrue(all(len(set(row))==5 for row in ids.tolist()))
            np.testing.assert_allclose(values,np.take_along_axis(ref,ids,axis=1),rtol=tolerance,atol=tolerance/10)
            np.testing.assert_allclose(values,-np.sort(-ref,axis=1)[:,:5],rtol=tolerance,atol=tolerance/10)

    def test_known_host_layouts_execute_both_authored_trials(self):
        for variant in plan.tx.CANDIDATES:
            policy=self.trial(variant)
            for layout in plan.LAYOUTS:
                docs,queries=self.request(layout=layout)
                self.check(policy.search(docs,queries),docs,queries)
                self.assertEqual(policy.last_decision.implementation,variant)
            self.assertEqual((policy.calls,policy.fallback_calls),(3,0))

    def test_unknown_layout_dtype_boundary_and_mlx_inputs_fall_back(self):
        policy=self.trial();docs,queries=self.request()
        readonly=docs.copy();readonly.flags.writeable=False
        unaligned=self.np.ndarray(docs.shape,dtype=self.np.float32,buffer=bytearray(docs.nbytes+1),offset=1)
        unaligned[:]=docs
        requests=[(docs[:,::-1],[q[:,::-1] for q in queries]),
                  (docs.astype(self.np.float16),[q.astype(self.np.float16) for q in queries]),
                  (readonly,queries),(unaligned,queries),
                  (self.np.broadcast_to(docs[:1],docs.shape),queries),
                  (self.mx.array(docs),[self.mx.array(q) for q in queries])]
        for a,b in requests:
            self.check(policy.search(a,b),a,b,tolerance=.03 if isinstance(a,self.np.ndarray) and a.dtype==self.np.float16 else 2e-5)
            self.assertEqual(policy.last_decision.implementation,'library')
        self.assertEqual(policy.calls,policy.fallback_calls)

    def test_changed_aliasing_inputs_and_nonfinite_recovery(self):
        np=self.np;policy=self.trial(rows=(37,37,37))
        for seed in (49021,49022):
            docs,_=self.request(seed);queries=[docs,docs,docs];before=docs.copy()
            self.check(policy.search(docs,queries),docs,queries)
            self.assertEqual(policy.last_decision.implementation,'simd32')
            np.testing.assert_array_equal(docs,before)
            bad=docs.copy();bad[0,0]=np.inf
            with self.assertRaises(ValueError):policy.search(bad,[bad,bad,bad])
            self.check(policy.search(docs,queries),docs,queries)

    def test_rank_adapter_uses_evaluated_scores_and_bounded_dispatch(self):
        np,mx=self.np,self.mx
        docs,queries=self.request();signature=plan.host_signature(docs,queries,5)
        scores=np.random.default_rng(49031).normal(size=(11,37)).astype(np.float32)
        expected=np.argsort(-scores,axis=1,kind='stable')[:,:5]
        for variant in plan.tx.CANDIDATES:
            policy=self.trial(variant);ids,values=policy._rank_trial(signature,mx.array(scores));mx.eval(ids,values)
            np.testing.assert_array_equal(np.array(ids),expected)
            np.testing.assert_array_equal(np.array(values).view(np.uint32),np.take_along_axis(scores.view(np.uint32),expected,axis=1))
            self.assertEqual((policy.calls,policy.fallback_calls),(1,0))
