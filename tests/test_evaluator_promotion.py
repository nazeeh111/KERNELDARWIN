"""Owning engine with CPU fake evaluator: no GPU timing or execution evidence."""
import importlib.util,unittest
from unittest.mock import patch
from types import SimpleNamespace
from kerneldarwin.engine import investigate_cuda
from kerneldarwin.records import WorkloadSpec,validate_record

@unittest.skipUnless(importlib.util.find_spec('numpy'),'existing NumPy runtime required')
class FakeEvaluatorTests(unittest.TestCase):
    def run_fake(self, *, compiled_loss=False, compiled_bad=False, budget=180):
        import numpy as np
        class Input:
            def stride(self):return [17,1]
        torch=SimpleNamespace(compile=lambda f,**k:f,cuda=SimpleNamespace(synchronize=lambda:None),
            allclose=lambda a,b,**k:np.allclose(a,b),isfinite=np.isfinite)
        def pairs(a,b,**kwargs):
            a();b()  # Execute evaluator callables, but timings below are declared fixtures.
            if compiled_loss and kwargs['seed'] in (1002,8001):return [1.]*20,[3.]*20
            return [10.]*20,[3.]*20
        with patch('kerneldarwin.gpu.require_cuda',return_value=(torch,SimpleNamespace(__version__='fake'))),patch('kerneldarwin.gpu.device_info',return_value={'kind':'cuda','name':'FAKE-CPU-TEST'}),patch('kerneldarwin.gpu.make_input',return_value=Input()),patch('kerneldarwin.gpu.reference',return_value=np.array([1.])),patch('kerneldarwin.gpu.candidate',return_value=np.array([1.])),patch('kerneldarwin.gpu.configurations',return_value=[{'block_size':256,'num_warps':4}]),patch('kerneldarwin.gpu.verify',return_value={'passed':True}),patch('kerneldarwin.gpu.time_pairs',side_effect=pairs),patch('kerneldarwin.gpu.source_pair',return_value=__import__('kerneldarwin.promotion',fromlist=['authored_source_pair']).authored_source_pair()):
            torch.__version__='fake'
            if compiled_bad:torch.compile=lambda f,**k:lambda x:np.array([2.])
            r=investigate_cuda(WorkloadSpec('sum',(1,17)),repeats=20,max_candidates=1,wall_seconds=budget)
            validate_record(r)
            return r

    def test_real_engine_decisions_with_declared_fake_evaluator(self):
        r=self.run_fake();self.assertEqual(r['status'],'accepted');self.assertEqual(r['winner_id'],'c000')
        for kwargs in ({'compiled_loss':True},{'compiled_bad':True},{'budget':0}):
            with self.subTest(kwargs=kwargs):
                r=self.run_fake(**kwargs);self.assertIsNone(r['winner_id']);self.assertEqual(r['status'],'no_improvement')
