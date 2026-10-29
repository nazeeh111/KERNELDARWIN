"""Executed with KD_RUN_METAL=1 on an authorized Apple Silicon GPU.

Default CI explicitly skips these tests. A skip is never GPU evidence.
"""
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.getenv("KD_RUN_METAL") == "1", "requires explicit authorized Metal execution")
class MetalWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        cls.mx, cls.np = mx, np
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(2*1024**3)

    def assert_reference(self, x, candidate):
        from kerneldarwin.workload import candidate_normalize, library_normalize
        mx, np = self.mx, self.np
        before = np.array(x).copy()
        a = before.astype(np.float64)
        # float64 max scaling is independent of the Metal float32 reduction.
        scale = np.maximum(np.max(np.abs(a),axis=1,keepdims=True),1e-12)
        z = a / scale
        reference = z / np.maximum(np.sqrt(np.sum(z*z,axis=1,keepdims=True)),1e-12/scale)
        y = library_normalize(x) if candidate == "library" else candidate_normalize(x,candidate)
        mx.eval(y)
        np.testing.assert_allclose(np.array(y),reference,rtol=2e-5,atol=2e-6,equal_nan=True)
        np.testing.assert_array_equal(np.array(x),before)

    def test_adversarial_magnitudes_cancellation_and_tails(self):
        from kerneldarwin.workload import CANDIDATES
        mx, np = self.mx, self.np
        rng=np.random.default_rng(177)
        for width in (1,3,17,31,32,33,127,129,384,513,768,1536,2048):
            a=rng.normal(size=(7,width)).astype(np.float32)
            a[0]=0;a[1]*=1e30;a[2]*=1e-30;a[3,::2]*=-1;a[4]*=1e-12
            a[5]=np.nextafter(np.float32(0),np.float32(1));a[6]=np.finfo(np.float32).max
            for candidate in ("library",*CANDIDATES):
                with self.subTest(width=width,candidate=candidate):self.assert_reference(mx.array(a),candidate)

    def test_layouts_offsets_repeated_calls_and_changed_contents(self):
        from kerneldarwin.workload import CANDIDATES
        mx, np = self.mx, self.np
        rng=np.random.default_rng(287)
        for seed in range(3):
            a=mx.array(rng.normal(size=(33,388)).astype(np.float32))
            layouts=[a[:,1:385],a[:,3:387],a[::-1,1:385],a[:,1:385].T,a[:,::3],a[::-1,::-1],mx.broadcast_to(a[:1,:],(17,388))]
            for i,x in enumerate(layouts):
                for candidate in CANDIDATES:
                    with self.subTest(seed=seed,layout=i,candidate=candidate):self.assert_reference(x,candidate)

    def test_nonfinite_rows_and_library_fallback_contract(self):
        from kerneldarwin.workload import CANDIDATES, Normalizer, candidate_normalize, library_normalize
        mx, np = self.mx, self.np
        x=mx.array([[float('nan'),1],[float('inf'),1],[-float('inf'),0],[0,0]],dtype=mx.float32)
        for name in ("library",*CANDIDATES):
            y=library_normalize(x) if name=="library" else candidate_normalize(x,name)
            actual=np.array(y)
            self.assertTrue(np.isnan(actual[:3]).all());self.assertTrue((actual[3]==0).all())
        normalize=Normalizer()
        for shape in ((0,17),(3,0)):
            self.assertEqual(normalize(mx.zeros(shape)).shape,shape)
        for dtype in (mx.float16,mx.bfloat16):
            y=normalize(mx.array([[3,4]],dtype=dtype));self.assertEqual(y.dtype,dtype)
            np.testing.assert_allclose(np.array(y.astype(mx.float32)),[[.6,.8]],rtol=.006,atol=.004)
        with self.assertRaises(ValueError):normalize(mx.ones((2,3)),mode='train')
        with self.assertRaises(ValueError):normalize(mx.ones((2,3,4)))
        with self.assertRaises(TypeError):normalize(mx.ones((2,3),dtype=mx.int32))

    def test_tiny_normal_values_are_not_silently_zeroed(self):
        from kerneldarwin.workload import CANDIDATES, candidate_normalize, library_normalize
        mx,np=self.mx,self.np
        a=np.array([[1e-30,-2e-30,3e-30],[1e-15,-2e-15,3e-15]],dtype=np.float32)
        expected=a.astype(np.float64)/1e-12
        for name in ("library",*CANDIDATES):
            x=mx.array(a);y=library_normalize(x) if name=="library" else candidate_normalize(x,name)
            np.testing.assert_allclose(np.array(y),expected,rtol=2e-5,atol=0)

    def test_optimized_library_alternatives_and_injected_regression(self):
        from kerneldarwin.library_baselines import eager_normalize,fast_library_normalize
        import importlib.util
        mx,np=self.mx,self.np
        values=np.array([[0,0,0],[1e30,-1e30,1e30],[1e-30,-2e-30,3e-30],[float('inf'),1,0],[float('nan'),0,1]],dtype=np.float32)
        reference=np.array([[0,0,0],[1/3**.5,-1/3**.5,1/3**.5],[1e-18,-2e-18,3e-18],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])
        for name,f in (("eager",eager_normalize),("fast_rms",fast_library_normalize)):
            y=np.array(f(mx.array(values)))
            np.testing.assert_allclose(y,reference,rtol=2e-5,atol=0,equal_nan=True)
        path=Path(__file__).resolve().parents[1]/'examples/regressions/unstable_norm.py'
        spec=importlib.util.spec_from_file_location('injected_regression',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        actual=np.array(module.unstable(mx.array(values[1:2])))
        self.assertFalse(np.allclose(actual,reference[1:2],rtol=2e-5,atol=2e-6))
        self.assertTrue((actual==0).all())

    def test_static_dispatch_changed_inputs_and_plan_invalidations(self):
        import hashlib,json,tempfile
        from kerneldarwin.workload import Normalizer,source_identity,environment_identity,_LOADED_SOURCE
        from kerneldarwin.study_identity import implementation_identity
        mx,np=self.mx,self.np
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name,source in _LOADED_SOURCE:(path/name).write_bytes(source)
            plan={"schema":1,"operation":"stable_l2_rows_v1","epsilon":1e-12,"source_sha256":source_identity(),"study_identity":implementation_identity(),"environment":environment_identity(),
                  "sources":{name:hashlib.sha256(source).hexdigest() for name,source in _LOADED_SOURCE},
                  "evidence":{"numerical_passed":True,"confirmation_passed":True},
                  "rules":[{"shape":[5,33],"candidate":"vector4","dtype":"float32","layout":"any_mlx_strides"}]}
            file=path/'manifest.json';file.write_text(json.dumps(plan))
            n=Normalizer(path);self.assertEqual(n.status,'installed')
            rng=np.random.default_rng(1004)
            for i in range(3):
                x=mx.array(rng.normal(size=(5,33)).astype(np.float32));expected=np.array(x).astype(np.float64)
                expected/=np.linalg.norm(expected,axis=1,keepdims=True)
                np.testing.assert_allclose(np.array(n(x)),expected,rtol=2e-5,atol=2e-6)
                self.assertEqual(n.last_decision.implementation,'vector4')
            np.testing.assert_allclose(np.array(n(mx.array([[3.,4.]]))),[[.6,.8]],rtol=2e-5,atol=2e-6)
            self.assertEqual(n.fallback_calls,1)
            for mutation in ('unconfirmed','environment','source','study','dtype','portfolio'):
                p=json.loads(json.dumps(plan))
                if mutation=='unconfirmed':p['evidence']['confirmation_passed']=False
                if mutation=='environment':p['environment']['os_build']='changed'
                if mutation=='source':p['source_sha256']='changed'
                if mutation=='study':p['study_identity']='changed'
                if mutation=='dtype':p['rules'][0]['dtype']='float16'
                if mutation=='portfolio':p['rules']=[{**p['rules'][0],'shape':[i+1,33],'candidate':name} for i,name in enumerate(('simd','simd4','vector4'))]
                file.write_text(json.dumps(p));rejected=Normalizer(path)
                self.assertTrue(rejected.status.startswith('fallback:'),mutation)
            file.write_text(json.dumps(plan));(path/'normalize_simd.metal').write_text('tampered')
            self.assertTrue(Normalizer(path).status.startswith('fallback:'))

    def test_float32_subnormal_behavior_is_explicit(self):
        from kerneldarwin.workload import CANDIDATES, candidate_normalize, library_normalize
        mx,np=self.mx,self.np
        tiny=np.nextafter(np.float32(0),np.float32(1))
        x=mx.array(np.array([[tiny,-tiny]],dtype=np.float32))
        for name in ("library",*CANDIDATES):
            y=library_normalize(x) if name=="library" else candidate_normalize(x,name)
            # Apple Metal float32 arithmetic flushes these input denormals.
            np.testing.assert_array_equal(np.array(y),[[0.,0.]])

    def test_application_changes_inputs_and_matches_independent_scores(self):
        from kerneldarwin.search import CosineIndex,CompiledCosineIndex
        from kerneldarwin.workload import candidate_normalize,library_normalize
        mx, np = self.mx, self.np
        rng=np.random.default_rng(388)
        for changed in range(2):
            docs=rng.normal(size=(41,33)).astype(np.float32)
            qs=rng.normal(size=(5,33)).astype(np.float32)
            docs64=docs.astype(np.float64);q64=qs.astype(np.float64)
            scores=(q64/np.linalg.norm(q64,axis=1,keepdims=True))@(docs64/np.linalg.norm(docs64,axis=1,keepdims=True)).T
            reference=np.sort(scores,axis=-1)[:,-5:][:,::-1]
            # The direct offline candidate has no dtype/empty fallback. This
            # adapter supplies the same semantic fallback as a runtime plan.
            guarded=lambda x:candidate_normalize(x,'vector4') if x.dtype==mx.float32 and x.size else library_normalize(x)
            for factory in (lambda:CosineIndex(mx.array(docs),guarded),lambda:CosineIndex(mx.array(docs)),lambda:CompiledCosineIndex(mx.array(docs),'stable'),lambda:CompiledCosineIndex(mx.array(docs),'fast')):
                index=factory();ids,actual=index.search(mx.array(qs),5)
                np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=2e-6)
                self.assertEqual(len(ids),5)
                for row in ids:
                    self.assertEqual(len(set(row)),5)
                    self.assertTrue(all(type(i) is int and 0<=i<len(docs) for i in row))
                np.testing.assert_allclose(actual,np.take_along_axis(scores,np.array(ids),axis=1),rtol=2e-5,atol=2e-6)
                self.assertEqual(index.search(mx.zeros((0,33)),1),([],[]))
                self.assertEqual(len(index.search(mx.array(qs),1)[0][0]),1)
                self.assertEqual(len(index.search(mx.array(qs,dtype=mx.float16),2)[0][0]),2)

    def test_application_rejects_nonfinite_vectors_explicitly(self):
        from kerneldarwin.search import CosineIndex,CompiledCosineIndex
        mx=self.mx
        with self.assertRaisesRegex(ValueError,'must be finite'):
            CosineIndex(mx.array([[float('nan'),1.],[1.,2.]]))
        for factory in (CosineIndex,lambda x:CompiledCosineIndex(x,'stable'),lambda x:CompiledCosineIndex(x,'fast')):
            index=factory(mx.array([[1.,2.],[3.,4.]]));before=index.search(mx.array([[1.,2.]]),1)
            with self.assertRaisesRegex(ValueError,'must be finite'):index.search(mx.array([[float('inf'),1.]]),1)
            with self.assertRaisesRegex(ValueError,'must be finite'):index.refresh(mx.array([[float('nan'),1.],[2.,3.]]))
            self.assertEqual(index.search(mx.array([[1.,2.]]),1),before)


if __name__=='__main__':unittest.main()
