"""Actual Metal checks for target two. Skipped by CPU-only CI explicitly."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.getenv("KD_RUN_METAL") == "1", "Actual Metal execution is opt-in")
class FusedSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import kerneldarwin.fused_search as fused
        cls.mx, cls.np, cls.fused = mx, np, fused
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(2 * 1024**3)

    def reference(self, q, docs):
        np = self.np
        q = np.array(q).astype(np.float64)
        # Reflect the explicitly declared native float32 subnormal behavior.
        q[np.abs(q) < np.finfo(np.float32).tiny] = 0
        peak = np.maximum(np.max(np.abs(q), axis=1, keepdims=True), 1e-12)
        z = q / peak
        normalized = z / np.maximum(np.sqrt(np.sum(z*z, axis=1, keepdims=True)), 1e-12/peak)
        return normalized @ np.array(docs).astype(np.float64).T

    def documents(self, n, d, seed=7):
        rng = self.np.random.default_rng(seed)
        raw = rng.normal(size=(n,d))
        raw /= self.np.linalg.norm(raw,axis=1,keepdims=True)
        return self.mx.array(raw.astype(self.np.float32))

    def assert_scores(self, q, docs, *, atol=2e-6, rtol=2e-5):
        expected = self.reference(q, docs)
        before_q, before_d = self.np.array(q).copy(), self.np.array(docs).copy()
        for name in self.fused.SCORE_CANDIDATES:
            with self.subTest(candidate=name, shape=q.shape, documents=docs.shape):
                value = self.fused.candidate_scores(q, docs, name)
                self.mx.eval(value)
                self.np.testing.assert_allclose(self.np.array(value), expected, rtol=rtol, atol=atol)
        for kind in ("stable", "fast"):
            value = self.fused.library_scores(q, docs, kind)
            self.mx.eval(value)
            self.np.testing.assert_allclose(self.np.array(value), expected, rtol=rtol, atol=atol)
        self.np.testing.assert_array_equal(self.np.array(q), before_q)
        self.np.testing.assert_array_equal(self.np.array(docs), before_d)

    def test_widths_tails_and_changed_contents(self):
        rng = self.np.random.default_rng(11)
        for width in (1,3,31,32,33,127,128,129,513,2048):
            for count in (1,5):
                docs = self.documents(count,width)
                q = self.mx.array(rng.normal(size=(3,width)).astype(self.np.float32))
                self.assert_scores(q,docs)
                self.assert_scores(q * -1.7 + .13,docs)

    def test_strides_offsets_negative_and_broadcast(self):
        mx, np = self.mx, self.np
        rng = np.random.default_rng(13)
        raw = rng.normal(size=(4, 129)).astype(np.float32)
        docs = self.documents(7,129)
        padded = np.zeros((4, 259),np.float32); padded[:,1::2] = raw
        qviews = [mx.array(raw.T.copy()).T, mx.array(padded)[:,1::2], mx.array(raw)[:,::-1], mx.array(raw)[::-1], mx.array(raw)[1:3], mx.broadcast_to(mx.array(raw[:1]),(4,129))]
        dh = np.array(docs); dp = np.zeros((7,259),np.float32);dp[:,1::2]=dh
        dviews = [mx.array(dh.T.copy()).T,mx.array(dp)[:,1::2],docs[:,::-1],docs[::-1],docs[1:6],mx.broadcast_to(docs[:1],(7,129))]
        for q in qviews:
            for d in dviews:
                self.assert_scores(q,d)

    def test_extremes_cancellation_and_tiny_values(self):
        mx, np = self.mx,self.np
        docs = self.documents(5,513,43)
        rng = np.random.default_rng(17)
        raw = rng.normal(size=(4,513)).astype(np.float32)
        raw[0] *= np.finfo(np.float32).max / (2*np.max(np.abs(raw[0])))
        raw[1] *= 1e-30
        raw[2] = 0
        raw[3] = np.where(np.arange(513)%2,-1.,1.)
        self.assert_scores(mx.array(raw),docs,atol=2e-6)
        self.assert_scores(mx.full((2,513),1e-35),mx.full((5,513),1/np.sqrt(513)),atol=0,rtol=3e-5)
        self.assert_scores(mx.full((2,513),np.nextafter(np.float32(0),np.float32(1))),docs,atol=0,rtol=0)

    def test_nonfinite_rows_are_nan(self):
        mx, np = self.mx,self.np
        q = np.ones((4,33),np.float32)
        q[0,3]=np.nan;q[1,5]=np.inf;q[2,1]=-np.inf
        for variant in self.fused.SCORE_CANDIDATES:
            out = np.array(self.fused.candidate_scores(mx.array(q),self.documents(5,33),variant))
            self.assertTrue(np.isnan(out[:3]).all())
            self.assertTrue(np.isfinite(out[3]).all())

    def test_application_ids_empty_mixed_dtype_and_rollback(self):
        mx,np=self.mx,self.np
        rng=np.random.default_rng(61)
        vectors=mx.array(rng.normal(size=(9,33)).astype(np.float32))
        queries=mx.array(rng.normal(size=(3,33)).astype(np.float32))
        for name in self.fused.SCORE_CANDIDATES:
            index=self.fused.FusedCosineIndex(vectors,lambda q,d,name=name:self.fused.candidate_scores(q,d,name))
            expected=self.reference(queries,index.vectors)
            ids,values=index.search(queries,k=4)
            for r,(row,scores) in enumerate(zip(ids,values)):
                self.assertEqual(len(set(row)),4)
                self.assertTrue(all(0<=v<9 for v in row))
                np.testing.assert_allclose(scores,expected[r,row],rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(scores,np.sort(expected[r])[::-1][:4],rtol=2e-5,atol=2e-6)
            self.assertEqual(index.search(mx.zeros((0,33))),([],[]))
            before=np.array(index.vectors).copy()
            with self.assertRaises(ValueError):index.refresh(mx.full((9,33),float('nan')))
            np.testing.assert_array_equal(np.array(index.vectors),before)
            with self.assertRaises(ValueError):index.search(mx.full((3,33),float('inf')))
        fallback=self.fused.FusedPlan()
        index=fallback.index(vectors.astype(mx.float16))
        for dtype in (mx.float16,mx.bfloat16,mx.float32):
            ids,values=index.search(queries.astype(dtype),k=3)
            self.assertEqual(len(ids),3)
        self.assertEqual(fallback.calls,3)
        self.assertEqual(fallback.fallback_calls,3)

    def manifest(self):
        f=self.fused
        return {"schema":1,"operation":"stable_l2_scores_v1","epsilon":1e-12,
            "source_sha256":f.score_source_identity(),"study_identity":f.fused_implementation_identity(),
            "environment":f.environment_identity(),"sources":{name:hashlib.sha256(content).hexdigest() for name,content in f._SOURCES},
            "rules":[{"shape":[3,9,33],"candidate":"score_tile4","dtype":"float32","layout":"any_mlx_strides"}],
            "evidence":{"numerical_passed":True,"confirmation_passed":False}}

    def test_static_plan_changed_input_fallback_and_invalid_artifacts(self):
        f,mx,np=self.fused,self.mx,self.np
        manifest=self.manifest()
        trial=f.FusedPlan._trial(manifest)
        vectors=self.documents(9,33)
        for seed in (1,2,3):
            q=mx.array(np.random.default_rng(seed).normal(size=(3,33)).astype(np.float32))
            self.assert_scores(q,vectors)
            np.testing.assert_allclose(np.array(trial.score(q,vectors)),self.reference(q,vectors),rtol=2e-5,atol=2e-6)
            trial.index(vectors).search(q,k=3)
        self.assertEqual(trial.calls,6);self.assertEqual(trial.fallback_calls,0)
        trial.index(vectors).search(mx.ones((2,33)),k=3)
        self.assertEqual(trial.fallback_calls,1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name,content in f._SOURCES:(path/name).write_bytes(content)
            (path/'manifest.json').write_text(json.dumps(manifest))
            self.assertTrue(f.FusedPlan(path).status.startswith('fallback:'))
            # Manufactured confirmed fixtures test guards only, not performance.
            manifest['evidence']['confirmation_passed']=True
            (path/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(f.FusedPlan(path).status,'installed')
            for key,bad in (('source_sha256','0'*64),('study_identity','0'*64),('environment',{})):
                altered=copy.deepcopy(manifest);altered[key]=bad
                (path/'manifest.json').write_text(json.dumps(altered))
                self.assertTrue(f.FusedPlan(path).status.startswith('fallback:'))
            (path/'manifest.json').write_text(json.dumps(manifest))
            (path/f.SCORE_SOURCE_FILES[0]).write_text('tampered')
            self.assertTrue(f.FusedPlan(path).status.startswith('fallback:'))
        with self.assertRaises(ValueError):trial.score(mx.ones((3,33)),vectors,mode='autodiff')

    def test_candidate_limits_and_regression_detection(self):
        f,mx,np=self.fused,self.mx,self.np
        for q,docs in ((mx.ones((0,3)),mx.ones((5,3))), (mx.ones((1,2049)),mx.ones((1,2049))), (mx.ones((129,1)),mx.ones((1,1)))):
            with self.assertRaises(ValueError):f.candidate_scores(q,docs,'score_simd')
        with self.assertRaises(TypeError):f.candidate_scores(mx.ones((1,3),dtype=mx.float16),mx.ones((5,3)),'score_simd')
        with self.assertRaises(ValueError):f.candidate_scores(mx.ones((1,3)),mx.ones((5,4)),'score_simd')
        with self.assertRaises(ValueError):f.candidate_scores(mx.ones((128,2048)),mx.ones((2048,2048)),'score_simd')
        self.assert_scores(mx.ones((128,1)),mx.ones((2048,1)))
        q=mx.full((2,33),1e30);docs=self.documents(5,33)
        broken=(q/mx.sqrt(mx.sum(q*q,axis=1,keepdims=True)))@docs.T
        with self.assertRaises(AssertionError):np.testing.assert_allclose(np.array(broken),self.reference(q,docs),rtol=2e-5,atol=2e-6)

    def test_compiled_plan_query_and_score_use_changed_arrays(self):
        mx,np,f=self.mx,self.np,self.fused
        rng=np.random.default_rng(911)
        # Both variants, narrow/irregular/wide rows and document tails exercise
        # the production compiled graph, with only synthetic dispatch rules.
        for variant in f.SCORE_CANDIDATES:
            for q_count,n,width in ((1,5,1),(3,9,33),(5,17,513),(17,161,1536),(2,5,2048)):
                manifest=self.manifest()
                manifest['rules']=[{'shape':[q_count,n,width],'candidate':variant,'dtype':'float32','layout':'any_mlx_strides'}]
                plan=f.FusedPlan._trial(manifest)
                for change in (0,1):
                    raw=mx.array(rng.normal(size=(n,width)).astype(np.float32))
                    q=mx.array(rng.normal(size=(q_count,width)).astype(np.float32))
                    if change:q=q[:,::-1]
                    index=plan.index(raw)
                    expected=self.reference(q,index.vectors)
                    before=plan.calls
                    ids,values=index.search(q,k=min(3,n))
                    self.assertEqual(plan.calls,before+1)
                    self.assertEqual(plan.fallback_calls,0)
                    ids=np.asarray(ids);values=np.asarray(values)
                    self.assertTrue(all(len(set(row.tolist()))==min(3,n) for row in ids))
                    self.assertTrue(np.all((ids>=0)&(ids<n)))
                    np.testing.assert_allclose(values,np.take_along_axis(expected,ids,axis=1),rtol=2e-5,atol=2e-6)
                    np.testing.assert_allclose(values,-np.sort(-expected,axis=1)[:,:min(3,n)],rtol=2e-5,atol=2e-6)
                    np.testing.assert_allclose(np.array(plan.score(q,index.vectors)),expected,rtol=2e-5,atol=2e-6)
