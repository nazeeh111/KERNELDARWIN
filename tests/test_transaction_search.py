"""Batch transaction semantics; CPU contracts and opt-in actual Metal checks."""
import os
import unittest


class TransactionContractTests(unittest.TestCase):
    def test_execution_mode_and_batch_bound_precede_backend_work(self):
        from kerneldarwin.transaction_search import search_batches
        for batches in ([], [None]*17, None):
            with self.assertRaises(ValueError):search_batches(None,batches)
        with self.assertRaisesRegex(ValueError,'inference'):
            search_batches(None,[None],mode='training')

    def test_static_bounds_reject_unsupported_custom_domains(self):
        from kerneldarwin.transaction_search import _supported_custom
        self.assertTrue(_supported_custom([1,8,17],161,1536,5))
        for args in (([0,1],161,384,5),([64,65],161,384,5),([1],4,384,5),
                     ([1],161,2049,5),([128],2048,2048,5),([1],161,384,True),
                     ([True],161,384,5),([1],161,384.0,5)):
            self.assertFalse(_supported_custom(*args),args)


@unittest.skipUnless(os.getenv('KD_RUN_METAL')=='1','Actual Metal execution is opt-in')
class TransactionSearchTests(unittest.TestCase):
    completed_methods=[]
    gpu_executed=False

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import kerneldarwin.transaction_search as transaction
        if not mx.metal.is_available():raise unittest.SkipTest('Native Metal unavailable')
        mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
        probe=mx.arange(128,dtype=mx.float32);value=mx.sum(probe*probe);mx.eval(value)
        if value.item()!=690880:raise AssertionError('Actual GPU arithmetic probe failed')
        cls.mx,cls.np,cls.tx=mx,np,transaction
        cls.completed_methods=[];cls.gpu_executed=True

    def tearDown(self):
        self.mx.synchronize();type(self).completed_methods.append(self._testMethodName)

    def arrays(self,n=37,d=97,rows=(1,3,7),seed=44101):
        rng=self.np.random.default_rng(seed)
        return self.mx.array(rng.normal(size=(n,d)).astype('float32')),[
            self.mx.array(rng.normal(size=(q,d)).astype('float32')) for q in rows]

    def reference(self,docs,query):
        np,mx=self.np,self.mx
        def normalized(value):
            x=np.array(value.astype(mx.float32)).astype(np.float64)
            scale=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-12)
            x=x/scale
            return x/np.maximum(np.sqrt(np.sum(x*x,axis=1,keepdims=True)),1e-12/scale)
        return normalized(query)@normalized(docs).T

    def check(self,output,docs,queries,k=5,tolerance=2e-5):
        np=self.np
        self.assertEqual(len(output),len(queries))
        for (ids,values),query in zip(output,queries):
            if query.shape[0]==0:
                self.assertEqual((ids,values),([],[]));continue
            ids,values=np.asarray(ids),np.asarray(values)
            self.assertEqual(ids.shape,(query.shape[0],k));self.assertEqual(ids.shape,values.shape)
            self.assertTrue(np.issubdtype(ids.dtype,np.integer));self.assertTrue(np.all(ids>=0));self.assertTrue(np.all(ids<docs.shape[0]))
            self.assertTrue(all(len(set(row))==k for row in ids.tolist()))
            reference=self.reference(docs,query)
            np.testing.assert_allclose(values,np.take_along_axis(reference,ids,axis=1),rtol=tolerance,atol=tolerance/10)
            np.testing.assert_allclose(values,-np.sort(-reference,axis=1)[:,:k],rtol=tolerance,atol=tolerance/10)

    def test_public_variable_batch_boundaries(self):
        for rows in ((1,),(2,5),(0,1,4),(1,)*16):
            docs,queries=self.arrays(rows=rows)
            self.check(self.tx.search_batches(docs,queries),docs,queries)

    def test_all_library_and_authored_paths_match_batch_reference(self):
        for n,d,rows in ((64,384,(1,1,1)),(161,768,(8,8,8)),(161,1536,(17,17,17)),(5,31,(1,3,2))):
            docs,queries=self.arrays(n,d,rows)
            canonical=self.tx.search_batches(docs,queries)
            for method in (*self.tx.LIBRARIES,*self.tx.CANDIDATES):
                result=self.tx._execute(method,docs,queries);self.check(result,docs,queries)
                if method in self.tx.CANDIDATES:
                    for (ids,values),(refids,refvalues) in zip(result,canonical):
                        self.np.testing.assert_array_equal(ids,refids)
                        self.np.testing.assert_array_equal(self.np.asarray(values,dtype='float32').view('uint32'),self.np.asarray(refvalues,dtype='float32').view('uint32'))

    def test_strides_offsets_broadcasts_and_input_preservation(self):
        mx,np=self.mx,self.np
        docs,queries=self.arrays();host=np.array(docs);qh=[np.array(q) for q in queries]
        def view(x,kind):
            if kind=='negative':return mx.array(x)[:,::-1]
            if kind=='transpose':return mx.array(x.T.copy()).T
            if kind=='broadcast':return mx.broadcast_to(mx.array(x[:1]),x.shape)
            backing=np.zeros((x.shape[0]+2,x.shape[1]*2+3),np.float32)
            backing[1:-1,1:1+x.shape[1]*2:2]=x
            return mx.array(backing)[1:-1,1:1+x.shape[1]*2:2]
        for kind in ('negative','transpose','broadcast','offset_stride2'):
            a,b=view(host,kind),[view(q,kind) for q in qh]
            before=[np.array(x).copy() for x in [a,*b]]
            for method in ('stable_library',*self.tx.CANDIDATES):self.check(self.tx._execute(method,a,b),a,b)
            for x,original in zip([a,*b],before):np.testing.assert_array_equal(np.array(x),original)

    def test_changed_contents_and_changed_documents(self):
        for seed in (44111,44112,44113):
            docs,queries=self.arrays(seed=seed)
            for method in ('stable_library',*self.tx.CANDIDATES):
                self.check(self.tx._execute(method,docs,queries),docs,queries)
            docs[0,:]=self.mx.ones((docs.shape[1],))
            queries[0][:]=0
            self.check(self.tx.search_batches(docs,queries),docs,queries)

    def test_zero_ties_and_adversarial_magnitudes(self):
        mx,np=self.mx,self.np
        for scale in (0.0,1e-20,1e20):
            docs,queries=self.arrays();docs=docs*scale;queries=[q*scale for q in queries]
            for method in (*self.tx.LIBRARIES,*self.tx.CANDIDATES):
                output=self.tx._execute(method,docs,queries);self.check(output,docs,queries)
                if scale==0:
                    for (ids,_),q in zip(output,queries):np.testing.assert_array_equal(ids,np.broadcast_to(np.arange(5),(q.shape[0],5)))

    def test_nonfinite_rejection_does_not_mutate_inputs(self):
        mx,np=self.mx,self.np
        for bad in (float('inf'),float('-inf'),float('nan')):
            for where in ('docs','query'):
                docs,queries=self.arrays();(docs if where=='docs' else queries[1])[0,0]=bad
                original=[np.array(x).copy() for x in [docs,*queries]]
                for method in ('stable_library',*self.tx.CANDIDATES):
                    with self.assertRaises(ValueError):self.tx._execute(method,docs,queries)
                for x,before in zip([docs,*queries],original):np.testing.assert_array_equal(np.array(x),before)

    def test_empty_batches_still_validate_document_refresh(self):
        docs,queries=self.arrays(rows=(0,0,0))
        self.assertEqual(self.tx.search_batches(docs,queries),[([],[])]*3)
        docs[0,0]=float('inf')
        with self.assertRaisesRegex(ValueError,'index'):self.tx.search_batches(docs,queries)

    def test_unsupported_custom_shapes_and_k_use_library_results(self):
        for n,rows,k in ((37,(64,65),5),(3,(2,3),3),(37,(2,3),1)):
            docs,queries=self.arrays(n=n,rows=rows)
            for method in self.tx.CANDIDATES:self.check(self.tx._execute(method,docs,queries,k=k),docs,queries,k)

    def test_lower_precision_and_mixed_query_dtypes(self):
        mx=self.mx
        docs,queries=self.arrays()
        for dtype in (mx.float16,mx.bfloat16):
            a,b=docs.astype(dtype),[q.astype(dtype) for q in queries]
            self.check(self.tx.search_batches(a,b),a,b,tolerance=.03)
        mixed=[queries[0].astype(mx.float16),queries[1],queries[2].astype(mx.bfloat16)]
        for method in ('stable_library',*self.tx.CANDIDATES):self.check(self.tx._execute(method,docs,mixed),docs,mixed,tolerance=.03)

    def test_invalid_shapes_types_and_k(self):
        docs,queries=self.arrays()
        for k in (True,0,38,1.5):
            with self.assertRaises(ValueError):self.tx.search_batches(docs,queries,k=k)
        with self.assertRaises(TypeError):self.tx.search_batches(self.np.array(docs),queries)
        with self.assertRaises(TypeError):self.tx.search_batches(docs,[queries[0].astype(self.mx.int32)])
        with self.assertRaises(ValueError):self.tx.search_batches(docs,[self.mx.zeros((1,96))])
        with self.assertRaises(ValueError):self.tx.search_batches(docs,[self.mx.zeros((97,))])
        with self.assertRaises(ValueError):self.tx.search_batches(self.mx.zeros((0,97)),queries)
