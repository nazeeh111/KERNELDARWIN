"""CPU parser/resource guards; actual compute is separately validated on Metal."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from kerneldarwin import transaction_cli


class TransactionCLITests(unittest.TestCase):
    def test_batch_command_parser_preserves_explicit_artifact(self):
        from kerneldarwin.cli import parser
        args=parser().parse_args(['workload','batch','input.npz','--artifact','plan','--k','3'])
        self.assertEqual((args.workload_command,args.vectors,args.artifact,args.k),('batch',Path('input.npz'),Path('plan'),3))

    def test_two_and_three_dimensional_queries_preserve_boundaries(self):
        docs=np.arange(30,dtype=np.float32).reshape(6,5)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz'
            for q in (np.zeros((2,5),np.float32),np.zeros((3,2,5),np.float32)):
                np.savez(path,documents=docs,queries=q)
                a,b=transaction_cli.load_batches(path);np.testing.assert_array_equal(a,docs)
                self.assertEqual([x.shape for x in b],[(2,5)]*(1 if q.ndim==2 else 3))

    def test_numbered_queries_keep_variable_rows_and_dtype(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz'
            np.savez(path,documents=np.zeros((6,5),np.float32),queries_0=np.zeros((1,5),np.float16),queries_1=np.zeros((0,5),np.float32))
            _,queries=transaction_cli.load_batches(path)
            self.assertEqual([q.shape[0] for q in queries],[1,0]);self.assertEqual(queries[0].dtype,np.float16)

    def test_query_keys_are_unambiguous_and_bounded(self):
        docs=np.zeros((6,5),np.float32);q=np.zeros((1,5),np.float32)
        cases=[{'documents':docs},{'documents':docs,'queries':q,'queries_0':q},
               {'documents':docs,'queries_1':q},{'documents':docs,'queries_00':q},
               {'documents':docs,**{f'queries_{i}':q for i in range(17)}}]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz'
            for arrays in cases:
                np.savez(path,**arrays)
                with self.assertRaises(ValueError):transaction_cli.load_batches(path)

    def test_object_and_implicit_double_arrays_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz'
            for dtype in (np.float64,object):
                np.savez(path,documents=np.zeros((6,5),dtype),queries=np.zeros((1,5),np.float32))
                with patch.object(np,'load',side_effect=AssertionError('Unvalidated payload loaded')):
                    with self.assertRaises(ValueError):transaction_cli.load_batches(path)

    def test_declared_huge_shape_rejected_before_array_allocation(self):
        header=io.BytesIO();np.lib.format.write_array_header_1_0(header,{'descr':'<f4','fortran_order':False,'shape':(100_000_000,100_000_000)})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz'
            with zipfile.ZipFile(path,'w') as z:z.writestr('documents.npy',header.getvalue())
            with patch.object(np,'load',side_effect=AssertionError('Unvalidated allocation')):
                with self.assertRaisesRegex(ValueError,'payload'):transaction_cli.load_batches(path)

    def test_duplicate_zip_entries_and_non_npz_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz';path.write_bytes(b'not a zip')
            with self.assertRaises(ValueError):transaction_cli.load_batches(path)
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('documents.npy',b'x')
                with self.assertWarns(UserWarning):z.writestr('documents.npy',b'x')
            with self.assertRaisesRegex(ValueError,'entries'):transaction_cli.load_batches(path)

    def test_unsupported_entry_names_are_not_extracted(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz'
            with zipfile.ZipFile(path,'w') as z:z.writestr('../outside.npy',b'x')
            with self.assertRaisesRegex(ValueError,'flat numeric'):transaction_cli.load_batches(path)
            self.assertFalse((Path(temp).parent/'outside.npy').exists())
