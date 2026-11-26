"""Snapshot and complete no-clobber publication controls."""
import concurrent.futures,json,tempfile,unittest,warnings
from pathlib import Path
from unittest.mock import patch
from test_core import record
from kerneldarwin.records import ExperimentStore,validate_record

class RecordPublicationTests(unittest.TestCase):
    def test_snapshot_fault_retry_collision_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            store=ExperimentStore(d);r=record();real=validate_record
            def mutate(snapshot):
                real(snapshot);r['title']='changed caller'
            with patch('kerneldarwin.records.validate_record',side_effect=mutate):path=store.save(r)
            self.assertNotIn('title',json.loads(path.read_bytes()))
            saved=path.read_bytes()
            with self.assertRaises(FileExistsError):store.save(r)
            self.assertEqual(path.read_bytes(),saved)
            r['id']='fault'
            with patch('kerneldarwin.records.os.fsync',side_effect=OSError('late failure')):
                with self.assertRaises(OSError):store.save(r)
            self.assertFalse((Path(d)/'fault.json').exists());self.assertFalse(list(Path(d).glob('.record-*')))
            path=store.save(r);self.assertEqual(store.load('fault')['title'],'changed caller')
            r['id']='cleanup'
            with patch('pathlib.Path.unlink',side_effect=PermissionError('cleanup denied')),warnings.catch_warnings(record=True) as caught:
                path=store.save(r)
            self.assertTrue(path.exists());self.assertIn('Published complete output',str(caught[0].message))
            self.assertEqual(len(list(Path(d).glob('.record-*'))),1)

    def test_concurrent_writers_have_one_complete_owner(self):
        with tempfile.TemporaryDirectory() as d:
            store=ExperimentStore(d);r=record()
            def write(i):
                try:store.save({**r,'title':str(i)});return 'saved'
                except FileExistsError:return 'collision'
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:self.assertEqual(sorted(pool.map(write,range(2))),['collision','saved'])
            self.assertIn(store.load(r['id'])['title'],('0','1'));self.assertFalse(list(Path(d).glob('.record-*')))
