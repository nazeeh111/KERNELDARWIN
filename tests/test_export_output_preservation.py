"""Output persistence controls only; no GPU execution or evidence attestation.

Execute the actual export function body with inert source/spec dependencies so
these filesystem regressions also run without Torch, Triton or a CUDA device.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class ExportOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / 'new-directory' / 'optimized.py'
        self.original = b'another writer owns these bytes\n'
        source = Path(__file__).resolve().parents[1] / 'src/kerneldarwin/cli.py'
        tree = ast.parse(source.read_text())
        bodies = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'export_record']
        self.assertEqual(len(bodies), 1)
        package = ModuleType('_kernel_output_control')
        package.__path__ = []
        gpu = ModuleType('_kernel_output_control.gpu')
        gpu.source_pair = lambda: {'baseline': 'inert', 'candidate': 'inert'}
        package.gpu = gpu
        modules = patch.dict(sys.modules, {'_kernel_output_control': package, '_kernel_output_control.gpu': gpu})
        modules.start()
        self.addCleanup(modules.stop)
        namespace = {'__package__': '_kernel_output_control', 'json': json,
                     'WorkloadSpec': SimpleNamespace(from_dict=lambda data: SimpleNamespace(to_dict=lambda: dict(data)))}
        exec(compile(ast.Module(body=bodies, type_ignores=[]), str(source), 'exec'), namespace)
        self.export = namespace['export_record']
        self.record = {
            'id': 'filesystem-control', 'evidence_kind': 'cuda_run', 'status': 'accepted',
            'winner_id': 'candidate', 'device': {'name': 'inert-device'},
            'workload': {'family': 'sum', 'shape': [1, 1], 'dtype': 'float32', 'stride': 1, 'eps': 1e-5, 'seed': 0},
            'experiments': [{'id': 'candidate', 'config': {}, 'confirmation': {'comparison': {'accepted': True}}}],
            'provenance': {'source_sha256': hashlib.sha256(json.dumps(gpu.source_pair(), sort_keys=True).encode()).hexdigest()},
        }

    def test_new_export_preserves_serialized_workload_and_source_identity(self):
        result = self.export(self.record, self.output)
        self.assertEqual(result['record_id'], self.record['id'])
        module = ast.parse(self.output.read_text())
        spec = next(node for node in module.body if isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and target.id == 'SPEC' for target in node.targets))
        metadata = json.loads(spec.value.args[0].value)
        self.assertEqual(metadata['workload'], self.record['workload'])
        self.assertEqual(metadata['source_sha256'], self.record['provenance']['source_sha256'])
        self.assertEqual(metadata['device'], self.record['device'])

    def test_initial_existing_output_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(self.original)
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            self.export(self.record, self.output)
        self.assertEqual(self.output.read_bytes(), self.original)

    def test_output_created_after_existence_check_is_preserved(self):
        mkdir = Path.mkdir
        def other_writer(path, *args, **kwargs):
            result = mkdir(path, *args, **kwargs)
            if path == self.output.parent:
                self.output.write_bytes(self.original)
            return result
        with patch.object(Path, 'mkdir', new=other_writer):
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                self.export(self.record, self.output)
        self.assertEqual(self.output.read_bytes(), self.original)

    def test_dangling_output_symlink_does_not_create_its_target(self):
        self.output.parent.mkdir()
        target = self.root / 'must-remain-absent.py'
        try:
            self.output.symlink_to(target)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f'This platform cannot create the test symlink: {error}')
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            self.export(self.record, self.output)
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())

    def test_serialization_error_does_not_create_output(self):
        self.record['experiments'][0]['config'] = {'invalid': float('nan')}
        with self.assertRaises(ValueError):
            self.export(self.record, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
