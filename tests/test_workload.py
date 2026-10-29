"""CPU-compatible contract checks; GPU arithmetic lives in test_metal_workload."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin.workload import CANDIDATES, Normalizer, source_identity


class WorkloadPlanTests(unittest.TestCase):
    def test_unavailable_native_identity_probe_keeps_library_fallback(self):
        import subprocess,sys,types
        from kerneldarwin.workload import environment_identity
        fake=types.ModuleType('mlx.core');fake.device_info=lambda:{}
        parent=types.ModuleType('mlx');parent.core=fake
        for error in (subprocess.TimeoutExpired('system_profiler',10),subprocess.CalledProcessError(1,'system_profiler')):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'manifest.json'
                path.write_text(json.dumps({'schema':1,'operation':'stable_l2_rows_v1','epsilon':1e-12,'source_sha256':source_identity(),'environment':{}}))
                with patch.dict(sys.modules,{'mlx':parent,'mlx.core':fake}),patch('subprocess.check_output',side_effect=error):
                    with self.assertRaisesRegex(RuntimeError,'probe unavailable'):environment_identity()
                    normalize=Normalizer(path)
                    self.assertTrue(normalize.status.startswith('fallback:'))
                    self.assertFalse(normalize._rules)

    def test_installed_measurement_source_pin_matches_repository(self):
        import hashlib
        import kerneldarwin.study_identity as binding
        metadata=json.loads((Path(binding.__file__).parent/'study-source.json').read_text())
        script=Path(__file__).resolve().parents[1]/'scripts/workload_study.py'
        self.assertEqual(metadata['measurement_script_sha256'],hashlib.sha256(script.read_bytes()).hexdigest())

    def test_study_identity_changes_with_application_or_competing_baseline(self):
        from kerneldarwin.study_identity import implementation_identity
        expected=implementation_identity()
        for target in ('kerneldarwin.search._LOADED_APPLICATION_IDENTITY','kerneldarwin.library_baselines._LOADED_BASELINE_IDENTITY'):
            with patch(target,'changed'):
                self.assertNotEqual(implementation_identity(),expected)

    def test_no_plan_is_explicit_library_mode(self):
        normalize = Normalizer()
        self.assertEqual(normalize.status, "library_only")
        self.assertEqual(normalize._rules, {})
        self.assertEqual(len(CANDIDATES), 4)
        self.assertEqual(len(source_identity()), 64)

    def test_loaded_source_identity_does_not_relabel_cached_code_on_disk_change(self):
        expected=source_identity()
        with patch("pathlib.Path.read_bytes", side_effect=AssertionError("source must be snapshotted")):
            self.assertEqual(source_identity(),expected)

    def test_malformed_missing_oversized_and_stale_plan_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"manifest.json"
            self.assertTrue(Normalizer(p).status.startswith("fallback:"))
            for content in ("{", "[]", "null", "{}", json.dumps({"schema": 999}), " "*1_000_001):
                p.write_text(content)
                normalizer = Normalizer(p)
                self.assertTrue(normalizer.status.startswith("fallback:"))
                self.assertFalse(normalizer._rules)

    def test_stale_source_never_probes_device_or_loads_code(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"manifest.json"
            p.write_text(json.dumps({"schema":1,"operation":"stable_l2_rows_v1","epsilon":1e-12,"source_sha256":"old"}))
            with patch("kerneldarwin.workload.environment_identity", side_effect=AssertionError("must short circuit")):
                normalize=Normalizer(p)
            self.assertIn("source or environment", normalize.status)
            self.assertFalse(normalize._rules)

    def test_artifact_source_cannot_be_relabelled_by_repinning_manifest(self):
        import hashlib
        from kerneldarwin.workload import _LOADED_SOURCE
        from kerneldarwin.study_identity import implementation_identity
        for mutation in ('bytes','bytes_and_manifest','manifest_only','extra_source'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                for name,content in _LOADED_SOURCE:(root/name).write_bytes(content)
                plan={'schema':1,'operation':'stable_l2_rows_v1','epsilon':1e-12,
                      'source_sha256':source_identity(),'study_identity':implementation_identity(),'environment':{},
                      'sources':{name:hashlib.sha256(content).hexdigest() for name,content in _LOADED_SOURCE},
                      'rules':[{'shape':[1,33],'candidate':'simd','dtype':'float32','layout':'any_mlx_strides'}],
                      'evidence':{'numerical_passed':True,'confirmation_passed':True}}
                path=root/'manifest.json';path.write_text(json.dumps(plan))
                with patch('kerneldarwin.workload.environment_identity',return_value={}):
                    self.assertEqual(Normalizer(root).status,'installed')
                    name=_LOADED_SOURCE[0][0]
                    if mutation in ('bytes','bytes_and_manifest'):(root/name).write_bytes(b'changed authored source')
                    if mutation=='bytes_and_manifest':plan['sources'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
                    if mutation=='manifest_only':plan['sources'][name]='0'*64
                    if mutation=='extra_source':plan['sources']['unapproved.metal']='0'*64
                    path.write_text(json.dumps(plan))
                    rejected=Normalizer(root)
                    self.assertTrue(rejected.status.startswith('fallback:'))
                    self.assertEqual(rejected._rules,{})

    def test_duplicate_plan_rules_fail_closed(self):
        import hashlib
        from kerneldarwin.workload import _LOADED_SOURCE
        from kerneldarwin.study_identity import implementation_identity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,content in _LOADED_SOURCE:(root/name).write_bytes(content)
            rule={'shape':[1,33],'candidate':'simd','dtype':'float32','layout':'any_mlx_strides'}
            plan={'schema':1,'operation':'stable_l2_rows_v1','epsilon':1e-12,
                  'source_sha256':source_identity(),'study_identity':implementation_identity(),'environment':{},
                  'sources':{name:hashlib.sha256(content).hexdigest() for name,content in _LOADED_SOURCE},
                  'rules':[rule,rule], 'evidence':{'numerical_passed':True,'confirmation_passed':True}}
            (root/'manifest.json').write_text(json.dumps(plan))
            with patch('kerneldarwin.workload.environment_identity',return_value={}):
                rejected=Normalizer(root)
                self.assertIn('duplicate',rejected.status)
                self.assertEqual(rejected._rules,{})


if __name__ == "__main__":
    unittest.main()
