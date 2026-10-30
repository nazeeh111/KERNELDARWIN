"""CPU checks for source-bound fused installation; fixtures are synthetic."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin.fused_search import FusedPlan,score_source_identity,fused_implementation_identity


class FusedContractTests(unittest.TestCase):
    def test_installed_pin_matches_measurement_source(self):
        import kerneldarwin.fused_search as f
        root=Path(__file__).resolve().parents[1]
        pin=json.loads((Path(f.__file__).parent/'fused-study-source.json').read_text())
        self.assertEqual(pin['measurement_script_sha256'],hashlib.sha256((root/'scripts/fused_study.py').read_bytes()).hexdigest())
        self.assertEqual(len(fused_implementation_identity()),64)

    def test_fused_identity_includes_base_application(self):
        expected=fused_implementation_identity()
        with patch('kerneldarwin.search._LOADED_APPLICATION_IDENTITY','changed'):
            self.assertNotEqual(fused_implementation_identity(),expected)
        with patch('pathlib.Path.read_bytes',side_effect=AssertionError('must use source snapshot')):
            self.assertEqual(score_source_identity(),score_source_identity())

    def test_malformed_stale_and_oversized_plan_keep_library(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            for contents in ('{','[]','null','{}',' '*1_000_001,json.dumps({'schema':1,'operation':'stable_l2_scores_v1','epsilon':1e-12,'source_sha256':'old'})):
                path.write_text(contents)
                with patch('kerneldarwin.fused_search.environment_identity',side_effect=AssertionError('invalid source must not probe GPU')):
                    plan=FusedPlan(path)
                self.assertTrue(plan.status.startswith('fallback:'))
                self.assertEqual(plan._rules,{})

    def test_no_artifact_is_explicit_library_mode(self):
        plan=FusedPlan()
        self.assertEqual(plan.status,'library_only')
        self.assertEqual(plan.calls,0)


if __name__=='__main__':unittest.main()
