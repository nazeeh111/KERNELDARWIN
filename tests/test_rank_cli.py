"""CPU command routing checks; no numerical or performance evidence."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin import workload_cli


class RankCliTests(unittest.TestCase):
    def test_rank_manifest_uses_guarded_rank_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"operation": "stable_top5_pairs_v1"}))
            # This deliberately incomplete fixture selects the correct adapter,
            # which must then reject it and retain complete library fallback.
            adapter, scope = workload_cli.application_adapter(root)
            self.assertEqual(scope, "top5_ranking")
            self.assertTrue(adapter.status.startswith("fallback:"))
            self.assertEqual(adapter._rules, {})

    def test_rank_export_uses_independent_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"target": "stable_top5_pairs_v1"}))
            args = argparse.Namespace(workload_command="export", study=root, output=root / "artifact")
            with patch("kerneldarwin.rank_export.export_artifact", return_value={"routed": True}) as export:
                self.assertEqual(workload_cli.run(args), {"routed": True})
                export.assert_called_once_with(root, root / "artifact")

    def test_incomplete_rank_study_cannot_create_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"target": "stable_top5_pairs_v1"}))
            args = argparse.Namespace(workload_command="export", study=root, output=root / "artifact")
            with self.assertRaises((OSError, ValueError)):
                workload_cli.run(args)
            self.assertFalse(args.output.exists())
