"""Workflow gates and actual CPU controls; GPU execution is not simulated."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin.cli import main, export_record, regression
from kerneldarwin.engine import blocked_record, run_isolated
from kerneldarwin.records import ExperimentStore, WorkloadSpec, validate_record


class WorkflowTests(unittest.TestCase):
    def test_cuda_unavailable_records_blocker_and_exits_two(self):
        capability = {"cuda_available": False, "triton_installed": False,
                      "requirements": "NVIDIA CUDA host required"}
        with tempfile.TemporaryDirectory() as directory, patch("kerneldarwin.gpu.doctor", return_value=capability):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["investigate", "--shape", "3", "17", "--store", directory])
            self.assertEqual(code, 2)
            summary = json.loads(output.getvalue())
            saved = json.loads(Path(summary["record"]).read_text())
            validate_record(saved)
            self.assertEqual(saved["status"], "blocked")
            self.assertEqual(saved["evidence_kind"], "reference")
            self.assertEqual(saved["experiments"], [])
            self.assertFalse(saved["provenance"]["gpu_execution"])
            with self.assertRaises(ValueError):
                export_record(saved, Path(directory) / "unverified.py")
            self.assertFalse((Path(directory) / "unverified.py").exists())

    def test_resource_gate_precedes_backend_work(self):
        for argv in (["control", "--repeats", "19"], ["investigate", "--shape", "100000000", "20"]):
            with self.subTest(argv=argv), patch("kerneldarwin.gpu.doctor") as doctor, redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 1)
                doctor.assert_not_called()

    def test_worker_uses_allowlisted_environment_and_timeout(self):
        # Artificial environment verifies policy without reading credential stores.
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PATH": "/usr/bin", "UNNECESSARY_SECRET": "test-only"}, clear=True):
                with patch("kerneldarwin.engine.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
                    run_isolated(WorkloadSpec("sum", (3, 17)), Path(directory) / "result.json", wall_seconds=9)
                passed = run.call_args.kwargs
                self.assertNotIn("UNNECESSARY_SECRET", passed["env"])
                self.assertNotIn("HOME", passed["env"])
                self.assertEqual(passed["timeout"], 39)
                self.assertEqual(passed["env"]["PYTHONNOUSERSITE"], "1")
                self.assertNotIn("shell", passed)

    def test_record_comparison_refuses_semantic_mismatch(self):
        before = blocked_record(WorkloadSpec("sum", (3, 17)), "blocked")
        before.update(status="no_improvement", evidence_kind="cpu_control")
        before["baseline"] = {"samples_us": [10.] * 20}
        after = copy.deepcopy(before)
        after["workload"]["shape"] = [3, 18]
        with self.assertRaises(ValueError):
            regression(before, after)
        after = copy.deepcopy(before)
        after["baseline"]["samples_us"] = [20.] * 20
        result = regression(before, after)
        self.assertTrue(result["regression_detected"])
        self.assertEqual(result["latency_ratio_after_over_before"], 2)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "install .[control] for actual NumPy controls")
class ActualControlTests(unittest.TestCase):
    def test_all_families_strides_and_dtypes_validate_roundtrip(self):
        from kerneldarwin.control import run_control
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(directory)
            for family in ("sum", "softmax", "layernorm", "fused"):
                for dtype in ("float32", "float16"):
                    with self.subTest(family=family, dtype=dtype):
                        spec = WorkloadSpec(family, (3, 17), dtype=dtype, stride=2)
                        record = run_control(spec, repeats=20, batches=(1, 4))
                        validate_record(record)
                        self.assertEqual(record["evidence_kind"], "cpu_control")
                        self.assertFalse(record["provenance"]["gpu_execution"])
                        self.assertEqual(record["workload"]["actual_strides"], [34, 2])
                        self.assertTrue(all(exp["correctness"]["passed"] for exp in record["experiments"]))
                        self.assertEqual(len(record["experiments"]), 2)
                        self.assertEqual(WorkloadSpec.from_dict(record["workload"]), spec)
                        store.save(record)
                        self.assertEqual(store.load(record["id"]), record)

    def test_skipped_computation_is_rejected(self):
        import numpy as np
        from kerneldarwin.control import run_control
        def cheating_candidate(family, x, eps, batch):
            return np.zeros_like(x)
        with patch("kerneldarwin.control.batched", new=cheating_candidate):
            record = run_control(WorkloadSpec("softmax", (3, 17)), repeats=20, batches=(4,))
        self.assertEqual(record["status"], "no_improvement")
        self.assertIsNone(record["winner_id"])
        self.assertEqual(record["experiments"][0]["status"], "invalid")
        self.assertFalse(record["experiments"][0]["correctness"]["passed"])


if __name__ == "__main__":
    unittest.main()
