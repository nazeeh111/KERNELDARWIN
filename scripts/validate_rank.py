#!/usr/bin/env python3
"""Run untimed production rank Metal tests and write one immutable report."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before importing a backend. Concurrent callers cannot both run.
    # A terminated run leaves its partial file intact and cannot be relabeled.
    with args.output.open("x") as handle:
        started = time.perf_counter()
        flags = sorted({"MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "METAL_DEVICE_WRAPPER_TYPE",
                        "METAL_DEBUG_ERROR_MODE"} | {
            key for key in os.environ if key.startswith(("MTL_DEBUG", "MTL_SHADER_VALIDATION"))})
        record = {"schema": 1, "kind": "executed_rank_metal_numerical_validation",
            "source_sha256": None, "study_identity": None, "environment": None,
            "gpu_executed": False, "success": False, "tests_run": 0,
            "failures": 0, "errors": 0, "skipped": 0,
            "requested_validation_environment": {key: os.getenv(key) for key in flags},
            "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "test_sources": {"test_rank_search.py": hashlib.sha256(
                (ROOT / "tests/test_rank_search.py").read_bytes()).hexdigest()},
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "candidate_timing": False, "candidate_selection": False, "accepted_artifact": False,
            "limitations": ["Numerical validity alone does not establish acceleration.",
                "Native Metal and GPU-validation flags require external log verification.",
                "Synthetic manifests exercise production guards and never establish performance acceptance.",
                "FTZ comparison keys preserve selected original float32 bits; inference-only contract."]}
        stream = io.StringIO()
        module = None
        try:
            os.environ["KD_RUN_METAL"] = "1"
            from kerneldarwin.rank_search import rank_source_identity, rank_implementation_identity
            from kerneldarwin.workload import environment_identity
            record.update(source_sha256=rank_source_identity(), study_identity=rank_implementation_identity(),
                          environment=environment_identity())
            spec = importlib.util.spec_from_file_location("kd_rank_validation_tests", ROOT / "tests/test_rank_search.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.RankSearchTests)
            planned = suite.countTestCases()
            if planned < 8:
                raise ValueError("At least eight actual Metal test methods are required")
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            record.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                          skipped=len(result.skipped), gpu_executed=module.RankSearchTests.gpu_executed,
                          completed_methods=list(module.RankSearchTests.completed_methods))
            if record["gpu_executed"]:
                record["peak_memory_bytes"] = module.RankSearchTests.mx.get_peak_memory()
            record["success"] = (result.wasSuccessful() and not result.skipped and result.testsRun >= 8
                and len(record["completed_methods"]) == result.testsRun and record["gpu_executed"] is True)
        except Exception:
            record["errors"] += 1
            stream.write(traceback.format_exc())
            if module is not None:
                record["gpu_executed"] = bool(module.RankSearchTests.gpu_executed)
        record["wall_seconds"] = time.perf_counter() - started
        record["output"] = stream.getvalue()
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(stream.getvalue())
    print(json.dumps({key: record[key] for key in
        ("success", "gpu_executed", "tests_run", "failures", "errors", "skipped", "source_sha256")}))
    return 0 if record["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
