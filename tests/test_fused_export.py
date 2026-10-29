"""Synthetic mocked CPU promotion fixtures. These are never GPU evidence."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin import fused_export as exporter
from kerneldarwin import fused_search as fused


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orders(methods, seed, count=40):
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        row = list(methods); rng.shuffle(row); result.append(row)
    return result


def fixture(root):
    """Invented costs and report flags test the verifier, not hardware origin."""
    env = {"fixture": "synthetic CPU fixture; no GPU was executed"}
    source, implementation, script = "a" * 64, "b" * 64, "c" * 64
    write(root / "fused-study-source.json", {"measurement_script_sha256": script})
    write(root / "corpus.json", {"chunks": ["synthetic fixture text"] * 64})
    corpus = sha(root / "corpus.json")
    identity = {"source_sha256": source, "study_identity": implementation, "corpus_sha256": corpus, "environment": env}
    sources = {name: hashlib.sha256(value).hexdigest() for name, value in fused._SOURCES}
    tasks = [{"id": "notes", "rows": 64, "width": 384, "queries": 1, "layout": "contiguous"},
             {"id": "manuals", "rows": 64, "width": 768, "queries": 8, "layout": "column_stride2"},
             {"id": "archive", "rows": 64, "width": 1536, "queries": 17, "layout": "transpose"}]
    unseen = {"id": "unseen", "rows": 113, "width": 641, "queries": 5, "layout": "column_stride2"}
    protocol = {**identity, "schema": 1, "target": "fused_query_scores_v1", "script_sha256": script,
                "sources": sources, "source_bytes": sum(len(value) for _, value in fused._SOURCES),
                "limits": exporter.LIMITS, "corpus_chunks": 64, "tasks": tasks,
                "confirmation": {"sessions": 3, "repeats_per_session": 40, "seeds": exporter.SEEDS,
                                 "sequence_orders": exporter.SEQUENCES, "unseen_task": unseen},
                "development": {"seed": 104, "repeats": 40, "order_seed": 1207, "authored_configurations": 2,
                                "score_observations_per_round": 5, "application_adapter_observations_per_round": 10},
                "measurement_batch": 32, "application_measurement_batch": 4,
                "criterion": {"score_ci95_lower_speedup": 1.1, "application_ci95_lower_speedup": 1.03,
                              "portfolio_over_fixed_minimum_total_cost_gain": .02, "numerical_rtol": 2e-5, "numerical_atol": 2e-6},
                "baseline_noise": [{"task": task, "baseline_a_us": [10.] * 40, "baseline_b_us": [10.] * 40} for task in tasks]}
    write(root / "protocol.json", protocol)
    numerical = {**identity, "kind": "executed_fused_metal_numerical_validation", "fixture": "SYNTHETIC mocked report, not executed hardware evidence", "success": True,
                 "gpu_executed": True, "tests_run": 7, "failures": 0, "errors": 0, "skipped": 0}
    write(root / "numerical-validation.json", numerical)
    rows, rng = [], random.Random(1207)
    dev_costs = {name: [value] * 40 for name, value in zip(exporter.DEVELOPMENT_METHODS, (10., 8., 12., 2., 3.))}
    # Global fixed and routed per-shape methods deliberately have different
    # winners. Substituting a global refresh cost must not pass these fixtures.
    routed_costs = {name: [value] * 40 for name, value in zip(exporter.ROUTED_METHODS, (10., 1., 4., 5.))}
    for task in tasks:
        sequence, application_sequence = [], []
        for _ in range(40):
            row = list(exporter.DEVELOPMENT_METHODS); rng.shuffle(row); sequence.append(row)
            app_row = list(exporter.APPLICATION_ARMS); rng.shuffle(app_row); application_sequence.append(app_row)
        rows.append({"task": task, "shape": [task["queries"], task["rows"], task["width"]], "weight": 3,
                     "score_samples_us": dev_costs, "application_samples_us": dev_costs, "orders": sequence,
                     "application_orders": application_sequence,
                     "application_numerical_checks": {name: {"scores_passed": True, "ids_association_passed": True} for name in exporter.APPLICATION_ARMS},
                     "routed_application_samples_us": routed_costs, "plan_library_application_samples_us": [6.] * 40,
                     "first_score_in_development_ms": {name: 1. for name in exporter.DEVELOPMENT_METHODS},
                     "first_application_in_development_ms": {name: 2. for name in exporter.DEVELOPMENT_METHODS},
                     "first_adapter_application_in_development_ms": {name: 2. for name in exporter.APPLICATION_ARMS[5:]},
                     "numerical_check": {"passed": True, "rtol": 2e-5, "atol": 2e-6}})
    development = {**identity, "protocol_sha256": sha(root / "protocol.json"), "rows": rows,
                   "search_wall_seconds": 1., "peak_memory_bytes": 1024}
    write(root / "development.json", development)
    rules = [{"shape": row["shape"], "candidate": "score_simd", "dtype": "float32", "layout": "any_mlx_strides"} for row in rows]
    policy = {**identity, "schema": 1, "operation": "stable_l2_scores_v1", "epsilon": 1e-12, "sources": sources,
              "rules": rules, "evidence": {"numerical_passed": True, "confirmation_passed": False, "corpus_sha256": corpus,
              "protocol_sha256": sha(root / "protocol.json"), "numerical_report_sha256": sha(root / "numerical-validation.json"),
              "development_sha256": sha(root / "development.json")},
              "policy_selection": {"best_fixed": "score_simd", "joint_subset": ["score_simd"],
              "development_application_cost_us": 6., "best_one_subset_application_cost_us": 6., "best_fixed_application_cost_us": 6.,
              "development_per_shape_application_cost_us": 3.,
              "per_shape": [{"shape": row["shape"], "candidate": "fast_library"} for row in rows]}}
    write(root / "policy-frozen.json", policy)
    costs = {name: [value] * 40 for name, value in zip(exporter.METHODS, (10., 8., 12., 2., 1.5, 1.))}
    for i, seed in enumerate(exporter.SEEDS):
        record = {**identity, "session": i, "seed": seed, "sequence_order": exporter.SEQUENCES[i],
                  "protocol_sha256": sha(root / "protocol.json"), "policy_sha256": sha(root / "policy-frozen.json"),
                  "checks": [{"task": task["id"], "method": method, "scores_passed": True, "ids_association_passed": True}
                             for task in [*tasks, unseen] for method in exporter.METHODS],
                  "score_samples_us": costs, "application_samples_us": costs, "orders": orders(exporter.METHODS, seed),
                  "first_sequence_in_session_ms": {name: 2. for name in exporter.METHODS}, "artifact_load_ms": 1.,
                  "peak_memory_bytes": 2048, "fallback_counts": {"dispatcher": {"calls": 100, "fallback_calls": 20}}}
        write(root / f"confirmation-{i}.json", record)
    cold_rows, rng = [], random.Random(1667)
    for repeat in range(3):
        sequence = list(exporter.METHODS); rng.shuffle(sequence)
        for method in sequence:
            cold_rows.append({**identity, "repeat": repeat, "method": method, "policy_sha256": sha(root / "policy-frozen.json"),
                              "runtime_setup_ms": 1., "frozen_checks_ms": 1., "policy_setup_ms": 1., "input_prepare_ms": 1.,
                              "first_application_sequence_ms": 2., "process_wall_ms": 10., "peak_memory_bytes": 2048})
    write(root / "cold-study.json", {**identity, "schema": 1, "policy_sha256": sha(root / "policy-frozen.json"),
                                    "rows": cold_rows, "wall_seconds": 1.})
    return env, source, implementation


class SyntheticPlan:
    """Mock runtime installation check without importing MLX or using a GPU."""
    fail_install = False

    @classmethod
    def _trial(cls, policy):
        return None

    def __init__(self, directory):
        for name, value in fused._SOURCES:
            if (directory / name).read_bytes() != value:
                raise AssertionError("Exporter copied external kernel bytes")
        if json.loads((directory / "manifest.json").read_text())["evidence"]["confirmation_passed"] is not True:
            raise AssertionError("Confirmation flag not present in staged manifest")
        self.status = "fallback: synthetic failure" if self.fail_install else "installed"


class FusedExportTests(unittest.TestCase):
    def context(self, root):
        env, source, implementation = fixture(root)
        stack = ExitStack()
        stack.enter_context(patch.object(fused, "score_source_identity", return_value=source))
        stack.enter_context(patch.object(fused, "fused_implementation_identity", return_value=implementation))
        stack.enter_context(patch.object(fused, "__file__", str(root / "adapter.py")))
        stack.enter_context(patch.object(fused, "FusedPlan", SyntheticPlan))
        stack.enter_context(patch("kerneldarwin.workload.environment_identity", return_value=env))
        self.addCleanup(stack.close)
        return stack

    def mutate(self, root, name, operation):
        record = json.loads((root / name).read_text()); operation(record); write(root / name, record)

    def repin_policy(self, root):
        """Rebind synthetic fixture bytes so deeper guards are actually reached."""
        self.mutate(root, "policy-frozen.json", lambda p: p["evidence"].update(development_sha256=sha(root / "development.json")))
        policy_sha = sha(root / "policy-frozen.json")
        for i in range(3):
            self.mutate(root, f"confirmation-{i}.json", lambda p: p.update(policy_sha256=policy_sha))
        def cold(record):
            record["policy_sha256"] = policy_sha
            for row in record["rows"]:
                row["policy_sha256"] = policy_sha
        self.mutate(root, "cold-study.json", cold)

    def test_complete_synthetic_atomic_export_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                verified = exporter.verify_study(root)
                self.assertTrue(verified["confirmation_passed"])
                self.assertIn("per_shape", verified["per_session"][0]["application"])
                result = exporter.export_artifact(root, root / "artifact")
                self.assertTrue(result["confirmation_passed"])
                self.assertFalse(json.loads((root / "policy-frozen.json").read_text())["evidence"]["confirmation_passed"])
                self.assertTrue(json.loads((root / "artifact/manifest.json").read_text())["evidence"]["confirmation_passed"])
                self.assertEqual(set(path.name for path in (root / "artifact").iterdir()), {"manifest.json", "verification.json", *fused.SCORE_SOURCE_FILES})
                with self.assertRaises(FileExistsError):
                    exporter.export_artifact(root, root / "artifact")

    def test_one_fast_session_loss_is_not_hidden_by_pooled_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                self.mutate(root, "confirmation-2.json", lambda p: [p[key].update(fast_library=[.5] * 40) for key in ("score_samples_us", "application_samples_us")])
                # Keep descriptive pooling cheap. Every 40-pair acceptance gate
                # still uses the real estimator on these synthetic constants.
                original = exporter._interval
                def interval(a, b, seed=1404):
                    if len(a) == 40:
                        return original(a, b, seed)
                    ratio = sum(a) / sum(b)
                    return {"speedup_total_cost": ratio, "ci95": [ratio, ratio]}
                with patch.object(exporter, "_interval", side_effect=interval):
                    result = exporter.verify_study(root)
                    self.assertGreater(result["comparisons"]["application"]["dispatcher"]["fast_library"]["ci95"][0], 1.03)
                    self.assertFalse(result["confirmation_passed"])
                    with self.assertRaises(ValueError):
                        exporter.export_artifact(root, root / "artifact")
                self.assertFalse((root / "artifact").exists())

    def test_changed_or_false_pinned_evidence_refuses(self):
        cases = [("corpus.json", lambda p: p["chunks"].append("changed")),
                 ("policy-frozen.json", lambda p: p["evidence"].update(numerical_passed=False)),
                 ("numerical-validation.json", lambda p: p.update(success=False)),
                 ("development.json", lambda p: p.update(search_wall_seconds=901)),
                 ("protocol.json", lambda p: p["criterion"].update(application_ci95_lower_speedup=1.)),
                 ("cold-study.json", lambda p: p["rows"][0].update(corpus_sha256="changed"))]
        for name, operation in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, name, operation)
                    with self.assertRaises(ValueError):
                        exporter.export_artifact(root, root / "artifact")
                    self.assertFalse((root / "artifact").exists())

    def test_incomplete_malformed_or_failed_session_refuses(self):
        cases = [lambda p: p["score_samples_us"]["dispatcher"].pop(),
                 lambda p: p["application_samples_us"]["eager_library"].__setitem__(0, -1),
                 lambda p: p["score_samples_us"]["dispatcher"].__setitem__(0, True),
                 lambda p: p["checks"].pop(),
                 lambda p: p["checks"][0].update(ids_association_passed=False),
                 lambda p: p["orders"].pop(),
                 lambda p: p["fallback_counts"]["dispatcher"].update(fallback_calls=0),
                 lambda p: p.update(seed=1),
                 lambda p: p.update(peak_memory_bytes=3 * 1024 ** 3)]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "confirmation-0.json", operation)
                    with self.assertRaises(ValueError):
                        exporter.verify_study(root)

    def test_incomplete_cold_and_numerical_counts_refuse_even_if_repinned(self):
        for mutation in ("cold", "numerical"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    if mutation == "cold":
                        self.mutate(root, "cold-study.json", lambda p: p["rows"].pop())
                    else:
                        self.mutate(root, "numerical-validation.json", lambda p: p.update(tests_run=6))
                        self.mutate(root, "policy-frozen.json", lambda p: p["evidence"].update(numerical_report_sha256=sha(root / "numerical-validation.json")))
                    with self.assertRaises(ValueError):
                        exporter.verify_study(root)

    def test_failed_runtime_install_leaves_no_output_or_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root), patch.object(SyntheticPlan, "fail_install", True):
                with self.assertRaises(ValueError):
                    exporter.export_artifact(root, root / "artifact")
                self.assertFalse((root / "artifact").exists())
                self.assertEqual(list(root.glob("kd-fused-export-*")), [])

    def test_racing_empty_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = exporter._rename_new
            def race(source, target):
                target.mkdir()
                original(source, target)
            with self.context(root), patch.object(exporter, "_rename_new", side_effect=race):
                with self.assertRaises(FileExistsError):
                    exporter.export_artifact(root, root / "artifact")
                self.assertTrue((root / "artifact").is_dir())
                self.assertEqual(list((root / "artifact").iterdir()), [])
                self.assertEqual(list(root.glob("kd-fused-export-*")), [])

    def test_repinned_development_budget_still_refuses(self):
        for key, value in (("search_wall_seconds", 901), ("peak_memory_bytes", 3 * 1024 ** 3)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "development.json", lambda p: p.update({key: value}))
                    self.mutate(root, "policy-frozen.json", lambda p: p["evidence"].update(development_sha256=sha(root / "development.json")))
                    with self.assertRaisesRegex(ValueError, "over-budget"):
                        exporter.verify_study(root)

    def test_cost_validation_rejects_nonfinite_and_unbounded_samples(self):
        for bad in (float("nan"), float("inf"), 0., -1., True, 1e13):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                exporter._interval([bad] * 40, [1.] * 40)

    def test_per_shape_uses_actual_routed_cost_not_global_fixed_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                policy = json.loads((root / "policy-frozen.json").read_text())
                self.assertEqual(policy["policy_selection"]["best_fixed"], "score_simd")
                self.assertTrue(all(rule["candidate"] == "fast_library" for rule in policy["policy_selection"]["per_shape"]))
                self.assertTrue(exporter.verify_study(root)["confirmation_passed"])
                self.mutate(root, "policy-frozen.json", lambda p: [rule.update(candidate="score_simd") for rule in p["policy_selection"]["per_shape"]])
                self.repin_policy(root)
                with self.assertRaisesRegex(ValueError, "per-shape"):
                    exporter.verify_study(root)

    def test_actual_empty_plan_cost_can_choose_a_null_portfolio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                self.mutate(root, "development.json", lambda p: [row.update(plan_library_application_samples_us=[.25] * 40) for row in p["rows"]])
                def policy(record):
                    record["rules"] = []
                    record["policy_selection"].update(joint_subset=[], development_application_cost_us=.75, best_one_subset_application_cost_us=.75)
                self.mutate(root, "policy-frozen.json", policy)
                self.repin_policy(root)
                for i in range(3):
                    self.mutate(root, f"confirmation-{i}.json", lambda p: p["fallback_counts"]["dispatcher"].update(fallback_calls=100))
                self.assertFalse(exporter.verify_study(root)["confirmation_passed"])
                with self.assertRaises(ValueError):
                    exporter.export_artifact(root, root / "artifact")
                self.assertFalse((root / "artifact").exists())

    def test_added_adapter_cost_arrays_cannot_be_missing_incomplete_or_invalid(self):
        cases = [lambda p: p["rows"][0].pop("routed_application_samples_us"),
                 lambda p: p["rows"][0]["routed_application_samples_us"].pop("fast_library"),
                 lambda p: p["rows"][0]["routed_application_samples_us"]["score_simd"].pop(),
                 lambda p: p["rows"][0]["routed_application_samples_us"]["score_tile4"].__setitem__(0, 0),
                 lambda p: p["rows"][0].pop("plan_library_application_samples_us"),
                 lambda p: p["rows"][0]["plan_library_application_samples_us"].pop(),
                 lambda p: p["rows"][0]["plan_library_application_samples_us"].__setitem__(0, True),
                 lambda p: p["rows"][0]["application_orders"].pop(),
                 lambda p: p["rows"][0]["application_orders"][0].__setitem__(0, "unmeasured"),
                 lambda p: p["rows"][0]["application_numerical_checks"].pop("plan_library"),
                 lambda p: p["rows"][0]["application_numerical_checks"]["routed:fast_library"].update(ids_association_passed=False),
                 lambda p: p["rows"][0]["first_adapter_application_in_development_ms"].pop("plan_library"),
                 lambda p: p["rows"][0]["plan_library_application_samples_us"].__setitem__(0, 1e10)]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "development.json", operation)
                    self.repin_policy(root)
                    with self.assertRaises(ValueError):
                        exporter.verify_study(root)

    def test_nonfinite_added_adapter_costs_refuse_even_if_repinned(self):
        for key in ("routed_application_samples_us", "plan_library_application_samples_us"):
            for value in (float("nan"), float("inf")):
                with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with self.context(root):
                        path = root / "development.json"
                        record = json.loads(path.read_text())
                        values = record["rows"][0][key]
                        (values["fast_library"] if isinstance(values, dict) else values)[0] = value
                        # Deliberately malformed JSON numbers, which the study's
                        # allow_nan=False writer itself would never emit.
                        path.write_text(json.dumps(record))
                        self.repin_policy(root)
                        with self.assertRaises(ValueError):
                            exporter.verify_study(root)


if __name__ == "__main__":
    unittest.main()
