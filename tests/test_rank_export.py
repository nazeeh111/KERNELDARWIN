"""CPU-only synthetic evidence replay regressions, never GPU/performance evidence."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin import rank_export as export
from kerneldarwin import rank_search as rank


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orders(rank_names, app_names, rng):
    ranks, apps = [], []
    for _ in range(40):
        names = list(rank_names); rng.shuffle(names); ranks.append(names)
        names = list(app_names); rng.shuffle(names); apps.append(names)
    return ranks, apps


def comparison_rows(record, seed):
    return {method: {base: {stage: export._interval(record[key][base], record[key][method], seed)
            for stage, key in (("rank", "rank_samples_us"), ("application", "application_samples_us"))}
            for base in export.LIBRARIES} for method in export.METHODS}


def portfolio_rows(record, two_configs=False):
    comparisons = {m: export._interval(record["application_samples_us"][m],
        record["application_samples_us"]["dispatcher"], record["seed"]) for m in ("best_fixed", "per_shape")}
    record["portfolio_comparisons"] = comparisons
    record["portfolio_session_gate"] = {"over_best_fixed": comparisons["best_fixed"]["ci95"][0] > 1.02,
        "over_per_shape": comparisons["per_shape"]["ci95"][0] > 1., "two_authored_configs": two_configs}


def synthetic_telemetry():
    return {"time_utc": "2026-11-01T00:00:00Z", "load_average": [0., 0., 0.],
        "thermal": {"unavailable": "Synthetic CPU fixture, no observation"},
        "power": {"unavailable": "Synthetic CPU fixture, no observation"},
        "thermal_limit": "CPU fixture does not observe a GPU or temperature"}


def fixture(root):
    """Fabricated constants exercise guards only. Runtime/GPU identities mocked."""
    environment = {"device": "SYNTHETIC CPU TEST ONLY", "mlx": "not executed"}
    adapter = b"# Synthetic CPU-only fixture, never executable GPU evidence\n"
    (root / "adapter.py").write_bytes(adapter)
    source = hashlib.sha256(adapter + b"".join(n.encode() + v for n, v in rank._SOURCES)).hexdigest()
    implementation, script = "b" * 64, "c" * 64
    identity = {"source_sha256": source, "study_identity": implementation, "environment": environment}
    metal = root / "metal"; metal.mkdir()
    for name, data in rank._SOURCES:
        (metal / name).write_bytes(data)
    write(root / "rank-study-source.json", {"measurement_script_sha256": script})
    write(root / "corpus.json", {"chunks": ["synthetic licensed corpus " + str(i) for i in range(161)]})
    corpus = sha(root / "corpus.json")
    binding = {**identity, "corpus_sha256": corpus}
    flags = dict.fromkeys(export.DEBUG_FLAGS)
    write(root / "numerical-validation.json", {**identity, "kind": "executed_rank_metal_numerical_validation",
        "success": True, "gpu_executed": True, "tests_run": 12, "failures": 0, "errors": 0, "skipped": 0,
        "SYNTHETIC_FIXTURE": "This is fabricated CPU test data, not executed GPU evidence"})
    numerical = sha(root / "numerical-validation.json")
    tasks = [{"id": "notes", "rows": 64, "width": 384, "queries": 1, "layout": "contiguous"},
             {"id": "manuals", "rows": 161, "width": 768, "queries": 8, "layout": "column_stride2"},
             {"id": "archive", "rows": 161, "width": 1536, "queries": 17, "layout": "transpose"}]
    unseen = {"id": "unseen", "rows": 97, "width": 1025, "queries": 7, "layout": "column_stride2"}
    noise, rng = [], random.Random(21441)
    for task in tasks:
        noise_orders = []
        for _ in range(40):
            row = [0, 1]; rng.shuffle(row); noise_orders.append(row)
        noise.append({"task": task, "baseline_a_us": [10.] * 40, "baseline_b_us": [10.] * 40,
                      "comparison": export._interval([10.] * 40, [10.] * 40), "orders": noise_orders, "first_rank_ms": 1.})
    app_orders, app_rng = [], random.Random(21443)
    for _ in range(40):
        order = [0, 1]; app_rng.shuffle(order); app_orders.append(order)
    app_noise = {"baseline_a_us": [10.] * 40, "baseline_b_us": [10.] * 40, "orders": app_orders,
        "comparison": export._interval([10.] * 40, [10.] * 40), "first_application_sequence_ms": 1.,
        "input_seeds": list(range(21101, 21141)),
        "sequence_orders": [[[0, 1, 2], [1, 2, 0], [2, 0, 1]][r % 3] for r in range(40)],
        "method": "stable_library", "sequences_per_block": 16}
    protocol = {**binding, "schema": 2, "measurement_design": "application_primary_symmetric_noise_v2",
        "uncertainty": {"resamples": 4000, "circular_block_lengths": [4, 8], "block_seed_stride": 100003,
                        "primary_interval": "envelope_iid_block4_block8"},
        "calibration": {"rank_order_seed": 21441, "application_order_seed": 21443,
            "application_input_seeds": list(range(21101, 21141)), "application_sequence_orders": app_noise["sequence_orders"],
            "negative_control_method": "stable_library", "application_first_warmup_order": [0, 1, 2], "shared_method_instance": True},
        "calibration_wall_seconds": 2., "application_baseline_noise": app_noise,
        "telemetry_before": synthetic_telemetry(), "telemetry": synthetic_telemetry(),
        "target": "stable_top5_pairs_v1", "script_sha256": script,
        "limits": export.LIMITS, "sources": {n: hashlib.sha256(v).hexdigest() for n, v in rank._SOURCES},
        "source_bytes": sum(len(v) for _, v in rank._SOURCES), "corpus_chunks": 161,
        "tasks": tasks, "confirmation": {"sessions": 3, "repeats_per_session": 40, "seeds": [23101, 24203, 25307],
            "sequence_orders": [[1, 0, 3, 2], [2, 3, 0, 1], [0, 2, 1, 3]], "unseen_task": unseen,
            "minimum_separation_seconds": 120},
        "development": {"seed": 21104, "repeats": 40, "order_seed": 21207, "authored_configurations": 2,
                        "rank_observations_per_round": 9, "application_adapter_observations_per_round": 16},
        "measurement_batch": 128, "application_measurement_batch": 16,
        "methods": list(export.METHODS), "libraries": list(export.LIBRARIES),
        "canonical_rank_libraries": list(export.CANONICAL_RANK_LIBRARIES),
        "global_names": list(export.GLOBAL_NAMES), "routed_choices": list(export.ROUTED_CHOICES),
        "rank_arms": list(export.RANK_ARMS), "application_arms": list(export.APP_ARMS),
        "application": "One index refresh plus three executed changed-query top5 searches per task",
        "weights": "Equal task frequency; sum three rank costs per task and complete application costs",
        "qos": "user_initiated_current_thread_only", "source_budget_scope": "All three authored MSL files including shared header",
        "baseline_noise": noise, "criterion": {"rank_ci95_lower_speedup": 1.1, "application_ci95_lower_speedup": 1.03,
            "portfolio_over_fixed_minimum_total_cost_gain": .02, "numerical_rtol": 2e-5, "numerical_atol": 2e-6},
        "numerical_report_sha256": numerical, "peak_memory_bytes": 2048, "timing_validation_flags": flags}
    write(root / "protocol.json", protocol)
    rng, rows = random.Random(21207), []
    for task in tasks:
        rank_orders, app_orders = orders(export.RANK_ARMS, export.APP_ARMS, rng)
        rank_costs = {n: [10.] * 40 for n in export.RANK_ARMS}
        rank_costs.update(simd32=[1.] * 40, group128=[2.] * 40,
                          **{"plan:simd32": [2.] * 40, "plan:group128": [3.] * 40})
        app_costs = {n: [10.] * 40 for n in export.APP_ARMS}
        app_costs.update({"global:simd32": [1.] * 40, "global:group128": [2.] * 40,
                          "routed:fast_library": [2.] * 40, "routed:simd32": [4.] * 40,
                          "routed:group128": [5.] * 40, "plan:simd32": [3.] * 40,
                          "plan:group128": [4.] * 40, "plan_library": [8.] * 40})
        row = {"task": task, "shape": [task["queries"], task["rows"], task["width"], 5], "weight": 1,
            "rank_samples_us": rank_costs, "application_samples_us": app_costs,
            "rank_orders": rank_orders, "application_orders": app_orders,
            "rank_numerical_checks": {n: {"ids_passed": True, "original_bits_passed": True} for n in export.RANK_ARMS},
            "application_numerical_checks": {n: {"scores_passed": True, "ids_association_passed": True} for n in export.APP_ARMS},
            "first_rank_in_development_ms": {n: 1. for n in export.RANK_ARMS},
            "first_application_in_development_ms": {n: 1. for n in export.APP_ARMS}}
        row["comparisons"] = {n: {b: {"rank": export._interval(rank_costs[b], rank_costs["plan:" + n])}
                                   for b in export.CANONICAL_RANK_LIBRARIES} for n in export.CANDIDATES}
        rows.append(row)
    write(root / "development.json", {**binding, "schema": 1, "protocol_sha256": sha(root / "protocol.json"),
        "numerical_report_sha256": numerical, "rows": rows, "search_wall_seconds": 20.,
        "peak_memory_bytes": 2048, "timing_validation_flags": flags})
    policy = {**identity, "schema": 1, "operation": "stable_top5_pairs_v1", "epsilon": 1e-12, "fixed_k": 5,
        "key_policy": "native_ftz_stable_index_original_bits_v1", "compile_options": {"math_mode": "safe"},
        "sources": protocol["sources"],
        "rules": [{"shape": r["shape"], "candidate": "simd32", "dtype": "float32", "layout": "any_mlx_strides"} for r in rows],
        "evidence": {"numerical_passed": True, "confirmation_passed": False, "corpus_sha256": corpus,
                     "protocol_sha256": sha(root / "protocol.json"), "development_sha256": sha(root / "development.json"),
                     "numerical_report_sha256": numerical},
        "policy_selection": {"best_fixed": "simd32", "joint_subset": ["simd32"], "development_application_cost_us": 9.,
            "best_one_subset_application_cost_us": 9., "best_fixed_application_cost_us": 3.,
            "development_per_shape_application_cost_us": 6.,
            "per_shape": [{"shape": r["shape"], "candidate": "fast_library"} for r in rows]}}
    write(root / "policy-frozen.json", policy)
    policy_sha = sha(root / "policy-frozen.json")
    fallback_counts = {m: {"calls": 20488, "fallback_calls": 20488} for m in export.METHODS}
    fallback_counts.update(best_fixed={"calls": 28192, "fallback_calls": 0},
                           per_shape={"calls": 28192, "fallback_calls": 28192},
                           dispatcher={"calls": 28192, "fallback_calls": 7048})
    for i, seed in enumerate(export.SEEDS):
        rank_orders, app_orders = orders(export.METHODS, export.METHODS, random.Random(seed))
        record = {**binding, "schema": 1, "session": i, "seed": seed, "sequence_order": export.SEQUENCES[i],
            "session_start_unix_ns": 1_700_000_000_000_000_000 + i * 130_000_000_000,
            "session_end_unix_ns": 1_700_000_001_000_000_000 + i * 130_000_000_000,
            "session_start_monotonic_ns": 1_000_000_000_000 + i * 130_000_000_000,
            "session_end_monotonic_ns": 1_001_000_000_000 + i * 130_000_000_000,
            "process_id": 10000 + i, "telemetry_before": synthetic_telemetry(), "telemetry_after": synthetic_telemetry(),
            "protocol_sha256": sha(root / "protocol.json"), "policy_sha256": policy_sha,
            "development_sha256": sha(root / "development.json"), "numerical_report_sha256": numerical,
            "rank_samples_us": {m: [10. if m in export.LIBRARIES else 1.] * 40 for m in export.METHODS},
            "application_samples_us": {m: [10. if m in export.LIBRARIES else 1.] * 40 for m in export.METHODS},
            "rank_orders": rank_orders, "application_orders": app_orders,
            "checks": [{"task": t["id"], "method": m, "rank_ids_passed": True, "rank_original_bits_passed": True,
                        "scores_passed": True, "ids_association_passed": True} for t in [*tasks, unseen] for m in export.METHODS],
            "first_rank_in_session_ms": {m: [1.] * 4 for m in export.METHODS},
            "first_application_in_session_ms": {m: [1.] * 4 for m in export.METHODS},
            "artifact_load_ms": 1., "peak_memory_bytes": 2048, "timing_validation_flags": flags,
            "fallback_counts": fallback_counts, "session_gate": {"rank": True, "application": True}}
        record["comparisons"] = comparison_rows(record, seed)
        portfolio_rows(record)
        write(root / f"confirmation-{i}.json", record)
    rng, cold_orders, cold_rows = random.Random(21667), [], []
    for repeat in range(3):
        names = list(export.METHODS); rng.shuffle(names); cold_orders.append(names)
        for method in names:
            cold_rows.append({**binding, "repeat": repeat, "method": method, "policy_sha256": policy_sha,
                "runtime_setup_ms": 1., "frozen_checks_ms": 1., "policy_setup_ms": 1., "input_prepare_ms": 1.,
                "first_application_sequence_ms": 1., "process_wall_ms": 10., "peak_memory_bytes": 2048,
                "timing_validation_flags": flags})
    write(root / "cold-study.json", {**binding, "schema": 1, "policy_sha256": policy_sha,
                                    "rows": cold_rows, "orders": cold_orders, "wall_seconds": 1.})
    return environment, source, implementation


class SyntheticPlan:
    """Data-only install stand-in. No MLX, Metal, shader or timing execution."""
    fail_install = False

    @classmethod
    def _trial(cls, policy):
        return None

    def __init__(self, staging):
        self.status = "fallback: synthetic" if self.fail_install else "installed"
        assert json.loads((staging / "manifest.json").read_text())["evidence"]["confirmation_passed"] is True
        for name, content in rank._SOURCES:
            assert (staging / name).read_bytes() == content


class RankExportTests(unittest.TestCase):
    def test_checked_in_measurement_pin_matches_actual_script_bytes(self):
        """Plain file hash check. Does not import or execute the study script."""
        repository = Path(__file__).resolve().parents[1]
        pin = json.loads((Path(rank.__file__).parent / "rank-study-source.json").read_text())
        self.assertEqual(pin["measurement_script_sha256"], sha(repository / "scripts/rank_study.py"))

    def context(self, root):
        env, source, implementation = fixture(root)
        stack = ExitStack()
        stack.enter_context(patch.object(rank, "rank_source_identity", return_value=source))
        stack.enter_context(patch.object(rank, "rank_implementation_identity", return_value=implementation))
        stack.enter_context(patch.object(rank, "__file__", str(root / "adapter.py")))
        stack.enter_context(patch.object(rank, "RankPlan", SyntheticPlan))
        stack.enter_context(patch("kerneldarwin.workload.environment_identity", return_value=env))
        self.addCleanup(stack.close)
        return stack

    def mutate(self, root, name, operation):
        record = json.loads((root / name).read_text()); operation(record); write(root / name, record)

    def repin(self, root):
        """Rebind fabricated bytes to exercise semantic checks beyond hashes."""
        numerical, protocol = [sha(root / n) for n in ("numerical-validation.json", "protocol.json")]
        self.mutate(root, "development.json", lambda p: p.update(protocol_sha256=protocol, numerical_report_sha256=numerical))
        development = sha(root / "development.json")
        self.mutate(root, "policy-frozen.json", lambda p: p["evidence"].update(numerical_report_sha256=numerical,
                    protocol_sha256=protocol, development_sha256=development))
        policy = sha(root / "policy-frozen.json")
        for i in range(3):
            self.mutate(root, f"confirmation-{i}.json", lambda p: p.update(policy_sha256=policy,
                protocol_sha256=protocol, development_sha256=development, numerical_report_sha256=numerical))
        def cold(p):
            p["policy_sha256"] = policy
            for r in p["rows"]:
                r["policy_sha256"] = policy
        self.mutate(root, "cold-study.json", cold)

    def test_complete_atomic_export_preserves_policy_and_exact_msl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                verified = export.verify_study(root)
                self.assertTrue(all(verified[k] for k in ("numerical_passed", "rank_passed", "application_passed", "resource_passed", "confirmation_passed")))
                self.assertTrue(verified["break_even"]["estimate_conditions_passed"])
                self.assertFalse(verified["break_even"]["reliable_payback"])
                self.assertFalse(verified["break_even"]["guaranteed"])
                export.export_artifact(root, root / "artifact")
                self.assertFalse(json.loads((root / "policy-frozen.json").read_text())["evidence"]["confirmation_passed"])
                self.assertEqual({p.name for p in (root / "artifact").iterdir()}, {"manifest.json", "verification.json", *export.SOURCE_FILES})
                with self.assertRaises(FileExistsError):
                    export.export_artifact(root, root / "artifact")

    def test_weak_or_changed_protocol_rejected_even_if_repinned(self):
        cases = [lambda p: p["criterion"].update(rank_ci95_lower_speedup=1.),
                 lambda p: p["criterion"].update(application_ci95_lower_speedup=1.),
                 lambda p: p["limits"].update(portfolio_kernels=3),
                 lambda p: p["development"].update(repeats=True),
                 lambda p: p.update(measurement_batch=127),
                 lambda p: p.update(application_measurement_batch=8),
                 lambda p: p["confirmation"]["unseen_task"].update(width=1024),
                 lambda p: p.update(rank_arms=list(export.GLOBAL_NAMES)),
                 lambda p: p.update(peak_memory_bytes=True),
                 lambda p: p["timing_validation_flags"].update(MTL_DEBUG_LAYER="1")]
        cases.extend([lambda p: p.update(schema=1), lambda p: p.update(measurement_design="old_rank_only"),
            lambda p: p["uncertainty"].update(circular_block_lengths=[4]),
            lambda p: p["uncertainty"].update(resamples=3999),
            lambda p: p["uncertainty"].update(block_seed_stride=1),
            lambda p: p["uncertainty"].update(primary_interval="iid"),
            lambda p: p["calibration"].update(shared_method_instance=False),
            lambda p: p["calibration"]["application_input_seeds"].__setitem__(0, 23101),
            lambda p: p["calibration"].update(application_first_warmup_order=[2, 1, 0]),
            lambda p: p["calibration"].update(extra_control=True),
            lambda p: p.update(freeze_wall_seconds_ignored=True),
            lambda p: p.update(calibration_wall_seconds=121),
            lambda p: p.update(calibration_wall_seconds=.01),
            lambda p: p["confirmation"].update(minimum_separation_seconds=119),
            lambda p: p.pop("telemetry_before"),
            lambda p: p["telemetry"].pop("power")])
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "protocol.json", operation); self.repin(root)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)

    def test_prospective_noise_all_three_and_order_are_mandatory(self):
        cases = [lambda p: p["baseline_noise"].pop(),
                 lambda p: p["baseline_noise"][2]["baseline_b_us"].__setitem__(0, .01),
                 lambda p: p["baseline_noise"][1].update(baseline_b_us=[5.] * 40),
                 lambda p: p["baseline_noise"][0]["orders"][0].reverse()]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "protocol.json", operation); self.repin(root)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)

    def test_v2_block_bootstrap_golden_case_and_input_counts(self):
        # Fixed known CPU calculation makes the widening visible: IID passes
        # 1.03 while a contiguous run of slow observations fails block8.
        # Exactly representable input sums keep this oracle independent of
        # Python-version differences in floating-point summation.
        interval = export._interval([18.] * 20 + [8.] * 20, [10.] * 40, 23101)
        self.assertEqual(interval, {"speedup_total_cost": 1.3, "ci95": [.925, 1.65],
            "baseline_total_us": 520., "candidate_total_us": 400., "iid_ci95": [1.15, 1.45],
            "block4_ci95": [1.025, 1.6], "block8_ci95": [.925, 1.65]})
        for length in (39, 41, 44, 1001):
            with self.subTest(length=length), self.assertRaises(ValueError):
                export._interval([1.] * length, [1.] * length)
        with self.assertRaises(ValueError):
            export._interval([1.] * 40, [1.] * 40, True)

    def test_rank_entry_cap_relaxed_but_symmetric_squared_threshold_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                def controls(p):
                    row = p["baseline_noise"][0]
                    row["baseline_b_us"] = [5.] * 40
                    row["comparison"] = export._interval(row["baseline_a_us"], row["baseline_b_us"])
                    p["criterion"]["rank_ci95_lower_speedup"] = 4.
                self.mutate(root, "protocol.json", controls); self.repin(root)
                self.assertTrue(export.verify_study(root)["confirmation_passed"])
                self.mutate(root, "protocol.json", lambda p: p["criterion"].update(rank_ci95_lower_speedup=3.))
                self.repin(root)
                with self.assertRaisesRegex(ValueError, "criterion"):
                    export.verify_study(root)

    def test_application_threshold_comes_from_symmetric_noise_in_all_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                def controls(p):
                    row = p["application_baseline_noise"]
                    row["baseline_a_us"] = [10.5] * 40
                    row["comparison"] = export._interval(row["baseline_a_us"], row["baseline_b_us"])
                    p["criterion"]["application_ci95_lower_speedup"] = 1.05 ** 2
                self.mutate(root, "protocol.json", controls); self.repin(root)
                for i in range(3):
                    def session(p):
                        for base in export.LIBRARIES:
                            p["application_samples_us"][base] = [1.05] * 40
                        p["comparisons"] = comparison_rows(p, p["seed"])
                        p["session_gate"]["application"] = False
                    self.mutate(root, f"confirmation-{i}.json", session)
                result = export.verify_study(root)
                self.assertFalse(result["application_passed"])
                self.assertFalse(result["confirmation_passed"])
                self.assertTrue(all(s["application"]["dispatcher"]["stable_library"]["ci95"][0] > 1.03 for s in result["per_session"]))

    def test_repinned_application_control_iid_pass_does_not_hide_block_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                def controls(p):
                    row = p["application_baseline_noise"]
                    row["baseline_a_us"], row["baseline_b_us"] = [1.2] * 20 + [.8] * 20, [1.] * 40
                    row["comparison"] = export._interval(row["baseline_a_us"], row["baseline_b_us"])
                    interval = row["comparison"]
                    self.assertGreaterEqual(interval["iid_ci95"][0], .9)
                    self.assertLessEqual(interval["iid_ci95"][1], 1.1)
                    self.assertLess(interval["ci95"][0], .9)
                    p["criterion"]["application_ci95_lower_speedup"] = max(1., interval["ci95"][1], 1 / interval["ci95"][0]) ** 2
                self.mutate(root, "protocol.json", controls); self.repin(root)
                with self.assertRaisesRegex(ValueError, "noise stability"):
                    export.verify_study(root)

    def test_application_control_exact_calibration_fields_and_families(self):
        cases = [lambda a: a["orders"][0].reverse(), lambda a: a["input_seeds"].__setitem__(0, 21100),
            lambda a: a["sequence_orders"][1].reverse(), lambda a: a.update(method="fast_library"),
            lambda a: a.update(sequences_per_block=8), lambda a: a.update(first_application_sequence_ms=True),
            lambda a: a["baseline_a_us"].pop(), lambda a: a["comparison"].pop("block8_ci95"),
            lambda a: a["comparison"].update(block4_ci95=[1., 2.]), lambda a: a.update(extra_control=True)]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "protocol.json", lambda p: operation(p["application_baseline_noise"]))
                    self.repin(root)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)

    def test_repinned_session_iid_win_with_block_loss_is_complete_losing_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                def session(p):
                    p["application_samples_us"]["stable_library"] = [1.8] * 20 + [.8] * 20
                    p["comparisons"] = comparison_rows(p, p["seed"])
                    p["session_gate"]["application"] = False
                    self.assertGreater(p["comparisons"]["dispatcher"]["stable_library"]["application"]["iid_ci95"][0], 1.03)
                self.mutate(root, "confirmation-0.json", session)
                result = export.verify_study(root)
                self.assertFalse(result["application_passed"])
                self.assertFalse(result["confirmation_passed"])
                self.assertIsNone(result["break_even"]["full_four_task_sequences"])
                # Forging the primary interval from the narrower IID family is
                # rejected even with all other fields left internally intact.
                self.mutate(root, "confirmation-0.json", lambda p: p["comparisons"]["dispatcher"]["stable_library"]["application"].update(
                    ci95=p["comparisons"]["dispatcher"]["stable_library"]["application"]["iid_ci95"]))
                with self.assertRaisesRegex(ValueError, "raw costs"):
                    export.verify_study(root)

    def test_development_rejects_malformed_arrays_checks_budgets_and_global_cost_substitution(self):
        cases = [lambda p: p.update(search_wall_seconds=901),
                 lambda p: p.update(search_wall_seconds=.1),
                 lambda p: p.update(peak_memory_bytes=3 * 1024 ** 3),
                 lambda p: p["environment"].update(device="changed"),
                 lambda p: p["rows"][0]["rank_samples_us"]["plan:simd32"].pop(),
                 lambda p: p["rows"][0]["application_samples_us"].pop("plan_library"),
                 lambda p: p["rows"][0]["application_samples_us"]["routed:fast_library"].__setitem__(0, True),
                 lambda p: p["rows"][0]["rank_numerical_checks"]["plan:simd32"].update(ids_passed=False),
                 lambda p: p["rows"][0]["application_numerical_checks"]["plan_library"].update(ids_association_passed=False),
                 lambda p: p["rows"][0]["first_application_in_development_ms"].pop("plan:group128"),
                 lambda p: p["rows"][0]["application_orders"].pop(),
                 lambda p: p["rows"][0]["rank_orders"][0].reverse(),
                 lambda p: p["rows"][0]["shape"].__setitem__(3, True),
                 lambda p: p["rows"][0]["application_samples_us"].update(**{"plan:simd32": [1.] * 40}),
                 lambda p: p["rows"][0]["application_samples_us"].update(**{"routed:simd32": [.5] * 40}),
                 lambda p: p["rows"][0]["rank_samples_us"].update(**{"plan:simd32": [20.] * 40})]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "development.json", operation); self.repin(root)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)

    def test_actual_plan_rank_cost_determines_eligibility_after_comparisons_repinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                def change(p):
                    for row in p["rows"]:
                        row["rank_samples_us"]["plan:simd32"] = [20.] * 40
                        row["comparisons"]["simd32"] = {b: {"rank": export._interval([10.] * 40, [20.] * 40)} for b in export.CANONICAL_RANK_LIBRARIES}
                self.mutate(root, "development.json", change); self.repin(root)
                with self.assertRaisesRegex(ValueError, "portfolio"):
                    export.verify_study(root)

    def test_two_kernel_portfolio_requires_two_percent_gain(self):
        for archive_cost, two_allowed in ((2.9, False), (2., True)):
            with self.subTest(archive_cost=archive_cost), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "development.json", lambda p: p["rows"][2]["application_samples_us"].update(
                        **{"plan:group128": [archive_cost] * 40}))
                    self.repin(root)
                    if not two_allowed:
                        # An apparent 1.1% two-kernel gain still selects one.
                        self.assertTrue(export.verify_study(root)["confirmation_passed"])
                    def two(p):
                        p["rules"][2]["candidate"] = "group128"
                        p["policy_selection"].update(joint_subset=["group128", "simd32"],
                            development_application_cost_us=6. + archive_cost)
                    self.mutate(root, "policy-frozen.json", two); self.repin(root)
                    if two_allowed:
                        for i in range(3):
                            self.mutate(root, f"confirmation-{i}.json", lambda p: portfolio_rows(p, True))
                        result = export.verify_study(root)
                        self.assertTrue(result["confirmation_passed"])
                        self.assertTrue(result["portfolio"]["two_authored_configs"])
                        self.assertFalse(result["portfolio"]["two_authored_kernel_advantage_over_fixed"])
                    else:
                        with self.assertRaisesRegex(ValueError, "portfolio"):
                            export.verify_study(root)

    def test_measured_empty_plan_can_select_null_study_but_cannot_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                self.mutate(root, "development.json", lambda p: [r["application_samples_us"].update(
                    plan_library=[.25] * 40) for r in p["rows"]])
                def empty(p):
                    p["rules"] = []
                    p["policy_selection"].update(joint_subset=[], development_application_cost_us=.75,
                        best_one_subset_application_cost_us=.75)
                self.mutate(root, "policy-frozen.json", empty); self.repin(root)
                for i in range(3):
                    self.mutate(root, f"confirmation-{i}.json", lambda p: p["fallback_counts"]["dispatcher"].update(fallback_calls=28192))
                result = export.verify_study(root)
                self.assertFalse(result["confirmation_passed"])
                self.assertFalse(result["break_even"]["reliable_payback"])
                with self.assertRaises(ValueError):
                    export.export_artifact(root, root / "artifact")

    def test_rank_gate_requires_both_canonical_algorithms_in_every_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                def loss(p):
                    p["rank_samples_us"]["stable_reduction"] = [.5] * 40
                    p["comparisons"] = comparison_rows(p, p["seed"])
                    p["session_gate"]["rank"] = False
                self.mutate(root, "confirmation-2.json", loss)
                result = export.verify_study(root)
                self.assertFalse(result["rank_passed"])
                self.assertTrue(result["application_passed"])
                self.assertFalse(result["confirmation_passed"])

    def test_every_session_library_gate_matters_and_pooled_gain_cannot_rescue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                def loss(p):
                    p["application_samples_us"]["fast_reduction"] = [.5] * 40
                    p["comparisons"] = comparison_rows(p, p["seed"])
                    p["session_gate"]["application"] = False
                self.mutate(root, "confirmation-2.json", loss)
                result = export.verify_study(root)
                self.assertFalse(result["confirmation_passed"])
                self.assertFalse(result["application_passed"])
                self.assertTrue(result["rank_passed"])
                self.assertGreater(result["comparisons"]["application"]["dispatcher"]["fast_reduction"]["ci95"][0], 1.03)
                self.assertFalse(result["break_even"]["reliable_payback"])
                self.assertIsNone(result["break_even"]["full_four_task_sequences"])
                with self.assertRaises(ValueError):
                    export.export_artifact(root, root / "artifact")
                self.assertFalse((root / "artifact").exists())

    def test_confirmation_requires_rank_ids_bits_complete_samples_orders_counts_and_bindings(self):
        cases = [lambda p: p["checks"][0].pop("rank_ids_passed"),
                 lambda p: p["checks"][0].update(rank_original_bits_passed=False),
                 lambda p: p["checks"].pop(),
                 lambda p: p["rank_samples_us"]["dispatcher"].pop(),
                 lambda p: p["application_samples_us"]["fast_reduction"].__setitem__(0, 0),
                 lambda p: p["rank_orders"].pop(),
                 lambda p: p["application_orders"][0].reverse(),
                 lambda p: p["fallback_counts"]["dispatcher"].update(fallback_calls=0),
                 lambda p: p["fallback_counts"]["dispatcher"].update(calls=True),
                 lambda p: p["fallback_counts"]["per_shape"].update(fallback_calls=0),
                 lambda p: p["first_rank_in_session_ms"]["dispatcher"].pop(),
                 lambda p: p.update(seed=23102),
                 lambda p: p.update(development_sha256="stale"),
                 lambda p: p.update(peak_memory_bytes=3 * 1024 ** 3)]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "confirmation-0.json", operation)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)

    def test_confirmation_requires_positive_two_clock_spans_gaps_and_pid(self):
        cases = [lambda p: p.pop("session_start_unix_ns"),
            lambda p: p.update(session_start_unix_ns=True), lambda p: p.update(session_end_unix_ns=1),
            lambda p: p.update(session_end_monotonic_ns=p["session_start_monotonic_ns"]),
            lambda p: p.update(session_start_unix_ns=1_700_000_120_000_000_000),
            lambda p: p.update(session_start_monotonic_ns=1_120_000_000_000),
            lambda p: p.update(process_id=0), lambda p: p.update(process_id=True),
            lambda p: p.update(process_id=1.5), lambda p: p.pop("telemetry_after"),
            lambda p: p["telemetry_before"].pop("thermal")]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "confirmation-1.json", operation)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                self.mutate(root, "confirmation-1.json", lambda p: p.update(
                    session_start_unix_ns=1_700_000_121_000_000_000, session_start_monotonic_ns=1_121_000_000_000))
                result = export.verify_study(root)
                self.assertTrue(result["confirmation_passed"])
                self.assertEqual(result["sessions"][1]["unix_gap_seconds"], 120.)
                self.assertEqual(result["sessions"][1]["monotonic_gap_seconds"], 120.)

    def test_untouched_portfolio_flags_are_separate_and_require_all_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                result = export.verify_study(root)
                self.assertTrue(result["confirmation_passed"])
                self.assertFalse(result["portfolio"]["over_best_fixed"])
                for i in range(3):
                    def session(p):
                        p["application_samples_us"].update(best_fixed=[2.] * 40, per_shape=[1.1] * 40)
                        p["comparisons"] = comparison_rows(p, p["seed"])
                        portfolio_rows(p)
                    self.mutate(root, f"confirmation-{i}.json", session)
                result = export.verify_study(root)
                self.assertTrue(result["portfolio"]["over_best_fixed"])
                self.assertTrue(result["portfolio"]["over_per_shape"])
                self.assertFalse(result["portfolio"]["two_authored_kernel_advantage_over_fixed"])
                def loss(p):
                    p["application_samples_us"]["best_fixed"] = [1.01] * 40
                    p["comparisons"] = comparison_rows(p, p["seed"])
                    portfolio_rows(p)
                self.mutate(root, "confirmation-2.json", loss)
                result = export.verify_study(root)
                self.assertTrue(result["confirmation_passed"])
                self.assertFalse(result["portfolio"]["over_best_fixed"])
                self.assertTrue(result["portfolio"]["over_per_shape"])
                self.mutate(root, "confirmation-2.json", lambda p: p["portfolio_session_gate"].update(over_best_fixed=True))
                with self.assertRaisesRegex(ValueError, "portfolio flags"):
                    export.verify_study(root)

    def test_numerical_exact_kind_counts_and_current_identities(self):
        cases = [lambda p: p.update(tests_run=True), lambda p: p.update(tests_run=11),
                 lambda p: p.update(errors=True), lambda p: p.update(skipped=1),
                 lambda p: p.update(kind="executed_fused_metal_numerical_validation"),
                 lambda p: p.update(source_sha256="stale"), lambda p: p.update(gpu_executed=False),
                 lambda p: (p.pop("tests_run"), p.update(methods_run=100))]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "numerical-validation.json", operation)
                    self.mutate(root, "protocol.json", lambda p: p.update(numerical_report_sha256=sha(root / "numerical-validation.json")))
                    self.mutate(root, "development.json", lambda p: p.update(protocol_sha256=sha(root / "protocol.json"), numerical_report_sha256=sha(root / "numerical-validation.json")))
                    self.repin(root)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)

    def test_cold_24_exact_order_components_and_budgets(self):
        cases = [lambda p: p["rows"].pop(), lambda p: p["rows"].reverse(),
                 lambda p: p["orders"][0].reverse(), lambda p: p.update(wall_seconds=181),
                 lambda p: p.update(wall_seconds=.1), lambda p: p["rows"][0].update(process_wall_ms=1),
                 lambda p: p["rows"][0].update(runtime_setup_ms=False),
                 lambda p: p["rows"][0].update(corpus_sha256="stale"),
                 lambda p: p["rows"][0]["environment"].update(device="changed"),
                 lambda p: p["rows"][0]["timing_validation_flags"].update(MTL_SHADER_VALIDATION="1")]
        for operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.context(root):
                    self.mutate(root, "cold-study.json", operation)
                    with self.assertRaises(ValueError):
                        export.verify_study(root)

    def test_source_snapshot_mismatch_and_bounds_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                (root / "metal" / export.SOURCE_FILES[0]).write_text("tampered")
                with self.assertRaises(ValueError):
                    export.verify_study(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                (root / "adapter.py").write_text("# hot replacement")
                with self.assertRaisesRegex(ValueError, "adapter source"):
                    export.verify_study(root)
        for maximum in (12 * 1024, 20_000_000):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "large"
                with path.open("wb") as handle:
                    handle.truncate(maximum + 1)
                with patch.object(Path, "open", side_effect=AssertionError("must stat before read")):
                    with self.assertRaises(ValueError):
                        export._bytes(path, maximum)

    def test_cost_validation_rejects_nonfinite_bool_zero_unbounded(self):
        for value in (float("nan"), float("inf"), True, 0, -1, 1e13):
            with self.subTest(value=value), self.assertRaises(ValueError):
                export._interval([value] * 40, [1.] * 40)

    def test_duplicate_json_keys_and_nan_rejected(self):
        for value in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"metadata": {"value": 1e999}}', '{"a": -1e999}'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "data.json"; path.write_text(value)
                with self.assertRaises(ValueError):
                    export._read(path)

    def test_failed_runtime_guard_cleans_only_own_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root), patch.object(SyntheticPlan, "fail_install", True):
                sentinel = root / "keep"; sentinel.write_text("preserve")
                with self.assertRaises(ValueError):
                    export.export_artifact(root, root / "artifact")
                self.assertFalse((root / "artifact").exists())
                self.assertEqual(list(root.glob("kd-rank-export-*")), [])
                self.assertEqual(sentinel.read_text(), "preserve")

    def test_racing_empty_destination_and_existing_symlink_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                original = export._rename_new
                def race(source, target):
                    target.mkdir(); original(source, target)
                with patch.object(export, "_rename_new", side_effect=race), self.assertRaises(FileExistsError):
                    export.export_artifact(root, root / "artifact")
                self.assertEqual(list((root / "artifact").iterdir()), [])
                self.assertEqual(list(root.glob("kd-rank-export-*")), [])
                link = root / "broken"; link.symlink_to(root / "missing")
                with self.assertRaises(FileExistsError):
                    export.export_artifact(root, link)
                self.assertTrue(link.is_symlink())

    def test_evidence_changes_after_replay_refuses_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                original = export.verify_study
                def race(study):
                    result = original(study)
                    self.mutate(root, "corpus.json", lambda p: p["chunks"].append("changed"))
                    return result
                with patch.object(export, "verify_study", side_effect=race), self.assertRaises(ValueError):
                    export.export_artifact(root, root / "artifact")
                self.assertFalse((root / "artifact").exists())

    def test_policy_read_is_hashed_before_parsing_against_revert_race(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.context(root):
                verified = export.verify_study(root)
                bad = json.loads((root / "policy-frozen.json").read_text())
                bad["rules"][0]["candidate"] = "group128"
                bad_bytes = json.dumps(bad).encode()
                original = export._bytes
                seen = False
                def race(path, maximum=20_000_000):
                    nonlocal seen
                    if path.name == "policy-frozen.json" and not seen:
                        seen = True
                        return bad_bytes  # Subsequent reads see the original again.
                    return original(path, maximum)
                with patch.object(export, "verify_study", return_value=verified), patch.object(export, "_bytes", side_effect=race):
                    with self.assertRaisesRegex(ValueError, "Policy changed"):
                        export.export_artifact(root, root / "artifact")
                self.assertFalse((root / "artifact").exists())


if __name__ == "__main__":
    unittest.main()
