"""Replay a frozen fused-score study before atomically installing its plan.

Local JSON provenance is a reproducibility check, not hardware attestation.
Only the installed immutable authored Metal sources can enter an artifact.
"""
from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any

METHODS = ("stable_library", "fast_library", "eager_library", "best_fixed", "per_shape", "dispatcher")
LIBRARIES = METHODS[:2]
CANDIDATES = ("score_simd", "score_tile4")
DEVELOPMENT_METHODS = (*METHODS[:3], *CANDIDATES)
ROUTED_METHODS = (*LIBRARIES, *CANDIDATES)
APPLICATION_ARMS = (*("global:" + name for name in DEVELOPMENT_METHODS),
                    *("routed:" + name for name in ROUTED_METHODS), "plan_library")
SEEDS = [17101, 18203, 19307]
SEQUENCES = [[2, 0, 1, 3], [1, 3, 2, 0], [3, 0, 2, 1]]
LIMITS = {"candidates_per_shape": 2, "portfolio_kernels": 2, "max_rules": 128,
          "source_bytes": 24 * 1024, "development_wall_seconds": 900,
          "memory_bytes": 2 * 1024 ** 3, "cache_bytes": 256 * 1024 ** 2}


def _read(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 20_000_000:
        raise ValueError("Evidence exceeds 20 MB")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Evidence must be an object")
    return value


def _sha(path: Path) -> str:
    if path.stat().st_size > 20_000_000:
        raise ValueError("Evidence exceeds 20 MB")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rename_new(source: Path, target: Path) -> None:
    """Publish one directory atomically without replacing a racing output."""
    if sys.platform == "win32":
        source.rename(target)  # Windows rename refuses an existing target.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(source), os.fsencode(target), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, os.fsencode(source), -100, os.fsencode(target), 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError("Atomic exclusive directory publication is unavailable")
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(code, "Output appeared during export", str(target))
        raise OSError(code, os.strerror(code), str(target))


def _number(value: Any, maximum: float, *, zero: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not (0 <= value if zero else 0 < value) or value > maximum:
        raise ValueError("Invalid or over-budget resource/cost")


def _samples(values: Any, count: int = 40) -> None:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Incomplete paired samples")
    for value in values:
        _number(value, 1e12)


def _interval(a: list[float], b: list[float], seed: int = 1404) -> dict[str, Any]:
    if len(a) != len(b) or not 40 <= len(a) <= 1000:
        raise ValueError("Need 40..1000 paired costs")
    _samples(a, len(a)); _samples(b, len(b))
    # Every resample of a constant ratio has exactly that ratio. This also keeps
    # synthetic guard fixtures light without substituting a different estimator.
    ratio = sum(a) / sum(b)
    if all(x == a[0] for x in a) and all(x == b[0] for x in b):
        bounds = [ratio, ratio]
    else:
        rng = random.Random(seed)
        ratios = []
        for _ in range(4000):
            ids = rng.choices(range(len(a)), k=len(a))
            ratios.append(sum(a[i] for i in ids) / sum(b[i] for i in ids))
        ratios.sort()
        bounds = [ratios[100], ratios[-101]]
    return {"speedup_total_cost": ratio, "ci95": bounds,
            "baseline_total_us": sum(a), "candidate_total_us": sum(b)}


def _orders(rows: Any, methods: tuple[str, ...], seed: int, count: int = 40) -> None:
    rng = random.Random(seed)
    expected = []
    for _ in range(count):
        order = list(methods)
        rng.shuffle(order)
        expected.append(order)
    if rows != expected:
        raise ValueError("Incomplete or changed paired measurement order")


def _method_samples(record: dict[str, Any], methods: tuple[str, ...]) -> None:
    for key in ("score_samples_us", "application_samples_us"):
        if set(record[key]) != set(methods):
            raise ValueError("Incomplete method samples")
        for values in record[key].values():
            _samples(values)


def verify_study(directory: Path) -> dict[str, Any]:
    """Verify identities, raw costs, coverage and all three independent gates.

    A complete losing study returns confirmation_passed=False. Malformed,
    incomplete, changed or over-budget evidence raises ValueError instead.
    """
    from . import fused_search as fused
    from .workload import environment_identity

    directory = Path(directory)
    names = ["protocol.json", "policy-frozen.json", "development.json", "numerical-validation.json", "corpus.json", "cold-study.json",
             *[f"confirmation-{i}.json" for i in range(3)]]
    records, hashes = {}, {}
    for name in names:
        path = directory / name
        if path.stat().st_size > 20_000_000:
            raise ValueError("Evidence exceeds 20 MB")
        content = path.read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        records[name] = json.loads(content)
        if not isinstance(records[name], dict):
            raise ValueError("Evidence must be an object")
    protocol, policy = records["protocol.json"], records["policy-frozen.json"]
    development, numerical = records["development.json"], records["numerical-validation.json"]
    source, implementation, environment = fused.score_source_identity(), fused.fused_implementation_identity(), environment_identity()
    sources = {name: hashlib.sha256(value).hexdigest() for name, value in fused._SOURCES}
    source_bytes = sum(len(value) for _, value in fused._SOURCES)
    script_pin = _read(Path(fused.__file__).parent / "fused-study-source.json")["measurement_script_sha256"]

    def identity(record: dict[str, Any], *, env: bool = True) -> None:
        if record.get("source_sha256") != source or record.get("study_identity") != implementation:
            raise ValueError("Operation/application/baseline/study source changed")
        if env and record.get("environment") != environment:
            raise ValueError("Study runtime changed")
        if record.get("corpus_sha256") != hashes["corpus.json"]:
            raise ValueError("Frozen corpus changed")

    identity(protocol)
    identity(policy | {"corpus_sha256": policy["evidence"].get("corpus_sha256")})
    if protocol.get("schema") != 1 or protocol.get("target") != "fused_query_scores_v1" or protocol.get("script_sha256") != script_pin:
        raise ValueError("Incompatible frozen protocol or measurement script")
    if protocol.get("limits") != LIMITS or protocol.get("sources") != sources or protocol.get("source_bytes") != source_bytes or source_bytes > LIMITS["source_bytes"]:
        raise ValueError("Changed source/resource budget")
    chunks = records["corpus.json"]["chunks"]
    if not isinstance(chunks, list) or len(chunks) < 64 or not all(isinstance(value, str) for value in chunks) or protocol.get("corpus_chunks") != len(chunks):
        raise ValueError("Invalid frozen corpus")
    tasks = [{"id": "notes", "rows": 64, "width": 384, "queries": 1, "layout": "contiguous"},
             {"id": "manuals", "rows": min(256, len(chunks)), "width": 768, "queries": 8, "layout": "column_stride2"},
             {"id": "archive", "rows": len(chunks), "width": 1536, "queries": 17, "layout": "transpose"}]
    confirmation = {"sessions": 3, "repeats_per_session": 40, "seeds": SEEDS, "sequence_orders": SEQUENCES,
                    "unseen_task": {"id": "unseen", "rows": 113, "width": 641, "queries": 5, "layout": "column_stride2"}}
    development_protocol = {"seed": 104, "repeats": 40, "order_seed": 1207, "authored_configurations": 2,
                            "score_observations_per_round": 5, "application_adapter_observations_per_round": 10}
    if protocol.get("tasks") != tasks or protocol.get("confirmation") != confirmation or protocol.get("development") != development_protocol or protocol.get("measurement_batch") != 32 or protocol.get("application_measurement_batch") != 4:
        raise ValueError("Changed frozen workload or independent session protocol")
    noise = protocol["baseline_noise"]
    if len(noise) != len(tasks) or [row["task"] for row in noise] != tasks:
        raise ValueError("Incomplete prospective baseline checks")
    worst = 0.0
    for row in noise:
        _samples(row["baseline_a_us"]); _samples(row["baseline_b_us"])
        low, high = _interval(row["baseline_a_us"], row["baseline_b_us"])["ci95"]
        if low < .9 or high > 1.1:
            raise ValueError("Prospective baseline stability gate failed")
        worst = max(worst, abs(low - 1), abs(high - 1))
    criterion = {"score_ci95_lower_speedup": max(1.10, 1 + 2 * worst), "application_ci95_lower_speedup": 1.03,
                 "portfolio_over_fixed_minimum_total_cost_gain": .02, "numerical_rtol": 2e-5, "numerical_atol": 2e-6}
    if protocol["criterion"] != criterion:
        raise ValueError("Frozen acceptance criterion changed")
    evidence = policy["evidence"]
    if evidence.get("numerical_passed") is not True or evidence.get("confirmation_passed") is not False:
        raise ValueError("Expected an unconfirmed numerically verified frozen policy")
    for key, name in (("protocol_sha256", "protocol.json"), ("numerical_report_sha256", "numerical-validation.json"), ("development_sha256", "development.json")):
        if evidence.get(key) != hashes[name]:
            raise ValueError("Frozen policy evidence bytes changed")
    if policy.get("schema") != 1 or policy.get("operation") != "stable_l2_scores_v1" or policy.get("epsilon") != 1e-12 or policy.get("sources") != sources:
        raise ValueError("Incompatible frozen policy")
    # This calls only compatibility/shape guards. No kernel is compiled or run.
    fused.FusedPlan._trial(policy)
    if numerical.get("kind") != "executed_fused_metal_numerical_validation" or numerical.get("source_sha256") != source or numerical.get("study_identity") != implementation or numerical.get("environment") != environment or numerical.get("success") is not True or numerical.get("gpu_executed") is not True or type(numerical.get("tests_run")) is not int or numerical["tests_run"] < 7 or any(type(numerical.get(key)) is not int or numerical[key] != 0 for key in ("failures", "errors", "skipped")):
        raise ValueError("Independent executed current-source GPU numerical validation missing or failed")
    identity(development, env=False)
    if development.get("protocol_sha256") != hashes["protocol.json"]:
        raise ValueError("Development protocol changed")
    _number(development["search_wall_seconds"], 900)
    _number(development["peak_memory_bytes"], LIMITS["memory_bytes"], zero=True)
    rows = development["rows"]
    if len(rows) != len(tasks) or [row["task"] for row in rows] != tasks:
        raise ValueError("Incomplete development tasks")
    rng = random.Random(1207)
    recorded_development_us = 0.
    for row, task in zip(rows, tasks):
        if row["shape"] != [task["queries"], task["rows"], task["width"]] or row.get("weight") != 3 or row.get("numerical_check", {}).get("passed") is not True or row["numerical_check"].get("rtol") != 2e-5 or row["numerical_check"].get("atol") != 2e-6:
            raise ValueError("Development shape or numerical check missing")
        _method_samples(row, DEVELOPMENT_METHODS)
        routed_samples = row.get("routed_application_samples_us")
        if not isinstance(routed_samples, dict) or set(routed_samples) != set(ROUTED_METHODS):
            raise ValueError("Incomplete routed-adapter application samples")
        for values in routed_samples.values():
            _samples(values)
        _samples(row.get("plan_library_application_samples_us"))
        recorded_development_us += sum(sum(values) for values in row["score_samples_us"].values()) * 32
        recorded_development_us += sum(sum(values) for values in row["application_samples_us"].values())
        recorded_development_us += sum(sum(values) for values in routed_samples.values()) + sum(row["plan_library_application_samples_us"])
        adapter_checks = row.get("application_numerical_checks")
        if not isinstance(adapter_checks, dict) or set(adapter_checks) != set(APPLICATION_ARMS) or any(not isinstance(check, dict) or check.get("scores_passed") is not True or check.get("ids_association_passed") is not True for check in adapter_checks.values()):
            raise ValueError("Incomplete or failed application-adapter numerical checks")
        for key in ("first_score_in_development_ms", "first_application_in_development_ms"):
            if set(row[key]) != set(DEVELOPMENT_METHODS):
                raise ValueError("Incomplete first-development-call costs")
            for value in row[key].values():
                _number(value, 900_000)
                recorded_development_us += value * 1000
        adapter_first = row.get("first_adapter_application_in_development_ms")
        if not isinstance(adapter_first, dict) or set(adapter_first) != set(APPLICATION_ARMS[5:]):
            raise ValueError("Incomplete first-adapter costs")
        for value in adapter_first.values():
            _number(value, 900_000)
            recorded_development_us += value * 1000
        expected, expected_applications = [], []
        for _ in range(40):
            order = list(DEVELOPMENT_METHODS); rng.shuffle(order); expected.append(order)
            app_order = list(APPLICATION_ARMS); rng.shuffle(app_order); expected_applications.append(app_order)
        if row["orders"] != expected or row.get("application_orders") != expected_applications:
            raise ValueError("Changed development measurement order")
    if recorded_development_us > development["search_wall_seconds"] * 1e6:
        raise ValueError("Recorded development costs exceed its wall-time budget")
    known = {tuple(row["shape"]): row for row in rows}
    rules = policy["rules"]
    for rule in rules:
        row = known.get(tuple(rule["shape"]))
        if row is None or any(_interval(row["score_samples_us"][base], row["score_samples_us"][rule["candidate"]])["ci95"][0] <= criterion["score_ci95_lower_speedup"] for base in LIBRARIES):
            raise ValueError("Rule lacks development eligibility against both libraries")
    selection = policy["policy_selection"]
    cost = lambda row, name: statistics.mean(row["application_samples_us"][name])
    routed_cost = lambda row, name: statistics.mean(row["routed_application_samples_us"][name])
    plan_cost = lambda row, name: statistics.mean(row["plan_library_application_samples_us"]) if name == "stable_library" else cost(row, name)
    choices = (*LIBRARIES, *CANDIDATES)
    fixed = min(choices, key=lambda name: sum(cost(row, name) for row in rows))
    per_shape = [{"shape": row["shape"], "candidate": min(choices, key=lambda name: routed_cost(row, name))} for row in sorted(rows, key=lambda row: tuple(row["shape"]))]
    if selection.get("best_fixed") != fixed or selection.get("per_shape") != per_shape or selection.get("joint_subset") != sorted({rule["candidate"] for rule in rules}) or selection.get("development_per_shape_application_cost_us") != sum(min(routed_cost(row, name) for name in choices) for row in rows):
        raise ValueError("Frozen fixed/per-shape selection differs from raw development")
    eligible = {(tuple(row["shape"]), name): all(_interval(row["score_samples_us"][base], row["score_samples_us"][name])["ci95"][0] > criterion["score_ci95_lower_speedup"] for base in LIBRARIES)
                for row in rows for name in CANDIDATES}
    portfolios = []
    for count in (0, 1, 2):
        for subset in itertools.combinations(CANDIDATES, count):
            chosen, total = {}, 0.
            for row in rows:
                options = ["stable_library", *[name for name in subset if eligible[tuple(row["shape"]), name]]]
                name = min(options, key=lambda name: plan_cost(row, name))
                total += plan_cost(row, name)
                if name in CANDIDATES:
                    chosen[tuple(row["shape"])] = name
            portfolios.append({"subset": subset, "rules": chosen, "total_us": total})
    best_one = min((entry for entry in portfolios if len(entry["subset"]) <= 1), key=lambda entry: entry["total_us"])
    joint = min(portfolios, key=lambda entry: entry["total_us"])
    if joint["total_us"] > best_one["total_us"] * .98:
        joint = best_one
    selected_rules = [{"shape": list(shape), "candidate": name, "dtype": "float32", "layout": "any_mlx_strides"} for shape, name in sorted(joint["rules"].items())]
    if rules != selected_rules or selection.get("development_application_cost_us") != joint["total_us"] or selection.get("best_one_subset_application_cost_us") != best_one["total_us"] or selection.get("best_fixed_application_cost_us") != sum(cost(row, fixed) for row in rows):
        raise ValueError("Frozen joint portfolio differs from raw development selection")
    all_tasks = [*tasks, confirmation["unseen_task"]]
    expected_checks = {(task["id"], method) for task in all_tasks for method in METHODS}
    sessions = []
    for i in range(3):
        result = records[f"confirmation-{i}.json"]
        identity(result)
        if result.get("session") != i or result.get("seed") != SEEDS[i] or result.get("sequence_order") != SEQUENCES[i] or result.get("protocol_sha256") != hashes["protocol.json"] or result.get("policy_sha256") != hashes["policy-frozen.json"]:
            raise ValueError("Independent session identity/protocol/policy changed")
        checks = result["checks"]
        if len(checks) != len(expected_checks) or {(c["task"], c["method"]) for c in checks} != expected_checks or any(c.get("scores_passed") is not True or c.get("ids_association_passed") is not True for c in checks):
            raise ValueError("Incomplete or failed task/method top-k checks")
        _method_samples(result, METHODS)
        _orders(result["orders"], METHODS, SEEDS[i])
        if set(result["first_sequence_in_session_ms"]) != set(METHODS):
            raise ValueError("Incomplete first-session costs")
        for value in result["first_sequence_in_session_ms"].values():
            _number(value, 1e9)
        _number(result["artifact_load_ms"], 1e9)
        _number(result["peak_memory_bytes"], LIMITS["memory_bytes"], zero=True)
        counts = result["fallback_counts"]["dispatcher"]
        if type(counts["calls"]) is not int or type(counts["fallback_calls"]) is not int or not 0 < counts["fallback_calls"] <= counts["calls"] or (rules and counts["fallback_calls"] == counts["calls"]):
            raise ValueError("Custom dispatch and unseen whole-query fallback must both execute")
        sessions.append(result)
    per_session = [{stage: {method: {base: _interval(s[key][base], s[key][method], s["seed"]) for base in LIBRARIES}
                            for method in METHODS if method not in LIBRARIES}
                    for stage, key in (("score", "score_samples_us"), ("application", "application_samples_us"))} for s in sessions]
    score_passed = all(s["score"]["dispatcher"][base]["ci95"][0] > criterion["score_ci95_lower_speedup"] for s in per_session for base in LIBRARIES)
    application_passed = all(s["application"]["dispatcher"][base]["ci95"][0] > 1.03 for s in per_session for base in LIBRARIES)
    pooled = {}
    for stage, key in (("score", "score_samples_us"), ("application", "application_samples_us")):
        values = {method: [v for session in sessions for v in session[key][method]] for method in METHODS}
        pooled[stage] = {method: {base: _interval(values[base], values[method]) for base in LIBRARIES}
                         for method in METHODS if method not in LIBRARIES}
        pooled[stage]["dispatcher_vs_fixed"] = _interval(values["best_fixed"], values["dispatcher"])
        pooled[stage]["dispatcher_vs_per_shape"] = _interval(values["per_shape"], values["dispatcher"])
    cold = records["cold-study.json"]
    identity(cold, env=False)
    if cold.get("policy_sha256") != hashes["policy-frozen.json"] or cold.get("schema") != 1:
        raise ValueError("Cold-study policy changed")
    _number(cold["wall_seconds"], 180)
    expected_cold = []
    rng = random.Random(1667)
    for repeat in range(3):
        order = list(METHODS); rng.shuffle(order)
        expected_cold.extend((repeat, method) for method in order)
    if len(cold["rows"]) != 18 or [(row["repeat"], row["method"]) for row in cold["rows"]] != expected_cold:
        raise ValueError("Incomplete or changed independent cold-process order")
    for row in cold["rows"]:
        identity(row)
        if row.get("policy_sha256") != hashes["policy-frozen.json"]:
            raise ValueError("Cold worker policy changed")
        for key in ("runtime_setup_ms", "frozen_checks_ms", "policy_setup_ms", "input_prepare_ms", "first_application_sequence_ms", "process_wall_ms"):
            _number(row[key], 180_000)
        _number(row["peak_memory_bytes"], LIMITS["memory_bytes"], zero=True)
    if sum(row["process_wall_ms"] for row in cold["rows"]) > cold["wall_seconds"] * 1000:
        raise ValueError("Cold costs exceed recorded wall time")
    cold_summary = {method: {"median_process_wall_ms": statistics.median(row["process_wall_ms"] for row in cold["rows"] if row["method"] == method),
                             "median_first_sequence_ms": statistics.median(row["first_application_sequence_ms"] for row in cold["rows"] if row["method"] == method)} for method in METHODS}
    return {"schema": 1, "numerical_passed": True, "score_passed": score_passed, "application_passed": application_passed,
            "resource_passed": True, "confirmation_passed": bool(rules) and score_passed and application_passed,
            "source_sha256": source, "study_identity": implementation, "environment": environment,
            "per_session": per_session, "comparisons": pooled, "cold_start": cold_summary,
            "evidence_sha256": hashes, "authored_source_bytes": source_bytes,
            "limitations": ["Local JSON provenance is not cryptographic hardware attestation.", "Pooled intervals are descriptive; every independent session must beat both optimized libraries.", "Synchronized host-wall costs include allocation, dispatch and materialization.", "Fresh processes retain shared OS/driver caches; pure compiler time is not isolated."]}


def export_artifact(study: Path, output: Path) -> dict[str, Any]:
    """Install only a complete accepted study, preserving all existing outputs."""
    from . import fused_search as fused
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new artifact directory")
    verified = verify_study(study)
    if verified["confirmation_passed"] is not True:
        raise ValueError("Frozen confirmation failed; preserve the null result")
    policy = copy.deepcopy(_read(study / "policy-frozen.json"))
    # Recheck bytes after the potentially long raw bootstrap replay.
    for name, digest in verified["evidence_sha256"].items():
        if _sha(study / name) != digest:
            raise ValueError("Study changed during verification")
    if not policy["rules"]:
        raise ValueError("No confirmed custom rules")
    policy["evidence"] = {**policy["evidence"], "confirmation_passed": True, "verification": verified["evidence_sha256"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kd-fused-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        for name, content in fused._SOURCES:
            (staging / name).write_bytes(content)
        (staging / "manifest.json").write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
        (staging / "verification.json").write_text(json.dumps(verified, indent=2, allow_nan=False) + "\n")
        if fused.FusedPlan(staging).status != "installed":
            raise ValueError("Staged artifact failed installed runtime guards")
        if output.exists():
            raise FileExistsError("Output appeared during export")
        _rename_new(staging, output)
    return {"artifact": str(output), "confirmation_passed": True, "source_sha256": verified["source_sha256"],
            "bytes": sum(path.stat().st_size for path in output.iterdir())}
