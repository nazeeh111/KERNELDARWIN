"""Strict, CPU-only replay and exclusive export of a frozen top-five study.

Local evidence hashes establish reproducibility, not cryptographic GPU
attestation. This verifier never compiles kernels or measures performance.
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

METHODS = ("stable_library", "fast_library", "stable_reduction", "fast_reduction",
           "eager_library", "best_fixed", "per_shape", "dispatcher")
LIBRARIES = METHODS[:4]
CANONICAL_RANK_LIBRARIES = ("stable_library", "stable_reduction")
CANDIDATES = ("simd32", "group128")
GLOBAL_NAMES = (*LIBRARIES, "eager_library", *CANDIDATES)
ROUTED_CHOICES = (*LIBRARIES, *CANDIDATES)
RANK_ARMS = (*GLOBAL_NAMES, *("plan:" + name for name in CANDIDATES))
APP_ARMS = (*("global:" + name for name in GLOBAL_NAMES),
            *("routed:" + name for name in ROUTED_CHOICES),
            *("plan:" + name for name in CANDIDATES), "plan_library")
SOURCE_FILES = ("rank_key.metal", "rank_simd32.metal", "rank_group128.metal")
SEEDS = [23101, 24203, 25307]
SEQUENCES = [[1, 0, 3, 2], [2, 3, 0, 1], [0, 2, 1, 3]]
DESIGN = "application_primary_symmetric_noise_v2"
UNCERTAINTY = {"resamples": 4000, "circular_block_lengths": [4, 8],
               "block_seed_stride": 100003, "primary_interval": "envelope_iid_block4_block8"}
CALIBRATION_SEEDS = list(range(21101, 21141))
CALIBRATION_ORDERS = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
CALIBRATION = {"rank_order_seed": 21441, "application_order_seed": 21443,
    "application_input_seeds": CALIBRATION_SEEDS,
    "application_sequence_orders": [CALIBRATION_ORDERS[r % 3] for r in range(40)],
    "negative_control_method": "stable_library", "application_first_warmup_order": [0, 1, 2],
    "shared_method_instance": True}
LIMITS = {"candidates_per_shape": 2, "portfolio_kernels": 2, "max_rules": 128,
          "source_bytes": 12 * 1024, "development_wall_seconds": 900,
          "cold_study_wall_seconds": 180, "freeze_wall_seconds": 120, "memory_bytes": 2 * 1024 ** 3,
          "cache_bytes": 256 * 1024 ** 2}
DEBUG_FLAGS = ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "MTL_SHADER_VALIDATION_ENABLE_PIPELINES",
               "METAL_DEVICE_WRAPPER_TYPE", "METAL_DEBUG_ERROR_MODE", "MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS",
               "MTL_DEBUG_LAYER_VALIDATE_STORE_ACTIONS", "MTL_DEBUG_LAYER_VALIDATE_UNRETAINED_RESOURCES")


def _bytes(path: Path, maximum: int = 20_000_000) -> bytes:
    size = path.stat().st_size
    if not 0 <= size <= maximum:
        raise ValueError("Evidence/source exceeds bounded size")
    with path.open("rb") as handle:
        content = handle.read(maximum + 1)
    if len(content) != size:
        raise ValueError("Evidence/source changed during bounded read")
    return content


def _decode(content: bytes) -> dict[str, Any]:
    def invalid(value):
        raise ValueError("Nonfinite JSON number: " + value)
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    value = json.loads(content, parse_constant=invalid, parse_float=finite_float, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Evidence must be an object")
    return value


def _read(path: Path) -> dict[str, Any]:
    return _decode(_bytes(path))


def _sha(path: Path) -> str:
    return hashlib.sha256(_bytes(path)).hexdigest()


def _same(actual: Any, expected: Any) -> bool:
    # JSON equality distinguishes True from 1 and fractional metadata from ints.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def _require(actual: Any, expected: Any, message: str) -> None:
    if not _same(actual, expected):
        raise ValueError(message)


def _number(value: Any, maximum: float, *, zero: bool = False) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not (0 <= value if zero else 0 < value) or value > maximum):
        raise ValueError("Invalid or over-budget resource/cost")


def _count(value: Any, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("Invalid or over-budget integer count")


def _samples(values: Any, count: int = 40) -> None:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Incomplete paired samples")
    for value in values:
        _number(value, 1e12)


def _interval(a: list[float], b: list[float], seed: int = 1404) -> dict[str, Any]:
    if len(a) != len(b) or not 40 <= len(a) <= 1000 or len(a) % 8:
        raise ValueError("Need 40..1000 paired costs divisible by 4 and 8")
    if type(seed) is not int:
        raise ValueError("Invalid interval seed")
    _samples(a, len(a)); _samples(b, len(b))
    ratio = sum(a) / sum(b)
    _number(ratio, sys.float_info.max)
    if all(x == a[0] for x in a) and all(x == b[0] for x in b):
        families = {name: [ratio, ratio] for name in ("iid_ci95", "block4_ci95", "block8_ci95")}
    else:
        families = {}
        for block, name in ((0, "iid_ci95"), (4, "block4_ci95"), (8, "block8_ci95")):
            rng = random.Random(seed if block == 0 else seed + block * 100003)
            ratios = []
            for _ in range(4000):
                if block == 0:
                    ids = rng.choices(range(len(a)), k=len(a))
                else:
                    ids = []
                    for _ in range(len(a) // block):
                        start = rng.randrange(len(a))
                        ids.extend((start + offset) % len(a) for offset in range(block))
                ratios.append(sum(a[i] for i in ids) / sum(b[i] for i in ids))
            if any(not math.isfinite(value) or value <= 0 for value in ratios):
                raise ValueError("Nonfinite or underflowed resampled cost ratio")
            ratios.sort()
            families[name] = [ratios[100], ratios[-101]]
    bounds = [min(interval[0] for interval in families.values()), max(interval[1] for interval in families.values())]
    return {"speedup_total_cost": ratio, "ci95": bounds,
            "baseline_total_us": sum(a), "candidate_total_us": sum(b), **families}


def _arrays(record: dict, key: str, names: tuple[str, ...]) -> None:
    values = record.get(key)
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError("Incomplete " + key)
    for samples in values.values():
        _samples(samples)


def _orders(rank_orders, app_orders, rank_names, app_names, rng, count=40):
    rank, app = [], []
    for _ in range(count):
        order = list(rank_names); rng.shuffle(order); rank.append(order)
        order = list(app_names); rng.shuffle(order); app.append(order)
    _require(rank_orders, rank, "Changed rank measurement order")
    _require(app_orders, app, "Changed application measurement order")


def _flags(record: dict) -> None:
    flags = record.get("timing_validation_flags")
    if not isinstance(flags, dict) or not set(DEBUG_FLAGS) <= set(flags):
        raise ValueError("Missing timing validation flags")
    for key, value in flags.items():
        if key not in DEBUG_FLAGS and not key.startswith(("MTL_DEBUG", "MTL_SHADER_VALIDATION")):
            raise ValueError("Unknown timing validation flag")
        if value is not None and (not isinstance(value, str) or value.lower() not in ("", "0", "false", "no", "off")):
            raise ValueError("Timing under native debug/validation is prohibited")


def _telemetry(observed: Any) -> None:
    if not isinstance(observed, dict) or not isinstance(observed.get("time_utc"), str):
        raise ValueError("Missing observed telemetry timestamp")
    loads = observed.get("load_average")
    if not isinstance(loads, list) or len(loads) != 3:
        raise ValueError("Missing observed load telemetry")
    for load in loads:
        _number(load, 1e9, zero=True)
    for name in ("thermal", "power"):
        probe = observed.get(name)
        if not isinstance(probe, dict):
            raise ValueError("Missing observed or explicitly unavailable telemetry")
        if set(probe) == {"unavailable"}:
            if not isinstance(probe["unavailable"], str) or not probe["unavailable"]:
                raise ValueError("Invalid unavailable telemetry")
        elif (set(probe) != {"returncode", "observation"} or type(probe["returncode"]) is not int
              or not isinstance(probe["observation"], str)):
            raise ValueError("Invalid observed telemetry")
    if not isinstance(observed.get("thermal_limit"), str) or not observed["thermal_limit"]:
        raise ValueError("Missing telemetry limitations")


def _source_snapshot(rank):
    if tuple(rank.RANK_SOURCE_FILES) != SOURCE_FILES or tuple(rank.RANK_CANDIDATES) != CANDIDATES:
        raise ValueError("Authored rank source/configuration set changed")
    if tuple(name for name, _ in rank._SOURCES) != SOURCE_FILES:
        raise ValueError("Immutable rank snapshot changed")
    sources = {}
    size = 0
    for name, content in rank._SOURCES:
        size += len(content)
        if size > LIMITS["source_bytes"]:
            raise ValueError("Authored MSL over-budget")
        if _bytes(Path(rank.__file__).parent / "metal" / name, LIMITS["source_bytes"]) != content:
            raise ValueError("Installed source differs from immutable rank snapshot")
        sources[name] = hashlib.sha256(content).hexdigest()
    loaded = hashlib.sha256(_bytes(Path(rank.__file__)) + b"".join(
        name.encode() + content for name, content in rank._SOURCES)).hexdigest()
    if loaded != rank.rank_source_identity():
        raise ValueError("Rank adapter source changed after import")
    return sources, size


def _loaded_application_sources() -> None:
    """A hot-replaced file must not relabel already loaded Python/Metal code."""
    from . import library_baselines, search, study_identity, workload
    for module, loaded in ((search, search._LOADED_APPLICATION_IDENTITY),
                           (library_baselines, library_baselines._LOADED_BASELINE_IDENTITY),
                           (study_identity, study_identity._LOADED_BINDING_IDENTITY)):
        if hashlib.sha256(_bytes(Path(module.__file__))).hexdigest() != loaded:
            raise ValueError("Loaded application/baseline/binding source changed")
    data = _bytes(Path(workload.__file__))
    for name, content in workload._LOADED_SOURCE:
        if _bytes(Path(workload.__file__).parent / "metal" / name) != content:
            raise ValueError("Loaded normalization source changed")
        data += name.encode() + content
    if hashlib.sha256(data).hexdigest() != workload.source_identity():
        raise ValueError("Loaded normalization adapter source changed")


def _verify(directory: Path) -> dict[str, Any]:
    from . import rank_search as rank
    from .workload import environment_identity

    names = ["protocol.json", "policy-frozen.json", "development.json", "numerical-validation.json",
             "corpus.json", "cold-study.json", *[f"confirmation-{i}.json" for i in range(3)]]
    # Hash exactly the bytes parsed, then recheck their hashes before publication.
    records, hashes = {}, {}
    for name in names:
        content = _bytes(directory / name)
        hashes[name] = hashlib.sha256(content).hexdigest()
        records[name] = _decode(content)
    protocol, policy = records["protocol.json"], records["policy-frozen.json"]
    development, numerical = records["development.json"], records["numerical-validation.json"]
    _loaded_application_sources()
    source, implementation, environment = rank.rank_source_identity(), rank.rank_implementation_identity(), environment_identity()
    sources, source_bytes = _source_snapshot(rank)
    pin = _read(Path(rank.__file__).parent / "rank-study-source.json")["measurement_script_sha256"]
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("Invalid measurement script pin")

    def identity(record, *, corpus=True):
        if record.get("source_sha256") != source or record.get("study_identity") != implementation:
            raise ValueError("Rank/application/baseline/study source changed")
        _require(record.get("environment"), environment, "Study runtime changed")
        if corpus and record.get("corpus_sha256") != hashes["corpus.json"]:
            raise ValueError("Frozen corpus changed")

    def bindings(record, pairs):
        for key, name in pairs:
            if record.get(key) != hashes[name]:
                raise ValueError("Frozen evidence bytes changed: " + key)

    identity(protocol)
    _require(protocol.get("schema"), 2, "Invalid V2 protocol schema")
    _require(protocol.get("measurement_design"), DESIGN, "Invalid prospective measurement design")
    if protocol.get("target") != "stable_top5_pairs_v1" or protocol.get("script_sha256") != pin:
        raise ValueError("Incompatible rank target or measurement pin")
    _require(protocol.get("limits"), LIMITS, "Changed resource limits")
    _require(protocol.get("sources"), sources, "Rank source hashes changed")
    _require(protocol.get("source_bytes"), source_bytes, "Rank source bytes changed")
    bindings(protocol, [("numerical_report_sha256", "numerical-validation.json")])
    _count(protocol["peak_memory_bytes"], LIMITS["memory_bytes"])
    _flags(protocol)
    chunks = records["corpus.json"].get("chunks")
    if not isinstance(chunks, list) or not 97 <= len(chunks) <= 2048 or not all(isinstance(c, str) and c for c in chunks):
        raise ValueError("Invalid or unsupported frozen corpus")
    _require(protocol.get("corpus_chunks"), len(chunks), "Corpus count changed")
    tasks = [{"id": "notes", "rows": 64, "width": 384, "queries": 1, "layout": "contiguous"},
             {"id": "manuals", "rows": min(256, len(chunks)), "width": 768, "queries": 8, "layout": "column_stride2"},
             {"id": "archive", "rows": len(chunks), "width": 1536, "queries": 17, "layout": "transpose"}]
    confirmation = {"sessions": 3, "repeats_per_session": 40, "seeds": SEEDS, "sequence_orders": SEQUENCES,
                    "unseen_task": {"id": "unseen", "rows": 97, "width": 1025, "queries": 7, "layout": "column_stride2"},
                    "minimum_separation_seconds": 120}
    expected = {"tasks": tasks, "confirmation": confirmation,
        "uncertainty": UNCERTAINTY, "calibration": CALIBRATION,
        "development": {"seed": 21104, "repeats": 40, "order_seed": 21207, "authored_configurations": 2,
                        "rank_observations_per_round": 9, "application_adapter_observations_per_round": 16},
        "measurement_batch": 128, "application_measurement_batch": 16,
        "methods": list(METHODS), "libraries": list(LIBRARIES), "canonical_rank_libraries": list(CANONICAL_RANK_LIBRARIES),
        "global_names": list(GLOBAL_NAMES), "routed_choices": list(ROUTED_CHOICES),
        "rank_arms": list(RANK_ARMS), "application_arms": list(APP_ARMS),
        "application": "One index refresh plus three executed changed-query top5 searches per task",
        "weights": "Equal task frequency; sum three rank costs per task and complete application costs",
        "qos": "user_initiated_current_thread_only",
        "source_budget_scope": "All three authored MSL files including shared header"}
    for key, value in expected.items():
        _require(protocol.get(key), value, "Changed frozen workload/protocol: " + key)
    allowed = set(expected) | {"schema", "target", "measurement_design", "frozen_at", "limits",
        "criterion", "selection", "comparator_domain", "stability_gate", "timing", "confirmation_rule",
        "environment", "telemetry_before", "telemetry", "baseline_noise", "application_baseline_noise",
        "calibration_wall_seconds", "timing_validation_flags", "peak_memory_bytes", "source_sha256",
        "study_identity", "sources", "source_bytes", "script_sha256", "numerical_report_sha256",
        "corpus_chunks", "corpus_sha256"}
    if set(protocol) - allowed:
        raise ValueError("Unknown frozen protocol field")
    for task in tasks:
        if task["queries"] * task["rows"] * task["width"] > 16_777_216:
            raise ValueError("Unsupported frozen rank task shape")
    # Rank controls calibrate the symmetric margin without a rank-only entry cap.
    noise = protocol["baseline_noise"]
    if not isinstance(noise, list) or len(noise) != 3:
        raise ValueError("Incomplete prospective noise")
    rng, rank_envelope, measured_calibration_us = random.Random(21441), 1., 0.
    for row, task in zip(noise, tasks):
        if not isinstance(row, dict) or set(row) != {"task", "baseline_a_us", "baseline_b_us", "orders", "comparison", "first_rank_ms"}:
            raise ValueError("Unknown or missing rank control field")
        _require(row["task"], task, "Noise task changed")
        _samples(row["baseline_a_us"]); _samples(row["baseline_b_us"])
        orders = []
        for _ in range(40):
            order = [0, 1]; rng.shuffle(order); orders.append(order)
        _require(row["orders"], orders, "Changed noise pair order")
        _number(row["first_rank_ms"], 120_000)
        comparison = _interval(row["baseline_a_us"], row["baseline_b_us"])
        _require(row["comparison"], comparison, "Noise summary differs from raw timings")
        low, high = comparison["ci95"]
        rank_envelope = max(rank_envelope, high, 1 / low)
        measured_calibration_us += 128 * (sum(row["baseline_a_us"]) + sum(row["baseline_b_us"])) + row["first_rank_ms"] * 1000
    app_noise = protocol["application_baseline_noise"]
    for key in ("baseline_a_us", "baseline_b_us"):
        _samples(app_noise[key])
    rng, orders = random.Random(21443), []
    for _ in range(40):
        order = [0, 1]; rng.shuffle(order); orders.append(order)
    control = {"orders": orders, "input_seeds": CALIBRATION_SEEDS,
        "sequence_orders": CALIBRATION["application_sequence_orders"],
        "method": "stable_library", "sequences_per_block": 16}
    if set(app_noise) != {*control, "baseline_a_us", "baseline_b_us", "comparison", "first_application_sequence_ms"}:
        raise ValueError("Unknown or missing application control field")
    for key, value in control.items():
        _require(app_noise.get(key), value, "Changed application calibration protocol: " + key)
    app_comparison = _interval(app_noise["baseline_a_us"], app_noise["baseline_b_us"])
    _require(app_noise["comparison"], app_comparison, "Application noise summary differs from raw timings")
    app_low, app_high = app_comparison["ci95"]
    if not .9 <= app_low <= app_high <= 1.1:
        raise ValueError("Prospective complete-application noise stability gate failed")
    app_envelope = max(1., app_high, 1 / app_low)
    _number(app_noise["first_application_sequence_ms"], 120_000)
    measured_calibration_us += 16 * (sum(app_noise["baseline_a_us"]) + sum(app_noise["baseline_b_us"])) + app_noise["first_application_sequence_ms"] * 1000
    _number(protocol["calibration_wall_seconds"], 120)
    if measured_calibration_us > protocol["calibration_wall_seconds"] * 1e6:
        raise ValueError("Recorded calibration costs exceed complete calibration wall time")
    _telemetry(protocol["telemetry_before"]); _telemetry(protocol["telemetry"])
    criterion = {"rank_ci95_lower_speedup": max(1.10, rank_envelope ** 2),
                 "application_ci95_lower_speedup": max(1.03, app_envelope ** 2),
                 "portfolio_over_fixed_minimum_total_cost_gain": .02, "numerical_rtol": 2e-5, "numerical_atol": 2e-6}
    _require(protocol["criterion"], criterion, "Changed frozen acceptance criterion")
    identity(numerical, corpus=False)
    if (numerical.get("kind") != "executed_rank_metal_numerical_validation"
            or numerical.get("success") is not True or numerical.get("gpu_executed") is not True
            or type(numerical.get("tests_run")) is not int or numerical["tests_run"] < 12
            or any(type(numerical.get(k)) is not int or numerical[k] != 0 for k in ("failures", "errors", "skipped"))):
        raise ValueError("Independent current-source executed rank Metal numerical report required")
    evidence = policy["evidence"]
    identity(policy, corpus=False)
    if evidence.get("numerical_passed") is not True or evidence.get("confirmation_passed") is not False:
        raise ValueError("Expected numerically verified unconfirmed frozen policy")
    bindings(evidence, [("corpus_sha256", "corpus.json"), ("protocol_sha256", "protocol.json"),
        ("development_sha256", "development.json"), ("numerical_report_sha256", "numerical-validation.json")])
    policy_contract = {"schema": 1, "operation": "stable_top5_pairs_v1", "epsilon": 1e-12, "fixed_k": 5,
        "key_policy": "native_ftz_stable_index_original_bits_v1", "compile_options": {"math_mode": "safe"}, "sources": sources}
    for key, value in policy_contract.items():
        _require(policy.get(key), value, "Incompatible frozen rank policy: " + key)
    rank.RankPlan._trial(policy)  # Data-only guard; no kernel execution.
    identity(development)
    _require(development.get("schema"), 1, "Invalid development schema")
    bindings(development, [("protocol_sha256", "protocol.json"), ("numerical_report_sha256", "numerical-validation.json")])
    _number(development["search_wall_seconds"], 900)
    _count(development["peak_memory_bytes"], LIMITS["memory_bytes"])
    _flags(development)
    rows = development["rows"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Incomplete development tasks")
    rng, measured_us, eligible = random.Random(21207), 0., {}
    for row, task in zip(rows, tasks):
        shape = [task["queries"], task["rows"], task["width"], 5]
        _require(row["task"], task, "Changed development task")
        _require(row["shape"], shape, "Unsupported development shape")
        _require(row["weight"], 1, "Changed equal task weight")
        _arrays(row, "rank_samples_us", RANK_ARMS); _arrays(row, "application_samples_us", APP_ARMS)
        _orders(row["rank_orders"], row["application_orders"], RANK_ARMS, APP_ARMS, rng)
        for key, names, fields in (("rank_numerical_checks", RANK_ARMS, ("ids_passed", "original_bits_passed")),
                                  ("application_numerical_checks", APP_ARMS, ("scores_passed", "ids_association_passed"))):
            checks = row.get(key)
            if not isinstance(checks, dict) or set(checks) != set(names) or any(
                    not isinstance(check, dict) or any(check.get(field) is not True for field in fields) for check in checks.values()):
                raise ValueError("Incomplete/failed development numerical checks")
        measured_us += sum(sum(v) for v in row["rank_samples_us"].values()) * 128
        measured_us += sum(sum(v) for v in row["application_samples_us"].values())
        for key, names in (("first_rank_in_development_ms", RANK_ARMS), ("first_application_in_development_ms", APP_ARMS)):
            first = row[key]
            if not isinstance(first, dict) or set(first) != set(names):
                raise ValueError("Incomplete individual first-development costs")
            for value in first.values():
                _number(value, 900_000); measured_us += value * 1000
        comparisons = {}
        for name in CANDIDATES:
            comparisons[name] = {base: {"rank": _interval(row["rank_samples_us"][base], row["rank_samples_us"]["plan:" + name])}
                                 for base in CANONICAL_RANK_LIBRARIES}
            eligible[tuple(shape), name] = all(comparisons[name][base]["rank"]["ci95"][0] > criterion["rank_ci95_lower_speedup"]
                                                for base in CANONICAL_RANK_LIBRARIES)
        _require(row["comparisons"], comparisons, "Development comparisons differ from actual PLAN rank costs")
    if measured_us > development["search_wall_seconds"] * 1e6:
        raise ValueError("Recorded development costs exceed total staging/validation/warmup/analysis wall budget")
    # Reconstruct fixed, routed per-shape and every subset, with distinct adapter costs.
    cost = lambda row, arm: statistics.mean(row["application_samples_us"][arm])
    fixed = min((*LIBRARIES, *CANDIDATES), key=lambda name: sum(cost(row, "global:" + name) for row in rows))
    per_shape = {tuple(row["shape"]): min(ROUTED_CHOICES, key=lambda name: cost(row, "routed:" + name)) for row in rows}
    portfolios = []
    for count in (0, 1, 2):
        for subset in itertools.combinations(CANDIDATES, count):
            selected, total = {}, 0.
            for row in rows:
                options = ["plan_library", *("plan:" + name for name in subset if eligible[tuple(row["shape"]), name])]
                arm = min(options, key=lambda label: cost(row, label))
                total += cost(row, arm)
                if arm != "plan_library":
                    selected[tuple(row["shape"])] = arm.removeprefix("plan:")
            portfolios.append({"rules": selected, "total_us": total, "subset": subset})
    best_one = min((p for p in portfolios if len(p["subset"]) <= 1), key=lambda p: p["total_us"])
    joint = min(portfolios, key=lambda p: p["total_us"])
    if joint["total_us"] > best_one["total_us"] * .98:
        joint = best_one
    rules = [{"shape": list(shape), "candidate": name, "dtype": "float32", "layout": "any_mlx_strides"}
             for shape, name in sorted(joint["rules"].items())]
    selection = {"best_fixed": fixed, "joint_subset": sorted(set(joint["rules"].values())),
        "development_application_cost_us": joint["total_us"], "best_one_subset_application_cost_us": best_one["total_us"],
        "best_fixed_application_cost_us": sum(cost(row, "global:" + fixed) for row in rows),
        "development_per_shape_application_cost_us": sum(cost(row, "routed:" + per_shape[tuple(row["shape"])]) for row in rows),
        "per_shape": [{"shape": list(shape), "candidate": name} for shape, name in sorted(per_shape.items())]}
    _require(policy["rules"], rules, "Frozen rules differ from raw actual PLAN portfolio")
    _require(policy["policy_selection"], selection, "Frozen fixed/per-shape/joint selection differs from raw adapter costs")
    all_tasks = [*tasks, confirmation["unseen_task"]]
    expected_checks = {(task["id"], method) for task in all_tasks for method in METHODS}
    sessions, per_session, portfolio_sessions, session_spans = [], [], [], []
    for i in range(3):
        result = records[f"confirmation-{i}.json"]
        identity(result); _flags(result)
        for key, value in (("schema", 1), ("session", i), ("seed", SEEDS[i]), ("sequence_order", SEQUENCES[i])):
            _require(result.get(key), value, "Changed independent session identity/order")
        bindings(result, [("protocol_sha256", "protocol.json"), ("policy_sha256", "policy-frozen.json"),
                         ("development_sha256", "development.json"), ("numerical_report_sha256", "numerical-validation.json")])
        _count(result["process_id"], 2 ** 31 - 1)
        if result["process_id"] == 0:
            raise ValueError("Positive confirmation process ID required")
        span = {"session": i, "process_id": result["process_id"]}
        for clock in ("unix", "monotonic"):
            start, end = result[f"session_start_{clock}_ns"], result[f"session_end_{clock}_ns"]
            for stamp in (start, end):
                _count(stamp, 2 ** 63 - 1)
            if start <= 0 or end <= start:
                raise ValueError("Confirmation must have positive spans on both clocks")
            span[clock + "_span_seconds"] = (end - start) / 1e9
            gap = None if i == 0 else start - sessions[-1][f"session_end_{clock}_ns"]
            if gap is not None and gap < 120_000_000_000:
                raise ValueError("Confirmation sessions lack 120-second separation on both clocks")
            span[clock + "_gap_seconds"] = None if gap is None else gap / 1e9
        session_spans.append(span)
        _telemetry(result["telemetry_before"]); _telemetry(result["telemetry_after"])
        checks = result["checks"]
        if (not isinstance(checks, list) or len(checks) != len(expected_checks)
                or {(c["task"], c["method"]) for c in checks} != expected_checks
                or any(any(c.get(k) is not True for k in ("rank_ids_passed", "rank_original_bits_passed", "scores_passed", "ids_association_passed")) for c in checks)):
            raise ValueError("Incomplete/failed rank IDs/original values/application IDs+values checks")
        for key in ("rank_samples_us", "application_samples_us"):
            _arrays(result, key, METHODS)
        _orders(result["rank_orders"], result["application_orders"], METHODS, METHODS, random.Random(SEEDS[i]))
        for key in ("first_rank_in_session_ms", "first_application_in_session_ms"):
            first = result[key]
            if not isinstance(first, dict) or set(first) != set(METHODS):
                raise ValueError("Incomplete first-session costs")
            for values in first.values():
                if not isinstance(values, list) or len(values) != 4:
                    raise ValueError("Incomplete per-task first-session costs")
                for value in values:
                    _number(value, 1e9)
        _number(result["artifact_load_ms"], 1e9)
        _count(result["peak_memory_bytes"], LIMITS["memory_bytes"])
        counts = result["fallback_counts"]
        if not isinstance(counts, dict) or set(counts) != set(METHODS):
            raise ValueError("Incomplete method fallback counts")
        for method in METHODS:
            chosen = fixed if method == "best_fixed" else method
            routed = chosen in CANDIDATES or method in ("per_shape", "dispatcher")
            calls = 4 * (7048 if routed else 5122)
            if method == "dispatcher":
                fallback = (4 - len(rules)) * 7048
            elif method == "per_shape":
                fallback = (1 + sum(v not in CANDIDATES for v in per_shape.values())) * 7048
            else:
                fallback = 0 if chosen in CANDIDATES else calls
            _require(counts[method], {"calls": calls, "fallback_calls": fallback}, "Invalid custom/fallback execution counts")
        comparisons = {method: {base: {stage: _interval(result[key][base], result[key][method], SEEDS[i])
                       for stage, key in (("rank", "rank_samples_us"), ("application", "application_samples_us"))}
                       for base in LIBRARIES} for method in METHODS}
        _require(result["comparisons"], comparisons, "Session comparisons differ from raw costs")
        gates = {"rank": all(comparisons["dispatcher"][b]["rank"]["ci95"][0] > criterion["rank_ci95_lower_speedup"] for b in CANONICAL_RANK_LIBRARIES),
                 "application": all(comparisons["dispatcher"][b]["application"]["ci95"][0] > criterion["application_ci95_lower_speedup"] for b in LIBRARIES)}
        _require(result["session_gate"], gates, "Session gate differs from raw costs")
        portfolio_comparisons = {method: _interval(result["application_samples_us"][method],
            result["application_samples_us"]["dispatcher"], SEEDS[i]) for method in ("best_fixed", "per_shape")}
        portfolio_gates = {"over_best_fixed": portfolio_comparisons["best_fixed"]["ci95"][0] > 1.02,
            "over_per_shape": portfolio_comparisons["per_shape"]["ci95"][0] > 1.,
            "two_authored_configs": len(selection["joint_subset"]) == 2}
        _require(result["portfolio_comparisons"], portfolio_comparisons, "Untouched portfolio comparisons differ from raw costs")
        _require(result["portfolio_session_gate"], portfolio_gates, "Untouched portfolio flags differ from raw costs/config count")
        portfolio_sessions.append({"comparisons": portfolio_comparisons, "gates": portfolio_gates})
        per_session.append({stage: {method: {base: comparisons[method][base][stage] for base in LIBRARIES}
                                   for method in METHODS} for stage in ("rank", "application")})
        sessions.append(result)
    rank_passed = all(s["rank"]["dispatcher"][b]["ci95"][0] > criterion["rank_ci95_lower_speedup"] for s in per_session for b in CANONICAL_RANK_LIBRARIES)
    app_passed = all(s["application"]["dispatcher"][b]["ci95"][0] > criterion["application_ci95_lower_speedup"] for s in per_session for b in LIBRARIES)
    pooled = {}
    for stage, key in (("rank", "rank_samples_us"), ("application", "application_samples_us")):
        values = {m: [v for s in sessions for v in s[key][m]] for m in METHODS}
        pooled[stage] = {m: {b: _interval(values[b], values[m]) for b in LIBRARIES} for m in METHODS}
        pooled[stage]["dispatcher_vs_fixed"] = _interval(values["best_fixed"], values["dispatcher"])
        pooled[stage]["dispatcher_vs_per_shape"] = _interval(values["per_shape"], values["dispatcher"])
    cold = records["cold-study.json"]
    identity(cold)
    _require(cold.get("schema"), 1, "Invalid cold schema")
    bindings(cold, [("policy_sha256", "policy-frozen.json")])
    _number(cold["wall_seconds"], 180)
    rng, orders, expected_rows = random.Random(21667), [], []
    for repeat in range(3):
        order = list(METHODS); rng.shuffle(order); orders.append(order)
        expected_rows.extend((repeat, m) for m in order)
    _require(cold["orders"], orders, "Changed cold-process random order")
    cold_rows = cold["rows"]
    if not isinstance(cold_rows, list) or len(cold_rows) != 24:
        raise ValueError("Incomplete 24-process cold coverage")
    for row, (repeat, method) in zip(cold_rows, expected_rows):
        identity(row); _flags(row)
        _require(row.get("repeat"), repeat, "Changed cold repeat")
        _require(row.get("method"), method, "Changed cold process order")
        bindings(row, [("policy_sha256", "policy-frozen.json")])
        keys = ("runtime_setup_ms", "frozen_checks_ms", "policy_setup_ms", "input_prepare_ms", "first_application_sequence_ms")
        for key in (*keys, "process_wall_ms"):
            _number(row[key], 180_000)
        if sum(row[k] for k in keys) > row["process_wall_ms"]:
            raise ValueError("Cold component costs exceed process wall time")
        _count(row["peak_memory_bytes"], LIMITS["memory_bytes"])
    if sum(row["process_wall_ms"] for row in cold_rows) > cold["wall_seconds"] * 1000:
        raise ValueError("Cold process costs exceed total wall budget")
    cold_summary = {m: {"median_process_wall_ms": statistics.median(r["process_wall_ms"] for r in cold_rows if r["method"] == m),
                        "median_first_sequence_ms": statistics.median(r["first_application_sequence_ms"] for r in cold_rows if r["method"] == m)} for m in METHODS}
    passed = bool(rules) and rank_passed and app_passed
    baseline_sequence_us = {b: statistics.mean([v for s in sessions for v in s["application_samples_us"][b]]) for b in LIBRARIES}
    dispatcher_us = statistics.mean([v for s in sessions for v in s["application_samples_us"]["dispatcher"]])
    savings = min(baseline_sequence_us.values()) - dispatcher_us
    # One full confirmation sequence has four refresh+three-query tasks. Include
    # offline development and positive incremental median process startup cost.
    cold_extra_ms = max(0., cold_summary["dispatcher"]["median_process_wall_ms"] - min(cold_summary[b]["median_process_wall_ms"] for b in LIBRARIES))
    payback = math.ceil((development["search_wall_seconds"] * 1e6 + cold_extra_ms * 1000) / savings) if passed and savings > 0 else None
    over_fixed = all(s["gates"]["over_best_fixed"] for s in portfolio_sessions)
    over_per_shape = all(s["gates"]["over_per_shape"] for s in portfolio_sessions)
    two_configs = len(selection["joint_subset"]) == 2
    return {"schema": 1, "numerical_passed": True, "rank_passed": rank_passed, "application_passed": app_passed,
        "resource_passed": True, "confirmation_passed": passed, "source_sha256": source, "study_identity": implementation,
        "measurement_design": DESIGN, "uncertainty": UNCERTAINTY, "criterion": criterion,
        "environment": environment, "per_session": per_session, "comparisons": pooled, "cold_start": cold_summary,
        "evidence_sha256": hashes, "authored_source_bytes": source_bytes, "selection": selection,
        "search_cost": {"wall_seconds": development["search_wall_seconds"], "recorded_first_and_sample_seconds": measured_us / 1e6,
                        "authored_configurations": 2, "rank_arms": 9, "application_arms": 16},
        "calibration": {"wall_seconds": protocol["calibration_wall_seconds"],
            "recorded_first_and_sample_seconds": measured_calibration_us / 1e6,
            "rank_symmetric_envelope": rank_envelope, "application_symmetric_envelope": app_envelope},
        "sessions": session_spans, "cold_wall_seconds": cold["wall_seconds"],
        "fallback_counts": [s["fallback_counts"] for s in sessions],
        "portfolio": {"per_session": portfolio_sessions, "over_best_fixed": over_fixed,
            "over_per_shape": over_per_shape, "two_authored_configs": two_configs,
            "two_authored_kernel_advantage_over_fixed": two_configs and over_fixed,
            "two_authored_kernel_advantage_over_per_shape": two_configs and over_per_shape},
        "memory": {"limit_bytes": LIMITS["memory_bytes"], "cache_limit_bytes": LIMITS["cache_bytes"],
                   "maximum_observed_peak_bytes": max(protocol["peak_memory_bytes"], development["peak_memory_bytes"],
                       *[s["peak_memory_bytes"] for s in sessions], *[r["peak_memory_bytes"] for r in cold_rows])},
        "break_even": {"full_four_task_sequences": payback, "reliable_payback": False, "guaranteed": False,
                       "estimate_conditions_passed": passed and payback is not None,
                       "included_costs": "Development wall time plus positive incremental median cold-process startup cost",
                       "excluded_research_costs": "Calibration, independent numerical validation, confirmation execution and inter-session waiting",
                       "baseline_mean_sequence_us": baseline_sequence_us, "dispatcher_mean_sequence_us": dispatcher_us},
        "limitations": ["Local JSON provenance is not cryptographic GPU attestation.",
            "Every independent session must pass both canonical rank and all four complete-application library gates; pooled intervals are descriptive.",
            "Synchronized host wall includes dispatch, allocation and materialization; rank excludes normalization/GEMM.",
            "Fresh processes retain OS/driver caches; pure compiler time is not isolated.",
            "IID/block4/block8 envelopes are conservative sensitivity decisions, not guaranteed formal 95% coverage, stationarity or independence; longer dependence remains possible.",
            "Two-clock session separation and process IDs are provenance and operational separation, not proof of independent environments.",
            "Break-even uses mean confirmation savings and development plus incremental median cold-process cost; excludes freeze/validation/confirmation research costs and is not a guaranteed bound."]}


def verify_study(study: Path) -> dict[str, Any]:
    """Return separate gates for complete studies; reject malformed/changed data."""
    try:
        return _verify(Path(study))
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed or incomplete rank study") from exc


def _rename_new(source: Path, target: Path) -> None:
    """Atomic exclusive directory install, including against racing empty dirs."""
    if sys.platform == "win32":
        source.rename(target)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(source), os.fsencode(target), 4)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    else:
        raise RuntimeError("Atomic exclusive directory publication unavailable")
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(code, "Output appeared during export", str(target))
        raise OSError(code, os.strerror(code), str(target))


def export_artifact(study: Path, output: Path) -> dict[str, Any]:
    """Export only a nonempty strictly confirmed policy, preserving other paths."""
    from . import rank_search as rank
    study, output = Path(study), Path(output)
    if os.path.lexists(output):
        raise FileExistsError("Choose a new artifact directory")
    verified = verify_study(study)
    if verified["confirmation_passed"] is not True:
        raise ValueError("Frozen confirmation failed; preserve the null result")
    policy_content = _bytes(study / "policy-frozen.json")
    if hashlib.sha256(policy_content).hexdigest() != verified["evidence_sha256"]["policy-frozen.json"]:
        raise ValueError("Policy changed after verification")
    policy = copy.deepcopy(_decode(policy_content))
    for name, digest in verified["evidence_sha256"].items():
        if _sha(study / name) != digest:
            raise ValueError("Study changed during verification")
    if not policy["rules"]:
        raise ValueError("No confirmed custom rules")
    _source_snapshot(rank)
    policy["evidence"] = {**policy["evidence"], "confirmation_passed": True, "verification": verified["evidence_sha256"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kd-rank-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        for name, content in rank._SOURCES:
            (staging / name).write_bytes(content)
        (staging / "manifest.json").write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
        (staging / "verification.json").write_text(json.dumps(verified, indent=2, allow_nan=False) + "\n")
        if rank.RankPlan(staging).status != "installed":
            raise ValueError("Staged artifact failed data-only installed runtime guards")
        _rename_new(staging, output)
    return {"artifact": str(output), "confirmation_passed": True, "source_sha256": verified["source_sha256"],
            "bytes": sum(path.stat().st_size for path in output.iterdir())}
