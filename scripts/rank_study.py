#!/usr/bin/env python3
"""V2 application-primary fixed-top-five study. No stage runs on import.

Synchronized host-wall rank samples exclude norm/GEMM; application samples
include input transfers, one refresh and three executed changed-query searches.
Every stage is exclusive and failed/timeout evidence is preserved. This study
never revises either older study and never executes code from an artifact.
The new application entry control and squared symmetric noise floors amend
V1 prospectively; no earlier rejection is reinterpreted as a success.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EVIDENCE = ROOT / "evidence/workload-2026-11-01/rank-v2"
MEASUREMENT_DESIGN = "application_primary_symmetric_noise_v2"
UNCERTAINTY = {"resamples": 4000, "circular_block_lengths": [4, 8],
               "block_seed_stride": 100003, "primary_interval": "envelope_iid_block4_block8"}
METHODS = ("stable_library", "fast_library", "stable_reduction", "fast_reduction",
           "eager_library", "best_fixed", "per_shape", "dispatcher")
LIBRARIES = METHODS[:4]
CANONICAL_RANK_LIBRARIES = ("stable_library", "stable_reduction")
RANK_NAMES = ("simd32", "group128")
GLOBAL_NAMES = (*LIBRARIES, "eager_library", *RANK_NAMES)
ROUTED_CHOICES = (*LIBRARIES, *RANK_NAMES)
RANK_ARMS = (*GLOBAL_NAMES, *("plan:" + name for name in RANK_NAMES))
APP_ARMS = (*("global:" + name for name in GLOBAL_NAMES),
            *("routed:" + name for name in ROUTED_CHOICES),
            *("plan:" + name for name in RANK_NAMES), "plan_library")
MEASUREMENT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEBUG_FLAGS = ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "MTL_SHADER_VALIDATION_ENABLE_PIPELINES",
               "METAL_DEVICE_WRAPPER_TYPE", "METAL_DEBUG_ERROR_MODE", "MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS",
               "MTL_DEBUG_LAYER_VALIDATE_STORE_ACTIONS", "MTL_DEBUG_LAYER_VALIDATE_UNRETAINED_RESOURCES")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def reserve(stage, outcomes):
    """A persistent exclusive start marker prevents retry after partial failure."""
    for name in outcomes:
        if (EVIDENCE / name).exists():
            raise FileExistsError("Preserve previous study outcome: " + name)
    save(EVIDENCE / (stage + "-started.json"),
         {"stage": stage, "script_sha256": MEASUREMENT_SHA, "pid": os.getpid(),
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def current_implementation():
    import kerneldarwin.rank_search as rank
    import kerneldarwin.search as search
    import kerneldarwin.library_baselines as baselines
    import kerneldarwin.workload as workload
    import kerneldarwin.study_identity as binding
    pin = json.loads((Path(rank.__file__).parent / "rank-study-source.json").read_text())
    if pin.get("measurement_script_sha256") != MEASUREMENT_SHA or digest(__file__) != MEASUREMENT_SHA:
        raise ValueError("Rank measurement bytes differ from installed pin")
    for module, loaded in ((search, search._LOADED_APPLICATION_IDENTITY),
                           (baselines, baselines._LOADED_BASELINE_IDENTITY),
                           (binding, binding._LOADED_BINDING_IDENTITY)):
        if digest(module.__file__) != loaded:
            raise ValueError("Application/baseline/binding changed after import")
    def snapshot(module, files):
        return hashlib.sha256(Path(module.__file__).read_bytes() + b"".join(
            name.encode() + (Path(module.__file__).parent / "metal" / name).read_bytes()
            for name in files)).hexdigest()
    if (snapshot(workload, workload.SOURCE_FILES) != workload.source_identity()
            or snapshot(rank, rank.RANK_SOURCE_FILES) != rank.rank_source_identity()):
        raise ValueError("Runtime or MSL differs from loaded identity")
    return rank.rank_implementation_identity()


def source_metadata():
    import kerneldarwin.rank_search as rank
    directory = Path(rank.__file__).parent / "metal"
    sources = {name: digest(directory / name) for name in rank.RANK_SOURCE_FILES}
    if sources != {name: hashlib.sha256(value).hexdigest() for name, value in rank._SOURCES}:
        raise ValueError("Rank MSL changed after import")
    size = sum((directory / name).stat().st_size for name in sources)
    if size > 12 * 1024 or tuple(rank.RANK_CANDIDATES) != RANK_NAMES:
        raise ValueError("Prospective two-configuration/12 KiB limit violated")
    return sources, size


def timing_flags():
    names = set(DEBUG_FLAGS) | {key for key in os.environ
        if key.startswith(("MTL_DEBUG", "MTL_SHADER_VALIDATION"))}
    return {key: os.environ.get(key) for key in sorted(names)}


def setup():
    # Validation is deliberately a separate native process. Never time it.
    enabled = {key: value for key, value in timing_flags().items() if value is not None
               and value.lower() not in ("", "0", "false", "no", "off")}
    if enabled:
        raise RuntimeError("Metal debug/validation timing prohibited: " + ", ".join(sorted(enabled)))
    import ctypes
    code = ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(0x19, 0)
    if code:
        raise RuntimeError(f"Cannot set own current-thread QoS: {code}")
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(2 * 1024 ** 3)
    mx.set_cache_limit(256 * 1024 ** 2)
    return mx


def telemetry():
    result = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "load_average": os.getloadavg()}
    for command, key in ((["pmset", "-g", "therm"], "thermal"), (["pmset", "-g", "batt"], "power")):
        try:
            observed = subprocess.run(command, text=True, capture_output=True, timeout=10)
            result[key] = {"returncode": observed.returncode,
                           "observation": (observed.stdout + observed.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = {"unavailable": str(exc)}
    result["thermal_limit"] = "Observed power/thermal/load state; no direct GPU temperature measurement"
    return result


def corpus():
    path = ROOT / "evidence/workload-2026-10-29/corpus.json"
    return json.loads(path.read_text())


def arrays(task, texts, seed):
    import numpy as np
    from kerneldarwin.search import text_vectors
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    docs = [texts[order[i % len(order)]] + f" revision{seed % 17}" for i in range(task["rows"])]
    queries = [texts[order[(i * 7 + 3) % len(order)]] + f" query{seed % 23}" for i in range(task["queries"])]
    return np.array(text_vectors(docs, task["width"])), np.array(text_vectors(queries, task["width"]))


def layout_array(host, layout):
    import mlx.core as mx
    import numpy as np
    if layout == "column_stride2":
        backing = np.zeros((host.shape[0], host.shape[1] * 2), np.float32)
        backing[:, ::2] = host
        return mx.array(backing)[:, ::2]
    if layout == "transpose":
        return mx.array(host.T.copy()).T
    return mx.array(host)


def measure(function):
    import mlx.core as mx
    mx.synchronize()
    start = time.perf_counter_ns()
    output = function()
    # Rank graphs return a pair; a burst returns 128 pairs. Application outputs
    # are already host lists. Do not add a walk over all their scalar results.
    values = []
    if isinstance(output, mx.array):
        values = [output]
    elif isinstance(output, tuple) and all(isinstance(item, mx.array) for item in output):
        values = list(output)
    elif isinstance(output, list) and output and all(
            isinstance(pair, tuple) and len(pair) == 2
            and all(isinstance(item, mx.array) for item in pair) for pair in output):
        values = [item for pair in output for item in pair]
    if values:
        mx.eval(*values)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1000


def measure_burst(method, staged):
    if len(staged) != 128:
        raise ValueError("Exactly 128 distinct staged rank inputs required")
    return measure(lambda: [rank_call(method, *inputs) for inputs in staged]) / 128


def rank_inputs(task, documents, queries, count=128):
    """Raw query, stable refreshed docs and evaluated scores, all changed arrays."""
    import mlx.core as mx
    from kerneldarwin.search import _compiled_refresh
    from kerneldarwin.library_baselines import eager_normalize
    if count != 128:
        raise ValueError("Rank staging is frozen at 128")
    staged = []
    for i in range(count):
        docs, finite = _compiled_refresh("stable")(layout_array(documents + (i + 1) * 0.0007, task["layout"]))
        query = layout_array(queries + (i + 1) * 0.0011, "contiguous")
        scores = eager_normalize(query) @ docs.T
        mx.eval(query, docs, scores, finite)
        if not finite.item():
            raise ValueError("Nonfinite staged persistent documents")
        staged.append((query, docs, scores))
    return staged


def shape_of(task):
    return (task["queries"], task["rows"], task["width"], 5)


def within_bounds(query, docs):
    import mlx.core as mx
    from kerneldarwin.rank_search import _bounded
    return (query.dtype == docs.dtype == mx.float32
            and _bounded((query.shape[0], docs.shape[0], query.shape[1], 5)))


@lru_cache(maxsize=1)
def eager_rank():
    import mlx.core as mx
    def graph(scores):
        ids = mx.argsort(-scores, axis=-1)[:, :5]
        return ids, mx.take_along_axis(scores, ids, axis=-1)
    return graph


class StudyMethod:
    """Matched global or routed comparator; the dispatcher is actual RankPlan.

    Global libraries retain their own refresh. Fixed custom uses one bounded
    variant and stable refresh. Routed adapters use stable storage and select
    stable/fast sort/reduction queries or either variant, paying actual routing.
    Eager is descriptive and is excluded from acceptance/selection comparators.
    """
    def __init__(self, choice="stable_library", rules=None):
        if choice not in GLOBAL_NAMES:
            raise ValueError("Unknown study method")
        self.choice, self.rules = choice, rules
        self.calls = self.fallback_calls = 0

    def route(self, query, docs):
        from kerneldarwin.workload import _check
        _check(query)
        _check(docs)
        if query.shape[1] != docs.shape[1]:
            raise ValueError("Query/document width mismatch")
        self.calls += 1
        shape = (query.shape[0], docs.shape[0], query.shape[1], 5)
        name = self.rules.get(shape, "stable_library") if self.rules is not None else self.choice
        if name in RANK_NAMES and not within_bounds(query, docs):
            name = "stable_library"
        if name not in RANK_NAMES:
            self.fallback_calls += 1
        return name

    def index(self, vectors):
        from kerneldarwin.search import CompiledCosineIndex, CosineIndex
        from kerneldarwin.rank_search import ReductionIndex
        from kerneldarwin.library_baselines import eager_normalize
        if self.rules is None and self.choice in LIBRARIES:
            kind = "fast" if self.choice.startswith("fast") else "stable"
            return (ReductionIndex if self.choice.endswith("reduction") else CompiledCosineIndex)(vectors, kind)
        if self.rules is None and self.choice == "eager_library":
            return CosineIndex(vectors, eager_normalize)
        return study_index_class()(vectors, self)


@lru_cache(maxsize=1)
def study_index_class():
    from kerneldarwin.search import CompiledCosineIndex, _compiled_search, _search_inputs, _search_results
    from kerneldarwin.rank_search import compiled_query, compiled_reduction_query
    class StudyIndex(CompiledCosineIndex):
        def __init__(self, vectors, method):
            self.method = method
            super().__init__(vectors, "stable")
        def search(self, query, k=5):
            _search_inputs(query, self.vectors, k)
            if query.shape[0] == 0:
                return [], []
            name = self.method.route(query, self.vectors)
            if name in RANK_NAMES:
                graph = compiled_query(name)
            elif name.endswith("reduction"):
                graph = compiled_reduction_query("fast" if name.startswith("fast") else "stable")
            else:
                graph = _compiled_search("fast" if name == "fast_library" else "stable", k)
            return _search_results(graph(query, self.vectors))
    return StudyIndex


def rank_call(method, query, docs, scores):
    from kerneldarwin.rank_search import RankPlan, compiled_rank, library_rank, library_reduction_rank
    # Actual guards/counters run on actual query/docs, even though scores are staged.
    if isinstance(method, RankPlan):
        name = method._route(query, docs, 5)
    else:
        name = method.route(query, docs)
    if name in RANK_NAMES:
        return compiled_rank(name)(scores)
    if name in ("stable_reduction", "fast_reduction"):
        return library_reduction_rank()(scores)
    if name == "eager_library":
        return eager_rank()(scores)
    return library_rank()(scores)


def application(method, task, documents, queries):
    index = method.index(layout_array(documents, task["layout"]))
    return [index.search(layout_array(queries + j * 0.001, "contiguous"), k=5) for j in range(3)]


def float64_scores(documents, queries):
    import numpy as np
    def normalize(values):
        values = np.asarray(values, np.float64)
        peak = np.maximum(np.max(np.abs(values), axis=1, keepdims=True), 1e-12)
        scaled = values / peak
        return scaled / np.maximum(np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True)), 1e-12 / peak)
    return normalize(queries) @ normalize(documents).T


def check_application(output, documents, queries):
    import numpy as np
    if len(output) != 3:
        raise AssertionError("Three searches must actually execute")
    for j, (ids, values) in enumerate(output):
        reference = float64_scores(documents, queries + j * 0.001)
        ids, values = np.asarray(ids), np.asarray(values)
        if ids.shape != (queries.shape[0], 5) or values.shape != ids.shape:
            raise AssertionError("Wrong top-five result shape")
        if (not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
                or np.any(ids >= documents.shape[0]) or any(len(set(row.tolist())) != 5 for row in ids)):
            raise AssertionError("Invalid/duplicate document IDs")
        np.testing.assert_allclose(values, np.take_along_axis(reference, ids, axis=1), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(values, -np.sort(-reference, axis=1)[:, :5], rtol=2e-5, atol=2e-6)


def check_rank(output, scores):
    """Independent CPU stable FTZ keys and original selected float32 bits."""
    import numpy as np
    values = np.asarray(scores, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise AssertionError("Study staging must be finite; numeric suite owns nonfinite cases")
    bits = values.view(np.uint32)
    keys = values.copy()
    keys[np.abs(keys) < np.finfo(np.float32).tiny] = 0
    expected = np.argsort(-keys, axis=1, kind="stable")[:, :5]
    ids, selected = (np.asarray(item) for item in output)
    np.testing.assert_array_equal(ids, expected)
    np.testing.assert_array_equal(selected.astype(np.float32).view(np.uint32),
                                  np.take_along_axis(bits, expected, axis=1))


def intervals(baseline, candidate, seed=1404, count=4000):
    n = len(baseline)
    if not 40 <= n <= 1000 or len(candidate) != n or n % 8:
        raise ValueError("Need 40..1000 paired samples divisible by both four and eight")
    if count != UNCERTAINTY["resamples"]:
        raise ValueError("V2 requires exactly 4000 bootstrap resamples per family")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in [*baseline, *candidate]):
        raise ValueError("Every timing must be finite and positive")
    point = sum(baseline) / sum(candidate)
    families = {}
    if len(set(baseline)) == len(set(candidate)) == 1:
        families = {name: [point, point] for name in ("iid_ci95", "block4_ci95", "block8_ci95")}
    else:
        for block, name in ((1, "iid_ci95"), (4, "block4_ci95"), (8, "block8_ci95")):
            rng = random.Random(seed if block == 1 else seed + block * UNCERTAINTY["block_seed_stride"])
            samples = []
            for _ in range(count):
                if block == 1:
                    indices = rng.choices(range(n), k=n)
                else:
                    indices = [(start + offset) % n for start in
                               (rng.randrange(n) for _ in range(n // block)) for offset in range(block)]
                samples.append(sum(baseline[i] for i in indices) / sum(candidate[i] for i in indices))
            samples.sort()
            families[name] = [samples[100], samples[-101]]
    return {"speedup_total_cost": point,
            "ci95": [min(value[0] for value in families.values()), max(value[1] for value in families.values())],
            **families, "baseline_total_us": sum(baseline), "candidate_total_us": sum(candidate)}


def calibration_spec():
    return {"rank_order_seed": 21441, "application_order_seed": 21443,
            "application_input_seeds": list(range(21101, 21141)),
            "application_sequence_orders": [[0, 1, 2], [1, 2, 0], [2, 0, 1]] * 13 + [[0, 1, 2]],
            "negative_control_method": "stable_library", "application_first_warmup_order": [0, 1, 2],
            "shared_method_instance": True}


def symmetric_noise_floor(comparisons, minimum):
    envelope = max(max(1., row["ci95"][1], 1. / row["ci95"][0]) for row in comparisons)
    return max(minimum, envelope ** 2)


def verify_numerical():
    from kerneldarwin.rank_search import rank_source_identity
    from kerneldarwin.workload import environment_identity
    result = read("numerical-validation.json")
    if (result.get("kind") != "executed_rank_metal_numerical_validation"
            or result.get("success") is not True or result.get("gpu_executed") is not True
            or result.get("source_sha256") != rank_source_identity()
            or result.get("study_identity") != current_implementation()
            or result.get("environment") != environment_identity()
            or any(type(result.get(key)) is not int or result[key] != 0
                   for key in ("failures", "errors", "skipped"))
            or type(result.get("tests_run")) is not int or result["tests_run"] < 12):
        raise ValueError("Current independent executed rank Metal numerical report required")
    return result


def verify_frozen(policy=None):
    from kerneldarwin.rank_search import rank_source_identity
    from kerneldarwin.workload import environment_identity
    protocol = read("protocol.json")
    if (protocol.get("schema") != 2 or protocol.get("measurement_design") != MEASUREMENT_DESIGN
            or protocol.get("uncertainty") != UNCERTAINTY or protocol.get("calibration") != calibration_spec()
            or protocol["limits"].get("freeze_wall_seconds") != 120
            or protocol["confirmation"].get("minimum_separation_seconds") != 120
            or protocol["study_identity"] != current_implementation()
            or protocol["source_sha256"] != rank_source_identity()
            or protocol["environment"] != environment_identity()
            or protocol["script_sha256"] != MEASUREMENT_SHA
            or protocol["corpus_sha256"] != digest(EVIDENCE / "corpus.json")
            or protocol["corpus_sha256"] != digest(ROOT / "evidence/workload-2026-10-29/corpus.json")
            or protocol["sources"] != source_metadata()[0]
            or protocol["numerical_report_sha256"] != digest(EVIDENCE / "numerical-validation.json")):
        raise ValueError("Frozen source, environment, corpus or numerical evidence changed")
    verify_numerical()
    if policy is not None:
        if (policy["study_identity"] != protocol["study_identity"]
                or policy["source_sha256"] != protocol["source_sha256"]
                or policy["environment"] != protocol["environment"]
                or any(policy["evidence"][key] != value for key, value in {
                    "protocol_sha256": digest(EVIDENCE / "protocol.json"),
                    "corpus_sha256": protocol["corpus_sha256"],
                    "development_sha256": digest(EVIDENCE / "development.json"),
                    "numerical_report_sha256": digest(EVIDENCE / "numerical-validation.json")}.items())):
            raise ValueError("Frozen policy evidence changed")
    return protocol


def manifest(protocol, rules):
    return {"schema": 1, "operation": "stable_top5_pairs_v1", "epsilon": 1e-12,
            "fixed_k": 5, "key_policy": "native_ftz_stable_index_original_bits_v1",
            "compile_options": {"math_mode": "safe"},
            "source_sha256": protocol["source_sha256"], "study_identity": protocol["study_identity"],
            "environment": protocol["environment"], "sources": protocol["sources"], "rules": rules,
            "evidence": {"numerical_passed": True, "confirmation_passed": False}}


def rule(shape, name):
    return {"shape": list(shape), "candidate": name, "dtype": "float32", "layout": "any_mlx_strides"}


def trial_plan(protocol, rules):
    from kerneldarwin.rank_search import RankPlan
    return RankPlan._trial(manifest(protocol, rules))


def freeze():
    start = time.perf_counter()
    reserve("freeze", ("protocol.json", "rejected-baseline.json"))
    mx = None
    state = {"schema": 2, "measurement_design": MEASUREMENT_DESIGN,
             "script_sha256": MEASUREMENT_SHA, "baseline_noise": [], "stage": "setup",
             "timing_validation_flags": timing_flags()}
    def budget(stage):
        state["stage"] = stage
        elapsed = time.perf_counter() - start
        if elapsed > 120:
            raise TimeoutError("120-second calibration wall budget exhausted; no retry")
        if mx is not None and mx.get_peak_memory() > 2 * 1024 ** 3:
            raise RuntimeError("Calibration exceeded 2 GiB MLX peak memory")
        return elapsed
    try:
        state["telemetry_before"] = telemetry()
        mx = setup()
        from kerneldarwin.rank_search import rank_source_identity
        from kerneldarwin.workload import environment_identity
        implementation = current_implementation()
        verify_numerical()
        sources, size = source_metadata()
        state.update(source_sha256=rank_source_identity(), study_identity=implementation,
                     environment=environment_identity(), timing_validation_flags=timing_flags())
        budget("initial_identity_and_numerical_validation")
        data = corpus()
        n = len(data["chunks"])
        if n < 97:
            raise ValueError("Need at least 97 real licensed corpus chunks")
        tasks = [
            {"id": "notes", "rows": 64, "width": 384, "queries": 1, "layout": "contiguous"},
            {"id": "manuals", "rows": min(256, n), "width": 768, "queries": 8, "layout": "column_stride2"},
            {"id": "archive", "rows": n, "width": 1536, "queries": 17, "layout": "transpose"}]
        calibration = calibration_spec()
        rng = random.Random(calibration["rank_order_seed"])
        noise = state["baseline_noise"]
        # One instance throughout rank/application A and B; no alternative path.
        baseline = StudyMethod(calibration["negative_control_method"])
        for task in tasks:
            docs, query = arrays(task, data["chunks"], 21100)
            staged = rank_inputs(task, docs, query)
            active = {"task": task, "baseline_a_us": [], "baseline_b_us": [], "orders": []}
            state["partial_rank_control"] = active
            budget("rank_staging")
            active["first_rank_ms"] = measure(lambda: rank_call(baseline, *staged[0])) / 1000
            check_rank(rank_call(baseline, *staged[0]), staged[0][2])
            budget("rank_first_call_and_validation")
            for _ in range(5):
                measure_burst(baseline, staged)
                budget("rank_warmup")
            for _ in range(40):
                order = [0, 1]
                rng.shuffle(order)
                active["orders"].append(order)
                for arm in order:
                    active["baseline_a_us" if arm == 0 else "baseline_b_us"].append(measure_burst(baseline, staged))
                    budget("rank_negative_control")
            active["comparison"] = intervals(active["baseline_a_us"], active["baseline_b_us"])
            noise.append(active)
            del state["partial_rank_control"]
            budget("rank_control_analysis")
        # Do not exercise the held-out task or confirmation-only seeds here.
        prepared = [arrays(task, data["chunks"], 21100) for task in tasks]
        fixed_order = calibration["application_first_warmup_order"]
        app = {"baseline_a_us": [], "baseline_b_us": [], "orders": [],
               "input_seeds": calibration["application_input_seeds"],
               "sequence_orders": calibration["application_sequence_orders"],
               "method": "stable_library", "sequences_per_block": 16}
        state["application_baseline_noise"] = app
        budget("application_initial_input_staging")
        app["first_application_sequence_ms"] = measure(lambda: [
            application(baseline, tasks[i], *prepared[i]) for i in fixed_order]) / 1000
        for i in fixed_order:
            check_application(application(baseline, tasks[i], *prepared[i]), *prepared[i])
        budget("application_first_sequence_and_validation")
        for _ in range(5):
            measure(lambda: [[application(baseline, tasks[i], *prepared[i])
                              for i in fixed_order] for _ in range(16)]) / 16
            budget("application_warmup_block")
        rng = random.Random(calibration["application_order_seed"])
        for seed, sequence_order in zip(app["input_seeds"], app["sequence_orders"]):
            prepared = [arrays(task, data["chunks"], seed) for task in tasks]
            budget("application_changed_input_staging")
            order = [0, 1]
            rng.shuffle(order)
            app["orders"].append(order)
            for arm in order:
                elapsed = measure(lambda: [[application(baseline, tasks[i], *prepared[i])
                    for i in sequence_order] for _ in range(16)]) / 16
                app["baseline_a_us" if arm == 0 else "baseline_b_us"].append(elapsed)
                budget("application_negative_control")
        app["comparison"] = intervals(app["baseline_a_us"], app["baseline_b_us"])
        rank_floor = symmetric_noise_floor([row["comparison"] for row in noise], 1.10)
        app_floor = symmetric_noise_floor([app["comparison"]], 1.03)
        budget("control_analysis")
        observed = telemetry()
        state["telemetry"] = observed
        if current_implementation() != implementation or source_metadata() != (sources, size):
            raise ValueError("Source changed during calibration")
        verify_numerical()
        if any(value is not None and value.lower() not in ("", "0", "false", "no", "off")
               for value in timing_flags().values()):
            raise RuntimeError("Debug/validation timing flags changed during calibration")
        budget("final_source_resource_and_numerical_checks")
        if any(not 0.9 <= bound <= 1.1 for bound in app["comparison"]["ci95"]):
            raise RuntimeError("Application negative-control primary CI outside [0.9,1.1]; no candidate timing")
        save(EVIDENCE / "corpus.json", data)
        protocol = {"schema": 2, "target": "stable_top5_pairs_v1", "measurement_design": MEASUREMENT_DESIGN,
            "frozen_at": observed["time_utc"], "uncertainty": dict(UNCERTAINTY), "calibration": calibration,
            "application": "One index refresh plus three executed changed-query top5 searches per task",
            "tasks": tasks, "corpus_chunks": n, "corpus_sha256": digest(EVIDENCE / "corpus.json"),
            "weights": "Equal task frequency; sum three rank costs per task and complete application costs",
            "methods": list(METHODS), "libraries": list(LIBRARIES),
            "canonical_rank_libraries": list(CANONICAL_RANK_LIBRARIES),
            "global_names": list(GLOBAL_NAMES), "routed_choices": list(ROUTED_CHOICES),
            "rank_arms": list(RANK_ARMS), "application_arms": list(APP_ARMS),
            "development": {"seed": 21104, "repeats": 40, "order_seed": 21207,
                            "authored_configurations": 2, "rank_observations_per_round": 9,
                            "application_adapter_observations_per_round": 16},
            "measurement_batch": 128, "application_measurement_batch": 16,
            "confirmation": {"sessions": 3, "repeats_per_session": 40, "seeds": [23101, 24203, 25307],
                "sequence_orders": [[1, 0, 3, 2], [2, 3, 0, 1], [0, 2, 1, 3]],
                "minimum_separation_seconds": 120,
                "unseen_task": {"id": "unseen", "rows": 97, "width": 1025, "queries": 7, "layout": "column_stride2"}},
            "criterion": {"rank_ci95_lower_speedup": rank_floor,
                          "application_ci95_lower_speedup": app_floor,
                          "portfolio_over_fixed_minimum_total_cost_gain": 0.02,
                          "numerical_rtol": 2e-5, "numerical_atol": 2e-6},
            "limits": {"candidates_per_shape": 2, "portfolio_kernels": 2, "max_rules": 128,
                       "source_bytes": 12 * 1024, "development_wall_seconds": 900,
                       "freeze_wall_seconds": 120, "cold_study_wall_seconds": 180,
                       "memory_bytes": 2 * 1024 ** 3, "cache_bytes": 256 * 1024 ** 2},
            "selection": "Same two configs/observations: fixed global costs, per-shape routed costs, joint actual RankPlan costs including empty fallback; rank eligibility uses actual one-rule plan versus both canonical rank libraries",
            "comparator_domain": "Global libraries own stable/fast refresh; fixed custom stable refresh; routed stable storage with stable/fast sort/reduction or custom query. Eager descriptive only. Dispatcher actual RankPlan/RankIndex.",
            "stability_gate": "Complete three-development-task application16 negative control primary CI in [0.9,1.1]; rank-only entry cap relaxed prospectively; squared symmetric noise floors retain minimum 1.10/1.03",
            "timing": "Synchronized host wall us; rank128 evaluated inputs excludes normalization/GEMM and forces IDs+values; application16 retains nested host outputs and includes transfers, allocations, refresh and three changed searches per task",
            "qos": "user_initiated_current_thread_only",
            "confirmation_rule": "No retries/retuning or pooled rescue; primary library gates every session; separate processes at least120 seconds apart by both clocks; no favorable-load scheduling",
            "environment": environment_identity(), "telemetry_before": state["telemetry_before"],
            "telemetry": observed, "baseline_noise": noise, "application_baseline_noise": app,
            "timing_validation_flags": timing_flags(), "peak_memory_bytes": mx.get_peak_memory(),
            "source_sha256": rank_source_identity(), "study_identity": implementation,
            "sources": sources, "source_bytes": size, "script_sha256": MEASUREMENT_SHA,
            "numerical_report_sha256": digest(EVIDENCE / "numerical-validation.json"),
            "source_budget_scope": "All three authored MSL files including shared header"}
        protocol["calibration_wall_seconds"] = budget("final_before_protocol_save")
        save(EVIDENCE / "protocol.json", protocol)
        print(json.dumps({"required_rank_speedup": rank_floor, "required_application_speedup": app_floor}))
    except Exception as exc:
        state.update(reason=type(exc).__name__ + ": " + str(exc),
                     calibration_wall_seconds=time.perf_counter() - start,
                     peak_memory_bytes=mx.get_peak_memory() if mx is not None else None)
        if "telemetry" not in state:
            state["telemetry"] = telemetry()
        save(EVIDENCE / "rejected-baseline.json", state)
        raise


def development_arms(protocol, task):
    globals_ = {name: StudyMethod(name) for name in GLOBAL_NAMES}
    routed = {name: StudyMethod(rules={shape_of(task): name}) for name in ROUTED_CHOICES}
    plans = {"plan:" + name: trial_plan(protocol, [rule(shape_of(task), name)]) for name in RANK_NAMES}
    rank = {**globals_, **plans}
    app = {**{"global:" + name: method for name, method in globals_.items()},
           **{"routed:" + name: method for name, method in routed.items()},
           **plans, "plan_library": trial_plan(protocol, [])}
    assert tuple(rank) == RANK_ARMS and tuple(app) == APP_ARMS
    return rank, app


def select_policy(protocol, rows):
    """Pure cost replay, also used by independent synthetic checks."""
    cost = lambda row, label: statistics.mean(row["application_samples_us"][label])
    eligible = lambda row, name: all(row["comparisons"][name][base]["rank"]["ci95"][0]
        > protocol["criterion"]["rank_ci95_lower_speedup"] for base in CANONICAL_RANK_LIBRARIES)
    fixed = min((*LIBRARIES, *RANK_NAMES), key=lambda name: sum(cost(row, "global:" + name) for row in rows))
    per_shape = {tuple(row["shape"]): min(ROUTED_CHOICES, key=lambda name: cost(row, "routed:" + name)) for row in rows}
    choices = []
    for count in (0, 1, 2):
        for subset in itertools.combinations(RANK_NAMES, count):
            rules, total = {}, 0.0
            for row in rows:
                options = ["plan_library", *("plan:" + name for name in subset if eligible(row, name))]
                best = min(options, key=lambda label: cost(row, label))
                total += cost(row, best)
                if best != "plan_library":
                    rules[tuple(row["shape"])] = best.removeprefix("plan:")
            choices.append({"subset": list(subset), "total_us": total, "rules": rules})
    best_one = min((item for item in choices if len(item["subset"]) <= 1), key=lambda item: item["total_us"])
    joint = min(choices, key=lambda item: item["total_us"])
    if joint["total_us"] > best_one["total_us"] * (1 - protocol["criterion"]["portfolio_over_fixed_minimum_total_cost_gain"]):
        joint = best_one
    return joint["rules"], {"best_fixed": fixed, "joint_subset": sorted(set(joint["rules"].values())),
        "development_application_cost_us": joint["total_us"],
        "best_one_subset_application_cost_us": best_one["total_us"],
        "best_fixed_application_cost_us": sum(cost(row, "global:" + fixed) for row in rows),
        "development_per_shape_application_cost_us": sum(cost(row, "routed:" + per_shape[tuple(row["shape"])]) for row in rows),
        "per_shape": [{"shape": list(shape), "candidate": name} for shape, name in sorted(per_shape.items())]}


def tune():
    start = time.perf_counter()
    reserve("tune", ("development.json", "development-timeout.json", "policy-frozen.json"))
    mx = setup()
    protocol = verify_frozen()
    rows, active = [], {}
    def budget(stage):
        elapsed = time.perf_counter() - start
        if elapsed > protocol["limits"]["development_wall_seconds"]:
            save(EVIDENCE / "development-timeout.json", {"completed_rows": rows, "partial": active,
                "elapsed_seconds": elapsed, "stage": stage, "source_sha256": protocol["source_sha256"],
                "study_identity": current_implementation(), "protocol_sha256": digest(EVIDENCE / "protocol.json")})
            raise TimeoutError("900-second development budget exhausted; no retry")
        return elapsed
    texts = read("corpus.json")["chunks"]
    rng = random.Random(protocol["development"]["order_seed"])
    budget("setup_and_validation")
    for task in protocol["tasks"]:
        rank_arms, app_arms = development_arms(protocol, task)
        docs, query = arrays(task, texts, protocol["development"]["seed"])
        staged = rank_inputs(task, docs, query)
        active = {"task": task, "shape": list(shape_of(task)), "weight": 1,
            "rank_samples_us": {name: [] for name in rank_arms},
            "application_samples_us": {name: [] for name in app_arms},
            "rank_orders": [], "application_orders": [], "first_rank_in_development_ms": {},
            "first_application_in_development_ms": {}, "rank_numerical_checks": {},
            "application_numerical_checks": {}}
        budget("staging")
        for name, method in rank_arms.items():
            active["first_rank_in_development_ms"][name] = measure(lambda m=method: rank_call(m, *staged[0])) / 1000
            check_rank(rank_call(method, *staged[0]), staged[0][2])
            active["rank_numerical_checks"][name] = {"ids_passed": True, "original_bits_passed": True}
            budget("first_rank_and_validation")
            for _ in range(5):
                measure_burst(method, staged)
                budget("rank_warmup")
        for name, method in app_arms.items():
            active["first_application_in_development_ms"][name] = measure(lambda m=method: application(m, task, docs, query)) / 1000
            check_application(application(method, task, docs, query), docs, query)
            active["application_numerical_checks"][name] = {"scores_passed": True, "ids_association_passed": True}
            budget("first_application_and_validation")
            for _ in range(5):
                measure(lambda m=method: application(m, task, docs, query))
                budget("application_warmup")
        for repeat in range(40):
            docs, query = arrays(task, texts, protocol["development"]["seed"] + repeat + 1)
            staged = rank_inputs(task, docs, query)
            budget("changed_input_staging")
            names = list(rank_arms)
            rng.shuffle(names)
            active["rank_orders"].append(names)
            for name in names:
                active["rank_samples_us"][name].append(measure_burst(rank_arms[name], staged))
                budget("rank_measurement")
            labels = list(app_arms)
            rng.shuffle(labels)
            active["application_orders"].append(labels)
            for label in labels:
                active["application_samples_us"][label].append(measure(lambda m=app_arms[label]: application(m, task, docs, query)))
                budget("application_measurement")
        active["comparisons"] = {name: {base: {"rank": intervals(active["rank_samples_us"][base],
            active["rank_samples_us"]["plan:" + name])} for base in CANONICAL_RANK_LIBRARIES} for name in RANK_NAMES}
        active["first_call_note"] = "Individual first calls; preceding arms may share compiler caches; process cold measured separately"
        active["rank_input_layout"] = "Raw query, stable compiled refresh output, separately evaluated changed score matrix; actual MLX strides"
        rows.append(active)
        active = {}
        budget("row_analysis")
        print("development", task["id"], "complete", flush=True)
    rules, selection = select_policy(protocol, rows)
    development = {"schema": 1, "source_sha256": protocol["source_sha256"],
        "study_identity": current_implementation(), "environment": protocol["environment"],
        "protocol_sha256": digest(EVIDENCE / "protocol.json"), "corpus_sha256": protocol["corpus_sha256"],
        "numerical_report_sha256": protocol["numerical_report_sha256"], "rows": rows,
        "peak_memory_bytes": mx.get_peak_memory(), "telemetry": telemetry(),
        "timing_validation_flags": timing_flags(),
        "budget": "Same two authored configurations, nine rank/sixteen app arms; staging, individual first calls, validation and analysis count in 900 seconds"}
    development["search_wall_seconds"] = budget("final_before_development_save")
    save(EVIDENCE / "development.json", development)
    policy = manifest(protocol, [rule(shape, name) for shape, name in sorted(rules.items())])
    policy["policy_selection"] = selection
    policy["evidence"].update({"corpus_sha256": protocol["corpus_sha256"],
        "protocol_sha256": digest(EVIDENCE / "protocol.json"),
        "development_sha256": digest(EVIDENCE / "development.json"),
        "numerical_report_sha256": protocol["numerical_report_sha256"]})
    budget("final_before_policy_save")
    save(EVIDENCE / "policy-frozen.json", policy)
    print(json.dumps(policy, indent=2))


def policies(policy, requested=None):
    from kerneldarwin.rank_search import RankPlan
    result = {}
    for name in METHODS if requested is None else (requested,):
        if name == "best_fixed":
            result[name] = StudyMethod(policy["policy_selection"]["best_fixed"])
        elif name == "per_shape":
            result[name] = StudyMethod(rules={tuple(row["shape"]): row["candidate"] for row in policy["policy_selection"]["per_shape"]})
        elif name == "dispatcher":
            before = time.perf_counter()
            result[name] = RankPlan._trial(policy)
            result[name].artifact_load_ms = (time.perf_counter() - before) * 1000
        else:
            result[name] = StudyMethod(name)
    return result


def session_preflight(session, policy, protocol, start_unix_ns, start_monotonic_ns):
    """Pure file/clock checks, before reserving a session or doing GPU work."""
    if session == 0:
        return
    previous = read(f"confirmation-{session - 1}.json")
    expected = {"schema": 1, "session": session - 1, "source_sha256": protocol["source_sha256"],
                "study_identity": protocol["study_identity"], "environment": protocol["environment"],
                "corpus_sha256": protocol["corpus_sha256"],
                "protocol_sha256": digest(EVIDENCE / "protocol.json"),
                "policy_sha256": digest(EVIDENCE / "policy-frozen.json"),
                "development_sha256": digest(EVIDENCE / "development.json"),
                "numerical_report_sha256": protocol["numerical_report_sha256"],
                "seed": protocol["confirmation"]["seeds"][session - 1],
                "sequence_order": protocol["confirmation"]["sequence_orders"][session - 1]}
    if (any(previous.get(key) != value for key, value in expected.items())
            or policy["study_identity"] != protocol["study_identity"]
            or type(previous.get("process_id")) is not int or previous["process_id"] <= 0):
        raise ValueError("Previous session identity/provenance differs from the frozen study")
    separation = protocol["confirmation"]["minimum_separation_seconds"] * 1_000_000_000
    for clock, current in (("unix", start_unix_ns), ("monotonic", start_monotonic_ns)):
        start = previous.get(f"session_start_{clock}_ns")
        end = previous.get(f"session_end_{clock}_ns")
        if type(start) is not int or type(end) is not int or start <= 0 or end <= start:
            raise ValueError("Previous session must have positive spans on both clocks")
        if current - end < separation:
            raise RuntimeError("Confirmation sessions require 120 seconds separation on both clocks; no marker created")


def confirm(session):
    if session not in (0, 1, 2):
        raise ValueError("Session must be 0, 1 or 2")
    start_unix_ns, start_monotonic_ns = time.time_ns(), time.monotonic_ns()
    policy = read("policy-frozen.json")
    session_preflight(session, policy, read("protocol.json"), start_unix_ns, start_monotonic_ns)
    protocol = verify_frozen(policy)
    reserve(f"confirmation-{session}", (f"confirmation-{session}.json",))
    mx = setup()
    seed = protocol["confirmation"]["seeds"][session]
    order = protocol["confirmation"]["sequence_orders"][session]
    tasks = [*protocol["tasks"], protocol["confirmation"]["unseen_task"]]
    texts, methods = read("corpus.json")["chunks"], policies(policy)
    prepared = [arrays(task, texts, seed) for task in tasks]
    before, first_rank, first_app, checks = telemetry(), {}, {}, []
    for name, method in methods.items():
        first_rank[name], first_app[name] = [], []
        for i, task in enumerate(tasks):
            staged = rank_inputs(task, *prepared[i])
            first_rank[name].append(measure(lambda m=method: rank_call(m, *staged[0])) / 1000)
            check_rank(rank_call(method, *staged[0]), staged[0][2])
            first_app[name].append(measure(lambda m=method: application(m, task, *prepared[i])) / 1000)
            check_application(application(method, task, *prepared[i]), *prepared[i])
            checks.append({"task": task["id"], "method": name, "rank_ids_passed": True,
                           "rank_original_bits_passed": True, "scores_passed": True, "ids_association_passed": True})
    ranks, apps = {name: [] for name in methods}, {name: [] for name in methods}
    rank_orders, app_orders, rng = [], [], random.Random(seed)
    for repeat in range(40):
        prepared = [arrays(task, texts, seed + repeat + 1) for task in tasks]
        staged = [rank_inputs(task, *prepared[i]) for i, task in enumerate(tasks)]
        names = list(methods)
        rng.shuffle(names)
        rank_orders.append(names)
        for name in names:
            ranks[name].append(sum(3 * measure_burst(methods[name], staged[i]) for i in order))
        names = list(methods)
        rng.shuffle(names)
        app_orders.append(names)
        for name in names:
            apps[name].append(measure(lambda m=methods[name]: [[application(m, tasks[i], *prepared[i])
                for i in order] for _ in range(16)]) / 16)
        if (repeat + 1) % 10 == 0:
            print("session", session, "repeat", repeat + 1, flush=True)
    comparisons = {name: {base: {"rank": intervals(ranks[base], ranks[name], seed),
        "application": intervals(apps[base], apps[name], seed)} for base in LIBRARIES} for name in METHODS}
    gates = {"rank": all(comparisons["dispatcher"][base]["rank"]["ci95"][0]
             > protocol["criterion"]["rank_ci95_lower_speedup"] for base in CANONICAL_RANK_LIBRARIES),
             "application": all(comparisons["dispatcher"][base]["application"]["ci95"][0]
             > protocol["criterion"]["application_ci95_lower_speedup"] for base in LIBRARIES)}
    portfolio = {name: intervals(apps[name], apps["dispatcher"], seed) for name in ("best_fixed", "per_shape")}
    portfolio_gate = {"over_best_fixed": portfolio["best_fixed"]["ci95"][0] > 1.02,
                      "over_per_shape": portfolio["per_shape"]["ci95"][0] > 1.00,
                      "two_authored_configs": len(set(policy["policy_selection"]["joint_subset"])) == 2}
    result = {"schema": 1, "session": session, "seed": seed, "sequence_order": order,
        "session_start_unix_ns": start_unix_ns, "session_start_monotonic_ns": start_monotonic_ns,
        "process_id": os.getpid(),
        "source_sha256": policy["source_sha256"], "study_identity": current_implementation(),
        "environment": protocol["environment"], "corpus_sha256": protocol["corpus_sha256"],
        "protocol_sha256": digest(EVIDENCE / "protocol.json"), "policy_sha256": digest(EVIDENCE / "policy-frozen.json"),
        "development_sha256": digest(EVIDENCE / "development.json"),
        "numerical_report_sha256": protocol["numerical_report_sha256"],
        "telemetry_before": before, "telemetry_after": telemetry(), "timing_validation_flags": timing_flags(),
        "first_rank_in_session_ms": first_rank, "first_application_in_session_ms": first_app,
        "artifact_load_ms": methods["dispatcher"].artifact_load_ms,
        "cold_note": "Individual first calls; preceding methods share possible caches; cold-study uses fresh processes",
        "rank_samples_us": ranks, "application_samples_us": apps,
        "rank_orders": rank_orders, "application_orders": app_orders, "checks": checks,
        "peak_memory_bytes": mx.get_peak_memory(), "comparisons": comparisons, "session_gate": gates,
        "portfolio_comparisons": portfolio, "portfolio_session_gate": portfolio_gate,
        "fallback_count_scope": "Calls across validation, rank bursts and application searches; global libraries count rank adapter calls but app calls bypass their adapter",
        "fallback_counts": {name: {"calls": method.calls, "fallback_calls": method.fallback_calls} for name, method in methods.items()}}
    result.update(session_end_unix_ns=time.time_ns(), session_end_monotonic_ns=time.monotonic_ns())
    if (result["session_end_unix_ns"] <= start_unix_ns
            or result["session_end_monotonic_ns"] <= start_monotonic_ns):
        raise RuntimeError("Confirmation session must have positive spans on both clocks")
    save(EVIDENCE / f"confirmation-{session}.json", result)
    print(json.dumps({"session_gate": gates, "dispatcher": comparisons["dispatcher"]}))


def cold(method):
    before = time.perf_counter()
    mx = setup()
    runtime_ms = (time.perf_counter() - before) * 1000
    policy = read("policy-frozen.json")
    before = time.perf_counter()
    protocol = verify_frozen(policy)
    probe_ms = (time.perf_counter() - before) * 1000
    before = time.perf_counter()
    selected = policies(policy, method)[method]
    load_ms = (time.perf_counter() - before) * 1000
    before = time.perf_counter()
    tasks = [*protocol["tasks"], protocol["confirmation"]["unseen_task"]]
    texts = read("corpus.json")["chunks"]
    prepared = [arrays(task, texts, 26661) for task in tasks]
    input_ms = (time.perf_counter() - before) * 1000
    first = measure(lambda: [application(selected, task, *prepared[i]) for i, task in enumerate(tasks)]) / 1000
    print(json.dumps({"method": method, "source_sha256": policy["source_sha256"],
        "study_identity": current_implementation(), "environment": protocol["environment"],
        "corpus_sha256": protocol["corpus_sha256"], "policy_sha256": digest(EVIDENCE / "policy-frozen.json"),
        "runtime_setup_ms": runtime_ms, "frozen_checks_ms": probe_ms, "policy_setup_ms": load_ms,
        "input_prepare_ms": input_ms, "first_application_sequence_ms": first,
        "peak_memory_bytes": mx.get_peak_memory(), "timing_validation_flags": timing_flags(),
        "cache_note": "Fresh Python/MLX process; OS/driver caches retained; compilation included, pure compiler time not isolated"}))


def cold_study():
    start = time.perf_counter()
    reserve("cold-study", ("cold-study.json", "cold-study-timeout.json", "cold-study-failure.json"))
    policy = read("policy-frozen.json")
    protocol = verify_frozen(policy)
    rng, rows, orders = random.Random(21667), [], []
    def budget():
        elapsed = time.perf_counter() - start
        if elapsed >= protocol["limits"]["cold_study_wall_seconds"]:
            save(EVIDENCE / "cold-study-timeout.json", {"rows": rows, "orders": orders, "wall_seconds": elapsed})
            raise TimeoutError("180-second cold-study budget exhausted")
        return protocol["limits"]["cold_study_wall_seconds"] - elapsed
    for repeat in range(3):
        order = list(METHODS)
        rng.shuffle(order)
        orders.append(order)
        for method in order:
            remaining = budget()
            before = time.perf_counter()
            try:
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "cold", "--method", method,
                    "--directory", str(EVIDENCE)], cwd=ROOT, text=True, capture_output=True, timeout=min(60, remaining))
            except subprocess.TimeoutExpired as exc:
                save(EVIDENCE / "cold-study-timeout.json", {"rows": rows, "orders": orders,
                    "method": method, "repeat": repeat, "wall_seconds": time.perf_counter() - start,
                    "stderr_tail": str(exc.stderr or "")[-3000:]})
                raise
            if child.returncode:
                save(EVIDENCE / "cold-study-failure.json", {"method": method, "repeat": repeat,
                    "returncode": child.returncode, "stderr_tail": child.stderr[-3000:], "completed_rows": rows})
                raise RuntimeError("Cold process failed: " + child.stderr[-1500:])
            row = json.loads(child.stdout)
            row.update(repeat=repeat, process_wall_ms=(time.perf_counter() - before) * 1000)
            rows.append(row)
            budget()
            print("cold", repeat, method, flush=True)
    result = {"schema": 1, "source_sha256": policy["source_sha256"], "study_identity": current_implementation(),
        "corpus_sha256": protocol["corpus_sha256"], "policy_sha256": digest(EVIDENCE / "policy-frozen.json"),
        "environment": protocol["environment"], "rows": rows, "orders": orders, "telemetry": telemetry(),
        "method": "24 serialized fresh processes, eight methods x three; randomized orders; OS/driver caches retained"}
    result["wall_seconds"] = protocol["limits"]["cold_study_wall_seconds"] - budget()
    save(EVIDENCE / "cold-study.json", result)


def main():
    global EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "tune", "confirm", "cold", "cold-study"))
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--method", choices=METHODS, default="dispatcher")
    parser.add_argument("--directory", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    EVIDENCE = args.directory.resolve()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "tune":
        tune()
    elif args.stage == "confirm":
        confirm(args.session)
    elif args.stage == "cold":
        cold(args.method)
    else:
        cold_study()


if __name__ == "__main__":
    main()
