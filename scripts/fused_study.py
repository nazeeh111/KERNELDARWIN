#!/usr/bin/env python3
"""Frozen second-target study: query normalization fused with similarity scores.

This file is staged until the first target's unchanged study is finished. Host
wall samples include dispatch/materialization; they are not GPU-only timers.
Each stage writes a new record and refuses to overwrite any previous outcome.
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
EVIDENCE = ROOT / "evidence/workload-2026-10-29/fused-v1"
METHODS = ("stable_library", "fast_library", "eager_library", "best_fixed", "per_shape", "dispatcher")
LIBRARIES = ("stable_library", "fast_library")
SCORE_NAMES = ("score_simd", "score_tile4")
ROUTED_CHOICES = (*LIBRARIES, *SCORE_NAMES)
MEASUREMENT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The exclusive open also rejects concurrent duplicate stage invocations.
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def current_implementation():
    import kerneldarwin.fused_search as fused
    pin = json.loads((Path(fused.__file__).parent / "fused-study-source.json").read_text())
    if pin.get("measurement_script_sha256") != MEASUREMENT_SHA or digest(__file__) != MEASUREMENT_SHA:
        raise ValueError("Fused measurement script differs from installed pin; rebuild before a new study")
    return fused.fused_implementation_identity()


def source_metadata():
    import kerneldarwin.fused_search as fused
    directory = Path(fused.__file__).parent / "metal"
    sources = {name: digest(directory / name) for name in fused.SCORE_SOURCE_FILES}
    if sources != {name: hashlib.sha256(value).hexdigest() for name, value in fused._SOURCES}:
        raise ValueError("Source bytes changed after import; restart before a new study")
    return sources, sum((directory / name).stat().st_size for name in sources)


def setup():
    import ctypes
    system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    code = system.pthread_set_qos_class_self_np(0x19, 0)
    if code:
        raise RuntimeError(f"Cannot set own current-thread QoS: {code}")
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(2 * 1024 ** 3)
    mx.set_cache_limit(256 * 1024 ** 2)
    return mx


def telemetry():
    result = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load_average": os.getloadavg()}
    for command, key in ((["pmset", "-g", "therm"], "thermal"), (["pmset", "-g", "batt"], "power")):
        observed = subprocess.run(command, text=True, capture_output=True, timeout=10)
        result[key] = (observed.stdout + observed.stderr).strip()
    result["thermal_limit"] = "pmset observations are not direct GPU temperature measurements"
    return result


def corpus():
    path = ROOT / "evidence/workload-2026-10-29/corpus.json"
    if not path.exists():
        raise FileNotFoundError("Copy the original licensed documentation corpus; do not regenerate from changed docs")
    return json.loads(path.read_text())


def arrays(task, texts, seed):
    import numpy as np
    from kerneldarwin.search import text_vectors
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    documents = [texts[order[i % len(order)]] + f" revision{seed % 17}" for i in range(task["rows"])]
    queries = [texts[order[(i * 7 + 3) % len(order)]] + f" query{seed % 23}" for i in range(task["queries"])]
    return np.array(text_vectors(documents, task["width"])), np.array(text_vectors(queries, task["width"]))


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
    if isinstance(output, mx.array):
        mx.eval(output)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1000


def measure_burst(function, staged):
    import mlx.core as mx
    return measure(lambda: mx.eval(*[function(q, docs) for q, docs in staged])) / len(staged)


def score_inputs(task, documents, queries, count):
    """Use the actual compiled-refresh output layout for persistent score input.

    Index refresh is included independently in every full application sample.
    Each burst has distinct changed evaluated inputs, never cached answers.
    """
    import mlx.core as mx
    from kerneldarwin.search import _compiled_refresh
    result = []
    for i in range(count):
        docs, finite = _compiled_refresh("stable")(layout_array(documents + (i + 1) * 0.0007, task["layout"]))
        mx.eval(docs, finite)
        if not finite.item():
            raise ValueError("Nonfinite persistent documents in staged input")
        query = layout_array(queries + (i + 1) * 0.0011, "contiguous")
        mx.eval(query, docs)
        result.append((query, docs))
    return result


@lru_cache(maxsize=1)
def eager_scores():
    from kerneldarwin.library_baselines import eager_normalize
    return lambda queries, docs: eager_normalize(queries) @ docs.T


def within_bounds(queries, docs):
    import mlx.core as mx
    return (queries.dtype == docs.dtype == mx.float32 and queries.ndim == docs.ndim == 2
            and 0 < queries.shape[0] <= 128 and 0 < docs.shape[0] <= 2048
            and 0 < queries.shape[1] == docs.shape[1] <= 2048
            and queries.shape[0] * docs.shape[0] * queries.shape[1] <= 16 * 1024 ** 2)


class StudyMethod:
    """Baseline or bounded fixed/per-shape comparator, using the same app class.

    The final dispatcher is a real FusedPlan, not this research comparator.
    Library routes use whole-query compiled graphs in application timing.
    """
    def __init__(self, choice="stable_library", rules=None):
        self.choice = choice
        self.rules = rules
        self.calls = 0
        self.fallback_calls = 0

    def route(self, queries, docs):
        from kerneldarwin.workload import _check
        _check(queries)
        _check(docs)
        if queries.shape[1] != docs.shape[1]:
            raise ValueError("Query and document widths must match")
        self.calls += 1
        key = (queries.shape[0], docs.shape[0], queries.shape[1])
        name = self.rules.get(key, "stable_library") if self.rules is not None else self.choice
        if name in SCORE_NAMES and not within_bounds(queries, docs):
            name = "stable_library"
        if name not in SCORE_NAMES:
            self.fallback_calls += 1
        return name

    def score(self, queries, docs):
        from kerneldarwin.fused_search import _compiled_candidate_scores, library_scores
        name = self.route(queries, docs)
        if name in SCORE_NAMES:
            return _compiled_candidate_scores(name)(queries, docs)
        if name == "eager_library":
            return eager_scores()(queries, docs)
        return library_scores(queries, docs, "fast" if name == "fast_library" else "stable")

    def index(self, vectors):
        from kerneldarwin.search import CompiledCosineIndex, CosineIndex
        from kerneldarwin.library_baselines import eager_normalize
        # Global library comparators retain optimized refresh as well as query.
        if self.rules is None and self.choice in LIBRARIES:
            return CompiledCosineIndex(vectors, "fast" if self.choice == "fast_library" else "stable")
        if self.rules is None and self.choice == "eager_library":
            return CosineIndex(vectors, eager_normalize)
        # Research comparators receive the same optimized whole-query fallback.
        # The dispatcher below is separately measured through the real public
        # FusedPlan adapter, including its exact source/env/rule guards.
        return study_index_class()(vectors, self)

    def __call__(self, queries, docs):
        return self.score(queries, docs)


@lru_cache(maxsize=1)
def study_index_class():
    # Lazy import keeps CPU-only script inspection free of Metal setup. Cache
    # the class itself, so a comparator does not pay class-construction cost on
    # every application refresh.
    from kerneldarwin.fused_search import FusedCosineIndex, _compiled_score_search
    from kerneldarwin.search import _compiled_search, _search_inputs, _search_results
    class StudyIndex(FusedCosineIndex):
        def search(self, queries, k=5):
            _search_inputs(queries, self.vectors, k)
            if queries.shape[0] == 0:
                return [], []
            name = self.scorer.route(queries, self.vectors)
            if name in LIBRARIES:
                kind = "fast" if name == "fast_library" else "stable"
                return _search_results(_compiled_search(kind, k)(queries, self.vectors))
            return _search_results(_compiled_score_search(name, k)(queries, self.vectors))
    return StudyIndex


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


def float64_score_operation(queries, normalized_documents):
    """The low-level score contract does not renormalize persistent documents."""
    import numpy as np
    q = np.asarray(queries, np.float64)
    peak = np.maximum(np.max(np.abs(q), axis=1, keepdims=True), 1e-12)
    scaled = q / peak
    normalized = scaled / np.maximum(np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True)), 1e-12 / peak)
    return normalized @ np.asarray(normalized_documents, np.float64).T


def check_application(output, documents, queries):
    import numpy as np
    for j, (ids, values) in enumerate(output):
        reference = float64_scores(documents, queries + j * 0.001)
        ids = np.asarray(ids)
        values = np.asarray(values)
        if ids.shape != (queries.shape[0], 5) or values.shape != ids.shape:
            raise AssertionError("Wrong top-k result shape")
        if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= documents.shape[0]):
            raise AssertionError("Invalid top-k document ids")
        if any(len(set(row.tolist())) != 5 for row in ids):
            raise AssertionError("Repeated top-k document id")
        associated = np.take_along_axis(reference, ids, axis=1)
        np.testing.assert_allclose(values, associated, rtol=2e-5, atol=2e-6)
        expected = -np.sort(-reference, axis=1)[:, :5]
        np.testing.assert_allclose(values, expected, rtol=2e-5, atol=2e-6)


def intervals(baseline, candidate, seed=1404, count=4000):
    if not 40 <= len(baseline) <= 1000 or len(candidate) != len(baseline):
        raise ValueError("Need 40..1000 paired samples")
    if any(not math.isfinite(v) or v <= 0 for v in [*baseline, *candidate]):
        raise ValueError("Every timing must be finite and positive")
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        ids = rng.choices(range(len(baseline)), k=len(baseline))
        samples.append(sum(baseline[i] for i in ids) / sum(candidate[i] for i in ids))
    samples.sort()
    return {"speedup_total_cost": sum(baseline) / sum(candidate),
            "ci95": [samples[count // 40], samples[-count // 40 - 1]],
            "baseline_total_us": sum(baseline), "candidate_total_us": sum(candidate)}


def verify_numerical():
    from kerneldarwin.fused_search import score_source_identity
    from kerneldarwin.workload import environment_identity
    result = read("numerical-validation.json")
    if (result.get("success") is not True or result.get("gpu_executed") is not True
            or result.get("source_sha256") != score_source_identity()
            or result.get("study_identity") != current_implementation()
            or result.get("environment") != environment_identity()):
        raise ValueError("Independent current-source GPU numerical report is required")
    return result


def verify_frozen(policy=None):
    from kerneldarwin.fused_search import score_source_identity
    from kerneldarwin.workload import environment_identity
    protocol = read("protocol.json")
    if (protocol["study_identity"] != current_implementation()
            or protocol["source_sha256"] != score_source_identity()
            or protocol["environment"] != environment_identity()
            or protocol["script_sha256"] != MEASUREMENT_SHA
            or protocol["corpus_sha256"] != digest(EVIDENCE / "corpus.json")):
        raise ValueError("Frozen implementation, runtime or corpus changed")
    if policy is not None:
        if (policy["study_identity"] != protocol["study_identity"]
                or policy["source_sha256"] != protocol["source_sha256"]
                or policy["environment"] != protocol["environment"]
                or policy["evidence"]["protocol_sha256"] != digest(EVIDENCE / "protocol.json")
                or policy["evidence"]["corpus_sha256"] != protocol["corpus_sha256"]
                or policy["evidence"]["development_sha256"] != digest(EVIDENCE / "development.json")
                or policy["evidence"]["numerical_report_sha256"] != digest(EVIDENCE / "numerical-validation.json")):
            raise ValueError("Frozen policy evidence changed")
    return protocol


def freeze():
    mx = setup()
    from kerneldarwin.fused_search import score_source_identity, library_scores
    from kerneldarwin.workload import environment_identity
    for name in ("protocol.json", "rejected-baseline.json"):
        if (EVIDENCE / name).exists():
            raise FileExistsError("Choose a fresh directory; never overwrite a study outcome")
    implementation = current_implementation()
    data = corpus()
    n = len(data["chunks"])
    if n < 64:
        raise ValueError("Need at least 64 actual document chunks")
    tasks = [
        {"id": "notes", "rows": 64, "width": 384, "queries": 1, "layout": "contiguous"},
        {"id": "manuals", "rows": min(256, n), "width": 768, "queries": 8, "layout": "column_stride2"},
        {"id": "archive", "rows": n, "width": 1536, "queries": 17, "layout": "transpose"},
    ]
    sources, source_bytes = source_metadata()
    if source_bytes > 24 * 1024:
        raise ValueError("Authored fused source exceeds prospective 24 KiB budget")
    rng = random.Random(1441)
    noise = []
    baseline = lambda q, docs: library_scores(q, docs, "stable")
    for task in tasks:
        docs, query = arrays(task, data["chunks"], 100)
        staged = score_inputs(task, docs, query, 32)
        for _ in range(5):
            measure_burst(baseline, staged)
        pairs = []
        for _ in range(40):
            order = [0, 1]
            rng.shuffle(order)
            values = {arm: measure_burst(baseline, staged) for arm in order}
            pairs.append([values[0], values[1]])
        a, b = map(list, zip(*pairs))
        noise.append({"task": task, "baseline_a_us": a, "baseline_b_us": b, "comparison": intervals(a, b),
                      "score_input_layout": "Direct compiled stable refresh output and contiguous query staging; Python MLX 0.32 does not expose array strides; authored shaders receive actual stride metadata"})
    worst = max(abs(bound - 1) for row in noise for bound in row["comparison"]["ci95"])
    observed = telemetry()
    if worst > 0.10:
        save(EVIDENCE / "rejected-baseline.json", {
            "reason": "Prospective identical-baseline 95% interval outside [0.9,1.1]; no tuning",
            "batch": 32, "noise": noise, "telemetry": observed,
            "source_sha256": score_source_identity(), "study_identity": implementation})
        raise RuntimeError("Baseline stability gate failed; no candidate timing is authorized by this protocol")
    save(EVIDENCE / "corpus.json", data)
    protocol = {
        "schema": 1, "target": "fused_query_scores_v1", "frozen_at": observed["time_utc"],
        "application": "One persistent index refresh followed by three changed-query top-k cosine searches per task",
        "tasks": tasks, "corpus_chunks": n, "corpus_sha256": digest(EVIDENCE / "corpus.json"),
        "weights": "Equal task frequency; three score calls and one refresh per application task; sum total costs",
        "development": {"seed": 104, "repeats": 40, "order_seed": 1207,
                        "authored_configurations": 2, "score_observations_per_round": 5,
                        "application_adapter_observations_per_round": 10},
        "measurement_batch": 32, "application_measurement_batch": 4,
        "confirmation": {"sessions": 3, "repeats_per_session": 40, "seeds": [17101, 18203, 19307],
                         "sequence_orders": [[2, 0, 1, 3], [1, 3, 2, 0], [3, 0, 2, 1]],
                         "unseen_task": {"id": "unseen", "rows": 113, "width": 641, "queries": 5, "layout": "column_stride2"}},
        "criterion": {"score_ci95_lower_speedup": max(1.10, 1 + 2 * worst),
                      "application_ci95_lower_speedup": 1.03,
                      "portfolio_over_fixed_minimum_total_cost_gain": 0.02,
                      "numerical_rtol": 2e-5, "numerical_atol": 2e-6},
        "limits": {"candidates_per_shape": 2, "portfolio_kernels": 2, "max_rules": 128,
                   "source_bytes": 24 * 1024, "development_wall_seconds": 900,
                   "memory_bytes": 2 * 1024 ** 3, "cache_bytes": 256 * 1024 ** 2},
        "selection": "Two authored configurations on the same changed-input distribution; fixed uses global application costs, per-shape uses its four separately measured routed adapter costs, joint uses actual candidate-plan and empty-plan fallback application costs; custom rules need score CI eligibility versus both libraries",
        "stability_gate": "Every identical stable-score baseline 95% total-cost interval in [0.9,1.1] before tuning",
        "timing": "Synchronized host wall microseconds including allocation, dispatch, materialization and full-app input transfers; score burst excludes persistent index normalization; full-app timing includes refresh",
        "qos": "user_initiated_current_thread_only",
        "confirmation_rule": "Never retune or weaken gates after confirmation; include untouched-shape whole-query fallback, every loss and every independent session",
        "environment": environment_identity(), "telemetry": observed, "baseline_noise": noise,
        "source_sha256": score_source_identity(), "study_identity": implementation,
        "sources": sources, "source_bytes": source_bytes, "script_sha256": MEASUREMENT_SHA,
        "source_budget_scope": "Two authored MSL files; adapter/package size is reported separately at export",
    }
    save(EVIDENCE / "protocol.json", protocol)
    print(json.dumps({"tasks": tasks, "required_score_speedup": protocol["criterion"]["score_ci95_lower_speedup"]}, indent=2))


def candidate_methods(protocol, task):
    """Time each authored configuration through the actual guarded plan API."""
    methods = {name: StudyMethod(name) for name in (*LIBRARIES, "eager_library")}
    for name in SCORE_NAMES:
        rules = [{"shape": [task["queries"], task["rows"], task["width"]], "candidate": name,
                  "dtype": "float32", "layout": "any_mlx_strides"}]
        methods[name] = trial_plan(protocol, rules)
    return methods


def trial_plan(protocol, rules):
    """Private in-memory numerical adapter, with confirmation kept false."""
    from kerneldarwin.fused_search import FusedPlan
    manifest = {"schema": 1, "operation": "stable_l2_scores_v1", "epsilon": 1e-12,
                "source_sha256": protocol["source_sha256"], "study_identity": protocol["study_identity"],
                "environment": protocol["environment"], "sources": protocol["sources"], "rules": rules,
                "evidence": {"numerical_passed": True, "confirmation_passed": False}}
    return FusedPlan._trial(manifest)


def tune():
    for name in ("development.json", "development-timeout.json", "policy-frozen.json"):
        if (EVIDENCE / name).exists():
            raise FileExistsError("Development already ran; preserve its outcome rather than retune")
    mx = setup()
    import numpy as np
    from kerneldarwin.fused_search import score_source_identity
    from kerneldarwin.workload import environment_identity
    protocol = verify_frozen()
    verify_numerical()
    start = time.perf_counter()
    texts = read("corpus.json")["chunks"]
    rng = random.Random(protocol["development"]["order_seed"])
    rows = []
    def require_budget(stage, **partial):
        elapsed = time.perf_counter() - start
        if elapsed > protocol["limits"]["development_wall_seconds"]:
            save(EVIDENCE / "development-timeout.json", {"completed_rows": rows, "elapsed_seconds": elapsed,
                 "stage": stage, "source_sha256": score_source_identity(),
                 "study_identity": current_implementation(), **partial})
            raise TimeoutError("Second target's 900-second development budget exhausted")
        return elapsed

    for task in protocol["tasks"]:
        methods = candidate_methods(protocol, task)
        shape = (task["queries"], task["rows"], task["width"])
        routed_methods = {name: StudyMethod(rules={shape: name}) for name in ROUTED_CHOICES}
        plan_library = trial_plan(protocol, [])
        app_arms = {**{"global:" + name: method for name, method in methods.items()},
                    **{"routed:" + name: method for name, method in routed_methods.items()},
                    "plan_library": plan_library}
        docs, query = arrays(task, texts, protocol["development"]["seed"])
        staged = score_inputs(task, docs, query, protocol["measurement_batch"])
        first_score = {}
        first_app = {}
        first_adapter_app = {}
        application_checks = {}
        for name, method in methods.items():
            first_score[name] = measure(lambda m=method: m.score(*staged[0])) / 1000
            q, normalized = staged[0]
            reference = float64_score_operation(np.array(q), np.array(normalized))
            np.testing.assert_allclose(np.array(method.score(q, normalized)), reference, rtol=2e-5, atol=2e-6)
            first_app[name] = measure(lambda m=method: application(m, task, docs, query)) / 1000
            check_application(application(method, task, docs, query), docs, query)
            application_checks["global:" + name] = {"scores_passed": True, "ids_association_passed": True}
            for _ in range(5):
                measure_burst(method.score, staged)
                measure(lambda m=method: application(m, task, docs, query))
            require_budget("global_warmup", task=task, method=name)
        for label, method in app_arms.items():
            if label.startswith("global:"):
                continue
            first_adapter_app[label] = measure(lambda m=method: application(m, task, docs, query)) / 1000
            check_application(application(method, task, docs, query), docs, query)
            application_checks[label] = {"scores_passed": True, "ids_association_passed": True}
            for _ in range(5):
                measure(lambda m=method: application(m, task, docs, query))
            require_budget("adapter_warmup", task=task, method=label)
        score_samples = {name: [] for name in methods}
        app_samples = {name: [] for name in methods}
        routed_app_samples = {name: [] for name in routed_methods}
        plan_library_app_samples = []
        orders = []
        application_orders = []
        for repeat in range(protocol["development"]["repeats"]):
            # New app contents on each development repetition, shared by arms.
            app_docs, app_query = arrays(task, texts, protocol["development"]["seed"] + repeat + 1)
            staged = score_inputs(task, app_docs, app_query, protocol["measurement_batch"])
            names = list(methods)
            rng.shuffle(names)
            orders.append(names)
            for name in names:
                method = methods[name]
                score_samples[name].append(measure_burst(method.score, staged))
            labels = list(app_arms)
            rng.shuffle(labels)
            application_orders.append(labels)
            for label in labels:
                elapsed = measure(lambda m=app_arms[label]: application(m, task, app_docs, app_query))
                if label.startswith("global:"):
                    app_samples[label.removeprefix("global:")].append(elapsed)
                elif label.startswith("routed:"):
                    routed_app_samples[label.removeprefix("routed:")].append(elapsed)
                else:
                    plan_library_app_samples.append(elapsed)
            require_budget("measurement", task=task, partial_score_samples_us=score_samples,
                           partial_application_samples_us=app_samples,
                           partial_routed_application_samples_us=routed_app_samples,
                           partial_plan_library_application_samples_us=plan_library_app_samples)
        comparisons = {name: {baseline: {"score": intervals(score_samples[baseline], score_samples[name]),
                           "application": intervals(app_samples[baseline], app_samples[name])} for baseline in LIBRARIES} for name in SCORE_NAMES}
        rows.append({"task": task, "shape": [task["queries"], task["rows"], task["width"]],
                     "weight": 3, "score_samples_us": score_samples, "application_samples_us": app_samples,
                     "routed_application_samples_us": routed_app_samples,
                     "plan_library_application_samples_us": plan_library_app_samples,
                     "orders": orders, "application_orders": application_orders,
                     "application_numerical_checks": application_checks,
                     "first_score_in_development_ms": first_score, "first_application_in_development_ms": first_app,
                     "first_adapter_application_in_development_ms": first_adapter_app,
                     "first_call_note": "Not process-cold; preceding methods may share compiler/driver caches; see cold-study",
                     "score_input_layout": "Direct compiled stable refresh output and contiguous query staging; Python MLX 0.32 does not expose array strides; authored shaders receive actual stride metadata",
                     "numerical_check": {"passed": True, "reference": "float64 stable norm, full scores and top-k id association", "rtol": 2e-5, "atol": 2e-6},
                     "comparisons": comparisons})
        require_budget("row_analysis", task=task)
        print(task["id"], "score", {n: round(statistics.mean(v), 2) for n, v in score_samples.items()},
              "application", {n: round(statistics.mean(v), 2) for n, v in app_samples.items()}, flush=True)

    cost = lambda row, name: statistics.mean(row["application_samples_us"][name])
    routed_cost = lambda row, name: statistics.mean(row["routed_application_samples_us"][name])
    plan_cost = lambda row, name: statistics.mean(row["plan_library_application_samples_us"]) if name == "stable_library" else cost(row, name)
    eligible = lambda row, name: all(row["comparisons"][name][baseline]["score"]["ci95"][0] > protocol["criterion"]["score_ci95_lower_speedup"] for baseline in LIBRARIES)
    fixed = min((*LIBRARIES, *SCORE_NAMES), key=lambda name: sum(cost(row, name) for row in rows))
    per_shape = {tuple(row["shape"]): min(ROUTED_CHOICES, key=lambda name: routed_cost(row, name)) for row in rows}
    choices = []
    # Empty subset is a genuine optimized stable-library-only outcome.
    for count in (0, 1, 2):
        for subset in itertools.combinations(SCORE_NAMES, count):
            rules = {}
            total = 0.0
            for row in rows:
                options = ["stable_library", *[name for name in subset if eligible(row, name)]]
                best = min(options, key=lambda name: plan_cost(row, name))
                total += plan_cost(row, best)
                if best in SCORE_NAMES:
                    rules[tuple(row["shape"])] = best
            choices.append({"subset": subset, "total_us": total, "rules": rules})
    best_one = min((choice for choice in choices if len(choice["subset"]) <= 1), key=lambda choice: choice["total_us"])
    joint = min(choices, key=lambda choice: choice["total_us"])
    if joint["total_us"] > best_one["total_us"] * (1 - protocol["criterion"]["portfolio_over_fixed_minimum_total_cost_gain"]):
        joint = best_one
    selected = sorted(set(joint["rules"].values()))
    development = {"source_sha256": score_source_identity(), "study_identity": current_implementation(),
                   "protocol_sha256": digest(EVIDENCE / "protocol.json"), "corpus_sha256": protocol["corpus_sha256"],
                   "rows": rows, "peak_memory_bytes": mx.get_peak_memory(),
                   "telemetry": telemetry(), "budget": "Two authored configurations; five score observations and ten separately randomized application adapter observations per task/round on the same changed inputs. Fixed uses global costs, per-shape uses exact routed costs, joint uses actual candidate-plan and empty-plan costs; all share the 900-second development budget; no confirmation retuning"}
    development["search_wall_seconds"] = require_budget("post_selection")
    save(EVIDENCE / "development.json", development)
    policy = {"schema": 1, "operation": "stable_l2_scores_v1", "epsilon": 1e-12,
              "source_sha256": score_source_identity(), "study_identity": current_implementation(),
              "environment": environment_identity(), "sources": source_metadata()[0],
              "rules": [{"shape": list(shape), "candidate": name, "dtype": "float32", "layout": "any_mlx_strides"} for shape, name in sorted(joint["rules"].items())],
              "evidence": {"numerical_passed": True, "confirmation_passed": False,
                           "corpus_sha256": protocol["corpus_sha256"], "protocol_sha256": digest(EVIDENCE / "protocol.json"),
                           "numerical_report_sha256": digest(EVIDENCE / "numerical-validation.json"),
                           "development_sha256": digest(EVIDENCE / "development.json")},
              "policy_selection": {"best_fixed": fixed, "joint_subset": selected,
                                   "development_application_cost_us": joint["total_us"],
                                   "best_one_subset_application_cost_us": best_one["total_us"],
                                   "best_fixed_application_cost_us": sum(cost(row, fixed) for row in rows),
                                   "development_per_shape_application_cost_us": sum(routed_cost(row, per_shape[tuple(row["shape"])]) for row in rows),
                                   "per_shape": [{"shape": list(shape), "candidate": name} for shape, name in sorted(per_shape.items())]}}
    save(EVIDENCE / "policy-frozen.json", policy)
    print(json.dumps(policy, indent=2))


def policies(policy, requested=None):
    from kerneldarwin.fused_search import FusedPlan
    names = METHODS if requested is None else (requested,)
    result = {}
    for name in names:
        if name == "best_fixed":
            result[name] = StudyMethod(policy["policy_selection"]["best_fixed"])
        elif name == "per_shape":
            rules = {tuple(rule["shape"]): rule["candidate"] for rule in policy["policy_selection"]["per_shape"]}
            result[name] = StudyMethod(rules=rules)
        elif name == "dispatcher":
            before = time.perf_counter()
            dispatcher = FusedPlan._trial(policy)
            dispatcher.artifact_load_ms = (time.perf_counter() - before) * 1000
            if dispatcher.status != "numerical_trial":
                raise ValueError(f"Trial dispatcher rejected frozen policy: {dispatcher.status}")
            result[name] = dispatcher
        else:
            result[name] = StudyMethod(name)
    return result


def confirm(session):
    mx = setup()
    if session not in (0, 1, 2):
        raise ValueError("session must be 0, 1 or 2")
    target = EVIDENCE / f"confirmation-{session}.json"
    if target.exists():
        raise FileExistsError(target)
    policy = read("policy-frozen.json")
    protocol = verify_frozen(policy)
    seed = protocol["confirmation"]["seeds"][session]
    order = protocol["confirmation"]["sequence_orders"][session]
    tasks = [*protocol["tasks"], protocol["confirmation"]["unseen_task"]]
    texts = read("corpus.json")["chunks"]
    methods = policies(policy)
    prepared = [arrays(task, texts, seed) for task in tasks]
    before = telemetry()
    first = {}
    checks = []
    for name, method in methods.items():
        first[name] = measure(lambda m=method: [application(m, task, *prepared[i]) for i, task in enumerate(tasks)]) / 1000
        for i, task in enumerate(tasks):
            check_application(application(method, task, *prepared[i]), *prepared[i])
            checks.append({"task": task["id"], "method": name, "scores_passed": True, "ids_association_passed": True})
    score_samples = {name: [] for name in methods}
    app_samples = {name: [] for name in methods}
    orders = []
    rng = random.Random(seed)
    for repeat in range(protocol["confirmation"]["repeats_per_session"]):
        prepared = [arrays(task, texts, seed + repeat + 1) for task in tasks]
        staged = [score_inputs(task, *prepared[i], protocol["measurement_batch"]) for i, task in enumerate(tasks)]
        names = list(methods)
        rng.shuffle(names)
        orders.append(names)
        for name in names:
            method = methods[name]
            score_samples[name].append(sum(3 * measure_burst(method.score, inputs) for inputs in staged))
            app_samples[name].append(measure(lambda m=method: [[application(m, tasks[i], *prepared[i]) for i in order]
                                     for _ in range(protocol["application_measurement_batch"])] ) / protocol["application_measurement_batch"])
        if (repeat + 1) % 10 == 0:
            print("session", session, "repeat", repeat + 1, flush=True)
    comparisons = {name: {baseline: {"score": intervals(score_samples[baseline], score_samples[name], seed),
                              "application": intervals(app_samples[baseline], app_samples[name], seed)} for baseline in LIBRARIES}
                   for name in methods if name not in LIBRARIES}
    dispatcher = comparisons["dispatcher"]
    gates = {"score": all(dispatcher[baseline]["score"]["ci95"][0] > protocol["criterion"]["score_ci95_lower_speedup"] for baseline in LIBRARIES),
             "application": all(dispatcher[baseline]["application"]["ci95"][0] > protocol["criterion"]["application_ci95_lower_speedup"] for baseline in LIBRARIES)}
    result = {"session": session, "seed": seed, "sequence_order": order,
              "source_sha256": policy["source_sha256"], "study_identity": current_implementation(),
              "corpus_sha256": protocol["corpus_sha256"], "protocol_sha256": digest(EVIDENCE / "protocol.json"),
              "policy_sha256": digest(EVIDENCE / "policy-frozen.json"), "environment": protocol["environment"],
              "telemetry_before": before, "telemetry_after": telemetry(), "first_sequence_in_session_ms": first,
              "artifact_load_ms": methods["dispatcher"].artifact_load_ms,
              "cold_note": "Preceding methods may share compilation; use independent cold-study process starts",
              "score_samples_us": score_samples, "application_samples_us": app_samples, "orders": orders,
              "checks": checks, "peak_memory_bytes": mx.get_peak_memory(), "comparisons": comparisons,
              "session_gate": gates,
              "fallback_count_scope": "Static/fixed/per-shape/dispatcher route decisions across validation, score bursts and application calls; global library methods execute directly and have zero adapter counters",
              "fallback_counts": {name: {"calls": method.calls, "fallback_calls": method.fallback_calls} for name, method in methods.items()}}
    save(target, result)
    print(json.dumps({"dispatcher": dispatcher, "session_gate": gates}, indent=2))


def cold(method):
    before = time.perf_counter()
    mx = setup()
    runtime_ms = (time.perf_counter() - before) * 1000
    policy = read("policy-frozen.json")
    before = time.perf_counter()
    protocol = verify_frozen(policy)
    probe_ms = (time.perf_counter() - before) * 1000
    before = time.perf_counter()
    selected = policies(policy, requested=method)[method]
    load_ms = (time.perf_counter() - before) * 1000
    before = time.perf_counter()
    tasks = [*protocol["tasks"], protocol["confirmation"]["unseen_task"]]
    texts = read("corpus.json")["chunks"]
    prepared = [arrays(task, texts, 19661) for task in tasks]
    input_ms = (time.perf_counter() - before) * 1000
    first_ms = measure(lambda: [application(selected, task, *prepared[i]) for i, task in enumerate(tasks)]) / 1000
    print(json.dumps({"method": method, "source_sha256": policy["source_sha256"], "study_identity": current_implementation(),
                      "corpus_sha256": protocol["corpus_sha256"], "environment": protocol["environment"],
                      "policy_sha256": digest(EVIDENCE / "policy-frozen.json"),
                      "runtime_setup_ms": runtime_ms, "frozen_checks_ms": probe_ms, "policy_setup_ms": load_ms,
                      "input_prepare_ms": input_ms, "first_application_sequence_ms": first_ms,
                      "peak_memory_bytes": mx.get_peak_memory(),
                      "cache_note": "Fresh Python/MLX process; shared OS/driver caches are not deleted; first-call work includes compilation/dispatch; pure compiler time is not isolated"}))


def cold_study():
    policy = read("policy-frozen.json")
    protocol = verify_frozen(policy)
    target = EVIDENCE / "cold-study.json"
    if target.exists():
        raise FileExistsError(target)
    rng = random.Random(1667)
    rows = []
    start = time.perf_counter()
    for repeat in range(3):
        order = list(METHODS)
        rng.shuffle(order)
        for method in order:
            before = time.perf_counter()
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "cold", "--method", method,
                                    "--directory", str(EVIDENCE)], cwd=ROOT, text=True, capture_output=True, timeout=60)
            wall = (time.perf_counter() - before) * 1000
            if child.returncode:
                save(EVIDENCE / "cold-study-failure.json", {"method": method, "repeat": repeat,
                     "returncode": child.returncode, "stderr_tail": child.stderr[-3000:], "completed_rows": rows})
                raise RuntimeError("Cold process failed: " + child.stderr[-1500:])
            row = json.loads(child.stdout)
            row.update(repeat=repeat, process_wall_ms=wall)
            rows.append(row)
            print("cold", repeat, method, round(wall, 1), "ms", flush=True)
            if time.perf_counter() - start > 180:
                save(EVIDENCE / "cold-study-timeout.json", {"rows": rows, "wall_seconds": time.perf_counter() - start})
                raise TimeoutError("Cold study exceeded prospective 180-second budget")
    save(target, {"schema": 1, "source_sha256": policy["source_sha256"], "study_identity": current_implementation(),
                  "corpus_sha256": protocol["corpus_sha256"], "policy_sha256": digest(EVIDENCE / "policy-frozen.json"),
                  "rows": rows, "wall_seconds": time.perf_counter() - start, "telemetry": telemetry(),
                  "method": "18 serialized independent process starts; randomized six-method order; shared OS/driver caches retained; process wall includes source/environment/evidence probes"})


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
