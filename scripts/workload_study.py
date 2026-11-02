#!/usr/bin/env python3
"""Bounded, serial Metal workload study. Development and confirmation are separate.

Run from repository root with the pinned MLX environment. Every outcome is saved;
confirmation never changes the frozen policy. Host timings are not GPU timers.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/workload-2026-10-29"
METHODS=('library','eager_library','fast_library','compiled_application','fast_compiled_application','best_fixed','per_shape','dispatcher')
_MEASUREMENT_IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def current_implementation():
    import kerneldarwin.study_identity as binding
    pin=json.loads((Path(binding.__file__).parent/'study-source.json').read_text())
    if pin['measurement_script_sha256']!=_MEASUREMENT_IDENTITY or digest(__file__)!=_MEASUREMENT_IDENTITY:
        raise ValueError('Measurement script differs from installed source pin; update intentionally and rebuild before a fresh study')
    return binding.implementation_identity()


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def telemetry():
    data = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load_average": os.getloadavg()}
    for cmd, key in ((["pmset", "-g", "therm"], "thermal"), (["pmset", "-g", "batt"], "power")):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data[key] = (result.stdout + result.stderr).strip()
    return data


def setup():
    # A user-initiated priority applies only to this thread/process lifetime.
    # It changes neither system settings nor any other process. Both arms run
    # under the same priority. Record it; do not assume it removes contention.
    import ctypes
    system=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    code=system.pthread_set_qos_class_self_np(0x19,0)
    if code:raise RuntimeError(f'Cannot set own user-initiated QoS: {code}')
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(2 * 1024 ** 3)
    mx.set_cache_limit(256 * 1024 ** 2)
    return mx


def corpus():
    archived=ROOT/"evidence/workload-2026-10-29/corpus.json"
    if archived.exists():
        return json.loads(archived.read_text())
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    chunks = []
    provenance = []
    for path in files:
        provenance.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        words = path.read_text().split()
        for offset in range(0, len(words), 48):
            text = " ".join(words[offset:offset + 48])
            if len(text) > 80:
                chunks.append(text)
    return {"license": "Apache-2.0; KERNELDARWIN documentation, copyright Nazeeh Abdul-Hadi", "sources": provenance, "chunks": chunks}


def arrays(task, texts, seed=0):
    """Real document token vectors, changed corpus order/content on every seed."""
    import mlx.core as mx
    import numpy as np
    from kerneldarwin.search import text_vectors
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    selected = [texts[indices[i % len(indices)]] + f" revision{seed % 13}" for i in range(task["rows"])]
    queries = [texts[indices[(i * 7 + 3) % len(indices)]] + f" query{seed % 19}" for i in range(task["queries"])]
    documents = np.array(text_vectors(selected, task["width"]))
    query = np.array(text_vectors(queries, task["width"]))
    return documents, query


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
    result = function()
    if isinstance(result, mx.array):
        mx.eval(result)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1000


def burst_inputs(x, count, layout="contiguous"):
    import mlx.core as mx
    # Distinct, evaluated values defeat common-subexpression elimination. This
    # is input staging, never a cached output or repeated alias of one graph.
    import numpy as np
    host=np.array(x)
    values=[layout_array(host+(i+1)*0.001,layout) for i in range(count)]
    mx.eval(*values)
    return values


def measure_burst(function, inputs):
    import mlx.core as mx
    return measure(lambda:mx.eval(*[function(x) for x in inputs]))/len(inputs)


def application(function, task, documents, queries):
    from kerneldarwin.search import CosineIndex
    from kerneldarwin.library_baselines import FullGraphLibrary
    vectors=layout_array(documents,task['layout'])
    index = function.index(vectors) if isinstance(function,FullGraphLibrary) else CosineIndex(vectors,function)
    # Three changing query batches per index refresh, as frozen in the trace.
    return [index.search(layout_array(queries + j * 0.001, "contiguous"), k=5) for j in range(3)]


def check_application(output, documents, queries):
    """Validate complete top-k values and their document IDs independently."""
    import numpy as np
    def normalized(values):
        values=np.asarray(values,np.float64)
        peak=np.maximum(np.max(np.abs(values),axis=1,keepdims=True),1e-12)
        scaled=values/peak
        return scaled/np.maximum(np.sqrt(np.sum(scaled*scaled,axis=1,keepdims=True)),1e-12/peak)
    reference_documents=normalized(documents)
    if len(output)!=3:raise AssertionError('Need all three query requests')
    for j,(ids,values) in enumerate(output):
        reference=normalized(queries+j*0.001)@reference_documents.T
        ids=np.asarray(ids);values=np.asarray(values)
        if ids.shape!=(queries.shape[0],5) or values.shape!=ids.shape:
            raise AssertionError('Incorrect top-k result shape')
        if not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=documents.shape[0]):
            raise AssertionError('Invalid document identifiers')
        if any(len(set(row.tolist()))!=5 for row in ids):
            raise AssertionError('Repeated top-k document identifier')
        np.testing.assert_allclose(values,np.take_along_axis(reference,ids,axis=1),atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(values,-np.sort(-reference,axis=1)[:,:5],atol=2e-6,rtol=2e-5)


def intervals(baseline, candidate, seed=404, count=4000):
    """Paired total-cost ratio, never an average of selected speedup ratios."""
    rng = random.Random(seed)
    ratios = []
    for _ in range(count):
        ids = rng.choices(range(len(baseline)), k=len(baseline))
        ratios.append(sum(baseline[i] for i in ids) / sum(candidate[i] for i in ids))
    ratios.sort()
    return {"speedup_total_cost": sum(baseline) / sum(candidate), "ci95": [ratios[count // 40], ratios[-count // 40 - 1]], "baseline_total_us": sum(baseline), "candidate_total_us": sum(candidate)}


def freeze():
    mx = setup()
    implementation=current_implementation()
    from kerneldarwin.workload import library_normalize, environment_identity
    if (EVIDENCE/"protocol.json").exists() or (EVIDENCE/"rejected-baseline.json").exists():
        raise FileExistsError("Choose a fresh study directory; evidence is immutable")
    data = corpus()
    n = len(data["chunks"])
    if n < 64:
        raise ValueError("Need at least 64 actual document chunks")
    tasks = [
        {"id": "notes", "rows": 64, "width": 384, "queries": 1, "layout": "contiguous"},
        {"id": "manuals", "rows": min(256, n), "width": 768, "queries": 8, "layout": "column_stride2"},
        {"id": "archive", "rows": n, "width": 1536, "queries": 17, "layout": "transpose"},
    ]
    rng = random.Random(41)
    noise = []
    for task in tasks:
        host, query = arrays(task, data["chunks"])
        x = layout_array(host, task["layout"])
        mx.eval(x)
        staged=burst_inputs(x,32,task["layout"])
        for _ in range(5):
            measure_burst(library_normalize,staged)
        pairs = []
        for _ in range(40):
            order = [0, 1]; rng.shuffle(order)
            samples = {}
            for arm in order:
                samples[arm] = measure_burst(library_normalize,staged)
            pairs.append([samples[0], samples[1]])
        a, b = map(list, zip(*pairs))
        noise.append({"task": task, "baseline_a_us": a, "baseline_b_us": b, "comparison": intervals(a, b)})
    worst = max(max(abs(v - 1) for v in row["comparison"]["ci95"]) for row in noise)
    # Freeze a useful threshold above noise before any candidate timings.
    required = max(1.10, 1 + 2 * worst)
    observed=telemetry()
    if worst>0.10:
        save(EVIDENCE/"rejected-baseline.json",{"reason":"Identical baseline 95% interval exceeded +/-10%; prospective stability gate failed", "batch":32,"qos":"user_initiated_current_thread_only","noise":noise,"telemetry":observed})
        raise RuntimeError("Batched baseline still too noisy; no tuning authorized by this protocol")
    protocol = {
        "schema": 1, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "application": "Local cosine-index refresh + three top-k searches on original documentation token vectors",
        "corpus_chunks": n, "tasks": tasks,
        "weights": "One refresh and three query normalizations per task, equal task frequency; total costs summed",
        "development": {"seed": 101, "repeats": 40},
        "measurement_batch":32,
        "qos":"user_initiated_current_thread_only",
        "application_measurement_batch":4,
        "stability_gate":"Every identical-baseline 95% total-cost ratio must lie in [0.9,1.1] before tuning",
        "confirmation": {"sessions": 3, "repeats_per_session": 40, "seeds": [7101, 8203, 9307],
            "sequence_orders": [[2,0,1,3], [1,3,2,0], [3,0,2,1]],
            "unseen_task": {"id": "unseen", "rows": 97, "width": 513, "queries": 3, "layout": "column_stride2"}},
        "criterion": {"normalization_ci95_lower_speedup": required, "application_claim_ci95_lower_speedup": 1.03,
            "portfolio_over_fixed_minimum_total_cost_gain": 0.02, "numerical_rtol": 2e-5, "numerical_atol": 2e-6},
        "limits": {"candidates_per_shape": 4, "portfolio_kernels": 2, "max_rules": 128, "source_bytes": 12000,
            "development_wall_seconds": 900, "memory_bytes": 2*1024**3, "cache_bytes": 256*1024**2},
        "timing": "Synchronized host wall microseconds, allocation/dispatch/materialization included; device-only timing requires profiler",
        "confirmation_rule": "Do not retune after confirmation. Include unseen-shape fallback, all losses, all sessions and full application costs.",
        "environment": environment_identity(), "telemetry": observed, "baseline_noise": noise,
        "script_sha256": digest(__file__),
        "study_identity":implementation,
    }
    save(EVIDENCE / "corpus.json", data)
    protocol['corpus_sha256']=digest(EVIDENCE/'corpus.json')
    save(EVIDENCE / "protocol.json", protocol)
    print(json.dumps({"chunks": n, "tasks": tasks, "required_normalization_speedup": required}, indent=2))


def operation_inputs(tasks, texts, seed):
    import mlx.core as mx
    result = []
    for task in tasks:
        docs, queries = arrays(task, texts, seed)
        for name, host, layout, weight in (("index", docs, task["layout"], 1), ("query", queries, "contiguous", 3)):
            x = layout_array(host, layout);mx.eval(x)
            result.append({"id": task["id"]+"-"+name, "x": x, "shape": list(x.shape), "layout": layout, "weight": weight})
    return result


def tune():
    mx = setup()
    import numpy as np
    from kerneldarwin.workload import CANDIDATES, candidate_normalize, library_normalize, source_identity, environment_identity
    from kerneldarwin.library_baselines import eager_normalize,fast_library_normalize,FullGraphLibrary
    protocol = read("protocol.json"); texts = read("corpus.json")["chunks"]
    if digest(EVIDENCE/'corpus.json')!=protocol['corpus_sha256']:raise ValueError('Frozen corpus changed')
    numerical=read("numerical-validation.json")
    if numerical.get('source_sha256')!=source_identity() or numerical.get('study_identity')!=current_implementation() or numerical.get('success') is not True or numerical.get('gpu_executed') is not True:
        raise ValueError('Copy the independent current-source Metal validation report into the study before tuning')
    if environment_identity() != protocol["environment"]:
        raise ValueError("Environment changed since freeze")
    if protocol['study_identity']!=current_implementation():
        raise ValueError('Application, baselines or measurement implementation changed since freeze')
    start = time.perf_counter()
    rng = random.Random(207)
    results = []
    implementations={"library":library_normalize,"eager_library":eager_normalize,"fast_library":fast_library_normalize,
                     "compiled_application":FullGraphLibrary('stable'),"fast_compiled_application":FullGraphLibrary('fast'),
                     **{name:lambda x,name=name:candidate_normalize(x,name) for name in CANDIDATES}}
    compile_ms = {name: 0.0 for name in implementations}
    for item in operation_inputs(protocol["tasks"], texts, protocol["development"]["seed"]):
        x = item["x"]
        staged=burst_inputs(x,protocol["measurement_batch"],item["layout"])
        ah = np.array(x).astype(np.float64)
        ref = ah / np.maximum(np.linalg.norm(ah, axis=1, keepdims=True), 1e-12)
        functions = {name:lambda f=f:f(x) for name,f in implementations.items()}
        cold = {}
        for name, f in functions.items():
            cold[name] = measure(f) / 1000;compile_ms[name] += cold[name]
            np.testing.assert_allclose(np.array(f()), ref, rtol=2e-5, atol=2e-6)
            for _ in range(5):
                measure_burst(implementations[name],staged)
        samples = {name: [] for name in functions}
        orders = []
        for repeat in range(protocol["development"]["repeats"]):
            names = list(functions);rng.shuffle(names);orders.append(names)
            for name in names:
                f=implementations[name]
                samples[name].append(measure_burst(f,staged))
        row = {k:v for k,v in item.items() if k != "x"}
        row.update(numerical_check={"passed":True,"reference":"float64 independent max-scaled norm", "rtol":2e-5,"atol":2e-6},cold_first_call_ms=cold, samples_us=samples, orders=orders,
                   comparisons={name: intervals(samples["library"], samples[name]) for name in CANDIDATES})
        results.append(row)
        print(row["id"], {k:round(statistics.mean(v),1) for k,v in samples.items()}, flush=True)
        if time.perf_counter() - start > protocol["limits"]["development_wall_seconds"]:
            raise TimeoutError("Bounded development budget exhausted")
    # Minimize workload-weighted total latency, including the fallback as a choice.
    costs = {row["id"]:{name:statistics.mean(values) for name,values in row["samples_us"].items()} for row in results}
    choices = []
    for count in (1,2):
        for subset in itertools.combinations(CANDIDATES, count):
            rules = {}
            total = 0.0
            for row in results:
                # Only admit a rule with a development confidence lower bound >1.10.
                eligible = ["library", *[name for name in subset if row["comparisons"][name]["ci95"][0] > protocol["criterion"]["normalization_ci95_lower_speedup"]]]
                best = min(eligible, key=lambda name: costs[row["id"]][name])
                total += row["weight"]*costs[row["id"]][best]
                if best != "library":rules[tuple(row["shape"])] = best
            choices.append({"subset": subset, "total_us": total, "rules": rules})
    best_fixed = min((x for x in choices if len(x["subset"]) == 1), key=lambda x:sum(row["weight"]*costs[row["id"]][x["subset"][0]] for row in results))
    fixed_unconditional_cost=sum(row["weight"]*costs[row["id"]][best_fixed["subset"][0]] for row in results)
    joint = min(choices, key=lambda x:x["total_us"])
    if joint["total_us"] > best_fixed["total_us"]*(1-protocol["criterion"]["portfolio_over_fixed_minimum_total_cost_gain"]):
        joint = best_fixed
    per_shape = {tuple(row["shape"]): min(CANDIDATES, key=lambda n:costs[row["id"]][n]) for row in results}
    policy = {"schema":1, "operation":"stable_l2_rows_v1", "epsilon":1e-12,
        "study_identity":current_implementation(),
        "source_sha256":source_identity(), "environment":environment_identity(),
        "sources":{name:digest(Path(__import__('kerneldarwin.workload',fromlist=['SOURCE_FILES']).__file__).parent/"metal"/name) for name in __import__('kerneldarwin.workload',fromlist=['SOURCE_FILES']).SOURCE_FILES},
        "rules":[{"shape":list(shape), "candidate":name, "dtype":"float32", "layout":"any_mlx_strides"} for shape,name in sorted(joint["rules"].items())],
        "evidence":{"numerical_passed":True, "confirmation_passed":False, "corpus_sha256":protocol['corpus_sha256'],"protocol_sha256":digest(EVIDENCE/"protocol.json"),"numerical_report_sha256":digest(EVIDENCE/"numerical-validation.json")},
        "policy_selection":{"best_fixed":best_fixed["subset"][0], "joint_subset":list(joint["subset"]),
            "development_weighted_cost_us":joint["total_us"], "best_fixed_weighted_cost_us":fixed_unconditional_cost,
            "per_shape":[{"shape":list(shape),"candidate":name} for shape,name in sorted(per_shape.items())]},
    }
    save(EVIDENCE/"development.json", {"source_sha256":source_identity(), "study_identity":current_implementation(),"corpus_sha256":protocol['corpus_sha256'],"protocol_sha256":digest(EVIDENCE/"protocol.json"), "rows":results,
        "compile_first_call_total_ms":compile_ms,"search_wall_seconds":time.perf_counter()-start,"peak_memory_bytes":mx.get_peak_memory(), "telemetry":telemetry(),
        "budget":"Same four candidate measurements feed fixed, per-shape and joint selectors. No learned policy, no hidden online tuning."})
    policy['evidence']['development_sha256']=digest(EVIDENCE/'development.json')
    save(EVIDENCE/"policy-frozen.json", policy)
    print(json.dumps(policy,indent=2))


def policies(policy, requested=None):
    from kerneldarwin.workload import Normalizer, candidate_normalize, library_normalize
    from kerneldarwin.library_baselines import eager_normalize,fast_library_normalize,FullGraphLibrary
    import mlx.core as mx
    class Static:
        def __init__(self, rules):self.rules=rules;self.calls=0;self.fallback_calls=0
        def __call__(self, x):
            self.calls+=1
            name=self.rules.get(tuple(x.shape)) if x.dtype==mx.float32 else None
            if name is None:self.fallback_calls+=1;return library_normalize(x)
            return candidate_normalize(x,name)
    fixed=policy["policy_selection"]["best_fixed"]
    class Fixed:
        def __init__(self):self.calls=0;self.fallback_calls=0
        def __call__(self,x):
            self.calls+=1
            if x.dtype==mx.float32 and x.size and x.shape[1]<=2048:return candidate_normalize(x,fixed)
            self.fallback_calls+=1;return library_normalize(x)
    per_shape={tuple(rule["shape"]):rule["candidate"] for rule in policy["policy_selection"]["per_shape"]}
    # A provisional data-only plan is used for measuring the exact runtime adapter.
    # Its confirmation flag is not changed on disk and it cannot be exported.
    from kerneldarwin.workload import SOURCE_FILES
    import kerneldarwin.workload as operation
    result={"library":library_normalize,"eager_library":eager_normalize,"fast_library":fast_library_normalize,"compiled_application":FullGraphLibrary('stable'),"fast_compiled_application":FullGraphLibrary('fast'),"best_fixed":Fixed(),"per_shape":Static(per_shape)}
    if requested is None or requested=='dispatcher':
        with tempfile.TemporaryDirectory(prefix='kd-numerical-trial-') as directory:
            provisional=Path(directory)
            test_policy={**policy,"evidence":{**policy["evidence"],"confirmation_passed":True}}
            (provisional/"manifest.json").write_text(json.dumps(test_policy))
            for name in SOURCE_FILES:shutil.copyfile(Path(operation.__file__).parent/"metal"/name,provisional/name)
            start=time.perf_counter();dispatcher=Normalizer(provisional)
            dispatcher.artifact_load_ms=(time.perf_counter()-start)*1000
            if dispatcher.status!="installed":raise ValueError(dispatcher.status)
        result['dispatcher']=dispatcher
    return result


def confirm(session):
    mx=setup()
    import numpy as np
    from kerneldarwin.workload import environment_identity,source_identity
    protocol=read("protocol.json");policy=read("policy-frozen.json")
    if session not in (0,1,2):raise ValueError("session must be 0,1,2")
    target=EVIDENCE/f"confirmation-{session}.json"
    if target.exists():raise FileExistsError(target)
    if environment_identity()!=policy["environment"] or source_identity()!=policy["source_sha256"]:
        raise ValueError("Source or environment changed since tuning")
    if policy['study_identity']!=current_implementation() or protocol['study_identity']!=current_implementation():
        raise ValueError('Study implementation changed since freeze/tuning')
    seed=protocol["confirmation"]["seeds"][session]
    if digest(EVIDENCE/'corpus.json')!=protocol['corpus_sha256'] or policy['evidence']['corpus_sha256']!=protocol['corpus_sha256']:raise ValueError('Frozen corpus changed')
    texts=read("corpus.json")["chunks"]
    tasks=[*protocol["tasks"],protocol["confirmation"]["unseen_task"]]
    order=protocol["confirmation"]["sequence_orders"][session]
    functions=policies(policy);rng=random.Random(seed)
    before=telemetry();cold={};normal_samples={n:[] for n in functions};app_samples={n:[] for n in functions}
    checks=[];orders=[]
    # Warm and verify each full application with fresh contents before timing.
    prepared=[arrays(task,texts,seed) for task in tasks]
    for name,f in functions.items():
        cold[name]=measure(lambda:[application(f,t,*prepared[i]) for i,t in enumerate(tasks)])/1000
        for i,task in enumerate(tasks):
            actual=application(f,task,*prepared[i])
            check_application(actual,*prepared[i])
            checks.append({"task":task["id"],"method":name,"scores_passed":True,"ids_association_passed":True})
    for repeat in range(protocol["confirmation"]["repeats_per_session"]):
        # Contents change every repetition. Generation is excluded from the
        # compute API timing, but input MLX allocation and materialization count.
        prepared=[arrays(task,texts,seed+repeat+1) for task in tasks]
        op_inputs=operation_inputs(tasks,texts,seed+repeat+1)
        for item in op_inputs:item["staged"]=burst_inputs(item["x"],protocol["measurement_batch"],item["layout"])
        names=list(functions);rng.shuffle(names);orders.append(names)
        for name in names:
            f=functions[name]
            normal_samples[name].append(sum(item["weight"]*measure_burst(f,item["staged"]) for item in op_inputs))
            app_samples[name].append(measure(lambda:[[application(f,tasks[i],*prepared[i]) for i in order] for _ in range(protocol["application_measurement_batch"])] )/protocol["application_measurement_batch"])
        if (repeat+1)%10==0:print('session',session,'repeat',repeat+1,flush=True)
    result={"session":session,"seed":seed,"sequence_order":order,"source_sha256":source_identity(),"study_identity":current_implementation(),"corpus_sha256":protocol['corpus_sha256'],"protocol_sha256":digest(EVIDENCE/"protocol.json"),
        "policy_sha256":digest(EVIDENCE/"policy-frozen.json"),"environment":environment_identity(),"telemetry_before":before,"telemetry_after":telemetry(),
        "first_sequence_in_session_ms":cold,"artifact_load_ms":functions['dispatcher'].artifact_load_ms,
        "cold_note":"Not a process-cold comparison: preceding methods can share compilation. Use cold-study.json for independent process starts.",
        "normalization_samples_us":normal_samples,"application_samples_us":app_samples,"orders":orders,"checks":checks,
        "peak_memory_bytes":mx.get_peak_memory(),"fallback_counts":{name:{"calls":getattr(f,"calls",None),"fallback_calls":getattr(f,"fallback_calls",None)} for name,f in functions.items()},
        "comparisons":{name:{"normalization":intervals(normal_samples["library"],normal_samples[name],seed),"application":intervals(app_samples["library"],app_samples[name],seed)} for name in functions if name!="library"}}
    save(target,result)
    print(json.dumps(result["comparisons"],indent=2))


def cold(method):
    start=time.perf_counter();mx=setup();runtime_ms=(time.perf_counter()-start)*1000
    protocol=read('protocol.json');policy=read('policy-frozen.json')
    from kerneldarwin.workload import environment_identity
    before=time.perf_counter();environment=environment_identity();probe_ms=(time.perf_counter()-before)*1000
    if environment!=policy['environment']:raise ValueError('Cold-study runtime changed')
    if current_implementation()!=policy['study_identity']:raise ValueError('Stale cold-study implementation')
    if digest(EVIDENCE/'corpus.json')!=protocol['corpus_sha256']:raise ValueError('Stale cold-study corpus')
    texts=read('corpus.json')['chunks'];tasks=[*protocol['tasks'],protocol['confirmation']['unseen_task']]
    before=time.perf_counter();functions=policies(policy,requested=method);load_ms=(time.perf_counter()-before)*1000
    before=time.perf_counter();prepared=[arrays(task,texts,9661) for task in tasks];prepare_ms=(time.perf_counter()-before)*1000
    first_ms=measure(lambda:[application(functions[method],task,*prepared[i]) for i,task in enumerate(tasks)])/1000
    print(json.dumps({'method':method,'source_sha256':policy['source_sha256'],'study_identity':policy['study_identity'],
        'runtime_setup_ms':runtime_ms,'environment_probe_ms':probe_ms,'policy_setup_ms':load_ms,'input_prepare_ms':prepare_ms,'first_application_sequence_ms':first_ms,
        'peak_memory_bytes':mx.get_peak_memory(),'environment':environment,
        'cache_note':'Fresh Python and MLX process; shared OS/driver caches are not deleted. First-call work includes compilation/dispatch; pure compiler time is not isolated.'}))


def cold_study():
    protocol=read('protocol.json');policy=read('policy-frozen.json');rng=random.Random(667)
    methods=list(METHODS);rows=[];start=time.perf_counter()
    for repeat in range(3):
        order=methods.copy();rng.shuffle(order)
        for method in order:
            before=time.perf_counter()
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'cold','--method',method,'--directory',str(EVIDENCE)],capture_output=True,text=True,timeout=60,cwd=ROOT)
            wall_ms=(time.perf_counter()-before)*1000
            if child.returncode:raise RuntimeError('Cold worker failed: '+child.stderr[-2000:])
            value=json.loads(child.stdout);value.update(repeat=repeat,process_wall_ms=wall_ms);rows.append(value)
            print('cold',repeat,method,round(wall_ms,1),'ms',flush=True)
            if time.perf_counter()-start>180:raise TimeoutError('Cold-start study exceeded 180-second budget')
    save(EVIDENCE/'cold-study.json',{'schema':1,'source_sha256':policy['source_sha256'],'study_identity':current_implementation(),
        'corpus_sha256':protocol['corpus_sha256'],'policy_sha256':digest(EVIDENCE/'policy-frozen.json'),'rows':rows,
        'wall_seconds':time.perf_counter()-start,
        'method':'24 serialized independent process starts, randomized method order; compilation/first dispatch and process/input setup reported separately; process wall includes harness identity probes',
        'telemetry':telemetry()})


def main():
    global EVIDENCE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','tune','confirm','cold','cold-study']);p.add_argument('--session',type=int,default=0)
    p.add_argument('--method',choices=METHODS,default='dispatcher')
    p.add_argument('--directory',type=Path,default=ROOT/"evidence/workload-2026-10-29/study-v1")
    args=p.parse_args()
    EVIDENCE=args.directory.resolve()
    if args.stage=='freeze':freeze()
    elif args.stage=='tune':tune()
    elif args.stage=='confirm':confirm(args.session)
    elif args.stage=='cold':cold(args.method)
    else:cold_study()


if __name__=='__main__':main()
