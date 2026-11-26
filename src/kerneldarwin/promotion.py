"""Pure bounded promotion checks. Consistency is not execution attestation."""
import hashlib
import json
import math
import inspect
from pathlib import Path

# Bind the currently loaded evaluator implementation as well as recorded context.
EVALUATOR_IDENTITY=hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in ('promotion.py','statistics.py','engine.py','gpu.py','kernels.py','records.py'))).hexdigest()


from . import gpu as _gpu
_CUDA_SOURCE={'baseline':inspect.getsource(_gpu.reference),
              'candidate':Path(__file__).with_name('kernels.py').read_text()+'\n'+inspect.getsource(_gpu.candidate)}


def authored_source_pair():
    """Match gpu.source_pair text without importing optional Triton kernels."""
    return dict(_CUDA_SOURCE)


def cpu_reference_identity():
    from . import control
    sources={'baseline':inspect.getsource(control.reference),'candidate':inspect.getsource(control.batched)}
    digest=hashlib.sha256((sources['baseline']+sources['candidate']+inspect.getsource(control.run_control)).encode()).hexdigest()
    return sources,digest


def evidence_identity(record, candidate):
    # JSON encodes bool, integer and floating scalar types distinctly.
    value={key:record.get(key) for key in ('workload','environment','device','evidence_kind','provenance')}
    value['config']=candidate.get('config')
    value['source']=candidate.get('source')
    value['baseline']=record.get('baseline')
    value['evaluator_identity']=EVALUATOR_IDENTITY
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_phase(phase: dict, label: str) -> None:
    """Recompute the claimed statistical gate; never trust an accepted flag."""
    from .statistics import compare_paired
    if not isinstance(phase, dict):
        raise ValueError(f"{label} requires paired timing evidence")
    baseline = phase.get("baseline_us", [])
    candidate = phase.get("candidate_us", [])
    if not isinstance(baseline, list) or not isinstance(candidate, list) or len(baseline) != len(candidate) or not 20 <= len(baseline) <= 10000:
        raise ValueError(f"{label} requires 20..10000 matched timing pairs")
    for duration in baseline + candidate:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{label} timing samples must be positive finite numbers")
    comparison = phase.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"{label} requires comparison metadata")
    threshold = comparison.get("min_speedup")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 1.03:
        raise ValueError(f"{label} minimum speedup must be finite and >=1.03")
    seed, resamples = comparison.get("seed"), comparison.get("bootstrap_resamples")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError(f"{label} requires an integer bootstrap seed in [0, 2**64)")
    if type(resamples) is not int or not 100 <= resamples <= 10000 or resamples * len(baseline) > 2_000_000:
        raise ValueError(f"{label} bootstrap validation exceeds bounded resampling budget")
    if type(comparison.get("n")) is not int or comparison["n"] != len(baseline):
        raise ValueError(f"{label} sample count differs from raw samples")
    expected = compare_paired(baseline, candidate, seed=seed, min_speedup=threshold, resamples=resamples)
    if not expected["accepted"] or comparison.get("accepted") is not True:
        raise ValueError(f"{label} raw samples do not clear the declared improvement gate")
    for key in ("speedup", "ci_low", "ci_high", "confidence_level"):
        claimed = comparison.get(key)
        if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or not math.isclose(claimed, expected[key], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"{label} {key} differs from recomputed raw-sample statistics")
    if comparison.get("method") != expected["method"]:
        raise ValueError(f"{label} statistical method is unsupported or inconsistent")



def require_candidate(record, winner, *, confirmation=True):
    from .records import WorkloadSpec, CandidateConfig
    WorkloadSpec.from_dict(record.get('workload',{}))
    if record.get('status') not in ('accepted','no_improvement'):
        raise ValueError('Blocked or failed run cannot promote')
    if record.get('evidence_kind') not in ('cuda_run','cpu_control'):
        raise ValueError('Reference or unknown evidence cannot promote')
    if winner.get('status') in ('invalid','failed') or winner.get('correctness',{}).get('passed') is not True:
        raise ValueError('Candidate correctness missing or failed')
    validate_phase(winner.get('timing'), 'Discovery')
    if record['evidence_kind']=='cuda_run':
        config=winner.get('config')
        if type(config) is not dict or set(config)!={'block_size','num_warps'}:
            raise ValueError('Explicit CUDA candidate configuration required')
        CandidateConfig(**config)
        provenance=record.get('provenance',{})
        source=provenance.get('source_sha256')
        if record.get('device',{}).get('kind')!='cuda' or provenance.get('gpu_execution') is not True or not isinstance(source,str) or len(source)!=64 or any(c not in '0123456789abcdef' for c in source):
            raise ValueError('CUDA execution/source provenance missing or invalid')
        if record.get('baseline',{}).get('name')!='pytorch_eager':
            raise ValueError('Named PyTorch eager reference missing or inconsistent')
        sources=winner.get('source')
        if sources!=authored_source_pair() or hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()!=source:
            raise ValueError('Candidate source is missing, changed or inconsistent with declared/current CUDA source')
        baseline=record.get('compiled_baseline',{})
        if baseline.get('status')!='measured' or baseline.get('correctness_passed') is not True:
            raise ValueError('Correct compiled comparator is required; promotion inconclusive')
    else:
        sources,digest=cpu_reference_identity()
        provenance=record.get('provenance',{})
        baseline=record.get('baseline',{})
        if record.get('device',{}).get('kind')!='cpu' or baseline.get('name')!='numpy_float64_reference' or provenance.get('gpu_execution') is not False or provenance.get('source_sha256')!=digest or winner.get('source')!=sources or winner.get('correctness',{}).get('independent_float64_reference') is not True:
            raise ValueError('Named CPU reference/source/provenance missing, changed or inconsistent')
        if provenance.get('measurement')!='perf_counter_ns' or provenance.get('timing_mode')!='host_cpu' or not isinstance(record.get('environment',{}).get('numpy'),str) or not record['environment']['numpy']:
            raise ValueError('CPU reference runtime/measurement provenance missing')
        samples=baseline.get('samples_us')
        if type(samples) is not list or not 20<=len(samples)<=10000:raise ValueError('CPU reference requires20..10000 baseline samples')
        from .statistics import summarize
        summarize(samples)
    expected=evidence_identity(record,winner)
    for label,phase in [('Discovery',winner.get('timing'))]+([('Independent confirmation',winner.get('confirmation'))] if confirmation else []):
        if not isinstance(phase,dict) or phase.get('evidence_identity')!=expected:
            raise ValueError(label+' source/reference/workload/config/environment binding is missing or stale')
        if record['evidence_kind']=='cuda_run':
            compiled=phase.get('compiled')
            validate_phase(compiled,label+' compiled comparator')
            if compiled.get('correctness_passed') is not True:
                raise ValueError(label+' compiled correctness missing or failed')
    if not confirmation:return
    confirmation = winner.get("confirmation")
    validate_phase(confirmation, "Independent confirmation")
    if confirmation.get("correctness_passed") is not True:
        raise ValueError("Independent confirmation requires passing correctness")
    confirmation_seed = confirmation.get("seed")
    if type(confirmation_seed) is not int or not 0 <= confirmation_seed < 2**64:
        raise ValueError("Independent confirmation requires an explicit fresh input seed")
    discovery_seeds = {record["workload"].get("seed", 0)}
    for experiment in record.get("experiments",[]):
        discovery = experiment.get("timing", {})
        for seed in (discovery.get("seed"), discovery.get("comparison", {}).get("seed")):
            if type(seed) is int:
                discovery_seeds.add(seed)
        compiled_seed=discovery.get("compiled",{}).get("comparison",{}).get("seed")
        if type(compiled_seed) is int:discovery_seeds.add(compiled_seed)
        for test in experiment.get("correctness", {}).get("tests", []):
            if isinstance(test, dict) and type(test.get("seed")) is int:
                discovery_seeds.add(test["seed"])
    if confirmation_seed in discovery_seeds or confirmation["comparison"]["seed"] in discovery_seeds:
        raise ValueError("Independent confirmation seeds must differ from discovery seeds")
    if record["evidence_kind"]=="cuda_run" and confirmation["compiled"]["comparison"]["seed"] in discovery_seeds:
        raise ValueError("Compiled confirmation seed must differ from discovery seeds")
    declared_strides = record["workload"].get("actual_strides")
    confirmed_strides = confirmation.get("actual_strides")
    if declared_strides is not None and confirmed_strides != declared_strides:
        raise ValueError("Independent confirmation must preserve actual workload strides")
    if confirmed_strides is not None:
        from .records import WorkloadSpec
        WorkloadSpec.from_dict({**record["workload"], "actual_strides": confirmed_strides})


def promotion_decision(record, candidate_id, *, confirmation=True):
    """Return a conservative decision without mutating records or trusting flags."""
    try:
        record=json.loads(json.dumps(record,allow_nan=False))
        matches=[e for e in record.get('experiments',[]) if e.get('id')==candidate_id]
        if len(matches)!=1:raise ValueError('Candidate id must identify one experiment')
        require_candidate(record,matches[0],confirmation=confirmation)
        return {'eligible':True,'status':'confirmed' if confirmation else 'nominated',
                'candidate_id':candidate_id,'scope':'cuda_eager_and_compiled' if record['evidence_kind']=='cuda_run' else 'cpu_named_reference',
                'execution_attested':False}
    except (ValueError,KeyError,TypeError,OverflowError,RecursionError,AttributeError) as exc:
        return {'eligible':False,'status':'inconclusive' if 'required' in str(exc) or 'missing' in str(exc) else 'refused',
                'candidate_id':candidate_id,'reason':str(exc),'execution_attested':False}
