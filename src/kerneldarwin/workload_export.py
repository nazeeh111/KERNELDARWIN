"""Validate a completed frozen study before copying an installable plan.

This is a reproducibility check, not a cryptographic attestation of local records.
Only the installed authored operation is copied; records cannot supply code.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import tempfile
import statistics
import math
from typing import Any

METHODS = ('library','eager_library','fast_library','compiled_application','fast_compiled_application','best_fixed','per_shape','dispatcher')
DEVELOPMENT_METHODS = (*METHODS[:5], 'simd','simd4','vector4','group128')
LIMITS = {'candidates_per_shape':4,'portfolio_kernels':2,'max_rules':128,'source_bytes':12000,
          'development_wall_seconds':900,'memory_bytes':2*1024**3,'cache_bytes':256*1024**2}
SEEDS = [7101,8203,9307]
SEQUENCES = [[2,0,1,3],[1,3,2,0],[3,0,2,1]]


def _read(path: Path) -> dict[str,Any]:
    if path.stat().st_size>20_000_000:raise ValueError("Evidence file exceeds 20 MB")
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    if path.stat().st_size>20_000_000:raise ValueError('Evidence file exceeds 20 MB')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(a: list[float], b: list[float], seed: int = 404) -> dict[str,Any]:
    import math
    if len(a)!=len(b) or not 40<=len(a)<=1000 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in [*a,*b]):
        raise ValueError("Need at least 40 finite positive paired costs")
    if all(v==a[0] for v in a) and all(v==b[0] for v in b):
        ratio=sum(a)/sum(b)
        return {'speedup_total_cost':ratio,'ci95':[ratio,ratio],
                'baseline_total_us':sum(a),'candidate_total_us':sum(b)}
    rng=random.Random(seed);ratios=[]
    for _ in range(4000):
        ids=rng.choices(range(len(a)),k=len(a))
        ratios.append(sum(a[i] for i in ids)/sum(b[i] for i in ids))
    ratios.sort()
    return {"speedup_total_cost":sum(a)/sum(b),"ci95":[ratios[100],ratios[-101]],
            "baseline_total_us":sum(a),"candidate_total_us":sum(b)}


def _costs(values: Any, count: int = 40) -> None:
    if not isinstance(values,list) or len(values)!=count or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=1e12 for v in values):
        raise ValueError('Need complete finite positive paired costs')


def _resource(value: Any, maximum: float, *, zero: bool = False) -> bool:
    return type(value) in (int,float) and math.isfinite(value) and (0<=value if zero else 0<value) and value<=maximum


def _protocol(protocol: dict[str,Any], policy: dict[str,Any], corpus: dict[str,Any]) -> list[dict[str,Any]]:
    """Reconstruct the prospective protocol instead of trusting its own gates."""
    from .workload import CANDIDATES,MAX_ELEMENTS,_LOADED_SOURCE
    from . import study_identity
    chunks=corpus.get('chunks')
    if not isinstance(chunks,list) or len(chunks)<64 or any(not isinstance(v,str) for v in chunks):
        raise ValueError('Invalid frozen corpus')
    tasks=[{'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},
           {'id':'manuals','rows':min(256,len(chunks)),'width':768,'queries':8,'layout':'column_stride2'},
           {'id':'archive','rows':len(chunks),'width':1536,'queries':17,'layout':'transpose'}]
    if any(not 0<task[key]*task['width']<=MAX_ELEMENTS or not 0<task['width']<=2048 for task in tasks for key in ('rows','queries')):
        raise ValueError('Frozen development shapes exceed candidate bounds')
    confirmation={'sessions':3,'repeats_per_session':40,'seeds':SEEDS,'sequence_orders':SEQUENCES,
                  'unseen_task':{'id':'unseen','rows':97,'width':513,'queries':3,'layout':'column_stride2'}}
    pin=_read(Path(study_identity.__file__).parent/'study-source.json')['measurement_script_sha256']
    if protocol.get('schema')!=1 or protocol.get('tasks')!=tasks or protocol.get('corpus_chunks')!=len(chunks) or protocol.get('development')!={'seed':101,'repeats':40} or protocol.get('measurement_batch')!=32 or protocol.get('application_measurement_batch')!=4 or protocol.get('confirmation')!=confirmation or protocol.get('script_sha256')!=pin:
        raise ValueError('Changed prospective workload/session/measurement protocol')
    if protocol.get('limits')!=LIMITS or sum(len(source) for _,source in _LOADED_SOURCE)>LIMITS['source_bytes']:
        raise ValueError('Changed prospective resource or source budget')
    noise=protocol.get('baseline_noise')
    if not isinstance(noise,list) or len(noise)!=3 or [r.get('task') for r in noise]!=tasks:
        raise ValueError('Incomplete prospective baseline noise checks')
    worst=0.
    for row in noise:
        _costs(row['baseline_a_us']);_costs(row['baseline_b_us'])
        low,high=_interval(row['baseline_a_us'],row['baseline_b_us'])['ci95']
        if low<.9 or high>1.1:raise ValueError('Prospective identical-baseline stability gate failed')
        worst=max(worst,abs(low-1),abs(high-1))
    criterion={'normalization_ci95_lower_speedup':max(1.10,1+2*worst),'application_claim_ci95_lower_speedup':1.03,
               'portfolio_over_fixed_minimum_total_cost_gain':.02,'numerical_rtol':2e-5,'numerical_atol':2e-6}
    if protocol.get('criterion')!=criterion:raise ValueError('Changed prospective acceptance criterion')
    sources={name:hashlib.sha256(content).hexdigest() for name,content in _LOADED_SOURCE}
    if policy.get('schema')!=1 or policy.get('operation')!='stable_l2_rows_v1' or policy.get('epsilon')!=1e-12 or policy.get('sources')!=sources:
        raise ValueError('Incompatible policy operation or authored source')
    rules=policy.get('rules')
    if not isinstance(rules,list) or len(rules)>128:raise ValueError('Invalid dispatch rule count')
    seen=set();candidates=set()
    known={(t[key],t['width']) for t in tasks for key in ('rows','queries')}
    for rule in rules:
        shape=rule.get('shape')
        if not isinstance(shape,list) or len(shape)!=2 or any(type(v) is not int for v in shape) or not 0<shape[0]*shape[1]<=MAX_ELEMENTS or not 0<shape[1]<=2048 or tuple(shape) not in known or tuple(shape) in seen:
            raise ValueError('Invalid or duplicate bounded dispatch shape')
        if rule.get('candidate') not in CANDIDATES or rule.get('dtype')!='float32' or rule.get('layout')!='any_mlx_strides':
            raise ValueError('Unsupported dispatch contract')
        seen.add(tuple(shape));candidates.add(rule['candidate'])
    if len(candidates)>2:raise ValueError('Portfolio exceeds two authored kernels')
    return tasks


def verify_study(directory: Path) -> dict[str,Any]:
    from .workload import source_identity,environment_identity
    from .study_identity import implementation_identity
    directory=Path(directory)
    protocol=_read(directory/"protocol.json");policy=_read(directory/"policy-frozen.json")
    if policy["evidence"].get("numerical_passed") is not True:
        raise ValueError("Policy has not passed numerical verification")
    if policy["evidence"]["protocol_sha256"]!=_sha(directory/"protocol.json"):
        raise ValueError("Frozen protocol changed")
    if policy["source_sha256"]!=source_identity() or policy["environment"]!=environment_identity():
        raise ValueError("Operation source or runtime changed since study")
    if protocol.get('environment')!=policy['environment']:
        raise ValueError('Frozen protocol runtime changed')
    if policy['study_identity']!=implementation_identity() or protocol['study_identity']!=implementation_identity():
        raise ValueError('Application, library baseline or measurement source changed')
    corpus_sha=_sha(directory/'corpus.json')
    if corpus_sha!=protocol['corpus_sha256'] or corpus_sha!=policy['evidence']['corpus_sha256']:
        raise ValueError('Frozen workload corpus changed')
    tasks=_protocol(protocol,policy,_read(directory/'corpus.json'))
    numerical=_read(directory/"numerical-validation.json")
    if numerical.get('study_identity')!=implementation_identity():raise ValueError('Numerical application/baseline/study source changed')
    if numerical.get("source_sha256")!=policy["source_sha256"] or numerical.get("environment")!=policy["environment"] or numerical.get("success") is not True or numerical.get("gpu_executed") is not True or type(numerical.get('tests_run')) is not int or numerical['tests_run']<6 or any(type(numerical.get(k)) is not int or numerical[k]!=0 for k in ("failures","errors","skipped")):
        raise ValueError("Independent current-source GPU numerical validation missing or failed")
    if policy["evidence"].get("numerical_report_sha256")!=_sha(directory/"numerical-validation.json"):
        raise ValueError("Numerical report changed after tuning")
    development=_read(directory/"development.json")
    if development.get("source_sha256")!=policy["source_sha256"] or development.get("protocol_sha256")!=_sha(directory/"protocol.json"):
        raise ValueError("Development source or protocol changed")
    if development.get('study_identity')!=implementation_identity() or policy['evidence'].get('development_sha256')!=_sha(directory/'development.json'):
        raise ValueError('Development implementation or frozen bytes changed')
    if development.get('corpus_sha256')!=corpus_sha:raise ValueError('Development corpus mismatch')
    if not development.get("rows") or any(row.get("numerical_check",{}).get("passed") is not True for row in development["rows"]):
        raise ValueError("Development numerical evidence missing or failed")
    expected_rows=[{'id':task['id']+'-'+label,'shape':[task[key],task['width']],
                    'layout':task['layout'] if label=='index' else 'contiguous','weight':weight}
                   for task in tasks for label,key,weight in (('index','rows',1),('query','queries',3))]
    if len(development['rows'])!=6:raise ValueError('Incomplete development shape coverage')
    rng=random.Random(207)
    for row,expected in zip(development['rows'],expected_rows):
        if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('Changed development shape/layout/weight')
        if row['numerical_check'].get('rtol')!=2e-5 or row['numerical_check'].get('atol')!=2e-6:
            raise ValueError('Changed development numerical tolerance')
        if set(row['samples_us'])!=set(DEVELOPMENT_METHODS):raise ValueError('Incomplete candidate/library development samples')
        for samples in row['samples_us'].values():_costs(samples)
        expected_orders=[]
        for _ in range(40):
            order=list(DEVELOPMENT_METHODS);rng.shuffle(order);expected_orders.append(order)
        if row['orders']!=expected_orders:raise ValueError('Changed paired development order')
    sessions=[]
    methods=set(METHODS)
    baselines=('library','fast_library','compiled_application','fast_compiled_application')
    expected_checks={(task['id'],method) for task in [*protocol['tasks'],protocol['confirmation']['unseen_task']] for method in methods}
    for i in range(3):
        result=_read(directory/f"confirmation-{i}.json")
        if result["session"]!=i or result["seed"]!=protocol["confirmation"]["seeds"][i]:raise ValueError("Wrong independent session identity")
        if result['sequence_order']!=protocol['confirmation']['sequence_orders'][i]:raise ValueError('Wrong untouched sequence order')
        if result["source_sha256"]!=policy["source_sha256"] or result["environment"]!=policy["environment"]:
            raise ValueError("Confirmation source/runtime mismatch")
        if result.get('study_identity')!=implementation_identity():raise ValueError('Confirmation application/baseline/study source mismatch')
        if result.get('corpus_sha256')!=corpus_sha:raise ValueError('Confirmation corpus mismatch')
        if result["protocol_sha256"]!=_sha(directory/"protocol.json") or result["policy_sha256"]!=_sha(directory/"policy-frozen.json"):
            raise ValueError("Confirmation plan/protocol changed")
        if not result["checks"] or any(c.get("scores_passed") is not True or c.get('ids_association_passed') is not True for c in result["checks"]):raise ValueError("Application scores or top-k ID association correctness failed")
        if len(result['checks'])!=len(expected_checks) or {(c['task'],c['method']) for c in result['checks']}!=expected_checks:
            raise ValueError('Incomplete application/method numerical checks')
        for key in ('normalization_samples_us','application_samples_us'):
            if set(result[key])!=methods or any(len(v)!=protocol['confirmation']['repeats_per_session'] for v in result[key].values()):
                raise ValueError('Incomplete paired method samples')
            for samples in result[key].values():_costs(samples)
        rng=random.Random(result['seed']);expected_orders=[]
        for _ in range(40):
            order=list(METHODS);rng.shuffle(order);expected_orders.append(order)
        if result.get('orders')!=expected_orders:raise ValueError('Changed paired confirmation order')
        counts=result["fallback_counts"]["dispatcher"]
        if counts["fallback_calls"]<1:raise ValueError("Unseen-input fallback was not exercised")
        sessions.append(result)
    comparisons={}
    for stage,key in (("normalization","normalization_samples_us"),("application","application_samples_us")):
        combined={name:[v for s in sessions for v in s[key][name]] for name in sorted(methods)}
        comparisons[stage]={name:_interval(combined["library"],combined[name]) for name in combined if name!="library"}
        comparisons[stage]["dispatcher_vs_fixed"]=_interval(combined["best_fixed"],combined["dispatcher"])
        comparisons[stage]["dispatcher_vs_fast_library"]=_interval(combined["fast_library"],combined["dispatcher"])
    threshold=protocol["criterion"]["normalization_ci95_lower_speedup"]
    per_session=[{stage:{baseline:_interval(s[key][baseline],s[key]['dispatcher'],s['seed']) for baseline in baselines} for stage,key in (('normalization','normalization_samples_us'),('application','application_samples_us'))} for s in sessions]
    normalization_passed=all(s['normalization'][baseline]['ci95'][0]>threshold for s in per_session for baseline in baselines)
    application_passed=all(s['application'][baseline]['ci95'][0]>protocol['criterion']['application_claim_ci95_lower_speedup'] for s in per_session for baseline in baselines)
    # Independent sessions must each support the application direction. A pooled
    # interval cannot hide one session with a point-estimate regression.
    application_consistent=all(sum(s["application_samples_us"]["library"])>sum(s["application_samples_us"]["dispatcher"]) for s in sessions)
    resource_passed=_resource(development['search_wall_seconds'],900) and _resource(development['peak_memory_bytes'],LIMITS['memory_bytes'],zero=True) and all(_resource(s['peak_memory_bytes'],LIMITS['memory_bytes'],zero=True) for s in sessions)
    cold=_read(directory/'cold-study.json')
    if cold.get('study_identity')!=implementation_identity() or cold.get('corpus_sha256')!=corpus_sha or cold.get('policy_sha256')!=_sha(directory/'policy-frozen.json'):
        raise ValueError('Cold-start study source/data/policy mismatch')
    if len(cold['rows'])!=3*len(methods) or {(r['repeat'],r['method']) for r in cold['rows']}!={(i,m) for i in range(3) for m in methods}:
        raise ValueError('Incomplete independent cold-start study')
    for row in cold['rows']:
        if row.get('environment')!=policy['environment'] or row.get('study_identity')!=implementation_identity() or row.get('source_sha256')!=source_identity():raise ValueError('Cold-start worker identity mismatch')
        if not all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>0 for k in ('process_wall_ms','first_application_sequence_ms','policy_setup_ms')):
            raise ValueError('Invalid cold-start costs')
    resource_passed=resource_passed and _resource(cold['wall_seconds'],180) and all(_resource(r['peak_memory_bytes'],LIMITS['memory_bytes'],zero=True) for r in cold['rows'])
    cold_summary={m:{'median_process_wall_ms':statistics.median(r['process_wall_ms'] for r in cold['rows'] if r['method']==m),
        'median_first_sequence_ms':statistics.median(r['first_application_sequence_ms'] for r in cold['rows'] if r['method']==m)} for m in methods}
    warm_baseline=min(sum(s['application_samples_us'][m][i] for s in sessions for i in range(len(s['application_samples_us'][m]))) for m in baselines)
    warm_candidate=sum(sum(s['application_samples_us']['dispatcher']) for s in sessions)
    savings=(warm_baseline-warm_candidate)/sum(len(s['application_samples_us']['dispatcher']) for s in sessions)
    cold_extra=max(0,cold_summary['dispatcher']['median_process_wall_ms']-min(cold_summary[m]['median_process_wall_ms'] for m in baselines))*1000
    break_even=math.ceil((development['search_wall_seconds']*1e6+cold_extra)/savings) if savings>0 else None
    return {"schema":1,"numerical_passed":True,"normalization_passed":normalization_passed,"application_passed":application_passed and application_consistent,
        "confirmation_passed":normalization_passed and application_passed and application_consistent and resource_passed,
        "comparisons":comparisons,"per_session":per_session,"resource_passed":resource_passed,
        'cold_start':cold_summary,'amortization_break_even_application_sequences':break_even,
        "source_sha256":source_identity(),"environment":policy["environment"],
        "evidence_sha256":{p.name:_sha(p) for p in [directory/"protocol.json",directory/"policy-frozen.json",directory/"development.json",directory/"numerical-validation.json",directory/'corpus.json',directory/'cold-study.json',*[directory/f"confirmation-{i}.json" for i in range(3)]]},
        "limitations":["Host wall timings include dispatch/allocation/materialization and observed background contention.",
            "Pooled paired bootstrap is descriptive; all independent session directions and per-session normalization intervals are additionally gated.",
            "No learned selector, cross-device speedup or adoption claim."]}


def export_artifact(study: Path, output: Path) -> dict[str,Any]:
    from .workload import _LOADED_SOURCE
    verified=verify_study(study)
    if verified["confirmation_passed"] is not True:raise ValueError("Frozen confirmation failed; export refused. Keep the null result.")
    if output.exists():raise FileExistsError("Choose a new artifact directory")
    policy=_read(study/"policy-frozen.json")
    if not policy["rules"]:raise ValueError("No independently confirmed optimized dispatch rules")
    if sum(len(source) for _,source in _LOADED_SOURCE)>12_000:raise ValueError("Source-size budget exceeded")
    policy["evidence"]={**policy["evidence"],"confirmation_passed":True,"verification":verified["evidence_sha256"]}
    from .workload import Normalizer
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kd-export-',dir=output.parent) as directory:
        staging=Path(directory)/'artifact';staging.mkdir()
        for name,source in _LOADED_SOURCE:(staging/name).write_bytes(source)
        (staging/"manifest.json").write_text(json.dumps(policy,indent=2)+"\n")
        (staging/"verification.json").write_text(json.dumps(verified,indent=2)+"\n")
        if Normalizer(staging).status!='installed':raise ValueError('Exported plan failed its own runtime compatibility guards')
        if output.exists():raise FileExistsError('Output appeared while preparing artifact')
        staging.rename(output)
    return {"artifact":str(output),"source_sha256":verified["source_sha256"],"confirmation_passed":True,"bytes":sum(p.stat().st_size for p in output.iterdir())}
