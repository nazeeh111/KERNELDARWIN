"""Budgeted experiments over authored templates; records are the product."""
from __future__ import annotations
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import __version__
from .records import WorkloadSpec, ExperimentStore
from .statistics import summarize, compare_paired
from .hypotheses import diagnose


def new_record(spec, kind: str, device: dict) -> dict:
    return {'schema_version':1,'id':uuid.uuid4().hex,'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'status':'failed','title':f'{spec.family} · {" × ".join(map(str,spec.shape))}', 'evidence_kind':kind,
            'workload':spec.to_dict(),'device':device,'environment':{'python':platform.python_version(),'platform':platform.platform()},
            'baseline':{},'compiled_baseline':{'status':'unavailable','reason':'Not measured'},'hypotheses':[], 'experiments':[],
            'winner_id':None,'trace':{'status':'unavailable','events':[],'reason':'Not requested'},
            'diagnosis':{'summary':'Investigation pending','limitations':[]},
            'provenance':{'gpu_execution':kind=='cuda_run','measurement':'CUDA events' if kind=='cuda_run' else 'perf_counter_ns',
                          'timing_mode':'normal_launch' if kind=='cuda_run' else 'host_cpu', 'author':f'KERNELDARWIN {__version__}'}}


def blocked_record(spec, reason: str) -> dict:
    record = new_record(spec,'reference',{'name':platform.machine(),'kind':'cpu'})
    record.update(status='blocked',title='CUDA validation blocked',diagnosis={'summary':reason,'limitations':['No GPU timings, correctness or learned GPU policy have been validated on this host.']})
    record['provenance']['gpu_execution']=False
    return record


def observation_rows(records: list[dict]) -> list[dict]:
    rows=[]
    for record in records:
        if record.get('evidence_kind') not in ('cuda_run','cpu_control'):
            continue
        for exp in record.get('experiments',[]):
            comparison=exp.get('timing',{}).get('comparison',{})
            valid=exp.get('correctness',{}).get('passed',False)
            if (not comparison or 'speedup' not in comparison) and exp.get('status') not in ('invalid','failed'):
                continue
            rows.append({'workload':record['workload'],'candidate':exp['config'],
                         'reward':math.log(comparison['speedup']) if comparison and 'speedup' in comparison else 0.,'cost':max(.001,exp.get('cost_seconds',.001)),
                         'valid':valid,
                         'group':record['workload']['family'], 'device_kind':record['device']['kind'],
                         'record_id':record['id']})
    return rows


def investigate_cuda(spec, *, max_candidates=6, repeats=40, warmup=10, wall_seconds=180, model=None, trace_path=None) -> dict:
    from . import gpu
    spec.validate()
    if spec.shape[1]>8192:raise ValueError('CUDA row width must be <=8192')
    torch,triton=gpu.require_cuda()
    start=time.monotonic()
    record=new_record(spec,'cuda_run',gpu.device_info(torch))
    record['environment']['triton']=triton.__version__
    record['environment']['torch_compile_mode']='default, fullgraph=True'
    x=gpu.make_input(spec)
    record['workload']['actual_strides']=list(x.stride())
    baseline=lambda:gpu.reference(spec.family,x,spec.eps)
    configs=gpu.configurations(spec)
    if model is not None:
        configs=[row['candidate'] for row in model.rank({**spec.to_dict(),'device_kind':'cuda'},configs)]
        method='learned_ridge_ucb'
    else:
        # Deterministic sparse first pass. This is a heuristic, not a trained policy.
        configs.sort(key=lambda c:(abs(c['num_warps']-4),c['block_size']))
        method='rule_based'
    record['selection']={'method':method,'candidate_count':len(configs),'hardware_transfer_validated':False}
    # Establish baseline independently of candidate success.
    a,_=gpu.time_pairs(baseline,baseline,repeats=repeats,warmup=warmup,seed=spec.seed)
    record['baseline']={'name':'pytorch_eager','samples_us':a,'summary':summarize(a)}
    record['hypotheses']=diagnose(spec.to_dict(),record['baseline'],record['device'])
    try:
        compiled=torch.compile(lambda value:gpu.reference(spec.family,value,spec.eps),fullgraph=True)
        before=time.perf_counter_ns(); compiled(x);torch.cuda.synchronize()
        compile_us=(time.perf_counter_ns()-before)/1000
        compiled_a,compiled_b=gpu.time_pairs(baseline,lambda:compiled(x),repeats=repeats,warmup=warmup,seed=spec.seed+1)
        correct=bool(torch.allclose(compiled(x),baseline(),atol=.03 if spec.dtype!='float32' else 2e-4,rtol=.03 if spec.dtype!='float32' else 2e-4))
        record['compiled_baseline']={'status':'measured','samples_us':compiled_b,'paired_eager_us':compiled_a,
                                     'summary':summarize(compiled_b),'compile_us':compile_us,'correctness_passed':correct}
    except Exception as exc:
        record['compiled_baseline']={'status':'unavailable','reason':str(exc)}
    sources=gpu.source_pair()
    for index,config in enumerate(configs[:max_candidates]):
        if time.monotonic()-start>wall_seconds:
            break
        exp_start=time.monotonic()
        exp={'id':f'c{index:03d}','parent_id':None,'config':config,'intervention':'fusion' if spec.family=='fused' else 'row_tiling',
             'status':'failed','source':sources,'safety':{'status':'unverified','reason':'Run compute-sanitizer independently before production use.'}}
        try:
            before=time.perf_counter_ns();gpu.candidate(spec.family,config,x,spec.eps);torch.cuda.synchronize()
            exp['compile_us']=(time.perf_counter_ns()-before)/1000
            exp['correctness']=gpu.verify(spec,config)
            if not exp['correctness']['passed']:
                exp['status']='invalid'
            else:
                a,b=gpu.time_pairs(baseline,lambda:gpu.candidate(spec.family,config,x,spec.eps),repeats=repeats,warmup=warmup,seed=spec.seed+index+2)
                comparison=compare_paired(a,b,seed=spec.seed+index)
                # Discovery pass can nominate but never accept a winner by itself.
                exp['timing']={'baseline_us':a,'candidate_us':b,'summary':summarize(b),'comparison':comparison,'phase':'discovery'}
                exp['status']='rejected'
        except Exception as exc:
            exp['error']=f'{type(exc).__name__}: {exc}'
        exp['cost_seconds']=time.monotonic()-exp_start
        record['experiments'].append(exp)
    eligible=[e for e in record['experiments'] if e.get('correctness',{}).get('passed') and e.get('timing',{}).get('comparison',{}).get('accepted')]
    if eligible and time.monotonic()-start < wall_seconds:
        best=max(eligible,key=lambda e:e['timing']['comparison']['ci_low'])
        # Re-measure one selected candidate after search, with independent inputs.
        x_holdout=gpu.make_input(spec,seed=spec.seed+7001)
        held_base=lambda:gpu.reference(spec.family,x_holdout,spec.eps)
        a,b=gpu.time_pairs(held_base,lambda:gpu.candidate(spec.family,best['config'],x_holdout,spec.eps),repeats=repeats,warmup=warmup,seed=spec.seed+7001)
        expected_hold=held_base();actual_hold=gpu.candidate(spec.family,best['config'],x_holdout,spec.eps)
        hold_correct=bool(torch.isfinite(actual_hold).all() and torch.allclose(actual_hold,expected_hold,atol=.03 if spec.dtype!='float32' else 2e-4,rtol=.06 if spec.dtype=='bfloat16' else .03 if spec.dtype=='float16' else 2e-4))
        confirm=compare_paired(a,b,seed=spec.seed+7001)
        if not hold_correct:confirm.update(accepted=False,reason='Independent confirmation correctness failed')
        best['confirmation']={'baseline_us':a,'candidate_us':b,'comparison':confirm,'seed':spec.seed+7001,'correctness_passed':hold_correct,'actual_strides':list(x_holdout.stride())}
        if confirm['accepted']:
            best['status']='accepted';record['winner_id']=best['id']
    record['status']='accepted' if record['winner_id'] else 'no_improvement'
    record['diagnosis']={'summary':'A candidate passed independent correctness and discovery/confirmation timing gates.' if record['winner_id'] else 'No candidate passed both correctness and repeatable improvement gates.',
                         'limitations':['Hypotheses are analytical explanations, not counter-proven causal conclusions.',
                                        'Compute Sanitizer has not been run by this process.','No hardware counters collected.',
                                        'Only the declared finite-input semantics and tested shapes are supported.',
                                        'Selection on GPU records requires held-out evaluation before a learning claim.']}
    if trace_path:
        record['trace']=gpu.capture_trace(baseline,Path(trace_path))
    record['budget']={'max_candidates':max_candidates,'used_candidates':len(record['experiments']),'wall_seconds':time.monotonic()-start,'limit_seconds':wall_seconds,
                      'note':'Scheduler stops between candidates; CLI worker enforces hard timeout.'}
    record['provenance']['source_sha256']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    return record


def run_isolated(spec, output: Path, *, max_candidates=6,repeats=40,warmup=10,wall_seconds=180,trace=False,model_path=None) -> subprocess.CompletedProcess:
    """Finite authored kernels in a credential-free subprocess, with timeout.

    This is process isolation, not an adversarial sandbox. It does not claim to
    prevent arbitrary network syscalls. There is no arbitrary-source input.
    """
    safe_keys=('PATH','SYSTEMROOT','WINDIR','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES')
    env={key:os.environ[key] for key in safe_keys if key in os.environ}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]),PYTHONNOUSERSITE='1',
               TOKENIZERS_PARALLELISM='false',CUDA_MODULE_LOADING='LAZY')
    args=[sys.executable,'-m','kerneldarwin','_worker',json.dumps(spec.to_dict()),str(output.resolve()),
          '--max-candidates',str(max_candidates),'--repeats',str(repeats),'--warmup',str(warmup),'--wall-seconds',str(wall_seconds)]
    if trace:args.append('--trace')
    if model_path:args.extend(['--model',str(Path(model_path).resolve())])
    return subprocess.run(args,env=env,capture_output=True,text=True,timeout=wall_seconds+30)
