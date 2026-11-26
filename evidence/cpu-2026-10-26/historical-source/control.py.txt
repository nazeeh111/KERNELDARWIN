"""Actual NumPy CPU experiments for testing the evidence and learning path.

These are never presented as CUDA optimizations or GPU-training evidence.
The intervention changes NumPy row batch size, not a GPU launch parameter.
"""
from __future__ import annotations
import inspect
import hashlib
import json
import platform
import random
import time
import math
from .engine import new_record
from .statistics import summarize, compare_paired
from .hypotheses import diagnose


def reference(family,x,eps):
    import numpy as np
    # Higher precision independent reference then cast to declared output dtype.
    value=x.astype(np.float64)
    if family=='sum':out=value.sum(axis=1)
    elif family=='softmax':
        e=np.exp(value-value.max(axis=1,keepdims=True));out=e/e.sum(axis=1,keepdims=True)
    elif family=='layernorm':
        mean=value.mean(axis=1,keepdims=True);diff=value-mean
        out=diff/np.sqrt((diff*diff).mean(axis=1,keepdims=True)+eps)
    elif family=='fused':out=np.maximum(x.astype(np.float32)*1.5+.25,0)
    else:raise ValueError('Unsupported family')
    return out.astype(x.dtype)


def batched(family,x,eps,batch):
    import numpy as np
    out=np.empty(x.shape[0] if family=='sum' else x.shape,dtype=x.dtype)
    for start in range(0,x.shape[0],batch):
        value=x[start:start+batch].astype(np.float32)
        if family=='sum':part=value.sum(axis=1)
        elif family=='softmax':
            e=np.exp(value-value.max(axis=1,keepdims=True));part=e/e.sum(axis=1,keepdims=True)
        elif family=='layernorm':
            diff=value-value.mean(axis=1,keepdims=True);part=diff/np.sqrt((diff*diff).mean(axis=1,keepdims=True)+eps)
        elif family=='fused':part=np.maximum(value*1.5+.25,0)
        else:raise ValueError('Unsupported family')
        out[start:start+batch]=part
    return out


def time_pairs(a,b,repeats,seed):
    for _ in range(5):a();b()
    rng=random.Random(seed);left=[];right=[]
    for _ in range(repeats):
        order=[0,1];rng.shuffle(order);times={}
        for which in order:
            start=time.perf_counter_ns();output=(a if which==0 else b)();times[which]=(time.perf_counter_ns()-start)/1000
            del output
        left.append(times[0]);right.append(times[1])
    return left,right


def run_control(spec, *, repeats=40, batches=(32,128,512)):
    import numpy as np
    spec.validate()
    if spec.dtype=='bfloat16':raise ValueError('NumPy CPU control supports float32/float16 only')
    start=time.monotonic();rng=np.random.default_rng(spec.seed)
    dtype=np.float32 if spec.dtype=='float32' else np.float16
    x=rng.normal(size=(spec.shape[0],spec.shape[1]*spec.stride)).astype(dtype)[:,::spec.stride]
    record=new_record(spec,'cpu_control',{'kind':'cpu','name':platform.machine(),'processor':platform.processor()})
    record['environment']['numpy']=np.__version__
    record['workload']['actual_strides']=[v//x.itemsize for v in x.strides]
    baseline=lambda:reference(spec.family,x,spec.eps)
    samples,_=time_pairs(baseline,baseline,repeats,spec.seed)
    record['baseline']={'name':'numpy_float64_reference','samples_us':samples,'summary':summarize(samples)}
    record['hypotheses']=diagnose(spec.to_dict(),record['baseline'],record['device'])
    record['selection']={'method':'exhaustive_cpu_control','note':'block_size means NumPy rows per batch; num_warps is an unused placeholder. Never transfer this policy to CUDA.'}
    for i,batch in enumerate(batches):
        exp_start=time.monotonic();valid=True;tests=[];max_error=0.
        for index,pattern in enumerate(('random','zero','constant','large','ramp')):
            rows=min(17,spec.shape[0]);cols=spec.shape[1]
            z=rng.normal(size=(rows,cols*spec.stride)).astype(dtype)[:,::spec.stride]
            if pattern=='zero':z.fill(0)
            elif pattern=='constant':z.fill(3.25)
            elif pattern=='large':z*=20
            elif pattern=='ramp':z[:]=np.linspace(-4,4,cols)
            original=z.copy();expected=reference(spec.family,z,spec.eps);actual=batched(spec.family,z,spec.eps,batch)
            max_error=max(max_error,float(np.max(np.abs(actual.astype(float)-expected.astype(float)))))
            passed=bool(np.isfinite(actual).all() and np.allclose(actual,expected,atol=.03 if spec.dtype=='float16' else 2e-4,rtol=.03 if spec.dtype=='float16' else 2e-4) and np.array_equal(original,z))
            tests.append({'pattern':pattern,'passed':passed,'shape':[rows,cols]});valid &= passed
        expected_full=reference(spec.family,x,spec.eps);actual_full=batched(spec.family,x,spec.eps,batch)
        full_ok=bool(np.isfinite(actual_full).all() and np.allclose(actual_full,expected_full,atol=.03 if spec.dtype=='float16' else 2e-4,rtol=.03 if spec.dtype=='float16' else 2e-4))
        valid &= full_ok;tests.append({'pattern':'full_shape_random','passed':full_ok,'shape':list(spec.shape)})
        a,b=time_pairs(baseline,lambda:batched(spec.family,x,spec.eps,batch),repeats,spec.seed+i+10)
        comparison=compare_paired(a,b,seed=spec.seed+i)
        exp={'id':f'cpu-{batch}','parent_id':None,'config':{'block_size':batch,'num_warps':1},'intervention':'numpy_row_batch',
             'status':'rejected','correctness':{'passed':valid,'tests':tests,'max_abs_error':max_error,'independent_float64_reference':True},
             'timing':{'baseline_us':a,'candidate_us':b,'summary':summarize(b),'comparison':comparison},
             'compile_us':0,'safety':{'status':'not_applicable','reason':'CPU NumPy control'},
             'source':{'baseline':inspect.getsource(reference),'candidate':inspect.getsource(batched)},
             'cost_seconds':time.monotonic()-exp_start}
        if not valid:exp['status']='invalid'
        record['experiments'].append(exp)
    eligible=[e for e in record['experiments'] if e['correctness']['passed'] and e['timing']['comparison']['accepted']]
    if eligible:
        best=max(eligible,key=lambda e:e['timing']['comparison']['ci_low'])
        holdout=rng.normal(size=(spec.shape[0],spec.shape[1]*spec.stride)).astype(dtype)[:,::spec.stride]
        a,b=time_pairs(lambda:reference(spec.family,holdout,spec.eps),lambda:batched(spec.family,holdout,spec.eps,best['config']['block_size']),repeats,spec.seed+7001)
        expected_hold=reference(spec.family,holdout,spec.eps);actual_hold=batched(spec.family,holdout,spec.eps,best['config']['block_size'])
        hold_correct=bool(np.isfinite(actual_hold).all() and np.allclose(actual_hold,expected_hold,atol=.03 if spec.dtype=='float16' else 2e-4,rtol=.03 if spec.dtype=='float16' else 2e-4))
        confirm=compare_paired(a,b,seed=spec.seed+7001)
        if not hold_correct:confirm.update(accepted=False,reason='Independent confirmation correctness failed')
        best['confirmation']={'baseline_us':a,'candidate_us':b,'comparison':confirm,'seed':spec.seed+7001,'correctness_passed':hold_correct,'actual_strides':[v//holdout.itemsize for v in holdout.strides]}
        if confirm['accepted']:
            best['status']='accepted';record['winner_id']=best['id']
    record['status']='accepted' if record['winner_id'] else 'no_improvement'
    record['diagnosis']={'summary':'Measured CPU control: NumPy row batching compared with an independent higher-precision reference. This is not a GPU benchmark.',
                         'limitations':['CPU timing and implementation differences do not establish GPU optimization gains.',
                                        'Reference float64 intermediates differ from candidate float32 intermediates within declared tolerance.',
                                        'CPU policy training exercises the learning pipeline only; no CUDA transfer has been established.',
                                        'No hardware counters or GPU trace available.']}
    record['provenance']['source_sha256']=hashlib.sha256((inspect.getsource(reference)+inspect.getsource(batched)+inspect.getsource(run_control)).encode()).hexdigest()
    record['budget']={'max_candidates':len(batches),'used_candidates':len(batches),'wall_seconds':time.monotonic()-start}
    return record
