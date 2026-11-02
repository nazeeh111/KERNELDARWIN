#!/usr/bin/env python3
"""Independent installed-package execution, without a source tree or tuning data.

Copy only this script into a fresh directory, install the wheel, unset PYTHONPATH
and run with a new --output. An optional accepted --artifact must dispatch custom
code as well as exercise an unfamiliar tuple's whole-query library fallback.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh result file')
    import mlx.core as mx
    import numpy as np
    import kerneldarwin
    from kerneldarwin.fused_search import (FusedPlan,FusedCosineIndex,SCORE_CANDIDATES,candidate_scores,
        score_source_identity,fused_implementation_identity,environment_identity)
    from kerneldarwin.search import CompiledCosineIndex
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(2*1024**3)
    package=Path(kerneldarwin.__file__).resolve()
    if 'site-packages' not in package.parts:
        raise RuntimeError('Consumer must use an independently installed wheel')
    plan=FusedPlan(args.artifact)
    q,n,d=5,37,97
    if args.artifact:
        if plan.status!='installed' or not plan._rules:
            raise RuntimeError('An explicit artifact must load real confirmed custom rules')
        q,n,d=next(iter(plan._rules))
    records=[]
    def normalize(host):
        host=np.asarray(host,np.float64)
        peak=np.maximum(np.max(np.abs(host),axis=1,keepdims=True),1e-12)
        z=host/peak
        return z/np.maximum(np.sqrt(np.sum(z*z,axis=1,keepdims=True)),1e-12/peak)
    for seed in (290101,290203,290307):
        rng=np.random.default_rng(seed)
        docs=rng.normal(size=(n,d)).astype(np.float32)
        queries=rng.normal(size=(q,d)).astype(np.float32)
        # Caller changes contents and presents a strided raw-document view.
        backing=np.zeros((n,d*2),np.float32);backing[:,::2]=docs
        device_docs=mx.array(backing)[:,::2]
        device_queries=mx.array(queries)
        ref=normalize(queries)@normalize(docs).T
        methods={'installed_plan':plan.index(device_docs),
            'stable_library':CompiledCosineIndex(device_docs,'stable'),
            'fast_library':CompiledCosineIndex(device_docs,'fast'),
            **{name:FusedCosineIndex(device_docs,lambda q,v,name=name:candidate_scores(q,v,name)) for name in SCORE_CANDIDATES}}
        k=min(5,n)
        for name,index in methods.items():
            ids,values=index.search(device_queries,k=k)
            ids=np.asarray(ids);values=np.asarray(values)
            if ids.shape!=(q,k) or any(len(set(row.tolist()))!=k for row in ids) or np.any(ids<0) or np.any(ids>=n):
                raise AssertionError('Invalid document identifiers')
            np.testing.assert_allclose(values,np.take_along_axis(ref,ids,axis=1),rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(values,-np.sort(-ref,axis=1)[:,:k],rtol=2e-5,atol=2e-6)
            records.append({'seed':seed,'method':name,'score_and_id_checks_passed':True})
    custom_calls=plan.calls-plan.fallback_calls
    if args.artifact and custom_calls<3:
        raise AssertionError('The installed artifact did not execute its custom path on changed inputs')
    # A separate width/shape not present in the plan must use the whole query.
    before=plan.fallback_calls
    plan.index(mx.ones((2,2))).search(mx.ones((1,2)),k=1)
    if plan.fallback_calls!=before+1:
        raise AssertionError('Unknown tuple did not use fallback')
    if args.artifact:
        from tempfile import TemporaryDirectory
        import shutil
        with TemporaryDirectory(prefix='kd-consumer-invalid-plan-') as directory:
            dest=Path(directory)/'artifact';shutil.copytree(args.artifact,dest)
            manifest=json.loads((dest/'manifest.json').read_text());manifest['environment']={}
            (dest/'manifest.json').write_text(json.dumps(manifest))
            stale=FusedPlan(dest)
            if not stale.status.startswith('fallback:'):raise AssertionError('Changed environment was accepted')
            stale.index(mx.ones((2,2))).search(mx.ones((1,2)),k=1)
    result={'schema':1,'success':True,'gpu_executed':True,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'package_location':str(package),'version':kerneldarwin.__version__,
        'source_sha256':score_source_identity(),'study_identity':fused_implementation_identity(),'environment':environment_identity(),
        'artifact_status':plan.status,'shape':[q,n,d],'custom_dispatch_calls':custom_calls,'fallback_calls':plan.fallback_calls,
        'checks':records,'unknown_tuple_fallback_passed':True,
        'limitations':['Independent numerical execution is not a performance confirmation.','Two authored candidates exercised directly; only an accepted explicit artifact counts as installed custom dispatch.']}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'success':True,'checks':len(records),'artifact_status':plan.status,'custom_dispatch_calls':custom_calls,'fallback_calls':plan.fallback_calls}))

if __name__=='__main__':main()
