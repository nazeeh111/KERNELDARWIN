#!/usr/bin/env python3
"""Execute the documented NPZ quickstart and explicit CLI failure paths."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh result file')
    import numpy as np
    import kerneldarwin
    from kerneldarwin.workload import source_identity
    from kerneldarwin.study_identity import implementation_identity
    package=Path(kerneldarwin.__file__).resolve()
    if 'site-packages' not in package.parts:raise RuntimeError('Use an independently installed wheel')
    environment=dict(os.environ)
    environment.pop('PYTHONPATH',None)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='kd-cli-inputs-') as directory:
        directory=Path(directory)
        rng=np.random.default_rng(27)
        documents=rng.standard_normal((64,384),dtype=np.float32)
        queries=documents[[3,27]]+np.float32(.01)*rng.standard_normal((2,384),dtype=np.float32)
        previous=None
        for change in (0,1):
            q=queries if change==0 else -queries+.01
            path=directory/f'changed-{change}.npz'
            np.savez(path,documents=documents,queries=q)
            result=subprocess.run([sys.executable,'-m','kerneldarwin','workload','search',str(path),'--k','5'],capture_output=True,text=True,env=environment,timeout=60)
            if result.returncode:raise RuntimeError(result.stderr)
            record=json.loads(result.stdout)
            ids=np.asarray(record['ids']);scores=np.asarray(record['scores'])
            d=documents.astype(np.float64);d/=np.linalg.norm(d,axis=1,keepdims=True)
            reference=q.astype(np.float64);reference/=np.linalg.norm(reference,axis=1,keepdims=True)
            reference=reference@d.T
            if ids.shape!=(2,5) or any(len(set(row.tolist()))!=5 for row in ids) or np.any(ids<0) or np.any(ids>=64):raise AssertionError('Wrong result identifiers')
            np.testing.assert_allclose(scores,np.take_along_axis(reference,ids,axis=1),rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(scores,-np.sort(-reference,axis=1)[:,:5],rtol=2e-5,atol=2e-6)
            if record['artifact_status']!='library_only' or record['fallback_calls']!=1:raise AssertionError('Expected explicit whole-query library fallback')
            if previous is not None and record['scores']==previous:raise AssertionError('Changed input returned stale scores')
            previous=record['scores']
            checks.append({'kind':'changed_npz','change':change,'independent_scores_and_ids':True,'artifact_status':record['artifact_status'],'fallback_calls':record['fallback_calls']})
        for name,docs,query,error in (('integer',documents.astype(np.int32),queries.astype(np.int32),'dtype'),('width',documents,queries[:,:-1],'width'),('empty_index',documents[:0],queries,'nonempty')):
            path=directory/f'{name}.npz';np.savez(path,documents=docs,queries=query)
            result=subprocess.run([sys.executable,'-m','kerneldarwin','workload','search',str(path)],capture_output=True,text=True,env=environment,timeout=60)
            if result.returncode!=1 or error not in result.stderr:raise AssertionError('CLI did not reject '+name+': '+result.stderr)
            checks.append({'kind':name,'returncode':result.returncode,'explicit_error':result.stderr.strip()})
    record={'schema':1,'success':True,'gpu_executed':True,'package_location':str(package),'version':kerneldarwin.__version__,
        'source_sha256':source_identity(),'study_identity':implementation_identity(),'checks':checks,'limitation':'A correctness/CLI check does not establish acceleration.'}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'success':True,'changed_input_checks':2,'failure_checks':3}))

if __name__=='__main__':main()
