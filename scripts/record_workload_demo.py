#!/usr/bin/env python3
"""Record real installed CLI outputs for the read-only product walkthrough."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cwd',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh recording')
    environment=dict(os.environ)
    for name in ('PYTHONPATH','MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):environment.pop(name,None)
    rows=[]
    for command in (['workload','doctor'],['workload','demo','--query','Metal GPU numerical precision'],['workload','demo','--query','unknown shapes library fallback']):
        child=subprocess.run([str(args.python),'-m','kerneldarwin',*command],cwd=args.cwd,env=environment,capture_output=True,text=True,timeout=60)
        if child.returncode:raise RuntimeError('Installed CLI failed: '+child.stderr[-1000:])
        result=json.loads(child.stdout)
        if command[-1]=='doctor' and result.get('metal_executed') is not True:
            raise RuntimeError('No actual Metal probe in this recording')
        if command[1]=='demo' and (result.get('artifact_status')!='library_only' or result.get('fallback_calls')!=1):
            raise RuntimeError('Preview demo must clearly use the actual complete library fallback')
        rows.append({'command':'kd '+' '.join(command),'result':result})
    if rows[1]['result']['scores']==rows[2]['result']['scores']:
        raise AssertionError('Changed query did not change the computation')
    record={'schema':1,'kind':'recorded_installed_gpu_cli_execution','recorded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'live_browser_compute':False,'performance_claim':False,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,
        'note':'These are recorded outputs from the installed CLI on Apple M4. The browser does not execute GPU code. Single-call cold times are not performance evidence.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(record,handle,indent=2);handle.write('\n')
    print(json.dumps({'recorded':len(rows),'output':str(args.output),'actual_metal_probe':True,'queries_changed':True,'fallback_shown':True}))

if __name__=='__main__':main()
