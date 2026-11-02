#!/usr/bin/env python3
"""Execute target-two Metal numerical checks, with immutable evidence output."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh validation report')
    os.environ['KD_RUN_METAL']='1'
    from kerneldarwin.fused_search import score_source_identity,fused_implementation_identity,environment_identity
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_fused_search.py')
    start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record={'schema':1,'kind':'executed_fused_metal_numerical_validation','source_sha256':score_source_identity(),
        'study_identity':fused_implementation_identity(),'environment':environment_identity(),
        'gpu_executed':True,'success':result.wasSuccessful() and not result.skipped,'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'wall_seconds':time.perf_counter()-start,'output':stream.getvalue(),
        'requested_validation_environment':{k:os.getenv(k) for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION')},
        'test_sources':{'test_fused_search.py':hashlib.sha256((ROOT/'tests/test_fused_search.py').read_bytes()).hexdigest()},
        'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'limitations':['F32 query normalization/scoring only; native subnormal flushing.','Normalized-document precondition is enforced by index refresh, not lower-level score calls.','Numerical validity does not prove workload acceleration.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(stream.getvalue());print(json.dumps({k:record[k] for k in ('success','tests_run','failures','errors','skipped','source_sha256')}))
    return 0 if record['success'] else 1

if __name__=='__main__':raise SystemExit(main())
