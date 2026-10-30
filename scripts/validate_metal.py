#!/usr/bin/env python3
"""Run current-source numerical and guard tests; save actual GPU evidence."""
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh validation report')
    os.environ['KD_RUN_METAL']='1'
    from kerneldarwin.workload import source_identity,environment_identity
    from kerneldarwin.study_identity import implementation_identity
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test*workload.py')
    start=time.perf_counter();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record={'schema':1,'kind':'executed_metal_numerical_validation','source_sha256':source_identity(),'study_identity':implementation_identity(),'environment':environment_identity(),
        'gpu_executed':True,'success':result.wasSuccessful() and not result.skipped,'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'wall_seconds':time.perf_counter()-start,'output':stream.getvalue(),
        'requested_validation_environment':{k:os.getenv(k) for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION')},
        'test_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tests/test_metal_workload.py',ROOT/'tests/test_workload.py')},
        'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'limitations':['Numerical correctness is separate from performance coverage.','Float32 input subnormals follow observed Metal flush-to-zero.','No CUDA sanitizer or cross-device pass is claimed.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(stream.getvalue());print(json.dumps({k:record[k] for k in ('success','tests_run','failures','errors','skipped','source_sha256')}))
    return 0 if record['success'] else 1


if __name__=='__main__':raise SystemExit(main())
