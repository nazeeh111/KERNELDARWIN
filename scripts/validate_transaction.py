#!/usr/bin/env python3
"""Execute source-bound batch-transaction numerical checks on actual Metal."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
# An installed validation root deliberately has no src tree.
if (ROOT/'src').is_dir():sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve the previous numerical report')
    os.environ['KD_RUN_METAL']='1'
    path=ROOT/'tests/test_transaction_search.py'
    spec=importlib.util.spec_from_file_location('test_transaction_search',path)
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    import kerneldarwin.transaction_search as transaction
    from kerneldarwin import rank_search,search,library_baselines,workload,study_identity
    modules=(transaction,rank_search,search,library_baselines,workload,study_identity)
    methods=tests.TransactionSearchTests.completed_methods
    passed=(result.wasSuccessful() and result.testsRun>=12 and len(methods)>=10
            and tests.TransactionSearchTests.gpu_executed is True and not result.skipped)
    identity,environment,unavailable=None,None,None
    try:
        identity=transaction.transaction_source_identity()
        environment=workload.environment_identity()
    except (ImportError,ValueError,RuntimeError,OSError) as error:
        passed=False
        unavailable=type(error).__name__+': '+str(error)
    backend=getattr(tests.TransactionSearchTests,'mx',None)
    report={'kind':'transaction_production_numerical_validation','schema':1,
        'passed':passed,'tests_run':result.testsRun,'native_metal_methods_completed':methods,
        'cpu_contract_methods':result.testsRun-len(methods),'gpu_executed':tests.TransactionSearchTests.gpu_executed,
        'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
        'source_sha256':identity,'environment':environment,'unavailable_or_changed_source':unavailable,
        'elapsed_seconds':time.perf_counter()-start,'peak_memory_bytes':backend.get_peak_memory() if backend is not None else None,
        'validation_root':str(ROOT),'validation_root_src_exists':(ROOT/'src').exists(),
        'actual_imports':{m.__name__:{'path':m.__file__,'sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()} for m in modules},
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'validation_flags':{k:os.getenv(k) for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION')},
        'candidate_timing':False,'accepted_artifact':False,
        'scope':'Public compiled-library batch API and private numerical authored trials. Lower-precision cases are smoke checks, not full float32 adversarial coverage. Numerical correctness does not confirm performance or an artifact.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':passed,'tests_run':result.testsRun,'native_methods':len(methods),'source_sha256':report['source_sha256']}))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
