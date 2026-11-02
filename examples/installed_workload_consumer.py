"""Run after installing the wheel, from a directory without the repository.

This checks changed-input arithmetic. Candidate execution is an explicit offline
numerical investigation, not an accepted dispatch plan or a timing claim.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifact',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh evidence output path')
    import numpy as np
    import mlx.core as mx
    import kerneldarwin
    from kerneldarwin.workload import Normalizer,CANDIDATES,candidate_normalize,environment_identity,source_identity
    from kerneldarwin.study_identity import implementation_identity
    from kerneldarwin.search import CosineIndex
    from kerneldarwin.library_baselines import eager_normalize,fast_library_normalize,FullGraphLibrary
    mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3)
    rng=np.random.default_rng(9167)
    normalizer=Normalizer(args.artifact)
    doc_shape=(41,33);query_rows=3
    if args.artifact:
        manifest=args.artifact/'manifest.json' if args.artifact.is_dir() else args.artifact
        plan=json.loads(manifest.read_text());rules=plan.get('rules',[])
        known=next((r['shape'] for r in rules if r['shape'][0]>=5),None)
        if known:doc_shape=tuple(known)
        elif rules:doc_shape=(41,rules[0]['shape'][1]);query_rows=rules[0]['shape'][0]
    checks=[]
    for invocation in range(3):
        docs=rng.normal(size=doc_shape).astype(np.float32)
        queries=rng.normal(size=(query_rows,doc_shape[1])).astype(np.float32)
        # One huge row and one zero row exercise stable semantics on real calls.
        docs[0]*=1e30;docs[1]=0
        d64=docs.astype(np.float64);q64=queries.astype(np.float64)
        d64=d64/np.maximum(np.linalg.norm(d64,axis=1,keepdims=True),1e-12)
        q64=q64/np.maximum(np.linalg.norm(q64,axis=1,keepdims=True),1e-12)
        expected=np.sort(q64@d64.T,axis=-1)[:,-5:][:,::-1]
        implementations={'installed_dispatch':normalizer,'eager_library':eager_normalize,'fast_library':fast_library_normalize,
                         'compiled_application':FullGraphLibrary('stable'),'fast_compiled_application':FullGraphLibrary('fast'),
                         **{name:lambda x,name=name:candidate_normalize(x,name) for name in CANDIDATES}}
        for name,normalize in implementations.items():
            index=normalize.index(mx.array(docs)) if isinstance(normalize,FullGraphLibrary) else CosineIndex(mx.array(docs),normalize)
            ids,scores=index.search(mx.array(queries),k=5)
            np.testing.assert_allclose(scores,expected,rtol=2e-5,atol=2e-6)
            for row in ids:assert len(set(row))==5 and all(type(i) is int and 0<=i<len(docs) for i in row)
            np.testing.assert_allclose(scores,np.take_along_axis(q64@d64.T,np.array(ids),axis=1),rtol=2e-5,atol=2e-6)
            checks.append({'invocation':invocation,'implementation':name,'passed':True,'input_sha256':hashlib.sha256(docs.tobytes()+queries.tobytes()).hexdigest()})
    # Malformed plans must never activate kernels. Unconfirmed, changed-source,
    # changed-environment and unknown-input cases have separate GPU guard tests.
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'manifest.json';path.write_text('{"schema":999}')
        bad=Normalizer(path)
        np.testing.assert_allclose(np.array(bad(mx.array([[3.,4.]]))),[[.6,.8]],rtol=2e-5,atol=2e-6)
        assert bad.fallback_calls==1 and bad.last_decision.implementation=='library'
        checks.append({'malformed_plan_fallback':True})
    np.testing.assert_allclose(np.array(normalizer(mx.array([[3.,4.]]))),[[.6,.8]],rtol=2e-5,atol=2e-6)
    if args.artifact:
        assert normalizer.status=='installed',normalizer.status
        assert normalizer.calls>normalizer.fallback_calls
    result={'schema':1,'kind':'installed_gpu_numerical_consumer','package_version':importlib.metadata.version('kerneldarwin'),
        'package_location':str(Path(kerneldarwin.__file__).parent),'environment':environment_identity(),'source_sha256':source_identity(),'study_identity':implementation_identity(),
        'checks':checks,'changed_inputs':True,'gpu_executed':True,'artifact_status':normalizer.status,
        'custom_dispatch_calls':normalizer.calls-normalizer.fallback_calls,'fallback_calls':normalizer.fallback_calls,
        'development_caches':'No source tree, tuning records, private files or Python/MLX process caches supplied. Shared OS driver caches are not deleted.',
        'performance_claim':False,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(checks),'artifact_status':normalizer.status,'source_sha256':source_identity()}))


if __name__=='__main__':main()
