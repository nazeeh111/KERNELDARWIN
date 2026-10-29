#!/usr/bin/env python3
"""Profile the actual installed search path; trace timings are diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh profile output')
    if not 1<=args.iterations<=32:raise ValueError('Iterations must be 1..32')
    import numpy as np
    import mlx.core as mx
    from kerneldarwin.search import CosineIndex,text_vectors,_compiled_search
    from kerneldarwin.library_baselines import FullGraphLibrary
    from kerneldarwin.workload import library_normalize,candidate_normalize,source_identity,environment_identity
    from kerneldarwin.study_identity import implementation_identity
    mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
    texts=json.loads(args.corpus.read_text())['chunks'][:161]
    d=np.array(text_vectors(texts,768));q=np.array(text_vectors(texts[:8],768))
    inputs=[]
    for i in range(args.iterations):
        host=d+np.float32(i*.001);backing=np.zeros((len(d),1536),np.float32);backing[:,::2]=host
        docs=mx.array(backing)[:,::2];queries=mx.array(q+np.float32(i*.001));mx.eval(docs,queries)
        a=host.astype(np.float64);b=np.array(queries).astype(np.float64)
        a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
        scores=b@a.T
        inputs.append((docs,queries,scores))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # This is the current complete query graph, not the earlier isolated helper.
    index=FullGraphLibrary('stable').index(inputs[0][0])
    graphs={}
    for kind in ('stable','fast'):
        outputs=_compiled_search(kind,5)(inputs[0][1],index.vectors)
        name=args.output.with_name(args.output.stem+'-'+kind+'.dot')
        mx.export_to_dot(str(name),*outputs,queries=inputs[0][1],index=index.vectors)
        graphs[kind]={'file':name.name,'sha256':hashlib.sha256(name.read_bytes()).hexdigest(),'primitive_rectangles':name.read_text().count('shape=rectangle')}
        mx.eval(*outputs)
    functions={'compiled_norm_application':library_normalize,'whole_stable_application':FullGraphLibrary('stable'),'whole_fast_application':FullGraphLibrary('fast'),
               **{name:lambda x,name=name:candidate_normalize(x,name) for name in ('simd','simd4','vector4','group128')}}
    phases=[]
    for name,f in functions.items():
        for _ in range(3):
            index=f.index(inputs[0][0]) if isinstance(f,FullGraphLibrary) else CosineIndex(inputs[0][0],f)
            index.search(inputs[0][1],5)
        mx.synchronize();start=time.time_ns()
        for docs,queries,reference in inputs:
            index=f.index(docs) if isinstance(f,FullGraphLibrary) else CosineIndex(docs,f)
            ids,values=index.search(queries,5)
            np.testing.assert_allclose(values,np.sort(reference,axis=1)[:,-5:][:,::-1],rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(values,np.take_along_axis(reference,np.array(ids),axis=1),rtol=2e-5,atol=2e-6)
        mx.synchronize();end=time.time_ns()
        phase={'method':name,'start_unix_ns':start,'end_unix_ns':end,'iterations':args.iterations,'numerical_passed':True}
        phases.append(phase);print(json.dumps(phase),flush=True)
        time.sleep(.03)
    result={'kind':'installed_application_profiler_execution','source_sha256':source_identity(),'study_identity':implementation_identity(),'environment':environment_identity(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'corpus_sha256':hashlib.sha256(args.corpus.read_bytes()).hexdigest(),'graphs':graphs,'phases':phases,
        'limitation':'Wall phase markers locate own-process GPU trace events. They include validation/host work and are not benchmark samples or causal proof.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
