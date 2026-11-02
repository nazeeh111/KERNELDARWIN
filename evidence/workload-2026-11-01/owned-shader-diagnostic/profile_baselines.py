#!/usr/bin/env python3
"""Owned-process baseline profiler diagnostic, not a tuning/confirmation study."""
from pathlib import Path
import argparse,hashlib,json,os,time

def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import numpy as np
 import mlx.core as mx
 from kerneldarwin.search import CompiledCosineIndex,text_vectors
 from kerneldarwin.workload import environment_identity,source_identity
 from kerneldarwin.study_identity import implementation_identity
 mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
 texts=json.loads(a.corpus.read_text())['chunks'];cases=[]
 for name,n,d,q,layout in [('notes',64,384,1,'contiguous'),('manuals',161,768,8,'stride2'),('archive',161,1536,17,'transpose')]:
  docs=np.asarray(text_vectors(texts[:n],d));queries=np.asarray(text_vectors(texts[3:3+q],d));cases.append((name,docs,queries,layout))
 def view(x,layout):
  if layout=='stride2':
   backing=np.zeros((len(x),x.shape[1]*2),np.float32);backing[:,::2]=x;return mx.array(backing)[:,::2]
  if layout=='transpose':return mx.array(x.T.copy()).T
  return mx.array(x)
 def normalize(x):
  x=np.asarray(x,np.float64);peak=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-12);z=x/peak;return z/np.maximum(np.sqrt(np.sum(z*z,axis=1,keepdims=True)),1e-12/peak)
 def application(kind,docs,q,layout):
  index=CompiledCosineIndex(view(docs,layout),kind)
  return [index.search(mx.array(q+np.float32(j*.001)),k=5) for j in range(3)]
 phases=[]
 for kind in ['stable','fast']:
  for name,docs,q,layout in cases:
   for _ in range(3):application(kind,docs,q,layout)
   mx.synchronize();start=time.time_ns();checks=0
   for iteration in range(4):
    changed_docs=docs+np.float32((iteration+1)*.0013);changed_q=q+np.float32((iteration+1)*.0017)
    result=application(kind,changed_docs,changed_q,layout)
    for j,(ids,values) in enumerate(result):
     ref=normalize(changed_q+np.float32(j*.001))@normalize(changed_docs).T;ids=np.asarray(ids);values=np.asarray(values)
     np.testing.assert_allclose(values,np.take_along_axis(ref,ids,axis=1),rtol=2e-5,atol=2e-6);np.testing.assert_allclose(values,-np.sort(-ref,axis=1)[:,:5],rtol=2e-5,atol=2e-6);checks+=1
   mx.synchronize();end=time.time_ns();phase={'method':kind,'task':name,'start_unix_ns':start,'end_unix_ns':end,'application_iterations':4,'independent_score_id_checks':checks};phases.append(phase);print(json.dumps(phase),flush=True);time.sleep(.03)
 report={'kind':'baseline_only_owned_process_counter_diagnostic','pid':os.getpid(),'success':True,'source_sha256':source_identity(),'study_identity':implementation_identity(),'environment':environment_identity(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'corpus_sha256':hashlib.sha256(a.corpus.read_bytes()).hexdigest(),'phases':phases,'peak_mlx_bytes':mx.get_peak_memory(),'limits':['Instrumented diagnostic only, not frozen performance or causal proof.','Only optimized stable/fast libraries; no custom candidate timing or selection.','Host phases include input preparation and numerical checks; use own-process trace data for device evidence.']}
 a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
