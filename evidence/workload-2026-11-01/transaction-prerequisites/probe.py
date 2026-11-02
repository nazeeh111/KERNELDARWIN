"""Numerical-only refresh plus three-query transaction composition proof."""
from pathlib import Path
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('rank_study',ROOT/'scripts/rank_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

@lru_cache(maxsize=6)
def graph(method):
    import mlx.core as mx
    from kerneldarwin.library_baselines import eager_normalize,fast_formula
    from kerneldarwin.rank_search import candidate_rank,reduction_rank_formula
    formula=fast_formula if method.startswith('fast') else eager_normalize
    def transaction(documents,q0,q1,q2):
        normalized_docs=formula(documents)
        queries=formula(mx.concatenate([q0,q1,q2],axis=0))
        doc_finite=mx.all(mx.isfinite(normalized_docs))
        query_finite=mx.all(mx.isfinite(queries))
        scores=queries@normalized_docs.T
        if method in ('simd32','group128'):
            ids,values=candidate_rank(scores,method)
        elif method.endswith('reduction'):
            ids,values=reduction_rank_formula(scores)
        else:
            ids=mx.argsort(-scores,axis=-1)[:,:5]
            values=mx.take_along_axis(scores,ids,axis=-1)
        return ids,values,doc_finite,query_finite
    return mx.compile(transaction)

def execute(method,task,documents,queries):
    import mlx.core as mx
    rawdocs=study.layout_array(documents,task['layout'])
    inputs=[study.layout_array(queries+j*.001,'contiguous') for j in range(3)]
    ids,values,df,qf=graph(method)(rawdocs,*inputs)
    mx.eval(ids,values,df,qf)
    if not df.item():raise ValueError('Search index vectors must be finite')
    if not qf.item():raise ValueError('Search query vectors must be finite')
    ids,values=ids.tolist(),values.tolist()
    q=task['queries']
    return [(ids[j*q:(j+1)*q],values[j*q:(j+1)*q]) for j in range(3)]

def main():
    import mlx.core as mx
    import numpy as np
    from kerneldarwin.rank_search import rank_source_identity
    mx.set_default_device(mx.gpu);mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
    start=time.perf_counter()
    state={'kind':'raw_refresh_three_changed_query_batch_composition_numerical_proof',
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'rank_source_sha256':rank_source_identity(),'study_identity':study.current_implementation(),
           'candidate_timing':False,'candidate_selection':False,'performance_claim':False,
           'validation_flags':{k:os.getenv(k) for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION')},
           'checks':[],'methods':['stable_library','fast_library','stable_reduction','fast_reduction','simd32','group128'],
           'scope':'Same raw documents and three changed-query batches; pure transaction, no state, no answer cache, normalization once per document/each query row; vendor matmul and stable finite/top5 contracts retained.'}
    with (HERE/'started.json').open('x') as f:json.dump(state,f,indent=2);f.write('\n')
    def budget():
        if time.perf_counter()-start>120:raise TimeoutError('120-second numerical-only cap')
        if mx.get_peak_memory()>2*1024**3:raise RuntimeError('2GiB memory cap')
    try:
        data=study.corpus()['chunks']
        tasks=[{'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},
               {'id':'manuals','rows':min(256,len(data)),'width':768,'queries':8,'layout':'column_stride2'},
               {'id':'archive','rows':len(data),'width':1536,'queries':17,'layout':'transpose'},
               {'id':'small','rows':5,'width':31,'queries':3,'layout':'column_stride2'}]
        rng=np.random.default_rng(41841)
        for task in tasks:
            for case in ('corpus','random','zeros','large','tiny'):
                docs,query=study.arrays(task,data,41800)
                if case!='corpus':
                    docs=rng.standard_normal((task['rows'],task['width'])).astype(np.float32)
                    query=rng.standard_normal((task['queries'],task['width'])).astype(np.float32)
                if case=='zeros':docs[:]=0;query[:]=0
                elif case=='large':docs*=np.float32(1e20);query*=np.float32(1e20)
                elif case=='tiny':docs*=np.float32(1e-20);query*=np.float32(1e-20)
                for method in state['methods']:
                    output=execute(method,task,docs,query)
                    study.check_application(output,docs,query)
                    original=study.application(study.StudyMethod(method),task,docs,query)
                    # Whole-application equality is numeric, not bitwise vendor-GEMM identity.
                    for (ids,values),(old_ids,old_values) in zip(output,original):
                        np.testing.assert_allclose(values,old_values,rtol=2e-5,atol=2e-6)
                    if case=='zeros':
                        for ids,_ in output:np.testing.assert_array_equal(ids,np.broadcast_to(np.arange(5),(task['queries'],5)))
                    state['checks'].append({'task':task['id'],'case':case,'method':method,
                        'three_searches':True,'float64_scores_and_id_association':True,'old_pipeline_scores_agree':True})
                    budget()
        task=tasks[0];docs,query=study.arrays(task,data,41800)
        for where in ('document','query'):
            bad_docs,bad_query=docs.copy(),query.copy()
            (bad_docs if where=='document' else bad_query)[0,0]=np.inf
            for method in state['methods']:
                try:execute(method,task,bad_docs,bad_query)
                except ValueError as error:
                    expected='index' if where=='document' else 'query'
                    if expected not in str(error):raise
                else:raise AssertionError('Nonfinite transaction accepted')
                state['checks'].append({'nonfinite':where,'method':method,'explicit_rejection':True})
                budget()
        state['completed']=True
    except Exception as error:
        state.update(completed=False,error=type(error).__name__+': '+str(error));raise
    finally:
        state.update(wall_seconds=time.perf_counter()-start,peak_memory_bytes=mx.get_peak_memory(),checks_run=len(state['checks']))
        with (HERE/'result.json').open('x') as f:json.dump(state,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({k:v for k,v in state.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
