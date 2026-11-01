"""One prospective library-only longer-block noise diagnostic, not tuning."""
from pathlib import Path
import gc, hashlib, importlib.util, json, random, sys, time
ROOT=Path('/Users/nazeeh/Documents/ChatGPT/Continue/work/kerneldarwin')
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('rank_study_diagnostic',ROOT/'scripts/rank_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
OUT=Path(__file__).parent
START=time.perf_counter();LIMIT=180

def budget():
 if time.perf_counter()-START>LIMIT:raise TimeoutError('Single 180-second diagnostic exhausted; no retry')

def main():
 with (OUT/'started.json').open('x') as f:json.dump({'kind':'library_only_rank_noise_assay','count':1024,'repeats':40,'wall_limit_seconds':LIMIT,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f)
 mx=study.setup()
 from kerneldarwin.search import _compiled_refresh
 from kerneldarwin.library_baselines import eager_normalize
 result={'kind':'library_only_rank_noise_assay','batch':1024,'repeats':40,'criterion':'Every paired total-cost CI inside [0.9,1.1]','staging':'One evaluated normalized persistent document matrix per task,1024 distinct rawqueries and separately evaluated scores; allrankoutputsforced; fullappnotmeasured','candidate_timing':False,'candidate_selection':False,'noise':[],'telemetry_before':study.telemetry(),'success':False}
 try:
  texts=study.corpus()['chunks'];tasks=json.loads((ROOT/'evidence/workload-2026-11-01/rank-v1/rejected-baseline.json').read_text())['noise']
  rng=random.Random(21731);baseline=study.StudyMethod('stable_library')
  def measure(staged):
   return study.measure(lambda:[study.rank_call(baseline,*x) for x in staged])/1024
  for row in tasks:
   budget();task=row['task'];dh,qh=study.arrays(task,texts,21100)
   docs,finite=_compiled_refresh('stable')(study.layout_array(dh,task['layout']));mx.eval(docs,finite)
   if not finite.item():raise AssertionError('Nonfinite normalized documents')
   staged=[]
   for i in range(1024):
    query=study.layout_array(qh+(i+1)*(0.0011*128/1024),'contiguous');scores=eager_normalize(query)@docs.T;mx.eval(query,scores);staged.append((query,docs,scores))
    if i%64==0:budget()
   study.check_rank(study.rank_call(baseline,*staged[0]),staged[0][2]);study.check_rank(study.rank_call(baseline,*staged[-1]),staged[-1][2])
   for _ in range(5):measure(staged);budget()
   a,b,orders=[],[],[]
   for _ in range(40):
    order=[0,1];rng.shuffle(order);v={arm:measure(staged) for arm in order};a.append(v[0]);b.append(v[1]);orders.append(order);budget()
   comparison=study.intervals(a,b);result['noise'].append({'task':task,'baseline_a_us':a,'baseline_b_us':b,'orders':orders,'comparison':comparison})
   del staged,query,scores,docs,finite;gc.collect();mx.clear_cache();budget();print(task['id'],comparison,flush=True)
  result['success']=all(.9<=x<=1.1 for r in result['noise'] for x in r['comparison']['ci95'])
 except Exception as exc:
  result['error']=type(exc).__name__+': '+str(exc)
 finally:
  result['telemetry_after']=study.telemetry()
  result.update(peak_memory_bytes=mx.get_peak_memory(),wall_seconds=time.perf_counter()-START)
  result['resource_passed']=result['wall_seconds']<=LIMIT and result['peak_memory_bytes']<=2*1024**3
  result['success']=result['success'] and result['resource_passed']
  with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:result[k] for k in ['success','peak_memory_bytes','wall_seconds','candidate_timing']}))
 return 0 if result['success'] else 1
if __name__=='__main__':raise SystemExit(main())
