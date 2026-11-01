"""One instrumented library-only mechanism diagnostic, never acceptance evidence."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,statistics,sys,time,traceback
ROOT=Path('/Users/nazeeh/Documents/ChatGPT/Continue/work/kerneldarwin')
spec=importlib.util.spec_from_file_location('frozen_rank_study',ROOT/'scripts/rank_study.py');study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
OUT=Path(__file__).parent
MAX_SECONDS=120

def main():
 with (OUT/'started.json').open('x') as f:json.dump({'kind':'library_only_rank_mechanism_probe','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limit_seconds':MAX_SECONDS,'profiled_bursts_per_task':32,'candidate_timing':False,'stability_gate_run':False},f)
 started=time.perf_counter();mx=study.setup()
 from kerneldarwin.rank_search import rank_source_identity
 from kerneldarwin.workload import environment_identity
 result={'kind':'instrumented_library_rank_mechanism_diagnostic','source_commit':'a57fe7e6f33bbc088e64a7f4a271d4cc73d7aa45','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'study_script_sha256':study.MEASUREMENT_SHA,'source_sha256':rank_source_identity(),'study_identity':study.current_implementation(),'environment':environment_identity(),'telemetry_before':study.telemetry(),'gc_enabled':gc.isenabled(),'gc_threshold':gc.get_threshold(),'candidate_timing':False,'stability_gate_run':False,'acceptance_or_selection':False,'rows':[],'success':False,'limitations':['Instrumented library-only phases; no candidate speed or acceptance result.','Graph construction/evaluation may include backend coordination; eval-wait is not pure device time.','Wall minus current-thread CPU time includes scheduling and other threads, not just GPU execution.','GC correlation alone would not establish causality.','All input score matrices are evaluated and changed; outputs are forced and numerically checked.']}
 events=[];phase='setup';active={}
 def callback(what,info):
  generation=info['generation']
  if what=='start':active[generation]=(time.perf_counter_ns(),phase)
  else:
   begin,where=active.pop(generation,(time.perf_counter_ns(),'unmatched'))
   events.append({'generation':generation,'start_ns':begin,'end_ns':time.perf_counter_ns(),'phase':where,'collected':info.get('collected'),'uncollectable':info.get('uncollectable')})
 def budget():
  if time.perf_counter()-started>MAX_SECONDS:raise TimeoutError('Single mechanism diagnostic exceeded120seconds')
 def stamp():return time.perf_counter_ns(),time.thread_time_ns()
 def delta(a,b):return {'wall_ns':b[0]-a[0],'thread_cpu_ns':b[1]-a[1]}
 gc.callbacks.append(callback)
 try:
  texts=study.corpus()['chunks'];method=study.StudyMethod('stable_library')
  for task in [{'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},{'id':'archive','rows':len(texts),'width':1536,'queries':17,'layout':'transpose'}]:
   phase='staging';dh,qh=study.arrays(task,texts,21100);staged=study.rank_inputs(task,dh,qh);budget()
   sample=study.rank_call(method,*staged[0]);mx.eval(*sample);study.check_rank(sample,staged[0][2]);tracking={'input_array':gc.is_tracked(staged[0][2]),'output_array':gc.is_tracked(sample[0]),'output_pair':gc.is_tracked(sample)};del sample
   phase='warmup'
   for _ in range(5):study.measure_burst(method,staged);budget()
   observations=[]
   for repeat in range(32):
    phase='synchronize';mx.synchronize();before_stats=gc.get_stats();event_start=len(events)
    phase='construct';t0=stamp();outputs=[study.rank_call(method,*values) for values in staged];t1=stamp()
    phase='flatten';flat=[item for pair in outputs for item in pair];t2=stamp()
    phase='evaluate_wait';mx.eval(*flat);mx.synchronize();t3=stamp()
    phase='release';del flat,outputs;t4=stamp()
    observations.append({'repeat':repeat,'construct':delta(t0,t1),'flatten':delta(t1,t2),'evaluate_wait':delta(t2,t3),'release':delta(t3,t4),'forced_outputs':256,'gc_events':events[event_start:],'gc_collection_delta':[b['collections']-a['collections'] for a,b in zip(before_stats,gc.get_stats())]});budget()
   phase='validation';last=study.rank_call(method,*staged[-1]);mx.eval(*last);study.check_rank(last,staged[-1][2]);del last
   summary={name:{field:statistics.median(row[name][field] for row in observations) for field in ['wall_ns','thread_cpu_ns']} for name in ['construct','flatten','evaluate_wait','release']}
   result['rows'].append({'task':task,'tracking':tracking,'observations':observations,'phase_medians':summary,'timed_gc_events':sum(len(row['gc_events']) for row in observations),'timed_gc_duration_ns':sum(e['end_ns']-e['start_ns'] for row in observations for e in row['gc_events'])})
   del staged;phase='between_tasks';gc.collect();mx.clear_cache();budget();print(task['id'],json.dumps({'phase_medians':summary,'timed_gc_events':result['rows'][-1]['timed_gc_events'],'tracking':tracking}),flush=True)
  result['success']=True
 except Exception as exc:result['error']=type(exc).__name__+': '+str(exc);result['traceback']=traceback.format_exc()
 finally:
  gc.callbacks.remove(callback);result.update(wall_seconds=time.perf_counter()-started,peak_memory_bytes=mx.get_peak_memory(),telemetry_after=study.telemetry(),all_gc_events=events)
  result['resource_passed']=result['wall_seconds']<=MAX_SECONDS and result['peak_memory_bytes']<=2*1024**3
  result['success']=result['success'] and result['resource_passed']
  with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:result[k] for k in ['success','resource_passed','wall_seconds','peak_memory_bytes']}))
 return 0 if result['success'] else 1
if __name__=='__main__':raise SystemExit(main())
