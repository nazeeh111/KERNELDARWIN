"""Prospective library-only block-size assay; no candidate timing/selection."""
from pathlib import Path
import importlib.util,hashlib,json,random
spec=importlib.util.spec_from_file_location('fused_study',Path('scripts/fused_study.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
mx=s.setup()
from kerneldarwin.fused_search import library_scores,score_source_identity
from kerneldarwin.workload import environment_identity
out=Path('baseline-batch128-assay.json')
if out.exists():raise FileExistsError(out)
data=s.corpus();n=len(data['chunks'])
tasks=[{'id':'notes','rows':64,'width':384,'queries':1,'layout':'contiguous'},{'id':'manuals','rows':min(256,n),'width':768,'queries':8,'layout':'column_stride2'},{'id':'archive','rows':n,'width':1536,'queries':17,'layout':'transpose'}]
rng=random.Random(1441);rows=[]
for task in tasks:
 docs,q=s.arrays(task,data['chunks'],100)
 staged=s.score_inputs(task,docs,q,128)
 for _ in range(5):s.measure_burst(library_scores,staged)
 pairs=[]
 for _ in range(40):
  order=[0,1];rng.shuffle(order)
  values={arm:s.measure_burst(library_scores,staged) for arm in order}
  pairs.append([values[0],values[1]])
 a,b=map(list,zip(*pairs))
 row={'task':task,'baseline_a_us':a,'baseline_b_us':b,'comparison':s.intervals(a,b)}
 rows.append(row);print(task['id'],row['comparison']['ci95'],flush=True)
passed=all(.9<=r['comparison']['ci95'][0] and r['comparison']['ci95'][1]<=1.1 for r in rows)
record={'kind':'prospective_library_only_block_assay','batch':128,'repeats':40,'gate':[.9,1.1],'passed':passed,'candidate_timing':False,'candidate_selection':False,'source_sha256':score_source_identity(),'study_identity':s.current_implementation(),'assay_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'environment':environment_identity(),'noise':rows,'peak_mlx_bytes':mx.get_peak_memory(),'telemetry':s.telemetry(),'limit':'Assay only; this does not freeze or validate a portfolio. No acceptance threshold was weakened.'}
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed':passed,'output':str(out)}))
