"""Verify retained bytes and replay their CPU timing gates, never measure or train."""
from pathlib import Path
import json,hashlib,sys,inspect
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'reader-source'))
m=json.loads((P/'bundle-manifest.json').read_text())
for f in m['files']:
 b=(P/f['path']).read_bytes()
 if len(b)!=f['bytes'] or hashlib.sha256(b).hexdigest()!=f['sha256']:raise SystemExit('Changed member: '+f['path'])
from kerneldarwin.records import validate_record
from kerneldarwin.statistics import compare_paired
from kerneldarwin.control import reference,batched,run_control
identity=hashlib.sha256((inspect.getsource(reference)+inspect.getsource(batched)+inspect.getsource(run_control)).encode()).hexdigest();count=0
for file in(P/'retained/records').glob('*.json'):
 r=json.loads(file.read_text());validate_record(r);assert r['provenance']['source_sha256']==identity and r['provenance']['gpu_execution'] is False;count+=1
assert count==24
rows=json.loads((P/'figure-data.json').read_text())['rows'];pairs=0
for row in rows:
 t=row['data'];c=t['comparison'];assert compare_paired(t['baseline_us'],t['candidate_us'],seed=c['seed'],min_speedup=c['min_speedup'],resamples=c['bootstrap_resamples'])==c;pairs+=len(t['baseline_us'])
assert pairs==280
print(json.dumps({'content_members_verified':len(m['files']),'records':24,'selected_paired_observations_replayed':pairs,'all_declared_bootstrap_fields_exact':True,'source_identity_matches':True,'no_new_timings_or_selector_fit':True,'scope':'Recorded CPU software controls, not CUDA or general performance evidence'},indent=2))
