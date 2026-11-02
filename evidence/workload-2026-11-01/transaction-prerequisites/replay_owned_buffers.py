"""Recompute the recorded library-only composition analysis; no GPU work."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import random
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--measurement-source',type=Path,default=Path('scripts/rank_study.py'))
args=parser.parse_args()
assert hashlib.sha256(args.measurement_source.read_bytes()).hexdigest()=='4470b506443feea5cf6a4f22b5fea01338e8a199621b02dd0f45f20f114f5b3a'
spec=importlib.util.spec_from_file_location('rank_study',args.measurement_source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=args.directory;d=json.loads((p/'owned-buffers-result.json').read_text())
assert d['completed'] and len(d['samples'])==640 and len(d['checks'])==48
assert d['pending_callbacks']==d['dropped_records']==d['observer_uninstall_result']==0
rng=random.Random(42317);i=0
for repeat in range(40):
    methods=list(m.LIBRARIES);rng.shuffle(methods)
    for method in methods:
        order=['sequential','transaction'];rng.shuffle(order);order+=order[::-1]
        for composition in order:
            row=d['samples'][i];i+=1
            assert row['repeat']==repeat and row['input_seed']==42201+repeat and row['method']==method
            assert row['method_order']==methods and row['composition']==composition and row['composition_order']==order
            assert row['command_buffers']==len(row['records'])
            assert len({r['sequence'] for r in row['records']})==len(row['records'])
            assert all(r['status']==4 and r['gpu_start']>0 and r['gpu_end']>=r['gpu_start'] for r in row['records'])
out={'kind':'library_only_owned_buffer_composition_analysis','acceptance_study':False,
     'candidate_kernel_timing':False,'raw_sha256':hashlib.sha256((p/'owned-buffers-result.json').read_bytes()).hexdigest(),
     'all_orders_and_counts_replayed':True,'comparisons':{},
     'limits':'Instrumented architecture pilot, not accepted study/custom kernel advantage; old and new each perform same raw refresh/threequeries per task. Device buffer duration is not pure kernel time; source/device binding and untouched confirmation remain required.'}
for method in m.LIBRARIES:
    rows=[r for r in d['samples'] if r['method']==method]
    pairs={arm:[statistics.mean(r['wall_us'] for r in rows if r['repeat']==repeat and r['composition']==arm)
                for repeat in range(40)] for arm in ('sequential','transaction')}
    out['comparisons'][method]={'wall':m.intervals(pairs['sequential'],pairs['transaction']),
        'command_buffers':{arm:sorted(set(r['command_buffers'] for r in rows if r['composition']==arm)) for arm in pairs},
        'median_wall_us':{arm:statistics.median(r['wall_us'] for r in rows if r['composition']==arm) for arm in pairs}}
assert out==json.loads((p/'owned-buffers-analysis.json').read_text())
print(json.dumps({'replay_matches':True,'groups_per_comparison':40,'samples':640,
                  'owned_buffer_records':sum(len(r['records']) for r in d['samples'])}))
