from pathlib import Path
import hashlib
import importlib.util
import json
import math
import random
import statistics

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rank_study',ROOT/'scripts/rank_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
d=json.loads((HERE/'counterbalanced-result.json').read_text())
contract=json.loads((HERE/'counterbalanced-preregistered.json').read_text())
assert d['completed'] is True
assert d['source_sha256']==contract['source_sha256']==hashlib.sha256((HERE/'counterbalanced_probe.py').read_bytes()).hexdigest()
assert d['wall_seconds']<=120 and d['peak_memory_bytes']<=2*1024**3
assert len(d['rows'])==320
rng=random.Random(40731)
index=0
for phase,task in [('rank','notes'),('rank','manuals'),('rank','archive'),('application',None)]:
    for repeat in range(40):
        designs=['short_ab','long_abba'];rng.shuffle(designs)
        for design in designs:
            order=['a','b'];rng.shuffle(order)
            if design=='long_abba':order+=order[::-1]
            row=d['rows'][index];index+=1
            assert row['phase']==phase and row.get('task')==task and row['repeat']==repeat
            assert row['design']==design and row['design_order']==designs and row['arm_order']==order
            assert [r['arm'] for r in row['windows']]==order
            if phase=='application':assert row['input_seed']==40600+repeat
            count=(32 if phase=='rank' else 2) if design=='long_abba' else 1
            inner_key='inner_per_operation_us' if phase=='rank' else 'inner_per_sequence_us'
            value_key='per_operation_us' if phase=='rank' else 'per_sequence_us'
            batch=128 if phase=='rank' else 16
            for window in row['windows']:
                inner=window[inner_key]
                assert len(inner)==count and all(math.isfinite(v) and v>0 for v in inner)
                assert math.isfinite(window['outer_wall_us']) and window['outer_wall_us']>0
                assert window[value_key]==window['outer_wall_us']/(count*batch)
                assert window['outer_wall_us']+1e-7>=sum(inner)*batch

out={'kind':'library_only_counterbalance_analysis','candidate_timing':False,'acceptance_study':False,
     'raw_sha256':hashlib.sha256((HERE/'counterbalanced-result.json').read_bytes()).hexdigest(),
     'all_orders_groups_subsamples_replayed':True,'groups_per_comparison':40,
     'arm_windows_per_long_group':2,'no_outlier_removal':True,'comparisons':{},
     'independence_limit':'Forty recorded paired groups are not proven independent environmental draws; block envelopes do not guarantee formal coverage or rule out longer dependence.'}
for phase,task in [('rank','notes'),('rank','manuals'),('rank','archive'),('application',None)]:
    key=task or 'application'
    out['comparisons'][key]={}
    for design in ('short_ab','long_abba'):
        rows=[r for r in d['rows'] if r['phase']==phase and r.get('task')==task and r['design']==design]
        value_key='per_operation_us' if phase=='rank' else 'per_sequence_us'
        pairs={arm:[statistics.mean(w[value_key] for w in r['windows'] if w['arm']==arm) for r in rows] for arm in ('a','b')}
        assert all(len(v)==40 for v in pairs.values())
        comparison=study.intervals(pairs['a'],pairs['b'])
        out['comparisons'][key][design]={'comparison':comparison,
            'a_us':pairs['a'],'b_us':pairs['b'],
            'median_outer_window_us':statistics.median(w['outer_wall_us'] for r in rows for w in r['windows'])}
out['descriptive_noise_floors']={design:{
    'rank':study.symmetric_noise_floor([out['comparisons'][task][design]['comparison'] for task in ('notes','manuals','archive')],1.10),
    'application':study.symmetric_noise_floor([out['comparisons']['application'][design]['comparison']],1.03)} for design in ('short_ab','long_abba')}
with (HERE/'counterbalanced-analysis.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for task,designs in out['comparisons'].items():
    print(task,{name:entry['comparison']['ci95'] for name,entry in designs.items()})
print(json.dumps(out['descriptive_noise_floors'],indent=2))
