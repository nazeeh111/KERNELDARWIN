#!/usr/bin/env python3
"""CPU-only audit of the closed, incomplete batch calibration. No GPU timing."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'src').is_dir():sys.path.insert(0,str(ROOT/'src'))
from kerneldarwin import transaction_export as reader


def replay(directory):
    directory=Path(directory)
    sha=lambda content:hashlib.sha256(content).hexdigest()
    metadata=json.loads((directory/'preregistered-source.json').read_text())
    if sha((directory/'preregistered-source.zip').read_bytes())!=metadata['archive_sha256']:
        raise ValueError('Preregistered source archive changed')
    import zipfile
    with zipfile.ZipFile(directory/'preregistered-source.zip') as archive:
        for name,digest in metadata['files'].items():
            if sha(archive.read(name))!=digest:raise ValueError('Archived source mismatch: '+name)
    if sha(Path(reader.__file__).read_bytes())!='42e78bf029da49537119336bd3fdf6a9f0a0e80db1c36a0c25e292cb58684362':
        raise ValueError('Independent reader changed')
    record=reader.audit._decode(reader._bytes(directory/'rejected-baseline.json'))
    if record['script_sha256']!='12ca2bad707b1acc2aaba8ac447d21178542973d7bd54bc803de888d180a9e95':
        raise ValueError('Measurement source changed')
    if (record['stage']!='application_controls'
            or record['reason']!='TimeoutError: 180-second calibration cap; no retry'
            or not 180<record['wall_seconds']<195):
        raise ValueError('Expected bounded incomplete calibration')
    reader.audit._count(record['peak_memory_bytes'],reader.LIMITS['memory_bytes'])
    rng=random.Random(44211);rank=[];measured=0.
    if len(record['rank_controls'])!=3:raise ValueError('Expected all3rank controls')
    for task,row in zip(reader.TASKS,record['rank_controls']):
        reader._equal(row['task'],task,'Changed development task')
        values=reader._samples(row['windows'],reader.METHODS,'rank')
        reader._equal(len(row['orders']),40,'Missing rank order')
        for order in row['orders']:reader._order(order,reader.METHODS,rng)
        comparisons={name:reader.audit._interval(values[name],values['dispatcher'])
                     for name in reader.CALIBRATION['equality_rank_labels']}
        reader._equal(row['comparisons'],comparisons,'Rank controls differ from raw windows')
        rank.append({'task':task['id'],'groups':40,'comparisons':comparisons})
        measured+=reader._outer_seconds(row['windows'],'rank')
    app=record['application_control'];groups=len(app['orders'])
    if not 0<groups<40 or set(app['windows'])!=set(reader.METHODS):raise ValueError('Expected partial application controls')
    rng=random.Random(44213)
    for order in app['orders']:reader._order(order,reader.METHODS,rng)
    for name,rows in app['windows'].items():
        if len(rows)!=groups:raise ValueError('Inconsistent partial group counts')
        for windows in rows:
            if len(windows)!=2:raise ValueError('Missing paired application window')
            for window in windows:reader._window(window,'application')
    measured+=reader._outer_seconds(app['windows'],'application')
    if measured>record['wall_seconds']:raise ValueError('Measurements exceed stage wall')
    forbidden=('protocol.json','freeze-completed.json','tune-started.json','development.json',
        'policy-frozen.json',*(f'confirmation-{i}-started.json' for i in range(3)),
        *(f'confirmation-{i}.json' for i in range(3)),'cold-study-started.json')
    if any((directory/name).exists() for name in forbidden):raise ValueError('Study continued after calibration failure')
    for field in ('telemetry_before','telemetry_after'):reader.audit._telemetry(record[field])
    return {'schema':1,'kind':'independent_incomplete_calibration_replay','passed':True,
        'source_archive_sha256':metadata['archive_sha256'],
        'rejected_baseline_sha256':sha((directory/'rejected-baseline.json').read_bytes()),
        'rank_controls':rank,'rank_comparisons_recomputed':21,
        'completed_rank_groups':120,'incomplete_application_groups':groups,'required_application_groups':40,
        'paired_windows_per_group':2,'recorded_window_seconds':measured,
        'wall_seconds':record['wall_seconds'],'peak_memory_bytes':record['peak_memory_bytes'],
        'application_equality_gate':'not reached; incomplete sample',
        'performance_criterion_frozen':False,'candidate_performance_timed':False,'heldout_executed':False,
        'candidate_policy_created':False,'accepted_artifact':False,
        'scope':'CPU replay validates retained raw controls and stop markers. It does not pass a GPU or performance gate.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=replay(args.directory)
    with args.output.open('x') as file:json.dump(report,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','rank_comparisons_recomputed','incomplete_application_groups','wall_seconds')}))


if __name__=='__main__':main()
