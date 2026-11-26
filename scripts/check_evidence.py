"""Check published evidence consistency, source identity and LAB copies."""
from pathlib import Path
import hashlib
import ast
import sys
import os
import stat
import json
import math
from kerneldarwin.records import WorkloadSpec,validate_record,read_record_file
from kerneldarwin import _historical_validation
from kerneldarwin.control import run_control
from kerneldarwin.engine import observation_rows
from kerneldarwin.selector import InterventionSelector,evaluate_grouped


def equal_numeric(left,right):
    if isinstance(left,float) or isinstance(right,float):
        return isinstance(left,(float,int)) and isinstance(right,(float,int)) and math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-11)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(equal_numeric(v,right[k]) for k,v in left.items())
    if isinstance(left,list):
        return isinstance(right,list) and len(left)==len(right) and all(equal_numeric(a,b) for a,b in zip(left,right))
    return left==right

def main():
    if not __debug__: raise RuntimeError('Evidence checker requires Python without optimization')
    root=Path(__file__).resolve().parents[1];folder=root/'evidence/cpu-2026-10-26'
    snapshots = folder/'historical-source'
    expected = {'control.py.txt':'1f10bc8261ee48d00cb7161d8aae5669408addf1518b258035b4732e72f3fc6a',
                'records.py.txt':'cb51e429ea62464e684efeaf48c836e6d0f94241ee8da6c3aa3ad982d16b2d19'}
    captured={}
    for name, sha in expected.items():
        fd=os.open(snapshots/name,os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size>1000000: raise ValueError('Invalid historical source file')
            data=b''
            while len(data)<=1000000:
                part=os.read(fd, min(65536,1000001-len(data)))
                if not part: break
                data+=part
            if len(data)>1000000 or hashlib.sha256(data).hexdigest()!=sha: raise ValueError(f'Historical source changed: {name}')
            captured[name]=data
        finally: os.close(fd)
    admitted={p.stem:read_record_file(p,validator=_historical_validation.validate_record,with_bytes=True) for p in sorted((folder/'records').glob('*.json'))}
    for identifier,(record,_) in admitted.items():
        if record['id']!=identifier: raise ValueError('Historical filename/id mismatch')
    records=[item[0] for item in admitted.values()]
    assert len(records)==24, 'Expected 24 complete CPU controls'
    source=captured['control.py.txt'].decode('utf-8')
    lines=source.splitlines(keepends=True)
    functions={node.name: ''.join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    digest=hashlib.sha256(''.join(functions[name] for name in ('reference','batched','run_control')).encode()).hexdigest()
    refused=0
    for record in records:
        assert record['evidence_kind']=='cpu_control' and record['device']['kind']=='cpu'
        assert record['provenance']['gpu_execution'] is False
        assert record['provenance']['source_sha256']==digest,'CPU source differs from published evidence'
        if record['status']=='accepted':
            try:
                validate_record(record)
            except ValueError as exc:
                assert 'Named CPU reference/source/provenance' in str(exc)
                refused += 1
            else:
                raise AssertionError('Historical accepted record unexpectedly passed current promotion validation')
        assert len(record['experiments'])==3
        for exp in record['experiments']:
            assert exp['source']=={'baseline':functions['reference'],'candidate':functions['batched']}
            assert len(exp['timing']['baseline_us'])==40
            assert len(exp['timing']['candidate_us'])==40
            assert exp['correctness']['passed'] is True
            assert any(t['pattern']=='full_shape_random' for t in exp['correctness']['tests'])
    rows=observation_rows(records);assert len(rows)==72
    report=json.loads((folder/'cpu-selector.evaluation.json').read_text())
    replay=evaluate_grouped(rows)
    assert equal_numeric(report['splits'],replay['splits']),'Stored replay differs from current deterministic evaluator'
    model=InterventionSelector.load(folder/'cpu-selector.json')
    recomputed=InterventionSelector().fit(rows)
    assert equal_numeric(model.weights,recomputed.weights) and equal_numeric(model.matrix,recomputed.matrix),'Fitted model is stale'
    manifest=json.loads((root/'lab/data/index.json').read_text())
    for name in manifest['records']:
        local=root/'lab/data'/name;record,lab_bytes=read_record_file(local,validator=_historical_validation.validate_record,with_bytes=True)
        assert lab_bytes==admitted[record['id']][1],'LAB record differs from raw experiment'
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['accepted']==sum(r['status']=='accepted' for r in records)
    assert refused == summary['accepted'] and refused > 0
    current=run_control(WorkloadSpec('softmax',(32,31)), repeats=20, batches=(32,))
    validate_record(current)
    print(f'PASS: {len(records)} historical CPU records and {len(rows)} observations under pinned historical source and historical compatibility validation; {refused} accepted historical records refused by current promotion; current CPU software smoke passed; historical model, evaluation and LAB copies match')

if __name__=='__main__':main()
