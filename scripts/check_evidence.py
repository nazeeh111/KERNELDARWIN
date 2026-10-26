"""Check published evidence consistency, source identity and LAB copies."""
from pathlib import Path
import hashlib
import inspect
import json
import math
from kerneldarwin.records import ExperimentStore,validate_record
from kerneldarwin.control import reference,batched,run_control
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
    root=Path(__file__).resolve().parents[1];folder=root/'evidence/cpu-2026-10-26'
    records=ExperimentStore(folder/'records').list()
    assert len(records)==24, 'Expected 24 complete CPU controls'
    digest=hashlib.sha256((inspect.getsource(reference)+inspect.getsource(batched)+inspect.getsource(run_control)).encode()).hexdigest()
    for record in records:
        assert record['evidence_kind']=='cpu_control' and record['device']['kind']=='cpu'
        assert record['provenance']['gpu_execution'] is False
        assert record['provenance']['source_sha256']==digest,'CPU source differs from published evidence'
        assert len(record['experiments'])==3
        for exp in record['experiments']:
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
        local=root/'lab/data'/name;record=json.loads(local.read_text());validate_record(record)
        assert local.read_bytes()==(folder/'records'/(record['id']+'.json')).read_bytes(),'LAB record differs from raw experiment'
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['accepted']==sum(r['status']=='accepted' for r in records)
    print(f'PASS: {len(records)} CPU records, {len(rows)} observations, source hashes, promotion replays, fitted model, evaluation and LAB copies')

if __name__=='__main__':main()
