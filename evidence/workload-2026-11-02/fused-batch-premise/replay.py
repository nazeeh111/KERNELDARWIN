"""Independent CPU check of this fixed numerical/resource premise; no GPU."""
from pathlib import Path
import collections
import hashlib
import json

HERE=Path(__file__).resolve().parent


def replay(directory=HERE):
    directory=Path(directory);read=lambda n:json.loads((directory/n).read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pin=read('preregistered.json');result=read('result.json');process=read('process.json')
    for name,digest in pin['files'].items():assert sha(directory/name)==digest,name
    assert result['completed'] is True and process['exit_code']==0 and process['hard_timeout'] is False
    assert 0<result['wall_seconds']<120 and result['wall_seconds']<=process['external_wall_seconds']<135
    assert result['peak_memory_bytes']<=2*1024**3 and result['authored_source_bytes']==7294<=12288
    assert result['observer_uninstall_result']==result['pending_callbacks']==result['dropped_records']==0
    assert result['candidate_performance_timed'] is result['acceptance_study'] is result['heldout_executed'] is False
    assert result['installed_source_sha256']==pin['installed_source_sha256'] and '/site-packages/' in result['actual_installed_import']
    checks=result['checks'];counts=collections.Counter(row['kind'] for row in checks)
    expected={'random_shapes_layouts':21,'zero_stable_ties':1,'adversarial_magnitude_cancellation_subnormal_and_tiny_atol0':1,
              'nonfinite_rejection_flags_and_nan_rows':12,'aliasing_and_nonfinite_recovery':1,'unsupported_request_library_fallback':6}
    assert dict(counts)==expected and len(checks)==42
    shapes=[(5,1,[1,1,1]),(7,31,[1,2,3]),(37,97,[1,3,7]),(64,384,[1,1,1]),
            (161,768,[8,8,8]),(161,1536,[17,17,17]),(2048,2048,[1,1,1])]
    actual=[(r['n'],r['d'],r['query_rows'],r['layout'],r['authored_paths'],r['library_paths']) for r in checks if r['kind']=='random_shapes_layouts']
    assert actual==[(n,d,rows,layout,2,4) for n,d,rows in shapes for layout in ('contiguous','column_stride2','transpose')]
    assert {(r['target'],r['value']) for r in checks if r['kind']=='nonfinite_rejection_flags_and_nan_rows'}=={(i,v) for i in range(4) for v in ('nan','inf','-inf')}
    control=result['positive_control'];assert [e['a'] for e in control if e['kind']==4]==[16]
    assert [(e['kind'],e['name']) for e in control if e['kind'] in (5,6)]==[(5,'kd_counter_control'),(6,'kd_counter_control')]
    assert sum(e['kind']==9 for e in control)==1
    methods=('stable_library','fast_library','stable_reduction','fast_reduction','simd32','group128')
    profiles=[('notes',64,384,1),('manuals',161,768,8),('archive',161,1536,17)]
    samples=result['samples'];assert len(samples)==18;rows=[]
    for tag,(sample,(profile,n,d,q,method)) in enumerate(zip(samples,[(p,n,d,q,m) for p,n,d,q in profiles for m in methods]),2):
        assert (sample['profile'],sample['n'],sample['d'],sample['query_rows'],sample['method'],sample['seed'])==(profile,n,d,[q]*3,method,50411)
        assert sample['float64_reference_passed'] is True
        events=sample['events'];assert len(events)<=4096 and all(e['tag']==tag and 1<=e['kind']<=9 for e in events)
        assert sum(e['kind']==9 for e in events)==1;boundary=next(i for i,e in enumerate(events) if e['kind']==9)
        kernels=[e for e in events if e['kind'] in (5,6)];assert len(kernels)==sample['compute_dispatches']
        assert dict(collections.Counter(e['name'] for e in kernels))==sample['kernel_counts']
        assert kernels and all(e['name'] and e['name']!='unknown' and all(e[k]>0 for k in ('a','b','c','d','e','f')) for e in kernels)
        api=sum(e['kind']==7 for e in events[:boundary]);diagnostic=sum(e['kind']==7 for e in events[boundary+1:])
        assert api==sample['api_command_buffers'] and diagnostic==sample['diagnostic_sync_command_buffers']==1
        assert sum(e['kind']==8 for e in events)==api+diagnostic and all(e['a']==4 for e in events if e['kind']==8)
        custom=method in ('simd32','group128')
        if custom:
            assert len(kernels)==6 and api==1
            assert sum('kd_normalize_pack3' in e['name'] for e in kernels)==1
            assert sum('kd_top5_'+method in e['name'] for e in kernels)==1
            assert not any('_copy' in e['name'] for e in kernels)
        else:
            index=('notes','manuals','archive').index(profile)
            expected_dispatches={'stable_library':27,'fast_library':29,'stable_reduction':51,'fast_reduction':53}[method]+index
            assert len(kernels)==expected_dispatches and api==(2 if method.endswith('reduction') else 1)
            copies=[e for e in kernels if e['name']=='gg1_copyfloat32float32']
            query_copies=[e for e in copies if (e['a'],e['b'],e['c'])==(q*d,1,1)]
            assert len(query_copies)==3
            if method.endswith('reduction'):
                assert len(copies)==8
                assert sum((e['a'],e['b'],e['c'])==(3*q,1,1) for e in copies)==5
                assert sum(e['name']=='gg1_copyuint32uint32' for e in kernels)==5
                assert sum(e['name']=='v_copyint32int64' and e['a']==n for e in kernels)==1
            else:assert len(copies)==3
        rows.append({'profile':profile,'method':method,'compute_dispatches':len(kernels),'api_command_buffers':api,'diagnostic_sync_command_buffers':diagnostic})
    log=(directory/'native.log').read_text();assert 'Metal API Validation Enabled' in log and 'Metal GPU Validation Enabled' in log
    return {'kind':'independent_cpu_fused_premise_replay','passed':True,'numerical_check_groups':42,
        'resource_samples_replayed':18,'result_sha256':sha(directory/'result.json'),'rows':rows,
        'performance_confirmed':False,'accepted_artifact':False,
        'limits':['CPU replay checks recorded evidence, not GPU attestation.','Numerical check flags are execution reports; raw Metal events validate resource counts.',
                  'Dispatch-count reduction is not latency acceleration.','One changed resource input seed and synthetic numerical cases are not frozen performance confirmation.']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=replay()
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':report['passed'],'resource_samples_replayed':report['resource_samples_replayed']}))
