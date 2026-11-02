"""Portable CPU reconstruction of fixed owned-call counters; no Metal import."""
from pathlib import Path
import collections
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent


def replay(directory=HERE):
    directory=Path(directory)
    def read(name):
        path=directory/name
        if path.stat().st_size>2_000_000:raise ValueError('Bounded accounting input exceeded')
        return json.loads(path.read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    pin=read('preregistered.json');result=read('result.json');process=read('process.json')
    assert sha(directory/'probe.py')==pin['script_sha256']==result['source_sha256']
    assert sha(directory/'call_counters.mm')==pin['helper_sha256']
    assert result['completed'] is True and process['exit_code']==0 and process['hard_timeout'] is False
    assert 0<result['wall_seconds']<=60 and result['wall_seconds']<=process['external_wall_seconds']<=75
    assert result['peak_memory_bytes']<=2*1024**3
    assert result['observer_uninstall_result']==result['dropped_records']==result['pending_callbacks']==0
    assert result['operation_source_sha256']==pin['operation_source_sha256']
    assert result['study_identity']==pin['study_identity'] and '/site-packages/' in result['actual_import']
    assert result['candidate_performance_timed'] is False and result['acceptance_study'] is False
    control=result['positive_control'];assert control['passed'] is True
    assert [e['a'] for e in control['events'] if e['kind']==4]==[16]
    assert [(e['kind'],e['name']) for e in control['events'] if e['kind'] in (5,6)]==[(5,'kd_counter_control'),(6,'kd_counter_control')]
    assert [(e['kind'],e['a'],e['c']) for e in control['events'] if e['kind'] in (1,2,3)][:3]==[(1,128,1),(1,128,1),(2,32,1)]
    assert sum(e['kind']==3 for e in control['events'])==1
    assert any(e['kind']==1 and e['a']==64 and e['c']==2 for e in control['events'])
    cases=[('notes',64,384,1,27),('manuals',161,768,8,28),('archive',161,1536,17,29),('notes_negative',64,384,1,27)]
    expected=[(seed,case) for seed in (50311,50312) for case in cases]
    assert len(result['samples'])==8
    rows=[]
    for tag,(sample,(seed,(name,n,d,q,dispatches))) in enumerate(zip(result['samples'],expected),2):
        assert (sample['case'],sample['seed'],sample['rows'],sample['width'],sample['query_rows'])==(name,seed,n,d,[q]*3)
        assert sample['numerical_float64_and_id_association'] is True and sample['inputs_preserved'] is True and sample['decision']=='library'
        events=sample['events'];assert len(events)<=4096 and all(e['tag']==tag and 1<=e['kind']<=8 for e in events)
        kernels=[e for e in events if e['kind'] in (5,6)]
        assert len(kernels)==sample['compute_dispatches']==dispatches
        assert all(e['name'] and e['name']!='unknown' and all(type(e[k]) is int and e[k]>0 for k in ('a','b','c','d','e','f')) for e in kernels)
        assert dict(collections.Counter(e['name'] for e in kernels))==sample['kernel_counts']
        copies=[e for e in kernels if e['name']=='gg1_copyfloat32float32']
        assert len(copies)==3 and all(e['kind']==6 and (e['a'],e['b'],e['c'])==(q*d,1,1) for e in copies)
        assert sum(e['kind']==7 for e in events)==sample['command_buffers']==2
        done=[e for e in events if e['kind']==8];assert len(done)==2 and all(e['a']==4 for e in done)
        assert not any(e['kind'] in (1,2,3,4) for e in events)
        assert sample['blit_copy_calls']==sample['blit_copy_bytes']==0
        logical=4*(n*d+3*q*d)
        bridge=4*((2*n*d-1 if name=='manuals' else n*d)+3*q*d)
        assert sample['host_input_logical_bytes']==logical
        assert sample['mlx_host_constructor_calls']==4 and sample['mlx_host_constructor_source_span_bytes']==bridge
        assert sample['host_adapter_copy_calls']==(4 if name=='notes_negative' else 0)
        assert sample['host_adapter_copy_logical_bytes']==(logical if name=='notes_negative' else 0)
        assert sample['output_numeric_payload_bytes']==3*q*5*8 and sample['finite_scalar_payload_bytes']==2
        expected_strides=(d*8,8) if name=='manuals' else (4,n*4) if name=='archive' else (d*4,-4) if name=='notes_negative' else (d*4,4)
        assert sample['document_strides']==list(expected_strides)
        assert all(0<=sample[field][kind]<=2*1024**3 for field in ('memory_before','memory_after') for kind in ('active','cache'))
        assert sample['memory_before']==sample['memory_after']
        rows.append({'case':name,'seed':seed,'compute_dispatches':dispatches,'generic_copy_dispatches':3,
            'generic_copy_elements':sum(math.prod(e[k] for k in ('a','b','c')) for e in copies),
            'observed_command_buffer_scope_including_final_synchronize':2,'new_metal_buffer_api_calls':0,
            'explicit_blit_copy_bytes':0,'host_input_logical_bytes':logical,'mlx_host_constructor_copy_span_bytes':bridge,
            'extra_host_adapter_copy_logical_bytes':logical if name=='notes_negative' else 0,
            'host_output_numeric_payload_bytes':3*q*5*8,'finite_scalar_payload_bytes':2})
    log=(directory/'native.log').read_text()
    assert 'Metal API Validation Enabled' in log and 'Metal GPU Validation Enabled' in log
    return {'kind':'independent_cpu_call_accounting_replay','passed':True,'samples_replayed':8,
        'result_sha256':sha(directory/'result.json'),'positive_control_replayed':True,'rows':rows,
        'limits':['No performance/acceptance gate is passed by CPU replay.',
            'Two observed submissions include the diagnostic final synchronize; exact API-only command-buffer split was not marked.',
            'Zero new-buffer API calls reflect warmed cache reuse, not zero logical allocation requests.',
            'Host copy spans and output payloads are source-derived logical quantities, not physical bus traffic.',
            'Three generic GPU copies match concatenation dimensions; buffer identities/offsets were not recorded, so operation attribution remains an inference.']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=replay()
    if args.output:
        with args.output.open('x') as file:json.dump(report,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({'passed':report['passed'],'samples_replayed':report['samples_replayed']}))
