#!/usr/bin/env python3
"""Extract only the launched process's shader records. Never infer ownership from GPU ID."""
from pathlib import Path
from datetime import datetime,timezone
import collections,hashlib,json,re,xml.etree.ElementTree as E
SOURCE=Path('/private/tmp/kd-counter-diag-20261101')
OUTPUT=Path(__file__).resolve().parent

def table(name):
    root=E.parse(SOURCE/name).getroot();index={e.get('id'):e for e in root.iter() if e.get('id')}
    def resolve(e):
        seen=set()
        while e is not None and e.get('ref'):
            ref=e.get('ref')
            if ref in seen:raise ValueError('Cyclic XML reference')
            seen.add(ref);e=index[ref]
        return e
    def text(e):
        e=resolve(e);return '' if e is None else (e.text or '').strip()
    def formatted(e):
        e=resolve(e);return '' if e is None else e.get('fmt',text(e))
    schemas=[]
    for s in root.iter('schema'):schemas.append({'name':s.get('name'),'columns':[c.findtext('mnemonic') for c in s.findall('col')]})
    return root,resolve,text,formatted,schemas

def union_ns(rows):
    end=None;total=0
    for start,stop in sorted(rows):
        if end is None or start>end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total

def main():
    app=json.loads((SOURCE/'application.json').read_text());pid=str(app['pid']);assert app['success']
    root,resolve,text,fmt,schemas=table('shader-intervals.xml');rows=[]
    for row in root.iter('row'):
        c=list(row);process=resolve(c[11]);p=process.find('pid') if process is not None else None
        if p is None or text(p)!=pid:continue
        rows.append({'start_ns':int(text(c[0])),'duration_ns':int(text(c[1])),'shader':fmt(c[2]),'event':fmt(c[5]),'shader_type':fmt(c[6]),'channel':fmt(c[13]),'depth':int(text(c[14]))})
    assert rows and all(r['duration_ns']>0 for r in rows)
    toc=E.parse(SOURCE/'toc.xml').getroot();start_text=next(e.text for e in toc.iter('start-date'));d=datetime.fromisoformat(start_text).astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc);start_ns=(d.days*86400+d.seconds)*10**9+d.microseconds*1000
    guard_ns=2_000_000;phases=[]
    for p in app['phases']:
        lo=p['start_unix_ns']-start_ns;hi=p['end_unix_ns']-start_ns
        selected=[r for r in rows if lo+guard_ns<=r['start_ns'] and r['start_ns']+r['duration_ns']<=hi-guard_ns]
        per_shader=collections.defaultdict(list)
        for r in selected:per_shader[re.sub(r' \(\d+\)$','',r['shader'])].append((r['start_ns'],r['start_ns']+r['duration_ns']))
        stats=[{'shader':name,'records':len(times),'interval_union_ns':union_ns(times)} for name,times in per_shader.items()]
        stats.sort(key=lambda x:(-x['interval_union_ns'],x['shader']))
        phases.append({'method':p['method'],'task':p['task'],'application_iterations':p['application_iterations'],'independent_score_id_checks':p['independent_score_id_checks'],'marker_interval_ns':hi-lo,'guarded_owned_records':len(selected),'guarded_owned_interval_union_ns':union_ns([(r['start_ns'],r['start_ns']+r['duration_ns']) for r in selected]),'by_shader':stats})
    root,resolve,text,fmt,shader_schemas=table('shader-list.xml');shader_names=set()
    for row in root.iter('row'):
        c=list(row);proc=resolve(c[8]);p=proc.find('pid') if proc is not None else None
        if p is not None and text(p)==pid:shader_names.add(fmt(c[1]))
    root,resolve,text,fmt,mapping_schemas=table('submission-map.xml');owners=collections.defaultdict(set);own_accelerators=set();owned_mapping_rows=0
    for row in root.iter('row'):
        c=list(row);p=text(c[11]);a=text(c[6]);owners[a].add(p)
        if p==pid:
            proc=resolve(c[12]);assert text(proc.find('pid'))==pid;owned_mapping_rows+=1;own_accelerators.add(a)
    exclusive=bool(own_accelerators) and all(owners[a]=={pid} for a in own_accelerators)
    root,resolve,text,fmt,counter_schemas=table('counter-info.xml');definitions=[]
    for row in root.iter('row'):
        c=list(row);definitions.append({'counter_id':text(c[1]),'name':text(c[2]),'description':text(c[5]),'type':text(c[7]),'requires_weighted_accumulation':text(c[9])})
    root,*_=table('counter-profile3.xml');counter_profile_rows=sum(1 for _ in root.iter('row'))
    report={'kind':'owned_baseline_shader_timeline_diagnostic','source_commit':'65c090f994af8e37ad53ff6e14eb9d2bc4c90043','pid':app['pid'],'application_success':app['success'],'base_implementation_identity':app['study_identity'],'row_source_identity':app['source_sha256'],'environment':app['environment'],'profile_script_sha256':app['script_sha256'],'sanitizer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'corpus_sha256':app['corpus_sha256'],'independent_score_id_checks':sum(p['independent_score_id_checks'] for p in app['phases']),'owned_shader_records':len(rows),'owned_registered_shader_names':len(shader_names),'channels':dict(collections.Counter(r['channel'] for r in rows)),'depths':dict(collections.Counter(r['depth'] for r in rows)),'candidate_timing':False,'candidate_selection':False,'frozen_benchmark':False,'boundary_guard_ns':guard_ns,'phase_summaries':phases,'numeric_counter_scope':{'owned_mapping_rows':owned_mapping_rows,'owned_accelerator_ids':len(own_accelerators),'shared_accelerator_ids':sum(len(owners[a])>1 for a in own_accelerators),'accelerator_id_proves_exclusive_process_ownership':exclusive,'counter_profile3_rows':counter_profile_rows,'exported_counter_definitions':definitions,'application_bandwidth_or_spill_measurement_established':False},'schemas':schemas+shader_schemas+mapping_schemas+counter_schemas,'input_sha256':{n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ['application.json','toc.xml','shader-intervals.xml','shader-list.xml','submission-map.xml','counter-info.xml','counter-profile3.xml']},'limits':['Shader timeline intervals from an instrumented baseline-only diagnostic, not frozen performance samples.','Timeline records are not assumed kernel launch counts; overlapping and nested intervals are unioned within each reported set.','Per-shader unions can overlap each other; their totals are not additive or percentages of application cost.','Host markers include numerical validation and input work, not just the compute API; no device/host speedup ratio is derived.','Wall start has millisecond precision; phase alignment is approximate. The 2ms boundary guard omits records, so phase summaries are partial.','Shared accelerator IDs do not identify the launched process; global numeric counter values are not attributed to it.','One exported definition describes raytracing-unit activity, not application bandwidth or register spills; shader counter profile3 had no rows.','Raw traces and XML contain unrelated system context and remain private. Only exact PID-filtered shader rows are emitted.','No accepted artifact, accelerated application, portfolio advantage or causal bottleneck is established.']}
    for name,value in [('summary.json',report),('owned-shader-rows.json',{'pid':app['pid'],'rows':rows})]:
        with (OUTPUT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'owned_shader_records':len(rows),'own_registered_shader_names':len(shader_names),'phases':len(phases),'independent_score_id_checks':report['independent_score_id_checks'],'exclusive_counter_ownership':exclusive,'counter_profile_rows':counter_profile_rows}))
if __name__=='__main__':main()
