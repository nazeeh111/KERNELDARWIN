#!/usr/bin/env python3
"""Account four fixed owned-process tables from the existing baseline trace.

Raw exports stay private. This is a historical evidence extractor, not a
profiler, benchmark runner or general trace service.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

APPLICATION_SHA256 = '95bffdc9b11a298dc2b37b82abec422cac94886a1e81f6350f12943c0a749bdd'
TABLES = {
 'submissions': ('kd-rank-owned-command-submissions-20261101.xml', 'metal-application-command-buffer-submissions',
  ('start','duration','event-type','gpu','track-label','num-encoders','encoder-time','process','thread','event-icon','event-label','encoder-time-label','frame-number','color','cmdbuffer-id')),
 'encoders': ('kd-rank-owned-encoder-list-20261101.xml', 'metal-application-encoders-list',
  ('start','duration','thread','process','gpu','frame-number','cmdbuffer-label','cmdbuffer-label-indexed','encoder-label','encoder-label-indexed','event-type','cmdbuffer-id','encoder-id')),
 'allocations': ('kd-rank-owned-resource-events-20261101.xml', 'metal-resource-allocations',
  ('timestamp','duration','process','thread','resource-id','parent-resource-id','label','vidmem-bytes','sysmem-bytes','resource-size','event-label','gpu','resource-type','event-type','backtrace','event-icon')),
 'memory_levels': ('kd-rank-owned-resource-accounting-20261101.xml', 'metal-current-allocated-size',
  ('start','duration','end','process','current-allocated-size','label','track-name','color')),
}


def bounded_bytes(path):
 size = path.stat().st_size
 if size > 20_000_000:
  raise ValueError('Bounded historical evidence exceeds 20 MB')
 with path.open('rb') as handle:
  content = handle.read(20_000_001)
 if len(content) != size:
  raise ValueError('Historical evidence changed while reading')
 return content


def digest(path):
 return hashlib.sha256(bounded_bytes(path)).hexdigest()


def owned_rows(content, schema_name, columns, pid):
 root = ET.fromstring(content)
 schemas = list(root.iter('schema'))
 if len(schemas) != 1 or schemas[0].get('name') != schema_name:
  raise ValueError('Unexpected historical table schema')
 if tuple(c.findtext('mnemonic') for c in schemas[0].findall('col')) != columns:
  raise ValueError('Historical table columns changed')
 index = {e.get('id'):e for e in root.iter() if e.get('id')}
 def resolve(element):
  seen = set()
  while element is not None and element.get('ref'):
   ref = element.get('ref')
   if ref in seen:
    raise ValueError('Cyclic XML reference')
   seen.add(ref)
   element = index[ref]
  return element
 def text(element):
  element = resolve(element)
  return '' if element is None else (element.text or '').strip()
 def fmt(element):
  element = resolve(element)
  return '' if element is None else element.get('fmt', text(element))
 for row in root.iter('row'):
  children = list(row)
  process_column = columns.index('process')
  if len(children) <= process_column:
   continue
  process = resolve(children[process_column])
  if process is None or text(process.find('pid')) != str(pid):
   continue
  # Validate widths only after ownership: unrelated records may use a different
  # row shape and are never materialized or emitted.
  cells = dict(zip(columns, children, strict=True))
  yield cells, text, fmt


def integer(cells, text, name, optional=False):
 value = text(cells[name])
 if optional and not value:
  return None
 result = int(value)
 if result < 0:
  raise ValueError('Negative count, size or timestamp')
 return result


def extract(application, raw_directory):
 application_bytes = bounded_bytes(application)
 if hashlib.sha256(application_bytes).hexdigest() != APPLICATION_SHA256:
  raise ValueError('Expected exact successful owned baseline application record')
 app = json.loads(application_bytes)
 if app['success'] is not True or app['pid'] != 47528:
  raise ValueError('Unexpected historical application identity')
 pid = app['pid']; output = {}; hashes = {}
 for kind, (filename, schema, columns) in TABLES.items():
  path = raw_directory / filename; content = bounded_bytes(path)
  hashes[filename] = hashlib.sha256(content).hexdigest(); rows = []
  for cells, text, fmt in owned_rows(content, schema, columns, pid):
   if kind == 'submissions':
    rows.append({'start_ns':integer(cells,text,'start'),'duration_ns':integer(cells,text,'duration'),
     'event_type':fmt(cells['event-type']),'encoder_count':integer(cells,text,'num-encoders'),
     'encoding_time_ns':integer(cells,text,'encoder-time',True),'command_buffer_id':text(cells['cmdbuffer-id'])})
   elif kind == 'encoders':
    rows.append({'start_ns':integer(cells,text,'start'),'duration_ns':integer(cells,text,'duration'),
     'event_type':fmt(cells['event-type']),'command_buffer_id':text(cells['cmdbuffer-id']),
     'encoder_id':text(cells['encoder-id']),'encoder_label':fmt(cells['encoder-label'])})
   elif kind == 'allocations':
    rows.append({'start_ns':integer(cells,text,'timestamp'),'duration_ns':integer(cells,text,'duration',True),
     'resource_id':text(cells['resource-id']),'parent_resource_id':text(cells['parent-resource-id']),
     'vidmem_bytes':integer(cells,text,'vidmem-bytes',True),'sysmem_bytes':integer(cells,text,'sysmem-bytes',True),
     'resource_size_bytes':integer(cells,text,'resource-size'),
     'resource_type':fmt(cells['resource-type']),'event_type':fmt(cells['event-type'])})
   else:
    rows.append({'start_ns':integer(cells,text,'start'),'duration_ns':integer(cells,text,'duration'),
     'end_ns':integer(cells,text,'end'),'current_allocated_bytes':integer(cells,text,'current-allocated-size')})
  if not rows:
   raise ValueError('No owned rows for required historical table: '+kind)
  output[kind] = rows
 submissions, encoders, allocations, levels = (output[k] for k in TABLES)
 buffers = {r['command_buffer_id'] for r in submissions}
 per_buffer = collections.Counter(r['command_buffer_id'] for r in encoders)
 if len(buffers) != len(submissions) or len({r['encoder_id'] for r in encoders}) != len(encoders):
  raise ValueError('Duplicate command buffer or encoder identity')
 if any(r['command_buffer_id'] not in buffers for r in encoders):
  raise ValueError('Owned encoder lacks owned command submission')
 if any(r['encoder_count'] != per_buffer[r['command_buffer_id']] for r in submissions):
  raise ValueError('Reported encoder counts disagree with actual encoder table')
 if len({r['resource_id'] for r in allocations}) != len(allocations):
  raise ValueError('Duplicate resource allocation identity')
 if any(r['end_ns']-r['start_ns'] != r['duration_ns'] for r in levels):
  raise ValueError('Memory interval units disagree')
 summary = {'kind':'owned_baseline_resource_accounting','source_commit':'65c090f994af8e37ad53ff6e14eb9d2bc4c90043',
  'application_sha256':APPLICATION_SHA256,'pid':pid,'environment':app['environment'],
  'base_implementation_identity':app['study_identity'],'extractor_sha256':digest(Path(__file__)),
  'raw_export_sha256':hashes,'owned_command_buffer_submissions':len(submissions),
  'reported_encoder_count_total':sum(r['encoder_count'] for r in submissions),
  'submissions_with_zero_reported_encoders':sum(r['encoder_count']==0 for r in submissions),
  'owned_encoder_records':len(encoders),'encoder_labels':dict(collections.Counter(r['encoder_label'] for r in encoders)),
  'owned_buffer_allocation_events':sum(r['resource_type']=='Buffer' and r['event_type']=='Allocation' for r in allocations),
  'allocation_resource_types':dict(collections.Counter(r['resource_type'] for r in allocations)),
  'allocation_event_types':dict(collections.Counter(r['event_type'] for r in allocations)),
  'allocation_resource_size_sum_bytes':sum(r['resource_size_bytes'] for r in allocations),
  'owned_mtl_current_allocated_size_records':len(levels),
  'max_observed_mtl_current_allocated_size_bytes':max(r['current_allocated_bytes'] for r in levels),
  'cross_table_encoder_counts_match':True,'candidate_timing':False,'candidate_selection':False,'accepted_artifact':False,
  'unavailable':['Exact kernel-dispatch count','Implicit contiguous-copy count','Tensor materialization count','CPU/GPU transfer byte count','Allocation-to-tensor association','Pure GPU elapsed time for each application call'],
  'limits':['Whole instrumented baseline process includes setup, compilation, input preparation, numerical validation and searches; counts are not per-search rates.',
   'Command-buffer and encoder counts are distinct from shader records and kernel launches.',
   'Encoder labels are recorded labels, not a guarantee that all copy activity uses a separately named blit encoder.',
   'Summed resource creation sizes are cumulative recorded allocation sizes, not live bytes or transfer/copy volume.',
   'MTLDevice.currentAllocatedSize and MLX peak allocation measure different resource scopes; neither is physical GPU residency.',
   'Observed memory levels are trace records; an unobserved transient maximum is not ruled out.',
   'Missing allocation duration or zero-encoder encoding-time fields remain null, not zero latency.',
   'No raw XML, foreign process row or allocation backtrace is emitted. Local provenance is not cryptographic attestation.',
   'No causal bottleneck, application headroom or acceleration follows from these counts.']}
 return summary, output


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--application',type=Path,default=Path(__file__).parent/'application.json')
 parser.add_argument('--raw-directory',type=Path,default=Path('/private/tmp'))
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 # Reserve only the fresh output folder. Existing evidence is never replaced.
 args.output.mkdir(parents=True,exist_ok=False)
 summary, rows=extract(args.application,args.raw_directory)
 for name,value in [('resource-accounting.json',summary),('owned-resource-accounting-rows.json',{'pid':summary['pid'],'tables':rows})]:
  with (args.output/name).open('x') as handle:
   json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
 print(json.dumps({k:summary[k] for k in ['owned_command_buffer_submissions','owned_encoder_records','owned_buffer_allocation_events','max_observed_mtl_current_allocated_size_bytes','cross_table_encoder_counts_match']}))

if __name__=='__main__':main()
