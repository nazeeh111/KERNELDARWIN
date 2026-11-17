"""Small CLI with explicit evidence states and a read-only results workbench."""
from __future__ import annotations
import argparse
import functools
import http.server
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from . import __version__
from .records import WorkloadSpec,ExperimentStore,validate_record


def dump(value):
    print(json.dumps(value,indent=2,allow_nan=False))


def positive(value):
    number=int(value)
    if number<1:raise argparse.ArgumentTypeError('Must be positive')
    return number


def parser():
    p=argparse.ArgumentParser(prog='kd',description='Evidence-first GPU performance investigations')
    p.add_argument('--version',action='version',version=__version__)
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor',help='Report GPU capability and exact missing requirements')
    for name in ('investigate','control'):
        c=commands.add_parser(name,help='Run CUDA investigation' if name=='investigate' else 'Measure actual CPU controls (not GPU results)')
        c.add_argument('--family',choices=['sum','softmax','layernorm','fused'],default='softmax')
        c.add_argument('--shape',type=positive,nargs=2,default=[128,1024],metavar=('ROWS','COLS'))
        c.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
        c.add_argument('--stride',type=positive,default=1)
        c.add_argument('--seed',type=int,default=0)
        c.add_argument('--repeats',type=positive,default=40)
        c.add_argument('--store',type=Path,default=Path('.runs'))
        if name=='investigate':
            c.add_argument('--max-candidates',type=positive,default=6)
            c.add_argument('--wall-seconds',type=positive,default=180)
            c.add_argument('--warmup',type=positive,default=10)
            c.add_argument('--trace',action='store_true')
            c.add_argument('--model',type=Path)
    for name in ('show','validate'):
        c=commands.add_parser(name);c.add_argument('record',type=Path)
    c=commands.add_parser('train',help='Train on recorded observations, with grouped held-out evaluation')
    c.add_argument('--store',type=Path,default=Path('.runs'));c.add_argument('--output',type=Path,required=True)
    c.add_argument('--device-kind',choices=['cpu','cuda'],required=True)
    c=commands.add_parser('regress',help='Compare compatible known runs; no automatic causal attribution')
    c.add_argument('before',type=Path);c.add_argument('after',type=Path)
    c=commands.add_parser('export',help='Export callable for an accepted CUDA candidate')
    c.add_argument('record',type=Path);c.add_argument('--output',type=Path,required=True)
    c=commands.add_parser('lab',help='Serve read-only LAB on localhost')
    c.add_argument('--port',type=positive,default=8765);c.add_argument('--directory',type=Path)
    c=commands.add_parser('_worker',help=argparse.SUPPRESS)
    c.add_argument('spec');c.add_argument('output',type=Path)
    c.add_argument('--max-candidates',type=positive,default=6);c.add_argument('--repeats',type=positive,default=40)
    c.add_argument('--warmup',type=positive,default=10);c.add_argument('--wall-seconds',type=positive,default=180);c.add_argument('--trace',action='store_true');c.add_argument('--model',type=Path)
    return p


def read_record(path):
    if path.stat().st_size > 50_000_000:raise ValueError('Record too large (>50 MB)')
    record=json.loads(path.read_text());validate_record(record);return record


def regression(before,after):
    from .statistics import summarize
    fields=('family','shape','dtype','stride','eps')
    if any(before['workload'].get(k)!=after['workload'].get(k) for k in fields):raise ValueError('Workload configurations differ')
    stable_device_keys=('kind','name','compute_capability','memory_bytes','multiprocessors')
    if any(before['device'].get(k)!=after['device'].get(k) for k in stable_device_keys):raise ValueError('Device identity differs; compare controlled runs')
    if before['evidence_kind']!=after['evidence_kind']:raise ValueError('Evidence types differ')
    if before['environment']!=after['environment']:raise ValueError('Software environments differ; record a controlled environment comparison separately')
    a=before['baseline']['samples_us'];b=after['baseline']['samples_us']
    if min(len(a),len(b))<20:raise ValueError('At least 20 samples per run required')
    import random,statistics
    rng=random.Random(0);ratios=[]
    for _ in range(2000):ratios.append(statistics.median(rng.choices(b,k=len(b)))/statistics.median(rng.choices(a,k=len(a))))
    ratios.sort();ratio=summarize(b)['median_us']/summarize(a)['median_us']
    return {'latency_ratio_after_over_before':ratio,'ci_low':ratios[50],'ci_high':ratios[1949],
            'regression_detected':ratios[50]>1.03,'method':'Independent bootstrap medians, 95% interval, >3% threshold',
            'limitations':['Between-run confounding remains possible. This comparison does not prove a cause or repair.'],
            'device_telemetry_changed':before['device'].get('nvidia_smi_observation')!=after['device'].get('nvidia_smi_observation'),
            'next_experiment':'Re-run both implementations interleaved on the same device; investigate launch, layout, fusion or scheduling changes.'}


def export_record(record,output):
    if record['evidence_kind']!='cuda_run' or record['status']!='accepted':raise ValueError('Only accepted CUDA evidence can export an optimized kernel')
    winner=next((e for e in record['experiments'] if e['id']==record['winner_id']),None)
    if not winner or not winner.get('confirmation',{}).get('comparison',{}).get('accepted'):raise ValueError('Independent confirmation required')
    from . import gpu
    import hashlib
    current_hash=hashlib.sha256(json.dumps(gpu.source_pair(),sort_keys=True).encode()).hexdigest()
    if record.get('provenance',{}).get('source_sha256')!=current_hash:raise ValueError('Current kernel source differs from verified source; re-investigate before export')
    spec=WorkloadSpec.from_dict(record['workload']);config=winner['config']
    # Exact spec/config values serialize as data. No source is executed from records.
    metadata={'workload':spec.to_dict(),'config':config,'record_id':record['id'],'safety':winner.get('safety'),'device':record['device'],'actual_strides':record['workload'].get('actual_strides'),'source_sha256':current_hash}
    source='''"""KERNELDARWIN exported callable. Requires installed kerneldarwin[cuda]."""\nimport json\nfrom kerneldarwin.gpu import candidate\nSPEC = json.loads(RECORD_DATA)\n\ndef run(x):\n    import hashlib\n    from kerneldarwin.gpu import source_pair\n    if hashlib.sha256(json.dumps(source_pair(),sort_keys=True).encode()).hexdigest() != SPEC['source_sha256']:\n        raise ValueError('Installed kernels differ from verified source; re-investigate')\n    w = SPEC['workload']\n    if list(x.shape) != list(w['shape']) or str(x.dtype).removeprefix('torch.') != w['dtype']:\n        raise ValueError('Input differs from verified shape/dtype')\n    if x.device.type != 'cuda' or x.stride(1) != w['stride']:\n        raise ValueError('Input differs from verified CUDA/stride contract')\n    if SPEC['actual_strides'] and list(x.stride()) != SPEC['actual_strides']:\n        raise ValueError('Input differs from verified full stride contract')\n    import torch\n    if torch.cuda.get_device_name(x.device) != SPEC['device']['name']:\n        raise ValueError('GPU differs from verified device')\n    return candidate(w['family'], SPEC['config'], x, w['eps'])\n'''
    source=source.replace('RECORD_DATA',repr(json.dumps(metadata,allow_nan=False)))
    if output.exists():raise ValueError('Output exists; choose a new export path')
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with output.open('x', encoding='utf-8') as handle:
            handle.write(source)
    except FileExistsError as exc:
        raise ValueError('Output exists; choose a new export path') from exc
    return {'export':str(output),'record_id':record['id'],'safety':winner.get('safety')}


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='doctor':
            from .gpu import doctor
            dump(doctor());return 0
        if args.command in ('show','validate'):
            record=read_record(args.record);dump(record if args.command=='show' else {'valid':True,'id':record['id']});return 0
        if args.command in ('investigate','control'):
            if args.repeats<20:raise ValueError('At least 20 repeats required for performance gates')
            spec=WorkloadSpec(family=args.family,shape=tuple(args.shape),dtype=args.dtype,stride=args.stride,seed=args.seed);spec.validate()
            if args.repeats>1000 or spec.shape[0]*spec.shape[1]*spec.stride>100_000_000:raise ValueError('Resource budget exceeded')
            if args.command=='control':
                from .control import run_control
                record=run_control(spec,repeats=args.repeats)
            else:
                from .gpu import doctor
                from .engine import blocked_record,run_isolated
                state=doctor()
                if not state['cuda_available'] or not state['triton_installed']:
                    record=blocked_record(spec,'CUDA + Triton unavailable. '+state['requirements'])
                else:
                    args.store.mkdir(parents=True,exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix='kd-worker-') as directory:
                        result=run_isolated(spec,Path(directory)/'record.json',max_candidates=args.max_candidates,repeats=args.repeats,warmup=args.warmup,wall_seconds=args.wall_seconds,trace=args.trace,model_path=args.model)
                        if result.returncode:raise RuntimeError('CUDA worker failed: '+result.stderr[-2000:])
                        record=read_record(Path(directory)/'record.json')
                        trace=Path(directory)/'trace.json'
                        if trace.exists():
                            trace_dir=args.store/'traces';trace_dir.mkdir(exist_ok=True)
                            target=trace_dir/(record['id']+'.trace.json');target.write_bytes(trace.read_bytes());record['trace']['source']='traces/'+target.name
            path=ExperimentStore(args.store).save(record)
            dump({'record':str(path),'status':record['status'],'evidence_kind':record['evidence_kind'],'winner_id':record['winner_id'],'summary':record['diagnosis']['summary']})
            return 2 if record['status']=='blocked' else 0
        if args.command=='_worker':
            from .engine import investigate_cuda
            spec=WorkloadSpec.from_dict(json.loads(args.spec))
            model=None
            if args.model:
                from .selector import InterventionSelector
                model=InterventionSelector.load(args.model)
                if model.device_kind!='cuda':raise ValueError('CUDA investigation requires CUDA-trained policy')
            record=investigate_cuda(spec,max_candidates=args.max_candidates,repeats=args.repeats,warmup=args.warmup,wall_seconds=args.wall_seconds,trace_path=args.output.with_name('trace.json') if args.trace else None,model=model)
            validate_record(record);args.output.write_text(json.dumps(record,allow_nan=False));return 0
        if args.command=='train':
            from .engine import observation_rows
            from .selector import InterventionSelector,evaluate_grouped
            rows=[row for row in observation_rows(ExperimentStore(args.store).list()) if row['device_kind']==args.device_kind]
            if len(rows)<12:raise ValueError('Need at least 12 measured observations across multiple workload groups')
            groups={row['group'] for row in rows}
            if len(groups)<2:raise ValueError('At least two workload families required for held-out evaluation')
            report=evaluate_grouped(rows)
            model=InterventionSelector();model.fit(rows)
            args.output.parent.mkdir(parents=True,exist_ok=True);model.save(args.output)
            report['limitations']=report.get('limitations',[])+(['CPU controls do not demonstrate GPU selection gains.'] if args.device_kind=='cpu' else ['Historical workload results do not establish generalization to new devices.'])
            report.update(evidence_kind='cpu_control' if args.device_kind=='cpu' else 'cuda_run',observations=len(rows),
                          device_kind=args.device_kind,training_record_ids=sorted({r['record_id'] for r in rows}))
            report_path=args.output.with_suffix('.evaluation.json');report_path.write_text(json.dumps(report,indent=2,allow_nan=False));dump(report);return 0
        if args.command=='regress':dump(regression(read_record(args.before),read_record(args.after)));return 0
        if args.command=='export':dump(export_record(read_record(args.record),args.output));return 0
        if args.command=='lab':
            root=args.directory or Path(__file__).resolve().parent/'lab'
            if not args.directory and not root.is_dir():root=Path(__file__).resolve().parents[2]/'lab'
            if not (root/'index.html').is_file():raise ValueError('LAB files missing; use --directory path/to/repo/lab')
            handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(root))
            print(f'KERNELDARWIN LAB: http://127.0.0.1:{args.port} (read-only; Ctrl-C to stop)',flush=True)
            with http.server.ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:server.serve_forever()
        return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError,subprocess.TimeoutExpired) as exc:
        print(f'kerneldarwin: {exc}',file=sys.stderr);return 1
    except ImportError as exc:
        print(f'Missing optional dependency: {exc}. Install .[control] or .[cuda] for the requested backend.',file=sys.stderr);return 1
    except KeyboardInterrupt:
        return 130

if __name__=='__main__':raise SystemExit(main())
