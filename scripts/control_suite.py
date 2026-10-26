"""Reproduce the shipped CPU evidence; does not execute GPU code."""
from pathlib import Path
import argparse
import json
from kerneldarwin.records import WorkloadSpec,ExperimentStore
from kerneldarwin.control import run_control
from kerneldarwin.engine import observation_rows
from kerneldarwin.selector import InterventionSelector,evaluate_grouped


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    store=ExperimentStore(args.output/'records')
    records=[]
    for family in ('sum','softmax','layernorm','fused'):
        for i,shape in enumerate(((32,31),(128,257),(128,1024),(256,513),(512,2048),(96,4095))):
            spec=WorkloadSpec(family=family,shape=shape,stride=2 if i==3 else 1,seed=2026+i)
            record=run_control(spec,repeats=40);store.save(record);records.append(record)
            print(f'{family} {shape}: {record["status"]}',flush=True)
    observations=observation_rows(records)
    model=InterventionSelector().fit(observations);model.save(args.output/'cpu-selector.json')
    evaluation=evaluate_grouped(observations)
    evaluation.update(evidence_kind='cpu_control',source_records=[r['id'] for r in records])
    (args.output/'cpu-selector.evaluation.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False)+'\n')
    manifest={'schema_version':1,'evidence_kind':'cpu_control','records':len(records),'observations':len(observations),
              'note':'Measured NumPy CPU controls. No GPU execution or GPU learning claim.',
              'accepted':sum(r['status']=='accepted' for r in records),'no_improvement':sum(r['status']=='no_improvement' for r in records)}
    (args.output/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))

if __name__=='__main__':main()
