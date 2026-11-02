"""Descriptive CPU analysis of retained negative controls, not acceptance replay."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();rows=[];hashes={}
 for name in ['rank-v1/rejected-baseline.json','rank-longblock-assay/result.json','rank-v1-lower-load-20261101/rejected-baseline.json']:
  path=ROOT/'evidence/workload-2026-11-01'/name;raw=path.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
  for record in json.loads(raw)['noise']:
   a,b=record['baseline_a_us'],record['baseline_b_us'];orders=record['orders'];first=[a[i] if order[0]==0 else b[i] for i,order in enumerate(orders)];second=[b[i] if order[0]==0 else a[i] for i,order in enumerate(orders)]
   rows.append({'input':name,'task':record['task']['id'],'a_first_count':sum(order[0]==0 for order in orders),'paired_rounds':len(orders),'total_a_over_b':sum(a)/sum(b),'total_first_over_second':sum(first)/sum(second),'median_first_over_second':statistics.median(x/y for x,y in zip(first,second)),'mean_log_first_over_second':statistics.mean(math.log(x/y) for x,y in zip(first,second)),'conditional_a_over_b':{str(arm):sum(a[i] for i,order in enumerate(orders) if order[0]==arm)/sum(b[i] for i,order in enumerate(orders) if order[0]==arm) for arm in (0,1)}})
 report={'kind':'posthoc_negative_control_order_description','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashes,'rows':rows,'candidate_timing':False,'old_gate_results_reinterpreted':False,'limits':['These descriptive grouped ratios do not identify a cause or prove absence of order effects.','No candidate result or revised acceptance decision is produced.']}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x') as out:json.dump(report,out,indent=2,allow_nan=False);out.write('\n')
 print(json.dumps({'descriptive_rows':len(rows),'candidate_timing':False}))
if __name__=='__main__':main()
