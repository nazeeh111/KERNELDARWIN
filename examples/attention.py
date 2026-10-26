"""Application-level attention benchmark using a verified softmax export.

Run on NVIDIA only, after `kd investigate --family softmax --shape 128 1024`
and `kd export RECORD --output optimized.py` succeed. This example does not
supply invented results or claim the entire attention implementation is fused.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from kerneldarwin.gpu import require_cuda,time_pairs
from kerneldarwin.statistics import summarize,compare_paired


def main():
    p=argparse.ArgumentParser();p.add_argument('export',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch,_=require_cuda()
    module_spec=importlib.util.spec_from_file_location('optimized',args.export)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    w=module.SPEC['workload']
    if w['family']!='softmax' or w['stride']!=1:raise ValueError('Contiguous softmax export required')
    m,n=w['shape'];dtype=getattr(torch,w['dtype'])
    torch.manual_seed(8701)
    q=torch.randn((m,64),device='cuda',dtype=dtype)
    k=torch.randn((n,64),device='cuda',dtype=dtype)
    v=torch.randn((n,64),device='cuda',dtype=dtype)
    def eager():return torch.softmax((q@k.T)/8,dim=-1)@v
    def optimized():return module.run((q@k.T)/8)@v
    torch.testing.assert_close(optimized(),eager(),atol=.03 if dtype!=torch.float32 else 5e-4,rtol=.03 if dtype!=torch.float32 else 5e-4)
    a,b=time_pairs(eager,optimized,repeats=80,seed=8701)
    result={'evidence_kind':'cuda_run','application':'single-head attention QK^T / sqrt(d) -> softmax -> V',
            'device':torch.cuda.get_device_name(0),'dtype':w['dtype'],'shape':[m,n,64],
            'eager_us':a,'optimized_us':b,'eager':summarize(a),'optimized':summarize(b),'comparison':compare_paired(a,b),
            'source_record':module.SPEC['record_id']}
    try:
        compiled=torch.compile(eager,fullgraph=True);compiled();torch.cuda.synchronize()
        torch.testing.assert_close(compiled(),eager(),atol=.03 if dtype!=torch.float32 else 5e-4,rtol=.03 if dtype!=torch.float32 else 5e-4)
        a,c=time_pairs(eager,compiled,repeats=80,seed=8702)
        result['torch_compile']={'samples_us':c,'summary':summarize(c),'paired_eager_us':a,'comparison':compare_paired(a,c)}
    except Exception as exc:result['torch_compile']={'status':'unavailable','reason':str(exc)}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
