"""Small CUDA correctness target for NVIDIA Compute Sanitizer.

compute-sanitizer --tool memcheck --error-exitcode 86 python scripts/sanitizer_check.py
compute-sanitizer --tool racecheck --error-exitcode 86 python scripts/sanitizer_check.py
compute-sanitizer --tool synccheck --error-exitcode 86 python scripts/sanitizer_check.py

These are separate evidence files; running a CPU test does not mark them passed.
"""
import json
from kerneldarwin.records import WorkloadSpec
from kerneldarwin.gpu import require_cuda,configurations,verify


def main():
    require_cuda();results=[]
    for family in ('sum','softmax','layernorm','fused'):
        for dtype in ('float32','float16','bfloat16'):
            spec=WorkloadSpec(family=family,shape=(17,513),dtype=dtype,stride=2,seed=9901)
            for config in configurations(spec):
                check=verify(spec,config);results.append({'workload':spec.to_dict(),'config':config,'correctness':check})
                if not check['passed']:raise RuntimeError(json.dumps(results[-1]))
    print(json.dumps({'checks':results,'note':'Inspect sanitizer exit status AND diagnostic log independently.'},indent=2))

if __name__=='__main__':main()
