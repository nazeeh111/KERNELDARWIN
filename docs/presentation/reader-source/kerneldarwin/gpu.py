"""CUDA adapters and verification. No CUDA import occurs in core modules."""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Any


class HardwareUnavailable(RuntimeError):
    pass


def doctor() -> dict:
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "cuda_available": False, "torch_installed": bool(importlib.util.find_spec('torch')),
              "triton_installed": bool(importlib.util.find_spec('triton')),
              "compute_sanitizer": bool(shutil.which('compute-sanitizer')),
              "ncu": bool(shutil.which('ncu'))}
    if result['torch_installed']:
        import torch
        result.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if result['cuda_available']:
            result['device'] = torch.cuda.get_device_name(0)
            result['compute_capability'] = list(torch.cuda.get_device_capability(0))
            result['supported_cuda'] = result['compute_capability'][0]>=8
    result['requirements'] = 'Linux x86_64/aarch64, NVIDIA GPU compute capability >=8.0 supported by installed PyTorch/Triton, CUDA driver; pip install .[cuda]'
    return result


def require_cuda():
    state = doctor()
    if not state['cuda_available'] or not state['triton_installed'] or not state.get('supported_cuda',False):
        raise HardwareUnavailable('CUDA + Triton required. This run cannot produce GPU performance evidence. ' + state['requirements'])
    import torch
    import triton
    return torch, triton


def device_info(torch) -> dict:
    prop = torch.cuda.get_device_properties(0)
    details = {"name": prop.name, "kind": "cuda", "compute_capability": list(torch.cuda.get_device_capability(0)),
               "memory_bytes": prop.total_memory, "multiprocessors": prop.multi_processor_count,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "uuid": str(getattr(prop,"uuid","unavailable"))}
    if shutil.which('nvidia-smi') and details['uuid']!='unavailable':
        try:
            proc = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu',
                                   '--format=csv,noheader,nounits', '--id='+details['uuid']], capture_output=True, text=True, timeout=5)
            details['nvidia_smi_observation'] = proc.stdout.strip() if proc.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            details['nvidia_smi_observation'] = 'unavailable'
    return details


def make_input(spec, seed=None, pattern='random'):
    import torch
    generator = torch.Generator(device='cuda').manual_seed(spec.seed if seed is None else seed)
    rows, cols = spec.shape
    base = torch.randn((rows, cols * spec.stride), dtype=getattr(torch, spec.dtype), device='cuda', generator=generator)
    x = base[:, ::spec.stride]
    if pattern == 'zero':
        x.zero_()
    elif pattern == 'constant':
        x.fill_(3.25)
    elif pattern == 'large':
        x.mul_(20)
    elif pattern == 'ramp':
        x.copy_(torch.linspace(-4, 4, cols, device='cuda', dtype=x.dtype).expand(rows, cols))
    return x


def reference(family: str, x, eps: float=1e-5):
    import torch
    if family == 'sum':
        return x.sum(dim=1)
    if family == 'softmax':
        return torch.softmax(x, dim=1)
    if family == 'layernorm':
        return torch.nn.functional.layer_norm(x, (x.shape[1],), eps=eps)
    if family == 'fused':
        # Explicit float32 intermediates are part of the declared semantics.
        return torch.relu(x.float() * 1.5 + .25).to(x.dtype)
    raise ValueError('Unsupported family')


def candidate(family: str, config: dict, x, eps: float=1e-5):
    import torch
    import triton
    from .kernels import row_kernel, fused_kernel
    if x.device.type != 'cuda' or x.ndim != 2:
        raise ValueError('CUDA 2D tensor required')
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Unsupported dtype')
    m, n = x.shape
    if m <= 0 or n <= 0 or n > 8192 or x.stride(0) <= 0 or x.stride(1) <= 0:
        raise ValueError('Positive shape/strides, columns <=8192 required')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be positive and finite')
    block, warps = config['block_size'], config['num_warps']
    if type(block) is not int or block not in (32,64,128,256,512,1024,2048,4096,8192,16384):
        raise ValueError('Invalid block size')
    if type(warps) is not int or warps not in (1,2,4,8,16):
        raise ValueError('Invalid warp count')
    if family == 'fused':
        out = torch.empty((m,n), device=x.device, dtype=x.dtype)
        fused_kernel[(triton.cdiv(m*n, block),)](x,out,n,m,*x.stride(),BLOCK=block,num_warps=warps)
    else:
        if block < n:
            raise ValueError('Row kernel block must cover all columns')
        out = torch.empty((m,) if family == 'sum' else (m,n), device=x.device,dtype=x.dtype)
        code = {'sum':0,'softmax':1,'layernorm':2}.get(family)
        if code is None:
            raise ValueError('Unsupported family')
        row_kernel[(m,)](x,out,n,*x.stride(),eps,code,block,num_warps=warps)
    return out


def configurations(spec) -> list[dict]:
    minimum = 1 << (spec.shape[1] - 1).bit_length()
    blocks = [128,256,512,1024] if spec.family == 'fused' else sorted(set([max(32, minimum), max(32, minimum*2)]))
    return [{'block_size':block,'num_warps':warps} for block in blocks if block <= 16384 for warps in (2,4,8)]


def verify(spec, config: dict) -> dict:
    """Independent inputs, adversarial patterns, strides and boundary shapes.

    A finite input domain is declared; NaN/Inf equivalence is not claimed.
    Repeated calls with changing tensors catch cached/constant-output mistakes.
    """
    import torch
    from .records import WorkloadSpec
    atol, rtol = {'float32':(2e-4,2e-4),'float16':(3e-2,3e-2),'bfloat16':(.15,.06)}[spec.dtype]
    max_error = 0.
    tests = []
    widths = sorted(set([1, min(31, spec.shape[1]), max(1,spec.shape[1]-1), spec.shape[1]]))
    for width in widths:
        check = WorkloadSpec(family=spec.family,shape=(min(spec.shape[0],17),width),dtype=spec.dtype,stride=spec.stride,eps=spec.eps,seed=spec.seed+991)
        for index,pattern in enumerate(('random','zero','constant','large','ramp')):
            x = make_input(check, seed=spec.seed+991+index, pattern=pattern)
            original = x.clone()
            expected = reference(spec.family,x,spec.eps)
            actual = candidate(spec.family,config,x,spec.eps)
            torch.cuda.synchronize()
            if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
                return {'passed':False,'tests':tests,'reason':'Output metadata mismatch'}
            error = (actual.float()-expected.float()).abs().max().item()
            max_error = max(max_error,error)
            valid = bool(torch.isfinite(actual).all() and torch.allclose(actual,expected,atol=atol,rtol=rtol) and torch.equal(original,x))
            if spec.family == 'softmax':
                valid &= bool(torch.allclose(actual.float().sum(1),torch.ones(actual.shape[0],device='cuda'),atol=.01,rtol=.01) and (actual>=0).all())
            tests.append({'shape':list(check.shape),'pattern':pattern,'seed':check.seed+index,'passed':valid})
            if not valid:
                return {'passed':False,'tests':tests,'max_abs_error':max_error,'atol':atol,'rtol':rtol,'reason':'Reference/semantic check failed'}
    # The benchmark shape itself is independently checked, even if large.
    for seed in (spec.seed+1777,spec.seed+1888):
        x = make_input(spec,seed=seed)
        actual,expected = candidate(spec.family,config,x,spec.eps),reference(spec.family,x,spec.eps)
        torch.cuda.synchronize()
        valid = bool(torch.isfinite(actual).all() and torch.allclose(actual,expected,atol=atol,rtol=rtol))
        tests.append({'shape':list(spec.shape),'pattern':'independent_random','seed':seed,'passed':valid})
        max_error = max(max_error,(actual.float()-expected.float()).abs().max().item())
        if not valid:
            return {'passed':False,'tests':tests,'max_abs_error':max_error,'reason':'Full-shape check failed'}
    return {'passed':True,'tests':tests,'max_abs_error':max_error,'atol':atol,'rtol':rtol,'finite_inputs_only':True,'input_mutation_checked':True}


def time_pairs(baseline: Callable, optimized: Callable, *, repeats=40, warmup=10, seed=0) -> tuple[list[float],list[float]]:
    """Normal launch CUDA events, one call/sample, randomized paired order.

    Event timing excludes compilation; it may include GPU idle gaps between
    launches. CUDA graph replay is deliberately a separate future experiment.
    """
    import torch
    for _ in range(warmup):
        baseline(); optimized()
    torch.cuda.synchronize()
    rng = random.Random(seed)
    a,b=[],[]
    for _ in range(repeats):
        times = {}
        order = [0,1]; rng.shuffle(order)
        for which in order:
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); output = (baseline if which == 0 else optimized)(); end.record()
            end.synchronize()
            times[which] = start.elapsed_time(end)*1000
            del output
        a.append(times[0]); b.append(times[1])
    return a,b


def capture_trace(fn: Callable, path: Path) -> dict:
    import torch
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        raw = json.loads(path.read_text())
        events = [{'name':e.get('name','kernel'),'start_us':e['ts'],'duration_us':e['dur'], 'category':e.get('cat','')}
                  for e in raw.get('traceEvents',[]) if e.get('ph') == 'X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'ts' in e and 'dur' in e]
        if events:
            origin = min(e['start_us'] for e in events)
            for event in events: event['start_us'] -= origin
        return {'status':'recorded','events':events,'source':path.name,'note':'Separate instrumented pass; not used as benchmark latency.'}
    except (RuntimeError,OSError,ValueError) as exc:
        return {'status':'unavailable','events':[],'reason':str(exc)}


def source_pair() -> dict:
    from . import kernels
    return {'baseline':inspect.getsource(reference),'candidate':inspect.getsource(kernels)+'\n'+inspect.getsource(candidate)}
