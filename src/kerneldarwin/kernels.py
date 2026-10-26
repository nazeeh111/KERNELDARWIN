"""Authored Triton templates. Import only with CUDA dependencies installed.

Row reductions intentionally bound columns to 8192. No external source is
compiled or evaluated by the CLI. See THIRD_PARTY_NOTICES.md for prior art.
"""
import triton
import triton.language as tl


@triton.jit
def row_kernel(X, Y, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
               EPS: tl.constexpr, FAMILY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * S0 + col * S1, col < N, other=0).to(tl.float32)
    if FAMILY == 0:
        out = tl.sum(value, 0)
        tl.store(Y + row, out)
    elif FAMILY == 1:
        masked = tl.where(col < N, value, -float('inf'))
        num = tl.exp(masked - tl.max(masked, 0))
        out = num / tl.sum(num, 0)
        tl.store(Y + row * N + col, out, col < N)
    else:
        mean = tl.sum(value, 0) / N
        diff = tl.where(col < N, value - mean, 0.)
        var = tl.sum(diff * diff, 0) / N
        out = (value - mean) * tl.rsqrt(var + EPS)
        tl.store(Y + row * N + col, out, col < N)


@triton.jit
def fused_kernel(X, Y, N: tl.constexpr, M: tl.constexpr, S0: tl.constexpr,
                 S1: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // N
    col = offset % N
    value = tl.load(X + row * S0 + col * S1, offset < M * N, other=0).to(tl.float32)
    out = tl.maximum(value * 1.5 + 0.25, 0.)
    tl.store(Y + offset, out, offset < M * N)
