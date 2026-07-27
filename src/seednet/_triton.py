from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _uniform(counter, seed):
    x = counter.to(tl.uint32) ^ seed.to(tl.uint32)
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    x *= 0x846CA68B
    x ^= x >> 16
    return (x >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _fwd(x, y, seed, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         sxm: tl.constexpr, sxk: tl.constexpr, sym: tl.constexpr, syn: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BF16: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    jn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(0, tl.cdiv(K, BK)):
        kk = kb * BK + tl.arange(0, BK)
        a = tl.load(x + im[:, None] * sxm + kk[None, :] * sxk,
                    mask=(im[:, None] < M) & (kk[None, :] < K), other=0.0)
        ctr = jn[None, :] * K + kk[:, None]
        w = (_uniform(ctr, seed) - 0.5) * tl.sqrt(12.0 / K)
        w = w.to(tl.bfloat16) if BF16 else w.to(tl.float16)
        acc += tl.dot(a, w)
    tl.store(y + im[:, None] * sym + jn[None, :] * syn, acc,
             mask=(im[:, None] < M) & (jn[None, :] < N))


@triton.jit
def _bwd_x(gy, gx, seed, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           sgm: tl.constexpr, sgn: tl.constexpr, sxm: tl.constexpr, sxk: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BF16: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    kk = tl.program_id(1) * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BK), tl.float32)
    for nb in range(0, tl.cdiv(N, BN)):
        nn = nb * BN + tl.arange(0, BN)
        g = tl.load(gy + im[:, None] * sgm + nn[None, :] * sgn,
                    mask=(im[:, None] < M) & (nn[None, :] < N), other=0.0)
        ctr = nn[:, None] * K + kk[None, :]
        w = (_uniform(ctr, seed) - 0.5) * tl.sqrt(12.0 / K)
        w = w.to(tl.bfloat16) if BF16 else w.to(tl.float16)
        acc += tl.dot(g, w)
    tl.store(gx + im[:, None] * sxm + kk[None, :] * sxk, acc,
             mask=(im[:, None] < M) & (kk[None, :] < K))


class _SeedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_features, seed):
        M, K, N = x.shape[0], x.shape[1], int(out_features)
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _fwd[grid](x, y, int(seed), M, N, K, x.stride(0), x.stride(1),
                   y.stride(0), y.stride(1), BM=64, BN=64, BK=32,
                   BF16=x.dtype == torch.bfloat16, num_warps=4, num_stages=2)
        ctx.shape, ctx.seed = (M, N, K), int(seed)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        M, N, K = ctx.shape
        grad_y = grad_y.contiguous()
        grad_x = torch.empty((M, K), device=grad_y.device, dtype=grad_y.dtype)
        grid = (triton.cdiv(M, 64), triton.cdiv(K, 64))
        _bwd_x[grid](grad_y, grad_x, ctx.seed, M, N, K,
                     grad_y.stride(0), grad_y.stride(1),
                     grad_x.stride(0), grad_x.stride(1),
                     BM=64, BN=32, BK=64,
                     BF16=grad_y.dtype == torch.bfloat16,
                     num_warps=4, num_stages=2)
        return grad_x, None, None


def fused_seed_gemm(x: torch.Tensor, out_features: int, seed: int) -> torch.Tensor:
    if x.ndim != 2 or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expected a rank-2 CUDA float16/bfloat16 tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    return _SeedGemm.apply(x, int(out_features), int(seed))
