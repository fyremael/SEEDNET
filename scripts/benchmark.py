from __future__ import annotations

import argparse
import time
import torch

from seednet import materialize_seed_weight
from seednet.functional import seed_gemm


def timed(fn, warmup=10, steps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16)
    w = materialize_seed_weight(args.n, args.k, args.seed, device="cuda",
                                dtype=torch.float16)
    fused = lambda: seed_gemm(x, args.n, args.seed, backend="triton")
    dense = lambda: x @ w.t()

    yf, yd = fused(), dense()
    torch.cuda.synchronize()
    print("max |fused-dense|:", (yf - yd).abs().max().item())
    print("fused ms:", timed(fused))
    print("dense ms:", timed(dense))
    print("persistent dense weight MiB:", w.numel() * w.element_size() / 2**20)


if __name__ == "__main__":
    main()
