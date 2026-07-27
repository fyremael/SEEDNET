from __future__ import annotations

import torch

from .random import (
    DISTRIBUTION_ID,
    HASH_ID,
    materialize_seed_weight,
)


def triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def _reference_seed_gemm(x: torch.Tensor, out_features: int, seed: int) -> torch.Tensor:
    weight = materialize_seed_weight(
        out_features,
        x.shape[-1],
        seed,
        device=x.device,
        dtype=x.dtype,
    )
    return x @ weight.transpose(0, 1)


def seed_gemm(
    x: torch.Tensor,
    out_features: int,
    seed: int,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute x @ W(seed).T for a contiguous rank-2 tensor."""
    if x.ndim != 2:
        raise ValueError(f"seed_gemm expects rank-2 input, got shape {tuple(x.shape)}")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if x.shape[1] * out_features >= 2**32:
        raise ValueError("seednet-hash32-v1 requires fewer than 2**32 matrix elements")
    if backend not in {"auto", "reference", "triton"}:
        raise ValueError("backend must be 'auto', 'reference', or 'triton'")

    can_fuse = (
        triton_available()
        and x.is_cuda
        and x.dtype in {torch.float16, torch.bfloat16}
    )
    use_triton = backend == "triton" or (backend == "auto" and can_fuse)

    if use_triton:
        if not can_fuse:
            raise RuntimeError(
                "Triton backend requires CUDA, Triton, and float16/bfloat16 input"
            )
        from ._triton import fused_seed_gemm
        return fused_seed_gemm(x.contiguous(), out_features, int(seed))

    return _reference_seed_gemm(x, out_features, int(seed))
