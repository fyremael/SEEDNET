"""Functional interface for procedural matrix multiplication."""

from __future__ import annotations

import torch

from .random import (
    DISTRIBUTION_ID,
    HASH_ID,
    materialize_seed_weight,
)


def triton_available() -> bool:
    """Return whether the fused Triton backend can be considered.

    The check requires both a CUDA-capable PyTorch runtime and a successful
    Triton import. It does not compile a kernel or guarantee that every device,
    shape, or dtype is supported.

    Returns:
        ``True`` when CUDA is available and Triton imports successfully;
        otherwise ``False``.
    """
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
    """Multiply by a deterministic procedural matrix.

    Computes ``x @ W(seed).T`` where ``x`` has shape ``(M, K)`` and the
    generated matrix has shape ``(out_features, K)``. The fused Triton backend
    regenerates weight tiles inside the GEMM and does not persist the base
    matrix in global memory. The reference backend materializes the matrix for
    correctness and fallback execution.

    Args:
        x: Contiguous rank-2 input tensor with shape ``(M, K)``. The Triton
            backend requires a CUDA ``float16`` or ``bfloat16`` tensor.
        out_features: Output width ``N`` and first dimension of the generated
            matrix.
        seed: Integer seed. The v1 format uses its low 32 bits.
        backend: ``"auto"`` selects Triton when compatible and otherwise uses
            the reference path. ``"triton"`` requires the fused backend.
            ``"reference"`` always materializes the generated matrix.

    Returns:
        A tensor with shape ``(M, out_features)`` and the same dtype and device
        as ``x``.

    Raises:
        ValueError: If dimensions or backend values are invalid, or the v1
            32-bit coordinate space would overflow.
        RuntimeError: If ``backend="triton"`` is requested without a compatible
            CUDA/Triton environment.

    Note:
        The seed is discrete and frozen. Autograd computes gradients with
        respect to ``x`` but not with respect to ``seed`` or ``out_features``.
    """
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
