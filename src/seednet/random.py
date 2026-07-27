"""Versioned deterministic weight materialization for SeedNet."""

from __future__ import annotations

import math
import torch

HASH_ID = "seednet-hash32-v1"
"""Identifier for the current coordinate-to-random-value hash contract."""

DISTRIBUTION_ID = "centred-uniform-var-1-over-k-v1"
"""Identifier for the current centred-uniform initialization transform."""

_U32 = 0xFFFFFFFF
_INV_24 = 1.0 / 16777216.0


def _hash32_torch(counter: torch.Tensor, seed: int) -> torch.Tensor:
    """Reference implementation of unsigned 32-bit overflow arithmetic."""
    x = (counter.to(torch.int64) ^ (int(seed) & _U32)) & _U32
    x = (x ^ (x >> 16)) & _U32
    x = (x * 0x7FEB352D) & _U32
    x = (x ^ (x >> 15)) & _U32
    x = (x * 0x846CA68B) & _U32
    x = (x ^ (x >> 16)) & _U32
    return x


def materialize_seed_weight(
    out_features: int,
    in_features: int,
    seed: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize the exact v1 procedural weight matrix.

    This function is the reference implementation used for CPU execution,
    correctness tests, and comparisons with the fused Triton kernel. It stores
    the complete generated matrix and therefore does not provide SeedNet's
    intended base-weight memory saving.

    Args:
        out_features: Number of matrix rows and output channels.
        in_features: Number of matrix columns and input channels.
        seed: Integer seed. The v1 model format uses its low 32 bits.
        device: Destination device accepted by PyTorch.
        dtype: Destination floating-point dtype.

    Returns:
        Tensor with shape ``(out_features, in_features)``. Values are centred
        uniform with variance approximately ``1 / in_features``.

    Raises:
        ValueError: If dimensions are non-positive or the matrix contains
            ``2**32`` elements or more, which would wrap the v1 counter.
    """
    if out_features <= 0 or in_features <= 0:
        raise ValueError("in_features and out_features must be positive")
    count = out_features * in_features
    if count >= 2**32:
        raise ValueError("seednet-hash32-v1 requires fewer than 2**32 matrix elements")

    counter = torch.arange(count, device=device, dtype=torch.int64)
    hashed = _hash32_torch(counter, seed)
    uniform = (hashed >> 8).to(torch.float32) * _INV_24
    scale = math.sqrt(12.0 / in_features)
    weight = (uniform - 0.5) * scale
    return weight.reshape(out_features, in_features).to(dtype=dtype)
