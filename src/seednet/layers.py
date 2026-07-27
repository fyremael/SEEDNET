"""PyTorch layers built on procedural frozen weight matrices."""

from __future__ import annotations

import math
import torch
from torch import nn

from .functional import seed_gemm


class SeedLinear(nn.Module):
    """Linear projection with a procedural frozen base and low-rank correction.

    The layer computes a generated base projection, applies an optional
    trainable output gain, and adds a LoRA-style low-rank correction and bias:

    ``y = seed_gemm(x) * gain + ((x @ A.T) @ B.T) / rank + bias``.

    The dense base matrix is absent from the parameter state. It is determined
    by ``seed``, the matrix dimensions, and the versioned SeedNet model format.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        seed: Integer seed for the frozen base matrix. The v1 format uses its
            low 32 bits.
        rank: Rank of the trainable correction. Set to zero to disable the
            adapter.
        bias: Whether to include a trainable output bias.
        trainable_gain: Whether the per-output multiplicative gain receives
            gradients. The gain tensor is present even when frozen.
        backend: Procedural GEMM backend: ``"auto"``, ``"reference"``, or
            ``"triton"``.

    Shape:
        - Input: ``(*, in_features)``
        - Output: ``(*, out_features)``

    Note:
        ``adapter_B`` is initialized to zero, so the adapter begins with exactly
        zero contribution. The initial function is therefore the seeded base
        projection, gain, and optional bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int,
        *,
        rank: int = 0,
        bias: bool = True,
        trainable_gain: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if rank < 0 or rank > min(in_features, out_features):
            raise ValueError("rank must satisfy 0 <= rank <= min(in_features, out_features)")
        if in_features * out_features >= 2**32:
            raise ValueError("seednet-hash32-v1 requires fewer than 2**32 matrix elements")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.backend = backend
        self.register_buffer("seed", torch.tensor(int(seed), dtype=torch.int64))

        gain = torch.ones(out_features)
        self.gain = nn.Parameter(gain, requires_grad=trainable_gain)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        if rank:
            self.adapter_A = nn.Parameter(torch.empty(rank, in_features))
            self.adapter_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.adapter_A, a=math.sqrt(5))
        else:
            self.register_parameter("adapter_A", None)
            self.register_parameter("adapter_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the seeded base projection and trainable correction.

        Args:
            x: Tensor whose final dimension equals ``in_features``.

        Returns:
            Tensor with the same leading dimensions as ``x`` and final
            dimension ``out_features``.

        Raises:
            ValueError: If the final input dimension does not match
                ``in_features``.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected final dimension {self.in_features}, got {x.shape[-1]}"
            )
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        base = seed_gemm(
            flat,
            self.out_features,
            int(self.seed.item()),
            backend=self.backend,
        )
        y = base * self.gain
        if self.rank:
            delta = flat @ self.adapter_A.transpose(0, 1)
            delta = delta @ self.adapter_B.transpose(0, 1)
            y = y + delta / self.rank
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*original_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"seed={int(self.seed.item())}, rank={self.rank}, backend={self.backend!r}"
        )

    def storage_report(self, element_bytes: int = 2) -> dict[str, int | float]:
        """Summarize represented dense storage and actual layer state.

        Args:
            element_bytes: Hypothetical bytes per element for the dense base
                matrix comparison, typically 2 for FP16/BF16 or 4 for FP32.

        Returns:
            Dictionary containing hypothetical dense base bytes, actual
            trainable parameter bytes, seed bytes, and the dense-base-to-seed
            storage ratio.
        """
        if element_bytes <= 0:
            raise ValueError("element_bytes must be positive")
        dense = self.in_features * self.out_features * element_bytes
        trainable = sum(p.numel() * p.element_size() for p in self.parameters())
        procedural = self.seed.numel() * self.seed.element_size()
        return {
            "dense_base_bytes_at_requested_precision": dense,
            "stored_trainable_bytes": trainable,
            "stored_seed_bytes": procedural,
            "base_storage_compression_ratio": dense / max(procedural, 1),
        }
