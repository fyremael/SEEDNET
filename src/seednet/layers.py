from __future__ import annotations

import math
import torch
from torch import nn

from .functional import seed_gemm


class SeedLinear(nn.Module):
    """Frozen procedural base projection with optional trainable low-rank delta."""

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
            delta = (flat @ self.adapter_A.transpose(0, 1))
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
        dense = self.in_features * self.out_features * element_bytes
        trainable = sum(p.numel() * p.element_size() for p in self.parameters())
        procedural = self.seed.numel() * self.seed.element_size()
        return {
            "dense_base_bytes_at_requested_precision": dense,
            "stored_trainable_bytes": trainable,
            "stored_seed_bytes": procedural,
            "base_storage_compression_ratio": dense / max(procedural, 1),
        }
