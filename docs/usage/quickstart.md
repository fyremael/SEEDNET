# Quickstart

## A seeded linear layer

```python
import torch
from seednet import SeedLinear

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

layer = SeedLinear(
    in_features=256,
    out_features=512,
    seed=0xBEEF,
    rank=8,
    bias=True,
    trainable_gain=True,
    backend="auto",
).to(device=device, dtype=dtype)

x = torch.randn(16, 256, device=device, dtype=dtype, requires_grad=True)
y = layer(x)
y.float().square().mean().backward()

print(y.shape)       # torch.Size([16, 512])
print(x.grad.shape)  # torch.Size([16, 256])
```

## Direct procedural GEMM

Use the functional API when no adapter, gain, or bias is required:

```python
import torch
from seednet import seed_gemm

x = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
y = seed_gemm(x, out_features=8192, seed=1234, backend="triton")
```

This computes `x @ W(seed).T` without storing `W(seed)` persistently.

## Materialize for inspection

```python
from seednet import materialize_seed_weight

weight = materialize_seed_weight(
    out_features=512,
    in_features=256,
    seed=0xBEEF,
    device="cpu",
)
```

Materialization is appropriate for tests and analysis, not for obtaining the
fused memory benefit.

## Measure represented storage

```python
print(layer.storage_report(element_bytes=2))
```

The report compares the procedural seed state with a hypothetical dense base
matrix at the requested bytes per element. It reports adapter and gain storage
separately as actual PyTorch parameter bytes.
