# SeedNet

**Weights as deterministic programs rather than stored arrays.**

SeedNet represents a frozen base matrix by a seed, a shape, and a versioned
counter-hash rule. The matrix is regenerated tile by tile inside a Triton GEMM
kernel, while a compact trainable correction supplies task-specific capacity.

\[
W_{n,k}=\sqrt{\frac{12}{K}}\left(
  h(\text{seed},nK+k)-\frac{1}{2}
\right),
\qquad
Y=XW^\top + \Delta Y.
\]

<div class="grid cards" markdown>

-   :material-memory:{ .lg .middle } **Reduce persistent weight state**

    ---

    The frozen base matrix is represented by metadata measured in bytes rather
    than a dense tensor measured in megabytes or gigabytes.

-   :material-chip:{ .lg .middle } **Trade bandwidth for arithmetic**

    ---

    Weight tiles are generated where they are consumed, avoiding persistent HBM
    loads for the procedural base matrix.

-   :material-tune-variant:{ .lg .middle } **Train a compact correction**

    ---

    Per-channel gains, bias, and an optional low-rank adapter provide a small
    trainable state over a fixed random feature map.

-   :material-source-branch:{ .lg .middle } **Preserve reproducibility**

    ---

    The seed, hash identifier, shape, and distribution identifier form an
    explicit model-format contract.

</div>

## Start here

```python
import torch
from seednet import SeedLinear

layer = SeedLinear(
    in_features=4096,
    out_features=8192,
    seed=0xBEEF,
    rank=8,
    backend="auto",
).cuda().half()

x = torch.randn(32, 4096, device="cuda", dtype=torch.float16)
y = layer(x)
y.float().square().mean().backward()
```

`backend="auto"` selects the fused Triton path for compatible CUDA tensors and
falls back to a deterministic PyTorch reference implementation elsewhere.

!!! important "What SeedNet does not claim"
    A seed does not normally approximate an arbitrary pretrained matrix by
    itself. This repository implements training in a seeded parameterization.
    Compressing an existing checkpoint requires seed search, a learned residual,
    or both.

## Documentation map

- [Motivation](motivation.md) explains why procedural weights are worth testing.
- [Architecture](design/architecture.md) specifies forward, backward, state, and backends.
- [Usage](usage/quickstart.md) covers installation, training, checkpointing, and benchmarking.
- [API reference](api/index.md) is generated from the current Python docstrings.
- [Limits and failure modes](design/limitations.md) states the boundaries explicitly.
