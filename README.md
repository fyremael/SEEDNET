# SeedNet

[Documentation](https://fyremael.github.io/SEEDNET/) · [Repository](https://github.com/fyremael/SEEDNET)

SeedNet treats a frozen neural weight matrix as a deterministic program:

$$
u_{n,k}=H(s,nK+k)\in[0,1),
\qquad
W_{n,k}=\sqrt{\frac{12}{K}}\left(u_{n,k}-\frac{1}{2}\right),
\qquad
Y=XW^\top.
$$

Here, $s$ is the seed, $0\le n<N$, and $0\le k<K$. The v1 hash
$H$ maps the seed and row-major counter $nK+k$ to a deterministic
uniform value. Centering by $1/2$ gives zero mean, and the factor
$\sqrt{12/K}$ gives nominal variance $1/K$ (up to finite-grid effects).

The full base matrix is not stored in the model state. A stateless counter hash
regenerates each tile inside a Triton GEMM kernel. Trainable capacity is supplied
by optional per-output gains, bias, and a low-rank correction.

## What this release contains

- `SeedLinear`: procedural frozen base weights plus an optional LoRA-style correction.
- A fused Triton forward kernel computing `X @ W(seed).T`.
- A separate fused transpose kernel computing the correct input gradient
  `grad_y @ W(seed)`.
- A deterministic PyTorch reference backend for CPU execution and validation.
- Tests for determinism, dense-reference equivalence, gradients, state size, and serialization.
- A self-contained demonstration notebook and command-line examples.

## Important corrections from the prototype

1. Triton kernels are Python programs. The executable kernel is
   `kernels/fused_seed_gemm.py`, not an importable `.cu` module.
2. The input gradient requires `grad_y @ W`, not another call to the forward
   orientation.
3. The seed, hash definition, counter layout, and initialization scale are part
   of the checkpoint format. Changing any of them changes the model.
4. The fused path avoids storing or loading the base matrix from HBM. It does
   not remove the `O(MNK)` arithmetic, output storage, activation memory, or
   adapter/optimizer state.
5. A seed alone does not generally reproduce an arbitrary pretrained matrix.
   This package supports training in the seeded parameterization; pretrained
   compression additionally requires seed search and/or a learned residual.

## Installation

Linux with an NVIDIA CUDA-capable PyTorch installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[cuda,demo,dev]"
```

CPU-only validation:

```bash
pip install -e ".[dev]"
pytest
```

Documentation development:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Quick start

```python
import torch
from seednet import SeedLinear

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

layer = SeedLinear(
    in_features=4096,
    out_features=8192,
    seed=0xBEEF,
    rank=8,
    backend="auto",
).to(device=device, dtype=dtype)

x = torch.randn(32, 4096, device=device, dtype=dtype)
y = layer(x)
loss = y.float().square().mean()
loss.backward()

print(y.shape)
print(layer.storage_report())
```

`backend="auto"` selects Triton for contiguous CUDA `float16`/`bfloat16`
2-D inputs when Triton is available. Otherwise it uses the deterministic
materialized reference backend.

## Direct fused operation

```python
from seednet.functional import seed_gemm

x = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
y = seed_gemm(x, out_features=8192, seed=1234, backend="triton")
```

The direct operation supports rank-2 tensors. `SeedLinear` accepts arbitrary
leading dimensions and flattens/restores them around the fused operation.

## Model state

The base matrix is represented by:

- `seed`: signed 64-bit buffer; the low 32 bits drive the current hash stream.
- `in_features`, `out_features`: architectural metadata.
- hash/version identifier: currently `seednet-hash32-v1`.
- initialization: centred uniform with variance `1 / in_features`.

Trainable state may include:

- `gain`: one scalar per output channel.
- `bias`: one scalar per output channel.
- `adapter_A`: shape `(rank, in_features)`.
- `adapter_B`: shape `(out_features, rank)`.

The adapter is initialized to zero effect by setting `adapter_B` to zero.

## Limits

- The v1 stream uses a 32-bit counter, so a single matrix should contain fewer
  than `2**32` elements to avoid counter wraparound.
- Generated weights are pseudorandom, not cryptographically random.
- Performance is hardware- and shape-dependent. Weight generation can dominate
  small matrices, and this educational kernel is not a replacement for full
  autotuning.
- The current fused implementation computes gradients only with respect to
  activations. Seeds are discrete and frozen.
- Per-output gain, bias, and adapters are applied as ordinary PyTorch operations
  after the fused base projection.

## Commands

```bash
python examples/quickstart.py
python scripts/benchmark.py --m 512 --k 4096 --n 4096
pytest
jupyter notebook notebooks/seednet_demo.ipynb
```
