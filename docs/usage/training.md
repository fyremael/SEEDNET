# Training

## Standard optimizer flow

`SeedLinear` exposes only the gain, bias, and adapter tensors as trainable
parameters. The base matrix is deterministic and has no gradient.

```python
import torch
from seednet import SeedLinear

model = torch.nn.Sequential(
    SeedLinear(768, 3072, seed=101, rank=16),
    torch.nn.GELU(),
    SeedLinear(3072, 768, seed=202, rank=16),
).cuda().half()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for x, target in loader:
    x = x.cuda().half()
    target = target.cuda()
    prediction = model(x)
    loss = objective(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## Adapter initialization

`adapter_A` receives a Kaiming initialization and `adapter_B` begins at zero.
Consequently, the low-rank branch initially contributes exactly zero while both
factors become trainable after gradients flow through the first update.

## Choosing rank

Rank controls the primary storage/capability trade-off:

\[
P_{\text{adapter}} = r(K+N).
\]

Use rank sweeps rather than assuming one universal value. At minimum, compare:

- `rank=0`: gain and bias only;
- a small adapter such as `rank=4` or `8`;
- a moderate adapter such as `rank=16` or `32`;
- a dense baseline with the same training budget.

## Checkpointing

```python
torch.save(
    {
        "model": model.state_dict(),
        "config": {
            "hash_id": "seednet-hash32-v1",
            "distribution_id": "centred-uniform-var-1-over-k-v1",
            "layers": [
                {"in_features": 768, "out_features": 3072, "seed": 101, "rank": 16},
                {"in_features": 3072, "out_features": 768, "seed": 202, "rank": 16},
            ],
        },
    },
    "checkpoint.pt",
)
```

The state dictionary contains seeds and trainable tensors, but constructor
shapes and format identifiers remain part of the durable architecture schema.

## Mixed architectures

Not every layer must be procedural. Practical experiments should compare:

- procedural MLPs with dense attention projections;
- dense embeddings and output heads with procedural internal blocks;
- selected dense boundary layers around a procedural core;
- multiple seeded bases combined by learned coefficients.
