# Public API

The API reference is rendered from the current source docstrings by
`mkdocstrings`. Documentation builds therefore fail when source references or
signatures become invalid.

The supported import surface is:

```python
from seednet import (
    DISTRIBUTION_ID,
    HASH_ID,
    SeedLinear,
    materialize_seed_weight,
    seed_gemm,
    triton_available,
)
```

- [`SeedLinear`](seedlinear.md) is the procedural PyTorch layer.
- [Functional interface](functional.md) covers direct seeded GEMM,
  materialization, and backend detection.
- [Model identifiers](identifiers.md) exposes the versioned format constants.
