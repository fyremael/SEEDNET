# Model format

A procedural matrix is identified by the tuple

```text
(hash_id, seed_u32, out_features, in_features, distribution_id)
```

Current identifiers:

```text
hash_id         = seednet-hash32-v1
distribution_id = centred-uniform-var-1-over-k-v1
```

## Coordinate contract

For absolute coordinates `(n, k)`:

```text
counter = uint32(n * in_features + k)
```

The counter is mixed with the low 32 bits of `seed` by the fixed v1 hash. The
upper 24 bits are mapped exactly to a float in `[0, 1)`, centred, and scaled by
`sqrt(12 / in_features)`.

## Why the format is versioned

A seed alone is ambiguous. Any change to the following changes every generated
weight:

- integer hash;
- counter layout;
- seed width or byte order;
- distribution transform;
- initialization scale;
- matrix shape.

Durable checkpoints must therefore preserve all identifiers and dimensions.
The current PyTorch state dictionary stores the seed and trainable tensors; the
architecture/configuration must preserve the remaining format metadata.

## Compatibility rule

Two implementations are compatible only when they produce bit-identical hash
outputs for the same seed and counter and agree on the floating-point mapping.
Numerical GEMM accumulation may still differ within ordinary dtype tolerances.

## Current bound

The v1 counter is 32-bit. A single procedural matrix must contain fewer than
`2**32` elements to avoid counter wraparound. A future format can introduce a
64-bit counter under a new `hash_id` without silently changing v1 checkpoints.
