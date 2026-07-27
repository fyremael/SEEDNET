# SeedNet model format v1

A procedural matrix is identified by the tuple:

```text
(hash_id, seed_u32, out_features, in_features, distribution_id)
```

Current values:

```text
hash_id         = seednet-hash32-v1
distribution_id = centred-uniform-var-1-over-k-v1
```

For absolute coordinates `(n, k)`, the counter is:

```text
counter = uint32(n * in_features + k)
```

The counter is mixed with the low 32 bits of `seed` by a fixed integer hash.
The resulting upper 24 bits are mapped exactly to a float in `[0, 1)`, centred,
and scaled by `sqrt(12 / in_features)`.

Checkpoint compatibility therefore requires preserving the hash algorithm,
counter convention, shape, distribution, and seed. A seed without this metadata
is not a complete model description.
