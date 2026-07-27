# Motivation

Conventional neural networks treat every weight as durable model state. That is
appropriate when every weight must encode learned information, but it is not the
only possible parameterization.

SeedNet asks a narrower question:

> How much useful computation can be supplied by a large, fixed, reproducible
> random operator while learning only a compact correction?

## State versus computation

A dense matrix is usually both:

1. a mathematical operator used during the forward pass; and
2. a stored array transferred through the memory hierarchy.

A procedural matrix separates these roles. The operator remains large, but its
persistent description becomes small:

```text
(seed, shape, hash version, distribution version)
```

The values are reconstructed only for the tile currently being multiplied.
This changes the systems trade-off from **load weights, then multiply** to
**generate weights while multiplying**.

## Why this may be useful

### Memory capacity

A model can expose an ambient random feature space much larger than the stored
base-weight state. The principal persistent costs become adapters, activations,
optimizer state, and metadata.

### Memory bandwidth

Large inference workloads are often constrained by moving parameters rather
than by arithmetic alone. Procedural generation can reduce base-weight traffic
when the generation cost is cheaper than fetching the same tile from HBM.

### Communication

Distributed training need only synchronize the trainable correction. Seeds and
format identifiers are static metadata that can be broadcast once.

### Regularization and controlled capacity

A fixed random backbone limits the directions available to learning. That may
act as a structural prior, but it also imposes a performance ceiling when the
adapter is too small.

## Intellectual lineage

SeedNet belongs to a broad family rather than a single established architecture:

- random-feature models and kernel approximations;
- reservoir computing and fixed recurrent substrates;
- random-subspace optimization;
- PRANC/NOLA-style random bases with learned coefficients;
- seed-based weight reconstruction and compression;
- low-rank adaptation over frozen operators.

The distinctive systems commitment here is that the random base matrix is not
merely frozen: it is **procedurally regenerated inside the matrix operation**.

## Research hypotheses

The repository is designed to test four falsifiable hypotheses:

1. **Storage hypothesis:** a useful model can be represented by seeds plus a
   substantially smaller trainable correction.
2. **Bandwidth hypothesis:** fused generation can outperform stored-weight GEMM
   for sufficiently bandwidth-bound shapes and hardware.
3. **Capability hypothesis:** the random substrate supplies reusable features
   whose task adaptation can be expressed at low rank.
4. **Scaling hypothesis:** useful ambient feature dimension can grow faster than
   persistent model state without making generation overhead prohibitive.

Failure on any one hypothesis is informative. A compact checkpoint that runs
slower, or a fast kernel whose random features cannot learn the task, is not a
successful SeedNet.
