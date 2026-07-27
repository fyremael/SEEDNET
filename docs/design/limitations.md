# Limits and failure modes

SeedNet changes where weight values come from. It does not repeal the cost of
matrix multiplication.

## No free FLOPs

The base projection remains \(O(MNK)\). Procedural generation adds integer and
floating-point work. The fused path is useful only when saved memory traffic is
more valuable than the added generation cost.

## No automatic pretrained compression

An arbitrary pretrained matrix is extremely unlikely to equal the output of one
random seed. Compressing an existing model requires one or more of:

- search over seeds or structured seed families;
- multiple seeded bases with learned coefficients;
- a sparse, low-rank, or quantized residual;
- retraining in the seeded parameterization.

## Random-feature ceiling

A fixed random operator can omit directions that the task needs. Increasing
adapter rank, adding multiple bases, or allowing selected dense layers may be
necessary. The correct comparison is quality at equal storage, throughput, and
training budget—not parameter count alone.

## Kernel maturity

The current Triton implementation is intentionally compact. It does not yet
provide:

- extensive autotuning across GPU generations;
- fused bias, gain, activation, or adapter operations;
- FP8 or integer output paths;
- split-K reductions;
- distributed tensor-parallel kernels;
- gradients with respect to the discrete seed.

## Shape and dtype constraints

The fused backend accepts contiguous rank-2 CUDA tensors in `float16` or
`bfloat16`. `SeedLinear` flattens arbitrary leading dimensions before calling
it. The reference backend supports broader CPU/GPU execution but materializes
the base matrix.

## Reproducibility hazards

Changing the PRNG algorithm while retaining the same seed silently changes the
model. Hash and distribution identifiers must be treated as checkpoint schema,
not as implementation details.

## Security

The generator is deterministic and non-cryptographic. Seeds must not be treated
as encryption keys, secrets, or a source of cryptographically secure randomness.
