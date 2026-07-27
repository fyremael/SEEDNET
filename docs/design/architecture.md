# Architecture

## Layer equation

For input \(X\in\mathbb{R}^{M\times K}\), a SeedNet linear layer computes

\[
Y = \left(XW_s^\top\right)\odot g
  + \frac{1}{r}(XA^\top)B^\top
  + b,
\]

where:

- \(W_s\in\mathbb{R}^{N\times K}\) is generated from seed \(s\);
- \(g\in\mathbb{R}^{N}\) is an optional trainable output gain;
- \(A\in\mathbb{R}^{r\times K}\) and \(B\in\mathbb{R}^{N\times r}\) form the
  low-rank correction;
- \(b\in\mathbb{R}^{N}\) is an optional bias.

The low-rank branch is initialized to zero effect by setting \(B=0\). The layer
therefore begins exactly at the seeded base operator.

## Procedural weight stream

The v1 stream assigns every matrix coordinate a deterministic counter:

```text
counter = uint32(output_index * in_features + input_index)
```

The counter is mixed with the low 32 bits of the seed by a versioned integer
hash. The upper 24 bits of the result are mapped to a centred uniform variate and
scaled to variance \(1/K\).

The same coordinate always produces the same value independently of tiling,
batch size, launch order, or device program identifier.

## Execution backends

### Reference backend

The PyTorch reference path materializes the full procedural matrix and calls a
standard matrix multiplication. It is intended for:

- CPU use;
- correctness tests;
- serialization tests;
- comparison against the fused kernel.

It does **not** provide the memory benefit of procedural generation.

### Triton backend

The fused kernel:

1. loads an activation tile;
2. computes the absolute coordinates of the required weight tile;
3. generates that tile in registers or on-chip storage;
4. applies `tl.dot`;
5. accumulates into the output tile.

No persistent base-weight tensor is loaded from global memory.

## Backward pass

The seed is discrete and frozen. The base branch therefore needs only the input
gradient:

\[
\nabla_X L = \nabla_Y L\,W_s.
\]

A separate transpose-oriented Triton kernel regenerates \(W_s\) with the same
coordinate rule. Reusing the forward orientation would be mathematically
incorrect.

The gain, bias, and adapter use ordinary PyTorch autograd operations and receive
standard trainable gradients.

## State inventory

A `SeedLinear` checkpoint stores:

| State | Shape | Trainable |
|---|---:|:---:|
| seed | scalar | no |
| gain | `out_features` | configurable |
| bias | `out_features` | optional |
| adapter A | `rank × in_features` | yes |
| adapter B | `out_features × rank` | yes |

The dense base matrix is absent from the state dictionary.

## Backend selection

`backend="auto"` uses Triton only when all of the following hold:

- CUDA is available;
- Triton imports successfully;
- the input is CUDA-resident;
- the input dtype is `float16` or `bfloat16`.

Otherwise the reference backend is selected. `backend="triton"` fails closed
when those requirements are not satisfied.
