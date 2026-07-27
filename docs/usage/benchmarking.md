# Benchmarking

A SeedNet benchmark must measure both systems efficiency and task quality.

## Kernel benchmark

```bash
python scripts/benchmark.py --m 512 --k 4096 --n 4096 --seed 1234
```

The script compares:

- fused procedural GEMM;
- dense GEMM using an explicitly materialized matching matrix;
- numerical agreement;
- persistent dense weight size.

## Metrics that matter

Record at least:

| Category | Metrics |
|---|---|
| Correctness | max/mean absolute error, forward and backward agreement |
| Latency | warm median, p95, compile time reported separately |
| Throughput | samples/s or effective TFLOP/s |
| Memory | peak allocated, peak reserved, persistent checkpoint bytes |
| Traffic | HBM read/write bytes from Nsight Compute where available |
| Quality | loss, perplexity, accuracy, or task-specific score |

## Fair comparison protocol

1. Exclude first-call Triton compilation from steady-state latency.
2. Use the same dtype and accumulation policy.
3. Generate the dense baseline from the identical seed stream.
4. Synchronize CUDA around timing regions.
5. Sweep realistic `(M, N, K)` shapes rather than one favourable case.
6. Report adapter operations and storage when benchmarking `SeedLinear`.
7. Compare end-to-end model latency, not only isolated kernel latency.

## Expected regimes

The fused approach is most plausible when:

- the dense base matrix is large;
- arithmetic intensity is otherwise limited by weight traffic;
- the batch or token dimension is not so large that dense GEMM fully amortizes
  weight loads;
- integer generation can be overlapped with tensor-core work.

For small matrices or highly compute-bound workloads, procedural generation may
be slower despite lower persistent storage.
