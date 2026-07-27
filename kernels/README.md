# Kernel entry point

`fused_seed_gemm.py` is the executable Triton source.

The `.cu` filename from the prototype is retained only as a correction notice:
Triton JIT kernels are authored and imported as Python. No `nvcc` build step is
required for this kernel.
