/*
Compatibility notice
====================

The original prototype used this filename, but a Triton kernel is Python source,
not a CUDA translation unit. It cannot be imported normally from a `.cu` file.

Use:

    kernels/fused_seed_gemm.py

or the installed API:

    from seednet.functional import seed_gemm

This file is retained only to make the correction explicit.
*/
