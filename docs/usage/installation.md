# Installation

## CPU/reference development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

This installs PyTorch and the test dependencies. The reference backend
materializes procedural matrices and works without Triton.

## CUDA and Triton

On Linux with a CUDA-capable PyTorch installation:

```bash
pip install -e ".[cuda,dev]"
python -c "from seednet import triton_available; print(triton_available())"
```

A result of `True` means `backend="auto"` may select the fused path for
compatible tensors.

!!! note
    Install the PyTorch build appropriate for the local CUDA driver before
    diagnosing Triton. PyTorch installation commands vary by platform and CUDA
    runtime.

## Documentation development

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open the local address printed by MkDocs. A strict production build is:

```bash
mkdocs build --strict
```

## Full demonstration environment

```bash
pip install -e ".[cuda,demo,dev,docs]"
jupyter notebook notebooks/seednet_demo.ipynb
```
