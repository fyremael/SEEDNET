import pytest
import torch

from seednet import materialize_seed_weight, triton_available
from seednet.functional import seed_gemm


@pytest.mark.skipif(not triton_available(), reason="CUDA Triton environment unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_forward_and_backward(dtype):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    x = torch.randn(65, 70, device="cuda", dtype=dtype, requires_grad=True)
    seed = 77
    y = seed_gemm(x, 73, seed, backend="triton")
    w = materialize_seed_weight(73, 70, seed, device="cuda", dtype=dtype)
    expected = x @ w.t()
    torch.testing.assert_close(y, expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(y)
    y.backward(grad)
    expected_grad = grad @ w
    torch.testing.assert_close(x.grad, expected_grad, rtol=2e-2, atol=2e-2)
