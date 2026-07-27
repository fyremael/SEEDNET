import torch

from seednet import materialize_seed_weight
from seednet.functional import seed_gemm


def test_determinism_and_seed_separation():
    a = materialize_seed_weight(7, 11, 123)
    b = materialize_seed_weight(7, 11, 123)
    c = materialize_seed_weight(7, 11, 124)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_reference_matches_explicit_dense():
    x = torch.randn(5, 11)
    w = materialize_seed_weight(7, 11, 91)
    actual = seed_gemm(x, 7, 91, backend="reference")
    expected = x @ w.t()
    torch.testing.assert_close(actual, expected)


def test_reference_input_gradient():
    x = torch.randn(4, 9, requires_grad=True)
    w = materialize_seed_weight(6, 9, 12)
    y = seed_gemm(x, 6, 12, backend="reference")
    y.square().sum().backward()
    expected = 2 * (x.detach() @ w.t()) @ w
    torch.testing.assert_close(x.grad, expected)
