import io
import torch

from seednet import SeedLinear


def test_shapes_adapter_gradients_and_zero_effect_initialization():
    layer = SeedLinear(13, 17, seed=7, rank=3, backend="reference")
    x = torch.randn(2, 5, 13, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 5, 17)

    base = SeedLinear(13, 17, seed=7, rank=0, backend="reference")
    with torch.no_grad():
        base.gain.copy_(layer.gain)
        base.bias.copy_(layer.bias)
    torch.testing.assert_close(y.detach(), base(x.detach()))

    y.square().mean().backward()
    assert x.grad is not None
    assert layer.adapter_A.grad is not None
    assert layer.adapter_B.grad is not None


def test_state_dict_round_trip_is_deterministic():
    layer = SeedLinear(8, 10, seed=1234, rank=2, backend="reference")
    x = torch.randn(3, 8)
    expected = layer(x)

    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)

    restored = SeedLinear(8, 10, seed=0, rank=2, backend="reference")
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x), expected)


def test_no_dense_base_parameter_is_stored():
    layer = SeedLinear(1024, 2048, seed=1, rank=4, backend="reference")
    names = dict(layer.named_parameters())
    assert "weight" not in names
    assert sum(p.numel() for p in layer.parameters()) < 1024 * 2048


def test_storage_report_rejects_nonpositive_element_size():
    layer = SeedLinear(8, 10, seed=1, backend="reference")
    import pytest
    with pytest.raises(ValueError, match="element_bytes"):
        layer.storage_report(element_bytes=0)
