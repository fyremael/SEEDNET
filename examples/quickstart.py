import torch
from seednet import SeedLinear

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

layer = SeedLinear(256, 512, seed=0xBEEF, rank=8).to(device=device, dtype=dtype)
x = torch.randn(16, 256, device=device, dtype=dtype, requires_grad=True)
y = layer(x)
loss = y.float().square().mean()
loss.backward()

print("device:", device)
print("output:", tuple(y.shape))
print("input gradient:", tuple(x.grad.shape))
print("storage:", layer.storage_report())
