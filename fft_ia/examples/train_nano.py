import torch
from fft_ia.core import FFTInspiredAttention

seq_len = 1024
x = torch.randn(2, seq_len, 512)

layer = FFTInspiredAttention(dim=512, heads=8)
out = layer(x)

print(f"Input: {x.shape} → Output: {out.shape}")
print("FFT-IA forward pass successful!")
assert out.shape == x.shape
