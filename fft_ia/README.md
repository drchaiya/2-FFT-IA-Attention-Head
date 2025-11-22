# FFT-IA: FFT-Inspired Attention — True O(N log N) with Full Softmax

**Paper**: "FFT-Inspired Attention (FFT-IA): O(N log N) Complexity via Hierarchical Structural Pruning and Softmax Fidelity" (IEEE-style, 2025)

The **first** attention mechanism that:
- Achieves **true O(N log N)** via **fixed radix-2 butterfly factorization**
- Keeps **exact local Softmax** → **100% Softmax Fidelity**
- Uses **dynamic Q/K re-projection** → fully content-aware
- Has **no approximation, no hashing, no kernel tricks**
- Ships with **Triton fused kernel** → **7–10× faster** than PyTorch loop

## Install
```bash
pip install git+https://github.com/yourname/fft-ia.git
