# FFT-Inspired Attention (\text{FFT}-\text{IA}): The O(N \log N) Transformer
Achieving Sub-Quadratic Complexity with Full Softmax Fidelity

FFT-Inspired Attention (FFT-IA)
\mathbf{O(N \log N)} Complexity via Hierarchical Structural Pruning and Softmax Fidelity
The Fast Fourier Transform-Inspired Attention (\mathbf{FFT}-\mathbf{IA}) framework is a novel, structurally enforced methodology designed to overcome the quadratic time complexity barrier of the Multi-Head Self-Attention (MHSA) mechanism in Transformers.
By drawing inspiration from the algorithmic factorization of the Dense Discrete Fourier Transform (DFT) into the Fast Fourier Transform (FFT) via the Cooley-Tukey algorithm, FFT-IA achieves a guaranteed \mathbf{O(N \log N)} asymptotic complexity in sequence length N.
💡 Core Innovation: Fixed Structural Factorization
The primary limitation of standard MHSA is its dense \mathbf{O(N^2)} interaction matrix. FFT-IA replaces this single, global operation with a cascade of sparse, local operations, structurally forcing efficiency without relying on mathematical approximations.
1. The Hierarchical Factorization Principle
The \mathbf{O(N^2)} attention computation is replaced by a product of \mathbf{L=\log_2 N} sequential sparse projection factors (P_1, \dots, P_L).
• Structure: The factorization enforces a fixed, radix-2 butterfly connection pattern at each sequential stage.
• Component: Each stage utilizes a Butterfly-Attention Block which performs a local, \mathbf{O(N)} operation, ensuring the total complexity remains \mathbf{O(N \log N)} globally.
2. Softmax Fidelity (Non-Approximate Efficiency)
A key distinction of FFT-IA is its commitment to high-fidelity attention scoring. Unlike many approximation methods that substitute or remove the Softmax non-linearity, FFT-IA retains it:
• Local Application: The mechanism computes exact attention scores and applies the Softmax non-linearity locally over the two connected tokens within the butterfly constraint set \mathcal{C}_j.
• Adaptive Pooling: This local Softmax acts as an adaptive normalized pooling step, with the global normalization being achieved compositionally through the cascade of \log_2 N stages.
3. Dynamic Re-projection
To ensure the attention remains content-dependent despite the fixed structural graph, Query (Q) and Key (K) vectors are dynamically re-projected from the intermediate state (V_{i-1}) at every sequential stage. This mechanism maintains the essential dynamism of the attention mechanism while strictly enforcing the fixed sparsity pattern.
🎯 Significance and Structural Inductive Bias
The fixed, structurally pruned connectivity graph acts as a powerful structural inductive bias. This is hypothesized to enhance model robustness and generalization by:
• Forcing Compositional Flow: Long-range dependencies must be established through a cascade of \log_2 N weighted aggregations, promoting deeper, compositional processing over simple associative memory lookups.
• Mitigating Spurious Correlations: The structural pruning limits the model's capacity to overfit to unnecessary or spurious global relationships.
⚙️ Paramount Technical Challenge
While the theoretical complexity is \mathbf{O(N \log N)}, the total computational cost is dominated by the repeated Q/K re-projection, resulting in a total cost of \mathbf{O(N d^2 \log N)}.
The realization of practical, wall-clock speedup is strictly contingent upon Kernel Fusion: the integration of the L=\log_2 N sequential, irregular operations into a single, highly optimized custom kernel for modern GPU architectures. This optimization is the necessary condition to eliminate the overhead of numerous sequential kernel launches.

Contribution
We welcome contributions from researchers and engineers specializing in model architecture, low-level GPU programming, and efficient deep learning systems. Please check the POC_SPEC.md for detailed requirements or open an Issue to discuss your potential contribution.
