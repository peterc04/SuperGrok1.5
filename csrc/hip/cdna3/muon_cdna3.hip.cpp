/*
 * Muon — CDNA3-Optimized (MI300X)
 *
 * BF16 Newton-Schulz GEMMs via MFMA instructions for large matrices (M >= 128).
 * CDNA3's BF16 MFMA provides ~2x throughput over FP32 for large matrices.
 * For small matrices (M < 128), BF16 conversion overhead exceeds MFMA benefit
 * and we fall back to the generic FP32 path.
 *
 * Element-wise kernels (momentum normalize, NS combine, update) use the
 * generic FP32 path as they are compute-bound on ALUs, not on matrix cores.
 */

#include <torch/extension.h>

// Forward declare generic launcher
void launch_muon_fused_step(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c);

void launch_muon_fused_step_cdna3(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c
) {
    auto M = param.dim() >= 2 ? param.size(0) : param.numel();

    if (M >= 128 && param.dim() >= 2) {
        // BF16 Newton-Schulz: convert momentum buffer to BF16 for MFMA GEMMs
        // 1. Momentum update: buf = momentum * buf + grad
        momentum_buffer.mul_(momentum).add_(grad);

        // 2. Normalize — device-side, no .item() CPU-GPU sync
        auto norm_tensor = momentum_buffer.norm();
        auto inv_norm_tensor = torch::where(
            norm_tensor > 1e-8f,
            torch::reciprocal(norm_tensor),
            torch::zeros_like(norm_tensor));

        // Convert to BF16 for NS iterations — MFMA BF16 gives ~2x throughput
        auto G = (momentum_buffer * inv_norm_tensor).to(torch::kBFloat16);
        int N_dim = param.size(1);
        auto G_2d = G.view({static_cast<int64_t>(M), N_dim});

        // 3. Newton-Schulz iterations with BF16 mm → MFMA instructions
        for (int i = 0; i < ns_steps; i++) {
            auto A_mat = torch::mm(G_2d.t(), G_2d);     // BF16 mm → MFMA
            auto AG = torch::mm(G_2d, A_mat);            // BF16 mm → MFMA
            auto AAG = torch::mm(AG, A_mat);             // BF16 mm → MFMA
            G_2d = a * G_2d + b * AG + c * AAG;
        }

        // 4. Update param in FP32
        auto orth = G_2d.to(torch::kFloat32).view_as(param);
        param.mul_(1.0f - lr * weight_decay);
        param.add_(orth, -lr);
    } else {
        // Small matrix — BF16 conversion overhead exceeds MFMA benefit.
        // Falls back to generic FP32 path.
        launch_muon_fused_step(param, momentum_buffer, grad,
                               lr, momentum, weight_decay, ns_steps, a, b, c);
    }
}
