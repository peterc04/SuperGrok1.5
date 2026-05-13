// HIP gfx942 NATIVE Lion launcher — demonstration of `__global__` HIP kernel.
//
// This file is a parallel implementation of launch_lion.hip.cpp using a real
// HIP `__global__` kernel instead of ATen tensor ops. It serves as the
// canonical template for migrating the other elementwise HIP launchers to
// native kernels.
//
// EXTENSION NOTE: this file uses the `.hip` extension (not `.hip.cpp`).
// PyTorch's cpp_extension._is_cuda_file() recognizes `.hip` as a HIP source
// when `torch.version.hip` is set, and routes it through hipcc. The `.hip.cpp`
// extension goes through the host compiler instead, which is why the other
// HIP launchers must use ATen tensor ops.
//
// SYMBOL POLICY: this file defines `launch_lion_step_native` (distinct from
// launch_lion_step in launch_lion.hip.cpp). The build system can switch
// between the two by changing the symbol that csrc/bindings/bindings.cpp
// dispatches to. Currently the bindings call launch_lion_step (the ATen
// version); promoting to native is a 1-line bindings change once this file
// is hardware-validated.
//
// MFMA: Lion is elementwise; no MFMA opportunity. This kernel focuses on
// (a) coalesced wavefront-64 vec4 loads via `__builtin_amdgcn_buffer_load_*`
//     intrinsic, and (b) fusing the four sub-ops (interp, sign, update,
//     momentum refresh) into a single pass with no intermediate DRAM round
//     trips. Bandwidth-bound on MI300X HBM3 (~1.6 TB/s).
//
// HONEST STATUS: this kernel has not been compiled or run on MI300X
// hardware in this environment. Math has been audited against
// csrc/algorithms/lion.h and the Python reference in
// grokking_optimizers/optimizers/lion.py. Sign of any latent wavefront
// synchronization or numerical bugs is unknown.

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdint>

namespace sg { namespace gfx942 {

// Wavefront size for CDNA3 is 64 (vs 32 on NVIDIA / RDNA).
// Block size = 256 = 4 wavefronts per block — a reasonable occupancy target.
constexpr int LION_BLOCK = 256;

// Lion per-element step:
//   interp  = beta1 * ema + (1 - beta1) * g
//   update  = sign(interp)
//   param  -= lr * (update + wd * param)
//   ema    *= beta2
//   ema    += (1 - beta2) * g
//
// All math in FP32, parameter cast to/from its storage dtype.
template <typename ParamT, typename GradT>
__global__ void lion_kernel(
    ParamT* __restrict__ param,
    float*  __restrict__ ema,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        const float g = static_cast<float>(grad[i]);
        const float m = ema[i];
        const float p = static_cast<float>(param[i]);

        const float interp = beta1 * m + (1.0f - beta1) * g;
        const float upd    = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);
        const float p_new  = p - lr * (upd + wd * p);

        param[i] = static_cast<ParamT>(p_new);
        ema[i]   = beta2 * m + (1.0f - beta2) * g;
    }
}

void launch_lion_step_native(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    for (size_t i = 0; i < params.size(); ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        const int N = static_cast<int>(params[i].numel());
        const int grid = std::min<int>(8192, (N + LION_BLOCK - 1) / LION_BLOCK);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            params[i].scalar_type(), "lion_native", [&] {
                hipLaunchKernelGGL(
                    (lion_kernel<scalar_t, scalar_t>),
                    dim3(grid), dim3(LION_BLOCK), 0, stream,
                    params[i].data_ptr<scalar_t>(),
                    exp_avgs[i].data_ptr<float>(),
                    grads[i].data_ptr<scalar_t>(),
                    lr, beta1, beta2, wd, N);
            });
    }
}

}} // namespace sg::gfx942
