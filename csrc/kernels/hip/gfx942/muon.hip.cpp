// =====================================================================
//  csrc/kernels/hip/gfx942/muon.hip.cpp
//
//  gfx942 Muon per-element kernels. The matmuls (X X^T, A X, A^2 X) are
//  done host-side via torch::mm in csrc/bindings/muon.cpp; this TU has
//  the elementwise pieces only.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/muon.hip.h"

namespace sg { namespace gfx942 { namespace muon_impl {

template <typename T>
__global__ void momentum_normalize_kernel(
    T* __restrict__ buf, T* __restrict__ X, const T* __restrict__ grad,
    const float momentum, const float inv_norm,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float b = static_cast<float>(buf[idx]);
    const float g = static_cast<float>(grad[idx]);
    const float new_b = momentum * b + g;
    buf[idx] = static_cast<T>(new_b);
    X[idx]   = static_cast<T>(new_b * inv_norm);
}

template <typename T>
__global__ void ns_combine_kernel(
    T* __restrict__ X_out,
    const T* __restrict__ X,
    const T* __restrict__ AX,
    const T* __restrict__ AAX,
    const float a, const float b, const float c,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float x   = static_cast<float>(X[idx]);
    const float ax  = static_cast<float>(AX[idx]);
    const float aax = static_cast<float>(AAX[idx]);
    X_out[idx] = static_cast<T>(a * x + b * ax + c * aax);
}

template <typename PT, typename OT>
__global__ void update_kernel(
    PT* __restrict__ param, const OT* __restrict__ orth,
    const float neg_lr_scale, const float decay_factor,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float p = static_cast<float>(param[idx]) * decay_factor;
    const float u = neg_lr_scale * static_cast<float>(orth[idx]);
    param[idx] = static_cast<PT>(p + u);
}

template <typename T>
__global__ void ns_combine_update_fused_kernel(
    T* __restrict__ param,
    const T* __restrict__ X,
    const T* __restrict__ AX,
    const T* __restrict__ AAX,
    const float a, const float b, const float c,
    const float neg_lr_scale, const float decay_factor,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float orth = a * static_cast<float>(X[idx])
                     + b * static_cast<float>(AX[idx])
                     + c * static_cast<float>(AAX[idx]);
    const float p = static_cast<float>(param[idx]) * decay_factor;
    param[idx] = static_cast<T>(p + neg_lr_scale * orth);
}

}}} // namespace sg::gfx942::muon_impl

namespace sg { namespace gfx942 {

#define MUON_DISPATCH3(VAR, BODY)                                              \
    do {                                                                       \
        switch (VAR.scalar_type()) {                                           \
            case at::kFloat:    { using T = float;            BODY; break; }   \
            case at::kBFloat16: { using T = __hip_bfloat16;   BODY; break; }   \
            case at::kHalf:     { using T = __half;           BODY; break; }   \
            default: TORCH_CHECK(false, "muon (gfx942): unsupported dtype");   \
        }                                                                      \
    } while (0)

void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
    float momentum, float inv_norm
) {
    using namespace sg::gfx942::common;
    const int64_t N = buf.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;

    MUON_DISPATCH3(buf, (
        sg::gfx942::muon_impl::momentum_normalize_kernel<T>
            <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
                reinterpret_cast<T*>(buf.data_ptr()),
                reinterpret_cast<T*>(X.data_ptr()),
                reinterpret_cast<const T*>(grad.data_ptr()),
                momentum, inv_norm, N)
    ));
    check_hip(hipGetLastError(), "muon_momentum_normalize");
}

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c
) {
    using namespace sg::gfx942::common;
    const int64_t N = X.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;

    MUON_DISPATCH3(X, (
        sg::gfx942::muon_impl::ns_combine_kernel<T>
            <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
                reinterpret_cast<T*>(X_out.data_ptr()),
                reinterpret_cast<const T*>(X.data_ptr()),
                reinterpret_cast<const T*>(AX.data_ptr()),
                reinterpret_cast<const T*>(AAX.data_ptr()),
                a, b, c, N)
    ));
    check_hip(hipGetLastError(), "muon_ns_combine");
}

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor
) {
    using namespace sg::gfx942::common;
    const int64_t N = param.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;

    // We assume orth has been .to(param.dtype()) by the host (binding does so).
    TORCH_CHECK(param.scalar_type() == orth.scalar_type(),
                "muon_update (gfx942): param and orth dtypes must match");
    MUON_DISPATCH3(param, (
        sg::gfx942::muon_impl::update_kernel<T, T>
            <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
                reinterpret_cast<T*>(param.data_ptr()),
                reinterpret_cast<const T*>(orth.data_ptr()),
                neg_lr_scale, decay_factor, N)
    ));
    check_hip(hipGetLastError(), "muon_update");
}

void launch_muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c,
    float neg_lr_scale, float decay_factor
) {
    using namespace sg::gfx942::common;
    const int64_t N = param.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;

    // X / AX / AAX must share dtype with param for this fused launcher.
    TORCH_CHECK(param.scalar_type() == X.scalar_type() &&
                param.scalar_type() == AX.scalar_type() &&
                param.scalar_type() == AAX.scalar_type(),
                "muon_ns_combine_update_fused (gfx942): all tensors must share dtype");
    MUON_DISPATCH3(param, (
        sg::gfx942::muon_impl::ns_combine_update_fused_kernel<T>
            <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
                reinterpret_cast<T*>(param.data_ptr()),
                reinterpret_cast<const T*>(X.data_ptr()),
                reinterpret_cast<const T*>(AX.data_ptr()),
                reinterpret_cast<const T*>(AAX.data_ptr()),
                a, b, c, neg_lr_scale, decay_factor, N)
    ));
    check_hip(hipGetLastError(), "muon_ns_combine_update_fused");
}

#undef MUON_DISPATCH3

}} // namespace sg::gfx942
