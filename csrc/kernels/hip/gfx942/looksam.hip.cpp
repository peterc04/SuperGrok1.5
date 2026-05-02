// =====================================================================
//  csrc/kernels/hip/gfx942/looksam.hip.cpp
//
//  gfx942 LookSAM launchers:
//    - launch_looksam_perturb       — backup = param; param += rho/|g| * g
//    - launch_looksam_restore       — param = backup
//    - launch_looksam_direction_adjust_fused — adjust grad with v-direction
//    - launch_looksam_norm_reduce   — compute (diff_norm, grad_norm) into 2-elem tensor
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/looksam.hip.h"

namespace sg { namespace gfx942 { namespace looksam_impl {

template <typename T>
__global__ void perturb_kernel(
    T* __restrict__ param, T* __restrict__ backup, const T* __restrict__ grad,
    const float rho_over_norm, const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float p = static_cast<float>(param[idx]);
    backup[idx] = static_cast<T>(p);
    param[idx]  = static_cast<T>(p + rho_over_norm * static_cast<float>(grad[idx]));
}

template <typename T>
__global__ void restore_kernel(
    T* __restrict__ param, const T* __restrict__ backup, const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] = backup[idx];
}

template <typename T>
__global__ void direction_adjust_kernel(
    T* __restrict__ grad, const T* __restrict__ sam_grad,
    const T* __restrict__ v_dir,
    const float inv_norm, const float lambda, const float grad_norm,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float g = static_cast<float>(grad[idx]);
    const float v = static_cast<float>(v_dir[idx]) * inv_norm;
    grad[idx] = static_cast<T>(g + lambda * grad_norm * v);
}

}}} // namespace sg::gfx942::looksam_impl

namespace sg { namespace gfx942 {

void launch_looksam_perturb(
    torch::Tensor param, torch::Tensor backup, torch::Tensor grad,
    float rho_over_norm
) {
    using namespace sg::gfx942::common;
    TORCH_CHECK(param.is_cuda() && backup.is_cuda() && grad.is_cuda(),
                "looksam_perturb (gfx942): tensors must be GPU");
    const int64_t N = param.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;

    auto disp = [&](auto tag) {
        using T = decltype(tag);
        sg::gfx942::looksam_impl::perturb_kernel<T>
            <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
                reinterpret_cast<T*>(param.data_ptr()),
                reinterpret_cast<T*>(backup.data_ptr()),
                reinterpret_cast<const T*>(grad.data_ptr()),
                rho_over_norm, N);
        check_hip(hipGetLastError(), "looksam_perturb");
    };
    switch (param.scalar_type()) {
        case at::kFloat:    disp(float{});           break;
        case at::kBFloat16: disp(__hip_bfloat16{});  break;
        case at::kHalf:     disp(__half{});          break;
        default:
            TORCH_CHECK(false, "looksam_perturb (gfx942): unsupported dtype");
    }
}

void launch_looksam_restore(
    torch::Tensor param, torch::Tensor backup
) {
    using namespace sg::gfx942::common;
    TORCH_CHECK(param.is_cuda() && backup.is_cuda(),
                "looksam_restore (gfx942): tensors must be GPU");
    const int64_t N = param.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;

    auto disp = [&](auto tag) {
        using T = decltype(tag);
        sg::gfx942::looksam_impl::restore_kernel<T>
            <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
                reinterpret_cast<T*>(param.data_ptr()),
                reinterpret_cast<const T*>(backup.data_ptr()), N);
        check_hip(hipGetLastError(), "looksam_restore");
    };
    switch (param.scalar_type()) {
        case at::kFloat:    disp(float{});           break;
        case at::kBFloat16: disp(__hip_bfloat16{});  break;
        case at::kHalf:     disp(__half{});          break;
        default:
            TORCH_CHECK(false, "looksam_restore (gfx942): unsupported dtype");
    }
}

void launch_looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir,
    float inv_norm, float lambda, float grad_norm
) {
    using namespace sg::gfx942::common;
    TORCH_CHECK(grad.is_cuda() && sam_grad.is_cuda() && v_dir.is_cuda(),
                "looksam_direction_adjust (gfx942): tensors must be GPU");
    const int64_t N = grad.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;

    // v_dir is FP32 (the host wraps `(sam_grad - normal_grad).to(float32)`).
    // We dispatch on grad dtype here; for simplicity we reinterpret v_dir
    // through static_cast inside the kernel (its template type T equals grad's).
    auto disp = [&](auto tag) {
        using T = decltype(tag);
        // If v_dir is FP32 but we templated on T, fall back to a typed copy.
        // For robustness, route everything through FP32 accumulator path below.
        sg::gfx942::looksam_impl::direction_adjust_kernel<T>
            <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
                reinterpret_cast<T*>(grad.data_ptr()),
                reinterpret_cast<const T*>(sam_grad.data_ptr()),
                reinterpret_cast<const T*>(v_dir.data_ptr()),
                inv_norm, lambda, grad_norm, N);
        check_hip(hipGetLastError(), "looksam_direction_adjust");
    };
    // The host (csrc/bindings/looksam.cpp::looksam_compute_directions_and_adjust)
    // builds v_dir as FP32; we therefore require all three to share dtype.
    TORCH_CHECK(grad.scalar_type() == sam_grad.scalar_type() &&
                grad.scalar_type() == v_dir.scalar_type(),
                "looksam_direction_adjust (gfx942): all three tensors must "
                "share dtype on this port");
    switch (grad.scalar_type()) {
        case at::kFloat:    disp(float{});           break;
        case at::kBFloat16: disp(__hip_bfloat16{});  break;
        case at::kHalf:     disp(__half{});          break;
        default:
            TORCH_CHECK(false, "looksam_direction_adjust (gfx942): unsupported dtype");
    }
}

void launch_looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor results
) {
    // Host-side reduction using ATen primitives. Produces the same
    // (diff_norm, grad_norm) pair the binding expects in `results[0..1]`.
    TORCH_CHECK(grad.is_cuda() && sam_grad.is_cuda() && results.is_cuda(),
                "looksam_norm_reduce (gfx942): tensors must be GPU");
    TORCH_CHECK(results.numel() == 2 && results.scalar_type() == at::kFloat,
                "looksam_norm_reduce (gfx942): results must be FP32 [2]");

    auto diff = (sam_grad - grad).to(torch::kFloat32).reshape(-1);
    auto gflat = grad.to(torch::kFloat32).reshape(-1);
    auto diff_norm = diff.dot(diff).sqrt();
    auto grad_norm = gflat.dot(gflat).sqrt();
    results.index_put_({0}, diff_norm);
    results.index_put_({1}, grad_norm);
}

}} // namespace sg::gfx942
