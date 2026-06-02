#ifndef GROKKING_KERNELS_SM90_PRODIGY_SM90_CUH_
#define GROKKING_KERNELS_SM90_PRODIGY_SM90_CUH_
// ============================================================================
// prodigy_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'prodigy'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_prodigy.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for Prodigy.
// Algorithm: csrc/algorithms/prodigy.h
//
// Three-kernel orchestration:
//   (1) reduce  — block-reduce (r, s) partial sums
//   (2) update  — single-thread d update
//   (3) apply   — Adam with d as effective learning rate
// All on-device; d_t never round-trips through CPU.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/prodigy.h"
// ── Autotuner-consumable launch parameters (inlined; see compile.py) ──
#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif
#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif
#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif
#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif
// §3.3 Autotuner-pickable launch cluster shape. compile.py emits the tuned
// cluster shape as three nvcc-safe scalar macros SG_TUNED_CLUSTER_SHAPE_{0,1,2}
// plus SG_TUNED_CLUSTER_SHAPE_VOLUME (a single `-Dfoo=2,1,1` is rejected by
// nvcc — the driver splits the value on commas). Default 1,1,1 = no cluster,
// so the DSMEM helper degrades to a plain block reduce.
#ifndef SG_TUNED_CLUSTER_SHAPE_0
#define SG_TUNED_CLUSTER_SHAPE_0 1
#endif
#ifndef SG_TUNED_CLUSTER_SHAPE_1
#define SG_TUNED_CLUSTER_SHAPE_1 1
#endif
#ifndef SG_TUNED_CLUSTER_SHAPE_2
#define SG_TUNED_CLUSTER_SHAPE_2 1
#endif
#ifndef SG_TUNED_CLUSTER_SHAPE_VOLUME
#define SG_TUNED_CLUSTER_SHAPE_VOLUME \
    (SG_TUNED_CLUSTER_SHAPE_0 * SG_TUNED_CLUSTER_SHAPE_1 * SG_TUNED_CLUSTER_SHAPE_2)
#endif
#include "csrc/backends/cuda/sm_90/primitives.cuh"
#include "grokking_optimizers/kernels/sm_90/common_sm90.cuh"

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::prodigy_partials_step;
using ::sg::algorithms::prodigy_update_d;
using ::sg::algorithms::prodigy_apply_step;

#ifndef SG_PRODIGY_MIN_BLOCKS
#define SG_PRODIGY_MIN_BLOCKS 4
#endif

// Compile-time cluster volume from the tuned shape, capped at the Hopper
// portable max (8). Used both for the host-side fits-in-one-cluster decision
// and to bound the DSMEM gather. (The __cluster_dims__ attribute below takes
// the three scalar components directly.)
namespace prodigy_detail {
constexpr int kRawClusterVolume = SG_TUNED_CLUSTER_SHAPE_VOLUME;
constexpr int kClusterVolume =
    (kRawClusterVolume > 8) ? 8 : kRawClusterVolume;
}  // namespace prodigy_detail

template <typename ParamT, typename GradT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_PRODIGY_MIN_BLOCKS)
prodigy_reduce_kernel(
    const ParamT* param, const ParamT* param_init, const GradT* grad,
    float d_prev,
    float* r_partial, float* s_partial,
    int64_t N
) {
    float r_local = 0.0f, s_local = 0.0f;
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        SG_SANITIZE_GRAD_INPLACE(grad, i);
        prodigy_partials_step(param, param_init, grad, d_prev, i,
                              r_local, s_local);
    }
    // Single fused two-value block reduction (one shared tile + one barrier
    // pair) → ONE atomicAdd per partial per block, instead of two separate
    // block reductions.
    float r_block, s_block;
    prim::block_reduce_sum2_f32(r_local, s_local, r_block, s_block);
    if (threadIdx.x == 0) {
        atomicAdd(r_partial, r_block);
        atomicAdd(s_partial, s_block);
    }
}

// Fused reduce + d-update for the SINGLE-tensor path: the LAST block to finish
// (last_block_finished + the gridDim counter) reads the fully-accumulated
// (r, s) partials and writes d_t directly — folding away the separate
// <<<1,1>>> prodigy_update_d_kernel launch. block_done_counter must be a
// device scalar zeroed before launch. The multi-tensor path keeps the separate
// d-update kernel (it accumulates across many launches before the update).
template <typename ParamT, typename GradT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_PRODIGY_MIN_BLOCKS)
prodigy_reduce_and_update_d_kernel(
    const ParamT* param, const ParamT* param_init, const GradT* grad,
    float d_prev,
    float* r_partial, float* s_partial, float* d_t,
    unsigned* block_done_counter,
    int64_t N, int total_blocks
) {
    float r_local = 0.0f, s_local = 0.0f;
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        SG_SANITIZE_GRAD_INPLACE(grad, i);
        prodigy_partials_step(param, param_init, grad, d_prev, i,
                              r_local, s_local);
    }
    float r_block, s_block;
    prim::block_reduce_sum2_f32(r_local, s_local, r_block, s_block);
    if (threadIdx.x == 0) {
        atomicAdd(r_partial, r_block);
        atomicAdd(s_partial, s_block);
    }
    // __threadfence so the last block sees the completed atomic results.
    __threadfence();
    if (prim::last_block_finished(block_done_counter, total_blocks)) {
        if (threadIdx.x == 0) {
            *d_t = prodigy_update_d(d_prev, *r_partial, *s_partial);
        }
    }
}

// §3.2 DSMEM cluster variant of the (r, s) partial reduce. Launched with a
// thread-block cluster (__cluster_dims__) so the whole reduction fits in ONE
// cluster; the cross-block sum is done via DSMEM (map_shared_rank + cl.sync)
// instead of a global atomicAdd. Only correct when the launch grid == the
// cluster volume (one cluster covers every block); the host launcher enforces
// that before selecting this kernel. Block 0 writes the final sum directly.
//
// Gated behind ENABLE_DSMEM_REDUCE at the call site; when the toggle is off,
// this kernel is never launched (the atomic kernel above is used) — so the
// cluster machinery adds NO cost on the default path.
#if defined(ENABLE_DSMEM_REDUCE) && (ENABLE_DSMEM_REDUCE != 0)
template <typename ParamT, typename GradT>
__global__ void __cluster_dims__(
    SG_TUNED_CLUSTER_SHAPE_0, SG_TUNED_CLUSTER_SHAPE_1, SG_TUNED_CLUSTER_SHAPE_2)
prodigy_reduce_dsmem_kernel(
    const ParamT* param, const ParamT* param_init, const GradT* grad,
    float d_prev,
    float* r_partial, float* s_partial,
    int64_t N
) {
    // One shared slot per reduced quantity for this block's published partial.
    __shared__ float r_slot;
    __shared__ float s_slot;

    float r_local = 0.0f, s_local = 0.0f;
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        SG_SANITIZE_GRAD_INPLACE(grad, i);
        prodigy_partials_step(param, param_init, grad, d_prev, i,
                              r_local, s_local);
    }
    // Full thread->warp->block->cluster tree; every block gets the same sum.
    float r_sum = prim::cluster_reduce_sum_f32_dsmem(r_local, &r_slot);
    float s_sum = prim::cluster_reduce_sum_f32_dsmem(s_local, &s_slot);

    // One block (rank 0, thread 0) publishes the final result — no atomic.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *r_partial = r_sum;
        *s_partial = s_sum;
    }
}
#endif  // ENABLE_DSMEM_REDUCE

__global__ void prodigy_update_d_kernel(
    float* d_t, const float* r_sum, const float* s_sum, float d_prev
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *d_t = prodigy_update_d(d_prev, *r_sum, *s_sum);
    }
}

template <typename ParamT, typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_PRODIGY_MIN_BLOCKS)
prodigy_apply_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* s_track,
    const GradT* grad, const float* d_ptr,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N
) {
    const float d = *d_ptr;
    const int64_t stride = prim::grid_stride() * UNROLL;
    const int64_t base0 = prim::grid_stride_index() * UNROLL;
    for (int64_t base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t i = base + u;
            if (i < N) {
                SG_SANITIZE_GRAD_INPLACE(grad, i);
                prodigy_apply_step(param, exp_avg, exp_avg_sq, s_track, grad,
                                   d, beta1, beta2, eps, wd, bc1, bc2, i);
            }
        }
    }
}

// FP32 vec4 apply: float4 traffic, canonical prodigy_apply_step CALLED 4×.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_PRODIGY_MIN_BLOCKS)
prodigy_apply_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, float4* exp_avg_sq4, float4* s_track4,
    const float4* grad4, const float* d_ptr,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N4
) {
    const float d = *d_ptr;
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N4; i += stride) {
        float4 p  = prim::ld_f32v4(param4 + i);
        float4 m  = prim::ld_f32v4(exp_avg4 + i);
        float4 v  = prim::ld_f32v4(exp_avg_sq4 + i);
        float4 st = prim::ld_f32v4(s_track4 + i);
        float4 g  = prim::ldg_f32v4(grad4 + i);
        ::grokking::sm90::sg_sanitize_grad4(g);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            prodigy_apply_step(&p.x, &m.x, &v.x, &st.x, &g.x,
                               d, beta1, beta2, eps, wd, bc1, bc2, u);
        }
        prim::st_f32v4(param4 + i, p);
        prim::st_f32v4(exp_avg4 + i, m);
        prim::st_f32v4(exp_avg_sq4 + i, v);
        prim::st_f32v4(s_track4 + i, st);
    }
}

void launch_prodigy_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& s_track,
    const torch::Tensor& param_init,
    const torch::Tensor& grad,
    torch::Tensor& d_t,
    torch::Tensor& r_partial,
    torch::Tensor& s_partial,
    float d_prev,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // §6.1: keep the Adam moments (m, v) L2-resident across the step.
    prim::L2PersistScope l2(stream,
        exp_avg.data_ptr(), exp_avg.nbytes(),
        exp_avg_sq.data_ptr(), exp_avg_sq.nbytes());

    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    r_partial.zero_();
    s_partial.zero_();

#if defined(ENABLE_DSMEM_REDUCE) && (ENABLE_DSMEM_REDUCE != 0)
    // §3.2 DSMEM cross-CTA path: only valid when the ENTIRE reduction grid fits
    // in ONE thread-block cluster (one cluster covers every block, so the
    // cross-block sum can ride DSMEM instead of a global atomic). When the grid
    // is larger than the cluster volume, fall back to the atomic kernel below.
    constexpr int kClusterVol = prodigy_detail::kClusterVolume;
    const bool dsmem_fits = (kClusterVol > 1) && (grid <= kClusterVol);
    if (dsmem_fits) {
        // Launch exactly kClusterVol blocks = one full cluster; the
        // __cluster_dims__ attribute supplies the cluster shape. The grid-
        // stride loop in the kernel still covers all N elements.
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            param.scalar_type(), "prodigy_reduce_dsmem", [&] {
                prodigy_reduce_dsmem_kernel<scalar_t, scalar_t>
                    <<<kClusterVol, block, 0, stream>>>(
                    param.data_ptr<scalar_t>(),
                    param_init.data_ptr<scalar_t>(),
                    grad.data_ptr<scalar_t>(),
                    d_prev,
                    r_partial.data_ptr<float>(),
                    s_partial.data_ptr<float>(),
                    N);
                SG_LAUNCH_CHECK(stream);
            });
        // DSMEM path wrote the final (r, s) directly; do the d-update on one
        // thread (the cluster kernel cannot use the last-block fold).
        prodigy_update_d_kernel<<<1, 1, 0, stream>>>(
            d_t.data_ptr<float>(),
            r_partial.data_ptr<float>(),
            s_partial.data_ptr<float>(),
            d_prev);
        SG_LAUNCH_CHECK(stream);
    } else
#endif  // ENABLE_DSMEM_REDUCE
    {
        // Atomic reduce + last-block d-update fold: the reduce kernel itself
        // writes d_t from the completed (r, s) partials — no separate <<<1,1>>>.
        auto counter = torch::zeros({1},
            torch::TensorOptions().device(param.device()).dtype(torch::kInt32));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            param.scalar_type(), "prodigy_reduce_update_d", [&] {
                prodigy_reduce_and_update_d_kernel<scalar_t, scalar_t>
                    <<<grid, block, 0, stream>>>(
                    param.data_ptr<scalar_t>(),
                    param_init.data_ptr<scalar_t>(),
                    grad.data_ptr<scalar_t>(),
                    d_prev,
                    r_partial.data_ptr<float>(),
                    s_partial.data_ptr<float>(),
                    d_t.data_ptr<float>(),
                    reinterpret_cast<unsigned*>(counter.data_ptr<int>()),
                    N, grid);
                SG_LAUNCH_CHECK(stream);
            });
    }

    const bool apply_fp32 = param.scalar_type() == torch::kFloat32 &&
                            grad.scalar_type() == torch::kFloat32;
    if (SG_TUNED_VEC_WIDTH == 4 && apply_fp32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg_sq.data_ptr(), N) &&
        prim::is_vec4_alignable(s_track.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int64_t N4 = N / 4;
        const int grid4 = static_cast<int>(
            std::min<int64_t>(65535, (N4 + block - 1) / block));
        prodigy_apply_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<float4*>(s_track.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            d_t.data_ptr<float>(),
            beta1, beta2, eps, wd, bc1, bc2, N4);
        SG_LAUNCH_CHECK(stream);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "prodigy_apply", [&] {
            prodigy_apply_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                s_track.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                d_t.data_ptr<float>(),
                beta1, beta2, eps, wd, bc1, bc2, N);
            SG_LAUNCH_CHECK(stream);
        });
}


// Per-tensor Prodigy step: d_lr is the current adaptive learning rate.
void launch_fused_prodigy_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor s, torch::Tensor param_init, torch::Tensor grad,
    float lr, float d_lr, float beta1, float beta2,
    float weight_decay, float eps, float bc1, float bc2
) {
    auto d_t = torch::tensor({d_lr},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    auto r_partial = torch::zeros({1},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    auto s_partial = torch::zeros({1},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    launch_prodigy_step(param, exp_avg, exp_avg_sq, s, param_init, grad,
                        d_t, r_partial, s_partial, d_lr,
                        beta1, beta2, eps, weight_decay, bc1, bc2);
}

// DLR reduction: numerator += <grad, param - param_init>,
// denominator += ||s|| * d + eps.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_PRODIGY_MIN_BLOCKS)
prodigy_dlr_reduce_kernel(
    const float* grad, const float* param, const float* param_init,
    const float* s, float* numerator, float* denominator,
    float eps, int64_t N
) {
    float num_acc = 0.0f, den_acc = 0.0f;
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        num_acc += ::grokking::sm90::sg_sanitize_grad(grad[i]) *
                   (param[i] - param_init[i]);
        den_acc += fabsf(s[i]);
    }
    // Fused two-value block reduction → ONE atomicAdd per partial per block
    // (was a per-THREAD atomicAdd, i.e. blockDim.x * gridDim.x atomics).
    float num_block, den_block;
    prim::block_reduce_sum2_f32(num_acc, den_acc, num_block, den_block);
    if (threadIdx.x == 0) {
        atomicAdd(numerator, num_block);
        atomicAdd(denominator, den_block);
    }
}

void launch_prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param, torch::Tensor param_init,
    torch::Tensor s, torch::Tensor numerator, torch::Tensor denominator,
    float eps
) {
    const int64_t N = grad.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    prodigy_dlr_reduce_kernel<<<grid, block, 0, stream>>>(
        grad.data_ptr<float>(), param.data_ptr<float>(),
        param_init.data_ptr<float>(), s.data_ptr<float>(),
        numerator.data_ptr<float>(), denominator.data_ptr<float>(),
        eps, N);
    SG_LAUNCH_CHECK(stream);
}

// Multi-tensor fused reduce + step: accumulate d_lr across all tensors,
// then apply the Prodigy update to each.
void launch_multi_tensor_prodigy_fused_reduce_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    torch::Tensor d_lr_buf,
    float beta1, float beta2, float lr, float wd, float eps
) {
    if (params.empty()) return;
    auto dev = params[0].device();
    auto r_partial = torch::zeros({1},
        torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto s_partial = torch::zeros({1},
        torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    float d_prev = d_lr_buf.item<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;

    for (size_t i = 0; i < params.size(); i++) {
        const int64_t N = params[i].numel();
        if (N == 0) continue;
        const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            params[i].scalar_type(), "prodigy_mt_reduce", [&] {
                prodigy_reduce_kernel<scalar_t, scalar_t>
                    <<<grid, block, 0, stream>>>(
                    params[i].data_ptr<scalar_t>(),
                    param_inits[i].data_ptr<scalar_t>(),
                    grads[i].data_ptr<scalar_t>(),
                    d_prev,
                    r_partial.data_ptr<float>(),
                    s_partial.data_ptr<float>(), N);
                SG_LAUNCH_CHECK(stream);
            });
    }

    prodigy_update_d_kernel<<<1, 1, 0, stream>>>(
        d_lr_buf.data_ptr<float>(),
        r_partial.data_ptr<float>(),
        s_partial.data_ptr<float>(), d_prev);
    SG_LAUNCH_CHECK(stream);

    for (size_t i = 0; i < params.size(); i++) {
        const int64_t N = params[i].numel();
        if (N == 0) continue;
        const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            params[i].scalar_type(), "prodigy_mt_apply", [&] {
                prodigy_apply_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                    <<<grid, block, 0, stream>>>(
                    params[i].data_ptr<scalar_t>(),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    s_bufs[i].data_ptr<float>(),
                    grads[i].data_ptr<scalar_t>(),
                    d_lr_buf.data_ptr<float>(),
                    beta1, beta2, eps, wd, bc1s[i], bc2s[i], N);
                SG_LAUNCH_CHECK(stream);
            });
    }
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_PRODIGY_SM90_CUH_
