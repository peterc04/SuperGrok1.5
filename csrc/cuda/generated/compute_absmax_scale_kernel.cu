/*
 * compute_absmax_scale_kernel — GPU-side max-reduce for FP8/INT8 scaling
 *
 * Replaces .item() in hopper_fp8_gemm by computing the absmax scale
 * entirely on-device, writing the result to device memory.
 * cublasGemmEx reads the scale via CUBLAS_POINTER_MODE_DEVICE.
 *
 * Algorithm:
 *   1. Each thread computes local max(|x[i]|) via grid-stride loop
 *   2. Warp shuffle reduction within each warp
 *   3. Shared memory reduction across warps
 *   4. atomicMax across all blocks (using float-as-int reinterpretation)
 *   5. Final scale = absmax / max_representable (written to device ptr)
 */

#include <torch/extension.h>
#include "platform.h"

// atomicMax for positive floats: reinterpret as int, use integer atomicMax
// Works because IEEE754 floats have monotonic int representation for non-negative values
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (val >= 0) {
        int* addr_as_int = reinterpret_cast<int*>(addr);
        int val_as_int = __float_as_int(val);
        atomicMax(addr_as_int, val_as_int);
    }
}

__launch_bounds__(256, 8)
__global__ void compute_absmax_kernel(
    const float* __restrict__ data,    // [N] input tensor
    float* __restrict__ out_absmax,    // [1] output: max(|data[i]|)
    const int N
) {
    constexpr int BLOCK = 256;
    constexpr int WARPS = BLOCK / WARP_SIZE;

    __shared__ float shared_max[WARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Grid-stride loop: each thread accumulates local max
    float local_max = 0.0f;
    for (int idx = blockIdx.x * BLOCK + tid; idx < N; idx += gridDim.x * BLOCK) {
        local_max = fmaxf(local_max, fabsf(data[idx]));
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, SHFL_DOWN(local_max, offset));
    }

    // Lane 0 writes to shared memory
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();

    // Cross-warp reduction (first warp only)
    if (warp_id == 0) {
        float warp_max = (lane_id < WARPS) ? shared_max[lane_id] : 0.0f;
        for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
            warp_max = fmaxf(warp_max, SHFL_DOWN(warp_max, offset));
        }
        // Lane 0 does atomic max to global output
        if (lane_id == 0) {
            atomicMaxFloat(out_absmax, warp_max);
        }
    }
}

__launch_bounds__(256, 8)
__global__ void compute_scale_from_absmax_kernel(
    const float* __restrict__ absmax,     // [1] absmax value
    float* __restrict__ out_scale,        // [1] output scale
    const float max_representable         // e.g. 448.0f for FP8 E4M3
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float am = absmax[0];
        out_scale[0] = (am > 0.0f) ? (am / max_representable) : 1.0f;
    }
}

void launch_compute_absmax_scale(
    torch::Tensor data,
    torch::Tensor out_absmax,
    torch::Tensor out_scale,
    float max_representable
) {
    const int N = data.numel();
    if (N == 0) return;

    // Zero the absmax accumulator
    out_absmax.zero_();

    // Cap grid to avoid excessive atomics
    const int max_blocks = 1024;
    const int grid = std::min((N + 255) / 256, max_blocks);

    compute_absmax_kernel<<<grid, 256>>>(
        data.data_ptr<float>(),
        out_absmax.data_ptr<float>(),
        N);

    // Compute scale from absmax (single thread)
    compute_scale_from_absmax_kernel<<<1, 1>>>(
        out_absmax.data_ptr<float>(),
        out_scale.data_ptr<float>(),
        max_representable);
}
