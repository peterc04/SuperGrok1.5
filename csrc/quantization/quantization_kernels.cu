/*
 * Quantization CUDA Kernels — FP8, INT8, INT4, MXFP4, NVFP4
 *
 * Provides GPU-accelerated quantize/dequantize routines for all formats
 * in the quantization registry. Each format uses per-tensor or per-block
 * scaling to maximize dynamic range.
 *
 * FP8 E4M3:  Hopper+ native, emulated on Ampere via FP32 cast
 * INT8:      Per-tensor absmax symmetric quantization
 * INT4:      Per-group (group_size=32) symmetric quantization, packed 2-per-byte
 * MXFP4:     Microscaling FP4 E2M1 with shared 8-bit exponent per 32-element block
 * NVFP4:     Blackwell-native FP4 E2M1 with per-16-element block scaling
 *
 * All kernels accept FP32 input and produce quantized output + scale tensors.
 * Dequantize kernels reverse the process for optimizer state reconstruction.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "platform.h"

// Block size for quantization kernels
constexpr int QUANT_BLOCK = 256;


// ═══════════════════════════════════════════════════════════════════════
//  FP8 E4M3 Quantize / Dequantize — NVIDIA only
//
//  FP8 hardware types are NVIDIA-specific (sm_89+). The software
//  emulation below uses standard FP32 bit manipulation, but FP8 is
//  only useful on NVIDIA GPUs. AMD GPUs should use BF16 or INT8.
//
//  E4M3: 4-bit exponent, 3-bit mantissa, 1 sign bit
//  Range: [-448, 448], min subnormal ~2^-9
//  Per-tensor scaling: scale = max(|x|) / 448
// ═══════════════════════════════════════════════════════════════════════

#if GROK_CUDA  // FP8 is NVIDIA-only

// FP8 E4M3 codebook constants
constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val, float inv_scale) {
    float scaled = val * inv_scale;
    scaled = fminf(fmaxf(scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX);

    // Encode: sign(1) | exponent(4) | mantissa(3)
    uint32_t bits = __float_as_uint(scaled);
    uint32_t sign = (bits >> 24) & 0x80;       // bit 31 → bit 7
    int32_t exp_val = ((bits >> 23) & 0xFF) - 127 + 7;  // rebias to E4M3 bias=7
    uint32_t mant = (bits >> 20) & 0x07;       // top 3 mantissa bits

    // Clamp exponent
    if (exp_val <= 0) {
        // Subnormal or zero
        if (exp_val < -3) return (uint8_t)sign;  // too small → zero
        mant = (mant | 0x08) >> (1 - exp_val);   // denormalize
        exp_val = 0;
    } else if (exp_val >= 15) {
        exp_val = 15;
        mant = 0x06;  // max normal (not NaN, E4M3 has no inf)
    }

    return (uint8_t)(sign | ((exp_val & 0x0F) << 3) | (mant & 0x07));
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
    uint32_t sign = (val & 0x80) ? 0x80000000u : 0;
    uint32_t exp_bits = (val >> 3) & 0x0F;
    uint32_t mant = val & 0x07;

    float result;
    if (exp_bits == 0) {
        // Subnormal: value = (-1)^s * 2^(-6) * (0.mant)
        result = ldexpf((float)mant, -6 - 3);  // 2^(-6) * mant/8
    } else if (exp_bits == 15 && mant == 0x07) {
        // NaN
        return __uint_as_float(sign | 0x7FC00000u);
    } else {
        // Normal: value = (-1)^s * 2^(exp-7) * (1.mant)
        float m = 1.0f + (float)mant / 8.0f;
        result = ldexpf(m, (int)exp_bits - 7);
    }
    return (sign ? -result : result);
}

__launch_bounds__(256, 8)
__global__ void quantize_fp8_e4m3_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale,         // [1] — per-tensor scale
    const int N
) {
    // Phase 1: find absmax via block reduction
    __shared__ float smem_max[QUANT_BLOCK];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float local_max = 0.0f;
    #pragma unroll 4
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }
    smem_max[tid] = local_max;
    __syncthreads();

    // Block reduction
    #pragma unroll 4
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)scale, __float_as_int(smem_max[0]));
    __syncthreads();

    // Phase 2: quantize
    float s_val = *scale;
    if (s_val == 0.0f) s_val = 1.0f;
    float inv_scale = FP8_E4M3_MAX / s_val;

    #pragma unroll 4
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        output[i] = float_to_fp8_e4m3(input[i], inv_scale);
    }

    // Write actual scale (absmax / FP8_MAX) for dequant
    if (blockIdx.x == 0 && tid == 0) {
        *scale = s_val / FP8_E4M3_MAX;
    }
}

__launch_bounds__(256, 8)
__global__ void dequantize_fp8_e4m3_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const int N
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    output[gid] = fp8_e4m3_to_float(input[gid]) * (*scale);
}

#endif  // GROK_CUDA (FP8 kernels)


// ═══════════════════════════════════════════════════════════════════════
//  INT8 Symmetric Quantize / Dequantize
//
//  Per-tensor absmax scaling: scale = max(|x|) / 127
//  Quantized value = clamp(round(x / scale), -127, 127)
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void quantize_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    const int N
) {
    __shared__ float smem_max[QUANT_BLOCK];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float local_max = 0.0f;
    #pragma unroll 4
    for (int i = gid; i < N; i += gridDim.x * blockDim.x)
        local_max = fmaxf(local_max, fabsf(input[i]));
    smem_max[tid] = local_max;
    __syncthreads();

    #pragma unroll 4
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)scale, __float_as_int(smem_max[0]));
    __syncthreads();

    float s_val = *scale;
    if (s_val == 0.0f) s_val = 1.0f;
    float inv_scale = 127.0f / s_val;

    #pragma unroll 4
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        float v = rintf(input[i] * inv_scale);
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        output[i] = (int8_t)v;
    }

    if (blockIdx.x == 0 && tid == 0) {
        *scale = s_val / 127.0f;
    }
}

__launch_bounds__(256, 8)
__global__ void dequantize_int8_kernel(
    const int8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const int N
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    output[gid] = (float)input[gid] * (*scale);
}


// ═══════════════════════════════════════════════════════════════════════
//  INT4 Per-Group Symmetric Quantize / Dequantize
//
//  Group size = 32 elements. Per-group scale = max(|group|) / 7.
//  Two INT4 values packed per byte (low nibble = even index, high = odd).
//  Output tensor size = N/2 bytes + N/group_size scales.
// ═══════════════════════════════════════════════════════════════════════

constexpr int INT4_GROUP_SIZE = 32;

__launch_bounds__(256, 8)
__global__ void quantize_int4_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,      // [N/2] packed
    float* __restrict__ scales,        // [num_groups]
    const int N
) {
    const int group_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = group_idx * INT4_GROUP_SIZE;
    if (base >= N) return;
    const int group_end = min(base + INT4_GROUP_SIZE, N);

    // Find group absmax
    __shared__ float smem[32];
    float local_max = 0.0f;
    if (tid < INT4_GROUP_SIZE && base + tid < N)
        local_max = fabsf(input[base + tid]);
    smem[tid] = local_max;
    __syncthreads();

    #pragma unroll 4
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float absmax = smem[0];
    float s_val = (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
    float inv_s = 1.0f / s_val;

    if (tid == 0) scales[group_idx] = s_val;
    __syncthreads();

    // Quantize and pack pairs
    if (tid < INT4_GROUP_SIZE / 2 && base + tid * 2 < N) {
        int idx0 = base + tid * 2;
        int idx1 = idx0 + 1;

        int8_t q0 = (int8_t)fminf(fmaxf(rintf(input[idx0] * inv_s), -7.0f), 7.0f);
        int8_t q1 = (idx1 < N) ?
            (int8_t)fminf(fmaxf(rintf(input[idx1] * inv_s), -7.0f), 7.0f) : 0;

        // Pack: low nibble = q0, high nibble = q1 (both in [-7,7], stored as unsigned offset)
        uint8_t packed = ((uint8_t)(q0 + 8) & 0x0F) | (((uint8_t)(q1 + 8) & 0x0F) << 4);
        output[base / 2 + tid] = packed;
    }
}

__launch_bounds__(256, 8)
__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scales,
    const int N
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_idx = gid;
    const int elem_idx = pair_idx * 2;
    if (elem_idx >= N) return;

    int group_idx = elem_idx / INT4_GROUP_SIZE;
    float s_val = scales[group_idx];

    uint8_t packed = input[pair_idx];
    int8_t q0 = (int8_t)((packed & 0x0F) - 8);
    int8_t q1 = (int8_t)(((packed >> 4) & 0x0F) - 8);

    output[elem_idx] = (float)q0 * s_val;
    if (elem_idx + 1 < N)
        output[elem_idx + 1] = (float)q1 * s_val;
}


// ═══════════════════════════════════════════════════════════════════════
//  MXFP4 (Microscaling FP4 E2M1) Quantize / Dequantize
//
//  32-element blocks share an 8-bit shared exponent.
//  Each element: 1 sign + 2 exponent + 1 mantissa = 4 bits.
//  E2M1 codebook: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
//  Two values packed per byte.
// ═══════════════════════════════════════════════════════════════════════

constexpr int MXFP4_BLOCK_SIZE = 32;

// E2M1 codebook (positive values)
__constant__ float MXFP4_CODEBOOK[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ uint8_t float_to_mxfp4(float val, float inv_block_scale) {
    float abs_scaled = fabsf(val * inv_block_scale);
    // Find nearest codebook entry
    uint8_t best = 0;
    float best_err = abs_scaled;  // error vs 0.0
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        float err = fabsf(abs_scaled - MXFP4_CODEBOOK[i]);
        if (err < best_err) { best_err = err; best = (uint8_t)i; }
    }
    uint8_t sign = (val < 0.0f) ? 0x08 : 0x00;
    return sign | best;
}

__device__ __forceinline__ float mxfp4_to_float(uint8_t code, float block_scale) {
    float val = MXFP4_CODEBOOK[code & 0x07];
    return (code & 0x08) ? -val * block_scale : val * block_scale;
}

__launch_bounds__(256, 8)
__global__ void quantize_mxfp4_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,         // [N/2] packed
    float* __restrict__ block_scales,     // [num_blocks]
    const int N
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = block_idx * MXFP4_BLOCK_SIZE;
    if (base >= N) return;

    // Find block absmax
    __shared__ float smem[32];
    float local_max = 0.0f;
    if (tid < MXFP4_BLOCK_SIZE && base + tid < N)
        local_max = fabsf(input[base + tid]);
    smem[tid] = local_max;
    __syncthreads();

    #pragma unroll 4
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float absmax = smem[0];
    // Scale so that absmax maps to max codebook value (6.0)
    float block_scale = (absmax > 0.0f) ? (absmax / 6.0f) : 1.0f;
    float inv_scale = 1.0f / block_scale;

    if (tid == 0) block_scales[block_idx] = block_scale;
    __syncthreads();

    // Quantize and pack pairs
    if (tid < MXFP4_BLOCK_SIZE / 2 && base + tid * 2 < N) {
        int idx0 = base + tid * 2;
        int idx1 = idx0 + 1;

        uint8_t q0 = float_to_mxfp4(input[idx0], inv_scale);
        uint8_t q1 = (idx1 < N) ? float_to_mxfp4(input[idx1], inv_scale) : 0;

        output[base / 2 + tid] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}

__launch_bounds__(256, 8)
__global__ void dequantize_mxfp4_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ block_scales,
    const int N
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_idx = gid;
    const int elem_idx = pair_idx * 2;
    if (elem_idx >= N) return;

    int block_idx = elem_idx / MXFP4_BLOCK_SIZE;
    float bs = block_scales[block_idx];

    uint8_t packed = input[pair_idx];
    output[elem_idx] = mxfp4_to_float(packed & 0x0F, bs);
    if (elem_idx + 1 < N)
        output[elem_idx + 1] = mxfp4_to_float((packed >> 4) & 0x0F, bs);
}


// ═══════════════════════════════════════════════════════════════════════
//  NVFP4 (Blackwell FP4 E2M1 with per-16-element blocks) — NVIDIA only
//
//  Same E2M1 codebook as MXFP4 but uses 16-element blocks (Blackwell
//  hardware native block size). Provides finer granularity scaling.
//  On non-Blackwell hardware, functionally identical to MXFP4 with
//  smaller block size.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_CUDA  // NVFP4 is NVIDIA Blackwell-specific

constexpr int NVFP4_BLOCK_SIZE = 16;

__launch_bounds__(256, 8)
__global__ void quantize_nvfp4_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ block_scales,
    const int N
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = block_idx * NVFP4_BLOCK_SIZE;
    if (base >= N) return;

    __shared__ float smem[16];
    float local_max = 0.0f;
    if (tid < NVFP4_BLOCK_SIZE && base + tid < N)
        local_max = fabsf(input[base + tid]);
    smem[tid] = local_max;
    __syncthreads();

    #pragma unroll 4
    for (int s = 8; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float absmax = smem[0];
    float block_scale = (absmax > 0.0f) ? (absmax / 6.0f) : 1.0f;
    float inv_scale = 1.0f / block_scale;

    if (tid == 0) block_scales[block_idx] = block_scale;
    __syncthreads();

    if (tid < NVFP4_BLOCK_SIZE / 2 && base + tid * 2 < N) {
        int idx0 = base + tid * 2;
        int idx1 = idx0 + 1;

        uint8_t q0 = float_to_mxfp4(input[idx0], inv_scale);
        uint8_t q1 = (idx1 < N) ? float_to_mxfp4(input[idx1], inv_scale) : 0;

        output[base / 2 + tid] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}

__launch_bounds__(256, 8)
__global__ void dequantize_nvfp4_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ block_scales,
    const int N
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_idx = gid;
    const int elem_idx = pair_idx * 2;
    if (elem_idx >= N) return;

    int block_idx = elem_idx / NVFP4_BLOCK_SIZE;
    float bs = block_scales[block_idx];

    uint8_t packed = input[pair_idx];
    output[elem_idx] = mxfp4_to_float(packed & 0x0F, bs);
    if (elem_idx + 1 < N)
        output[elem_idx + 1] = mxfp4_to_float((packed >> 4) & 0x0F, bs);
}

#endif  // GROK_CUDA (NVFP4 kernels)


// ═══════════════════════════════════════════════════════════════════════
//  Host launcher functions (called from ops.cpp)
// ═══════════════════════════════════════════════════════════════════════

// FP8 E4M3 — NVIDIA only
#if GROK_CUDA
std::vector<torch::Tensor> quantize_fp8_e4m3(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    auto input_f = input.to(torch::kFloat32).contiguous().reshape(-1);
    const int N = input_f.numel();

    auto output = torch::empty({N}, input.options().dtype(torch::kUInt8));
    auto scale = torch::zeros({1}, input.options().dtype(torch::kFloat32));

    int grid = (N + QUANT_BLOCK - 1) / QUANT_BLOCK;
    grid = min(grid, 128);  // cap grid for reduction phase
    quantize_fp8_e4m3_kernel<<<grid, QUANT_BLOCK>>>(
        input_f.data_ptr<float>(), output.data_ptr<uint8_t>(),
        scale.data_ptr<float>(), N);

    return {output, scale};
}

torch::Tensor dequantize_fp8_e4m3(torch::Tensor input, torch::Tensor scale, int64_t numel) {
    auto output = torch::empty({numel}, scale.options().dtype(torch::kFloat32));
    int grid = (numel + QUANT_BLOCK - 1) / QUANT_BLOCK;
    dequantize_fp8_e4m3_kernel<<<grid, QUANT_BLOCK>>>(
        input.data_ptr<uint8_t>(), output.data_ptr<float>(),
        scale.data_ptr<float>(), numel);
    return output;
}
#else
std::vector<torch::Tensor> quantize_fp8_e4m3(torch::Tensor input) {
    TORCH_CHECK(false, "FP8 E4M3 quantization requires NVIDIA GPU (sm_89+)");
    return {};
}
torch::Tensor dequantize_fp8_e4m3(torch::Tensor input, torch::Tensor scale, int64_t numel) {
    TORCH_CHECK(false, "FP8 E4M3 dequantization requires NVIDIA GPU (sm_89+)");
    return torch::Tensor();
}
#endif  // GROK_CUDA (FP8 launchers)

// INT8
std::vector<torch::Tensor> quantize_int8(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    auto input_f = input.to(torch::kFloat32).contiguous().reshape(-1);
    const int N = input_f.numel();

    auto output = torch::empty({N}, input.options().dtype(torch::kInt8));
    auto scale = torch::zeros({1}, input.options().dtype(torch::kFloat32));

    int grid = (N + QUANT_BLOCK - 1) / QUANT_BLOCK;
    grid = min(grid, 128);
    quantize_int8_kernel<<<grid, QUANT_BLOCK>>>(
        input_f.data_ptr<float>(), output.data_ptr<int8_t>(),
        scale.data_ptr<float>(), N);

    return {output, scale};
}

torch::Tensor dequantize_int8(torch::Tensor input, torch::Tensor scale, int64_t numel) {
    auto output = torch::empty({numel}, scale.options().dtype(torch::kFloat32));
    int grid = (numel + QUANT_BLOCK - 1) / QUANT_BLOCK;
    dequantize_int8_kernel<<<grid, QUANT_BLOCK>>>(
        input.data_ptr<int8_t>(), output.data_ptr<float>(),
        scale.data_ptr<float>(), numel);
    return output;
}

// INT4
std::vector<torch::Tensor> quantize_int4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    auto input_f = input.to(torch::kFloat32).contiguous().reshape(-1);
    const int N = input_f.numel();

    int num_groups = (N + INT4_GROUP_SIZE - 1) / INT4_GROUP_SIZE;
    int packed_size = (N + 1) / 2;

    auto output = torch::empty({packed_size}, input.options().dtype(torch::kUInt8));
    auto scales = torch::empty({num_groups}, input.options().dtype(torch::kFloat32));

    quantize_int4_kernel<<<num_groups, 32>>>(
        input_f.data_ptr<float>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), N);

    return {output, scales};
}

torch::Tensor dequantize_int4(torch::Tensor input, torch::Tensor scales, int64_t numel) {
    auto output = torch::empty({numel}, scales.options().dtype(torch::kFloat32));
    int num_pairs = (numel + 1) / 2;
    int grid = (num_pairs + QUANT_BLOCK - 1) / QUANT_BLOCK;
    dequantize_int4_kernel<<<grid, QUANT_BLOCK>>>(
        input.data_ptr<uint8_t>(), output.data_ptr<float>(),
        scales.data_ptr<float>(), numel);
    return output;
}

// MXFP4
std::vector<torch::Tensor> quantize_mxfp4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    auto input_f = input.to(torch::kFloat32).contiguous().reshape(-1);
    const int N = input_f.numel();

    int num_blocks = (N + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    int packed_size = (N + 1) / 2;

    auto output = torch::empty({packed_size}, input.options().dtype(torch::kUInt8));
    auto block_scales = torch::empty({num_blocks}, input.options().dtype(torch::kFloat32));

    quantize_mxfp4_kernel<<<num_blocks, 32>>>(
        input_f.data_ptr<float>(), output.data_ptr<uint8_t>(),
        block_scales.data_ptr<float>(), N);

    return {output, block_scales};
}

torch::Tensor dequantize_mxfp4(torch::Tensor input, torch::Tensor block_scales, int64_t numel) {
    auto output = torch::empty({numel}, block_scales.options().dtype(torch::kFloat32));
    int num_pairs = (numel + 1) / 2;
    int grid = (num_pairs + QUANT_BLOCK - 1) / QUANT_BLOCK;
    dequantize_mxfp4_kernel<<<grid, QUANT_BLOCK>>>(
        input.data_ptr<uint8_t>(), output.data_ptr<float>(),
        block_scales.data_ptr<float>(), numel);
    return output;
}

// NVFP4 — NVIDIA Blackwell only
#if GROK_CUDA
std::vector<torch::Tensor> quantize_nvfp4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    auto input_f = input.to(torch::kFloat32).contiguous().reshape(-1);
    const int N = input_f.numel();

    int num_blocks = (N + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;
    int packed_size = (N + 1) / 2;

    auto output = torch::empty({packed_size}, input.options().dtype(torch::kUInt8));
    auto block_scales = torch::empty({num_blocks}, input.options().dtype(torch::kFloat32));

    quantize_nvfp4_kernel<<<num_blocks, 16>>>(
        input_f.data_ptr<float>(), output.data_ptr<uint8_t>(),
        block_scales.data_ptr<float>(), N);

    return {output, block_scales};
}

torch::Tensor dequantize_nvfp4(torch::Tensor input, torch::Tensor block_scales, int64_t numel) {
    auto output = torch::empty({numel}, block_scales.options().dtype(torch::kFloat32));
    int num_pairs = (numel + 1) / 2;
    int grid = (num_pairs + QUANT_BLOCK - 1) / QUANT_BLOCK;
    dequantize_nvfp4_kernel<<<grid, QUANT_BLOCK>>>(
        input.data_ptr<uint8_t>(), output.data_ptr<float>(),
        block_scales.data_ptr<float>(), numel);
    return output;
}
#else
std::vector<torch::Tensor> quantize_nvfp4(torch::Tensor input) {
    TORCH_CHECK(false, "NVFP4 quantization requires NVIDIA Blackwell GPU (sm_100+)");
    return {};
}
torch::Tensor dequantize_nvfp4(torch::Tensor input, torch::Tensor block_scales, int64_t numel) {
    TORCH_CHECK(false, "NVFP4 dequantization requires NVIDIA Blackwell GPU (sm_100+)");
    return torch::Tensor();
}
#endif  // GROK_CUDA (NVFP4 launchers)
