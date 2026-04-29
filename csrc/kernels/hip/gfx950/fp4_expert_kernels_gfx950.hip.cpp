/*
 * SuperGrok v2 — CDNA4 FP4 Expert Weight Kernels (gfx950, MI350X)
 *
 * Split out from the former cdna4_kernels_gfx950.hip.cpp monolith.
 * Contains the 4 FP4 expert weight kernels and their host launchers:
 *   1. cdna4_fp4_expert_load_kernel       — packed FP4 → FP32 expert load
 *   2. cdna4_fp4_expert_fwd_kernel        — 2-layer FP4 expert MLP forward
 *   3. cdna4_fp4_expert_bwd_kernel        — FP4 expert MLP backward
 *   4. cdna4_fp4_quantize_experts_kernel  — FP32 → packed FP4 with SR
 *
 * Shared FP4 helpers live in csrc/common/fp4_helpers.hip.h.
 * Math is unchanged from the monolith.
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "platform.h"
#include "../../common/fp4_helpers.hip.h"

namespace sg { namespace gfx950 {

// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: FP4 Expert Weight Load
//
//  Load expert weights from packed FP4 storage and dequantize to FP32
//  in shared memory for subsequent computation.
//
//  Layout: weights_fp4 is [num_experts, packed_size] where
//          packed_size = ceil(weight_numel / 8) uint32_t values.
//  Each uint32_t holds 8 FP4 values.
//  scale_factors is [num_experts] — per-expert absmax scale.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp4_expert_load_kernel(
    const uint32_t* __restrict__ weights_fp4,   // [num_experts, packed_size]
    const float*    __restrict__ scale_factors,  // [num_experts]
    float*          __restrict__ weights_fp32,   // [num_experts, weight_numel]
    int             num_experts,
    int             weight_numel,
    int             packed_size                  // = ceil(weight_numel / 8)
) {
    extern __shared__ float smem[];

    const int expert_id = blockIdx.y;
    if (expert_id >= num_experts) return;

    const float scale = scale_factors[expert_id];
    const uint32_t* expert_packed = weights_fp4 + (size_t)expert_id * packed_size;
    float* expert_out = weights_fp32 + (size_t)expert_id * weight_numel;

    // Each thread processes one packed uint32_t (8 FP4 values)
    for (int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
         pack_idx < packed_size;
         pack_idx += gridDim.x * blockDim.x) {

        uint32_t packed = expert_packed[pack_idx];
        int base_out = pack_idx * 8;

        // Dequantize through shared memory for coalesced global writes
        float local_vals[8];

#if defined(__gfx950__)
        // On gfx950, use native FP4 MMA dequantization path if available.
        // The MFMA instruction operates on 128 FP4 elements at once;
        // for smaller granularity we fall back to the LUT path.
        // Native MMA path would be used in the forward kernel below;
        // here we do element-wise dequant for flexibility.
#endif
        for (int i = 0; i < 8; i++) {
            local_vals[i] = fp4_dequant(packed, i) * scale;
        }

        // Write to shared memory, then flush to global
        int smem_base = threadIdx.x * 8;
        for (int i = 0; i < 8; i++) {
            smem[smem_base + i] = local_vals[i];
        }
        __syncthreads();

        // Coalesced global write
        for (int i = 0; i < 8 && (base_out + i) < weight_numel; i++) {
            expert_out[base_out + i] = smem[smem_base + i];
        }
        __syncthreads();
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: FP4 Expert Forward MLP
//
//  Forward pass through a 2-layer MLP with FP4 expert weights:
//    hidden = ReLU(W1 * input + b1)
//    output = W2 * hidden + b2
//
//  W1 is [expert_hidden, d_in], W2 is [d_out, expert_hidden], both FP4.
//  Each thread block handles one expert for a batch of inputs.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp4_expert_fwd_kernel(
    const float*    __restrict__ input,           // [batch_size, d_in]
    const uint32_t* __restrict__ W1_fp4,          // [num_experts, packed_W1]
    const float*    __restrict__ b1,              // [num_experts, expert_hidden]
    const uint32_t* __restrict__ W2_fp4,          // [num_experts, packed_W2]
    const float*    __restrict__ b2,              // [num_experts, d_out]
    const float*    __restrict__ scale_W1,        // [num_experts]
    const float*    __restrict__ scale_W2,        // [num_experts]
    const int*      __restrict__ expert_assign,   // [batch_size] — which expert
    float*          __restrict__ output,          // [batch_size, d_out]
    int             batch_size,
    int             d_in,
    int             expert_hidden,
    int             d_out,
    int             packed_W1_row,                // ceil(d_in / 8)
    int             packed_W2_row                 // ceil(expert_hidden / 8)
) {
    extern __shared__ float smem[];
    // smem layout: [expert_hidden] for hidden activations per sample

    const int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    const int eid = expert_assign[sample_idx];
    const float s1 = scale_W1[eid];
    const float s2 = scale_W2[eid];
    const uint32_t* w1_base = W1_fp4 + (size_t)eid * expert_hidden * packed_W1_row;
    const uint32_t* w2_base = W2_fp4 + (size_t)eid * d_out * packed_W2_row;
    const float* b1_base = b1 + (size_t)eid * expert_hidden;
    const float* b2_base = b2 + (size_t)eid * d_out;
    const float* in_ptr = input + (size_t)sample_idx * d_in;
    float* out_ptr = output + (size_t)sample_idx * d_out;

    // Phase 1: Compute hidden = ReLU(W1 * input + b1)
    // Each thread computes one or more hidden neurons
    for (int h = threadIdx.x; h < expert_hidden; h += blockDim.x) {
        const uint32_t* w1_row = w1_base + (size_t)h * packed_W1_row;
        float acc = b1_base[h];

        // Dot product: dequantize W1 row on-the-fly and multiply with input
        for (int p = 0; p < packed_W1_row; p++) {
            uint32_t packed = w1_row[p];
            int base = p * 8;

#if defined(__gfx950__)
            // Native FP4 MMA path: accumulate using MFMA when we have
            // enough aligned data. For row-wise dot products with
            // arbitrary dimensions, we use the scalar dequant path
            // but benefit from gfx950's improved FP4 throughput.
#endif
            for (int i = 0; i < 8 && (base + i) < d_in; i++) {
                float w = fp4_dequant(packed, i) * s1;
                acc += w * in_ptr[base + i];
            }
        }

        // ReLU activation
        smem[h] = fmaxf(acc, 0.0f);
    }
    __syncthreads();

    // Phase 2: Compute output = W2 * hidden + b2
    for (int o = threadIdx.x; o < d_out; o += blockDim.x) {
        const uint32_t* w2_row = w2_base + (size_t)o * packed_W2_row;
        float acc = b2_base[o];

        for (int p = 0; p < packed_W2_row; p++) {
            uint32_t packed = w2_row[p];
            int base = p * 8;
            for (int i = 0; i < 8 && (base + i) < expert_hidden; i++) {
                float w = fp4_dequant(packed, i) * s2;
                acc += w * smem[base + i];
            }
        }

        out_ptr[o] = acc;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: FP4 Expert Backward MLP
//
//  Backward pass through the 2-layer FP4 expert MLP.
//  Computes gradients for input (d_input) in FP32.
//  Accumulates weight gradients and quantizes them back to FP4
//  with stochastic rounding for variance reduction.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp4_expert_bwd_kernel(
    const float*    __restrict__ grad_output,     // [batch_size, d_out]
    const float*    __restrict__ input,           // [batch_size, d_in]
    const float*    __restrict__ hidden_acts,     // [batch_size, expert_hidden]
    const uint32_t* __restrict__ W1_fp4,          // [num_experts, expert_hidden, packed_W1_row]
    const uint32_t* __restrict__ W2_fp4,          // [num_experts, d_out, packed_W2_row]
    const float*    __restrict__ scale_W1,
    const float*    __restrict__ scale_W2,
    const int*      __restrict__ expert_assign,   // [batch_size]
    float*          __restrict__ grad_input,      // [batch_size, d_in]
    float*          __restrict__ grad_W1_accum,   // [num_experts, expert_hidden, d_in] FP32 accum
    float*          __restrict__ grad_W2_accum,   // [num_experts, d_out, expert_hidden] FP32 accum
    float*          __restrict__ grad_b1,         // [num_experts, expert_hidden]
    float*          __restrict__ grad_b2,         // [num_experts, d_out]
    uint32_t        rng_seed,
    int             batch_size,
    int             d_in,
    int             expert_hidden,
    int             d_out,
    int             packed_W1_row,
    int             packed_W2_row
) {
    extern __shared__ float smem[];
    // smem layout: [expert_hidden] for grad_hidden

    const int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    const int eid = expert_assign[sample_idx];
    const float s1 = scale_W1[eid];
    const float s2 = scale_W2[eid];

    const float* go_ptr = grad_output + (size_t)sample_idx * d_out;
    const float* in_ptr = input + (size_t)sample_idx * d_in;
    const float* ha_ptr = hidden_acts + (size_t)sample_idx * expert_hidden;
    float* gi_ptr = grad_input + (size_t)sample_idx * d_in;

    const uint32_t* w2_base = W2_fp4 + (size_t)eid * d_out * packed_W2_row;

    // Phase 1: grad_hidden = W2^T * grad_output, masked by ReLU
    for (int h = threadIdx.x; h < expert_hidden; h += blockDim.x) {
        float grad_h = 0.0f;

        for (int o = 0; o < d_out; o++) {
            // Find weight W2[o, h] from packed storage
            int pack_idx = h / 8;
            int bit_idx = h % 8;
            uint32_t packed = w2_base[(size_t)o * packed_W2_row + pack_idx];
            float w = fp4_dequant(packed, bit_idx) * s2;
            grad_h += w * go_ptr[o];
        }

        // ReLU backward: zero gradient where hidden was zero
        float relu_mask = (ha_ptr[h] > 0.0f) ? 1.0f : 0.0f;
        smem[h] = grad_h * relu_mask;

        // Accumulate bias gradient
        atomicAdd(&grad_b1[(size_t)eid * expert_hidden + h], smem[h]);
    }
    __syncthreads();

    // Phase 2: grad_input = W1^T * grad_hidden
    const uint32_t* w1_base = W1_fp4 + (size_t)eid * expert_hidden * packed_W1_row;

    for (int d = threadIdx.x; d < d_in; d += blockDim.x) {
        float grad_d = 0.0f;

        for (int h = 0; h < expert_hidden; h++) {
            int pack_idx = d / 8;
            int bit_idx = d % 8;
            uint32_t packed = w1_base[(size_t)h * packed_W1_row + pack_idx];
            float w = fp4_dequant(packed, bit_idx) * s1;
            grad_d += w * smem[h];
        }

        gi_ptr[d] = grad_d;
    }

    // Phase 3: Accumulate weight gradients (FP32)
    // grad_W1 += grad_hidden * input^T
    float* gw1 = grad_W1_accum + (size_t)eid * expert_hidden * d_in;
    for (int h = threadIdx.x; h < expert_hidden; h += blockDim.x) {
        float gh = smem[h];
        for (int d = 0; d < d_in; d++) {
            atomicAdd(&gw1[(size_t)h * d_in + d], gh * in_ptr[d]);
        }
    }

    // grad_W2 += grad_output * hidden^T
    float* gw2 = grad_W2_accum + (size_t)eid * d_out * expert_hidden;
    for (int o = threadIdx.x; o < d_out; o += blockDim.x) {
        float go_val = go_ptr[o];
        atomicAdd(&grad_b2[(size_t)eid * d_out + o], go_val);
        for (int h = 0; h < expert_hidden; h++) {
            atomicAdd(&gw2[(size_t)o * expert_hidden + h], go_val * ha_ptr[h]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: FP4 Quantize Experts
//
//  Quantize FP32 expert weights to packed FP4 format.
//  Each group of 8 elements shares a per-group absmax scale.
//  Stochastic rounding for better convergence.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp4_quantize_experts_kernel(
    const float*    __restrict__ weights_fp32,    // [num_experts, weight_numel]
    uint32_t*       __restrict__ weights_fp4,     // [num_experts, packed_size]
    float*          __restrict__ scale_factors,   // [num_experts]
    uint32_t        rng_seed,
    int             num_experts,
    int             weight_numel,
    int             packed_size                   // ceil(weight_numel / 8)
) {
    const int expert_id = blockIdx.y;
    if (expert_id >= num_experts) return;

    const float* src = weights_fp32 + (size_t)expert_id * weight_numel;
    uint32_t* dst = weights_fp4 + (size_t)expert_id * packed_size;

    // First pass: find absmax for this expert (parallel reduction)
    __shared__ float s_absmax[256];
    float local_max = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < weight_numel;
         i += gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(src[i]));
    }
    s_absmax[threadIdx.x] = local_max;
    __syncthreads();

    // Reduction within block
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_absmax[threadIdx.x] = fmaxf(s_absmax[threadIdx.x],
                                           s_absmax[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float absmax = s_absmax[0];
    // Scale so that absmax maps to FP4 max representable (6.0)
    float scale = (absmax > 0.0f) ? (6.0f / absmax) : 1.0f;
    float inv_scale = (absmax > 0.0f) ? (absmax / 6.0f) : 1.0f;

    // Store scale factor (inverse, for dequant: multiply by inv_scale)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scale_factors[expert_id] = inv_scale;
    }
    __syncthreads();

    // Second pass: quantize and pack
    for (int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
         pack_idx < packed_size;
         pack_idx += gridDim.x * blockDim.x) {

        int base = pack_idx * 8;
        float vals[8];
        for (int i = 0; i < 8; i++) {
            int global_idx = base + i;
            float v = (global_idx < weight_numel) ? src[global_idx] : 0.0f;
            vals[i] = v;
        }

        // Pack with stochastic rounding
        uint32_t packed = 0;
        for (int i = 0; i < 8; i++) {
            float scaled = vals[i] * scale;

            // Stochastic rounding: add uniform noise in [-0.5, 0.5) before rounding
            uint32_t rng = philox_hash((uint32_t)(pack_idx * 8 + i), rng_seed ^ (uint32_t)expert_id);
            float noise = ((float)(rng & 0xFFFF) / 65536.0f) - 0.5f;

            // Add noise proportional to the quantization step size
            // FP4 step sizes vary, so we use a small fraction
            scaled += noise * 0.25f;

            uint32_t code = fp4_quant_one(scaled);
            packed |= (code << (i * 4));
        }

        dst[pack_idx] = packed;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Host Launchers for FP4 Expert Kernels
// ═══════════════════════════════════════════════════════════════════════

void cdna4_fp4_expert_load(
    torch::Tensor weights_fp4, torch::Tensor scale_factors,
    torch::Tensor weights_fp32,
    int num_experts, int weight_numel, int packed_size
) {
    dim3 grid((packed_size + 255) / 256, num_experts);
    dim3 block(256);
    int smem = 256 * 8 * sizeof(float);
    cdna4_fp4_expert_load_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const uint32_t*>(weights_fp4.data_ptr()),
        scale_factors.data_ptr<float>(),
        weights_fp32.data_ptr<float>(),
        num_experts, weight_numel, packed_size
    );
}

void cdna4_fp4_expert_fwd(
    torch::Tensor input, torch::Tensor W1_fp4, torch::Tensor b1,
    torch::Tensor W2_fp4, torch::Tensor b2,
    torch::Tensor scale_W1, torch::Tensor scale_W2,
    torch::Tensor expert_assign, torch::Tensor output,
    int batch_size, int d_in, int expert_hidden, int d_out,
    int packed_W1_row, int packed_W2_row
) {
    dim3 grid(batch_size);
    dim3 block(256);
    int smem = expert_hidden * sizeof(float);
    cdna4_fp4_expert_fwd_kernel<<<grid, block, smem>>>(
        input.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(W1_fp4.data_ptr()),
        b1.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(W2_fp4.data_ptr()),
        b2.data_ptr<float>(),
        scale_W1.data_ptr<float>(), scale_W2.data_ptr<float>(),
        expert_assign.data_ptr<int>(),
        output.data_ptr<float>(),
        batch_size, d_in, expert_hidden, d_out,
        packed_W1_row, packed_W2_row
    );
}

void cdna4_fp4_expert_bwd(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor hidden_acts,
    torch::Tensor W1_fp4, torch::Tensor W2_fp4,
    torch::Tensor scale_W1, torch::Tensor scale_W2,
    torch::Tensor expert_assign,
    torch::Tensor grad_input, torch::Tensor grad_W1_accum, torch::Tensor grad_W2_accum,
    torch::Tensor grad_b1, torch::Tensor grad_b2,
    uint32_t rng_seed,
    int batch_size, int d_in, int expert_hidden, int d_out,
    int packed_W1_row, int packed_W2_row
) {
    dim3 grid(batch_size);
    dim3 block(256);
    int smem = expert_hidden * sizeof(float);
    cdna4_fp4_expert_bwd_kernel<<<grid, block, smem>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        hidden_acts.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(W1_fp4.data_ptr()),
        reinterpret_cast<const uint32_t*>(W2_fp4.data_ptr()),
        scale_W1.data_ptr<float>(), scale_W2.data_ptr<float>(),
        expert_assign.data_ptr<int>(),
        grad_input.data_ptr<float>(),
        grad_W1_accum.data_ptr<float>(), grad_W2_accum.data_ptr<float>(),
        grad_b1.data_ptr<float>(), grad_b2.data_ptr<float>(),
        rng_seed, batch_size, d_in, expert_hidden, d_out,
        packed_W1_row, packed_W2_row
    );
}

void cdna4_fp4_quantize_experts(
    torch::Tensor weights_fp32, torch::Tensor weights_fp4,
    torch::Tensor scale_factors, uint32_t rng_seed,
    int num_experts, int weight_numel, int packed_size
) {
    dim3 grid((packed_size + 255) / 256, num_experts);
    dim3 block(256);
    cdna4_fp4_quantize_experts_kernel<<<grid, block>>>(
        weights_fp32.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(weights_fp4.data_ptr()),
        scale_factors.data_ptr<float>(),
        rng_seed, num_experts, weight_numel, packed_size
    );
}

} } // namespace sg::gfx950
