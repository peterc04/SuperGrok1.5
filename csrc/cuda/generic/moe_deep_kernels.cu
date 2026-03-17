/*
 * MoE Deep Optimization Kernels
 *
 * Nine CUDA kernels for advanced Mixture-of-Experts features:
 *
 *   Dynamic Expert Loading (1-3):
 *     1. moe_dynamic_expert_load_kernel   — Determine active experts, load weights to smem
 *     2. moe_dynamic_expert_fwd_kernel    — Forward pass through dynamically-loaded experts
 *     3. moe_dynamic_expert_bwd_kernel    — Backward through dynamic expert loading
 *
 *   Filtered Scan with Param Compaction (4-6):
 *     4. moe_filter_active_params_kernel  — Compact active params from expert assignments
 *     5. moe_scan_compacted_kernel        — Mamba scan on compacted param subset
 *     6. moe_scatter_results_kernel       — Scatter compacted results back to full array
 *
 *   Expert Activation Frequency Counter (7-9):
 *     7. moe_count_expert_activations_kernel   — Atomic count of expert selections
 *     8. moe_compute_load_balance_loss_kernel  — Auxiliary load-balancing loss
 *     9. moe_apply_frequency_scaling_kernel    — Scale expert LRs by frequency
 *
 * All meta-net state is FP32. Uses platform.h abstractions for CUDA/HIP portability.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>

#include "platform.h"
#include "types.h"
#include "ptx_intrinsics.cuh"


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Dynamic Expert Load
//
//  Given gate_logits and a threshold, determine active experts per block.
//  Only load active expert W1/b1/W2/b2 into shared memory (not all experts).
//  Uses __syncthreads() after computing active_mask.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_dynamic_expert_load_kernel(
    const float* __restrict__ gate_logits,   // [N, num_experts]
    const float* __restrict__ W1,            // [num_experts, expert_dim, input_dim]
    const float* __restrict__ b1,            // [num_experts, expert_dim]
    const float* __restrict__ W2,            // [num_experts, input_dim, expert_dim]
    const float* __restrict__ b2,            // [num_experts, input_dim]
    float* __restrict__ loaded_W1,           // [N, MAX_TOPK, expert_dim, input_dim]
    float* __restrict__ loaded_b1,           // [N, MAX_TOPK, expert_dim]
    float* __restrict__ loaded_W2,           // [N, MAX_TOPK, input_dim, expert_dim]
    float* __restrict__ loaded_b2,           // [N, MAX_TOPK, input_dim]
    int* __restrict__ active_expert_ids,     // [N, MAX_TOPK]
    int* __restrict__ num_active_per_sample, // [N]
    const float threshold,
    const int N,
    const int num_experts,
    const int input_dim,
    const int expert_dim
) {
    // Shared memory layout:
    //   active_mask[num_experts] (int) — which experts are active for this block's sample
    //   active_ids[MAX_TOPK] (int) — compacted list of active expert indices
    //   num_active (int) — count of active experts
    extern __shared__ char smem_raw[];
    int* active_mask = reinterpret_cast<int*>(smem_raw);
    int* active_ids = active_mask + num_experts;
    int* num_active = active_ids + MAX_TOPK;

    const int sample_idx = blockIdx.x;
    if (sample_idx >= N) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Step 1: Each thread evaluates gate logits and marks active experts
    if (tid == 0) {
        *num_active = 0;
    }
    for (int e = tid; e < num_experts; e += block_size) {
        float logit = gate_logits[sample_idx * num_experts + e];
        // Softmax numerator: use exp for gating
        active_mask[e] = (logit > threshold) ? 1 : 0;
    }
    __syncthreads();

    // Step 2: Thread 0 compacts the active expert list
    if (tid == 0) {
        int count = 0;
        for (int e = 0; e < num_experts && count < MAX_TOPK; e++) {
            if (active_mask[e]) {
                active_ids[count] = e;
                count++;
            }
        }
        *num_active = count;
        num_active_per_sample[sample_idx] = count;
    }
    __syncthreads();

    int n_active = *num_active;
    if (n_active == 0) return;

    // Step 3: Write active expert IDs to global memory
    if (tid < n_active) {
        active_expert_ids[sample_idx * MAX_TOPK + tid] = active_ids[tid];
    }

    // Step 4: Cooperatively load active expert weights into output buffers
    // Each thread handles a slice of the weight matrices
    for (int a = 0; a < n_active; a++) {
        int expert_id = active_ids[a];

        // Load W1: [expert_dim, input_dim]
        int w1_total = expert_dim * input_dim;
        for (int i = tid; i < w1_total; i += block_size) {
            int row = i / input_dim;
            int col = i % input_dim;
            float val = LDG(&W1[expert_id * w1_total + row * input_dim + col]);
            loaded_W1[((sample_idx * MAX_TOPK + a) * expert_dim + row) * input_dim + col] = val;
        }

        // Load b1: [expert_dim]
        for (int i = tid; i < expert_dim; i += block_size) {
            loaded_b1[(sample_idx * MAX_TOPK + a) * expert_dim + i] =
                LDG(&b1[expert_id * expert_dim + i]);
        }

        // Load W2: [input_dim, expert_dim]
        int w2_total = input_dim * expert_dim;
        for (int i = tid; i < w2_total; i += block_size) {
            int row = i / expert_dim;
            int col = i % expert_dim;
            float val = LDG(&W2[expert_id * w2_total + row * expert_dim + col]);
            loaded_W2[((sample_idx * MAX_TOPK + a) * input_dim + row) * expert_dim + col] = val;
        }

        // Load b2: [input_dim]
        for (int i = tid; i < input_dim; i += block_size) {
            loaded_b2[(sample_idx * MAX_TOPK + a) * input_dim + i] =
                LDG(&b2[expert_id * input_dim + i]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Dynamic Expert Forward
//
//  Forward pass through dynamically-loaded experts. Each thread processes
//  one parameter. Reads expert weights from shared memory (only active
//  experts loaded). Computes:
//    hidden = ReLU(W1 * input + b1)
//    output = sum(gate_weight * (W2 * hidden + b2))
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_dynamic_expert_fwd_kernel(
    const float* __restrict__ input,            // [N, input_dim]
    const float* __restrict__ gate_logits,      // [N, num_experts]
    const float* __restrict__ loaded_W1,        // [N, MAX_TOPK, expert_dim, input_dim]
    const float* __restrict__ loaded_b1,        // [N, MAX_TOPK, expert_dim]
    const float* __restrict__ loaded_W2,        // [N, MAX_TOPK, input_dim, expert_dim]
    const float* __restrict__ loaded_b2,        // [N, MAX_TOPK, input_dim]
    const int* __restrict__ active_expert_ids,  // [N, MAX_TOPK]
    const int* __restrict__ num_active_per_sample, // [N]
    float* __restrict__ output,                 // [N, input_dim]
    const int N,
    const int num_experts,
    const int input_dim,
    const int expert_dim
) {
    // Shared memory for input vector and hidden activations
    extern __shared__ float smem[];
    float* s_input = smem;                        // [input_dim]
    float* s_hidden = smem + input_dim;           // [expert_dim]
    float* s_gate_softmax = s_hidden + expert_dim; // [MAX_TOPK]

    const int sample_idx = blockIdx.x;
    if (sample_idx >= N) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int n_active = num_active_per_sample[sample_idx];

    if (n_active == 0) {
        // Zero output for samples with no active experts
        for (int d = tid; d < input_dim; d += block_size) {
            output[sample_idx * input_dim + d] = 0.0f;
        }
        return;
    }

    // Load input into shared memory
    for (int d = tid; d < input_dim; d += block_size) {
        s_input[d] = input[sample_idx * input_dim + d];
    }

    // Compute softmax gate weights for active experts
    if (tid == 0) {
        float max_logit = -1e30f;
        for (int a = 0; a < n_active; a++) {
            int eid = active_expert_ids[sample_idx * MAX_TOPK + a];
            float logit = gate_logits[sample_idx * num_experts + eid];
            if (logit > max_logit) max_logit = logit;
        }
        float sum_exp = 0.0f;
        for (int a = 0; a < n_active; a++) {
            int eid = active_expert_ids[sample_idx * MAX_TOPK + a];
            float logit = gate_logits[sample_idx * num_experts + eid];
            float exp_val = expf(logit - max_logit);
            s_gate_softmax[a] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int a = 0; a < n_active; a++) {
            s_gate_softmax[a] *= inv_sum;
        }
    }
    __syncthreads();

    // Initialize output accumulators in shared memory
    // Reuse a portion of smem after s_gate_softmax for output accumulation
    float* s_output = s_gate_softmax + MAX_TOPK;  // [input_dim]
    for (int d = tid; d < input_dim; d += block_size) {
        s_output[d] = 0.0f;
    }
    __syncthreads();

    // Process each active expert
    for (int a = 0; a < n_active; a++) {
        float gate_w = s_gate_softmax[a];
        int base_w1 = (sample_idx * MAX_TOPK + a) * expert_dim * input_dim;
        int base_b1 = (sample_idx * MAX_TOPK + a) * expert_dim;
        int base_w2 = (sample_idx * MAX_TOPK + a) * input_dim * expert_dim;
        int base_b2 = (sample_idx * MAX_TOPK + a) * input_dim;

        // Compute hidden = ReLU(W1 * input + b1)
        for (int h = tid; h < expert_dim; h += block_size) {
            float acc = loaded_b1[base_b1 + h];
            for (int d = 0; d < input_dim; d++) {
                acc += loaded_W1[base_w1 + h * input_dim + d] * s_input[d];
            }
            // ReLU activation
            s_hidden[h] = fmaxf(0.0f, acc);
        }
        __syncthreads();

        // Compute output contribution = gate_w * (W2 * hidden + b2)
        for (int d = tid; d < input_dim; d += block_size) {
            float acc = loaded_b2[base_b2 + d];
            for (int h = 0; h < expert_dim; h++) {
                acc += loaded_W2[base_w2 + d * expert_dim + h] * s_hidden[h];
            }
            s_output[d] += gate_w * acc;
        }
        __syncthreads();
    }

    // Write output to global memory
    for (int d = tid; d < input_dim; d += block_size) {
        output[sample_idx * input_dim + d] = s_output[d];
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: Dynamic Expert Backward
//
//  Backward through dynamic expert loading. Computes gradients for
//  gate_logits, W1, b1, W2, b2 for active experts only. Uses shared
//  memory reduction for weight gradient accumulation.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_dynamic_expert_bwd_kernel(
    const float* __restrict__ grad_output,       // [N, input_dim]
    const float* __restrict__ input,             // [N, input_dim]
    const float* __restrict__ gate_logits,       // [N, num_experts]
    const float* __restrict__ loaded_W1,         // [N, MAX_TOPK, expert_dim, input_dim]
    const float* __restrict__ loaded_b1,         // [N, MAX_TOPK, expert_dim]
    const float* __restrict__ loaded_W2,         // [N, MAX_TOPK, input_dim, expert_dim]
    const float* __restrict__ loaded_b2,         // [N, MAX_TOPK, input_dim]
    const int* __restrict__ active_expert_ids,   // [N, MAX_TOPK]
    const int* __restrict__ num_active_per_sample, // [N]
    float* __restrict__ grad_gate_logits,        // [N, num_experts]
    float* __restrict__ grad_W1,                 // [num_experts, expert_dim, input_dim]
    float* __restrict__ grad_b1,                 // [num_experts, expert_dim]
    float* __restrict__ grad_W2,                 // [num_experts, input_dim, expert_dim]
    float* __restrict__ grad_b2,                 // [num_experts, input_dim]
    const int N,
    const int num_experts,
    const int input_dim,
    const int expert_dim
) {
    // Shared memory layout:
    //   s_input[input_dim], s_grad_out[input_dim], s_hidden[expert_dim],
    //   s_gate_softmax[MAX_TOPK], s_expert_output[MAX_TOPK * input_dim]
    //   s_grad_w_reduce[expert_dim * input_dim] — for weight grad accumulation
    extern __shared__ float smem[];
    float* s_input = smem;
    float* s_grad_out = s_input + input_dim;
    float* s_hidden = s_grad_out + input_dim;
    float* s_gate_softmax = s_hidden + expert_dim;
    float* s_expert_output = s_gate_softmax + MAX_TOPK;
    float* s_grad_w_reduce = s_expert_output + MAX_TOPK * input_dim;

    const int sample_idx = blockIdx.x;
    if (sample_idx >= N) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int n_active = num_active_per_sample[sample_idx];

    if (n_active == 0) return;

    // Load input and grad_output into shared memory
    for (int d = tid; d < input_dim; d += block_size) {
        s_input[d] = input[sample_idx * input_dim + d];
        s_grad_out[d] = grad_output[sample_idx * input_dim + d];
    }
    __syncthreads();

    // Recompute gate softmax and per-expert outputs for gate gradient
    if (tid == 0) {
        float max_logit = -1e30f;
        for (int a = 0; a < n_active; a++) {
            int eid = active_expert_ids[sample_idx * MAX_TOPK + a];
            float logit = gate_logits[sample_idx * num_experts + eid];
            if (logit > max_logit) max_logit = logit;
        }
        float sum_exp = 0.0f;
        for (int a = 0; a < n_active; a++) {
            int eid = active_expert_ids[sample_idx * MAX_TOPK + a];
            float logit = gate_logits[sample_idx * num_experts + eid];
            float exp_val = expf(logit - max_logit);
            s_gate_softmax[a] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int a = 0; a < n_active; a++) {
            s_gate_softmax[a] *= inv_sum;
        }
    }
    __syncthreads();

    // Forward recomputation and backward for each active expert
    for (int a = 0; a < n_active; a++) {
        int expert_id = active_expert_ids[sample_idx * MAX_TOPK + a];
        float gate_w = s_gate_softmax[a];
        int base_w1 = (sample_idx * MAX_TOPK + a) * expert_dim * input_dim;
        int base_b1 = (sample_idx * MAX_TOPK + a) * expert_dim;
        int base_w2 = (sample_idx * MAX_TOPK + a) * input_dim * expert_dim;
        int base_b2 = (sample_idx * MAX_TOPK + a) * input_dim;

        // Recompute hidden = ReLU(W1 * input + b1)
        for (int h = tid; h < expert_dim; h += block_size) {
            float acc = loaded_b1[base_b1 + h];
            for (int d = 0; d < input_dim; d++) {
                acc += loaded_W1[base_w1 + h * input_dim + d] * s_input[d];
            }
            s_hidden[h] = fmaxf(0.0f, acc);
        }
        __syncthreads();

        // Compute per-expert output for gate gradient: expert_out = W2 * hidden + b2
        for (int d = tid; d < input_dim; d += block_size) {
            float acc = loaded_b2[base_b2 + d];
            for (int h = 0; h < expert_dim; h++) {
                acc += loaded_W2[base_w2 + d * expert_dim + h] * s_hidden[h];
            }
            s_expert_output[a * input_dim + d] = acc;
        }
        __syncthreads();

        // grad_W2: outer product of grad_output (scaled by gate) and hidden
        // Uses shared memory reduction
        for (int i = tid; i < input_dim * expert_dim; i += block_size) {
            int d = i / expert_dim;
            int h = i % expert_dim;
            float g = gate_w * s_grad_out[d] * s_hidden[h];
            s_grad_w_reduce[i] = g;
        }
        __syncthreads();

        // Atomically accumulate W2 gradients to global memory
        for (int i = tid; i < input_dim * expert_dim; i += block_size) {
            atomicAdd(&grad_W2[expert_id * input_dim * expert_dim + i], s_grad_w_reduce[i]);
        }

        // grad_b2: sum of grad_output scaled by gate weight
        for (int d = tid; d < input_dim; d += block_size) {
            atomicAdd(&grad_b2[expert_id * input_dim + d], gate_w * s_grad_out[d]);
        }

        // Backprop through W2 to get grad_hidden
        // grad_hidden[h] = sum_d(gate_w * grad_out[d] * W2[d, h]) * relu_mask
        for (int h = tid; h < expert_dim; h += block_size) {
            float acc = 0.0f;
            for (int d = 0; d < input_dim; d++) {
                acc += s_grad_out[d] * loaded_W2[base_w2 + d * expert_dim + h];
            }
            float relu_mask = (s_hidden[h] > 0.0f) ? 1.0f : 0.0f;
            float grad_h = gate_w * acc * relu_mask;

            // grad_b1
            atomicAdd(&grad_b1[expert_id * expert_dim + h], grad_h);

            // grad_W1: outer product of grad_hidden and input
            for (int d = 0; d < input_dim; d++) {
                atomicAdd(&grad_W1[expert_id * expert_dim * input_dim + h * input_dim + d],
                          grad_h * s_input[d]);
            }
        }
        __syncthreads();
    }

    // Compute gate gradient: d_loss/d_gate_logit_i
    // For softmax: d_loss/d_z_i = sum_j (g_i * (delta_ij - g_j)) * dot(grad_out, expert_out_j)
    for (int a = tid; a < n_active; a += block_size) {
        int eid = active_expert_ids[sample_idx * MAX_TOPK + a];
        float g_a = s_gate_softmax[a];

        // Compute dot(grad_output, expert_output_a)
        float dot_a = 0.0f;
        for (int d = 0; d < input_dim; d++) {
            dot_a += s_grad_out[d] * s_expert_output[a * input_dim + d];
        }

        // Compute weighted sum of all dots
        float weighted_sum = 0.0f;
        for (int b = 0; b < n_active; b++) {
            float dot_b = 0.0f;
            for (int d = 0; d < input_dim; d++) {
                dot_b += s_grad_out[d] * s_expert_output[b * input_dim + d];
            }
            weighted_sum += s_gate_softmax[b] * dot_b;
        }

        // Softmax gradient: g_a * (dot_a - weighted_sum)
        grad_gate_logits[sample_idx * num_experts + eid] = g_a * (dot_a - weighted_sum);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: Filter Active Parameters
//
//  Given expert assignments (from routing), create a compacted array of
//  only the parameters that are actively being updated by selected experts.
//  Output: compacted param/grad/state arrays + scatter indices.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_filter_active_params_kernel(
    const float* __restrict__ params,          // [total_params]
    const float* __restrict__ grads,           // [total_params]
    const float* __restrict__ state_m,         // [total_params] — optimizer 1st moment
    const float* __restrict__ state_v,         // [total_params] — optimizer 2nd moment
    const int* __restrict__ param_to_expert,   // [total_params] — expert assignment per param
    const int* __restrict__ expert_active,     // [num_experts] — 1 if expert is active, 0 otherwise
    float* __restrict__ compact_params,        // [max_active] — compacted output
    float* __restrict__ compact_grads,         // [max_active]
    float* __restrict__ compact_state_m,       // [max_active]
    float* __restrict__ compact_state_v,       // [max_active]
    int* __restrict__ scatter_indices,         // [max_active] — maps compact idx -> original idx
    int* __restrict__ compact_count,           // [1] — atomic counter for compacted size
    const int total_params
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < total_params; i += stride) {
        int expert_id = param_to_expert[i];
        if (expert_id >= 0 && expert_active[expert_id]) {
            // Atomically claim a slot in the compacted array
            int slot = atomicAdd(compact_count, 1);

            // Load with streaming hints (optimizer state is accessed once)
            compact_params[slot] = LDG(&params[i]);
            compact_grads[slot] = LDG(&grads[i]);
            compact_state_m[slot] = stream_load(&state_m[i]);
            compact_state_v[slot] = stream_load(&state_v[i]);
            scatter_indices[slot] = i;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 5: Mamba Scan on Compacted Parameters
//
//  Run Mamba scan only on the compacted (active) parameter subset.
//  Uses affine_combine_ptx() for Affine2x2 composition.
//  Identical scan logic to parallel_scan but on smaller N.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_scan_compacted_kernel(
    const float* __restrict__ compact_x,        // [compact_N, d_inner] — projected input
    const float* __restrict__ compact_dt,       // [compact_N, d_inner] — delta t
    const float* __restrict__ compact_B,        // [compact_N, d_state]
    const float* __restrict__ compact_C,        // [compact_N, d_state]
    const float* __restrict__ A_log,            // [d_inner, d_state]
    const float* __restrict__ D_param,          // [d_inner]
    const float* __restrict__ rope_freq,        // [d_inner, d_state/2]
    float* __restrict__ scan_output,            // [compact_N, d_inner]
    float* __restrict__ final_state,            // [d_inner, d_state]
    const float* __restrict__ initial_state,    // [d_inner, d_state] or nullptr
    const int compact_N,
    const int d_inner,
    const int d_state
) {
    const int j = blockIdx.x;  // d_inner index for this block
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory: Affine2x2 array for Blelloch scan [num_threads * 6 floats]
    extern __shared__ float smem[];

    const int chunk_size = (compact_N + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, compact_N);
    const int my_count = max(my_end - my_start, 0);

    const int half_d_state = d_state / 2;

    // Load per-d_inner constants
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[j * d_state + s]);
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[j * half_d_state + p];
    float D_val = D_param[j];

    // Load initial state
    float h_init_all[MAX_D_STATE];
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = initial_state[j * d_state + s];
    } else {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = 0.0f;
    }

    // Build affine transform for timestep t, pair p
    #define MOE_BUILD_AFFINE(t_idx, A_e, A_o, f_val, s_e, s_o, elem_out) do { \
        float dt_val = compact_dt[(t_idx) * d_inner + j]; \
        float x_val = compact_x[(t_idx) * d_inner + j]; \
        float B_e = compact_B[(t_idx) * d_state + (s_e)]; \
        float B_o = compact_B[(t_idx) * d_state + (s_o)]; \
        float A_bar_e = (1.0f + dt_val * (A_e) / 2.0f) / (1.0f - dt_val * (A_e) / 2.0f + 1e-8f); \
        float A_bar_o = (1.0f + dt_val * (A_o) / 2.0f) / (1.0f - dt_val * (A_o) / 2.0f + 1e-8f); \
        float cos_v, sin_v; \
        FAST_SINCOSF(dt_val * (f_val), &sin_v, &cos_v); \
        (elem_out).m00 = A_bar_e * cos_v; \
        (elem_out).m01 = -A_bar_e * sin_v; \
        (elem_out).m10 = A_bar_o * sin_v; \
        (elem_out).m11 = A_bar_o * cos_v; \
        (elem_out).b0 = dt_val * B_e * x_val; \
        (elem_out).b1 = dt_val * B_o * x_val; \
    } while(0)

    // Process each state pair
    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq[p];
        const float h_init_e = h_init_all[s_e];
        const float h_init_o = h_init_all[s_o];

        // Step 1: Sequential scan within chunk -> chunk summary
        Affine2x2 summary = affine_identity();
        for (int step = 0; step < my_count; step++) {
            int t = my_start + step;
            Affine2x2 elem;
            MOE_BUILD_AFFINE(t, A_e, A_o, f_val, s_e, s_o, elem);
            summary = affine_combine_ptx(summary, elem);
        }

        // Store summary in shared memory
        int base = ltid * 6;
        smem[base + 0] = summary.m00; smem[base + 1] = summary.m01;
        smem[base + 2] = summary.m10; smem[base + 3] = summary.m11;
        smem[base + 4] = summary.b0;  smem[base + 5] = summary.b1;
        __syncthreads();

        // Step 2: Blelloch exclusive prefix scan on chunk summaries
        // Up-sweep (reduction)
        for (int stride = 1; stride < num_threads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 combined = affine_combine_ptx(left, right);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) {
                __syncthreads();
            }
        }

        // Set last element to identity (exclusive scan)
        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            smem[last]   = 1.0f; smem[last+1] = 0.0f;
            smem[last+2] = 0.0f; smem[last+3] = 1.0f;
            smem[last+4] = 0.0f; smem[last+5] = 0.0f;
        }
        __syncthreads();

        // Down-sweep
        for (int stride = num_threads / 2; stride >= 1; stride /= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                // Swap and combine
                smem[(idx-stride)*6]   = right.m00; smem[(idx-stride)*6+1] = right.m01;
                smem[(idx-stride)*6+2] = right.m10; smem[(idx-stride)*6+3] = right.m11;
                smem[(idx-stride)*6+4] = right.b0;  smem[(idx-stride)*6+5] = right.b1;
                Affine2x2 combined = affine_combine_ptx(right, left);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) {
                __syncthreads();
            }
        }

        // Step 3: Apply prefix to each element within chunk, compute scan output
        Affine2x2 prefix = {smem[ltid*6], smem[ltid*6+1],
                            smem[ltid*6+2], smem[ltid*6+3],
                            smem[ltid*6+4], smem[ltid*6+5]};

        // Apply initial state to prefix bias
        float h_e = prefix.m00 * h_init_e + prefix.m01 * h_init_o + prefix.b0;
        float h_o = prefix.m10 * h_init_e + prefix.m11 * h_init_o + prefix.b1;

        // Sequential scan within chunk using prefix
        for (int step = 0; step < my_count; step++) {
            int t = my_start + step;
            Affine2x2 elem;
            MOE_BUILD_AFFINE(t, A_e, A_o, f_val, s_e, s_o, elem);

            float new_h_e = elem.m00 * h_e + elem.m01 * h_o + elem.b0;
            float new_h_o = elem.m10 * h_e + elem.m11 * h_o + elem.b1;
            h_e = new_h_e;
            h_o = new_h_o;

            // Accumulate scan output: y[t] += C[t,s] * h[s]
            float C_e = compact_C[t * d_state + s_e];
            float C_o = compact_C[t * d_state + s_o];
            // Atomic add since multiple pairs contribute to same output
            atomicAdd(&scan_output[t * d_inner + j], C_e * h_e + C_o * h_o);
        }

        // Store final state for this pair
        if (ltid == num_threads - 1 || (my_count > 0 && my_end == compact_N)) {
            if (my_end == compact_N) {
                final_state[j * d_state + s_e] = h_e;
                final_state[j * d_state + s_o] = h_o;
            }
        }
        __syncthreads();
    }

    #undef MOE_BUILD_AFFINE

    // Add D * x[t] skip connection
    for (int step = 0; step < my_count; step++) {
        int t = my_start + step;
        float x_val = compact_x[t * d_inner + j];
        atomicAdd(&scan_output[t * d_inner + j], D_val * x_val);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 6: Scatter Compacted Results
//
//  Scatter the compacted scan results back to the full parameter array
//  using the scatter indices from kernel 4.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_scatter_results_kernel(
    const float* __restrict__ compact_params,    // [compact_N] — updated params
    const float* __restrict__ compact_state_m,   // [compact_N] — updated 1st moment
    const float* __restrict__ compact_state_v,   // [compact_N] — updated 2nd moment
    const int* __restrict__ scatter_indices,     // [compact_N] — original indices
    float* __restrict__ params,                  // [total_params] — full param array
    float* __restrict__ state_m,                 // [total_params] — full 1st moment
    float* __restrict__ state_v,                 // [total_params] — full 2nd moment
    const int compact_N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < compact_N; i += stride) {
        int orig_idx = scatter_indices[i];

        // Write back updated values using streaming stores to avoid cache pollution
        float p = compact_params[i];
        float m = compact_state_m[i];
        float v = compact_state_v[i];

        params[orig_idx] = p;
        stream_store(&state_m[orig_idx], m);
        stream_store(&state_v[orig_idx], v);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 7: Count Expert Activations
//
//  Atomic counter: for each parameter, increment the count of which
//  expert was selected. Input: gate_logits, threshold.
//  Output: expert_counts[num_experts] array.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_count_expert_activations_kernel(
    const float* __restrict__ gate_logits,  // [N, num_experts]
    int* __restrict__ expert_counts,        // [num_experts] — output, must be pre-zeroed
    const float threshold,
    const int N,
    const int num_experts
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Use shared memory for block-local histogram to reduce global atomic contention
    extern __shared__ int s_counts[];

    // Initialize shared histogram
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        s_counts[e] = 0;
    }
    __syncthreads();

    // Each thread processes multiple parameters
    for (int i = tid; i < N; i += stride) {
        // Find which experts are active for this parameter
        for (int e = 0; e < num_experts; e++) {
            float logit = gate_logits[i * num_experts + e];
            if (logit > threshold) {
                atomicAdd(&s_counts[e], 1);
            }
        }
    }
    __syncthreads();

    // Flush shared histogram to global memory
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        if (s_counts[e] > 0) {
            atomicAdd(&expert_counts[e], s_counts[e]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 8: Compute Load Balance Loss
//
//  Compute auxiliary load-balancing loss from expert_counts.
//    L_balance = num_experts * sum(f_i * P_i)
//  where f_i = count_i / N, P_i = mean(gate_prob_i over all samples).
//  Uses shared memory reduction.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_compute_load_balance_loss_kernel(
    const int* __restrict__ expert_counts,   // [num_experts]
    const float* __restrict__ gate_logits,   // [N, num_experts]
    float* __restrict__ loss_out,            // [1] — scalar output
    const int N,
    const int num_experts
) {
    // Shared memory for reduction
    extern __shared__ float smem[];
    // s_gate_prob_sum[num_experts] — sum of gate probs per expert
    float* s_gate_prob_sum = smem;
    // s_partial_loss[blockDim.x] — partial loss per thread
    float* s_partial_loss = smem + num_experts;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Step 1: Compute mean gate probability per expert P_i = (1/N) * sum_n softmax(gate_logits[n])_i
    // Initialize accumulator
    for (int e = tid; e < num_experts; e += block_size) {
        s_gate_prob_sum[e] = 0.0f;
    }
    __syncthreads();

    // Each thread processes a subset of samples
    for (int n = tid; n < N; n += block_size) {
        // Compute softmax for this sample
        float max_logit = -1e30f;
        for (int e = 0; e < num_experts; e++) {
            float logit = gate_logits[n * num_experts + e];
            if (logit > max_logit) max_logit = logit;
        }
        float sum_exp = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            sum_exp += expf(gate_logits[n * num_experts + e] - max_logit);
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);

        for (int e = 0; e < num_experts; e++) {
            float prob = expf(gate_logits[n * num_experts + e] - max_logit) * inv_sum;
            atomicAdd(&s_gate_prob_sum[e], prob);
        }
    }
    __syncthreads();

    // Step 2: Compute L_balance = num_experts * sum_i(f_i * P_i)
    float local_loss = 0.0f;
    float inv_N = 1.0f / (float)(N > 0 ? N : 1);
    for (int e = tid; e < num_experts; e += block_size) {
        float f_i = (float)expert_counts[e] * inv_N;    // fraction of tokens routed to expert i
        float P_i = s_gate_prob_sum[e] * inv_N;          // mean gate probability for expert i
        local_loss += f_i * P_i;
    }

    // Shared memory reduction for final loss
    s_partial_loss[tid] = local_loss;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial_loss[tid] += s_partial_loss[tid + s];
        }
        if (s >= WARP_SIZE) {
            __syncthreads();
        }
    }

    if (tid == 0) {
        loss_out[0] = (float)num_experts * s_partial_loss[0];
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 9: Apply Frequency-Based LR Scaling
//
//  Scale expert learning rates inversely proportional to activation
//  frequency. Underused experts get higher LR, overused get lower.
//  Modifies the per-expert lr_scale buffer.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void moe_apply_frequency_scaling_kernel(
    const int* __restrict__ expert_counts,   // [num_experts] — activation counts
    float* __restrict__ lr_scale,            // [num_experts] — per-expert LR scale (in/out)
    const int num_experts,
    const int total_activations,             // sum of all expert_counts
    const float min_scale,                   // minimum LR scale (e.g., 0.1)
    const float max_scale,                   // maximum LR scale (e.g., 10.0)
    const float smoothing                    // EMA smoothing factor (e.g., 0.9)
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Expected uniform frequency: each expert should get 1/num_experts of activations
    float expected_freq = 1.0f / (float)num_experts;
    float inv_total = 1.0f / (float)(total_activations > 0 ? total_activations : 1);

    for (int e = tid; e < num_experts; e += stride) {
        float actual_freq = (float)expert_counts[e] * inv_total;

        // Inverse frequency scaling: underused experts get boosted
        // scale = expected_freq / actual_freq, clamped to [min_scale, max_scale]
        float raw_scale = expected_freq / (actual_freq + 1e-8f);
        float clamped_scale = fminf(max_scale, fmaxf(min_scale, raw_scale));

        // Exponential moving average with previous scale for stability
        float prev_scale = lr_scale[e];
        float new_scale = smoothing * prev_scale + (1.0f - smoothing) * clamped_scale;

        lr_scale[e] = new_scale;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ Launcher Functions for MoE Deep Kernels
// ═══════════════════════════════════════════════════════════════════════

void moe_dynamic_expert_load(
    torch::Tensor gate_logits, torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor loaded_W1, torch::Tensor loaded_b1,
    torch::Tensor loaded_W2, torch::Tensor loaded_b2,
    torch::Tensor active_expert_ids, torch::Tensor num_active_per_sample,
    float threshold, int N, int num_experts, int input_dim, int expert_dim
) {
    dim3 grid(N);
    dim3 block(256);
    int smem = (num_experts + MAX_TOPK + 1) * sizeof(int);

    moe_dynamic_expert_load_kernel<<<grid, block, smem>>>(
        gate_logits.data_ptr<float>(), W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        loaded_W1.data_ptr<float>(), loaded_b1.data_ptr<float>(),
        loaded_W2.data_ptr<float>(), loaded_b2.data_ptr<float>(),
        active_expert_ids.data_ptr<int>(), num_active_per_sample.data_ptr<int>(),
        threshold, N, num_experts, input_dim, expert_dim
    );
}

torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor input, torch::Tensor gate_logits,
    torch::Tensor loaded_W1, torch::Tensor loaded_b1,
    torch::Tensor loaded_W2, torch::Tensor loaded_b2,
    torch::Tensor active_expert_ids, torch::Tensor num_active_per_sample,
    int N, int num_experts, int input_dim, int expert_dim
) {
    auto output = torch::zeros({N, input_dim}, input.options());
    dim3 grid(N);
    dim3 block(256);
    int smem = (input_dim + expert_dim + MAX_TOPK + input_dim) * sizeof(float);

    moe_dynamic_expert_fwd_kernel<<<grid, block, smem>>>(
        input.data_ptr<float>(), gate_logits.data_ptr<float>(),
        loaded_W1.data_ptr<float>(), loaded_b1.data_ptr<float>(),
        loaded_W2.data_ptr<float>(), loaded_b2.data_ptr<float>(),
        active_expert_ids.data_ptr<int>(), num_active_per_sample.data_ptr<int>(),
        output.data_ptr<float>(), N, num_experts, input_dim, expert_dim
    );
    return output;
}

void moe_dynamic_expert_bwd(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor gate_logits,
    torch::Tensor loaded_W1, torch::Tensor loaded_b1,
    torch::Tensor loaded_W2, torch::Tensor loaded_b2,
    torch::Tensor active_expert_ids, torch::Tensor num_active_per_sample,
    torch::Tensor grad_gate_logits, torch::Tensor grad_W1, torch::Tensor grad_b1,
    torch::Tensor grad_W2, torch::Tensor grad_b2,
    int N, int num_experts, int input_dim, int expert_dim
) {
    dim3 grid(N);
    dim3 block(256);
    int smem = (input_dim + input_dim + expert_dim + MAX_TOPK +
                MAX_TOPK * input_dim + expert_dim * input_dim) * sizeof(float);

    moe_dynamic_expert_bwd_kernel<<<grid, block, smem>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        gate_logits.data_ptr<float>(),
        loaded_W1.data_ptr<float>(), loaded_b1.data_ptr<float>(),
        loaded_W2.data_ptr<float>(), loaded_b2.data_ptr<float>(),
        active_expert_ids.data_ptr<int>(), num_active_per_sample.data_ptr<int>(),
        grad_gate_logits.data_ptr<float>(),
        grad_W1.data_ptr<float>(), grad_b1.data_ptr<float>(),
        grad_W2.data_ptr<float>(), grad_b2.data_ptr<float>(),
        N, num_experts, input_dim, expert_dim
    );
}

void moe_filter_active_params(
    torch::Tensor params, torch::Tensor grads,
    torch::Tensor state_m, torch::Tensor state_v,
    torch::Tensor param_to_expert, torch::Tensor expert_active,
    torch::Tensor compact_params, torch::Tensor compact_grads,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices, torch::Tensor compact_count,
    int total_params
) {
    int block = 256;
    int grid = (total_params + block - 1) / block;

    moe_filter_active_params_kernel<<<grid, block>>>(
        params.data_ptr<float>(), grads.data_ptr<float>(),
        state_m.data_ptr<float>(), state_v.data_ptr<float>(),
        param_to_expert.data_ptr<int>(), expert_active.data_ptr<int>(),
        compact_params.data_ptr<float>(), compact_grads.data_ptr<float>(),
        compact_state_m.data_ptr<float>(), compact_state_v.data_ptr<float>(),
        scatter_indices.data_ptr<int>(), compact_count.data_ptr<int>(),
        total_params
    );
}

void moe_scan_compacted(
    torch::Tensor compact_x, torch::Tensor compact_dt,
    torch::Tensor compact_B, torch::Tensor compact_C,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int compact_N, int d_inner, int d_state
) {
    dim3 grid(d_inner);
    dim3 block(16);
    int smem = 16 * 6 * sizeof(float);
    const float* init_ptr = (initial_state.numel() > 0) ? initial_state.data_ptr<float>() : nullptr;

    moe_scan_compacted_kernel<<<grid, block, smem>>>(
        compact_x.data_ptr<float>(), compact_dt.data_ptr<float>(),
        compact_B.data_ptr<float>(), compact_C.data_ptr<float>(),
        A_log.data_ptr<float>(), D_param.data_ptr<float>(),
        rope_freq.data_ptr<float>(),
        scan_output.data_ptr<float>(), final_state.data_ptr<float>(),
        init_ptr, compact_N, d_inner, d_state
    );
}

void moe_scatter_results(
    torch::Tensor compact_params, torch::Tensor compact_state_m,
    torch::Tensor compact_state_v, torch::Tensor scatter_indices,
    torch::Tensor params, torch::Tensor state_m, torch::Tensor state_v,
    int compact_N
) {
    int block = 256;
    int grid = (compact_N + block - 1) / block;

    moe_scatter_results_kernel<<<grid, block>>>(
        compact_params.data_ptr<float>(), compact_state_m.data_ptr<float>(),
        compact_state_v.data_ptr<float>(), scatter_indices.data_ptr<int>(),
        params.data_ptr<float>(), state_m.data_ptr<float>(),
        state_v.data_ptr<float>(), compact_N
    );
}

void moe_count_expert_activations(
    torch::Tensor gate_logits, torch::Tensor expert_counts,
    float threshold, int N, int num_experts
) {
    int block = 256;
    int grid = (N + block - 1) / block;
    int smem = num_experts * sizeof(int);

    moe_count_expert_activations_kernel<<<grid, block, smem>>>(
        gate_logits.data_ptr<float>(), expert_counts.data_ptr<int>(),
        threshold, N, num_experts
    );
}

torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor expert_counts, torch::Tensor gate_logits,
    int N, int num_experts
) {
    auto loss_out = torch::zeros({1}, gate_logits.options());
    dim3 grid(1);
    dim3 block(256);
    int smem = (num_experts + 256) * sizeof(float);

    moe_compute_load_balance_loss_kernel<<<grid, block, smem>>>(
        expert_counts.data_ptr<int>(), gate_logits.data_ptr<float>(),
        loss_out.data_ptr<float>(), N, num_experts
    );
    return loss_out;
}

void moe_apply_frequency_scaling(
    torch::Tensor expert_counts, torch::Tensor lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing
) {
    int block = 256;
    int grid = (num_experts + block - 1) / block;

    moe_apply_frequency_scaling_kernel<<<grid, block>>>(
        expert_counts.data_ptr<int>(), lr_scale.data_ptr<float>(),
        num_experts, total_activations, min_scale, max_scale, smoothing
    );
}
