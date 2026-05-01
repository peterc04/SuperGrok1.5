// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok2_warp_specialized.cu
//
//  sm_90 (Hopper) SuperGrok v2 — warp-specialized scan TU.
//
//  Kernel + launcher definitions; declarations live in
//  supergrok2_warp_specialized.cuh. The math is identical to the
//  deleted baseline (commit 5505b50^:supergrok2_warp_specialized_sm90.cu)
//  with the I/O surface narrowed to raw pointers + a stream so it can
//  be invoked from the canonical SG2 forward dispatcher without taking
//  a torch::Tensor dependency in the .cuh header.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/supergrok2_warp_specialized.cuh"

#if GROK_CUDA

namespace sg { namespace sm90 { namespace supergrok2 {

// ---------------------------------------------------------------------
//  Kernel 1: warp-specialized scan (generic d_state)
//
//  Block layout: warp 0 = producer (one cp.async-style staged load per
//  step), warp 1 = consumer (executes the SSM recurrence + atomicAdd
//  accumulation into scan_output). Warps 2-3 idle (reserved for future
//  parallel d_state pair offload). Atomic phase flags in smem implement
//  the producer/consumer handshake without cluster-level barriers.
// ---------------------------------------------------------------------
__launch_bounds__(WS_BLOCK_SIZE, 4)
__global__ void scan_warp_specialized_kernel(
    const float* __restrict__ pre_x,
    const float* __restrict__ pre_z,
    const float* __restrict__ pre_dt,
    const float* __restrict__ pre_B,
    const float* __restrict__ pre_C,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ scan_state,
    float* __restrict__ scan_output,
    int N, int d_inner, int d_state
) {
    const int pair_idx = blockIdx.x;
    const int half_ds = d_state / 2;
    const int di = pair_idx / half_ds;
    const int ds_pair = pair_idx % half_ds;
    const int ds0 = ds_pair * 2;
    const int ds1 = ds0 + 1;
    if (di >= d_inner) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float smem_x[WS_NUM_BUFFERS];
    __shared__ float smem_z[WS_NUM_BUFFERS];
    __shared__ float smem_dt[WS_NUM_BUFFERS];
    __shared__ float smem_B0[WS_NUM_BUFFERS];
    __shared__ float smem_B1[WS_NUM_BUFFERS];
    __shared__ float smem_C0[WS_NUM_BUFFERS];
    __shared__ float smem_C1[WS_NUM_BUFFERS];
    __shared__ int smem_ready[WS_NUM_BUFFERS];
    __shared__ int smem_consumed[WS_NUM_BUFFERS];

    if (threadIdx.x == 0) {
        smem_ready[0] = 0; smem_ready[1] = 0;
        smem_consumed[0] = 1; smem_consumed[1] = 1;
    }
    __syncthreads();

    const float A0 = -fast_exp_ptx(A_log[di * d_state + ds0]);
    const float A1 = -fast_exp_ptx(A_log[di * d_state + ds1]);
    const float D_val = D_param[di];
    float h0 = scan_state[di * d_state + ds0];
    float h1 = scan_state[di * d_state + ds1];
    const float rope_f = rope_freq[ds_pair];

    if (warp_id == WS_PRODUCER_WARP_ID && lane_id == 0) {
        // ---------------- producer ---------------------------------
        for (int t = 0; t < N; t++) {
            const int buf = t % WS_NUM_BUFFERS;
            // wait for buffer to be empty
            while (atomicAdd(&smem_consumed[buf], 0) == 0) {
                __nanosleep(32);
            }
            smem_x[buf]  = pre_x[t * d_inner + di];
            smem_z[buf]  = pre_z[t * d_inner + di];
            smem_dt[buf] = pre_dt[t * d_inner + di];
            smem_B0[buf] = pre_B[t * d_state + ds0];
            smem_B1[buf] = pre_B[t * d_state + ds1];
            smem_C0[buf] = pre_C[t * d_state + ds0];
            smem_C1[buf] = pre_C[t * d_state + ds1];
            __threadfence_block();
            atomicExch(&smem_consumed[buf], 0);
            atomicExch(&smem_ready[buf], 1);
        }
    } else if (warp_id == 1 && lane_id == 0) {
        // ---------------- consumer ---------------------------------
        for (int t = 0; t < N; t++) {
            const int buf = t % WS_NUM_BUFFERS;
            while (atomicAdd(&smem_ready[buf], 0) == 0) {
                __nanosleep(32);
            }
            const float x_val  = smem_x[buf];
            const float z_val  = smem_z[buf];
            const float dt_val = smem_dt[buf];
            const float B0v = smem_B0[buf];
            const float B1v = smem_B1[buf];
            const float C0v = smem_C0[buf];
            const float C1v = smem_C1[buf];
            __threadfence_block();
            atomicExch(&smem_ready[buf], 0);
            atomicExch(&smem_consumed[buf], 1);

            const float dA0 = fast_exp_ptx(A0 * dt_val);
            const float dA1 = fast_exp_ptx(A1 * dt_val);
            const float dBx0 = B0v * x_val * dt_val;
            const float dBx1 = B1v * x_val * dt_val;
            h0 = dA0 * h0 + dBx0;
            h1 = dA1 * h1 + dBx1;

            float s_, c_;
            FAST_SINCOSF(rope_f * (float)t, &s_, &c_);
            const float h0r = h0 * c_ - h1 * s_;
            const float h1r = h0 * s_ + h1 * c_;

            const float y = C0v * h0r + C1v * h1r + D_val * x_val;
            const float silu_z = z_val / (1.0f + __expf(-z_val));
            atomicAdd(&scan_output[t * d_inner + di], y * silu_z);
        }
        scan_state[di * d_state + ds0] = h0;
        scan_state[di * d_state + ds1] = h1;
    }
}

// ---------------------------------------------------------------------
//  Kernel 2: warp-specialized scan, d_state=16 unrolled
//
//  All 8 RoPE pairs live in registers in the consumer warp. The
//  producer distributes B/C loads across lanes (lane_id < 16) to issue
//  16 parallel global reads per timestep.
// ---------------------------------------------------------------------
__launch_bounds__(WS_BLOCK_SIZE, 4)
__global__ void scan_warp_specialized_d16_kernel(
    const float* __restrict__ pre_x,
    const float* __restrict__ pre_z,
    const float* __restrict__ pre_dt,
    const float* __restrict__ pre_B,
    const float* __restrict__ pre_C,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ scan_state,
    float* __restrict__ scan_output,
    int N, int d_inner
) {
    constexpr int DSTATE = 16;
    constexpr int DPAIRS = 8;
    const int di = blockIdx.x;
    if (di >= d_inner) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float smem_x[WS_NUM_BUFFERS];
    __shared__ float smem_z[WS_NUM_BUFFERS];
    __shared__ float smem_dt[WS_NUM_BUFFERS];
    __shared__ float smem_B[WS_NUM_BUFFERS][DSTATE];
    __shared__ float smem_C[WS_NUM_BUFFERS][DSTATE];
    __shared__ int   smem_phase[WS_NUM_BUFFERS]; // 0=empty, 1=full

    if (threadIdx.x < WS_NUM_BUFFERS) smem_phase[threadIdx.x] = 0;
    __syncthreads();

    float A_vals[DSTATE];
    #pragma unroll
    for (int s = 0; s < DSTATE; s++)
        A_vals[s] = -fast_exp_ptx(A_log[di * DSTATE + s]);
    const float D_val = D_param[di];

    float h[DSTATE];
    #pragma unroll
    for (int s = 0; s < DSTATE; s++) h[s] = scan_state[di * DSTATE + s];

    float rope_f[DPAIRS];
    #pragma unroll
    for (int p = 0; p < DPAIRS; p++) rope_f[p] = rope_freq[p];

    if (warp_id == WS_PRODUCER_WARP_ID) {
        for (int t = 0; t < N; t++) {
            const int buf = t % WS_NUM_BUFFERS;
            while (atomicAdd(&smem_phase[buf], 0) != 0) {
                __nanosleep(32);
            }
            if (lane_id == 0) {
                smem_x[buf]  = pre_x[t * d_inner + di];
                smem_z[buf]  = pre_z[t * d_inner + di];
                smem_dt[buf] = pre_dt[t * d_inner + di];
            }
            if (lane_id < DSTATE) {
                smem_B[buf][lane_id] = pre_B[t * DSTATE + lane_id];
                smem_C[buf][lane_id] = pre_C[t * DSTATE + lane_id];
            }
            __threadfence_block();
            if (lane_id == 0) atomicExch(&smem_phase[buf], 1);
        }
    } else if (warp_id == 1) {
        for (int t = 0; t < N; t++) {
            const int buf = t % WS_NUM_BUFFERS;
            if (lane_id == 0) {
                while (atomicAdd(&smem_phase[buf], 0) != 1) {
                    __nanosleep(32);
                }
            }
            __syncwarp();

            const float x_val  = smem_x[buf];
            const float z_val  = smem_z[buf];
            const float dt_val = smem_dt[buf];

            float y_acc = 0.0f;
            #pragma unroll
            for (int p = 0; p < DPAIRS; p++) {
                const int s0 = p * 2;
                const int s1 = s0 + 1;
                const float dA0 = fast_exp_ptx(A_vals[s0] * dt_val);
                const float dA1 = fast_exp_ptx(A_vals[s1] * dt_val);
                const float dBx0 = smem_B[buf][s0] * x_val * dt_val;
                const float dBx1 = smem_B[buf][s1] * x_val * dt_val;
                h[s0] = dA0 * h[s0] + dBx0;
                h[s1] = dA1 * h[s1] + dBx1;
                float s_, c_;
                FAST_SINCOSF(rope_f[p] * (float)t, &s_, &c_);
                const float h0r = h[s0] * c_ - h[s1] * s_;
                const float h1r = h[s0] * s_ + h[s1] * c_;
                y_acc += smem_C[buf][s0] * h0r + smem_C[buf][s1] * h1r;
            }

            __threadfence_block();
            if (lane_id == 0) atomicExch(&smem_phase[buf], 0);

            y_acc += D_val * x_val;
            const float silu_z = z_val / (1.0f + __expf(-z_val));
            if (lane_id == 0) {
                scan_output[t * d_inner + di] = y_acc * silu_z;
            }
        }
        if (lane_id == 0) {
            #pragma unroll
            for (int s = 0; s < DSTATE; s++)
                scan_state[di * DSTATE + s] = h[s];
        }
    }
}

// ---------------------------------------------------------------------
//  Host launchers
// ---------------------------------------------------------------------
void launch_scan_warp_specialized(
    const float* pre_x, const float* pre_z, const float* pre_dt,
    const float* pre_B, const float* pre_C,
    const float* A_log, const float* D_param, const float* rope_freq,
    float* scan_state, float* scan_output,
    int N, int d_inner, int d_state,
    GpuStream_t stream
) {
    // Caller must pre-zero scan_output (the kernel uses atomicAdd).
    const int num_pairs = d_inner * (d_state / 2);
    dim3 grid(num_pairs);
    dim3 block(WS_BLOCK_SIZE);
    scan_warp_specialized_kernel<<<grid, block, 0, stream>>>(
        pre_x, pre_z, pre_dt, pre_B, pre_C,
        A_log, D_param, rope_freq,
        scan_state, scan_output,
        N, d_inner, d_state);
}

void launch_scan_warp_specialized_d16(
    const float* pre_x, const float* pre_z, const float* pre_dt,
    const float* pre_B, const float* pre_C,
    const float* A_log, const float* D_param, const float* rope_freq,
    float* scan_state, float* scan_output,
    int N, int d_inner,
    GpuStream_t stream
) {
    dim3 grid(d_inner);
    dim3 block(WS_BLOCK_SIZE);
    scan_warp_specialized_d16_kernel<<<grid, block, 0, stream>>>(
        pre_x, pre_z, pre_dt, pre_B, pre_C,
        A_log, D_param, rope_freq,
        scan_state, scan_output,
        N, d_inner);
}

}}} // namespace sg::sm90::supergrok2

#endif // GROK_CUDA
