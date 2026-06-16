// tests/hw/_mb3_scalar_probe.cu — MINIMAL standalone Milestone-A.1/B.1 probe for
// the Mamba-3 SCALAR fwd+bwd (model_stage_mamba3.cuh). NOT production: a single
// CTA loops all B samples, runs mb_forward_sample + mb_backward_sample, and
// accumulates grads with atomicAdd into a flat fp32 grad buffer. This isolates
// the Mamba-3 scan/RMSNorm/BCNorm/SwiGLU/block math (fp32) for comparison vs the
// fp64 oracle WITHOUT needing the TC rewrite or the persistent megakernel. JIT-
// loaded by tests/hw/test_mb3_scalar.py. (Determinism is irrelevant here — this
// is a correctness probe; the production path's deterministic reduce is separate.)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "csrc/fused/sm_90/model_stage_mamba3.cuh"

namespace {
using namespace sg::fused::sm90;

// One CTA, 256 threads. Loops the B samples; per sample runs fwd then bwd,
// atomicAdd-ing the per-sample grads into `grad` (flat, kMambaTotalElems). The
// loss is summed (fp64-ish via atomicAdd fp32) into loss_out[0]; CE mean over B
// is applied by passing B to mb_backward_sample (the dlogits/B) and dividing the
// loss sum by B on the host.
__global__ void __launch_bounds__(256)
mb3_scalar_fwdbwd(const float* __restrict__ params,
                  const int* __restrict__ tokens,   // [B, kSeq]
                  const int* __restrict__ targets,  // [B]
                  int B, float* __restrict__ grad, float* __restrict__ loss_out,
                  float* __restrict__ dbg) {
    extern __shared__ char smem_raw[];
    MambaSampleSmem* sm = reinterpret_cast<MambaSampleSmem*>(smem_raw);
    MambaWeights w = mb_bind(params);
    MambaGrad g = mb_bind_grad(grad);
    for (int s = 0; s < B; ++s) {
        const int* tok_s = tokens + (int64_t)s * mb::kSeq;
        int tgt = targets[s];
        float nll = mb_forward_sample(w, tok_s, tgt, sm);
        // DEBUG (sample 0 only): dump block-0 out, final_in, logits, then block-0
        // y_scan + x_in (mixer internals) appended after.
        if (s == 0 && dbg != nullptr) {
            for (int i = threadIdx.x; i < mb::kSeq * mb::kD; i += blockDim.x) {
                dbg[i] = sm->layer_in[1][i / mb::kD][i % mb::kD];                 // block-0 out
                dbg[mb::kSeq * mb::kD + i] = sm->final_in[i / mb::kD][i % mb::kD]; // block-1 out
            }
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x)
                dbg[2 * mb::kSeq * mb::kD + o] = sm->logits[o];                    // logits
            const int base = 2 * mb::kSeq * mb::kD + mb::kPHead;
            for (int i = threadIdx.x; i < mb::kSeq * mb::kDInner; i += blockDim.x) {
                dbg[base + i] = sm->act[0].y_scan[i / mb::kDInner][i % mb::kDInner];
                dbg[base + mb::kSeq * mb::kDInner + i] = sm->act[0].x_in[i / mb::kDInner][i % mb::kDInner];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) atomicAdd(loss_out, nll);
        mb_backward_sample(w, g, tok_s, tgt, B, sm);
    }
}

std::vector<torch::Tensor> scalar_fwdbwd(torch::Tensor params,
                                         torch::Tensor tokens,
                                         torch::Tensor targets) {
    TORCH_CHECK(params.is_cuda() && params.dtype() == torch::kFloat32, "params fp32 cuda");
    const int B = (int)tokens.size(0);
    auto grad = torch::zeros_like(params);
    auto loss = torch::zeros({1}, params.options());
    namespace mbn = sg::fused::sm90::mb;
    const int dbg_n = 2 * mbn::kSeq * mbn::kD + mbn::kPHead
                      + 2 * mbn::kSeq * mbn::kDInner;
    auto dbg = torch::zeros({dbg_n}, params.options());
    auto tok32 = tokens.to(torch::kInt32).contiguous();
    auto tgt32 = targets.to(torch::kInt32).contiguous();
    int64_t smem = kMambaSmemBytes;
    cudaFuncSetAttribute(mb3_scalar_fwdbwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    mb3_scalar_fwdbwd<<<1, 256, smem>>>(
        params.data_ptr<float>(), tok32.data_ptr<int>(), tgt32.data_ptr<int>(),
        B, grad.data_ptr<float>(), loss.data_ptr<float>(), dbg.data_ptr<float>());
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "kernel launch: ", cudaGetErrorString(e));
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "kernel sync: ", cudaGetErrorString(e));
    return {loss / (double)B, grad, dbg};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scalar_fwdbwd", &scalar_fwdbwd, "Mamba-3 scalar fwd+bwd probe");
    m.attr("TOTAL") = (int64_t)sg::fused::sm90::kMambaTotalElems;
}
