#ifndef SG_FUSED_SM90_SHARDED_OPTIMIZER_KERNEL_CUH_
#define SG_FUSED_SM90_SHARDED_OPTIMIZER_KERNEL_CUH_
// ============================================================================
// csrc/fused/sm_90/sharded_optimizer_kernel.cuh — the POST-reduce-scatter
// sharded optimizer kernel for the ZeRO-2/3 multi-GPU step.
//
// IMPLEMENTS: /workspace/.parallelism_design.md §2.3 (the new, small sharded-
// optimizer kernel) — the second half of the owner's B2-seam decomposition
// (design §0.1/§2.1):
//     [fwd+bwd megakernel] → reduce-scatter(grad) → [THIS kernel] → all-gather(params)
//
// WHAT IT IS: under ZeRO>=2 the fused megakernel early-exits right after the B2
// barrier (design §2.2, `if constexpr (Par::kShardOptGrad) return;`) with the
// reduced LOCAL grad assembled. The host then reduce-scatters that grad so each
// rank holds the reduced grad for ONLY its owned shard. THIS kernel applies the
// optimizer over that owned shard. For the ELEMENTWISE optimizers it is exactly
// P3 (the fused tail) with the work-steal-over-30-tensors loop replaced by a
// flat grid-stride over [0, shard_numel) and NO GridBarrier (single phase).
//
// ZERO NEW MATH (design §2.3, the parity guarantee): every per-element update is
// the EXACT SAME `apply_optimizer<Opt>` from opt_components.cuh — the canonical
// `csrc/algorithms/<opt>.h` device math the in-kernel P3 tail already calls and
// the parity gates already validate. This file INCLUDES opt_components.cuh and
// reuses it verbatim; it does NOT edit it and adds no new optimizer math, so the
// sharded path inherits the single-GPU A/A/A correctness per element (design
// §7.3: the DP=1 sharded kernel must be bit-identical to the in-kernel P3).
//
// ──────────────────────────────────────────────────────────────────────────
//  THE PER-TENSOR BOUNDARY (design §2.3 / §3.4 / §0.4) — read before using this
//  for muon / SuperGrok11 / SuperGrok15 / SuperGrok2:
//
//  `apply_optimizer<Opt>` is, for ALL 11 OptIds, a PURELY ELEMENTWISE update of
//  one `idx`: it consumes ONLY per-element state (exp_avg/exp_avg_sq/ema/...) and
//  PRE-COMPUTED inputs carried in FusedOptState (st.orth, st.mu, st.gate,
//  st.smart_grad, st.sam_dir, st.d_factor, st.clip_coef, ...). So this flat
//  grid-stride kernel COMPILES and RUNS correctly for any OptId *whose
//  FusedOptState inputs are already filled* — i.e. it is a complete, correct
//  optimizer step for:
//
//    ELEMENTWISE-DRIVABLE (this kernel alone is sufficient at DP):
//      adamw(0), lion(1), grokfast(2), neuralgrok(6),
//      and the ELEMENTWISE CORE of grokadamw(3)/looksam(4)/prodigy(5)
//      *given* their global/precomputed scalar already in st (clip_coef /
//      sam_dir / d_factor — see the next paragraph for where those come from).
//
//    PER-TENSOR / PER-MATRIX (this kernel is the APPLY ONLY — its st inputs need
//    an UPSTREAM stage that is OUT OF SCOPE for this kernel):
//      muon(7)        → st.orth     : Newton–Schulz over a WHOLE 2D weight (P2.7)
//      supergrok11(8) → st.mu/gate  : per-TENSOR meta-net mu/gate (P2.45)
//      supergrok15(9) → st.mu/gate  : per-TENSOR meta-net mu     (P2.45)
//      supergrok2(10) → st.smart_grad: per-TENSOR CSA/HCA/PEER/GRU + segmented
//                                      sort meta-net (STAGE -1 + P3-SG2)
//    Those P2.x stages cooperate across CTAs PER matrix/tensor on the GridBarrier
//    substrate, so they REQUIRE the tensor whole on one rank — which forces the
//    ZeRO-3 shard to be TENSOR-GRANULAR for these cells (design §3.4;
//    grokking_optimizers/parallel/shard_map.py::partition_tensor_granular). The
//    sharded-optimizer kernel FOR THEM is the *full* persistent megakernel
//    restricted to the tensors this rank owns (design §2.3) — NOT this flat
//    kernel. This flat kernel deliberately does NOT run those P2.x stages; it
//    only does the elementwise apply once their st inputs exist.
//
//  Likewise the GLOBAL-scalar cells (prodigy d, grokadamw grad-norm clip_coef,
//  looksam ‖g‖) need a cross-DP all-reduce of one/two scalars BEFORE this kernel
//  (design §2.4) so every rank's `st.d_factor`/`st.clip_coef` agree; that
//  all-reduce is host-side orchestration (design §6.2 step 4), also out of scope
//  here. This kernel just consumes the agreed scalar from st.
//
//  Net: every OptId COMPILE-CHECKS here (apply_optimizer is elementwise). Which
//  ones are *correct from this kernel alone* vs need the upstream stage/all-
//  reduce is the taxonomy above — documented per design §2.3/§3.4, enforced by
//  the host shard-mode selection (elementwise → flat-even; per-tensor →
//  tensor-granular).
// ──────────────────────────────────────────────────────────────────────────
//
// CPU/1-GPU AUTHORABLE (design §7.3): this is the cut the test_sharded_optimizer
// gate validates entirely on one GPU. Authored here as NEW code; the kernel
// builder owns wiring it into dispatch + the explicit instantiations.
// ============================================================================

#include <cstdint>
#include <cuda_runtime.h>

// The canonical per-optimizer elementwise math + the FusedOptState ABI. We REUSE
// apply_optimizer<Opt> / OptId / FusedOptState verbatim — INCLUDE, do not edit
// (design §2.3: zero new math, reuse the validated single-GPU tail).
#include "csrc/fused/sm_90/opt_components.cuh"

namespace sg { namespace fused { namespace sm90 {

// ─────────────────────────────────────────────────────────────────────────
//  sharded_optimizer_kernel<Opt, Par> — flat grid-stride over the OWNED shard.
//
//  Args (design §2.3 signature):
//    params_shard : [shard_numel] this rank's owned param slice (mutated in place)
//    grad_shard   : [shard_numel] the reduce-scattered grad for the owned slice
//    lr, step     : the step's learning rate + 1-based step counter
//    st_shard     : FusedOptState whose pointers are ALREADY rebased to the shard
//                   (the host offsets exp_avg/exp_avg_sq/ema/... to the owned
//                   slice before launch — the per-element ABI is shard-local, so
//                   index `i` into params_shard pairs with index `i` into every
//                   st_shard buffer, exactly like the in-kernel P3 owns each
//                   element once). st_shard also carries the per-step scalars
//                   already folded via apply_scalars (bc1/bc2/d_factor/clip_coef/
//                   gate/...), and the precomputed per-tensor pointers (orth/mu/
//                   smart_grad/sam_dir) for the per-tensor cells.
//
//  NO GridBarrier (single phase): the work-steal-over-30-tensors loop of P3 is
//  replaced by one flat grid-stride over [0, shard_numel). Each element is
//  written exactly once by the thread that owns its `i`, so there is no cross-CTA
//  dependence and no barrier (design §2.3). `lr` is threaded into st_shard.lr so
//  apply_optimizer reads a single source for the rate (matches the P3 contract).
//
//  Par is carried for ABI/symmetry with the megakernel template (design §1.1)
//  and to let a builder add `if constexpr (Par::kEmitComm)` hooks later (e.g. a
//  fused all-gather-on-write for ZeRO-3). It is otherwise unused by the
//  elementwise apply — the shard is already local when this kernel runs.
// ─────────────────────────────────────────────────────────────────────────
template <OptId Opt, class Par>
__global__ void sharded_optimizer_kernel(
        float* __restrict__ params_shard,        // [shard_numel] owned param slice
        const float* __restrict__ grad_shard,    // [shard_numel] reduce-scattered grad
        int64_t shard_numel,
        float lr, int step,
        FusedOptState st_shard /* pointers pre-rebased to the shard */) {
    // Single source for the rate (the P3 tail reads st.lr; mirror that here so a
    // caller that passes lr positionally and a caller that pre-folds it via
    // apply_scalars agree bit-for-bit).
    st_shard.lr = lr;

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i < shard_numel; i += stride) {
        // EXACT same canonical math as the in-kernel P3 tail (opt_components.cuh).
        // Shard-local index i pairs params_shard[i] with grad_shard[i] and every
        // pre-rebased st_shard buffer at [i]. Zero new math (design §2.3).
        apply_optimizer<Opt>(params_shard, grad_shard, i, step, st_shard);
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Host launcher — a thin grid-stride launch (design §2.3: "~30 lines, no
//  GridBarrier, single phase"). Sized to the shard, NOT the full param set. A
//  kernel builder routes this from dispatch behind the ZeRO>=2 path + the
//  explicit ParConfig instantiation allow-list (design §1.3/§7.2); it is left
//  here as the canonical small launcher so the kernel has a tested entry point.
//
//  NOTE: no GPU is launched in this CPU/1-GPU pre-work (design §7); this launcher
//  is COMPILE-CHECKED only. block=256 mirrors the per-op elementwise launchers'
//  default; grid is capped so the grid-stride covers any shard_numel.
// ─────────────────────────────────────────────────────────────────────────
template <OptId Opt, class Par>
inline cudaError_t launch_sharded_optimizer_kernel(
        float* params_shard, const float* grad_shard, int64_t shard_numel,
        float lr, int step, const FusedOptState& st_shard,
        cudaStream_t stream = nullptr, int block = 256, int max_grid = 65535) {
    if (shard_numel <= 0) return cudaSuccess;
    int64_t want = (shard_numel + block - 1) / block;
    int grid = static_cast<int>(want < max_grid ? want : max_grid);
    if (grid < 1) grid = 1;
    sharded_optimizer_kernel<Opt, Par><<<grid, block, 0, stream>>>(
        params_shard, grad_shard, shard_numel, lr, step, st_shard);
    return cudaGetLastError();
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_SHARDED_OPTIMIZER_KERNEL_CUH_
