#pragma once
// csrc/backends/cuda/sm_90/warp_specialize.cuh
// ─────────────────────────────────────────────────────────────────────────
// §3.4 / §4.5 — Hopper warp-specialization primitives.
//
// These are the building blocks the Stage-6 L3 persistent megakernel composes
// for its producer/consumer (load / compute) warp-group split:
//
//   * elect_one_sync      — single-leader election within a warp (the lane that
//                           issues TMA / cp.async.bulk on behalf of the group),
//                           via the `elect.sync` instruction (1 instr vs a
//                           ballot+ffs).
//   * Mbarrier            — shared-memory async transaction barrier
//                           (`mbarrier.*`), used to signal load→compute handoff
//                           with arrive/expect-tx and try_wait/parity.
//   * warpgroup_reg_alloc/dealloc — `setmaxnreg.{inc,dec}` register
//                           reallocation between the producer (few regs) and
//                           consumer (many regs) warp-groups.
//
// All are sm_90+ and guarded; on older arches the helpers degrade to a correct
// (if unspecialized) fallback so the codegen matrix still builds. Fence usage
// is kept explicit and minimal — loose `fence.proxy.async` placement is a known
// >10% perf footgun (ThunderKittens), so callers pair an mbarrier handoff with
// exactly one async-proxy fence, documented at each call site in Stage 6.
//
// NOTE: this header defines PRIMITIVES only — it is composed by the megakernel
// (Stage 6). It is intentionally NOT retrofitted into the elementwise optimizer
// kernels, where a producer/consumer split would be a no-op.
// ─────────────────────────────────────────────────────────────────────────

#include "csrc/common/platform.h"
#include <cuda_runtime.h>

namespace sg { namespace sm90 { namespace wgs {

// =========================================================================
//  elect.sync — elect a single leader lane among `membermask` (default: the
//  full warp). Returns true on exactly one lane. On pre-sm_90, falls back to
//  ballot + find-first-set (lane == lowest active lane).
// =========================================================================
__device__ __forceinline__ bool elect_one_sync(unsigned membermask = 0xffffffffu) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    unsigned pred;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "elect.sync _|p, %1;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(pred) : "r"(membermask));
    return pred != 0u;
#else
    unsigned active = __ballot_sync(membermask, 1);
    int leader = __ffs(active) - 1;
    return (threadIdx.x & 31) == leader;
#endif
}

// =========================================================================
//  Mbarrier — shared-memory async transaction barrier (mbarrier.*).
//
//  Producer/consumer handoff protocol (Stage-6 megakernel):
//    init(expected_arrivals)             // once, by one thread
//    producer: arrive_expect_tx(bytes)   // announce an incoming TMA/cp.async
//              <issue TMA / cp.async.bulk into shared>
//    consumer: while(!try_wait(parity)){} // spin until the transaction lands
//              <read shared>
//    flip parity each phase.
//
//  The barrier object is a single 8-byte shared word the caller owns.
// =========================================================================
struct Mbarrier {
    unsigned long long* addr;  // shared-memory barrier state (8 bytes)

    __device__ __forceinline__ explicit Mbarrier(unsigned long long* shared_word)
        : addr(shared_word) {}

    __device__ __forceinline__ unsigned smem_u32() const {
        return static_cast<unsigned>(__cvta_generic_to_shared(addr));
    }

    __device__ __forceinline__ void init(int expected_arrivals) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                     :: "r"(smem_u32()), "r"(expected_arrivals));
#else
        (void)expected_arrivals;
#endif
    }

    __device__ __forceinline__ void invalidate() const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(smem_u32()));
#endif
    }

    // Producer: arrive AND declare `tx_bytes` of pending async transactions.
    __device__ __forceinline__ void arrive_expect_tx(int tx_bytes) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"(smem_u32()), "r"(tx_bytes));
#else
        (void)tx_bytes;
#endif
    }

    // Plain arrive (no transaction bytes).
    __device__ __forceinline__ void arrive() const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(smem_u32()));
#endif
    }

    // Consumer: non-blocking test for the given phase parity. Returns true once
    // the expected arrivals + transaction bytes for this phase have completed.
    __device__ __forceinline__ bool try_wait(unsigned phase_parity) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        unsigned done;
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t"
            "}\n\t"
            : "=r"(done) : "r"(smem_u32()), "r"(phase_parity));
        return done != 0u;
#else
        (void)phase_parity;
        return true;  // unspecialized fallback: no async transactions in flight
#endif
    }

    // Blocking wait (spin) for the phase.
    __device__ __forceinline__ void wait(unsigned phase_parity) const {
        while (!try_wait(phase_parity)) { /* spin */ }
    }
};

// One async-proxy fence, paired with an mbarrier handoff. Kept as a named
// helper so Stage-6 call sites issue EXACTLY ONE (loose placement is a >10%
// regression). Orders generic-proxy writes before async-proxy (TMA) consumers.
__device__ __forceinline__ void fence_async_proxy() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("fence.proxy.async.shared::cta;");
#endif
}

// =========================================================================
//  setmaxnreg — warp-group register reallocation (Hopper). The producer
//  (TMA-issuing) warp-group deallocates down to N; the consumer (WGMMA)
//  warp-group allocates up to N. Must be called uniformly by all warps in
//  the warp-group, and only in a kernel launched with the matching
//  __launch_bounds__/cluster config (Stage 6 sets that up).
//
//  N is a template parameter: setmaxnreg requires a compile-time-constant
//  register count (PTX immediate operand).
// =========================================================================
template <int N>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N));
#endif
}

template <int N>
__device__ __forceinline__ void warpgroup_reg_alloc() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N));
#endif
}

}}} // namespace sg::sm90::wgs
