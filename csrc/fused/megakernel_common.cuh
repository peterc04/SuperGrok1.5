#pragma once
// csrc/fused/megakernel_common.cuh
// ─────────────────────────────────────────────────────────────────────────
// Stage 6 — sm_90 L3 persistent-megakernel internals (§1.1–§1.4, §1.13, §1.14).
//
// This header is the **uniform scheduler + barrier substrate** that every
// generated L3 megakernel (csrc/fused/sm_90/*.cu) composes on top of the §3.4
// warp-specialization primitives (csrc/backends/cuda/sm_90/warp_specialize.cuh).
// It defines ONLY the cross-CTA coordination machinery; the per-(model,
// optimizer) compute stages live in the generated TUs.
//
// The four pieces, all sm_90-real (compile-verified via
// scripts/compile_to_object.sh) and degrading to a correct fallback on older
// arches so the codegen matrix still builds:
//
//   §1.1  TaskQueue        — a single global atomic work counter that every
//                            persistent CTA pulls parameter-tensor indices
//                            from (the ThunderKittens-interpreter pattern).
//   §1.2  work-stealing     — there is NO static partition: an idle CTA simply
//                            keeps calling TaskQueue::next(), so a CTA that
//                            finishes its tensor early steals the next one.
//                            (atomicAdd on the shared counter IS the steal.)
//   §1.3  sm_id()           — reads %smid so the host can pin one persistent
//                            CTA per SM and keep that SM's optimizer-state
//                            slice L2-warm across the step.
//   §1.4  GridBarrier       — a hand-built arrive/wait grid barrier from two
//                            global atomics + a sense-reversing generation
//                            counter. Reusable, needs NO cooperative launch
//                            (so it scales past the cooperative-grid CTA cap).
//
// §1.14 fences: acquire/release are placed EXPLICITLY and MINIMALLY — exactly
// one __threadfence() before the arrival publish and one acquire load after
// the release wait at the barrier, and one fence at the task hand-off. No loose
// fences (a loose fence here is both a correctness and a >10% perf footgun).
// ─────────────────────────────────────────────────────────────────────────

#include "csrc/common/platform.h"
#include "csrc/backends/cuda/sm_90/warp_specialize.cuh"
#include <cuda_runtime.h>

// SG_TUNED_MEGA_BLOCK — threads per persistent megakernel CTA, shared by EVERY
// L3 megakernel that composes this substrate (the fused elementwise megakernel
// + the decoder / vit / mamba model drivers + the SuperGrok2 stage). Default
// 256 (= two 128-thread Hopper warp-groups → byte-identical untuned build).
// NEEDS-PARITY before shipping a non-default winner: the §3.4 producer/consumer
// warp-specialization and the per-model shared staging/reduction buffers assume
// the 256-thread (2 warp-group) shape, so any swept value must remain a
// multiple of the warp-group size AND be re-validated on the H100 parity gate —
// the autotuner cannot prove launch-shape correctness on its own.
#ifndef SG_TUNED_MEGA_BLOCK
#define SG_TUNED_MEGA_BLOCK 256
#endif

namespace sg { namespace fused {

// =========================================================================
//  §1.3  SM-pinning: read the streaming-multiprocessor id via %smid.
//
//  The host launches exactly gridDim.x == #SMs CTAs (one persistent CTA per
//  SM) and uses sm_id() inside the kernel to index that SM's resident slice
//  of optimizer state, so the slice stays warm in this SM's L2 partition for
//  the whole step. On pre-sm_90 the asm is still valid (%smid exists since
//  sm_20); the guard only exists to keep a host-compile of this header clean.
// =========================================================================
__device__ __forceinline__ unsigned sm_id() {
#if defined(__CUDA_ARCH__)
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
#else
    return 0u;
#endif
}

// =========================================================================
//  §1.1 / §1.2  TaskQueue — global work counter + work-stealing.
//
//  One int in global memory (`g_next_task`, zero-initialized by the host).
//  Every CTA runs:
//      for (int t = q.next(); t < n_tasks; t = q.next()) process(t);
//  next() does `atomicAdd(g_next_task, 1)` — a CTA that drains its tensor
//  early just grabs the next index, so faster SMs steal work from the queue
//  with no static load-balancing (§1.2). The atomic is the ONLY synchronizer;
//  task bodies are independent (one parameter tensor each), so no fence is
//  needed on the *pull* — the fence belongs at the grid barrier between the
//  forward / backward / optimizer phases (§1.4), not on the per-task pull.
//
//  By convention ONE thread per CTA (the elected leader) pulls a task and
//  broadcasts it to the block via shared memory; `next_block()` packages that.
// =========================================================================
struct TaskQueue {
    int* g_next_task;   // global counter (host memset to 0 before launch)
    int  n_tasks;       // total parameter tensors to process this phase

    __device__ __forceinline__ explicit TaskQueue(int* counter, int total)
        : g_next_task(counter), n_tasks(total) {}

    // Per-thread atomic pull (used when every thread independently claims work,
    // e.g. a warp-granular schedule). Returns a task index that may be >=
    // n_tasks (caller bounds-checks).
    __device__ __forceinline__ int next() {
        return atomicAdd(g_next_task, 1);
    }

    // Per-CTA pull: the leader thread does the single atomicAdd and broadcasts
    // the claimed index to the whole block through `smem_slot`. All threads in
    // the block then cooperate on that one tensor. Returns the claimed index.
    __device__ __forceinline__ int next_block(int* smem_slot) {
        if (threadIdx.x == 0) {
            *smem_slot = atomicAdd(g_next_task, 1);
        }
        __syncthreads();
        int t = *smem_slot;
        __syncthreads();   // ensure every thread read before slot is reused
        return t;
    }
};

// =========================================================================
//  §1.4  GridBarrier — hand-built, reusable, sense-reversing grid barrier.
//
//  State (all in global memory, host-zero-initialized):
//    arrived    — count of CTAs that have reached the current barrier
//    generation — flips/advances each time the barrier is crossed (the "sense"
//                 a late CTA waits on so an early CTA can't lap it)
//
//  The classic correctness hazard with a counter barrier is *reuse*: if you
//  just reset `arrived` to 0, a fast CTA can enter the NEXT barrier and
//  increment the counter before a slow CTA has observed the release of the
//  PREVIOUS one. The sense-reversing generation fixes this: each CTA samples
//  the generation on entry, then waits until the generation has ADVANCED
//  (the last arriver advances it), never touching `arrived` for reset on the
//  consumer side. The last CTA to arrive resets `arrived` to 0 and bumps the
//  generation in one release; everyone else spins on the generation.
//
//  Fences (§1.14):
//    * The releasing writes a CTA makes BEFORE the barrier (its phase output)
//      must be visible to peers AFTER the barrier. We publish arrival with a
//      __threadfence() (device-scope release) before the atomicAdd, and the
//      waiters do an acquire by reading the generation with volatile semantics
//      and a __threadfence() after the wait completes. Exactly two fences per
//      CTA per barrier — the minimum for a correct release/acquire pair.
//
//  Only thread 0 of each CTA participates in the cross-CTA handshake; the rest
//  of the block is gated by a __syncthreads() so the whole CTA leaves together.
// =========================================================================
struct GridBarrier {
    unsigned* g_arrived;      // CTAs arrived at current phase
    unsigned* g_generation;   // sense-reversing generation counter
    unsigned  n_ctas;         // gridDim.x (CTAs that must arrive)

    __device__ __forceinline__ GridBarrier(unsigned* arrived,
                                            unsigned* generation,
                                            unsigned grid_ctas)
        : g_arrived(arrived), g_generation(generation), n_ctas(grid_ctas) {}

    __device__ __forceinline__ void sync() const {
        // The block must fully reach the barrier before its leader publishes.
        __syncthreads();

        if (threadIdx.x == 0) {
            // Sample the sense we are waiting to leave.
            unsigned my_gen =
                *reinterpret_cast<volatile unsigned*>(g_generation);

            // §1.14 RELEASE: make this CTA's phase writes visible device-wide
            // BEFORE we announce arrival, so a peer that observes our arrival
            // (and crosses) also observes our outputs.
            __threadfence();

            // Atomically join the arrival count. The CTA whose add completes
            // the set (== n_ctas) is the "last arriver".
            unsigned prev = atomicAdd(g_arrived, 1u);
            if (prev + 1u == n_ctas) {
                // Last arriver: clear the count for the NEXT phase, then bump
                // the generation. Order matters — reset arrived to 0 first,
                // then a release-store of the advanced generation so a CTA that
                // sees the new generation also sees arrived==0 when it next
                // enters. atomicExch gives the store device-scope visibility.
                atomicExch(g_arrived, 0u);
                __threadfence();
                atomicAdd(g_generation, 1u);
            } else {
                // Not last: spin until the generation advances past our sample.
                // §1.14b BACKOFF: a tight volatile-read spin hammers the L2 /
                // memory subsystem with the generation poll and steals issue
                // slots from arriving CTAs. __nanosleep parks the spinning warp
                // for an exponentially-growing nap (capped) so the poll rate
                // drops and the memory subsystem is left to the laggard CTAs.
                // Behavior-preserving: still exits the instant the generation
                // advances; only the *poll cadence* changes (sm_70+; on older
                // arches the asm is a no-op fallback).
                unsigned backoff = 32u;
                while (*reinterpret_cast<volatile unsigned*>(g_generation)
                       == my_gen) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
                    __nanosleep(backoff);
                    backoff = (backoff < 1024u) ? (backoff << 1) : 1024u;
#endif
                }
            }

            // §1.14 ACQUIRE: after observing the generation advance, fence so
            // the peers' phase writes (released above) are visible to us.
            __threadfence();
        }

        // Release the rest of the block once the leader has crossed.
        __syncthreads();
    }

    // #4 — barrier with a FUSED task-counter reset. The L3 phase loop used to do
    //   fwd → sync → reset_counter → sync → bwd → sync → reset_counter → sync
    // (4 grid barriers + 2 standalone in-kernel counter resets). The reset is
    // folded into THIS barrier's last-arriver critical section: the last CTA
    // zeroes `*reset_counter` BEFORE the release-store that advances the
    // generation, so any CTA that later observes the new generation (and thus
    // crosses) is guaranteed to see the counter == 0 for the next phase. That
    // removes the two standalone resets AND their two barriers — 4 → 2 grid
    // barriers per L3 step — with identical phase ordering/visibility. Pass the
    // task-queue counter (ctx.g_next_task) as `reset_counter`.
    __device__ __forceinline__ void sync_reset(int* reset_counter) const {
        __syncthreads();

        if (threadIdx.x == 0) {
            unsigned my_gen =
                *reinterpret_cast<volatile unsigned*>(g_generation);
            __threadfence();   // §1.14 RELEASE: publish phase writes before arrival.
            unsigned prev = atomicAdd(g_arrived, 1u);
            if (prev + 1u == n_ctas) {
                // Last arriver: clear arrival count AND reset the task counter
                // for the next phase, then a release-store of the advanced
                // generation. Ordering: both resets, then __threadfence(), then
                // the generation bump — so observers of the new gen see both
                // arrived==0 and reset_counter==0.
                atomicExch(g_arrived, 0u);
                atomicExch(reinterpret_cast<unsigned*>(reset_counter), 0u);
                __threadfence();
                atomicAdd(g_generation, 1u);
            } else {
                unsigned backoff = 32u;
                while (*reinterpret_cast<volatile unsigned*>(g_generation)
                       == my_gen) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
                    __nanosleep(backoff);
                    backoff = (backoff < 1024u) ? (backoff << 1) : 1024u;
#endif
                }
            }
            __threadfence();   // §1.14 ACQUIRE.
        }

        __syncthreads();
    }
};

// =========================================================================
//  PersistentContext — the bundle of global scratch a generated L3 megakernel
//  needs. The host allocates these (all zero-initialized) and passes their
//  device pointers to the kernel. Grouping them keeps the generated kernel
//  signatures uniform across all 99 cells.
// =========================================================================
struct PersistentContext {
    int*      g_next_task;     // §1.1 TaskQueue counter
    unsigned* g_arrived;       // §1.4 GridBarrier arrival count
    unsigned* g_generation;    // §1.4 GridBarrier generation
    int       n_tasks;         // parameter tensors this step
    unsigned  n_ctas;          // persistent CTAs (== #SMs, one per SM, §1.3)

    __device__ __forceinline__ TaskQueue queue() const {
        return TaskQueue(g_next_task, n_tasks);
    }
    __device__ __forceinline__ GridBarrier barrier() const {
        return GridBarrier(g_arrived, g_generation, n_ctas);
    }
};

}} // namespace sg::fused
