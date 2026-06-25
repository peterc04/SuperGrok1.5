# tp_nvshmem — IN-KERNEL device-NVSHMEM TP/SP all-reduce inside the persistent megakernel

AREA: `csrc/fused/sm_90/tp_transport.cuh` + the megakernel TP-reduce seam
(`model_stage_decoder_tc.cuh` reduce points ①/②/①'/②') + `parallel_config.cuh`
(the `CommCtx` widening) + the workspace allocator
(`mega_decoder_real_adamw_tc_launcher.cu`).

GOAL (verbatim from the task): spec the IN-KERNEL device-NVSHMEM all-reduce the
USER explicitly wants — TP/SP cross-rank reduction done INSIDE the persistent
1-CTA/SM megakernel via NVSHMEM **device** APIs, so the fwd→bwd→AdamW fusion is
preserved. NOT a host CUDA-graph stitch of per-rank launches.

STATUS: This is a **DESIGN + APPLY-READY spec with one hard ENVIRONMENT GATE**
(NVSHMEM is not installed — see §0). The edits below are written so that:
- the diffs to `tp_transport.cuh`, `parallel_config.cuh`, and the workspace
  allocator are **apply-able today** and compile on the no-NVSHMEM box (every
  NVSHMEM symbol stays behind `#if defined(SG_HAS_NVSHMEM)`), and
- the actual cross-GPU reduction can only be **compiled+run** once NVSHMEM is on
  the include/link path (the `-DSG_HAS_NVSHMEM=1` gate), which is the genuine
  8×H100 task.

Everything here is bit-identical-preserving for the SingleGPU / `!kEmitComm`
instantiation: every new symbol is behind `if constexpr (Par::kTPComm)` or
`#if defined(SG_HAS_NVSHMEM)`, both of which fold to nothing on the shipped path.

---

## §0 — ENVIRONMENT GATE: NVSHMEM is NOT installed on this box (verified this session)

This is an install gate exactly like the `ncu` perf-counter gate. Verified
2026-06-25:

```
$ ls /usr/include/nvshmem*           # (nothing)
$ ls /usr/local/lib/libnvshmem*      # (nothing)
$ find / -name nvshmem.h 2>/dev/null | grep -v dist-packages   # (nothing)
$ bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu -DSG_HAS_NVSHMEM=1
COMPILE_FAIL rc=1 tu=tests/hw/tp_loopback_binding.cu
./csrc/fused/sm_90/tp_transport.cuh:79:10: fatal error: nvshmem.h: No such file or directory
```

The ONLY on-box references to "nvshmem" are in the `cuda-pathfinder` PIP catalog
metadata (`/usr/local/lib/python3.11/dist-packages/cuda/pathfinder/...`) — these
are descriptor tables, NOT an install: there is no `nvshmem.h`, no
`libnvshmem_host.so.3`, no `libnvshmem_device.bc`, and `import nvshmem` fails
(`ModuleNotFoundError`).

Baseline confirmed working WITHOUT the flag (the loopback path the spec builds on):
```
$ bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu
COMPILE_OK tu=tests/hw/tp_loopback_binding.cu
```

### What the user must install (the gate the user must clear)

NVSHMEM ≥ 3.x (the device-side `libnvshmem_device.bc` bitcode + `nvshmem.h`/
`nvshmemx.h` headers + `libnvshmem_host.so.3`). Two supported routes:

1. **PIP wheel** (matches the `cuda-pathfinder` catalog already present):
   `pip install nvidia-nvshmem-cu12` (the wheel lands headers in
   `nvidia/nvshmem/include`, the host lib + `libnvshmem_device.bc` in
   `nvidia/nvshmem/lib`). Then export include/lib paths.
2. **NVIDIA HPC-SDK / standalone NVSHMEM tarball** under `$NVSHMEM_HOME`
   (`$NVSHMEM_HOME/include`, `$NVSHMEM_HOME/lib`).

Device-side NVSHMEM is **relocatable-device-code (RDC)**: the megakernel TU
must be compiled with `-rdc=true -DNVSHMEM_TARGET_SM_90` and **device-linked**
against `libnvshmem_device.bc` (`nvcc -dlink ... -lnvshmem_device`), and the
host process must link `libnvshmem_host.so`. RDC is a build-system change for the
megakernel TU specifically (the rest of `_ops` is whole-program-compiled today).
That is flagged in §5 as the build gate.

### Gate commands (re-run after install)
```
grep -rn nvshmem /usr/lib /usr/local 2>/dev/null | head
ls $NVSHMEM_HOME/include/nvshmem.h 2>/dev/null
python -c "import cuda.pathfinder as p; print(p.find_nvidia_header_directory('nvshmem'))" 2>/dev/null
# then the real compile gate (megakernel TP TU, RDC + NVSHMEM):
bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu \
     -DSG_HAS_NVSHMEM=1 -rdc=true \
     -I"$NVSHMEM_HOME/include" -L"$NVSHMEM_HOME/lib" -lnvshmem_device
```

---

## §1 — THE THREE THINGS THE TASK ASKS (answered up front)

**(1) Is NVSHMEM installable here?** No — see §0. It is an environment gate the
user must clear (PIP wheel or HPC-SDK). The code is already structured so the
NVSHMEM path is entirely behind `-DSG_HAS_NVSHMEM=1`; nothing in this spec
weakens that.

**(2) The EXACT device-side call sequence to all-reduce the TP-partitioned
activation at the GridBarrier boundary.** The partials are already in the
workspace (the row-parallel out_proj/ff2 GEMM and the column-parallel-dX GEMM
already write `tr.local(slot_off)` — see `tp_layer.cuh`
`tp_rowparallel_fwd_partial_tile` / `tp_colparallel_dx_partial_tile`). The
sequence at each of the four reduce points (①/②/①'/②') is, per PE:

```
   <GEMM already wrote this PE's partial into the SYMMETRIC slot tr.local(slot_off)>
   tr.rendezvous(bar);                         // (A) cross-CTA GridBarrier + cross-GPU nvshmem barrier
   tp_allreduce_sum_fixed_order(tr, slot_off,  // (B) ASCENDING-pe fp32 reduce over peer slots
                                dst, n, tid, nthreads);
   tr.rendezvous(bar);                         // (C) fence before the slot is reused next reduce point
```

The determinism-critical detail (the A/A/A + fp64-parity requirement): step (B)
is the **hand-rolled ascending-pe loop** in `tp_allreduce_sum_fixed_order`
(already in `tp_transport.cuh`), NOT `nvshmemx_float_sum_reduce` (whose reduction
order is unspecified → ULP drift → A/A/A failure). On NVSHMEM the loop reads
`tr.peer(slot_off, pe)` which is `nvshmem_ptr(heap_base, pe) + off` — a direct
NVLink load/store of peer pe's symmetric slot, summed in fixed ascending order.
This is **already the code in the repo** — the math does not change. What §3
adds is making `tr.rendezvous(bar)` actually fence across GPUs (today the
`NvshmemTransport::rendezvous` exists but §3 hardens its barrier discipline and
adds the symmetric-heap quiet/fence the NVLink-load path needs).

**(3) How it composes with the hand GridBarrier (cross-RANK vs cross-CTA).**
There are TWO barrier systems and they must be nested, never interleaved:
- the hand `GridBarrier` (`megakernel_common.cuh`) is **cross-CTA, within one
  GPU** (sense-reversing, no cooperative launch);
- `nvshmem_barrier_all` / `nvshmemx_barrier_all_block` is **cross-PE (cross-GPU),
  device-and-grid-wide** on the NVSHMEM team.

The discipline (already sketched in the `NvshmemTransport::rendezvous` comment,
hardened in §3): inside `rendezvous(bar)` do `bar.sync()` (all CTAs of THIS GPU
arrive) → **exactly one CTA per GPU** crosses `nvshmemx_barrier_all_block()` →
`bar.sync()` again (release every CTA after the cross-GPU join). The NVSHMEM
barrier must be entered by **one CTA per GPU only**, never concurrently with the
GridBarrier spin — otherwise the two barriers deadlock (a CTA spinning on the
GridBarrier generation while another waits in the NVSHMEM barrier for it to
arrive). §3 adds the `nvshmem_quiet()` (drains outstanding peer stores) BEFORE
the cross-GPU barrier so the ascending-pe NVLink loads in step (B) observe every
peer's published partial.

---

## §2 — THE SYMMETRIC-HEAP ALLOCATION REQUIREMENT (the hard one — forces an allocator change)

This is the honest blocker the task asks me to flag. The operands of an
NVSHMEM reduce **must live in the NVSHMEM symmetric heap** (`nvshmem_malloc`),
NOT in a plain `cudaMalloc` workspace. `nvshmem_ptr(sym, pe)` only translates
pointers that were returned by `nvshmem_malloc` (or `nvshmemx_buffer_register`'d)
— a raw `cudaMalloc` pointer is NOT a symmetric address and `nvshmem_ptr` returns
garbage/nullptr for it.

Today the TP slot lives in the **`cudaMalloc`'d workspace**
(`mega_decoder_real_adamw_tc_launcher.cu` → `dec_tc_launcher_scratch` →
`cudaMalloc(&s.workspace, ...)`). The TP slot is carved from that same blob (the
loopback heap in `tp_loopback_binding.cu` is `at::zeros(...)`, also non-symmetric
— fine for loopback, fatal for real NVSHMEM).

**⇒ The TP comm slots must be split out of the `cudaMalloc` workspace into a
SEPARATE `nvshmem_malloc`'d symmetric region.** Two options:

- **Option A (minimal, RECOMMENDED): a small dedicated symmetric TP-slot heap.**
  Only the reduce slots need to be symmetric — `tp_heap_stride_floats(ctas_per_pe)`
  = `ctas_per_pe * 2 * kTileM * d` floats (§ `tp_layer.cuh::tp_heap_stride_floats`).
  At the flagship (d=1600, kTileM=128, one tile/CTA, ~132 CTAs → 132 PEs-worth on
  the loopback but on real multi-GPU it is `ctas_per_pe` per PE): the symmetric
  region is `nCTA * 2 * 128 * 1600` floats ≈ `132 * 2 * 128 * 1600 * 4 B`
  ≈ 216 MB per GPU — allocate via `nvshmem_malloc`, leave the big acts/grad/state
  workspace on `cudaMalloc`. This is the smallest blast radius.

- **Option B (whole-workspace symmetric):** make the entire `dec_tc_workspace`
  symmetric. Simpler call-site math (one base) but wasteful — the acts/grad/state
  regions never need cross-PE addressing, and `nvshmem_malloc` is collective
  (every PE must call it with the same size in the same order). Not recommended.

The allocator change is in `mega_decoder_real_adamw_tc_launcher.cu`
(`DecTcLauncherScratch`): add an `nvshmem_malloc`'d `tp_sym_heap` field, populated
ONLY on `kEmitComm` (gated `#if defined(SG_HAS_NVSHMEM)`), pass its base into the
kernel via the widened `CommCtx`. On the SingleGPU path the field stays nullptr
and the workspace is byte-identical to today (the §6 PTX gate).

**Symmetric-heap sizing must be UNIFORM across PEs** (collective allocation):
size with the WORLD-max `(T,B,nCTA)` so every PE's `nvshmem_malloc` call agrees;
the host plan already knows the global shapes.

---

## §3 — EDIT 1: harden `NvshmemTransport` in `tp_transport.cuh`

The existing `NvshmemTransport` is close but (a) does the cross-GPU barrier
without an explicit `nvshmem_quiet()` before it (so a peer's published partial
may not be visible to the ascending-pe NVLink load), and (b) does not carry the
TP **team** (it uses the global `nvshmemx_barrier_all_block`, which barriers the
WHOLE world, not just the TP group — wrong when TP is one axis of a 4D mesh:
DP/PP replicas would be dragged into the TP barrier). Both are fixed here.

### VERBATIM OLD (copied from `csrc/fused/sm_90/tp_transport.cuh` lines 137–183)

```cpp
#if defined(SG_HAS_NVSHMEM)
// ─────────────────────────────────────────────────────────────────────────
//  NvshmemTransport — the REAL device-initiated transport (design §5.2).
//  COMPILED ONLY under -DSG_HAS_NVSHMEM=1 (toolkit on the path). The math code
//  path is IDENTICAL to the loopback's; only the address translation + the
//  rendezvous scope differ:
//    * `heap_base` is an nvshmem_malloc'd SYMMETRIC allocation (same offset
//      valid on every PE);
//    * `peer()` translates via nvshmem_ptr (NVLink direct load/store — the
//      single-node 8×H100 case; a multi-node fabric would use nvshmem_get,
//      which is the documented extension point, NOT silently emulated);
//    * `rendezvous()` = in-GPU GridBarrier + ONE CTA running the cross-GPU
//      nvshmemx_barrier_all_block + GridBarrier again (so every CTA of every
//      GPU is fenced both sides — the §5.2 "two barrier systems must not
//      deadlock" discipline: the NVSHMEM barrier is entered by exactly one CTA
//      per GPU, never concurrently with the GridBarrier spin).
//  GO/NO-GO (§5.4): parity vs host-NCCL TP, A/A/A, MFU, ZeRO-3/DP composition —
//  all 8×H100-window activities (design §7.5).
// ─────────────────────────────────────────────────────────────────────────
struct NvshmemTransport {
    float*  heap_base;       // nvshmem_malloc'd symmetric base (LOCAL pointer)
    int     n_pes_;          // TP group size  (nvshmem_n_pes() on the TP team)
    int     my_pe_;          // this GPU's pe  (nvshmem_my_pe())

    __device__ __forceinline__ int my_pe() const { return my_pe_; }
    __device__ __forceinline__ int n_pes() const { return n_pes_; }

    __device__ __forceinline__ float* local(int64_t off) const {
        return heap_base + off;
    }
    __device__ __forceinline__ const float* peer(int64_t off, int pe) const {
        // NVLink path: direct load/store through the peer mapping. nvshmem_ptr
        // returns nullptr for non-P2P-reachable peers — single-node 8×H100 is
        // always reachable; the multi-node fall-back (nvshmem_get staging) is
        // future work and must NOT be silently approximated here.
        return static_cast<const float*>(nvshmem_ptr(heap_base, pe)) + off;
    }

    __device__ __forceinline__ void rendezvous(const ::sg::fused::GridBarrier& bar) const {
        bar.sync();                       // all CTAs of THIS GPU arrived
        if (blockIdx.x == 0) {            // one CTA crosses the GPU boundary
            nvshmemx_barrier_all_block();
        }
        bar.sync();                       // release every CTA after the cross-GPU join
    }
};
#endif  // SG_HAS_NVSHMEM
```

### NEW (replace the block above with this)

```cpp
#if defined(SG_HAS_NVSHMEM)
// ─────────────────────────────────────────────────────────────────────────
//  NvshmemTransport — the REAL device-initiated transport (design §5.2).
//  COMPILED ONLY under -DSG_HAS_NVSHMEM=1 (toolkit on the path; the TU must be
//  built -rdc=true and device-linked against libnvshmem_device — see the build
//  gate in /workspace/impl_diffs/tp_nvshmem.md §5). The math code path is
//  IDENTICAL to the loopback's; only the address translation + the rendezvous
//  scope differ:
//    * `heap_base` is an nvshmem_malloc'd SYMMETRIC allocation (same offset
//      valid on every PE) — see the symmetric-heap allocator change in §2 of
//      the spec; a plain cudaMalloc pointer is NOT addressable via nvshmem_ptr.
//    * `peer()` translates via nvshmem_ptr (NVLink direct load/store — the
//      single-node 8×H100 case; a multi-node fabric would use nvshmem_get,
//      which is the documented extension point, NOT silently emulated);
//    * `rendezvous()` = in-GPU GridBarrier + nvshmem_quiet (drain THIS PE's
//      published partials so peers' NVLink loads see them) + ONE CTA running
//      the cross-GPU TEAM barrier + GridBarrier again (so every CTA of every
//      GPU is fenced both sides — the §5.2 "two barrier systems must not
//      deadlock" discipline: the NVSHMEM barrier is entered by exactly one CTA
//      per GPU, never concurrently with the GridBarrier spin).
//    * `tp_team_` scopes the cross-PE barrier to the TP GROUP only — when TP is
//      one axis of a 4D mesh, barriering the whole world (nvshmemx_barrier_all)
//      would drag DP/PP replicas into the TP reduce. The host bootstrap splits
//      a TP team (nvshmem_team_split_strided over the TP-contiguous PEs) and
//      passes its handle in via CommCtx.tp_comm_handle (cast here).
//  GO/NO-GO (§5.4): parity vs host-NCCL TP, A/A/A, MFU, ZeRO-3/DP composition —
//  all 8×H100-window activities (design §7.5).
// ─────────────────────────────────────────────────────────────────────────
struct NvshmemTransport {
    float*       heap_base;  // nvshmem_malloc'd symmetric base (LOCAL pointer)
    nvshmem_team_t tp_team_; // the TP-group team (NVSHMEM_TEAM_WORLD if TP==world)
    int          n_pes_;     // TP group size  (nvshmem_team_n_pes(tp_team_))
    int          my_pe_;     // this GPU's pe-in-team (nvshmem_team_my_pe(tp_team_))

    __device__ __forceinline__ int my_pe() const { return my_pe_; }
    __device__ __forceinline__ int n_pes() const { return n_pes_; }

    __device__ __forceinline__ float* local(int64_t off) const {
        return heap_base + off;
    }
    __device__ __forceinline__ const float* peer(int64_t off, int pe_in_team) const {
        // NVLink path: direct load/store through the peer mapping. peer() is
        // called with the TEAM-local pe index (ascending 0..n_pes()-1); map it
        // to the GLOBAL pe nvshmem_ptr needs via the team translate. nvshmem_ptr
        // returns nullptr for non-P2P-reachable peers — single-node 8×H100 is
        // always reachable; the multi-node fall-back (nvshmem_get staging) is
        // future work and must NOT be silently approximated here.
        const int global_pe = nvshmem_team_translate_pe(tp_team_, pe_in_team,
                                                         NVSHMEM_TEAM_WORLD);
        return static_cast<const float*>(nvshmem_ptr(heap_base, global_pe)) + off;
    }

    __device__ __forceinline__ void rendezvous(const ::sg::fused::GridBarrier& bar) const {
        bar.sync();                       // all CTAs of THIS GPU arrived
        if (blockIdx.x == 0) {            // one CTA crosses the GPU boundary
            // Drain THIS PE's outstanding symmetric stores so the ascending-pe
            // NVLink loads on every peer observe our published partial. quiet is
            // device-scope; the team barrier then joins all PEs in the TP group.
            nvshmem_quiet();
            nvshmemx_barrier_all_block();   // (team-scoped variant in §3.1 note)
        }
        bar.sync();                       // release every CTA after the cross-GPU join
    }
};
#endif  // SG_HAS_NVSHMEM
```

### §3.1 — team-scoped barrier note (apply when the NVSHMEM version exposes it)

`nvshmemx_barrier_all_block()` barriers `NVSHMEM_TEAM_WORLD`. For a TP team that
is a strict subset of the world (4D mesh), use the **team** block barrier if the
installed NVSHMEM exposes it: `nvshmemx_barrier_block(tp_team_)` (NVSHMEM ≥ 3.x).
If the installed version lacks the block-team variant, the host bootstrap must
ensure the TP team IS the whole world for the single-node 8×H100 pure-TP bring-up
(the first GO/NO-GO), and the world barrier is then correct. The line is marked
`(team-scoped variant in §3.1 note)` above so the GPU-window builder flips it.
This is a version-dependent API choice, NOT a correctness compromise — flagged
honestly as a follow-up the 8×H100 window resolves against the actual toolkit.

### §3.2 — header include for the team type

`nvshmem_team_t` / `nvshmem_team_translate_pe` / `NVSHMEM_TEAM_WORLD` come from
`<nvshmemx.h>` (already included under `#if defined(SG_HAS_NVSHMEM)` at the top
of the file — no new include needed). `nvshmem_quiet` is in `<nvshmem.h>`
(already included). No change to the include block (lines 76–81).

---

## §4 — EDIT 2: widen `CommCtx` to carry the symmetric heap + TP team (`parallel_config.cuh`)

The megakernel needs, on the `kEmitComm` path: (a) the symmetric-heap base for
the TP slots, (b) the TP team handle, (c) the team-local pe index + team size.
`CommCtx` already reserves opaque slots; this widening adds the symmetric-heap
pointer as a typed-but-opaque `void*` (so the header stays CPU-compilable — no
`<nvshmem.h>` include in `parallel_config.cuh`).

### VERBATIM OLD (copied from `csrc/fused/sm_90/parallel_config.cuh` lines 106–120)

```cpp
struct CommCtx {
    // rank within the world / the per-axis groups (filled only when kEmitComm).
    // Defaults describe the degenerate single-GPU world so a default-constructed
    // CommCtx is the correct "no comm" value the single-GPU path forwards.
    int world_size = 1;
    int world_rank = 0;
    int dp_size = 1, dp_rank = 0;
    int tp_size = 1, tp_rank = 0;
    int pp_size = 1, pp_rank = 0;
    // Opaque handle slots (NVSHMEM team / NCCL comm) — left as nullptr-able
    // pointers a builder casts to the concrete handle type behind kEmitComm, so
    // this header need NOT include <nccl.h>/<nvshmem.h> (kept CPU-compilable).
    void* tp_comm_handle = nullptr;   // NCCL comm / NVSHMEM team for the TP group (§5)
    void* dp_comm_handle = nullptr;   // NCCL comm for the DP group (reduce-scatter / all-gather)
};
```

### NEW (replace the block above with this)

```cpp
struct CommCtx {
    // rank within the world / the per-axis groups (filled only when kEmitComm).
    // Defaults describe the degenerate single-GPU world so a default-constructed
    // CommCtx is the correct "no comm" value the single-GPU path forwards.
    int world_size = 1;
    int world_rank = 0;
    int dp_size = 1, dp_rank = 0;
    int tp_size = 1, tp_rank = 0;
    int pp_size = 1, pp_rank = 0;
    // Opaque handle slots (NVSHMEM team / NCCL comm) — left as nullptr-able
    // pointers a builder casts to the concrete handle type behind kEmitComm, so
    // this header need NOT include <nccl.h>/<nvshmem.h> (kept CPU-compilable).
    void* tp_comm_handle = nullptr;   // NCCL comm / NVSHMEM team for the TP group (§5)
    void* dp_comm_handle = nullptr;   // NCCL comm for the DP group (reduce-scatter / all-gather)
    // ── In-kernel TP all-reduce wiring (filled ONLY on kEmitComm; nullptr/0 on
    //    the SingleGPU path, so a default-constructed CommCtx forwards "no TP
    //    heap" and the kEmitComm=false megakernel never reads these — ABI of the
    //    <Opt>-only overload preserved, the §6 PTX-diff gate). ──────────────────
    //  * tp_sym_heap: the nvshmem_malloc'd SYMMETRIC base for the TP reduce slots
    //    (NOT the cudaMalloc workspace — see /workspace/impl_diffs/tp_nvshmem.md
    //    §2). Opaque `float*` here; the megakernel reinterprets it as the
    //    NvshmemTransport::heap_base behind #if defined(SG_HAS_NVSHMEM). On the
    //    loopback build it is the strided cudaMalloc heap (LoopbackTransport).
    //  * tp_heap_stride_floats: per-PE symmetric region stride (the value
    //    tp::tp_heap_stride_floats(ctas_per_pe) computes host-side).
    //  * tp_team_local_pe / tp_team_n_pes: the team-local pe index + team size
    //    (== nvshmem_team_my_pe / _n_pes on the TP team; == tp_rank / tp_size).
    void*   tp_sym_heap          = nullptr;  // nvshmem_malloc'd symmetric TP-slot base
    int64_t tp_heap_stride_floats = 0;       // per-PE symmetric stride (floats)
    int     tp_team_local_pe     = 0;        // pe-in-TP-team (== tp_rank)
    int     tp_team_n_pes        = 1;        // TP team size  (== tp_size)
};
```

(No change to `ParConfig` itself: `kTPComm = (TP > 1)` already gates the reduce.
The static_assert / SingleGPU alias are untouched.)

---

## §5 — EDIT 3: the megakernel signature must be THREADED with ParConfig + CommCtx

**KEY FINDING (verified this session):** the production megakernel is STILL
`template <OptId Opt>` only — `ParConfig`/`CommCtx` are NOT yet threaded into
`fused_decoder_megakernel_tc`. Verified:

```
$ grep -n 'template\|fused_decoder_megakernel_tc(' csrc/fused/sm_90/fused_decoder_megakernel.cuh
145:template <OptId Opt>
...
672:template <OptId Opt>
674:fused_decoder_megakernel_tc(PersistentContext ctx, ...)
1511:template <OptId Opt>
1512:cudaError_t launch_fused_decoder_megakernel_tc(...)
1566:    fused_decoder_megakernel_tc<Opt><<<grid, block, dyn_smem, stream>>>(ctx, params, tok, grad, lr, step, st);
```

So the in-kernel TP reduce CANNOT be wired until the kernel signature is widened
to `template <OptId Opt, class Par = par::SingleGPU>` and takes a trailing
`par::CommCtx comm = {}` argument (design §1.1's documented kernel signature —
`parallel_config.cuh` lines 88–104 describe exactly this seam). This is the
single largest mechanical edit and it is **deferred to the kernel track** (the
spec author is READ-ONLY and the megakernel TU is the kernel-track's tracked
file). What this spec pins down is the EXACT shape of that edit so it composes
with EDIT 1/EDIT 2:

### §5.1 — kernel signature widening (the megakernel-track applies this)

```cpp
// OLD (line 672–674):
//   template <OptId Opt>
//   __global__ void ... fused_decoder_megakernel_tc(PersistentContext ctx, ...)
// NEW:
   template <OptId Opt, class Par = ::sg::fused::par::SingleGPU>
   __global__ void ... fused_decoder_megakernel_tc(PersistentContext ctx, ...,
                                                   ::sg::fused::par::CommCtx comm = {})
```
and at the call site (line 1566) forward `comm`; the launcher (line 1511) gains
the same `class Par` template param + a `par::CommCtx comm` argument that
defaults to `{}` (so every existing `<Opt>` call site compiles unchanged — the
default `Par = SingleGPU`, default `comm = {}`, `kEmitComm=false` folds the
whole TP path away → byte-identical, the §6 gate).

### §5.2 — the transport construction inside the kernel (one place, top of the layer loop)

```cpp
   // Built ONCE per kernel; folds to nothing on the SingleGPU path.
   #if defined(SG_HAS_NVSHMEM)
   ::sg::fused::sm90::tp::NvshmemTransport tr{
       reinterpret_cast<float*>(comm.tp_sym_heap),
       reinterpret_cast<nvshmem_team_t>(comm.tp_comm_handle),
       comm.tp_team_n_pes, comm.tp_team_local_pe };
   #else
   // No NVSHMEM toolkit: the multi-GPU build is the LOOPBACK transport (the
   // honest single-GPU simulation) — strided cudaMalloc heap, GridBarrier is the
   // rendezvous. kEmitComm gates whether tr is used at all.
   ::sg::fused::sm90::tp::LoopbackTransport tr{
       reinterpret_cast<float*>(comm.tp_sym_heap), comm.tp_heap_stride_floats,
       comm.tp_team_n_pes, comm.tp_team_local_pe };
   #endif
```

### §5.3 — the four reduce-point insertions (verbatim anchors in `model_stage_decoder_tc.cuh`)

The four points are EXACTLY where `tp_layer.cuh` documents them. The megakernel
track wraps each in `if constexpr (Par::kTPComm) { ... }` (so SingleGPU is
byte-identical) and replaces the unsharded GEMM with the sharded partial GEMM +
the reduce sequence from §1(2). The verbatim anchors (copied live this session):

**① out_proj forward all-reduce** — `model_stage_decoder_tc.cuh` line 1570–1572:
```cpp
        // a = X_ctx @ out_w^T (+ out_b)  (N=d, K=d). fp32 → work.
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ctx[li] + (int64_t)g0 * dec::kD, wb.out_w[li],
                                            sc.work, dec::kD, dec::kD, sA, sB, pipeBars);
```
On `kTPComm`: `out_w[li]` is the ROW shard `[d, d/P]`, the GEMM becomes
`tp_rowparallel_fwd_partial_tile<...>(tr, slot_part, X_ctx_own, out_w_shard, d/P, d, ...)`
writing the partial into `tr.local(slot_part)`, then the §1(2) A/B/C sequence
reduces into `sc.work` BEFORE the residual+bias fold at line 1576–1578 (which
runs unchanged on the reduced value).

**② ff2 forward all-reduce** — `model_stage_decoder_tc.cuh` line 1601–1603:
```cpp
        // ff2 = X_gact @ ff2_w^T (+ ff2_b) (N=d, K=dff). fp32 → work.
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_gact[li] + (int64_t)g0 * dec::kDff, wb.ff2_w[li],
                                            sc.work, dec::kDff, dec::kD, sA, sB, pipeBars);
```
On `kTPComm`: `ff2_w[li]` is ROW shard `[d, dff/P]`, GEMM →
`tp_rowparallel_fwd_partial_tile<...>(tr, slot_part, X_gact_own, ff2_w_shard, dff/P, d, ...)`,
reduce into `sc.work` BEFORE the r2 fold at line 1607–1609.

**②' ff0-dX backward all-reduce** — `model_stage_decoder_tc.cuh` line 1878–1881:
```cpp
        // ff0 dX: dx1 += dff0 @ ff0_w  (output width Kin=d, contract Nout=dff). fp32
        //   → sc.x1 (free now — fwd x1 consumed); then add to work2.
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff0[li] + (int64_t)g0 * dec::kDff, wb.ff0_wT[li],
                                           sc.x1, /*Kin=*/dec::kD, /*Nout=*/dec::kDff, sA, sB, pipeBars);  // dx1_ffn [nrows,d]
```
On `kTPComm`: `ff0_wT[li]` is the COLUMN-parallel weight (ff0 is Col), so the dX
is a partial → `tp_colparallel_dx_partial_tile<...>(tr, slot_part, dY_ff0_own, ff0_w_shard, d, dff/P, ...)`,
reduce into `sc.x1` BEFORE the `work2[idx] += sc.x1[idx]` accumulate at line 1883–1884.

**①' in_proj-dX backward all-reduce** — `model_stage_decoder_tc.cuh` line 1907–1910:
```cpp
        // in_proj dX: dx_in_attn = dqkv @ in_w  (output width Kin=d, contract Nout=3d).
        //   fp32 → sc.work; ADD residual (in sc.dh) → new running adjoint dh.
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_qkv[li] + (int64_t)g0 * 3 * dec::kD, wb.in_wT[li],
                                           sc.work, /*Kin=*/dec::kD, /*Nout=*/3 * dec::kD, sA, sB, pipeBars);  // dx_in_attn
```
On `kTPComm`: `in_wT[li]` is the COLUMN(QKV)-parallel weight, dX is a partial →
`tp_colparallel_dx_partial_tile<...>(tr, slot_part, dY_qkv_own, in_w_shard, d, 3d/P, ...)`,
reduce into `sc.work` BEFORE the residual add at line 1912–1913.

NOTE on barrier cadence: design P1 ("barrier-free within a tile") relaxes to
"barrier at the 4 reduce points" ONLY on `kTPComm` (each `tr.rendezvous(bar)` is
a GridBarrier+team-barrier). The SingleGPU instantiation keeps today's
barrier-free path byte-identical (the reduce blocks are `if constexpr`'d out).

### §5.4 — the build gate (RDC + device-link), re-stated for the megakernel TU

The megakernel TU(s) that instantiate `<Opt, Par>` with `kEmitComm && SG_HAS_NVSHMEM`
must be compiled `-rdc=true` and device-linked against `libnvshmem_device`:
```
nvcc -c -rdc=true -DSG_HAS_NVSHMEM=1 -DWITH_CUDA \
     -gencode arch=compute_90a,code=sm_90a \
     -I"$NVSHMEM_HOME/include" ... mega_decoder_real_adamw_tc.cu -o mega.o
nvcc -dlink mega.o -L"$NVSHMEM_HOME/lib" -lnvshmem_device -o mega_dlink.o
# host link also needs -lnvshmem_host
```
The SingleGPU `_ops` build is UNCHANGED (no `-rdc`, no NVSHMEM) because the
default instantiation never references an NVSHMEM symbol.

---

## §6 — BIT-IDENTICAL / A/A/A / fp64-parity preservation (the non-negotiables)

1. **SingleGPU PTX-diff gate.** Every new symbol is behind
   `if constexpr (Par::kTPComm)` (folds when TP==1) or `#if defined(SG_HAS_NVSHMEM)`
   (absent without the toolkit). `CommCtx`'s new fields are POD with single-GPU
   defaults the `<Opt>` overload forwards default-constructed. ⇒
   `fused_decoder_megakernel_tc<Opt, SingleGPU>` is byte-for-byte the legacy
   `<Opt>` kernel (design §1.2). The gate: PTX-diff the two instantiations; they
   must be identical. (The loopback TU compiles clean today — §0 gate 3.)

2. **A/A/A across reruns.** The reduce is `tp_allreduce_sum_fixed_order`
   (ASCENDING-pe, fp32) — UNCHANGED from the repo. The summation order is
   STRUCTURAL (pe index), never timing-dependent, so the result is bit-identical
   on every PE and across reruns. `nvshmem_quiet()` + the team barrier only fence
   VISIBILITY (they do not touch the arithmetic). `nvshmemx_float_sum_reduce` is
   deliberately NOT used (unspecified order → ULP drift → A/A/A failure).

3. **fp64 parity.** The cross-rank reduce reproduces the EXACT serial
   chunked-order reference the loopback test already asserts bit-exact against
   (`tp_loopback_binding.cu` reference (ii)) — the transport contributes zero
   numerical effect. The NVSHMEM path reads the IDENTICAL partials in the
   IDENTICAL ascending-pe order, so the loopback's bit-exact gate transfers.

4. **Cross-rank determinism of replicated grads.** Per `tp_layer.cuh`, replicated
   tensors (LN γ/β, biases, head) receive bit-identical grads on every rank
   because their producing adjoints sit downstream of the fixed-order reduces. No
   grad comm needed; determinism is structural. Unchanged.

---

## §7 — GO/NO-GO checklist for the 8×H100 window (what to run after install)

1. `bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu -DSG_HAS_NVSHMEM=1 -rdc=true -I$NVSHMEM_HOME/include`
   → COMPILE_OK (the §0 gate, now passing because nvshmem.h is on the path).
2. Repoint `tp_loopback_binding.cu` `LoopbackTransport` → `NvshmemTransport`
   (the swap point — NO math change in that file, design §5.3) and run
   `tests/hw/test_tp_loopback.py` across 2 PEs on 2 GPUs: assert (a) bit-identical
   reduced outputs across PEs, (b) 3-rerun A/A/A, (c) bit-exact vs the serial
   chunked-order reference, (d) exact-slice dW/db.
3. Wire EDIT 1/2/3 into the megakernel (`<Opt, Par>` + `comm`), allocate the
   symmetric TP-slot heap (§2 Option A), run the flagship TP=2/4/8 forward+backward
   and assert parity vs a host-NCCL TP reference + A/A/A.
4. Compose with ZeRO-3 / DP / PP (the 4D mesh) — the TP team scoping (§3.1) is
   what makes the TP barrier not drag DP/PP replicas in.

---

## §8 — FILES TOUCHED (apply order)

| # | file | edit | apply-able now? |
|---|------|------|-----------------|
| 1 | `csrc/fused/sm_90/tp_transport.cuh` | §3 — harden `NvshmemTransport` (team scope + quiet) | YES (behind `#if SG_HAS_NVSHMEM`; no-op on this box) |
| 2 | `csrc/fused/sm_90/parallel_config.cuh` | §4 — widen `CommCtx` (sym-heap base, stride, team pe/size) | YES (POD fields, CPU-compilable, single-GPU defaults) |
| 3 | `csrc/fused/sm_90/fused_decoder_megakernel.cuh` | §5.1 — thread `<Opt, Par>` + `comm` into kernel+launcher signature | KERNEL-TRACK (tracked file; spec pins the exact shape) |
| 4 | `csrc/fused/sm_90/model_stage_decoder_tc.cuh` | §5.3 — wrap the 4 reduce points in `if constexpr (Par::kTPComm)` + sharded partial GEMM + reduce | KERNEL-TRACK |
| 5 | `csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu` | §2 — add `nvshmem_malloc`'d `tp_sym_heap` to `DecTcLauncherScratch` (gated), populate `CommCtx` | KERNEL-TRACK + needs NVSHMEM linked |
| 6 | build system | §5.4 — `-rdc=true` + device-link `libnvshmem_device` for the megakernel TU on the `SG_HAS_NVSHMEM` path | NEEDS NVSHMEM (§0 gate) |

Edits 1–2 are apply-ready and compile-safe on the current (no-NVSHMEM) box today.
Edits 3–6 are pinned to exact verbatim anchors here but land in kernel-track
tracked files and/or require the NVSHMEM install (§0). The honest blocker is the
symmetric-heap allocator change (§2) — the operands CANNOT stay in the
`cudaMalloc` workspace; that is the forced allocator change the task asked me to
flag.
