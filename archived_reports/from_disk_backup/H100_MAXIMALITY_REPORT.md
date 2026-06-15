# H100 (sm_90a) MAXIMALITY AUDIT — SuperGrok1.5

Host: real NVIDIA H100 80GB HBM3, driver 570.211.01 (CUDA 12.8 driver), nvcc 12.4.131,
cuobjdump 12.4, torch 2.4.1+cu124. Budget (mk._arch_budget sm_90): 255 regs/thread, 233472 B smem.
Shipped binary: grokking_optimizers/_ops.cpython-311-x86_64-linux-gnu.so (33.9 MB, built Jun 9 05:03).

## PROFILE_MAXIMAL RESULT (PYTHONPATH=. python3 grokking_optimizers/profile_maximal.py)
Final: 6 PASS / 5 FAIL / 9 SKIP-silicon (20 probes, 636 s). Exit 1.

Premise correction: the "5 GPU-deferred" probes on the CPU host were the nvcc-GATED tiers
(A + B). This harness is host-side binary inspection (nvcc+ptxas+cuobjdump), not live-GPU
execution. On this H100, nvcc IS present so A+B RAN FOR REAL. The 9 SKIPs are NOT GPU-deferred —
they are Tier C (gfx942, needs clang/llvm — absent), Tier D (jax — absent), Tier E
(jax/pallas — absent). All three target NON-H100 backends and are orthogonal to the H100 verdict.

TIER A (sm_90 GEMM instruction maximality) — all 4 FAIL, but ONLY on 8B spill:
  decoder    : WGMMA=72 TMA=152 mbarrier=42 C7509=0 spills=8B
  vit        : WGMMA=88 TMA=208 mbarrier=56 C7509=0 spills=8B
  mamba      : WGMMA=56 TMA=124 mbarrier=28 C7509=0 spills=8B
  supergrok2 : WGMMA=40 TMA=96  mbarrier=21 C7509=0 spills=8B
  => WGMMA present, TMA present, C7509=0 (wgmma mainloop NOT serialized — the real perf gate
     PASSES). The ONLY failing condition is spill_bytes==0 (got 8B). Tensor-core path is live.

TIER B (sm_90 resource health, 0 spills = maximal):
  cell mamba3/adamw            : 32 regs, 4232B smem, 0B spills   PASS
  cell mamba3/supergrok2       : 40 regs, 4232B smem, 0B spills   PASS
  cell transformer_decoder/muon: 30 regs, 136B  smem, 0B spills   PASS
  cell vit/prodigy             : 32 regs, 4232B smem, 0B spills   PASS
  opt launch_adamw             : 44 regs, 0B spills               PASS
  opt launch_muon              : 32 regs, 0B spills               PASS
  opt launch_supergrok2        : 168 regs, 8B spills              FAIL  <- same TF32 CUTLASS kernel
  (mamba3 "32 regs / 0 spills" matches the docstring's stated Tier B target exactly.)

TIER C/D/E: all SKIP (clang/llvm absent for gfx942; jax absent for functional+TPU). Out of scope
for the H100 verdict; not a regression.

## SASS EVIDENCE TABLE (shipped .so, cuobjdump -sass / -elf / --list-elf / -res-usage)
Embedded archs:  47 ELF objects, ALL sm_90a (cuobjdump --list-elf). cuobjdump -elf shows
  47x "arch = sm_90a". PTX fallback compute_90 also present. NO sm_80/sm_70 SASS, not PTX-only-JIT.
ptxasOptions baked in binary: "--opt-level=3 -v --warn-on-spills --generate-line-info" (= the
  shipped NVCC_DEVICE_BASE + setup.py -lineinfo). The .so reflects the real install build.

| Feature (prompt grep)            | shipped .so count | actual SASS mnemonic            | verdict |
|----------------------------------|-------------------|---------------------------------|---------|
| `grep -ci "wgmma"`               | 0                 | (PTX spelling; not in SASS)     | ARTIFACT|
| WGMMA (Hopper warpgroup MMA)     | 256               | HGMMA.* (the SASS spelling)     | PRESENT |
|   - HGMMA.64x128x8.F32.TF32      | 144               |                                 |         |
|   - HGMMA.64x128x16.F32.BF16     | 56                |                                 |         |
|   - HGMMA.64x128x16.F32 (F16)    | 56                |                                 |         |
| TMA bulk copy                    | 622               | UTMALDG.3D (580) + bulk variants| PRESENT |
| cp.async (Ampere async copy)     | 166               | LDGSTS (in hand-written attn)   | PRESENT |
| tensor-core mma / HMMA / IMMA    | 0                 | (Hopper uses HGMMA, not HMMA)   | N/A     |
| MBARRIER/ELECT/ARRIVES/BAR/DEPBAR| 1650              |                                 | PRESENT |
| MUFU (transcendentals)           | 2607              |                                 |         |
| FFMA                             | 14943             |                                 |         |

CRITICAL: the prompt's `grep -ciE "wgmma"` returns 0 ONLY because the Hopper warpgroup-MMA
mnemonic in SASS is HGMMA / GMMA — "wgmma" is the PTX-level spelling, which cuobjdump -sass
does not emit. The `\b[HUWIQ]?GMMA\b` pattern (which the harness itself uses) finds 256. A
reviewer reading only "0" would wrongly conclude FAIL.

Attribution (per-function split of the SASS): 21 distinct kernels carry all 256 HGMMA + 580
UTMALDG. Every one is
  cutlass::device_kernel<GemmUniversal<... MainloopSm90TmaGmma[Rmem A]WarpSpecialized ...,
  KernelTmaWarpSpecializedCooperative, {half_t|bfloat16_t|tfloat32_t},
  SM90::GMMA::MMA_64x128x{8,16}_F32{F16F16|BF16BF16|TF32TF32}_{SS|RS}, SM90_TMA_LOAD>>
— i.e. exactly the CUTLASS Sm90 collective that csrc/backends/cuda/sm_90/mma.cuh builds
(GemmUniversalAdapter from CollectiveBuilder<arch::Sm90, OpClassTensorOp>). The cp.async/LDGSTS
(166) lives in 8 hand-written attention kernels (fmha_softmax, hca/csa_attention) — the in-tree
flash-attention async-copy path, separate from the CUTLASS TMA path.

VERDICT ON THE WGMMA/TMA CLAIM: HOLDS. The README claim that the sm_90 path is the CUTLASS
WGMMA/TMA fast path is TRUE on the shipped binary. WITH_CUTLASS was compiled in (cutlass
submodule present at third_party/cutlass, 118 MB; setup.py auto-enables on CUDA>=12 + headers;
mma.cuh #errors without WITH_CUTLASS so the kernels could not exist otherwise). The 256 HGMMA +
580 UTMALDG are physically in the SASS.

## SPILL / REG / OCCUPANCY TABLE
Authoritative spill signal = ptxas "N bytes spill stores" (Tier A/B) + cuobjdump -res-usage STACK
on the .so, reconciled against an STL/LDL SASS scan. Max REG across ALL 456 shipped kernels = 168
(<< 255 budget) — NO kernel is register-limited against the contract.

Heavy / GEMM-bearing kernels (from -res-usage on the .so + ptxas -v recompile):
| kernel                                          | regs | spill stores/loads | static smem | note |
|-------------------------------------------------|------|--------------------|-------------|------|
| CUTLASS GEMM, half_t  SS (MMA_64x128x16 F16)    | 168  | 0 / 0              | 1024B*      | OK   |
| CUTLASS GEMM, bf16_t  SS (MMA_64x128x16 BF16)   | 168  | 0 / 0              | 1024B*      | OK   |
| CUTLASS GEMM, tf32_t  SS (MMA_64x128x8  TF32)   | 168  | 0 / 0              | 1024B*      | OK   |
| CUTLASS GEMM, tf32_t  RS (MMA_64x128x8 TF32_RS) | 168  | 8 / 8             | 1024B*      | SPILL|
| (17 of 21 GEMM kernels STACK=0; the 4 TF32-RS instances carry the 8B)  |
| mega_* fused cells (mamba3/transformer/vit, all opts) | 30-40 | 0 / 0       | 136-4232B   | OK   |
| launch_adamw / launch_muon                      | 44/32| 0 / 0              | small       | OK   |
| scan_backward_kernel (mamba)                    | 72   | 0 / 0  (STACK 34816B = local array, not spill) | local | OK |
| sequential_scan / moe_scan_compacted            | 40   | 0 / 0  (STACK 1024B = local array)             | local | OK |
| sg11/sg15_sweep_a, sg11_mu_metanet, neuralgrok  | 64   | 0 / 0  (STACK 632-856B = metanet local arrays) | local | OK |
*CUTLASS smem is allocated DYNAMICALLY at launch (cudaFuncSetAttribute); static SHARED=1024B
 understates it. These GEMMs are smem-bound to ~1 CTA/SM by design (warp-specialized Hopper
 operating point) — that is the intended config, NOT a contract gap. Achieved occupancy is
 ncu-only (silicon-gated).

KEY RECONCILIATION: 31 of 456 kernels show STL/LDL or STACK>0, but ptxas attributes "N bytes
spill stores" to ONLY ONE kernel (the TF32-RS GEMM, 8B). All large STACK frames (scan_backward
34816B, sweep_a/metanet 632-856B, moe_scan 1024B) are explicit per-thread LOCAL ARRAYS (alloca) —
ptxas reports "0 bytes spill stores" for every one. Local arrays != register spills.

Occupancy sanity: no kernel is reg-pressure-limited (max 168 / 255). Nothing pathological (<25%)
from register pressure. The scalar fused cells (30-40 regs, <=4232B static smem) achieve high
theoretical occupancy. The CUTLASS GEMMs are smem-bound to ~1 CTA/SM by design.

## VERDICT: IS THE H100 PATH MAXIMAL PER THE CODEBASE'S OWN CONTRACT?
PER THE CONTRACT (profile_maximal, which IS the codebase's own maximality harness): FAIL — it
exits 1, 5 of 5 FAILs all caused by ONE 8-byte-spilling CUTLASS TF32-RS kernel.
SUBSTANTIVELY: maximal in every load-bearing respect — the Hopper WGMMA/TMA tensor-core fast path
genuinely ships and the single spilling kernel is runtime-DEAD (never launched). The contract FAIL
is real but isolated, cosmetic (8B on a never-executed kernel), and closeable without a kernel rewrite.

What HOLDS (the load-bearing claims):
  - WGMMA tensor-core MMA: 256 HGMMA in the shipped SASS, across 21 CUTLASS Sm90 collective
    kernels. The "0 wgmma" from the literal grep is a spelling artifact, not a fail.
  - TMA: 622 bulk-copy ops (580 UTMALDG.3D). SM90_TMA_LOAD in the kernel types.
  - Arch: 47 sm_90a ELFs, no sm_80 fallback, real SASS not PTX-only. CORRECT.
  - wgmma NOT serialized: C7509=0 on all 4 GEMM TUs (NDEBUG working). PASS — the real perf gate.
  - Spills: 0 on every scalar fused cell + every scalar opt kernel + the half/bf16/TF32-SS GEMMs.
  - Register pressure: max 168/255 across all 456 kernels. No occupancy-limiting reg pressure.

GAP (the single itemized deviation — and it is RUNTIME-DEAD, see reachability):
  GAP-1: Exactly ONE kernel in the entire 456-kernel binary spills: the CUTLASS TF32
    A-transposed register-source cooperative GEMM
    (SM90::GMMA::MMA_64x128x8_F32TF32TF32_**RS_TN**, MainloopSm90TmaGmmaRmemAWarpSpecialized),
    8B stack frame / 8B spill stores / 8B spill loads, 168 regs. It is compiled into all 4 GEMM
    TUs (decoder/vit/mamba/supergrok2) + launch_supergrok2 (1 copy each = 4 instances), so it
    trips 5 of the 5 profile_maximal FAILs (4 Tier A + 1 Tier B). Enumeration of all 21 GEMM
    kernels in the .so: 7x BF16_SS (0B), 7x F16_SS (0B), 3x TF32_SS_TN (0B), 4x TF32_RS_TN (8B).
    Only the RS_TN (A-transposed) TF32 variant spills; every SS variant and every half/bf16
    kernel is 0B. The 8B lives inside CUTLASS's own register-source mainloop, not project code.

  REACHABILITY (this downgrades GAP-1 to cosmetic): the TF32 A-transposed kernel
    (Sm90GemmAT<tfloat32_t> via sm90_run_gemm_atb<tfloat32_t>) is reached at runtime ONLY through
    vit_run_gemm_atb<T> (vit_sm90.cuh:489), whose `if constexpr (std::is_same<T,float>)` branch —
    the ONLY branch whose Elem maps to tfloat32_t (cutlass_gemm_elem<float>=tfloat32_t) — returns
    cudaErrorNotSupported and FALLS BACK TO THE SCALAR gemm_grad_weight_kernel (the ViT float
    wgrad). mma.cuh's own DOCUMENT-STOP (lines 720-740) documents exactly this: the TF32
    A-transposed collective hits a hard CUTLASS static_assert and is deliberately routed to the
    scalar path. So the spilling kernel BODY is emitted by ptxas (hence the real 8B in the binary)
    but is NEVER LAUNCHED for float/TF32 at runtime. The LIVE TF32 model GEMMs use
    sm90_run_gemm_tf32_bt -> Sm90GemmBT<tfloat32_t> = the SS path (TF32_SS_TN, STACK=0, mma.cuh:808).
    => GAP-1 has ZERO runtime perf impact; it is a profile_maximal *static-inspection* FAIL on a
    kernel that the runtime guard prevents from ever executing.

  NOT gaps (explicitly ruled out as false positives):
    - "0 wgmma": spelling artifact (HGMMA is the SASS mnemonic; wgmma is the PTX spelling).
      256 HGMMA tensor-core MMAs ARE present.
    - large STACK frames (scan_backward 34816B, sweep_a/metanet 632-856B, moe_scan 1024B): these
      are explicit per-thread LOCAL ARRAYS (alloca) — ptxas reports "0 bytes spill stores" for
      every one. Verified directly. Local arrays != register spills.
    - low CUTLASS occupancy / ~1 CTA/SM: by design (warp-specialized Hopper; smem is dynamic),
      contract-deferred to silicon, not a gap.
    - Tier C/D/E SKIPs: non-H100 backends, toolchain-gated (clang/jax absent), orthogonal.

## MINIMAL CHANGE TO CLOSE GAP-1 (flag-level, NOT a kernel rewrite) — TESTED ON THIS H100
TESTED: re-probed launch_supergrok2.cu with `-Xptxas --maxrregcount=192` AND `=255` on this H100.
The TF32-RS kernel STILL reports `8 bytes spill stores` in BOTH cases (the other 0-spill kernels
stay at 0). So a register CEILING does not help — the kernel uses 168 regs and 192/255 are
non-binding above that; the spill is not relievable by capping. (And --allow-expensive-
optimizations is CUDA 12.5+, unavailable on this 12.4 host.) The realistic minimal options:
  (a) ACCEPT / DOCUMENT as a CUTLASS-internal residual on a runtime-DEAD kernel (RECOMMENDED).
      The spilling TF32-RS GEMM is never launched (the float A-transposed path is guarded to the
      scalar fallback). 8B on a 168-reg cooperative GEMM that never runs is functionally and
      perf-wise a non-event. The cleanest flag-level closure is to make profile_maximal's Tier A
      spill assertion ignore CUTLASS-internal spills on kernels not in the runtime call graph
      (e.g. exclude the RS_TN TF32 instantiation), OR document the 8B as an accepted CUTLASS
      upstream residual. (Requires owner sign-off; it changes the harness threshold, not a kernel.)
  (b) STOP EMITTING the dead kernel: prevent the TF32 A-transposed collective from being
      instantiated at all (it is only ODR-reached via templates whose runtime guard already
      returns NotSupported). If a translation-unit-local change can drop the
      Sm90GemmAT<tfloat32_t> instantiation that the live code never calls, the 8B disappears from
      the binary AND the harness passes — with zero runtime behavior change. This is a
      source-level instantiation-pruning change (one TU), NOT a kernel rewrite, and is the only
      way to actually clear the FAIL flag rather than relax it.
  There is NO flag (gencode / WITH_CUTLASS / maxrregcount) that closes GAP-1 — verified. The fix
  is either (a) accept+document, or (b) prune the dead TF32-RS instantiation. A kernel rewrite is
  out of scope and unnecessary (the kernel is CUTLASS's and is never launched).

## CONFIDENCE + WHAT I COULD NOT MEASURE
Confidence HIGH on: WGMMA/TMA presence + attribution (direct SASS of the shipped .so, demangled
to the CUTLASS Sm90 collective), arch embedding (sm_90a only), C7509=0, the spill being exactly
8B isolated to the TF32-RS GEMM (ptxas per-function attribution + cuobjdump -res-usage agree),
register caps (max 168/255), and the local-array-vs-spill distinction (ptxas "0 bytes spill
stores" on the big-STACK kernels).

Could NOT measure (silicon-gated / out of scope, consistent with the harness's own residual list):
  - Achieved (live) occupancy, DRAM/L2 bandwidth, SM duty cycle — needs ncu/nsight on a running
    kernel; theoretical occupancy of the CUTLASS GEMMs is dominated by DYNAMIC smem set at launch,
    which static analysis cannot see.
  - Wall-clock latency / throughput / fused-vs-ATen speedup — needs kernel execution + timing.
  - Tier C (gfx942 v_mfma ISA) — needs clang+llvm-objdump (absent); MI300X target, not H100.
  - Tier D/E (functional descent, TPU HLO) — needs jax (absent); not H100.
  - Whether --maxrregcount clears the 8B — TESTED (192 and 255): it does NOT. (No repo edits made.)
  - The exact template trigger that ODR-instantiates Sm90GemmAT<tfloat32_t> despite no live float
    call site — not fully traced (the runtime guard makes it moot for the verdict); confidence
    HIGH that it is never *launched* for float, MEDIUM on the precise instantiation mechanism.
