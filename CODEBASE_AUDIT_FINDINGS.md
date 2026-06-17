# Codebase audit — findings (8-agent line-by-line sweep, 2026-06-17)

Per the recurring-audit process (CODEBASE_AUDIT.md). 8 read-only agents, one per subtree + cross-cutting,
each mapping *where it stands · zero-quality-loss efficiency · bugs · dead code/files*. The **high-stakes
actionable subset is being adversarially re-verified** by the `verify-audit-fixes` workflow before any code
is touched; this doc is the full synthesis. **Nothing here trades quality** — every efficiency/cleanup item
is `QUALITY-IMPACT: none` unless flagged.

## Verification outcome (post-workflow — added before commit)
The high-stakes actionable subset was adversarially re-verified (the `verify-audit-fixes` workflow):
**5 apply-now** (the `has_kernels` probe, scratch-workspace removal, grad-hooks fail-fast at
construction, the decoder-mirror test, scan-adapter removal) — applied + self-test **265 passed** in
`.claude/worktrees/agent-af1e817de3563e3e5/.perf/audit_safe_fixes.diff`, awaiting the GPU `_ops` rebuild
+ fp64 gate before landing on the branch; **6 needs-care** (verify_all re-anchor, the DOA profilers, the
PP patch 0001, the DP=2 assert, the `mma.cuh`/CUTLASS removal, the committed build-artifacts); **3
REJECTED** — premises FALSE against live code: the mamba `FusedOptState` "uninit" (benign on the AdamW
path), the `zero3.py` non-contiguous broadcast, and the `PIPE==2` "half-wired null-deref" (the engine in
fact compiles + places at 1 CTA/SM — confirmed by the PIPE=2 tournament). Treat the P0 table below as the
raw synthesis; those three rows are struck by verification. Final verdicts track in OPTIMIZATION_LEDGER.

## Headline
The **shipping single-GPU path is sound**: the 33-cell L3-TC dispatch is consistent end-to-end (Python sets
≡ C++ tables ≡ opt-id coverage; element/state mirrors exact), the wgmma substrate + the decoder/ViT/mamba
megakernels are production-mature, the fp64 + A/A/A gate is genuinely anchored to fp64 ground truth (no
accidental loosening that lets a bad single-step kernel pass), and the dormant Phase-2/HIP/Pallas code is
correct-and-isolated. **The real defects cluster in three buckets, none on the shipping numeric path:**
(A) a **cleanup-#10 regression class** in the *dev/verification harnesses* + the public `has_kernels()`;
(B) **dead weight** (an 820-LoC unused CUTLASS TU dragging the whole submodule into the build, committed
build artifacts, ~750 MB never-read scratch); (C) **doc drift** (PERF_ANALYSIS bills the reverted P0 as live).

## Per-subtree status
1. **decoder+ViT megakernels** — production-mature, no production race; one latent defect: `PIPE==2` engine
   half-wired (null-deref if built; PIPE=0/1 unaffected). ViT TC is one optimization-gen behind decoder.
2. **Mamba megakernel** — solid + gate-green (scalar scan validated); issues are doc-drift ("13 matrices"→17,
   "conv1d", "~145KB"→210.8KB, "BLOCKED"→converted) + an uninitialized `FusedOptState st;` (benign now).
3. **backend primitives** — sm_90 wgmma substrate mature/correct; `mma.cuh` (CUTLASS) is dead; a misnamed
   warp-only `cluster_reduce_sum_f32` footgun; `tile_pipeline.cuh` is the validated reference (KEEP).
4. **grokking_optimizers/ (≠compile.py)** — production dispatch sound; **#10-regression: `has_kernels()`
   probes removed symbols → False on a healthy build; `verify_all`/profilers target the removed world.**
5. **tuning/scripts** — first-class tooling coherent; `fast_triage` agreement-ledger wired-but-never-called
   (over-claim); ~13 orphaned one-off campaign probes; FLOPs formula duplicated 3×.
6. **tests/** — the gate is sound + fp64-anchored; Phase-2 NOT bring-up-ready (PP patch dormant, DP=2 assert
   tautological, 3D test only `thr>0`); `decoder_kernel_mirror.py` orphan → decoder structural gate unrun.
7. **Phase-2 scaffolding + build + docs** — shard math/TP geometry/PP ownership SOUND + isolated (zero risk
   to `_ops`); build-system (sccache/PYTORCH_NVCC) correct; PP patch 0001 bit-rotted; docs drifted.
8. **cross-cutting dead-file graph** — tree is healthy; few provably-dead files; most "dead" refs are
   dormant-but-intended (flag-and-keep).

## P0 — correctness / regression (fix; all zero-quality-loss)
| sev | item | where | fix |
|---|---|---|---|
| MAJOR | `has_kernels()` returns False on a healthy `_ops.so` (probes 4 #10-removed symbols) — breaks the public API + `_HAS_CUDA` | dispatch.py:~542 | `_KERNEL_PROBE_NAMES=("fused_step","sg2_fused_step")` |
| MAJOR | `verify_all.py` false-fails a healthy tree + never checks the production path | verify_all.py | re-anchor to `mega_*_real_adamw_tc*.cu` + add a `fused_train_step` phase |
| MAJOR | GPU profilers DOA (drive the raising eager `.step()` / compile deleted TUs) | profile.py, profile_maximal.py, utilization.py | repoint to `dispatch.fused_train_step` / `*_real_adamw_tc.cu` |
| MAJOR | `--grad-hooks` → eager `_single_param_step` raises mid-backward | dispatch grad-hooks path | raise at construction, or delete the plumbing |
| MED | `FusedOptState st;` stack-uninitialized; pointer-arith on indeterminate ptrs (benign for AdamW) | mega_mamba_real_adamw_tc.cu | `FusedOptState st{};` |
| MED | `PIPE==2` engine half-wired → null-deref if built (latent; PIPE=0/1 safe) | model_stage_decoder_tc.cuh:~860 | thread `pipeBars` or demote PIPE to 0/1 (a tournament agent may be wiring it now) |
| LOW | `zero3.py` non-contiguous broadcast (dormant, pre-`world>1`) | parallel/zero3.py:~234 | `.contiguous()` before broadcast |

## P0 — gate soundness (Phase-2 bring-up + coverage)
- **DP=2 cross-rank determinism assert (a) is near-tautological** (compares halves of the same all-gather) — compare pre-gather owned shards. The parity-vs-ref (b) carries the real weight.
- **`decoder_kernel_mirror.py` orphan** → the decoder megakernel *structural/aliasing* gate never runs (vit/mamba twins ARE gated; decoder *math* still gated) — add `test_decoder_kernel_mirror_matches_oracle`.
- **PP=2 bit-exact gate dormant** — patch `0001` bit-rotted vs the now-2592-line decoder header; regenerate before 8×H100 PP bring-up. 3D-parallel test asserts only `thr>0` — add a cross-DP param-identity invariant.
- ViT TC loss tol (5e-3) uncalibrated vs decoder's 1e-4; SG2 selftest "fails" only on a CPU-infeasible 1.0s timing assert (numerics exact) — gate the timing assert on `cuda.is_available()`.

## P1 — efficiency / dead weight (zero quality loss; removals pending workflow verification)
- **Remove `mma.cuh`** (~820 LoC CUTLASS GEMM, zero `#include`rs) + retire the default-ON `WITH_CUTLASS` plumbing → drops the whole CUTLASS submodule from the build. *Biggest win.* (verify zero-ref first)
- **Remove `csrc/scan/mamba_scan_adapter.cuh`** (orphan; payload duplicated in live headers).
- **Remove committed build artifacts**: `build.log`, `compiled_cache.db`, `p{3,4,5}_*_timing.txt` (+ gitignore).
- **Reclaim ~750 MB** `Decoder/ViT/Mamba Scratch.workspace` (allocated, never read; TC launchers get nullptr) — dispatch.cpp.
- ViT fwd/dX over-issue ~4× ragged M-atoms (`vittc_gemm_fwd_f32`/`_dx_f32` hardcode `m_atoms=17`) — thread real `m_atoms` (gate-confirmed inert) — model_stage_vit_tc.cuh.
- De-dup the vit/mamba GEMM-FLOP formulas re-inlined in `roofline_bench._flops()`; the `build_variant`/`measure` loops duplicated across the 3 benches + roofline_bench.

## P2 — dead code (defined, uncalled; not compile.py — see COMPILE_AUDIT.md)
- `dispatch.py` `supports_async_copy/tma/block_clusters`, `fused_l3_real_cells()` (0 callers).
- `megakernel_codegen.py` sm_90 emit path (`_emit_cuda`/`dispatch_table_sm90` → deleted headers/orphan `.inc`); `megakernel_engine.py` `dispatch_fused_megakernel`/`all_reduce_optimizer_hooks` (0 callers).
- Large eager residue in `optimizers/*.py` (`_group_cache`/`_fused_step`/`step_full`-ending-in-`self.step()`); `helpers.h` `SG_DISPATCH*`/`check_*_device_side` (0 C++ callers). `cluster_reduce_sum_f32` warp-only footgun.
- `fast_triage` `record_screen_gate` ledger wired but never called (downgrade the "continuously verified" docstring OR wire it from the gate path).

## P3 — doc drift (reconcile to OPTIMIZATION_LEDGER as the canonical truth)
- **PERF_ANALYSIS.md** — billed reverted-P0 as the live #1 lever + "~2% is a ceiling" → **FIXED this session** (supersede banner added).
- **COMPILE_AUDIT.md** pinned to ba19e29 / pre-fix (P0/P1 fixed in af9b720+ad45a88) — add RESOLVED markers.
- **AUTOTUNE_LINKAGE.md** correct flow but dead TU/header paths (`launch_<opt>.cu`, `parity_gate_h100.py`).
- **kernels/README.md** inverts compiled/not-compiled status; the per-cell mamba "BLOCKED" comments contradict the converted reality.

## Flag-and-KEEP (dormant-but-intended; do NOT remove)
HIP+Pallas backends; Phase-2 scaffolding (`tp_layer`/`pp_stage`/`sharded_optimizer_kernel`/`parallel_config`/
`distributed.py`/`zero3.py` — test-only, not in `_ops`); fp64 oracles + `*_kernel_mirror.py`; the ~15
underscore dev/campaign probes; `third_party/cutlass` submodule (until the `mma.cuh` removal is confirmed);
`.STOP_TUNING` sentinel + its guard (operator kill-switch — but delete the sentinel before a real tuning run).

*Apply order (after workflow verification): P0-regression (cheap, fixes the public API + the self-verifier) →
the dead-file removals (verified-safe) → the ViT ragged + workspace reclaim → doc reconciliation → dead-code.
Every change re-gated (fp64 + self-test). Removals stay conservative.*
