# Phase-2 authoring report — Lane C (#25), 2026-06-12

**Mission:** finish the multi-GPU (Phase-2) authoring as far as one GPU's worth
of hardware allows, per the owner's provisioning rule: template + DP/ZeRO/PP/TP
code AUTHORED and single-GPU-unit-tested as far as possible; only NVSHMEM-TP
transport validation + real scaling measurements may remain for the 8×.
**Constraint honored:** CPU-only lane — every GPU test below is AUTHORED +
nvcc-COMPILED (sm_90a) but NOT RUN; `.phase2/RUNBOOK.md` is the exact
execution plan. No tracked file was modified (changes to tracked files ship as
`.phase2/patches/*`); `csrc/bindings/*` and `grokking_optimizers/compile.py`
were not touched (other lane's surface).

**NVSHMEM verdict: NOT INSTALLED** (verified: `find / -name "nvshmem*"` over
the filesystem, `pip list`, `ldconfig -p`, CUDA 12.4 include tree — zero hits).
Therefore TP was authored against the transport interface with a bit-exact
single-process loopback transport (the design-§5.3/§5.4-sanctioned path), with
the real `NvshmemTransport` surface compiled only under `-DSG_HAS_NVSHMEM`.

---

## 1. Authored-vs-missing matrix (against /workspace/.parallelism_design.md)

Legend: **[pre]** existed before this pass (ef433ac + earlier) · **[NEW]**
authored this pass · **[8×]** genuinely needs the 8×H100 window · **[follow-up]**
1-GPU-authorable later, explicitly not done.

| Axis | Contract item (§) | Status |
|---|---|---|
| **Template** | `ParConfig<DP,TP,PP,SP,Z>` + `SingleGPU` + `CommCtx` (§1.1) | [pre] `csrc/fused/sm_90/parallel_config.cuh` |
| | Allow-list instantiation compile gate (§7.2) | [pre] `tests/hw/test_parallel_instantiation.py` (CPU-green) |
| | Megakernel `<Opt, Par>` threading + `if constexpr` comm gates (§1.2/§2.2/§9) | [8×]-adjacent: dispatch/bindings surface (other lane) + transport-choice-dependent; the B2-cut semantics are bit-gated NOW via the capture-grad/re-apply equivalence (ef433ac D1 + the new §6.2 module) |
| **DP** | DP=2 loopback cross-rank A/A/A, fixed-order reduce (§7.1/§6.4/§2.7) | [pre] `tests/hw/test_dp2_loopback_determinism.py` (16/16 green on HW) |
| | Production-shaped `fused_train_step_distributed` (§6.2 [0]-[5]) | [NEW] `grokking_optimizers/parallel/distributed_step.py` + gate `tests/hw/test_distributed_step.py` (world=1 bit-identity ×2 steps; world=2 torchrun loopback through the module) |
| | Rank-aware wiring_check mode (§6.3) | [follow-up] edit to tracked `wiring_check.py`; trivially authorable but only meaningfully testable under a launched job |
| | Real 1→8 weak scaling (≥70% bar) | [8×] |
| **ZeRO-1/2** | Sharded-opt kernel == in-kernel P3 bit-parity (§2.3/§7.3) | [pre] `sharded_optimizer_kernel.cuh` + `tests/hw/test_sharded_optimizer.py` (9 cells bit-exact on HW) |
| | Z2-as-Z3-minus-param-shard (same code path, §3.1) | [pre]/[NEW] realized by `FlatShardPlan` + store (param residency is the only delta) |
| **ZeRO-3** | Flat-blob param partition honoring §3.4 (elementwise-even / tensor-granular) | [pre] `parallel/shard_map.py`; [NEW] `parallel/zero3.py::flat_plan_for_optimizer` (maps plans to megakernel-ABI flat slices, fingerprinted) |
| | §3.2(a) full pre-gather / release around the fused step | [NEW] `Zero3FlatParamStore.gather_full/release` (virtual-peer loopback + dist paths; loud on stale-world) |
| | Sharded opt-state checkpoint / save-resume, bit-exact | [NEW] `save/load_sharded_checkpoint` (plan-fingerprint-guarded) + CPU gate `tests/test_zero3_plan.py` (green) + GPU gate `tests/hw/test_zero3_roundtrip.py` (2-step sharded chain == production chain; resume bit-exact; guards loud) |
| | §3.2(c) in-kernel NVSHMEM param gather | [8×]+conditional (only if (a) OOMs at flagship size — design says do NOT build speculatively) |
| **PP** | Layer-range stage cut of the TC tile functions (§4.1) | [NEW] `.phase2/patches/0001-dectc-layer-range-pp.patch` (tracked-file change as patch; PTX-identity proof below) |
| | Stage kernels (fwd-only; fwd-recompute+bwd+owned-dW) + launchers | [NEW] `csrc/fused/sm_90/pp_stage_decoder_tc.cuh` (P0→B0→P1→B1→P2 per stage; ends at the B2-seam; loud `#error` without the patch) |
| | Activation hand-off buffers (bf16 X_in[L] fwd; fp32 dh bwd) | [NEW] in the patch + `pipeline.py::handoff_plan` (fp32 dh = the bit-preserving carrier) |
| | 1F1B schedule + validator + host driver + loopback P2P (§4.2) | [NEW] `grokking_optimizers/parallel/pipeline.py` + CPU gate `tests/test_pipeline_schedule.py` (green) |
| | Single-GPU 2-stage loopback A/A/A vs fused step | [NEW] `tests/hw/pp_stage_binding.cu` + `tests/hw/test_pp2_loopback_determinism.py` (grad+loss+closure BIT-exact asserts; ownership cross-check) |
| | PP bubble/throughput at real stage counts; real P2P transport | [8×] (and `LoopbackP2P`→dist swap) |
| | vit/mamba stage twins | [follow-up] same recipe once decoder is HW-validated |
| **TP** | Transport interface + loopback (symmetric-heap sim) + NVSHMEM surface (§5.2/§5.3) | [NEW] `csrc/fused/sm_90/tp_transport.cuh` (`LoopbackTransport` real+testable; `NvshmemTransport` compiled only under `-DSG_HAS_NVSHMEM` — honest absence, not a stub) |
| | Fixed-order ascending-pe fp32 all-reduce (§5.2 determinism non-negotiable) | [NEW] `tp_allreduce_sum_fixed_order` (one code path for both transports) |
| | Megatron col/row split geometry of the 30-tensor decoder layout + QKV head-aligned 3-block shard + dW exact-slice property + the 4 reduce-point insertion map (§5.1) | [NEW] `csrc/fused/sm_90/tp_layer.cuh` (shard table, pack maps, per-tile sharded wgmma block functions reusing the production `dectc_gemm_*`) |
| | Single-GPU loopback gate (TP∈{2,4}: cross-rank identity, A/A/A, transport-neutrality bit-exact, dW slice-exactness bit-exact, parity vs unsharded) | [NEW] `tests/hw/tp_loopback_binding.cu` + `tests/hw/test_tp_loopback.py` |
| | NVSHMEM transport validation + §5.4 go/no-go (vs host-NCCL TP) | [8×] — THE residual; the swap point is one type name in the binding |
| | TP insertion into the production kernel body (`if constexpr (Par::kTPComm)` at the 4 marked points) | [8×]-scheduled by design (TP is the LAST increment, transport-choice-dependent); insertion map documented at file:line in `tp_layer.cuh` |
| **SP** | Expressible, pinned 1 (static_assert) | [pre] `parallel_config.cuh` + §7.2 gate |
| **Graph** | 1-GPU capture of the decomposed step (§7.4) | [pre] `tests/hw/test_step_graph_capture.py` (green; mixed megakernel+NCCL capture documented as 8×) |

## 2. What was authored (file list)

**New CUDA/C++ headers (committed):**
- `csrc/fused/sm_90/tp_transport.cuh` — the TP transport seam (loopback + NVSHMEM surface + fixed-order reduce).
- `csrc/fused/sm_90/tp_layer.cuh` — TP shard geometry/table/pack maps + sharded wgmma tile functions + reduce-point insertion map.
- `csrc/fused/sm_90/pp_stage_decoder_tc.cuh` — PP stage spec/ownership + stage kernels + launchers (patch-gated, loud).

**New JIT test bindings (committed):**
- `tests/hw/tp_loopback_binding.cu` — TP=P virtual ranks + unsharded/chunked references.
- `tests/hw/pp_stage_binding.cu` — the 3-launch PP=2 loopback step + ownership/layout introspection.

**New Python (committed):**
- `grokking_optimizers/parallel/pipeline.py` — 1F1B build/validate/driver, stage partition, handoff plan, LoopbackP2P.
- `grokking_optimizers/parallel/zero3.py` — FlatShardPlan, Zero3FlatParamStore (gather/release), sharded checkpoint save/load.
- `grokking_optimizers/parallel/distributed_step.py` — §6.2 `fused_train_step_distributed` (dependency-injected sharded apply).

**New tests (committed):** CPU (RUN, GREEN): `tests/test_pipeline_schedule.py`,
`tests/test_zero3_plan.py` — 55 passed together. GPU (NOT run):
`tests/hw/test_tp_loopback.py`, `tests/hw/test_pp2_loopback_determinism.py`,
`tests/hw/test_zero3_roundtrip.py`, `tests/hw/test_distributed_step.py`.

**Patches (tracked-file changes, NOT applied to the tree):**
- `.phase2/patches/0001-dectc-layer-range-pp.patch` — layer-range templates +
  fp32 boundary-adjoint args on `dectc_forward_tile`/`dectc_backward_tile`
  (model_stage_decoder_tc.cuh). REQUIRED by the PP stage header/test.
- `.phase2/patches/0002-parallel-init-exports.patch` — OPTIONAL re-exports in
  `grokking_optimizers/parallel/__init__.py` (modules import by path without it).

## 3. Compile + test proof (all on this CPU-only box, 2026-06-12)

| Artifact | Command shape | Result |
|---|---|---|
| TP headers harness (`tp_layer.cuh` chain) | `nvcc -c -std=c++17 -O2/-O3 --use_fast_math --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a -I repo (+torch incs)` | OK (`tp_compile_check.o`) |
| `tp_loopback_binding.cu` | same + `-DTORCH_EXTENSION_NAME=…` | OK (503 KB .o) |
| Patched shadow: production cell `mega_decoder_real_adamw_tc.cu` | shadow tree `/dev/shm/ppshadow` with patch applied | OK (769 KB .o) — proves every existing call site compiles unchanged |
| **PTX identity of the patch's default instantiation** | `nvcc -ptx` of the explicit `fused_decoder_megakernel_tc<OptId::AdamW>` instantiation, unpatched vs patched | **16,543 PTX lines each; sole delta = one `mov.u32 %r3655, 0;` scheduled one line earlier** (same instruction/register — scheduling jitter, zero semantic delta; within the §1.2 gate) |
| `pp_stage_binding.cu` (patched shadow) | same flags | OK (792 KB .o) |
| `pp_stage_binding.cu` vs UNPATCHED tree | same | `#error` fires with the apply instruction (the loud gate verified) |
| Both patches | `git apply --check` | clean |
| CPU tests | `pytest tests/test_pipeline_schedule.py tests/test_zero3_plan.py -q` | **55 passed** |
| ruff | all new python files | clean |
| Patched `__init__` (patch 0002) | shadow package import + smoke | OK |
| Tracked tree | `git status` | zero tracked files modified by this lane |

PTX-diff reproduction command (for the runbook's optional spot-check):
```bash
# unpatched vs patched (shadow tree with 0001 applied at /dev/shm/ppshadow):
TORCH_INC="-I $(python -c 'import torch.utils.cpp_extension as c;print(" -I ".join(c.include_paths()))') -I /usr/include/python3.11"
nvcc -ptx -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -DWITH_CUDA \
  -gencode arch=compute_90a,code=compute_90a -I <tree> $TORCH_INC \
  <probe instantiating fused_decoder_megakernel_tc<OptId::AdamW>> -o out.ptx
diff unpatched.ptx patched.ptx   # expect: 4 diff lines (one mov reordered)
```

## 4. Honesty register (no functionality suppression)

- The TP loopback is a TRANSPORT simulation, not a math simulation: the shard
  GEMMs, publish, rendezvous and fixed-order reduce are the real code path on
  the production wgmma tiles; the test's transport-neutrality assert (TP result
  == serial chunked-order, bit-exact) makes a fake-success loopback impossible.
- `NvshmemTransport` does not exist unless `-DSG_HAS_NVSHMEM` — selecting it
  without the toolkit is a compile error, never a silent fallback.
- The PP stage header `#error`s without the patch; the PP test SKIPS with the
  apply instruction. No unpatched silent path exists.
- `distributed_step` documents the §2.2 redundant-P3 inefficiency (the kernel's
  early-exit is bindings-lane wiring): correctness-identical (the discarded P3
  result is overwritten from the sharded shards), bit-gated by its world=1 test.
- Per-tensor optimizers (muon/SG11/SG15/SG2) are REJECTED loudly by the flat
  sharded apply (`ValueError` pointing at §2.3); their tensor-granular path is
  planned, not faked.
- The vit/mamba PP/TP twins are NOT authored (listed as follow-up): the design
  specifies the decoder flagship; claiming three-model PP coverage would be false.

## 5. Genuinely-needs-8× residual

See `.phase2/RUNBOOK.md` final section for the full list with swap points:
(1) NVSHMEM-TP transport validation + §5.4 go/no-go [THE residual];
(2) TP insertion into the production kernel body (transport-choice-dependent,
mechanical at the 4 marked points); (3) real scaling measurements (DP 1→8,
ZeRO-3 OOM threshold, PP bubble/microbatch sweeps); (4) cross-rank graph
capture with collectives; (5) PP real P2P transport swap; (6) vit/mamba twins
(1-GPU follow-up, not 8×-bound).
