# tuning/ — Exhaustive Analysis

**Scope:** `/workspace/SuperGrok1.5/tuning/` — 21 files, all read fully.

---

## 1. Directory Overview

The tuning package is the Python-side benchmark, profiler, sweep, and validation harness. It is organized into three tiers:

| Tier | Files | Purpose |
|------|-------|---------|
| **Primary drivers** | `flagship_distributed.py`, `_tp8_run.py`, `decoder_bench.py`, `mamba_bench.py`, `vit_bench.py`, `roofline.py`, `tune_optimizers.py`, `precision_analysis.py` | Main entry points for benchmarking/tuning |
| **Specialised validation** | `_grokadamw_multistep_parity.py`, `_prodigy_multistep_parity.py`, `_prodigy_owner_block_unit.py`, `_grokadamw_final_revalidate.py`, `_decoder_validate.py`, `_h3_dw_aaa.py`, `_embed_aaa.py`, `_h3_splitk_sweep.py`, `_mbtc_bypass_profile.py` | Targeted validation/sweep scripts |
| **Infrastructure** | `test_build_injection.py`, `__init__.py` | Build injection test; empty init |

---

## 2. flagship_distributed.py — 8-GPU Distributed Driver

**File:** `tuning/flagship_distributed.py` (498 lines)

### Role
Torchrun entry point for the flagship 1.5B decoder (d=1600, L=48, 1,475,884,899 params) across all 8 H100s. Claimed mesh: TP8 x DP1 x PP1 + ZeRO-3 (the "4D" operating point).

### Config-Derivation Mechanism (line-by-line)
The flagship mesh is NOT hardcoded — it is fully CLI-derived:

- `--tp` (default 8), `--pp` (default 1), `--dp` (default 1) are argparse args (lines 447-449).
- `--zero3` (default True) enables ZeRO-3 (no-op at DP=1) (line 451).
- Mesh rank coordinates computed at lines 132-135: `tp_rank = global_rank % tp`, `pp_rank = (global_rank // tp) % pp`, `dp_rank = global_rank // (tp * pp)`.
- nCTA is AUTO-derived or user-capped: `ncta = args.ncta_cap or fb.auto_ncta(args.opt, tp=tp, pp=pp, dp=dp, zero3=args.zero3, B=args.batch)` (line 142-143).
- Budget fit gate enforced at line 343: if `not bud.fits` → raise SystemExit with explicit OOM message.
- nCTA is adjusted to be a multiple of TP: `ncta = (ncta // tp) * tp; ncta = max(ncta, tp)` (lines 149-150).
- `ParallelConfig.validate_against_world(world)` asserts DP*TP*PP == world_size (line 204).
- `flagship_named_shapes()` builds exact per-tensor 2D shape table (lines 83-122) — validates the TP shard math produces `max_shard_numel == FLAGSHIP_NMAX // tp` exactly.

### Dry-Run Gate
`run_dry_plan()` (lines 195-230): CPU-only, validates mesh + TP weight shard + NVSHMEM bootstrap plan + budget for ALL ranks (or just this rank under torchrun). Exit 0 on success, SystemExit on OOM or shard drift.

### Live Run (`run_rank()`, lines 310-432)
Sequence:
1. (optional) import `grokking_optimizers.nvshmem_bringup_ext` to set NVSHMEM_DISABLE_NVLS=1 BEFORE init_process_group (line 316).
2. `dist.init_process_group(backend="nccl")`.
3. (optional, --nvshmem) NVSHMEM UID bootstrap via torch.distributed broadcast + `bringup_tp_team_live()` → carves TP team + mallocs symmetric heap (lines 361-387).
4. `seed_everything(args.seed)` + `build_flagship_module(tp, pp, dp, zero3, has_nvshmem)`.
5. `make_flagship_params(seed, device)` → flat fp32 `[FLAGSHIP_TOTAL_PARAMS]`.
6. State buffer: `torch.zeros(args.state_planes * FLAGSHIP_TOTAL_PARAMS)` (default 9 planes, covers SG2).
7. `make_real_batch(B, seed, device)` → real grokking_race_v2 data (p=97 modular arithmetic, tokens `[B,4]` int32, targets `[B]` int32).
8. Training loop: `mod.tc_train_step(params, tokens, targets, state, lr, beta1, beta2, eps, wd, bc1, bc2, step, ncta)` (line 406).
9. Cross-rank loss verification (lines 411-426): all_gather losses, assert `lmax < 1e-9`.

### CRITICAL DISCREPANCY: tc_train_step vs tc_train_step_tp8

**flagship_distributed.py calls `mod.tc_train_step(...)` at line 406**, not `tc_train_step_tp8`. Per `_tp8_scratch_pybind.cu` lines 5-9: "the committed mega_decoder_real_adamw_tc.cu pybind `tc_train_step` calls launch_fused_decoder_megakernel_tc<OptId::AdamW>() — the **SingleGPU template, tp_size=1, NO CommCtx** — so it NEVER fires the in-kernel TP all-reduce."

This means `flagship_distributed.py`'s `run_rank()` does NOT do actual in-kernel TP all-reduce — each rank runs an **independent single-GPU forward/backward** with the same seed/data. The cross-rank loss agreement check (lmax < 1e-9) trivially passes because all 8 ranks compute the SAME answer independently.

The REAL in-kernel TP all-reduce is only wired in `_tp8_run.py` which calls `tc_train_step_tp8` (which passes `tp_size=8` to the launcher's 18-arg `mega_decoder_real_adamw_tc(..., int tp_size)` function, reaching the ParTP8 arm).

**The `-DSG_FLAGSHIP_TP=8` compile flag** in `build_flagship_module()` (line 253) likely affects layout constants (the flagship layout header is force-included via `-include csrc/fused/sm_90/decoder_flagship_layout.cuh`), but does NOT make `tc_train_step` use the TP dispatch arm — that arm is gated on the RUNTIME `tp_size=8` argument, which only `tc_train_step_tp8` passes.

### Build flags for flagship module (lines 245-265)
- `-O3 -std=c++17 --expt-relaxed-constexpr`
- `-gencode=arch=compute_90a,code=sm_90a` + `compute_90a`
- `-DSG_TUNED_GEMM_IMPL=1`
- `-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1` (pre-defines the committed layout's include guard so flagship header wins)
- `-include csrc/fused/sm_90/decoder_flagship_layout.cuh` (force-includes flagship layout)
- `-DSG_FLAGSHIP_TP=8`, `-DSG_FLAGSHIP_PP=1`, `-DSG_FLAGSHIP_DP=1`, `-DSG_FLAGSHIP_ZERO=3`
- If `has_nvshmem`: `-DSG_HAS_NVSHMEM=1 -rdc=true -I${NVSHMEM_HOME}/include` + ldflags

### Post-build assertions (lines 273-276)
- `int(mod.D) == FLAGSHIP_D`
- `int(mod.LAYERS) == FLAGSHIP_LAYERS`
- `int(mod.TOTAL) == FLAGSHIP_TOTAL_PARAMS`

---

## 3. _tp8_run.py — Scratch 8-GPU TP Run Harness

**File:** `tuning/_tp8_run.py` (252 lines, marked NON-COMMITTED)

### Role
The ACTUAL 8-GPU in-kernel TP all-reduce runner. Explicitly calls `tc_train_step_tp8` (not `tc_train_step`).

### Hardcoded constraint (line 119)
```python
assert world == 8, "this is the 8-GPU TP8 run"
```
tp is also hardcoded to 8 at line 124 (pp=1, dp=1 implied). This is intentional for the scratch runner but means no generalization.

### Key sequencing (the NVSHMEM device-state order constraint)
**CRITICAL** ordering documented in lines 130-142 (the comments are precise and important):

1. **Build + dlopen the .so BEFORE nvshmem_init** (lines 130-142). Reason: `nvshmem_init` populates `nvshmemi_device_state_d` in every CUDA module loaded AT init time. The TP8 scratch .so is device-linked against its OWN copy of libnvshmem_device, so its in-kernel nvshmemx_barrier_block / nvshmem_ptr read that module's device state — which stays NULL (→ in-kernel IMA in the team barrier) unless the .so is dlopen'd BEFORE `nvshmem_init`. Rank 0 compiles; all ranks wait via `dist.barrier()` then dlopen.

2. **NVSHMEM bring-up** (lines 145-163): UID mint (rank 0) → torch.distributed broadcast → `mod_nv.init_with_uniqueid(rank, world, uid, local_rank)` → `team_split_strided(pe_start, pe_stride, pe_size)` → `malloc_symmetric_heap(sym_floats)`.

3. **Register device state** (line 171): `mod.init_device_state()` — calls `nvshmemx_cumodule_init` via the CUmodule recovered from the kernel symbol. MUST run AFTER nvshmem_init, BEFORE first step.

4. **Weight shard report + TC step loop**.

### Cross-rank tolerance (line 225-228)
Warns if `dloss >= 1e-6` (NOT a hard assert, unlike flagship_distributed.py's `1e-9` assert). Reports `RESULT_JSON` to stdout.

### Verification output (lines 240-243)
```python
print(f"[tp8] RESULT_JSON {'ran_8gpu': true, 'steps': ..., 'loss0': ..., 'lossN': ...,
      'descends': ..., 'cross_rank_agree': ..., 'per_rank_gib': ...}")
```

---

## 4. _tp8_build.sh — Manual 3-Step RDC Build

**File:** `tuning/_tp8_build.sh` (62 lines)

### Why manual (not torch JIT load)
Torch's `JIT load()` omits the `-dlink` (device-link) step for `-rdc=true` objects. Without device-link, `__cudaRegisterLinkedBinary` is unresolved and nvshmem device symbols (`nvshmemi_transfer_quiet`, `nvshmemi_device_state_d`) are undefined.

### 3-step build sequence
1. **Compile** (line 38): `nvcc $CUDA_CFLAGS -c _tp8_scratch_pybind.cu -o _tp8_scratch_pybind.cuda.o`
2. **Device-link** (lines 46-50): `nvcc -dlink -gencode=arch=compute_90a,code=sm_90a -Xcompiler -fPIC $BD/_tp8_scratch_pybind.cuda.o -L$NVSHMEM_HOME/lib -lnvshmem_device -lcudart -o _tp8_scratch_dlink.o`
   - Uses NAMED `-lnvshmem_device` (NOT `-l:libnvshmem_device.a`); the exact form leaves device symbols undefined (documented gotcha, line 44).
   - `-Xcompiler -fPIC` required: dlink object carries `__fatbinwrap` relocation that ld rejects without PIC (line 45).
3. **Host link** (lines 55-59): Produces `sg_tp8_scratch.so`. Includes `-lcuda` for driver API (`cuFuncGetModule` for `init_device_state`).

### Build dir
`SG_TP8_BD` env var (default `/workspace/.torch_ext/sg_tp8_scratch_wt`) — WORKTREE-SPECIFIC so it doesn't clobber main repo's cached .so.

---

## 5. _tp8_scratch_pybind.cu — TP8 Pybind Wiring

**File:** `tuning/_tp8_scratch_pybind.cu` (176 lines)

### Exposed functions
- **`tc_train_step_tp8`** (lines 90-157): One Fork-B TC step through the ParTP8 in-kernel all-reduce path. Calls `::sg::fused::sm90::mega_decoder_real_adamw_tc(..., tp_size=8)` at line 143.
- **`init_device_state`** (lines 61-73): Registers this .so's NVSHMEM device state via `nvshmemx_cumodule_init(CUmodule)`. Uses `cudaGetFuncBySymbol` + `cuFuncGetModule` to recover the CUmodule from the kernel symbol. MUST be called after `nvshmem_init`, before first step.
- **`D`, `LAYERS`, `TOTAL`, `NUMT`** module attributes (lines 171-174).

### ScratchParTP8 (line 47-48)
```cpp
using ScratchParTP8 = ::sg::fused::par::ParConfig<
    /*DP=*/8, /*TP=*/8, /*PP=*/1, /*SP=*/1, ::sg::fused::par::ZeROStage::Z3>;
```
This is the compile-time ParConfig that the megakernel specializes on for `init_device_state`'s symbol recovery.

### How it avoids editing committed sources
Includes the committed launcher directly: `#include "csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu"` (line 39). Gets the 18-arg `mega_decoder_real_adamw_tc` function verbatim. Compiled as ONE TU.

### Params layout note (lines 82-88)
"params [total] fp32 (the FULL flat decoder blob, replicated per rank — the kernel reads its TP-owned column/row shards via the layout; the host passes the whole blob and the kTPComm GEMM tiles index the rank's kTP slice)." This means each rank holds a FULL copy of all 1.476B parameters — no actual memory reduction per rank from TP here, which is the "A" bug (per-rank weight-shard offset) noted as still requiring the WIP patch.

### State layout (lines 103-109)
- loss_out = state + 3*total
- AdamW uses m (0..total), v (total..2*total), extra (2*total..3*total)
- Loss slot: state[3*total]

---

## 6. decoder_bench.py — Hill-Climb Benchmark + Profiler

**File:** `tuning/decoder_bench.py` (324 lines)

### Purpose
First-class coexisting bench-variant JIT build of `mega_decoder_real_adamw_tc.cu`. Supports:
- d=128 (production) or d=1024 (bench layout via `-DSG_DEC_BENCH_LAYOUT=1`)
- Per-phase clock64 profiling (`-DSG_DEC_PROFILE=1`)
- Fine GEMM sub-phase profiling (`-DSG_DEC_PROFILE_FWD_FINE=1`)
- Hill-climb knob overrides via `-D KEY=VAL`

### Build (lines 52-116)
Key: at d>=768, the scalar megakernel's `DecSampleSmem` exceeds 0x29000 bytes and ptxas hard-stops. Gate it OFF: `-DSG_DEC_SCALAR_MEGAKERNEL=0` (line 78). Production path (d=128) keeps the scalar path.

### Phase names (line 219-220)
```python
PHASE_NAMES = ["P1_fwd", "P1_bwd", "B1_barrier", "P2_dW_GEMM",
               "P2_grad_asm", "P3_opt_tail", "B2_barrier", "B0_barrier"]
```
8 slots. P2_dW_GEMM is index 3.

### Fine sub-phase names (lines 258-259)
```python
sub_names = ["ISSUE(cp.async)", "WAIT(drain)", "WGMMA(mma)", "EPI(store)", "BARRIER(sync)"]
ph_names = ["P1_fwd ring", "P1_bwd(dX) ring"]
```
Diagnosis: WAIT-dominant → DRAIN-bound; WGMMA/EPI-dominant → compute/epilogue-bound.

### FLOPs formula (lines 119-139)
`T = B * seq`. Per layer: in_proj (T×3d×d), out_proj (T×d×d), ff.0 (T×4d×d), ff.2 (T×d×4d). Head: B×vocab×d. fwd+bwd ≈ 3× forward. Attention score GEMMs (seq=4, negligible) included.

### Roofline fraction (line 232)
Reports against 989 TF/s bf16 dense roofline (H100 BF16 tensor-core peak).

### Host SM clock (line 48)
`SM_GHZ = 1.98` (H100 SM boost clock = 1980 MHz).

---

## 7. mamba_bench.py, vit_bench.py

Both mirror `decoder_bench.py` structurally.

### mamba_bench.py (166 lines)
- TU: `mega_mamba_real_adamw_tc.cu`
- `-DSG_MB_BENCH_LAYOUT=1` at d=1024, `-DSG_MB_SCALAR_MEGAKERNEL=0`
- `MambaSampleSmem` overflows ~228 KB/SM at d=1024 (documented reason for gating scalar megakernel off)
- Exposes `mod.DINNER` (d_inner), `mod.DTRANK` (dt_rank), `mod.STATE`, `mod.SEQ`
- FLOPs formula: per layer: in_proj (T×2d_inner×d), x_proj (T×dbc×d_inner), dt_proj (T×d_inner×dt_rank), out_proj (T×d×d_inner). Plus head.
- Uses oracle constants from `tests.hw.mamba_oracle` for VOCAB/P_HEAD/SEQ.

### vit_bench.py (234 lines)
- TU: `mega_vit_real_adamw_tc.cu`
- `-DSG_VIT_BENCH_LAYOUT=1` at d=1024, `-DSG_VIT_SCALAR_MEGAKERNEL=0`
- `VitSampleSmem` overflows 227 KB dynamic-smem cap at d=1024
- Per-phase profiling: 6 slots via `SG_VIT_PROFILE=1`, `g_vit_prof_max[6]`
- VIT_PHASE_NAMES: `["P1_fwd", "P1_bwd", "B1_barrier", "P2_dW_GEMM", "P2_grad_asm", "P3_opt_tail"]`
- Input: float32 patches `[B, NUM_PATCHES, PATCH_DIM]`, targets int32 `[B]`

---

## 8. roofline.py — Full H100 Roofline Analysis

**File:** `tuning/roofline.py` (1038 lines)

### H100 peaks (lines 44-54)
```
fp32_cuda: 66.9e12
tf32_tc:   494.7e12
bf16_tc:   989.4e12  ← the "989 TF/s" referenced throughout
fp8_tc:    1978.9e12 (dense, no sparsity; FP8 with sparsity = 3958)
int8_tc:   1978.9e12
hbm_bw:    3.35e12   (HBM3)
```

### State tensors per optimizer (lines 108-133)
Source-verified counts (load-bearing for bytes_per_step):
| Optimizer | State tensors |
|-----------|---------------|
| adamw | 2 |
| lion | 1 |
| grokfast | 3 |
| grokadamw | 3 |
| muon | 2 |
| prodigy | 4 |
| neuralgrok | 2 |
| looksam | 3 |
| supergrok | 4 |
| supergrok15 | 4 |
| supergrok2 | 9 |

supergrok2 = 9 because it has 5 param-sized tensors (exp_avg, exp_avg_sq, mu, slow, sharpness) + gru_states of shape (N, gru_hidden=4) = 4 elem/param → 5+4=9.

### 3-pass measurement (lines 354-443)
1. **Wall pass**: 25 warmup steps (discard), then diff of 15 vs (15+timed_steps) runs to cancel per-run setup overhead. Peak VRAM captured here (production path, use_fused=True).
2. **Per-10-step series**: eval_callback hook for per-10-step timing.
3. **FLOPs pass**: `torch.profiler(with_flops=True)` with `use_fused=False` — CRITICAL: fused megakernel registers 0 FLOPs in profiler (opaque kernel). FLOPs/step is dtype-AND-path-independent, so eager-profiled FLOPs / L3-megakernel wall = valid achieved TF/s.

### Batch saturation sweep — measured real data (lines 74-101)
**Key honest negative result** embedded in the source:
- Megakernel (TC, 1 CTA/SM) saturates at B≈2k (throughput 142k-143k samples/s at B=2048-4096), then DECLINES to 128k at B=65k+.
- Eager (lion, cuBLAS): saturates much later (~32-65k), peaks ~1.9M samples/s.
- **Verdict**: "the path to higher megakernel fraction is multi-CTA-per-tensor tiling, NOT a larger batch."
- The 16384 operating point is chosen as fair shared comparison (megakernel within 3.5% of its B=2048 peak; eager at its knee).

### Path detection (lines 296-402)
Wraps `g._try_fused_train_step` and `g._try_fused_step` to count L3/L1 firings and capture `g.LAST_L3_ENGINE` ("wgmma" or "scalar"). Engine-driven ceiling: wgmma → bf16 989TF, scalar → fp32 66.9TF.

### force-precision default (line 859)
`--force-precision bf16` (default). The old fp32-force existed because pre-wiring the precision gate declined L3 at bf16. Now that wgmma is live, bf16 is the correct measurement point.

### TC direct measurement (lines 528-688)
`measure_tc_cell()`: JIT-loads the *_tc.cu TUs directly, times both `tc_train_step` (bf16-TC) and `scalar_train_step` (fp32). Reports TC-faithful AI (acts stored bf16 → halved activation byte term) AND fp32-model AI for cross-row comparability.

---

## 9. tune_optimizers.py — 11-Optimizer Hyperparameter Tuner

**File:** `tuning/tune_optimizers.py` (694 lines)

### Architecture
- Optuna-based, JournalStorage (file-locked append-only .log)
- TPESampler (multivariate=True, n_startup_trials=12, 10_000+worker_seed)
- MedianPruner (n_startup_trials=12, n_warmup_steps=490, interval_steps=1)
- First prunable rung at step 500 (report every 10th step, 490 warmup means step-500 rung is first viable)

### Kill-switch (lines 62-65)
```python
if (ROOT / ".STOP_TUNING").exists():
    sys.stderr.write("[tune_optimizers] .STOP_TUNING sentinel present — refusing to run.\n")
    sys.exit(0)
```
Added 2026-06-16 to stop a runaway tuning relaunch-loop.

### Objective scoring (lines 365-382)
- **Grokked + test-confirmed (test_acc >= 0.90)**: value = grokking_step (∈ [1, cap])
- **Fake-grok (val-grok but test < 0.90)**: value = cap + (1 − final_test_acc) × cap
- **DNF**: value = cap + (1 − max(peak_val, 0.5×peak_train)) × cap
  - `0.5×peak_train` gives TPE gradient in the memorize-but-no-grok region
- **INFRA crash (OOM, CUDA error, TypeError)**: PRUNED (no score, doesn't poison region)
- **Config crash**: value = 2×cap + (1 − last_peak_val) × 100 (gradient-carrying)

### Caps (line 69-71)
- decoder: 5000 steps
- vit: 6000 steps  
- mamba: 12000 steps

### Confirm mode (lines 522-589)
Top-5 configs re-run on CONFIRM_SEEDS = (1002, 1003, 1004, 1005). Winner = min(-n_confirm_grokked, median_steps). Must grok ALL 4 confirm seeds to be "robust_all". Writes `results/tuning/tuned_configs_{model}.json`.

### No-suppression bounds
All search space lower bounds are > 0 (e.g., supergrok_meta_lr lower bound = 1e-5, not 1e-6, to prevent meta-net from silently zeroing its contribution). Architecture/compiled-shape knobs are FIXED (never tuned).

### Orphan-trial sweep (`_sweep_orphan_trials`, lines 428-461)
On worker startup, fails RUNNING trials older than 2 hours (dead-worker recovery). Per-trial try/except so a storage hiccup doesn't abort the sweep.

---

## 10. precision_analysis.py — 7-Arm Precision Grid

**File:** `tuning/precision_analysis.py` (310 lines)

### Grid
7 precisions × 3 models × 11 optimizers × 3 seeds = 693 runs.

Arms: `("fp32", "tf32", "bf16", "fp16amp", "fp8", "fp8e5m2", "int8")`

### Resumable via JSONL (lines 74-92)
`results/h100_grokking_race/precision_analysis_full.jsonl` — flock'd appends. On restart, diffs grid against recorded (precision, model, optimizer, seed) keys.

### Dead-stop heuristic (lines 136-139)
At `step >= cap//2` with `train_acc < 0.30` → abort (never memorized).

### Fake-grok guard (line 166-167)
`test_confirmed = grokking_step is not None and final_test_acc >= 0.90`

### Partial-crash accounting
Explicit tracking of crashes vs dead-stops vs successful runs per cell (not conflated).

---

## 11. test_build_injection.py — Build Injection Unit Test

**File:** `tuning/test_build_injection.py` (577 lines)

### Purpose
CPU-only unit test (no GPU, no build) for `grokking_optimizers/_tuned_inject.py`. Verifies:
1. TU path → optimizer mapping (ambiguous TUs → None)
2. Per-source nvcc flag computation from flat schema
3. build.ninja rewrite (cuda_post_cflags override injection)
4. export_winner read-merge-write round-trip (atomic JSON, nested per-model schema)
5. Drift guard: macros in _tuned_inject.py MACROS table match `#define SG_TUNED_*` defaults in live csrc/ headers

### Drift headers checked (lines 401-408)
```python
_DRIFT_HEADERS = (
    "csrc/backends/cuda/sm_90/wgmma.cuh",
    "csrc/backends/cuda/sm_90/tile_pipeline.cuh",
    "csrc/fused/megakernel_common.cuh",
    "csrc/fused/sm_90/model_stage_decoder_tc.cuh",
    "csrc/fused/sm_90/model_stage_vit_tc.cuh",
    "csrc/fused/sm_90/model_stage_mamba_tc.cuh",
)
```

### Torch-integration test (lines 451-549)
Drives torch's real `_write_ninja_file_and_compile_objects` through a monkeypatch (same pattern setup.py uses) to verify the ninja parser matches torch's actual emission/escaping — not a synthetic fake.

---

## 12. Specialised Validation Scripts

### _decoder_validate.py (74 lines)
Pipeline: (1) `wiring_check --models decoder` for all 11 optimizers → expect L3-TC paths; (2) `test_l3tc_tail_gate.run_cell_gate` for adamw/lion/grokfast/decoder; (3) decoder TC TF/s re-measure via `roofline.measure_tc_cell`. Summary: "ALL GREEN" if ≥3/11 L3-TC + all tail gates pass.

### _h3_splitk_sweep.py (68 lines)
dW split-K G sweep at G ∈ {1,2,4,8} × d ∈ {1024,128}. Reports wall + P2_dW_GEMM phase ms. "Winner at both scales" requires beating the current default G=4 beyond noise at BOTH d (the GATE discipline).

### _embed_aaa.py (49 lines)
A/A/A bit-identity check: 3 runs of `tc_train_step` from identical inputs → sha256 of grad tensor must be identical. Also prints cross-run max-abs delta (should be exactly 0.0).

### _h3_dw_aaa.py (126 lines)
Two checks on d=1024 decoder TC cell:
- (A) Determinism: 3 runs, sha256 of (params+state+loss) must be identical.
- (P) Parity vs baseline ref: `git archive` the baseline ref, build from that tree, compare one step's (params+state+loss) sha256. The dW M-atom interleave reorders independent MMAs → must be bitwise-exact.

### _grokadamw_multistep_parity.py (363 lines)
80-step parity test for grokadamw. Runs TC kernel + eager reference + 2 controls side-by-side:
- Control (ii): clip-OFF (grad_clip=1e30) — must diverge ≥ 1e-3 from clip-ON ref to prove clip is load-bearing
- Control (iii): static-α (no grokking signal ever set) — must diverge ≥ 1e-3 to prove adaptive-α is load-bearing
- Tolerance: TOL=1e-4 for kernel vs eager (fp32 reorder is ~1e-7, dropped mechanism ~1e-3)
- Also includes `_verify_race_alpha_path()` which runs real `train_grokadamw` for > alpha_freq steps to verify production α-cadence.

### _grokadamw_final_revalidate.py (70 lines)
Orchestrates: (1) single-step tail gates for 4 cells; (2) runs `_grokadamw_multistep_parity.py` TWICE to verify P2.5 clipped-regime reduction is bit-deterministic (A==B bitwise worst-errors).

### _mbtc_bypass_profile.py (102 lines)
Differential bypass profiler for Mamba TC megakernel. Builds 4 variants with bypass defines (`SG_MBTC_BYPASS_DW_GEMM`, `SG_MBTC_BYPASS_SCAN`, `SG_MBTC_BYPASS_EMBED_SCAN`) to attribute wall to phases by delta. Writes `results/h100_grokking_race/mbtc_bypass_profile.json`. ncu is blocked in the container (ERR_NVGPUCTRPERM).

### _prodigy_multistep_parity.py (257 lines)
80-step parity for prodigy. Tracks (m, v, s) state + the adaptive d_lr (state[4*total+3]). Control: d_coef=0 freezes d at d0. Tolerance TOL=1e-4. NaN-mask guard: explicitly checks all worst-* values are finite (max(x, NaN) = x in Python, would false-green). Production d0=1e-6/d_coef=1.0 is typically inert (d never trips over d0 in 80 steps for decoder/vit).

### _prodigy_owner_block_unit.py (135 lines)
Unit test for prodigy P2.6 d-update arithmetic (unreachable by multi-step parity when d is inert). Injects controlled delta (param_init = p_e + delta), resets persisted r_ema/s_ema/d_lr to cold start, runs ONE kernel step, compares persisted scalars to canonical fp64 formula. Key: d_coef≠1 makes the "persist UNSCALED, scale only candidate" line load-bearing (d_coef=1 makes it invisible).

---

## 13. Config-Derivation Summary (tuning/layer)

| Mechanism | Where | How |
|-----------|-------|-----|
| Parallelism (tp/pp/dp) | flagship_distributed.py:130-155 | CLI args → rank coords → build_host_plan |
| nCTA selection | flagship_distributed.py:142-151 | `fb.auto_ncta(opt, tp, pp, dp, zero3, B)` or --ncta-cap; clamped to multiple of TP |
| Budget gate | flagship_distributed.py:343-345 | `fb.per_rank_budget()` → OOM = hard SystemExit |
| TP weight shard | flagship_distributed.py:138-139 | `partition_tensor_parallel(flagship_named_shapes(), tp, tp_rank)` |
| TP shard correctness | flagship_distributed.py:165-188 | Assert `nmax == FLAGSHIP_NMAX // tp` |
| NVSHMEM bootstrap | flagship_distributed.py:361-387 | `bringup_tp_team_live()` → real NVSHMEM team; or dry-run `bootstrap_tp_team(allow_dry=True)` |
| Kernel specialization | _tp8_scratch_pybind.cu:47-48 | `ScratchParTP8 = ParConfig<DP=8,TP=8,PP=1,SP=1,ZeRO::Z3>` for init_device_state |
| Build variant isolation | decoder_bench.py:95-100 | Distinct module name + build dir per (d, profile, defines) |

---

## 14. Discrepancies vs Claimed State

### CRITICAL: flagship_distributed.py does NOT fire in-kernel TP all-reduce
- **Claimed**: "ONE 1.5B model spread across all 8 GPUs... in-kernel TP all-reduce"
- **Actual**: `run_rank()` calls `mod.tc_train_step(...)` (line 406) — the COMMITTED pybind which uses tp_size=1 (SingleGPU template, per _tp8_scratch_pybind.cu comment lines 5-9). Each rank does an independent single-GPU forward with the same seed. The cross-rank loss agreement (lmax < 1e-9) passes trivially.
- The REAL TP path is `_tp8_run.py` → `tc_train_step_tp8` (tp_size=8).

### _tp8_run.py has hardcoded `assert world == 8`
- Line 119: `assert world == 8, "this is the 8-GPU TP8 run"`.
- Also hardcodes `tp = 8` at line 124.
- This is flagged as "NON-COMMITTED scratch runner" so it is intentional, not a design violation.

### Loss agreement tolerance difference
- flagship_distributed.py: `assert lmax < 1e-9` (hard fail)
- _tp8_run.py: `if dloss >= 1e-6: cross_ok = False; print(WARNING)` (soft warn)
- The harder tolerance in flagship_distributed.py is "easier" to pass since each rank runs independently (same answer deterministically).

### Batch saturation: megakernel saturates at B≈2k, NOT 16k
- The BATCH_SATURATION_SWEEP data (lines 74-101 of roofline.py) explicitly shows megakernel throughput peaks at B=2048-4096 (~142k samples/s) and DECLINES to 128k at B=65k+.
- The "B≈16k" floor choice is for a "fair shared comparison with eager" — not the megakernel's actual saturation point.

### tune_optimizers.py `_tp8_scratch_pybind.cu` DP=8 in ParConfig
- `ScratchParTP8 = ParConfig<DP=8, TP=8, PP=1, SP=1, ZeRO::Z3>` — DP=8 in the ParConfig seems odd for a pure-TP8 mesh (DP should be 1). This may be a carry-over or the DP dimension in ParConfig has a different meaning here (perhaps the total world size). Not confirmed without reading the ParConfig definition.

---

## 15. Open Items / Bugs / TODOs

1. **TP data-path bugs (A, B, C)**: Per RESUME.md, bugs A (per-rank weight-shard offset) and B (25-heads-not-%8 attention) are fixed in `phase6/tp_datapath_fix_WIP.patch` (ungated). Bug C (IMA from bug B) unconfirmed. These prevent `_tp8_run.py` from producing correct results.

2. **flagship_distributed.py run_rank() doesn't call tc_train_step_tp8**: As detailed in §3 above. The function sets up NVSHMEM infrastructure (when --nvshmem passed) but the kernel call falls back to tp_size=1. The fix would need to either (a) use `_tp8_scratch_pybind.cu`'s `tc_train_step_tp8`, or (b) expose a `tc_train_step_tp8` in the flagship module's committed pybind.

3. **ScratchParTP8 DP=8**: May be incorrect (should be DP=1 for a TP8-only mesh). Needs verification against `par::ParConfig` definition.

4. **precision_analysis.py: fp8/int8 arms** may require `lowprec.py` module that implements the custom fp8/int8 training paths. Not verified to be implemented.

5. **tune_optimizers.py results**: The RESUME.md notes "11-opt decoder ranking (overfit placeholder)" — the tuning study results are not confirmed as real production results yet.

6. **Real-data benchmark**: Still pending (RESUME.md states as next after TP data-path fix).

7. **Full 33-cell roofline**: Still pending (RESUME.md).
