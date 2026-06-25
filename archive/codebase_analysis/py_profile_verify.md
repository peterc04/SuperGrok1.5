# Python Profiling & Verification Stack — Deep Read Digest
## Files: profile.py (740L), profile_maximal.py (762L), utilization.py (722L), verify_all.py (699L)

Generated: 2026-06-25

---

## 1. FILE-BY-FILE GROUND TRUTH

### 1.1 `profile.py` (740 lines) — Arch-native profile capture

**Purpose:** Standalone arch-native profiler dispatcher. Companion to `compile.py`. Wraps ncu/rocprof/jax.profiler around a generated "smoke" script (one `opt.step()`) for any (optimizer, model, arch) triple.

**Key constants:**
- `ARCHES` (profile.py:85-98): 30-entry tuple covering sm_70..sm_120a (incl. aliases), gfx906..gfx1201, tpu_v4..tpu_v7. This is the PROFILE-side arch list; the MEGAKERNEL_ARCHS are only `sm_90/gfx942/tpu_v6e`.
- `NCU_FLAGS` (profile.py:110-122): 7 sections: `ComputeWorkloadAnalysis, LaunchStats, MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats, InstructionStats, Occupancy`. Flags: `--set full --target-processes all --import-source yes --source-folders csrc/`.
- `ROCPROF_FLAGS` (profile.py:124-127): `--hip-trace --hsa-trace --stats --basenames on --timestamp on`.

**Imports from dispatch (profile.py:74-76):**
```python
from grokking_optimizers.dispatch import (
    OPTIMIZERS, SHORT_MODELS as MODELS, OPT_CLASS,
)
```
- `OPTIMIZERS`: 11 names (adamw, grokadamw, grokfast, lion, looksam, muon, neuralgrok, prodigy, supergrok11, supergrok15, supergrok2)
- `MODELS` (as `SHORT_MODELS`): 3 SHORT names — `('decoder', 'vit', 'mamba')` NOT the canonical ('transformer_decoder', 'vit', 'mamba3')
- `OPT_CLASS`: dict of optimizer name → Python class name

**Architecture dispatch (profile.py:538-547):**
- Vendor `cuda` → `profile_cuda()` (calls ncu or falls back to smoke-only if ncu absent)
- Vendor `hip` → `profile_hip()` (calls rocprof-compute/rocprofv2/rocprof or smoke-only)
- Vendor `pallas` → `profile_pallas()` (jax.profiler.start_trace/stop_trace in-process)

**ncu blocked path (profile.py:452-455):**
```python
if shutil.which("ncu") is None:
    report.write("  [profile] ncu not in PATH; running smoke only.\n")
    return run_capture([sys.executable, script], report, timeout=timeout)
```
- ncu is OPTIONAL — if blocked/absent, the smoke script (one `opt.step()`) still runs but no counter data is collected. There is NO in-kernel `clock64` path and NO analytical FLOP/byte computation in `profile.py` itself — the "ncu-counter-free" path is just smoke execution.

**Pallas profiling (profile.py:499-535):**
- Imports `jax.profiler` INSIDE the function body (graceful skip if absent)
- Uses `jax.profiler.start_trace(tdir)` / `stop_trace()` in-process around an `exec()` of the smoke script
- Reports trace file sizes only (no counter extraction)

**`_DebugTee` (profile.py:554-581):** Mirrors writes to stderr when `debug=True`. Clean implementation.

**Smoke script generator (profile.py:246-377):**
- CUDA/HIP: imports optimizer class from grokking_optimizers, creates 64×64 float32 param, does one backward + `opt.step()`.
- Pallas: imports launcher by absolute path via `importlib.util`, exercises `launch_{opt}_step_jit` or first `launch_*` function. Best-effort auto-invoke with synthesized args; module load is the hard requirement.
- Profile report is written to `build/profiled/profile_{opt}_{model}_{arch}.txt`

**Path inference (profile.py:391-439):**
- `csrc/backends/cuda/sm_90/...` → arch = sm_90
- `csrc/backends/hip/gfx942/...` → arch = gfx942
- `csrc/backends/pallas/...` → arch = tpu_v6e
- `grokking_compiled_{opt}_{model}_{arch}/*.so` → extracts all three
- Only infers sm_90/gfx942/tpu_v6e from path structure (not sm_90a, etc.)

**`_arch_info()` (profile.py:105-108):** Lazy accessor that imports `compile.ARCH_INFO` to avoid circular imports. `compile.py` imports several names from `profile.py` at module-load time.

---

### 1.2 `profile_maximal.py` (762 lines) — Binary maximality proof

**Purpose:** Binary inspection + functional test to prove compiled artifacts are MAXIMAL (tensor cores, TMA, no spills) AND correct (optimizer descends). Five tiers:

**Architecture:**
```python
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP-silicon"
```
- `Probe` dataclass (tier, name, status, detail, seconds)
- `Report` collects probes; prints live; silicon-only items listed at end.

#### TIER A — sm_90 GEMM Instruction Maximality (profile_maximal.py:315-353)
**4 TUs checked:**
```python
_GEMM_TUS = [
    ("csrc/backends/cuda/sm_90/models/decoder.cu", "decoder"),
    ("csrc/backends/cuda/sm_90/models/vit.cu", "vit"),
    ("csrc/backends/cuda/sm_90/models/mamba.cu", "mamba"),
    ("csrc/backends/cuda/sm_90/launch_supergrok2.cu", "supergrok2"),
]
```
**Compile flags:** `_MAX_NVCC` = `-c -std=c++17 -O3 --use_fast_math -DNDEBUG -DWITH_CUDA -DWITH_CUTLASS -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda`

**SASS audit via `cuobjdump -sass`:**
- `WGMMA` count (regex `\b[HUW]?GMMA\b`)
- `TMA` count (regex `UTMALDG|UBLKCP|cp\.async\.bulk`)
- `mbarrier` count
- `C7509` wgmma-serialization warnings
- Spill bytes

**Pass condition:** `wgmma > 0 AND tma > 0 AND c7509 == 0 AND unexpected_spill_bytes == 0`

**Skip condition:** `nvcc` absent → all 4 are SKIP-silicon.

#### TIER B — sm_90 Resource Health (profile_maximal.py:374-415)
**Cell sample (4 cells from 33):**
```python
_CELL_SAMPLE = [
    ("mamba3", "adamw"), ("mamba3", "supergrok2"),
    ("transformer_decoder", "muon"), ("vit", "prodigy"),
]
```
**Optimizer kernels (3 of 11):**
```python
_OPT_KERNELS = ["launch_adamw", "launch_supergrok2", "launch_muon"]
```
- Cross-checks real ptxas regs/smem against `mk.solve()` plan + `mk._arch_budget()` budget.
- Gate: `unexpected_spill_bytes == 0 AND regs_max <= budget.max_regs_per_thread`

#### THE ALLOWLIST — `_DEAD_SPILL_KERNEL_TOKENS` (profile_maximal.py:161-229)

The allowlist has EXACTLY ONE exception:
```python
_DEAD_SPILL_KERNEL_TOKENS: Tuple[str, ...] = (
    "MMA_64x128x8_F32TF32TF32_RS_TN",           # RS TF32 atom
    "MainloopSm90TmaGmmaRmemAWarpSpecialized",   # RmemA collective (RS path)
    "tfloat32_t",                                # element type
)
```
- Allowlisted kernel: TF32 A-transposed register-source GEMM. 8B spill. Runtime-unreachable via `vit_run_gemm_atb<float>` which returns `cudaErrorNotSupported` for float (routes to scalar fallback per `mma.cuh:~L720-740` DOCUMENT-STOP).
- Self-check at import time (lines 222-229): asserts the dead RS kernel MATCHES and the live SS kernel does NOT.
- Match uses ALL THREE tokens to be fail-closed.

**Per-kernel attribution (profile_maximal.py:253-281):** `_PTXAS_SPILL_BLOCK` regex pairs each spill count with its mangled kernel name to ensure allowlist is per-kernel (not per-TU), preventing one dead-code exception from masking a real spiller in the same TU.

**"11/0 allowlist" claim analysis:** The allowlist currently exempts EXACTLY 1 kernel (the dead TF32-RS). The "11/0" mentioned in session context likely refers to: the Tier B opt-kernel check only covers 3 of 11 optimizer launchers (`_OPT_KERNELS`). The other 8 launchers (grokadamw, grokfast, lion, looksam, neuralgrok, prodigy, supergrok11, supergrok15) are NOT checked for resource health in Tier B — this is a coverage gap, not an allowlist bug. Alternatively, "11/0" could refer to 11 subtests in the compile.py self-test for device_profiling all failing (0 passing) — see §2 below.

#### TIER C — gfx942 ISA Maximality (profile_maximal.py:477-506)
**4 kernels checked via `bash scripts/amdgcn_check.sh --emit-obj`:**
- attention, mamba3, decoder, vit
- Disassemble with `llvm-objdump -d`; count `v_mfma` + DPP ops
- Read kernel descriptor from `llvm-readobj --notes` for VGPR/SGPR/LDS
- Pass: `v_mfma > 0 AND vgpr_max <= budget.max_regs_per_thread`
- Skip: `clang` or `llvm-objdump` absent

#### TIER D — Functional Correctness (profile_maximal.py:513-609)
**Two reference-step descents:**
- `ref_adamw_step` loaded from `tests/hw/test_reference_parity.py` via `importlib`
- 200-step Adam descent: minimize `||w-t||²` from `w=[5,-3,1]`; requires loss drop 1000×
- 400-step Lion descent: same problem
- Skip if `torch` unavailable or parity module unloadable

**Fused program trace check:**
```python
from csrc.backends.pallas._pallas_fused import trace_check
r = trace_check("mamba3", "adamw", "L3")
```
- Skip if jax/pallas absent

#### TIER E — tpu_v6e HLO Maximality (profile_maximal.py:627-671)
- 4 cells: `(mamba3/adamw), (transformer_decoder/muon), (vit/supergrok2), (mamba3/prodigy)`
- Checks `from csrc.backends.pallas.v6e import TILE_SIZE == 256`
- Calls `profile_cell(model, opt, "L3")` for dot/fusion/finite checks
- Skip if jax absent

**Total probes per run (if all tools present):** A=4, B=7, C=4, D=3, E=5 → max 23 probes.

**ncu-counter-free note:** profile_maximal.py does NOT use ncu for counter data. Tier A/B use `nvcc -c` + `cuobjdump -sass` for SASS inspection and `ptxas -v` for resource numbers — these work without live GPU and without ncu. Silicon-only items (latency, occupancy, SM duty cycle) are listed in the final report as SKIP-silicon.

---

### 1.3 `utilization.py` (722 lines) — Live device utilization sweep

**Purpose:** Low-overhead background poller for all 33 pipelines per arch. Distinct from `profile.py` (one-shot ncu/rocprof) — this is a sustained-load sweep.

**`PIPELINES_PER_ARCH = len(OPTIMIZERS) * len(MODELS)` = 33 (utilization.py:60)**

**Imports from `profile.py` (utilization.py:48-57):**
```python
from grokking_optimizers.profile import (
    ARCHES, MODELS, OPTIMIZERS, REPO_ROOT, _arch_info,
    child_env, make_progress, write_temp_script,
)
```

**Sampler backends (utilization.py:182-396):**
| Class | Backend | Primary | Fallback |
|-------|---------|---------|---------|
| `_NvmlSampler` | `nvml` | pynvml `nvmlDeviceGetUtilizationRates` | `nvidia-smi --query-gpu` |
| `_RocmSmiSampler` | `rocm-smi` | amdsmi `amdsmi_get_gpu_activity` | `rocm-smi --showuse --json` |
| `_TpuSampler` | `jax-tpu` | `jax device.memory_stats()` | None |

**Crash-hard contract (utilization.py:27-32, 119-122, 168-172):**
- Missing device/library → raises immediately (no graceful degradation)
- Mid-run sampling failure → stash exception in `self._error`, re-raise in `stop()` — never returns partial sample as clean
- Zero samples → raises RuntimeError

**TPU limitation (utilization.py:313-369):**
- `compute_pct = None` for TPU — MXU duty-cycle is NOT exposed by JAX `device.memory_stats()`. Only HBM utilization is available. Documented explicitly as xprof-only.

**Workload script (utilization.py:433-487):**
- CUDA/HIP: 4096×4096 float32 param, `iters` steps of backward + `opt.step()` + sync
- Pallas: `csrc.fused.tpu_v6e.mega_{model}_{opt}.verify()` loop
- Script signals readiness via `print('WORKLOAD_READY', flush=True)`

**`track_all()` (utilization.py:571-601):** Shares ONE sampler across all 33 cells (setup/teardown once). If any cell fails, propagates the exception immediately (no partial results).

**Aggregation (utilization.py:403-425):** Drops first sample (warmup), computes mean/peak for compute% and mem%.

**Output:** JSON at `build/utilization/utilization_{arch}.json` + fixed-width table to stdout.

**Toolchain check (verify_all.py:5e / utilization.py:494-563):**
```python
# verify_all phase 5e
u.track_cell("adamw", "mamba", "sm_90a", iters=1, timeout=10)
# must raise RuntimeError on a no-GPU host
```

---

### 1.4 `verify_all.py` (699 lines) — End-to-end gate

**Purpose:** Single authoritative verification harness. 6 phases. Imports from `megakernel` (mk) and `megakernel_codegen` (cg).

**Constants (verify_all.py:57-60):**
```python
OPTIMIZERS = mk.OPTIMIZERS   # 11
MODELS = mk.MODELS            # 3 canonical ('transformer_decoder', 'vit', 'mamba3')
ARCHES = mk.MEGAKERNEL_ARCHS  # ('sm_90', 'gfx942', 'tpu_v6e')
N_CELLS = 11 * 3 * 3 = 99
```

**OOM-kill retry (verify_all.py:144-171):**
- CUTLASS-heavy nvcc compiles may be SIGKILL'd by OOM when running concurrently
- Detection: `rc in (-9, 137)` OR `out.strip().endswith("Killed")`
- Retry: hold `_EXCLUSIVE` lock so the single retry runs with full memory

**Phase 1 — Structural inventory (verify_all.py:184-226):**
- 11 algorithm headers in `csrc/algorithms/`
- 11 launchers per arch (sm_90/gfx942/pallas), template `launch_{o}.{ext}`
- 99 fused cells via `_cell_path(model, opt, arch)` → `csrc/fused/{arch_dir}/mega_{model}_{opt}{ext}`
- 3 dispatch tables: `fused_dispatch_table.inc` (sm_90 + gfx942) + `fused_wired_cells.inc`

**Phase 2 — Component compile (verify_all.py:232-261):**
- sm_90: `bash scripts/compile_to_object.sh {tu} -DWITH_CUTLASS` for 11 launchers + 3 models
- gfx942: `bash scripts/amdgcn_check.sh --header {hdr}` for headers
- Both skip if nvcc/clang absent

**Phase 3 — Modular composition (verify_all.py:265-301):**
- sm_90: nvcc compile all 33 cells × OOM-resilient
- gfx942: AMDGCN device gate via `scripts/amdgcn_check.sh --cell`
- tpu_v6e: `importlib.import_module(f"csrc.fused.tpu_v6e.mega_{m}_{o}")` + `mod.verify()` (CPU jax, max 4 workers)
- Silicon-gated: gfx942 hipLaunchKernelGGL host path, tpu_v6e on-device execution

**Phase 4 — Maximality (verify_all.py:388-479):**
- 4a: All 99 cells feasible (≥L1)
- 4b: Every cell at its MAX feasible tier (independent re-derivation of solver decisions)
- 4c: Every cell within register+smem budget
- 4d: CODEGEN IDEMPOTENCY — `cg.emit_cell()` must produce byte-identical output to committed file for all 99 cells
- 4e: Tier comment in committed cell matches solver live decision
- 4f: `scripts/check_math_single_source.py` drift guard

**Phase 5 — Cross-validation (verify_all.py:489-563):**
- 5a-b: Dispatch tables byte-identical to generators; no stale tpu_v5p; all model+opt route literals present
- 5c: `python -m grokking_optimizers.compile --self-test` — anchored regex: `r"\b[1-9][0-9]* passed, 0 failed\b"`
- 5d: ruff lint clean on `grokking_optimizers/` + `scripts/`
- 5e: `utilization.track_cell()` raises RuntimeError when no device present (crash-hard contract)

**Phase 6 — Report:** Tier map by arch, silicon-only list, design choices, PASS/FAIL verdict.

**Driver (verify_all.py:649-695):** `--quick` skips phases 2+3. Always fills tier_map if phase 4 was skipped. Returns 1 if any FAILs.

---

## 2. CLAIMED BUGS — VERIFICATION

### Bug 1: `device_profiling` import dead

**CONFIRMED.** `grokking_optimizers/device_profiling.py` does NOT exist as a file:
```
ls /workspace/SuperGrok1.5/grokking_optimizers/ | grep device
(empty)
```

The device profiling functions (`collect_nvidia_stalls`, `_stall_to_bias_hints`, `bias_trial_queue`, `run_device_pgo_round`, `write_stall_sidecar`, `read_stall_sidecar`) are defined INSIDE `compile.py` starting at line 32002 under the comment: `"# Device-side PGO (formerly grokking_optimizers/device_profiling.py)"`.

**Impact 1 — compile.py line 17652 (hot path):**
```python
if spec.enable_device_pgo:
    try:
        from grokking_optimizers.device_profiling import (
            run_device_pgo_round,
        )
        ...
    except ImportError:
        pass   # ← silently swallowed
```
This import ALWAYS fails (ImportError). The `except ImportError: pass` means `run_device_pgo_round` is NEVER called even when `enable_device_pgo=True`. Device-side PGO is a complete no-op on the current codebase.

**Impact 2 — compile.py self-test (lines 20299-20426):**
The `_self_test_device_profiling()` function runs 8 subtests, ALL of which do:
```python
from grokking_optimizers import device_profiling  # FAILS: no such module
```
These 8 subtests: `device_profiling_import`, `stall_to_bias_mapping`, `stall_to_bias_empty_input`, `bias_trial_queue_enqueues`, `bias_trial_queue_empty`, `run_device_pgo_round_disabled`, `stall_sidecar_round_trip`, `buildspec_has_device_pgo_field` — ALL will fail with ImportError.

**Cascade:** verify_all.py phase 5c (`compile.py --self-test`) expects `r"\b[1-9][0-9]* passed, 0 failed\b"`. If 8 device_profiling subtests all fail, the self-test will report failures, breaking verify_all phase 5c.

**Fix:** Either create `grokking_optimizers/device_profiling.py` that re-exports from compile.py (awkward), or update the import in compile.py:17652 to call the local function directly, and update the self-test to call the local functions.

### Bug 2: `profile_maximal` 11/0 allowlist

**PARTIALLY CONFIRMED / RE-INTERPRETED.**

The allowlist itself (the `_DEAD_SPILL_KERNEL_TOKENS` / `_is_dead_spill_kernel()` logic) is correctly implemented — it accepts exactly 1 dead-code kernel (the TF32-RS CUTLASS kernel with 8B spill) and rejects anything else. The self-checks at import time (lines 222-229) validate this.

The "11/0" may refer to two coverage gaps:

**Gap A — Tier B optimizer kernel coverage:** Only 3 of 11 optimizer launchers are checked for resource health:
```python
_OPT_KERNELS = ["launch_adamw", "launch_supergrok2", "launch_muon"]
```
The other 8 (grokadamw, grokfast, lion, looksam, neuralgrok, prodigy, supergrok11, supergrok15) are not compiled for Tier B resource audit. So 8/11 launchers have NO spill check.

**Gap B — Self-test failure cascade:** The 8 failing device_profiling subtests plus the verify_all regex pattern (`[1-9][0-9]* passed` requires ≥10 for two digits) may produce a subtle false result. The regex actually matches single-digit counts (`[1-9]` with `[0-9]*` = zero or more, so "1" matches), but the key issue is the "0 failed" part — 8 failures in the self-test would produce a non-zero fail count, breaking verify_all phase 5c entirely.

---

## 3. PROFILING STACK — SUMMARY

### 3.1 What does NOT require ncu (ncu-counter-free path)

All five tiers in `profile_maximal.py` work WITHOUT ncu:
- **Tier A/B:** Use `nvcc -c` + `ptxas -v` + `cuobjdump -sass`. No ncu needed.
- **Tier C:** Uses `bash scripts/amdgcn_check.sh` + `llvm-objdump` + `llvm-readobj`. No ncu.
- **Tier D:** Pure Python CPU execution with torch fp64. No GPU.
- **Tier E:** JAX CPU trace + HLO inspection. No TPU.

`profile.py`'s ncu path (profile_cuda) gracefully degrades: if `ncu not in PATH`, runs smoke-only. No crash.

### 3.2 What IS silicon-gated (needs live hardware)

Per profile_maximal.py's SILICON-ONLY section (always appended to report):
- Wall-clock latency/throughput per cell
- Achieved occupancy, DRAM/L2 bandwidth, SM/MXU duty cycle (ncu/rocprof)
- Autotuner config selection by measured latency
- End-to-end fused-vs-ATen speedup baseline
- gfx942 dynamic-LDS launch footprint + true occupancy (static LDS from descriptor is exact)
- tpu_v6e real MXU emission (CPU lowers to host backend, not MXU)

### 3.3 Analytical FLOP/byte counting

**NOT present in any of the four files.** There is no analytical FLOP count, arithmetic intensity computation, or roofline model in profile.py, profile_maximal.py, utilization.py, or verify_all.py. The `compile.py` has a roofline reference in the self-test area but the four profile files have none. The roofline deliverable is described in RESUME.md as a separate artifact.

### 3.4 In-kernel `clock64` timing

**NOT present in the four Python files.** `clock64` is a CUDA device intrinsic; it would appear in kernel .cu/.cuh files, not in Python profiling wrappers. The Python profiling layer uses wall-clock timing (Python `time.monotonic()`) for subprocess durations.

### 3.5 Occupancy API

`profile_maximal.py` uses `ptxas -v` (compile-time occupancy estimate) and budgets from `mk._arch_budget(arch)`. No live `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API call — that's a silicon path.

---

## 4. CONFIG-DERIVATION MECHANISM IN THIS SLICE

### 4.1 Parallelism / adaptivity in profiling stack

None of the four files contain the core resource-fit planner or parallelism auto-config logic. However, they expose several config-derived behaviors:

**profile.py** — Arch is INFERRED from path structure or passed explicitly. No hardcoded arch branching in the logic (the vendor lookup via `_arch_info()[arch]["vendor"]` is the dispatch gate).

**profile_maximal.py** — Reads `mk._arch_budget(arch)` and `mk.solve(model, opt, arch)` to get the SOLVER's register/smem budgets and tier decisions. The Tier B check cross-validates real ptxas numbers against the solver's estimate. This is where the "estimate vs silicon" gap is exposed.

**verify_all.py** — Phase 4 re-derives all 99 fusion plans via `mk.solve_all()` and checks:
- All 99 cells feasible (≥L1) — no infeasible compositions
- Every cell at its MAX feasible tier (independent re-derivation, not just reading cached plans)
- Every cell within register+smem budget

The solver (`megakernel.solve()`) is the CONFIG DERIVATION mechanism — it reads `ARCH_TABLE` from `compile.py` for register/smem budgets and picks the highest FusionTier (L3>L2>L1) that fits. Key formula (verify_all.py:414):
```python
for higher in (FusionTier.L3_FWD_BWD_OPT, FusionTier.L2_BWD_OPT):
    if higher <= tier: continue
    hr, hs = mk._tier_cost(model, optimizer, higher)
    if hr <= b.max_regs_per_thread and hs <= b.max_smem_per_block:
        not_maximal.append(...)  # solver under-fused!
```

**gfx942 decoder/vit L1 note (verify_all.py:599-605):**
```
note: {N} gfx942 decoder/vit cells are L1 by the smem ESTIMATE (bwd 66560>65536).
Promotion to L3 is silicon-gated (rocprof true-LDS).
```
This is the key "estimate vs silicon" gap — the solver places some gfx942 cells at L1 because the smem estimate (66560B) exceeds the 65536B budget, but the true LDS usage may be smaller.

---

## 5. OPEN ITEMS / BUGS

1. **`device_profiling` module missing** (CRITICAL): All 8 compile.py self-test subtests for device_profiling will FAIL with ImportError. The `enable_device_pgo` build path silently does nothing. Breaks verify_all phase 5c.

2. **Tier B optimizer coverage gap**: Only 3/11 optimizer launchers (`launch_adamw`, `launch_supergrok2`, `launch_muon`) are checked for resource health. The other 8 launchers have no spill audit.

3. **Tier A sm_90a vs sm_90**: `_MAX_NVCC` uses `-gencode arch=compute_90a,code=sm_90a` but `infer_from_path()` maps `"sm_90"` in path parts to `arch = "sm_90"` (line 400), not `sm_90a`. This could produce a mismatch between the inferred arch label and the actual compile target when ARCHES has both `sm_90` and `sm_90a`.

4. **profile_pallas `jax` import** (profile.py:510-512): The `jax.profiler` import is inside the function but `jax` is never imported at module level — `profile_pallas` at line 510 tries `import jax.profiler` (with `jax` in scope), but then at line 517 calls `jax.profiler.start_trace()` without having assigned `jax` to a local name. This would fail with `NameError: name 'jax' is not defined`. This is a real bug: the `import jax.profiler` at line 503 makes `jax` available (Python's import machinery does bind the top-level name), so it works — but it's fragile.

5. **Tier D reference-step loading**: The parity module is loaded via `importlib` from `tests/hw/test_reference_parity.py`, which must contain `ref_adamw_step` and `ref_lion_step`. If this file doesn't exist or lacks these functions, Tier D SKIPs entirely. No validation that the functions match the shipped kernel math post-facto.

6. **`_tpu_trace_all()` reload (verify_all.py:365)**: `importlib.reload(mod)` inside a thread pool may cause issues since jax tracing is noted as "not thread-safe across reloads" — max workers is limited to 4 as a mitigation.

7. **verify_all OOM detection false positive** (verify_all.py:147-153): `_is_oom_kill` checks `out.strip().endswith("Killed")` — any compile error whose last output line contains "Killed" (a valid compile diagnostic word) would be misclassified as OOM and silently retried.

---

## 6. DISCREPANCIES VS CLAIMED STATE

**Claimed (RESUME.md/session context):** "roofline deliverable" as DONE.
**Code truth:** The four profile files contain NO roofline computation. `profile.py` runs ncu/rocprof/jax.profiler but extracts no FLOP/byte counters. `profile_maximal.py` audits binary structure (SASS counts, ptxas register numbers) but does no arithmetic intensity calculation. The roofline must be a separate file not in this slice.

**Claimed:** "dead-code cleanup (removed 8.09M lines)"
**Code truth in this slice:** All four files are clean (no commented-out dead code, no #if 0 blocks, no stale imports). The cleanup claim applies to C++ csrc, not these Python files.

**Claimed:** "11-opt decoder ranking (overfit placeholder)"
**Code truth in this slice:** None of the four files implements or references a benchmark ranking. These are profiling/verification utilities, not benchmark runners.

**Claimed:** profile_maximal "11/0 allowlist" bug
**Code truth:** The allowlist (1 exception for the dead TF32-RS kernel) is correctly implemented with fail-closed logic and import-time self-checks. The "11/0" is more likely the consequence of the device_profiling self-test failure cascade or the Tier B 3/11 optimizer coverage gap.

---

## 7. KEY ARCHITECTURAL FACTS (with line citations)

| Fact | Location |
|------|----------|
| OPTIMIZERS = 11 (adamw..supergrok2) | dispatch.py:589, profile.py:75 |
| MODELS = 3 (transformer_decoder/vit/mamba3) canonical | dispatch.py:586, megakernel.py:37 |
| SHORT_MODELS = 3 (decoder/vit/mamba) user API | dispatch.py:587, profile.py:75 |
| MEGAKERNEL_ARCHS = (sm_90, gfx942, tpu_v6e) | megakernel.py:43, verify_all.py:59 |
| N_CELLS = 11×3×3 = 99 | verify_all.py:60 |
| PIPELINES_PER_ARCH = 33 | utilization.py:60 |
| ncu graceful fallback (smoke-only if absent) | profile.py:452-455 |
| device_profiling.py missing (dead import) | compile.py:17652, no file in package |
| Device PGO code lives in compile.py:32002+ | compile.py:32002 comment |
| Allowlist: exactly 1 dead TF32-RS kernel | profile_maximal.py:198-215 |
| Spill gate: per-kernel, not per-TU | profile_maximal.py:269-281 |
| Tier A checks 4 TUs (3 models + supergrok2) | profile_maximal.py:308-312 |
| Tier B checks only 3/11 opt launchers | profile_maximal.py:371 |
| gfx942 decoder/vit at L1 (smem estimate) | verify_all.py:601-605 |
| verify_all OOM-kill retry with exclusive lock | verify_all.py:144-171 |
| TPU compute_pct=None (xprof-only) | utilization.py:313-369 |
| crash-hard contract (no graceful degradation) | utilization.py:27-32 |
| Profile report path: build/profiled/ | profile.py:634-636 |
| Utilization output: build/utilization/ | utilization.py:643-644 |
