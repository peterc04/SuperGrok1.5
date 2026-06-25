# Codegen + Build Wiring Deep-Read
**Files:** megakernel_codegen.py (1982L), megakernel.py (310L), megakernel_engine.py (521L), setup.py (1029L), wiring_check.py (367L)
**Date analyzed:** 2026-06-25

---

## 1. megakernel.py — Feasibility Solver

### Purpose
`grokking_optimizers/megakernel.py` is the **Stage 6 feasibility solver**: given a `(model, optimizer, arch)` triple, it selects the highest fusion tier that fits within the arch's register and shared-memory budget.

### Tier Hierarchy
```python
class FusionTier(enum.IntEnum):
    L3_FWD_BWD_OPT = 3   # one persistent kernel: forward + backward + optimizer
    L2_BWD_OPT = 2       # backward + optimizer fused; forward separate
    L1_OPT_ONLY = 1      # optimizer step fused across params; fwd/bwd separate
    L0_UNFUSED = 0       # no fusion — solver ERRORS before returning this
```
(line 46-53)

### Component Cost Model
Per-arch budget pulled from `compile.py::ARCH_TABLE` (lazy import, line 131). For sm_90a: `max_regs_per_thread=255`, `max_smem_per_block=228*1024` (compile.py:666-691).

Calibrated component costs (lines 74-115):
- Decoder/ViT fwd: regs=168, smem=49152
- Decoder/ViT bwd: regs=168, smem=66560
- Mamba3 fwd: regs=176, smem=40960
- Mamba3 bwd: regs=176, smem=50176
- Optimizer costs range from lion (regs=32) to supergrok2 (regs=96, smem=49152)

### L3 Tier Cost Formula (Critical for Config Derivation)
The solver uses a **Hopper warp-group split** (lines 171-204):
```python
if tier == FusionTier.L3_FWD_BWD_OPT:
    consumer_wg = max(fwd.regs, bwd.regs)          # heavy model stages
    producer_wg = max(opt.regs, _PRODUCER_WG_REGS)  # tail + staging
    regs = max(consumer_wg, producer_wg)            # §3.4 split: max, not sum
    smem = max(fwd.smem, bwd.smem, opt.smem)
```
The `_PRODUCER_WG_REGS = 32` is the Hopper setmaxnreg producer floor. This means for all current models, L3 regs = max(fwd/bwd, opt.regs), which = 168 or 176. For sm_90a with budget=255, ALL 33 non-TPU cells are feasible at L3.

### Solve Algorithm
`solve(model, optimizer, arch)` (lines 207-264): iterates tiers L3→L2→L1, returns the first that fits. Raises hard error (§1.11) if even L1 doesn't fit — NO silent fallback.

`solve_all()` (lines 267-286): iterates all 99 (3 models × 11 optimizers × 3 archs), returns L0 plan instead of raising for infeasible cells.

### Config Derivation Mechanism
The solver is a **pure function** of (model, optimizer, arch) — not runtime, not GPU-visible. It operates entirely on calibrated budget estimates. The tier it returns drives:
1. The codegen's choice of which C++ template instantiation to emit
2. The dispatch table entries
3. `MegakernelOptimizer.step()` behavior

**Caveat flagged in code**: component costs are marked 🟡 (line 26) — `ptxas -v` on real silicon is the true arbiter; the additive model may differ from actual register allocation.

---

## 2. megakernel_codegen.py — Code Generator

### Overview
A build-time-only Python module (1982 lines) that generates per-(model, optimizer, arch) fused megakernel source files. Has NO runtime call sites.

### Cell Configuration
- **3 models**: `transformer_decoder`, `vit`, `mamba3` → C++ enums `TransformerDecoder`, `ViT`, `Mamba3` (lines 46-50)
- **11 optimizers**: adamw, lion, grokfast, grokadamw, looksam, prodigy, neuralgrok, muon, supergrok11, supergrok15, supergrok2 → each gets its OWN `OptId::*` (lines 52-64)
- **3 archs**: sm_90, gfx942, tpu_v6e
- Total: 99 cells in the manifest; 33 per arch

### WIRED_CELLS Declaration (lines 72-79)
```python
WIRED_CELLS: Tuple[...] = tuple(
    (m, o, a) for m in (3 models) for o in (11 opts) for a in (3 archs)
)
```
This claims ALL 99 cells are wired — not just the 3 demo cells. This is a documentation claim in the Python, not a runtime guarantee.

### Per-Arch Emission

**CUDA (sm_90) — `_emit_cuda()` (lines 165-241):**
Emits a `.cu` file that:
1. `#include "csrc/fused/sm_90/fused_megakernel.cuh"` — the one composition substrate
2. Defines `mega_<model>_<opt>()` host launcher that binds m/v/extra state buffers and calls `launch_fused_megakernel<ModelId, OptId, FuseTier::L1 or L3>()`
3. Supports BOTH tiers at runtime via `opt_only` flag

**The `extra` state field per optimizer** (lines 99-111):
- `None` for adamw/lion/neuralgrok (no third buffer)
- `"ema"` for grokfast/grokadamw
- `"sam_dir"` for looksam
- `"s_track"` for prodigy
- `"orth"` for muon
- `"mu"` for supergrok11/supergrok15
- `"smart_grad"` for supergrok2

neuralgrok gets special psi-net weight binding from the packed `extra` buffer (lines 120-141).

**AMD (gfx942) — `_emit_hip()` (lines 244-326):**
AMD twin of `_emit_cuda`. Includes `fused_megakernel.hip.hpp`. Force-instantiates BOTH L1 and L3 template globals. Host launch is 🟡 (MI300X-gated, no hipcc in production).

**TPU (tpu_v6e) — `_emit_pallas()` (lines 329-366):**
Generates a small Python module binding to `csrc/backends/pallas/_pallas_fused.py::fused_step` via `functools.partial`. Described as "REAL fused composition" (not a stub) — jax.jit fuses fwd→bwd→opt.

### `--write-all` Path (lines 444-458)
`write_all(root="csrc/fused")` iterates all 99 plans from `mk.solve_all()`, calls `emit_cell()`, and writes to `csrc/fused/<arch>/mega_<model>_<opt>.<ext>`. This creates:
- `csrc/fused/sm_90/mega_<model>_<opt>.cu` (33 files)
- `csrc/fused/gfx942/mega_<model>_<opt>.hip` (33 files)
- `csrc/fused/tpu_v6e/mega_<model>_<opt>.py` (33 files)

### Dispatch Tables (lines 467-556)
`dispatch_table_sm90()`: emits a C++ `.inc` declaring extern C++ function signatures for all 33 sm_90 launchers and a `dispatch_sm90_cell()` if-chain.
`dispatch_table_gfx942()`: AMD twin.
`dispatch_table()`: emits the `wired_fused_cell()` C++ body used by `fused_wired_cells.inc`.

### Weight Layout Emitters — The Single Source of Truth

The codegen computes all parameter tensor shapes from scratch:

**Decoder (lines 559-917):**
- Production: d=128, layers=2, vocab=99, seq=4, heads=4 → 30 tensors, 422755 elems
- Bench: d=2048, SG_DEC_BENCH_LAYOUT flag
- Flagship: d=1600, layers=48, heads=25 → 582 tensors, 1,475,884,899 elems
- `_decoder_param_sizes()` formula: 2 + 12*L + 4 tensors (tok/pos, 12 per layer, norm+out)

**ViT (lines 976-1391):**
- Production: d=128, layers=2, vocab=97, patch=49, npatch=16, heads=4 → 32 tensors, 418017 elems
- Bench: d=2048, SG_VIT_BENCH_LAYOUT flag
- Flagship: d=1664, layers=48, heads=16 (explicit, NOT d//64=26) → 584 tensors, 1,596,200,417 elems
- CRITICAL: cls_token (leaf nn.Parameter) comes BEFORE patch_proj submodule params

**Mamba3 (lines 1393-1833):**
- Production: d=128, layers=2, vocab=99, phead=97, seq=8 → 45 tensors, 593713 elems
- Bench: d=1024, SG_MB_BENCH_LAYOUT flag
- Flagship: d=2048, layers=24 → 485 tensors, 1,265,411,169 elems (n_heads=64, NOT config's 32)
- Per-block structure: mixer_norm.w, then mixer OWN params (A_log/D/4-biases) BEFORE mixer submodules

**Key correctness note on Mamba smem**: `kMambaSmemFloats` is described as "SAFE LARGE PLACEHOLDER" in the code comments (line 1571) — the field count is NOT the production value but a placeholder that must be pinned by a `static_assert(sizeof(MambaSampleSmem) == kMambaSmemBytes)` in model_stage_mamba3.cuh that will "fail the build with the exact required value when it is pinned."

### Size-Adaptive Knob Selector (lines 596-730)
This is the "self-designing megakernel" mechanism:

```python
def _dec_is_large(d, layers, T, n_sms, tile_m=128) -> bool:
    if d >= 1024:  return True          # WIDTH trigger
    if _dec_token_tiles(T, tile_m) < n_sms:  return True  # GRID UNDER-FILL
    return False
```

`decoder_knobs_for_size(d, layers, T, n_sms)` returns:
- LARGE tier (d≥1024 or token tiles < n_sms): `par::SizeLarge` (CTA-tiling on)
- SMALL tier (d=128 production race): `par::SizeSmall` (persistent 1-CTA/SM shape)

Named decoder tiers (lines 708-713):
```
"production": (d=128,  layers=2,  T=512*4=2048)   → SizeSmall
"bench":      (d=2048, layers=2,  T=4096*4=16384) → SizeLarge (d≥1024)
"flagship":   (d=1600, layers=48, T=512*4=2048)   → SizeLarge (d≥1024)
```

The knob VALUES themselves are all defaulted to `_DEC_KNOB_DEFAULTS` for all tiers currently (no branch in `decoder_knobs_for_size` changes the knob values for LARGE). Only `size_config` changes. This is noted as an intentional incremental approach: "the §1 byte-identical-when-OFF invariant."

### CLI Entry Points (lines 1873-1982)
- `--emit <model> <opt> <arch>`: emit one cell to stdout
- `--emit-all`: print 99-cell manifest table
- `--write-all`: materialize all 99 to disk
- `--dispatch-table`, `--dispatch-table-sm90`, `--dispatch-table-gfx942`: emit dispatch tables
- `--decoder-layout`, `--decoder-layout-flagship`: emit decoder layout headers
- `--decoder-knobs`: print size-adaptive knob selection table
- `--vit-layout`, `--vit-layout-flagship`: emit ViT layout headers
- `--mamba-layout`, `--mamba-layout-flagship`: emit Mamba3 layout headers

---

## 3. megakernel_engine.py — Framework Adapter

### Purpose
Reconciles the framework's separate forward/backward/step interface with the fused L3 kernel's single-launch contract.

### Key Classes

**`FusedBackwardHook` (lines 205-283):**
- `intercept_backward(loss)`: returns True and no-ops the framework backward when tier≥L2 (fused kernel owns backward)
- `as_tensor_hook()`: returns a Tensor.register_hook callable that zeroes grad when fused path owns it
- `fused_update_applied` flag: set when fused launch has applied update

**`MegakernelOptimizer` (lines 289-416):**
- Wraps an inner optimizer (one of the 11 race optimizers)
- `step()`: if fused_owns_step(plan) AND hook.fused_update_applied → only bookkeeping, no math re-run
- Otherwise delegates to inner.step()
- Satisfies DeepSpeed `client_optimizer` duck-type

**`dispatch_fused_megakernel()` (lines 101-159):**
- Unified cross-arch entry: routes tpu_v6e to Pallas, sm_90/gfx942 to C++ `fused_step`
- For GPU: if solver does NOT award L3, HARD-FAILs loudly (no silent downgrade to L1)
- Calls `ops.fused_step(model, optimizer, params, inputs, grads, state, lr, opt_only=False, gemm_impl="wgmma")`
- NOTE: "pure L3-TC (task #10): the L1 optimizer-only tail and the scalar fp32 engine were removed" (line 142)

### Framework Control Flow (L3 case)
1. `forward()` → fused L3 launch runs fwd+bwd+opt, sets `_fused_owned=True`
2. `engine.backward(loss)` → hook intercepts → NO-OP
3. `engine.step()` → MegakernelOptimizer sees `fused_update_applied=True` → bookkeeping only

### Config Derivation Mechanism
`resolve_fusion_plan(model_name, optimizer_name, arch)` calls `solve()` → returns FusionPlan. Tier gates ALL control flow. No hardcoded model/optimizer names in the flow logic.

---

## 4. setup.py — Build Wiring

### Extension: `grokking_optimizers._ops`
Source lists (lines 584-591 CUDA, 504-530 HIP):
- **CUDA**: `csrc/bindings/bindings.cpp`, `csrc/bindings/dispatch.cpp`, `csrc/backends/cuda/sm_90/*.cu`, `csrc/backends/cuda/sm_90/models/*.cu`, `csrc/fused/sm_90/*.cu` (glob)
- **HIP**: same bindings + hip backends + `csrc/fused/gfx942/*.hip`

Source filtering `_collect()` (lines 453-490):
- Drops `*_overlay.*` files
- Drops `*_selftest.cu` files (own pybind module)
- Drops `.cu` files containing `PYBIND11_MODULE(TORCH_EXTENSION_NAME` (own module → would collide on `PyInit__ops`)

This last rule is why `mega_decoder_real_adamw_tc.cu` is excluded: it has its own pybind module. The `*_launcher.cu` variants (no own module) ARE included.

### Multi-arch Build (lines 270-376)
`_NVIDIA_CCS = ["70","75","80","86","89","90","100","103","120"]`
`_toolchain_accepts()` probe-filters at build time. Hopper+ uses `sm_90a` architecture-specific target. PTX of newest accepted CC for driver-JIT forward-compat.

### CUTLASS Auto-Enable Policy (lines 157-179)
Default ON when CUDA≥12 toolkit AND `third_party/cutlass/include/cutlass/cutlass.h` present. Set `WITH_CUTLASS=0` to force cuBLAS.

### Per-TU Tuned Flag Injection — `TunedBuildExtension` (lines 829-913)
Monkeypatches `torch.utils.cpp_extension._write_ninja_file` to inject per-TU `cuda_post_cflags` overrides from `_kernel_tuned.json`. This is how different SG_TUNED_* macro values and `--maxrregcount` are applied per-optimizer TU.

**Key design**: torch's ninja build uses ONE shared `cuda_post_cflags` for all TUs; this hook appends per-statement overrides in the generated ninja file. Degrades gracefully (no JSON → stock build).

### NVCC Drift Guard — `_verify_macro_names()` (lines 944-969)
Scans `csrc/` for `#ifndef SG_TUNED_*` macros and warns if `_tuned_inject.MACROS` contains any name not guarded by a header. Prevents silent dead `-D` flags.

---

## 5. wiring_check.py — The L3-TC Baseline Gate

### What It Does
For each of the 33 (model × optimizer) cells, runs `steps=4` real race train steps via `grokking_race_v2.OPTIMIZER_REGISTRY[opt]`. Wraps `_try_fused_train_step` and `_try_fused_step` to capture `LAST_L3_ENGINE["engine"]`.

**PASS criterion** (line 241):
```python
all_wgmma = (n_l3 > 0 and all(e == "wgmma" for e in seen_l3_engines)
             and not abi_stale and err is None)
```

### Cell Naming
- OPTS list uses race registry keys: `"supergrok"` (= SuperGrok11), `"supergrok15"`, `"supergrok2"` (lines 52-53)
- MODELS: `"decoder"`, `"vit"`, `"mamba"` (not the canonical long names)

### 33/33 Result (wiring_check.json, 2026-06-24)
The stored result at `/workspace/SuperGrok1.5/results/h100_grokking_race/wiring_check.json` shows:
- `converted_l3_tc: 33`, `blocked: 0`, `fraction_l3_tc: 1.0`
- `l3_steps_fired_total: 132` (33 × 4 steps)
- `l1_steps_fired_total: 0`, `abi_stale_any: False`, no errors

### Discrepancy with `_BLOCK_REASONS`
The code contains detailed `_BLOCK_REASONS` for supergrok, supergrok15, looksam, neuralgrok explaining why they're model-coupled and blocked. But the JSON shows all 33 as wgmma. This is NOT a contradiction: `_BLOCK_REASONS` is only consulted when `converted=False` — if the dispatch.cpp actually routes these to wgmma in the production build, they pass. The comments describe the history of blocking reasons, not permanent gates.

**Key: dispatch.cpp DOES include looksam/supergrok11/supergrok15/neuralgrok in wgmma routing** (line 779: `const bool is_sg = (optimizer == "supergrok11" || optimizer == "supergrok15")`; looksam handled via `wgmma_tail_opt_id` gate).

---

## 6. ARCHIVE GAPS — What Actually Exists vs. What Codegen Expects

### GAP #1: `fused_megakernel.cuh` Missing
The codegen `_emit_cuda()` generates cells that `#include "csrc/fused/sm_90/fused_megakernel.cuh"` (line 197 of megakernel_codegen.py). **This file does NOT exist** in `csrc/fused/sm_90/`:
```
$ ls /workspace/SuperGrok1.5/csrc/fused/sm_90/fused_megakernel.cuh
NOT FOUND
```

The actual wired TC launchers use **model-specific** megakernel headers:
- `mega_decoder_real_adamw_tc_launcher.cu` → `fused_decoder_megakernel.cuh`
- `mega_mamba_real_adamw_tc_launcher.cu` → `fused_mamba_megakernel.cuh`
- `mega_vit_real_adamw_tc_launcher.cu` → `fused_vit_megakernel.cuh`

This means: **if `--write-all` were run, the 33 generated sm_90 `.cu` files would fail to compile** because the unified `fused_megakernel.cuh` they `#include` doesn't exist.

### GAP #2: Generated sm_90 `.cu` Cells Missing
`WIRED_CELLS` declares 33 sm_90 cells wired, and `fused_dispatch_table.inc` declares extern signatures for all 33 `mega_<model>_<opt>` symbols. But **zero** of these 33 generated files exist on disk:
```
ls csrc/fused/sm_90/mega_transformer_decoder_adamw.cu → NOT FOUND (33/33 missing)
```

The 6 `.cu` files in `csrc/fused/sm_90/` are all `*_real_adamw_tc*` variants — the Fork-B TC drivers, NOT the codegen-generated cells.

### GAP #3: `launch_<opt>.cu` Shims Missing
No `launch_*.cu` files exist in `csrc/fused/sm_90/`:
```
ls csrc/fused/sm_90/launch_*.cu → NONE
```
The compile.py `ArchEntry.launcher_glob=("launch_*.cu",)` for sm_90a implies these were planned.

### What IS Present
- **gfx942**: 33/33 generated `.hip` cells ARE present (all `mega_<model>_<opt>.hip`)
- **tpu_v6e**: 33/33 generated `.py` cells ARE present
- **sm_90**: Only 6 files (3 model-specific TC drivers + 3 launcher variants), NOT the codegen-produced cells
- `fused_dispatch_table.inc`: Generated (declares all 33 sm_90 symbols) but its extern declarations reference symbols with NO corresponding compiled TUs

### Runtime Impact
The existing `fused_dispatch_table.inc` is included by dispatch.cpp, which declares the 33 `mega_<model>_<opt>` launchers as `extern`. But since no TUs implement those symbols, the build would link-fail unless dispatch.cpp only actually routes to the 3 real TC launchers (the Fork-B `*_launcher.cu` files).

Looking at dispatch.cpp: it uses its own internal routing to `launch_fused_{decoder,vit,mamba}_megakernel_tc` — the model-specific TC launchers — NOT to the `dispatch_sm90_cell` table from `fused_dispatch_table.inc`. The `fused_dispatch_table.inc` may be included for reference/declaration but the actual dispatch paths in dispatch.cpp bypass it for the wgmma routes.

---

## 7. Key Design Decisions — Self-Adaptation Mechanism

### How the Solver Derives Config
1. **Input**: `(model, optimizer, arch)` triple
2. **Budget source**: `compile.py::ARCH_TABLE[arch]` (max_regs_per_thread, max_smem_per_block)
3. **Cost model**: component-additive estimates per tier (L3 uses Hopper warp-group max, not sum)
4. **Output**: `FusionPlan.tier` → drives codegen template, dispatch routing, engine adapter

### How the Codegen Derives Layout
1. **Input**: `(d, layers, vocab, seq, heads)` — derived from model config
2. **Formula**: `_decoder_param_sizes(d, layers, vocab, seq)` — pure arithmetic, no torch import
3. **Output**: `kDecOffsets[]`, `kDecSizes[]` — device constant arrays, verified by compile-time `static_assert`

### How CTA-Tiling is Selected
`decoder_knobs_for_size(d, layers, T, n_sms)` — pure function with two structural triggers:
- d ≥ 1024 (width-wide enough to split N-range across CTAs)
- token tiles < n_sms (grid under-fill with 1-CTA/SM wave)

Returns `par::SizeLarge` or `par::SizeSmall` as the launcher's `SizeConfig` template arg.

---

## 8. Summary of State

| Component | State |
|-----------|-------|
| megakernel.py solver | Implemented, functional; calibrated estimates are 🟡 |
| megakernel_codegen.py emission logic | Implemented for all 3 archs |
| megakernel_codegen.py layout emitters | Implemented, parameterized for all 3 flagship tiers |
| size-adaptive knob selector | Implemented; knob values all default for now (only size_config changes) |
| fused_megakernel.cuh (the "one substrate") | **MISSING** — codegen would fail to compile if run |
| Generated sm_90 cells (33 .cu files) | **ALL MISSING** on disk |
| Generated gfx942 cells (33 .hip files) | ALL PRESENT (33/33) |
| Generated tpu_v6e cells (33 .py files) | ALL PRESENT (33/33) |
| launch_<opt>.cu shims | MISSING |
| Actual wired sm_90 TC path | 3 model-specific _launcher.cu files (wgmma, wired in _ops) |
| wiring_check.json 33/33 result | Present (2026-06-24); internally consistent; 33 cells × 4 steps |
| Mamba smem budget | PLACEHOLDER in codegen; real value pinned by static_assert in model_stage_mamba3.cuh |
| megakernel_engine.py | Implemented; hard-fails on non-L3 (scalar path removed) |
| setup.py TunedBuildExtension | Implemented; degrades gracefully without _kernel_tuned.json |
