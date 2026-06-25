# Python Runtime & Data Layer Analysis
## Slice: `grokking_optimizers/__init__.py`, `dataset_sources.py`, `_tuned_inject.py`, `tune_hook.py`, `lowprec.py`, `optimizers/` (all 12 files)

---

## 1. Package Init — `grokking_optimizers/__init__.py`

**Purpose:** Public entry point; imports all 11 optimizer classes + dispatch helpers.

**Key facts:**
- `__version__ = "3.0.0"` (line 32)
- Imports 12 classes: `AdamW` + `SuperGrok2/15/11` + `GrokAdamW/Grokfast/Lion/LookSAM/Muon/NeuralGrok/Prodigy` + `MoEAwareSuperGrok2` + `CompiledSuperGrok2`
- Module-level `_HAS_OPS = bool(get_ops())`, `_HAS_CUDA = has_kernels()`, `_HAS_CPU_OPS = hasattr(_ops, "supergrok2_cpu_step")` (lines 47-51)
- Lazy import pattern (PEP 562 `__getattr__`) for `ARCH_TABLE/ARCH_INFO/ArchEntry/get_arch_entry` from `compile.py` — defers the 23k-LOC module parse until first access (lines 99-114)
- `has_kernels()` is the public GPU-build capability probe; `_HAS_CUDA` retained for back-compat
- **EXPLICIT CONTRACT:** "There is NO pure-PyTorch/CPU fallback for the 11 optimizers. `.step()` on a CPU-only or unbuilt install raises a descriptive error." (lines 16-19)

---

## 2. Dataset Sources — `grokking_optimizers/dataset_sources.py`

**Purpose:** Layer-A pluggable dataset seam for grokking_race_v2. DEFAULT-OFF; only active when `c["data_source"] != "modular"`.

### Contract
`make_source_for_task(c, seed)` returns the same 6-tuple as the legacy `make_data_for_task`:
`(train_view, train_y, val_probe, val_y, test_probe, test_y)` as plain CPU tensors.

- Slots 0,1: fixed once-materialized train view (<=16384 rows, n_view = min(max(bs, budget), 16384))
- Slots 2,3: fixed val probe (n_probe = max(bs, min(c.get("eval_probe_rows",4096), 50000)))
- Slots 4,5: fixed test probe (independent draw)

### DatasetSource class (line 40)
- `__init__(gen, bs, n_view, vprobe, tprobe, seed)`: materializes train view once, stores fixed probes
- `stream(step)`: per-step fresh minibatch for callers wanting true streaming
- `as_tuple()`: returns the 6-tuple

### Seed design
- `_gen(seed, step, salt)`: CPU generator seeded by `(seed*1000003 + step*100003 + salt) & 0x7FFFFFFFFFFF`
- RNG salts 1, 7, 9, 11 for train/val-lm/val-forecast/val-vit probes

### Three reference stubs (deterministic synthetic)

| Stub | data_source key | model_type | Input shape | Target shape |
|------|----------------|------------|-------------|--------------|
| `_lm_stub` | `"fineweb_edu"` | `"decoder"` | `[B, 4]` int tokens (seq=4) | `[B]` int (vocab=num_tokens, default 99) |
| `_forecast_stub` | `"gifteval"` | `"mamba"` | `[B, seq_len]` int (seq=8 default) | `[B]` int (classes=p, default 97) |
| `_imagenet_stub` | `"imagenet1k"` | `"vit"` | `[B, 16, 49]` float (npatch=16, pdim=49) | `[B]` int (classes=p, default 97) |

- `"synthetic"` alias routes by model_type
- `_STUBS` dict maps keys to builders; `None` value = route by model_type
- `_route(c)` function at line 133 implements the dispatch

### Pluggability seam
Real Phase-4 loaders (FineWeb-Edu / ImageNet-1k / GiftEvalPretrain) drop in behind the IDENTICAL `(c, seed) -> 6-tuple` signature. Layer-B (real vocab/patch/class + kernel regen) is explicitly OUT OF SCOPE.

**STATUS:** Fully implemented deterministic stubs. Not connected to any real download logic. The c-dict keys it reads: `"data_source"`, `"model_type"`, `"num_tokens"`, `"seq_len"`, `"num_patches"`, `"patch_dim"`, `"p"`, `"eval_probe_rows"`, `"train_batch_size"`, `"early_stop_max_steps"`, `"max_steps"`, `"train_view_rows"`.

---

## 3. Tuned-Flag Injection — `grokking_optimizers/_tuned_inject.py` (849 lines)

**Purpose:** Single source of truth for injecting autotuner-winning launch parameters as nvcc macros into the shipped .so build.

### Design
- **Pure stdlib** — no torch, no CUDA, no grokking_optimizers imports. Loaded by `setup.py` and test code.
- Producer: `compile.py::build_jit` calls `export_winner()` → writes JSON
- Consumer: `setup.py::BuildExtension` reads JSON → injects `-DSG_TUNED_*` macros per TU

### MACROS floor (line 151) — live sm_90 macros only

| dim name | macro | default | kernel location |
|----------|-------|---------|-----------------|
| `mega_block` | `SG_TUNED_MEGA_BLOCK` | 256 | `csrc/fused/megakernel_common.cuh:50` |
| `tile_m` | `SG_TUNED_TILE_M` | 128 | `csrc/backends/cuda/sm_90/wgmma.cuh:136` |
| `tile_n` | `SG_TUNED_TILE_N` | 128 | `csrc/backends/cuda/sm_90/wgmma.cuh:133` |
| `dec_dw_splitk` | `SG_TUNED_DEC_DW_SPLITK` | 1 | `csrc/fused/sm_90/model_stage_decoder_tc.cuh:101` |
| `vit_dw_splitk` | `SG_TUNED_VIT_DW_SPLITK` | 4 | `csrc/fused/sm_90/model_stage_vit_tc.cuh:105` |
| `prod_regs` | `SG_TUNED_PROD_REGS` | 40 | `csrc/backends/cuda/sm_90/tile_pipeline.cuh:92` |
| `cons_regs` | `SG_TUNED_CONS_REGS` | 232 | `csrc/backends/cuda/sm_90/tile_pipeline.cuh:95` |
| `maxrregcount` | (ptxas flag) | 0 (unset) | `--maxrregcount=N` |

**DEAD macros removed:** `SG_TUNED_BLOCK_SIZE`, `VEC_WIDTH`, `UNROLL`, `ASYNC_DEPTH` (were for deleted eager per-op headers).  
**REMOVED:** `mb_dw_splitk`/`SG_TUNED_MB_DW_SPLITK` (Mamba-3 TC rewrite dropped output-stationary dW split-K, line 129-131).

### Schema #12: Per-model nested JSON
JSON shape: `{arch: {model: {optimizer: combo}}}` (canonical short model keys: `decoder/vit/mamba`).
- Old flat `{arch: {optimizer: combo}}` still read (backward compat) and migrated in-place on next `export_winner()` call.
- `_arch_block_is_nested()` distinguishes old/new shapes.
- `_lookup_combo()` handles both shapes.

### TU → optimizer mapping
- `optimizer_for_source(path)`: `launch_<opt>.cu` → opt; `mega_<model>_<opt>.cu` → opt (greedy, longest known suffix)
- `model_for_source(path)`: `mega_<model>_<opt>.cu` → short model key; `launch_<opt>.cu` → None (shared launcher)
- Standalone real-TC cells (`mega_<model>_real_adamw_tc.cu`) → None (not injection targets)

### Key functions
- `source_extra_nvcc_flags(optimizer, tuned, arch_key, model)`: produces per-TU flag list
- `compute_source_flags(sources, tuned, arch_key)`: maps all sources to their flag lists
- `export_winner(optimizer, model, arch, combo, ...)`: READ-MERGE-WRITE atomic JSON update
- `inject_overrides_into_ninja(ninja_path, ...)`: patches build.ninja with per-edge `cuda_post_cflags` overrides
- `parse_ninja_build_target(stmt)`: handles ninja `$:` / `$ ` escaping

**STATUS:** Fully implemented, self-contained. The `_kernel_tuned.example.json` file exists in the package dir as documentation of the schema.

---

## 4. Tune Hook — `grokking_optimizers/tune_hook.py` (327 lines)

**Purpose:** Provider implementation for compile.py's portable autotuner — drives the production L3-TC step against a variant .so and measures correctness + performance.

### Interface
```python
run(*, so_path, model, optimizer, arch, regime, seed) -> {"output": np.ndarray, "elapsed_ms": float}
```

### Regime → seed mapping (line 101)
| regime | seed offset |
|--------|-------------|
| `normal` | +0 |
| `large` | +101 |
| `small` | +202 |
| `adversarial` | +303 |
Unknown regime: stable hash `1000 + (sum(ord(ch) for ch in regime) % 9000)`.

### Timing (line 110)
- `_WARMUP = 8`, `_ITERS = 20`, `_REPS = 5`
- Median ms/step over 5 reps of 20 iterations each

### Key mechanisms
1. **`_alias_variant_ops(so_path)`** (line 129): evicts all `grokking_optimizers.*` from `sys.modules`, loads variant .so with `importlib.util.spec_from_file_location`, aliases as `sys.modules['grokking_optimizers._ops']`. Module name = pre-first-dot stem of filename.
2. **Workload construction** (line 237): 
   - `optimizer == "supergrok2"`: uses `tests.hw._sg2_l3tc_gate._build()` + `_sg2_factory()`
   - All others: uses `tests.hw.test_l3tc_tail_gate._CELLS[f"{optimizer}/{model}"]`
3. **Validation gate** (line 280): asserts `has_l3_real(canon, optimizer)` AND `gemm_impl_for_cell(canon, optimizer, "bf16") == "wgmma"`. Fails LOUD if not met.
4. **Output**: concatenated updated params after last production step, as `float32` numpy array.
5. **Determinism**: sets `GATE_SEED` env var, `torch.manual_seed`, `torch.cuda.manual_seed_all`, cudnn deterministic.

**STATUS:** Fully implemented. Imports are lazy (no CUDA required at module import time). SG2 has its own gate path; all others go through `_CELLS` factory dict.

---

## 5. Low Precision — `grokking_optimizers/lowprec.py` (233 lines)

**Purpose:** FP8 and INT8 Linear layer swaps for H100-native low-precision GEMM.

### Modes
`LOWPREC_MODES = ("fp8", "fp8e5m2", "int8")` (line 34)

### FP8 implementation (lines 56-127)
- `_FP8MatmulFn`: custom autograd Function
- Forward: dynamic per-tensor scaling → cast to fp8 → `torch._scaled_mm`
- Backward: E5M2 gradients for dX and dW (standard recipe)
- **SM_90 constraint:** `e4m3×e4m3` and `e5m2×e4m3` supported; `e5m2×e5m2` NOT supported → `fp8e5m2` uses E5M2 activations × E4M3 weights (line 69)
- dW GEMM pads token dim T to multiple of 16 (zero padding exact: zero rows → zero output)
- `_fp8_dims_ok`: K%16==0 AND N%16==0 required

### INT8 implementation (lines 130-166)
- `_Int8MatmulFn`: custom autograd Function
- Forward: symmetric per-tensor quant → `torch._int_mm` (IMMA cores)
- **Layout constraint:** mat2 must be COLUMN-major; `.t()` without `.contiguous()` (line 150, comment)
- Token dim padded to %16 for GEMM
- Backward: bf16 matmuls (int8 gradient training not defined)
- `_int8_dims_ok`: K%8==0 AND N%8==0

### `swap_linears_lowprec(model, mode)` (line 188)
- Walks all `nn.Linear` modules recursively
- Swaps compatible ones, keeps others as plain Linear (bf16 via autocast)
- Returns `{"mode", "swapped": [...], "fallback_bf16": [...]}` 
- LOUD warnings: if no swaps, or partial fallbacks
- FP8Linear holds fp32 master weights (shared param object), casts on forward
- Int8Linear same: fp32 master, casts on forward

**STATUS:** Fully implemented. The 99-logit head (vocab=99) and vit's 49-dim patch embed are mentioned as fallback examples (non-%16 dims). Not connected to the megakernel path; used independently.

---

## 6. Optimizers — `grokking_optimizers/optimizers/`

### Critical shared pattern across ALL 11 optimizers

**Every `step()` raises:**
```python
raise NotImplementedError(
    "L3-TC megakernel only; eager .step() removed — the megakernel owns "
    "the optimizer update via fused_train_step")
```

**Every `use_grad_hooks=True` raises at construction time** (fail-fast before mid-backward).

**Every `_single_param_step()` raises NotImplementedError.**

The plumbing below the `raise` in `_register_grad_hooks` (PyTorch >= 2.1 check, hook registration) is dead code retained "for when an eager path returns."

### 6.1 optimizers/__init__.py
Just re-exports all 12 classes (lines 16-41). No logic.

### 6.2 `adamw.py`
**Math (from docstring, lines 6-13):**
```
m_t = β1·m + (1-β1)·g
v_t = β2·v + (1-β2)·g²
m̂ = m_t / (1-β1^t); v̂ = v_t / (1-β2^t)
p_{t+1} = p_t - lr·(m̂/(√v̂+eps) + wd·p_t)
```
- Defaults: lr=1e-3, betas=(0.9,0.999), eps=1e-8, wd=1e-2
- State: `exp_avg` (fp32), `exp_avg_sq` (fp32), `step` counter
- `_validate_grad()`: rejects sparse/cpu/dtype-mismatch grads, densifies non-contiguous

### 6.3 `lion.py`
- Sign-based momentum optimizer
- Defaults: lr=3e-4, betas=(0.9,0.99), wd=3.0
- State: `exp_avg` (fp32 momentum buffer)

### 6.4 `muon.py`
- Auto-splits params: 2D (ndim==2) → Muon group; other ndim → AdamW group
- Defaults: lr=0.02, momentum=0.95, ns_steps=5, adamw_lr=1e-3, adamw_betas=(0.9,0.98)
- Two internal param groups: `group_type="muon"` and `group_type="adamw"`
- State: `momentum_buffer` (fp32) for Muon; `exp_avg/exp_avg_sq/step` (fp32) for AdamW
- `_step_muon()` and `_step_adamw()` both raise NotImplementedError

### 6.5 `grokfast.py`
- EMA gradient amplification: `ema = α·ema + (1-α)·g; g_amp = g + λ·ema`
- Defaults: lr=1e-3, betas=(0.9,0.98), eps=1e-8, wd=1.0, grokfast_alpha=0.98, grokfast_lamb=2.0
- State: `ema` (fp32, SEEDED WITH FIRST GRADIENT not zeros — critical design choice), `exp_avg`, `exp_avg_sq`, `step`
- First-gradient seed for EMA: prevents under-amplification in early grokking phase

### 6.6 `grokadamw.py`
- Published GrokAdamW algorithm (cognitivecomputations/grokadamw / QuixiAI/grokadamw)
- **Layer-wise β1 decay:** `β1_i = β1 * (1-γ)^i` (global enumeration across all groups)
- **Grokking-signal α:** `α_t = alpha_init * exp(-κ * signal)`, signal = `max(0, val-train)/max(val,train)` (max-normalised)
- Defaults: lr=1e-3, betas=(0.9,0.98), alpha=0.98, lamb=2.0, gamma=0.1, kappa=0.1, grad_clip=1.0
- `decay` arg: back-compat only, accepted but IGNORED (no published role)
- `_grokking_signal()` staticmethod returns 0 if either loss is None
- `_alpha_for_group()`: returns alpha_init if signal never set
- Host-side: `_grok_signal` + `_signal_active` on optimizer instance
- State: `exp_avg`, `exp_avg_sq`, `ema` (seeded with first gradient), `step`
- Step() has extra args: `train_loss=None, val_loss=None` — ALSO raises NotImplementedError; the production path sets opt._grok_signal host-side before fused launch

### 6.7 `looksam.py`
- AdamW + periodic SAM direction adjustment every k steps
- Defaults: lr=1e-3, rho=0.05, k=5, alpha=0.7
- `should_sam_step()`: returns `self._global_step % k == 0` (host-side helper, still works)
- `sam_step()` and `step()` both raise NotImplementedError
- `state_dict()`/`load_state_dict()`: save `_global_step` under `"_looksam"` key
- State: `exp_avg`, `exp_avg_sq`, `sam_direction` (fp32), `step`

### 6.8 `neuralgrok.py`
- MLP gradient amplifier: `g_amp = (ψ(|g|)*alpha + beta) * g`
- `KERNEL_PSI_HIDDEN = 16` (must match csrc compile-time `kPsiHidden`)
- `_Amplifier`: Linear(1,H) → ReLU → Linear(H,1); H=128 default (but race pins H=16)
- `psi_pack()`: returns `[3*H+1]` fp32 vector `[W1(H)|b1(H)|W2(H)|b2(1)]` for kernel's extra slice
- **A3 defect fix:** `train_amplifier_step()` (line 348): differentiable virtual step through amplifier, val-only lookahead objective. Previously amplifier was autograd-unreachable (frozen at random init).
- `maybe_train_amplifier()`: cadence wrapper, `every=1` by default
- Internally-owned Adam over amplifier params (lazily built, amplifier_lr=1e-3)
- `get_amplifier_optimizer()`: external Adam factory
- `inner_steps` arg: deprecated/dead, emits DeprecationWarning if ≠ 1
- State: `exp_avg`, `exp_avg_sq`, `step`

### 6.9 `prodigy.py`
- Distance-aware self-tuning Adam
- Defaults: lr=1.0, betas=(0.9,0.999), eps=1e-8, wd=1.0, d0=1e-6, d_coef=1.0
- **Persistent scalars** (NOT in per-param state): `_d_lr=d0`, `_r_ema=0.0`, `_s_ema=0.0`
- r_ema/s_ema are running EMAs (NOT instantaneous): decay by β3=sqrt(β2) across steps
- Design note: instantaneous max() form caused d blowup on post-memorization gradient noise → EMA form plateaus when params stop drifting
- `state_dict()`/`load_state_dict()`: save under `"_prodigy"` key; old checkpoints (without blob) load cleanly
- State per-param: `exp_avg`, `exp_avg_sq`, `s`, `param_init` (stored at step 0), `step`

### 6.10 `supergrok11.py`
**Gate signal:** `sigmoid(gate_temperature * cos_sim(grad, momentum))` — per-param cosine (vs. SG15's accuracy scalar).

**SharpnessMetaNet:** `output = grad + rescale * MLP(grad, sharpness)`, MLP: `Linear(2,H) → GELU → Linear(H,1)`, H=32 default.

**Hyperparams:** lr=1e-3, betas=(0.9,0.999), alpha_init=0.98, lamb=5.0, gamma=0.1, warmup_steps=100, warmup_ramp=100, gate_temperature=5.0, meta_update_freq=5, sam_rho=0.05.

**Layer-wise params (precomputed at init):**
- `_flat_layer_beta1s[i] = β1 * (1-gamma)^i`
- `_flat_layer_alphas[i] = (1-gamma_alpha)^(max_idx-i)` (or 1.0 if gamma_alpha==0)

**Host-side methods (KEPT, not removed):**
- `sam_step()`: functional_call SAM perturb → autograd.grad → stores sharpness diff
- `meta_step()`: two-term bilevel objective `val_CE(w+) + train_CE(w+)` via functional_call; batched meta-net forward; train_x/train_y optional
- `step_full()`: complete self-contained training step

**State:** 4 flat tensor lists (`_flat_exp_avgs`, `_flat_exp_avg_sqs`, `_flat_mus`, `_flat_sharpness`), `_flat_steps`, `_global_step`, `_cached_alpha`, `_cached_train_acc`.

**Checkpoint:** Custom `state_dict()`/`load_state_dict()` with `_EXTRA_KEY = "supergrok11_extra"`. Saves all flat moment tensors + meta_net state_dict + auto_meta_opt state_dict.

**Alpha update signal:** if train_acc >= zero_acc_threshold OR train_loss < zero_loss_threshold → signal=10.0 (memorized); else signal = max(0, (val-train)/train) (train-normalised, NOT max-normalised like GrokAdamW).

### 6.11 `supergrok15.py`
**Gate:** accuracy-scalar sigmoid gate `sigmoid(gate_scale * (train_acc - gate_thresh))`.

**Adaptive scheduling (sigmoid-driven):**
- SAM freq: `sam_freq_max - (max-min) * sam_heat` in [sam_freq_min=3, sam_freq_max=20]
- Bilevel freq: same pattern, [bilevel_freq_min=5, bilevel_freq_max=30]
- WD ramp: `base_wd * (1.0 + wd_ramp * sigmoid_val)`, wd_ramp=4.0

**`bilevel_step()`:** Two-term objective `val_CE(w+) + train_CE(w+)` (ALWAYS both terms; unlike SG11's optional train_x/train_y, this always uses train batch).

**SharpnessMetaNet:** Identical to SG11 (duplicated by design for self-containment).

**State and checkpoint:** Same pattern as SG11, key `"supergrok15_extra"`.

### 6.12 `supergrok2.py` (2397 lines)

**Architecture: CSA/HCA Hybrid Attention + 4-Head PEER + GRU**

Replaced bidirectional Mamba-3 scan with DeepSeek-V4-style compressed attention:

#### HybridCompressedAttention (line 332)
- `mode='csa'` (CSA): stride-4 weighted pooling, lightning indexer top-k=16, sliding window=8, multi-head softmax
- `mode='hca'` (HCA): stride-128 mean pool, dense attention over all compressed entries + sliding window
- Both: stateless across optimizer steps; d_model=8, n_heads=2, head_dim=4

#### CSAHCAMetaNet (line 563)
- `input_proj`: Linear(2, d_model)
- `csa_layer`: HybridCompressedAttention(mode='csa')
- `hca_layer`: HybridCompressedAttention(mode='hca')
- `gru`: MiniGRU(input_dim=2+2*d_model, hidden_dim=gru_hidden=4)
- PEER: 4 heads × (144 experts = 12×12 product-key), expert_hidden=16, top-k=4
- Dynamic expert recycling: interval=100, threshold=0.001; dead experts cloned from top performer
- `Mamba3PEERMetaNet = CSAHCAMetaNet` (back-compat alias, line 1135)

#### Forward pipeline (line 662):
1. Flatten grad + sharpness → sort by |grad|
2. `input_proj([g_sorted, s_sorted])` → x [N, d_model]
3. `csa_layer(x)` → csa_ctx [N, d_model]
4. `hca_layer(x)` → hca_ctx [N, d_model]
5. Unsort; GRU(`[g, s, csa_ctx, hca_ctx]`) → new_gru
6. PEER(`[new_gru, csa_ctx, hca_ctx, g, s]`) → expert_out (4 heads, top-k=4 per head)
7. `smart_grad = g + rescale * expert_out / num_heads`
8. Returns `(smart_grad, new_gru, None, None)` (None = legacy scan states)

#### PrecisionConfig (line 98)
- `projection_precision`: auto→bf16 if bf16 supported, else fp32; explicit: fp32/tf32/bf16/fp8
- `expert_precision`: fp32/int8/int4
- `dynamic=True`: adjusts tier based on grad_norm CV (coefficient of variation)
- 4 tiers: fp32→tf32→bf16→fp8 for projections; fp32→fp32→int8→int4 for experts

#### SuperGrok2 class (line 1143)
- Single param group ONLY (raises NotImplementedError if len>1, line 1229)
- `vocab_size` param (default 97) passed to kernel args
- `state_precision='config3'`: INT8 exp_avg + BF16 exp_avg_sq/mus/slows/sharpness/gru_states
- `_flat_slows`: NEW grokfast slow EMA buffer (restored grokfast term, distinct from mu)
- `_flat_gru_states`: per-param GRU hidden state `[N, gru_hidden]`
- `_flat_mamba_fwd/bwd_states`: None placeholders (back-compat)
- `bilevel_step()`: ALWAYS uses `_bilevel_step_autograd()` (C++ VJP path dropped, line 1721)
- `_bilevel_step_cuda()`: explicitly raises NotImplementedError (line 1793)
- `bilevel_step_distributed()`: thin wrapper; relies on bilevel_allreduce_meta_grads flag
- `_allreduce_meta_grads()`: all-reduce meta-net grads across ranks
- `_allreduce_expert_counts()`: all-reduce expert counts before recycling
- `_sync_mamba_states()`: no-op (attention is stateless)

#### MoEAwareSuperGrok2 (line 2226)
- Extends SuperGrok2 for MoE models
- `active_expert_indices` → compact filter → scatter back
- Ops: `moe_count_expert_activations`, `moe_compute_load_balance_loss`, `moe_apply_frequency_scaling`, `moe_filter_active_params`, `moe_scatter_results`
- Frequency-based per-expert LR scaling (min=0.1x, max=10x)
- `step()` also raises NotImplementedError (pure L3-TC)
- Load balance loss via `load_balance_coeff=0.01`

#### CompiledSuperGrok2 (line 2046)
- CUDA graph wrapper for SuperGrok2
- `step()` raises NotImplementedError (the underlying step_compiled is also removed)
- Infrastructure kept: `_capture_graph()`, static grad buffer management, `invalidate()`
- Warmup-capture-replay cycle design but effectively dead under L3-TC

#### Checkpoint fidelity
All three (SG11/15/2) override `state_dict()`/`load_state_dict()` to persist:
- All flat moment tensors (exp_avg/exp_avg_sq/mus/slows/sharpness/gru_states/steps)
- meta_net state_dict (the trained net, not just weights)
- auto_meta_opt state_dict (the internal Adam over meta params)
- Global step counters, cached_alpha, cached_train_acc

---

## 7. Front-End API — How to Use

### Call 1: Pick a model and optimizer
```python
from grokking_optimizers import SuperGrok2, Lion, GrokAdamW
# or any of: AdamW, SuperGrok15, SuperGrok11, Grokfast, LookSAM, Muon, NeuralGrok, Prodigy
```

### Call 2: Construct optimizer
```python
opt = SuperGrok2(model.parameters(), lr=1e-3, ...)
# OR for Muon: Muon(model.parameters(), ...)  # auto-splits 2D vs 1D params
```

### Call 3: Connect to dataset
```python
from grokking_optimizers.dataset_sources import make_source_for_task
tx, ty, vax, vay, tex, tey = make_source_for_task(config, seed)
# OR use the legacy make_data_for_task (modular path)
```

### Call 4: Training (L3-TC path only)
```python
from grokking_optimizers.dispatch import fused_train_step
fused_train_step(model_canon, optimizer_name, module, opt, tx, ty,
                 state_cache=state_cache, step=step, gemm_impl="wgmma")
```
All 11 optimizer `.step()` methods raise NotImplementedError; the only valid training path is `fused_train_step`.

---

## 8. Key Discrepancies vs. Claimed State

1. **SG2 bilevel C++ path**: RESUME.md/PROGRESS.md claim C++ bilevel VJP was implemented. Code at supergrok2.py:1721 shows it is DROPPED; `_bilevel_step_cuda()` raises NotImplementedError; autograd path is always used.

2. **CompiledSuperGrok2.step()**: The `CompiledSuperGrok2` class exists but its `step()` also raises NotImplementedError (L3-TC megakernel only). It is effectively a dead shell.

3. **Dataset stub sizes**: The LM stub uses `seq=4` (hardcoded, not from c["seq_len"]); the decoder kernel uses kSeq=4 — consistent. But the mamba stub uses `c.get("seq_len", 8)` which may differ from the kernel's actual seq dimension.

4. **NeuralGrok hidden dim mismatch risk**: The _Amplifier default `hidden_dim=128` differs from `KERNEL_PSI_HIDDEN=16`. The `OPTIMIZER_CONFIGS` is claimed to pin `neural_hidden=16`, but this is in dispatch.py (not visible here). The psi_pack assertion enforces this at runtime.

5. **MoEAwareSuperGrok2._moe_step()**: Falls through to `super().step(**kwargs)` at line 2353, but `super().step()` raises NotImplementedError. This path is unreachable unless ops expose `moe_filter_active_params`, but even then it would fail. The MoE path is effectively dead under L3-TC.

---

## 9. Open Items / Bugs

1. **All eager `.step()` methods removed** — training requires L3-TC `fused_train_step`. No Python reference path for correctness debugging outside the kernel.

2. **NeuralGrok: num_layers constraint** (neuralgrok.py:159): `psi_pack()` raises ValueError if `len(self.amplifier.net) != 3` (i.e., num_layers != 2). The default is num_layers=3 → psi_pack fails unless overridden to 2.

3. **GrokAdamW: per-tensor β1 ABI** (grokadamw.py:48): Comment says "until the vector ABI lands we partition the fused call per tensor; post rebuild a single call passes the whole β1 vector." Layer-wise β1 differences are NOT currently reflected in kernel dispatch (the kernel takes a single β1 scalar).

4. **dataset_sources.py seq=4 hardcoded** (line 77): For decoder, seq is hardcoded to 4 instead of read from config. Comment says "current decoder kSeq" — but this is a doc debt if kSeq ever changes.

5. **Prodigy EMA design** (prodigy.py:80-87): Comment claims `_r_ema`/`_s_ema` are EMAs decayed by β3=sqrt(β2), but the Python step() is REMOVED. Whether the kernel actually implements this EMA (vs the simpler instantaneous form) is unverifiable from Python alone.

6. **SuperGrok2 single param group restriction** (supergrok2.py:1222-1233): Raises NotImplementedError for multiple groups. This limits flexibility but is stated as queued work.

7. **Muon 2D auto-split**: `ndim != 2` → AdamW group. Conv weights (4D) go to AdamW; this may not be the intended behavior for all models.

8. **CompiledSuperGrok2**: Entirely dead code under pure L3-TC. The CUDA graph infrastructure exists but cannot be used.

---

## 10. Summary of Config-Derivation / Adaptivity in Python Layer

The Python layer handles these adaptive decisions (HOST-SIDE, before kernel launch):

| Component | Adaptive computation | Inputs |
|-----------|---------------------|--------|
| `dataset_sources._route()` | Selects stub by `data_source` then `model_type` | `c["data_source"]`, `c["model_type"]` |
| `lowprec.swap_linears_lowprec()` | Decides which layers can run native kernel | `in_features`, `out_features` |
| `PrecisionConfig.update_dynamic()` | Adjusts precision tier by grad_norm CV | `grad_norm_ema`, `grad_norm_var_ema`, step count |
| `SG15._get_effective_sam_freq()` | SAM frequency from training accuracy sigmoid | `_cached_train_acc`, `sam_scale`, `sam_thresh`, `sam_freq_min/max` |
| `SG15._get_effective_bilevel_freq()` | Meta-update frequency from accuracy sigmoid | same pattern |
| `SG15._get_effective_wd()` | Progressive weight decay from accuracy | `_cached_train_acc`, `wd_ramp`, `wd_scale`, `wd_thresh` |
| `SG11/15/2._update_alpha()` | EMA decay rate from grokking signal | `train_acc`, `train_loss`, `val_loss`, `zero_*_threshold` |
| `GrokAdamW._alpha_for_group()` | α from max-normalised val/train gap | `_grok_signal`, `alpha_init`, `kappa` |
| `GrokAdamW._layer_beta1()` | Per-param β1 from layer index and gamma | `gamma`, layer index |
| `_tuned_inject.source_extra_nvcc_flags()` | Per-TU nvcc flags from tuning JSON | arch, optimizer, model keys |
| `tune_hook._effective_seed()` | Regime → seed for deterministic workloads | regime name, base seed |

None of these is hardcoded based on model size or GPU count. Each is a pure function of the config dict / optimizer hyperparams / measured training statistics.
