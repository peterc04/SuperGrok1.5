# Hardware Validation Runbook — SuperGrok2

This is the **gate to promote any cell 🟡 → ✅**. Every stage of the
performance build appends its concrete on-silicon checks here. Nothing in this
repo is `✅` until the relevant section below passes on real
**H100 / MI300X / TPU v5p**.

> **Status legend**
> - 🟡 = implemented + structurally / compile verified (nvcc `-c` to object on a
>   CPU host, or preprocessor-equivalence for pure moves), **NOT yet
>   hardware-validated**.
> - ✅ = bit-level reference-checked + profiled on the real target accelerator
>   via the procedure in this file.

## 0. Environment bring-up (run once per machine)

### NVIDIA (H100, sm_90a)
```bash
# toolchain
nvidia-smi                      # confirm H100 visible, driver >= 545
nvcc --version                  # CUDA >= 12.3 for full Hopper (wgmma/TMA/cluster)
git submodule update --init third_party/cutlass

# build the extension for the real device
FORCE_CUDA=1 WITH_CUTLASS=1 TORCH_CUDA_ARCH_LIST="9.0a" \
  pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/build_h100.log
python -c "import grokking_optimizers._C as c; print('ext loaded', c)"
```

### AMD (MI300X, gfx942)
```bash
rocminfo | grep gfx942         # confirm CDNA3 visible
hipcc --version                # ROCm >= 6.1 for FP8 FNUZ + full MFMA
FORCE_HIP=1 PYTORCH_ROCM_ARCH=gfx942 \
  pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/build_mi300.log
python -c "import grokking_optimizers._C as c; print('ext loaded', c)"
```

### TPU (v5p)
```bash
python -c "import jax; print(jax.devices())"   # expect TPU v5p chips
# Pallas path is pure-Python; no extension build required.
```

## 1. Per-cell numerics reference-check (the ✅ gate)

For each (optimizer × model × arch) cell, the bit-level oracle compares the
fused/kernel output against the pure-PyTorch (CPU, fp64-accumulate) reference
math for N=20 steps. A cell is promoted to ✅ only when **all** params and
optimizer state match within `rtol=1e-3, atol=1e-5` (bf16 path) or `1e-5/1e-7`
(fp32 path).

```bash
# generic harness (to be implemented in tests/hw/test_reference_parity.py)
python -m tests.hw.test_reference_parity \
    --optimizer <adamw|lion|grokfast|grokadamw|looksam|muon|neuralgrok|prodigy|supergrok11|supergrok15|supergrok2> \
    --model <transformer|vit|mamba> \
    --arch <sm_90|gfx942|tpu_v5p> \
    --steps 20 --dtype <bf16|fp32> \
    --rtol 1e-3 --atol 1e-5
```

### Cell status matrix (filled in as stages land; all 🟡 until silicon)

| optimizer    | transformer | vit | mamba | notes |
|--------------|:-----------:|:---:|:-----:|-------|
| adamw        | 🟡 | 🟡 | 🟡 | elementwise math proven; kernel path hw-deferred |
| lion         | 🟡 | 🟡 | 🟡 | |
| grokfast     | 🟡 | 🟡 | 🟡 | |
| grokadamw    | 🟡 | 🟡 | 🟡 | fused + Q3 quant path needs numeric check |
| looksam      | 🟡 | 🟡 | 🟡 | SAM perturb/restore two-pass |
| muon         | 🟡 | 🟡 | 🟡 | Newton–Schulz fused combine+update |
| neuralgrok   | 🟡 | 🟡 | 🟡 | psi-net amplifier |
| prodigy      | 🟡 | 🟡 | 🟡 | d_lr global reduction |
| supergrok11  | 🟡 | 🟡 | 🟡 | cosine gate + meta-net |
| supergrok15  | 🟡 | 🟡 | 🟡 | global sigmoid gate |
| supergrok2   | 🟡 | 🟡 | 🟡 | CSA/HCA fwd + **bilevel backward (Stage 1A, hand-written adjoint, done)** |

## 2. Per-stage hardware checks (appended as stages complete)

### Stage 0 — compile correctness
- [ ] `FORCE_CUDA=1 WITH_CUTLASS=1 pip install -e .` links a loadable `_C`
      extension on an H100 box (the CPU-host `nvcc -c` gate cannot catch device
      link / ptxas-lowering errors).
- [ ] `cuobjdump -sass build/.../launch_adamw.o | head` shows real SASS.

<!-- Stage 1+ checks are appended below by each stage. -->

### Stage 1A — SG2 bilevel backward (CSA/HCA hand-written adjoint)

The saved-activation reverse-mode VJP through the SuperGrok2 CSA/HCA meta-net
(`input_proj+sort → CSA → HCA → GRU → PEER → smart_grad`). Implemented as a
REAL hand-written adjoint (NO autograd, NO autograd graph, NO throw). The
vendor-neutral math lives in `csrc/algorithms/supergrok2_bilevel_adjoint.h` and
is shared bit-for-bit by both backends:

- **sm_90** (`grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh`): the 4
  launchers (`launch_csa_hca_bilevel_fwd_save[_batched]`,
  `launch_csa_hca_backward[_batched]`) orchestrate the shared ATen adjoint
  (nvcc TU; ATen host orchestration, matching the existing `detail::` forward).
- **gfx942** (`grokking_optimizers/kernels/gfx942/supergrok2_gfx942.hip.hpp`):
  the same 4 launchers, ATen-based (host-compiler TU), per the AMD-native
  stance — Stage 5 lowers to raw HIP. The previous `csa_hca_bilevel_not_implemented`
  throws are GONE.

**Kernels / stages written (reverse pipeline order):**
1. `smart_grad → expert_out`: `d_total = rescale * d_smart_grad`, split over heads.
2. PEER expert-MLP + product-key routing backward (`peer_head_backward`): VJP
   through the relu 2-layer expert MLP (atomic `index_add_` into
   `d_expert_{W1,b1,W2,b2}`), the `soft_a⊗soft_b` routing softmax (×10 temp),
   the top-k gather (scatter back to `d_scores_{a,b}`), and the query projection
   (`d_peer_query_Ws`, `d_prod_keys_{A,B}`) → `d_peer_input`.
3. GRU backward (`bilevel_backward_driver` §4): VJP through z/r/h̃ gates →
   `d_gru_{Wz,bz,Wr,br,Wh,bh}`; gates recomputed/saved from `gru_input + h_old`.
4. HCA dense-attention backward (`hca_forward`/`hca_backward`): softmax VJP +
   mean-pool compression scatter + window scatter → `d_hca_{q,k,v,out}_W`.
5. CSA sparse-attention backward (`csa_forward`/`csa_backward`): joint
   (selected-compressed ∪ window) softmax VJP, learned-pool compression VJP
   (`d_csa_compress_w` via softmax-pool), selected-entry + window scatter →
   `d_csa_{q,k,v,out}_W`. Lightning-indexer (`d_csa_idx_{DQ,UQ,K}`) handled per
   the discrete-topk stop-gradient note below.
6. input_proj + sort backward: accumulate `d_x_sorted` from the q/k/v projection
   adjoints of both blocks, then `d_input_proj_W[:,0]+=Σ d_x·g_sorted`,
   `[:,1]+=Σ d_x·s_sorted`, `d_input_proj_b+=Σ d_x` (sort handled by the
   `unsort/sort_idx` permutation on the saved contexts).

**fwd_save save-set:** `x_sorted`, `sort_indices`, `csa_ctx`, `hca_ctx`,
`csa_saved_sel_idx`, `csa_saved_probs`, `hca_saved_probs`,
`csa_saved_denom`/`hca_saved_denom` (informational), plus the GRU
(`gru_input`, `gru_h_old`, `gru_z/r/h_tilde`) and PEER (`peer_input`,
`expert_indices`, `routing_weights`, `saved_z_hidden`, `saved_scores_{a,b}`,
`saved_top_{a,b}_idx`, `saved_soft_{a,b}`) adjoint tensors declared in the
bindings.cpp signature.

**Checkpointing choice (honors `checkpoint_interval ≤ MAX_CKPT_INTERVAL=32`):**
fwd_save persists the heavy per-layer contexts (`csa_ctx`, `hca_ctx`) and the
softmax probs/sel sets; the backward RECOMPUTES the cheap per-row q/k/v +
indexer projections and the compressed K/V pools from the saved `x_sorted` via
the shared `csa_forward`/`hca_forward` helpers (i.e. layer-boundary activation
checkpointing). This is a strict superset of any interval ≤ 32 — no information
is dropped, so the parameter is accepted and threaded through but the recompute
granularity is the layer boundary for this first correct cut.

**Numerics oracle (HARDWARE-DEFERRED — no GPU here):** bit-level autograd
parity. For N=20 fp32 rows, build the oracle `CSAHCAMetaNet.forward_for_bilevel`
(`grokking_optimizers/optimizers/supergrok2.py:734`), backprop a random
`d_smart_grad` with `torch.autograd.grad`, and compare each of the 24
`d_*_W/b` buffers from `launch_csa_hca_backward` against the autograd reference
at `rtol=1e-3, atol=1e-5`:

```python
# N=20, fp32, on an sm_90 / gfx942 device:
import torch
from grokking_optimizers.optimizers.supergrok2 import CSAHCAMetaNet
net = CSAHCAMetaNet(d_model=8).cuda().double().float().train()
g = torch.randn(20, device='cuda', requires_grad=False)
s = torch.randn(20, device='cuda')
h = torch.zeros(net.gru_hidden, device='cuda')
for p in net.parameters(): p.requires_grad_(True)
smart, *_ = net.forward_for_bilevel(g, s, h)
dsg = torch.randn_like(smart)
refs = torch.autograd.grad(smart, list(net.parameters()), dsg, allow_unused=True)
# ... run fwd_save + launch_csa_hca_backward, compare d_*_W vs refs (rtol1e-3/atol1e-5)
```

**Ledger — the 24 weight-grad buffers** (🟢 = full analytic adjoint, expected
bit-parity once run on device; 🟡 = analytic but flagged for on-device confirm):

| buffer | status | note |
|--------|--------|------|
| d_input_proj_W | 🟢 | from accumulated d_x_sorted × (g,s)_sorted |
| d_input_proj_b | 🟢 | Σ d_x_sorted |
| d_csa_q_W | 🟢 | dq.t()@x |
| d_csa_k_W | 🟢 | window + compressed-pool scatter |
| d_csa_v_W | 🟢 | window + compressed-pool scatter |
| d_csa_out_W | 🟢 | d_csa_out.t()@ctx |
| d_csa_compress_w | 🟡 | softmax-pool VJP through normalized weighted pool; confirm on device |
| d_csa_idx_DQ | 🟡 | indexer feeds only the discrete top-k index (stop-grad) → exactly 0 by construction; oracle likewise yields 0 grad to idx_* |
| d_csa_idx_UQ | 🟡 | same as idx_DQ |
| d_csa_idx_K | 🟡 | same as idx_DQ |
| d_hca_q_W | 🟢 | dq.t()@x (multi-head split) |
| d_hca_k_W | 🟢 | window + mean-pool scatter |
| d_hca_v_W | 🟢 | window + mean-pool scatter |
| d_hca_out_W | 🟢 | d_hca_out.t()@ctx |
| d_gru_Wz | 🟢 | sigmoid VJP, dz·z(1-z) |
| d_gru_bz | 🟢 | Σ d_pre_z |
| d_gru_Wr | 🟢 | sigmoid VJP |
| d_gru_br | 🟢 | Σ d_pre_r |
| d_gru_Wh | 🟢 | tanh VJP, (1-h̃²) |
| d_gru_bh | 🟢 | Σ d_pre_h |
| d_peer_query_Ws | 🟢 | d_query.t()@peer_input, per head |
| d_prod_keys_A | 🟢 | d_scores_a.t()@q_a, per head |
| d_prod_keys_B | 🟢 | d_scores_b.t()@q_b, per head |
| d_expert_W1 | 🟢 | atomic index_add of (d_pre_z·g) per active expert |
| d_expert_b1 | 🟢 | atomic index_add of d_pre_z |
| d_expert_W2 | 🟢 | atomic index_add of (d_out·z) per active expert |
| d_expert_b2 | 🟢 | atomic index_add of d_out |

**Stop-gradient note (d_csa_idx_*):** the lightning indexer's only consumer is
`idx_scores.topk(...).indices` — a non-differentiable argmax/top-k index. The
oracle's autograd returns `None` (zero) grad to `idx_DQ/idx_UQ/idx_K` for a pure
top-k-index path, so the adjoint correctly accumulates zero into those three
buffers. Marked 🟡 only to flag for explicit on-device confirmation that the
oracle indeed yields zero (rather than a surrogate) for those parameters.

**Stage-1A correctness re-review (Opus 4.8, 2026-06-01) — `scripts/REVIEW_1A.md`.**
The hand-written adjoint was re-verified by a line-by-line Python transcription
of `supergrok2_bilevel_adjoint.h` compared against `torch.autograd.grad` through
the real `forward_for_bilevel` (CPU fp32). **All 24 weight-grad buffers match
autograd to fp32 precision** across 4 configs (incl. production defaults
csa_compress=4/window=8/topk=16/hca_compress=128, and small-N=5 edge), each run
with zero AND nonzero `gru_state` to exercise the reset-gate path. No VJP / sign
/ dropped-gradient bug found; **nothing changed in the adjoint**. Two residual
🟡 items surfaced for on-silicon follow-up:

- 🟡 **GRU-gate recompute fallback drops biases.** If a caller passes *empty*
  `gru_z_gate/gru_r_gate/gru_h_tilde` to `launch_csa_hca_backward`, both backends
  recompute the gates **without** `gru_{bz,br,bh}` (sm90:1684/1687/1691,
  gfx942:875/878/882). The oracle gates include the biases, so the fallback is
  inexact. The documented save-set provides the gates (canonical path is exact,
  and is what the parity test validates), but the backward ABI cannot recompute
  them bias-correctly. **On-device check:** assert the fwd_save→backward harness
  threads the saved gru gates (non-empty), OR extend the backward ABI to accept
  `gru_{bz,br,bh}` and recompute with biases. Until then the no-bias fallback
  must never be hit.
- 🟡 **Output-buffer zero-init is a caller contract with no guard.** The driver
  accumulates (`add_`/`index_add_`) into the 24 `d_*` buffers; the bindings do
  NOT zero them. **On-device check:** the numerics harness must `zero_()` all 24
  `d_*` buffers before each `supergrok2_bilevel_backward` call (or add a guard in
  the binding), else grads accumulate across calls.

### Stage 1B — MoE compaction (`MoEAwareSuperGrok2._moe_step`)

Nine MoE-compaction kernels implemented: sm_90 as real `__global__` CUDA
(grid-stride + atomics), gfx942 as ATen tensor ops. Reachable (called by
`_moe_step`): `count_expert_activations`, `compute_load_balance_loss`,
`apply_frequency_scaling`, `filter_active_params`, `scatter_results`. Exported
but not yet called: `dynamic_expert_{load,fwd,bwd}` (real), `scan_compacted`
(VESTIGIAL — Mamba-era SSM; SG2's mixer is CSA/HCA). CPU-host `nvcc -c` /
host-ATen builds cannot run these; each needs a bit-level oracle on silicon.

Per-kernel bit-level oracle (run on H100 / MI300X; fp32 path, `rtol=1e-5,
atol=1e-7`):

```bash
# 1. moe_count_expert_activations — histogram vs (gate_logits>thr).sum(0)
python -m tests.hw.test_reference_parity --kernel moe_count_expert_activations \
    --arch <sm_90|gfx942> --ref "(gate_logits>thr).sum(0).int()" \
    --rtol 0 --atol 0
# 2. moe_compute_load_balance_loss — vs E*Σ_e (counts[e]/N)*mean_t softmax(gl)[t,e]
python -m tests.hw.test_reference_parity --kernel moe_compute_load_balance_loss \
    --arch <sm_90|gfx942> --dtype fp32 --rtol 1e-5 --atol 1e-7
# 3. moe_apply_frequency_scaling — vs clamp((1/E)/((c+s)/(tot+s*E)),lo,hi)
python -m tests.hw.test_reference_parity --kernel moe_apply_frequency_scaling \
    --arch <sm_90|gfx942> --dtype fp32 --rtol 1e-5 --atol 1e-7
# 4. moe_filter_active_params — kept set == {i: expert_active[p2e[i]]!=0};
#    sm_90 uses atomic compaction (order-agnostic), so compare as a SET keyed
#    by scatter_indices, not positionally.
python -m tests.hw.test_reference_parity --kernel moe_filter_active_params \
    --arch <sm_90|gfx942> --compare set-by-scatter-index --rtol 0 --atol 0
# 5. moe_scatter_results — round-trip: filter then scatter == identity on kept idx
python -m tests.hw.test_reference_parity --kernel moe_scatter_results \
    --arch <sm_90|gfx942> --roundtrip-with moe_filter_active_params --rtol 0 --atol 0
# 6. moe_dynamic_expert_load — packed slices == expert_*[active_mask!=0]
python -m tests.hw.test_reference_parity --kernel moe_dynamic_expert_load \
    --arch <sm_90|gfx942> --rtol 0 --atol 0
# 7. moe_dynamic_expert_fwd — vs rw*(W2_e@relu(W1_e@x+b1_e)+b2_e)
python -m tests.hw.test_reference_parity --kernel moe_dynamic_expert_fwd \
    --arch <sm_90|gfx942> --dtype fp32 --rtol 1e-5 --atol 1e-7
# 8. moe_dynamic_expert_bwd — gradcheck VJP vs torch.autograd on the #7 graph
python -m tests.hw.test_reference_parity --kernel moe_dynamic_expert_bwd \
    --arch <sm_90|gfx942> --gradcheck --dtype fp64 --rtol 1e-5 --atol 1e-7
# 9. moe_scan_compacted (VESTIGIAL) — vs sequential discretized SSM recurrence
python -m tests.hw.test_reference_parity --kernel moe_scan_compacted \
    --arch <sm_90|gfx942> --dtype fp32 --rtol 1e-5 --atol 1e-7
```

- [ ] sm_90: all 9 oracles pass on H100 (fp32).
- [ ] gfx942: all 9 oracles pass on MI300X (fp32).
- [ ] sm_90 SASS for `moe_*_kernel` shows real `red.global.add` / `atom.global.add`
      for the histogram + compaction atomics (not a serialized fallback).

## 3. Profiling commands (per arch)

```bash
# NVIDIA: kernel-level counters (occupancy, mem throughput, tensor-core util)
ncu --set full -k "regex:adamw|supergrok2|attention" \
    -o /tmp/ncu_sg2 python -m tests.hw.run_one_step ...

# AMD
rocprof --stats --hip-trace python -m tests.hw.run_one_step ...

# TPU
python -c "import jax; jax.profiler.start_trace('/tmp/tpu_trace'); ...; jax.profiler.stop_trace()"
```

## 4. Tensor-core / asm emission verification

A kernel that *should* use tensor cores is only ✅ once the instruction is
confirmed in the disassembly:

```bash
# NVIDIA — confirm WGMMA / HMMA actually emitted (not SIMT fallback)
cuobjdump -sass <obj>.o | grep -E 'wgmma|HMMA|hgmma'    # CUTLASS Sm90 GEMMs
cuobjdump -sass <obj>.o | grep -E 'cp.async|TMA|UTMALDG'
# AMD — confirm MFMA emitted
roc-obj-ls <obj>; roc-obj -d <obj> | grep -E 'v_mfma|buffer_load.*lds|v_.*dpp'
```

## 5. Autotuner effect verification (Stage 3/4)

Two distinct `SG_TUNED_BLOCK_SIZE` values must produce **different** SASS and
measured latency (today they only perturb the cache key):

```bash
SG_TUNED_BLOCK_SIZE=128 <build one obj> ; cuobjdump -sass > /tmp/a.sass
SG_TUNED_BLOCK_SIZE=512 <build one obj> ; cuobjdump -sass > /tmp/b.sass
diff <(grep -c . /tmp/a.sass) <(grep -c . /tmp/b.sass)   # expect non-empty diff
```

## 6. Distributed scaling (Stage 7)

```bash
# 3D-parallel 5–10B-param harness; record scaling efficiency vs ideal linear.
torchrun --nnodes=$N --nproc_per_node=8 \
  tests/hw/test_3d_parallel.py --params 7e9 --dp 8 --tp 4 --pp 2 --zero 3
# pass = >= 70% weak-scaling efficiency to 32 GPUs on IB/NVLink fabric.
```

## Stage 1C — decoder + ViT tensor-core (TMA+WGMMA) matmul paths

The transformer-decoder and ViT sm_90 kernels now route their heavy Linear
matmuls through the canonical Sm90 collective builder (`mma::sm90_run_gemm_bt`,
reused from `csrc/backends/cuda/sm_90/mma.cuh`, modeled byte-for-byte on
`attention_sm90.cuh`'s `fmha_sm90_gemm`). Converted matmuls (half/bf16; FP32
keeps the cuBLAS/scalar fallback, gated by `#ifdef WITH_CUTLASS` +
`if constexpr (cutlass_gemm_supported<T>)`):

- **decoder** (`transformer_decoder_sm90.cuh`): QKV projection, output
  projection, FFN up, FFN down, vocab head. (Attention S=Q·Kᵀ and O=P·V already
  run through `attention_sm90.cuh`'s CUTLASS FMHA path.)
- **vit** (`vit_sm90.cuh`): patch-embedding projection, QKV, output projection,
  MLP up, MLP down, classification head. (Attention S/O via the same FMHA path.)

Tensor-core disassembly check (the SASS spelling of the PTX `wgmma` warp-group
MMA is `HGMMA`; `wgmma` matches 0 in SASS — grep `HGMMA` instead. TMA bulk loads
appear as `UTMALDG`):

```bash
bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/models/decoder.cu -DWITH_CUTLASS  # COMPILE_OK
bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/models/vit.cu     -DWITH_CUTLASS  # COMPILE_OK
# build real .o (compile_to_object.sh emits to /dev/null), then disassemble:
nvcc -c -std=c++17 -DWITH_CUDA -DWITH_CUTLASS -gencode arch=compute_90a,code=sm_90a \
  -I. -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include \
  --expt-relaxed-constexpr --expt-extended-lambda \
  csrc/backends/cuda/sm_90/models/decoder.cu -o /tmp/decoder.o
cuobjdump -sass /tmp/decoder.o | grep -ciE 'wgmma|hgmma'   # 64  (HGMMA.64x128x16.F32[.BF16])
cuobjdump -sass /tmp/decoder.o | grep -ciE 'utmaldg'       # 50  (UTMALDG.3D — TMA)
cuobjdump -sass /tmp/vit.o     | grep -ciE 'wgmma|hgmma'   # 64
cuobjdump -sass /tmp/vit.o     | grep -ciE 'utmaldg'       # 50
```

Observed (arch=sm_90a): decoder.o → 64 HGMMA (32× `HGMMA.64x128x16.F32`,
32× `HGMMA.64x128x16.F32.BF16`) + 50 `UTMALDG.3D`; vit.o → identical 64 HGMMA +
50 `UTMALDG.3D`. Literal `grep wgmma` = 0 (PTX-level mnemonic; SASS = HGMMA).
Numerics (FP32-accumulate parity vs cuBLAS) are hardware-deferred.

---

### Deferred-check ledger (every runtime check the build could not run on CPU)
Each stage appends one line per deferred check: `STAGE | cell | what to verify | command`.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 0 | all | device link + SASS sanity | §2 Stage 0 |
| 1A | supergrok2/bilevel | 24 d_*_W/b buffers vs `torch.autograd.grad` through `forward_for_bilevel`, N=20, rtol1e-3/atol1e-5 | Stage 1A oracle snippet |
| 1A | supergrok2/bilevel | d_csa_compress_w softmax-pool VJP parity | Stage 1A ledger 🟡 |
| 1A | supergrok2/bilevel | d_csa_idx_{DQ,UQ,K} = 0 (discrete top-k stop-grad) parity | Stage 1A stop-grad note |
| 1A | supergrok2/bilevel | fwd_save save-set round-trips backward (x_sorted/ctx/probs) | Stage 1A save-set |
| 1C | decoder | QKV/out-proj/FFN-up/FFN-down/vocab-head emit HGMMA+TMA (not SIMT) | `cuobjdump -sass decoder.o \| grep -ciE 'wgmma\|hgmma'` (=64), `\| grep -ci utmaldg` (=50) |
| 1C | vit | patch-embed/QKV/out-proj/MLP-up/MLP-down/head emit HGMMA+TMA | `cuobjdump -sass vit.o \| grep -ciE 'wgmma\|hgmma'` (=64), `\| grep -ci utmaldg` (=50) |
| 1C | decoder+vit | bf16/fp16 GEMM numerics match cuBLAS FP32-acc within tol on H100 | run model fwd/bwd oracle on sm_90 device |
| 1B | supergrok2/moe | moe_count_expert_activations histogram parity | §2 Stage 1B #1 |
| 1B | supergrok2/moe | moe_compute_load_balance_loss aux-loss parity | §2 Stage 1B #2 |
| 1B | supergrok2/moe | moe_apply_frequency_scaling clamp parity | §2 Stage 1B #3 |
| 1B | supergrok2/moe | moe_filter_active_params kept-set (atomic order) | §2 Stage 1B #4 |
| 1B | supergrok2/moe | moe_scatter_results filter→scatter round-trip | §2 Stage 1B #5 |
| 1B | supergrok2/moe | moe_dynamic_expert_load masked-gather parity | §2 Stage 1B #6 |
| 1B | supergrok2/moe | moe_dynamic_expert_fwd MLP parity | §2 Stage 1B #7 |
| 1B | supergrok2/moe | moe_dynamic_expert_bwd VJP gradcheck | §2 Stage 1B #8 |
| 1B | supergrok2/moe | moe_scan_compacted (vestigial) SSM recurrence | §2 Stage 1B #9 |
| 1B | supergrok2/moe | sm_90 atomic-add SASS emission | §2 Stage 1B SASS |

## Stage 2 — L2 persistence (§6.1)

Per-step optimizer state (m, v, EMA, μ, momentum, tracking) is hinted L2-resident
across the step via `prim::L2PersistScope` (RAII; `cudaStreamSetAttribute` +
`cudaAccessPolicyWindow` — the safe runtime API, NOT `createpolicy` PTX). Gated
by `ENABLE_L2_PERSIST` + a runtime size check against
`cudaDevAttrMaxPersistingL2CacheSize` (no-op on pre-Hopper or when the state span
exceeds the reservable persisting-L2, ~50 MB on H100).

Wired into the primary step launcher of all 11 optimizers (state buffers persisted):
adamw, grokfast, grokadamw, neuralgrok, prodigy, supergrok11, supergrok15,
supergrok2 (exp_avg + exp_avg_sq); lion (exp_avg); muon (momentum buf);
looksam (exp_avg + exp_avg_sq in the apply step).

**Hardware checks (deferred — all 🟡):**
- L2 hit-rate uplift on the state buffers:
  `ncu --metrics lts__t_sector_hit_rate.pct,lts__t_sectors_aperture_device_op_read.sum -k "regex:adamw|supergrok2" python -m tests.hw.run_one_step ...`
  Expect a measurable rise in `lts__t_sector_hit_rate.pct` for the m/v reads vs an
  `ENABLE_L2_PERSIST=0` rebuild.
- Correctness must be a NO-OP on numerics: persisting L2 is a cache hint only;
  bit-level optimizer-state parity vs the ENABLE_L2_PERSIST=0 build must be exact
  (rtol=0). Run the Stage-1/§1 per-cell oracle under both builds and diff.
- Confirm the carve-out is released: after a step, `cudaCtxResetPersistingL2Cache`
  + `cudaDeviceSetLimit(...,0)` leave a clean L2 for the next op (check no
  residual persisting window via `ncu` on a following unrelated kernel).

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 2 | all opt | L2 hit-rate uplift on m/v | §Stage 2 ncu |
| 2 | all opt | numerics no-op vs ENABLE_L2_PERSIST=0 | §Stage 2 parity |
| 2 | all opt | carve-out released after step | §Stage 2 reset |

## Stage 3.1 — redux.sync integer reductions

§4.1 (NVIDIA PTX maximization). INTEGER warp/block reductions now use the
single-instruction Ampere/Hopper warp collective `redux.sync.add.u32` instead of
the 5-step `__shfl_xor` butterfly, and the two per-lane INTEGER-atomic histogram
sites in the MoE path are converted to warp-aggregated atomics
(`__ballot_sync` / `__match_any_sync` + `__popc`). FLOAT reductions (norms,
softmax, gradient atomics) are untouched — float `redux` is Blackwell-only
(sm_100+) and out of scope.

**Helpers added** (`csrc/backends/cuda/sm_90/primitives.cuh`,
`sg::sm90::primitives`, next to the float reducers):
- `warp_reduce_add_u32(unsigned)` — `redux.sync.add.u32` under
  `__CUDA_ARCH__ >= 800`, shuffle-xor tree fallback otherwise (pre-Ampere
  codegen matrix still compiles + works). Every lane receives the full-warp sum.
- `block_reduce_add_u32(unsigned)` — warp-reduce → `smem[warp]` → first-warp
  reduce. Structure (32-u32 smem, `__syncthreads` placement) mirrors
  `block_reduce_sum_f32` exactly.

**Sites converted** (`grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh`):
- `moe_count_expert_activations_kernel` (~L1862): per-element expert histogram.
  Was `atomicAdd(&expert_counts[e], 1)` per active lane. Now lanes ballot the
  hit predicate, `__match_any_sync` groups same-`e` lanes, the group leader
  (lowest set lane) issues one `atomicAdd(&expert_counts[e], __popc(mask))`.
  Trip count rounded up to a whole number of strides so every lane reaches the
  warp ballot each iteration; the `in_range && hit` predicate excludes tail
  lanes. Counts are BIT-IDENTICAL — `popc(mask)` = number of lanes adding 1.
- `moe_filter_active_params_kernel` (~L1983): stream-compaction slot allocation.
  Was `atomicAdd(&compact_count[0], 1)` per kept lane. Now the warp leader
  reserves a contiguous block of `popc(mask)` slots with one atomic; each kept
  lane writes to `base + intra-warp-rank`. Total count identical; ordering among
  kept elements was already irrelevant (scatter map written by stored index).

**Sites left as-is (with reason):**
- `last_block_finished` `atomicInc` (primitives.cuh): one-thread-per-block
  (`threadIdx.x==0` only) — no intra-warp aggregation possible.
- All remaining `atomicAdd` in the sm_90 tree are FLOAT (gradient accumulation
  in supergrok2 backward `db1/db2/dw1/dw2/dz1`, grad_A_log; prodigy
  numerator/denominator; supergrok11 gate num/den; supergrok15 sharpness) —
  out of scope (float redux is Blackwell-only).

**PTX / SASS survival (verified on this host, nvcc 12.x):**
- `nvcc -ptx -arch=sm_90a` on a tiny TU calling the helper emits
  `redux.sync.add.u32 %r1, %r2, 0xffffffff;` (PTX line 33). Instruction is NOT
  optimized away.
- `nvcc -cubin -arch=sm_90a` + `cuobjdump -sass` shows it lowers to
  `REDUX.SUM UR6, R2` in SASS.

**Hardware checks (deferred — all 🟡):**
- On-silicon SASS confirmation in the actual built kernels: after a real
  `-arch=sm_90a` build of `launch_supergrok2`, run
  `cuobjdump -sass <obj> | grep -i REDUX` and confirm `REDUX.SUM` appears in the
  MoE / any u32-reduce kernel (the standalone probe above already shows the
  helper lowers to REDUX; this confirms it in the linked TU).
- Expert-count histogram bit-parity: run the MoE count path under the
  warp-aggregated build and a per-lane-atomic reference build over the same
  `gate_logits`/`threshold`; `expert_counts` must match exactly (rtol=0, every
  element). Likewise `compact_count[0]` and the multiset of compacted rows from
  `moe_filter_active_params` must be identical (order may differ; content/count
  must not).
- Throughput: `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum`
  on the MoE count kernel should show a drop in global-atomic traffic vs the
  pre-change per-lane build, with no change in result.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 3.1 | moe count | REDUX.SUM in built SASS | §Stage 3.1 cuobjdump |
| 3.1 | moe count | expert_counts bit-parity vs per-lane atomics | §Stage 3.1 parity |
| 3.1 | moe filter | compact_count + content parity | §Stage 3.1 parity |
| 3.1 | moe count | global-atomic traffic drop | §Stage 3.1 ncu |

---

## Stage 3.2 — cp.async background loads (NVIDIA PTX maximization §4.2)

Hand-issued `cp.async.{cg,ca}.shared.global` + commit/wait pipelines for the
memory-bound global→shared staging loads in the model attention path and the
SG2 CSA/HCA attention path, with the loads-in-flight (pipeline) depth exposed
as the autotuner dimension `SG_TUNED_ASYNC_DEPTH`.

**Helpers added** (`csrc/backends/cuda/sm_90/primitives.cuh`, namespace
`sg::sm90::primitives`, new §4.2 section appended at end — the Stage 3.1
reduction section was NOT touched):
- `cp_async_cg_16(smem, gmem)` — 16B (float4) `cp.async.cg.shared.global`.
- `cp_async_ca_4(smem, gmem)`  — 4B (scalar float) `cp.async.ca.shared.global`.
- `cp_async_commit()`          — `cp.async.commit_group`.
- `cp_async_wait_group<N>()`   — `cp.async.wait_group N` (N a PTX immediate).
- `cp_async_wait_all()`        — `wait_group 0` (drain all).
All four are guarded `#if __CUDA_ARCH__ >= 800` with a **synchronous-copy
fallback** (plain `float4`/`float` store; commit/wait become no-ops) so the
pre-Ampere codegen matrix still compiles and is numerically identical.

**Kernels converted:**
- Model attention — `grokking_optimizers/kernels/sm_90/attention_sm90.cuh`,
  `fmha_softmax_kernel`: the raw score row `S[i*N + 0..N)` is staged into a new
  `sraw[N]` shared buffer via a depth-pipelined `cp.async.ca` loop (one strided
  tile of `blockDim.x` 4B copies per group, up to ASYNC_DEPTH groups in flight),
  then scale+mask+max read it from shared. Shared alloc grew `N → 2*N` floats
  (`sh[N]` working + `sraw[N]` staging). The fwd/bwd `smem_attention_*_kernel`
  compute QKᵀ directly into shared with no reused global→shared staging loop, so
  they were intentionally left as-is (no clean memory-bound staging target).
- SG2 CSA/HCA — `grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh`,
  `csa_attention_kernel` + `hca_attention_kernel`: the per-thread query vector
  `qv[0..head_dim)` is REUSED across every top-k entry and every window token —
  the reused global→shared load. New `csa_stage_query_async()` helper stages
  each thread's `head_dim`-float query slice into a private dynamic-shared slot
  (`block*head_dim` floats) via `cp.async.ca` split into `kCsaAsyncDepth`
  committed groups, then `wait_all`. Launches now pass the dynamic smem and call
  `set_attn_dyn_smem()` to opt into >48KB dynamic shared when needed. Streaming
  K/V stay direct (read once, no reuse → no staging benefit).

**How `SG_TUNED_ASYNC_DEPTH` is consumed:** it is the number of in-flight
cp.async groups / staging slots. In `fmha_softmax_kernel` it is the number of
row-tile groups primed and kept in flight (`cp_async_wait_group<depth-1>` per
consumed tile); in the CSA/HCA path it is the number of committed query-chunk
groups (`kCsaAsyncDepth`). Clamped to `[1,4]` at every use site. Two distinct
values produce different code (verified below).

**PTX / codegen survival (verified on this host, nvcc 12.0, `-arch=sm_90a`):**
- Tiny TU calling `cp_async_cg_16` / `cp_async_ca_4` / commit / wait emits, in
  the PTX: `cp.async.cg.shared.global [...], [...], 16;` (×1),
  `cp.async.ca.shared.global [...], [...], 4;` (×1), `cp.async.commit_group;`
  (×2), and `cp.async.wait_group 0;` / `cp.async.wait_group 1;`. None optimized
  away.
- Depth knob is live: compiling the staging-prime pattern with
  `-DSG_TUNED_ASYNC_DEPTH={1,2,4}` emits {1,2,4} `cp.async.commit_group`
  instructions and `cp.async.wait_group {0,1,3}` respectively — distinct SASS
  per depth, confirming the autotuner dimension is actually consumed.

**Hardware checks (deferred — all 🟡, no CUDA device in this env):**
- Async-copy + overlap confirmation:
  `ncu --metrics smsp__inst_executed_pipe_lsu,l1tex__data_pipe_lsu_wavefronts_mem_shared`
  on `fmha_softmax_kernel` and `csa/hca_attention_kernel` — expect cp.async
  (LDGSTS) issue on the LSU pipe and improved load/compute overlap (lower load
  stall) vs the pre-change synchronous-load build.
- Numeric parity: attention output (`P`/`csa_ctx`/`hca_ctx`) must be
  **bit-identical** (rtol=0, every element) to the pre-change synchronous-load
  path — a correctly waited cp.async lands byte-identical shared data. Run the
  same inputs through the Stage-3.2 build and a `-DSG_FORCE_SYNC_STAGING` (or
  pre-Ampere fallback) build and diff the output tensors.
- Depth sweep: build with `SG_TUNED_ASYNC_DEPTH ∈ {1,2,3,4}`; all must produce
  the identical output tensor (depth changes scheduling only, never numerics)
  while ncu LSU-overlap metrics shift — the autotuner picks the fastest depth.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 3.2 | fmha_softmax / csa+hca attn | cp.async LDGSTS + load/compute overlap | §Stage 3.2 ncu |
| 3.2 | fmha_softmax / csa+hca attn | output bit-parity vs sync-load path | §Stage 3.2 parity |
| 3.2 | all §4.2 kernels | invariant output across ASYNC_DEPTH 1–4 | §Stage 3.2 depth sweep |

## Stage 4.1 — TMA descriptor / operator reuse (NVIDIA memory §3.1)

**What changed (host-side, `csrc/backends/cuda/sm_90/mma.cuh`):** the generic
Sm90 collective GEMM launchers `sm90_run_gemm<>` and
`sm90_run_gemm_bt<,LayoutBT>` used to rebuild the CUTLASS `Gemm::Arguments`
(which materialises the Hopper TMA tensor-map descriptors via
`make_tma_copy_A/B_sm90` inside `to_underlying_arguments`), re-query the
workspace, and call `op.initialize()` on **every** invocation — even though
every optimizer step issues the **same** matmul shapes against the **same**
persistent buffers. They now route through a new shape-keyed host cache
`sm90_run_gemm_cached<Gemm,…>` that reuses the already-initialised operator.

**Exactly what is cached / reused:**
- The **initialised `GemmUniversalAdapter` operator** (its `params_` member,
  which holds the baked TMA descriptors for A, B and the C epilogue map), plus
  the workspace pointer it was initialised with. On a hit we skip
  `make_cute_packed_stride` ×3, `Gemm::Arguments` construction,
  `can_implement`, `get_workspace_size`, and `initialize` (the call that runs
  the `cuTensorMapEncode`s) and re-launch via the supported
  `op.run(stream)` overload — CUTLASS documents this overload as
  "re-launch the same kernel without updating internal params"
  (`gemm_universal_adapter.h:545`).
- **Not** reused / not reached into: CUTLASS's internal TMA mainloop. There is
  no public "swap the descriptor's base pointer" API in 3.6, so we do not
  fabricate one. The real, measurable win is skipping the per-step
  Arguments+initialize (the cuTensorMapEncode path), which is what this does.

**Cache key + eviction:**
- Key = `GemmCacheKey{ int M,N,K; const void* A; const void* B; void* C; }`.
  The key is **per-template-instantiation** (one static `Sm90GemmCache<Gemm>`
  each), so an FP16 vs BF16 GEMM, and a RowMajor-B vs ColumnMajor-B (Bᵀ) GEMM,
  never share a slot even at identical `(M,N,K,ptrs)`.
- Eviction: fixed-size **direct-mapped** table, `kSlots = 16`, slot =
  `FNV-1a(key) & (kSlots-1)`. A colliding signature evicts the resident entry
  and **`cudaFree`s its owned workspace** first → bounded memory, no leak. Only
  a handful of distinct signatures recur per run, so collisions are rare.
- Thread-safety: one `std::mutex` per `Sm90GemmCache<Gemm>`; the launcher is
  single-threaded per stream here but the lock makes concurrent host launchers
  safe and is off the hot path (host-side µs, dwarfed by the kernel).

**Descriptor address-stability analysis (the load-bearing correctness
invariant):** a Hopper TMA tensor map encodes the tensor's **global base
address** (CUTLASS feeds `make_tensor(ptr_A, layout)` into
`make_tma_copy_A_sm90`, `sm90_mma_tma_gmma_ss_warpspecialized.hpp:212-226`),
plus its box/shape and strides. Strides here are a pure function of `(M,N,K)`
(row-major packed). Therefore a cached operator is **bit-for-bit equivalent**
to a freshly rebuilt one **iff** `(M,N,K)` and the A/B/C base pointers match —
which is precisely the cache key. The buffers served (optimizer param/m/v
state, model weights, and the grow-once FP32 output scratch) are allocated
**once** and keep the same address every step ⇒ stable key ⇒ hit. If any buffer
is reallocated to a new address, the key changes ⇒ **miss ⇒ full rebuild**, so a
stale descriptor can never be launched against a moved buffer (that would be
wrong results / an illegal access). Workspace: for this builder (1×1×1 cluster,
`kGemm`, no split-K) `get_workspace_size()==0` and the adapter only adds a
barrier workspace when `cute::size(ClusterShape)>1`, so served GEMMs use
`ws==nullptr`; even so each entry **owns** its workspace allocation so a cached
operator's params can never dangle into the shared grow-only
`sm90_get_workspace` buffer.

**elect/mbarrier boundary note (honest):** §3.4's `elect_one_sync()` /
`Mbarrier` (`warp_specialize.cuh`, namespace `sg::sm90::wgs`) compose a
**hand-written** producer/consumer TMA staging loop. The CUTLASS GEMM is a
self-contained kernel that **owns** its TMA mainloop — there is no hand-written
TMA staging in `mma.cuh` into which a leader/barrier could be injected.
Pairing elect/mbarrier with TMA therefore belongs to the **Stage-6 megakernel's**
hand-written TMA path, **not** here. Stage 4.1 is the host-side
operator/descriptor cache only.

**Verification gate (this host, run today):**
- `compile_to_object.sh launch_supergrok2.cu -DWITH_CUTLASS` → `COMPILE_OK`
- `compile_to_object.sh launch_muon.cu -DWITH_CUTLASS`        → `COMPILE_OK`
- `compile_to_object.sh models/mamba.cu -DWITH_CUTLASS`       → `COMPILE_OK`
- `compile_to_object.sh models/decoder.cu -DWITH_CUTLASS`     → `COMPILE_OK`
- `python grokking_optimizers/compile.py --self-test` → `137 passed, 1 failed`
- `ruff check grokking_optimizers/` → `All checks passed!`

**Hardware checks (deferred — 🟡, no CUDA device in this env):**
- Fewer descriptor encodes / lower launch overhead per step:
  `ncu --metrics gpu__time_duration.sum --target-processes all` on a multi-step
  optimizer/model run, plus an API trace
  (`nsys profile --trace=cuda,nvtx`) — expect the per-step host-side
  `cuTensorMapEncodeTiled` calls (3 per GEMM: A, B, C) to drop to **zero after
  the first occurrence of each distinct shape** (only misses encode), and lower
  CPU launch overhead per step vs the rebuild-every-time build.
- Numeric parity: cached-operator GEMM output must be **bit-identical**
  (rtol=0, every element) to a `rebuild-every-step` build (revert
  `sm90_run_gemm*` to the inline initialize path, or force the cache to miss by
  perturbing the key). Same `args` ⇒ same `params_` ⇒ same descriptors ⇒ same
  launch, so equality is exact, not approximate. Drive identical inputs through
  the SG2 dt_proj GEMM, the decoder/ViT linear GEMMs, and the Muon
  Newton-Schulz GEMMs and diff the output tensors.
- Moved-buffer safety: re-run after forcing a buffer reallocation between steps
  (new base address) — the key must change and the rebuild must produce correct
  output (no illegal access from a stale TMA descriptor).

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 4.1 | sm90_run_gemm{,_bt} | cuTensorMapEncode count → 0 after warm-up; lower per-step launch overhead | §Stage 4.1 ncu/nsys |
| 4.1 | sm90_run_gemm{,_bt} | cached-op output bit-identical to rebuild-every-step | §Stage 4.1 parity |
| 4.1 | Sm90GemmCache | moved-buffer → key change → rebuild, correct output | §Stage 4.1 moved-buffer |

---

## Stage 4.2 — DSMEM cross-CTA reductions (NVIDIA memory features §3.2)

**What landed.** Replaced the `cluster_dsmem_reduce_sum` STUB (a bare warp
reduce that a comment admitted was a `cg::reduce` fallback) with a **real
Hopper thread-block-cluster DSMEM cross-CTA reduction**:
`sg::sm90::primitives::cluster_reduce_sum_f32_dsmem(val, cluster_smem_slot)` in
`csrc/backends/cuda/sm_90/primitives.cuh` (added as a NEW section after the
just-landed Stage 3.1 redux.sync and Stage 3.2 cp.async sections — neither was
touched). It is a **full multi-level tree** (§3.4) thread → warp → block(shared
tree) → cluster(DSMEM): each block publishes its block-reduced partial to its
OWN shared slot, then **every** block reads ALL peers' slots via
`cl.map_shared_rank` + `cl.sync()` and sums them (a full cluster-wide tree, not
a single-rank gather). Ordering is the critical part: **write own slot →
`cl.sync()` → read all peers → `cl.sync()`** (the second barrier prevents a
later reduction from clobbering the scratch mid-read). `cluster_size == 1` (the
common, no-cluster launch) short-circuits to the block reduce and **never
deadlocks** on a singleton-cluster barrier; pre-Hopper (`__CUDA_ARCH__ < 900`)
compiles to `block_reduce_sum_f32` with no cluster API referenced. The old
`cluster_dsmem_reduce_sum` in `csrc/common/utils.cuh` is **kept** as the
arch-portable warp-reduce fallback (signature unchanged — other sites depend on
it); cluster-aware sites route to the new helper.

**Toggle (§3.5).** `ENABLE_DSMEM_REDUCE` (default **0**). OFF ⇒ call sites use
the existing global-atomic reduction (today's behavior) and the DSMEM kernel is
never compiled-in at the call site — **zero overhead** at small scale. ON ⇒ the
host launcher selects the cluster kernel **only when the whole reduction grid
fits in one cluster** (`grid ≤ cluster_volume`), else it falls back to the
atomic kernel. Default OFF until the on-silicon checks below pass.

**Cluster-size knob (§3.3).** The launch cluster dimension is autotuner-pickable
via `SG_TUNED_CLUSTER_SHAPE` (already emitted by `compile.py`, volume capped
`m·n·p ≤ 8`). A latent bug was fixed: `-DSG_TUNED_CLUSTER_SHAPE=2,1,1` is
**rejected by nvcc** ("macro names must be identifiers" — the driver splits the
`-D` value on commas). `resolve_macros` now emits tuple dims as nvcc-safe scalar
macros `SG_TUNED_CLUSTER_SHAPE_{0,1,2}` + `SG_TUNED_CLUSTER_SHAPE_VOLUME`; the
C++ side reassembles them for `__cluster_dims__(…)`. **Megakernel caveat:** cap
the cluster at **≤ 2** inside any persistent megakernel context — large clusters
reduce the count of concurrently-resident cluster slots and can starve a
long-lived persistent grid.

**Sites wired vs deferred.**
- ✅ **Prodigy r/s sums** (`prodigy_reduce_kernel`): added
  `prodigy_reduce_dsmem_kernel` (`__cluster_dims__`), replacing the cross-block
  `atomicAdd(r_partial/s_partial)` with the DSMEM cluster tree; block 0 writes
  the final sum without an atomic. Wired into `launch_prodigy_step`.
- 🟡 **LookSAM ‖g‖** — `launch_looksam_norm_reduce` uses **ATen** `.norm()`
  (no custom CUDA reduction kernel), so there is no cross-CTA atomic to
  replace. Helper ready; no launch site to wire.
- 🟡 **Muon Frobenius norm** — `inv_norm` is computed **host-side** and passed
  into `muon_momentum_normalize_kernel` as a scalar; no device reduction kernel
  here. Helper ready; no launch site to wire.
- 🟡 **Attention softmax denom** (`fmha_softmax_kernel`) — **single-block per
  row** (`blockIdx.x == query row`); the row-sum is reduced entirely within one
  block, so there is no cross-CTA atomic. DSMEM not applicable as-is.
- 🟡 **Layernorm mean/var** (`transformer_decoder_sm90.cuh` /
  `vit_sm90.cuh`) — **single-block per row**; mean/variance are block-local
  warp+shared trees, no cross-CTA atomic. DSMEM not applicable as-is.
  Rationale: DSMEM only pays off where a reduction is **multi-block with an
  atomic accumulator that fits one cluster** — Prodigy is the only such site;
  the rest are single-block or host/ATen reductions. Correctness over coverage.

**Verification gate (this host, run today):**
- `compile_to_object.sh launch_prodigy.cu`  → `COMPILE_OK` (toggle off, and
  also verified `-DENABLE_DSMEM_REDUCE=1` and `…_1 -DSG_TUNED_CLUSTER_SHAPE_0=2`)
- `compile_to_object.sh launch_looksam.cu`  → `COMPILE_OK`
- `compile_to_object.sh launch_muon.cu`     → `COMPILE_OK`
- `compile_to_object.sh models/decoder.cu -DWITH_CUTLASS` → `COMPILE_OK`
- `compile_to_object.sh models/vit.cu -DWITH_CUTLASS`     → `COMPILE_OK`
- `compile_to_object.sh launch_adamw.cu`    → `COMPILE_OK` (header clean —
  primitives.cuh new section adds nothing to a non-cluster TU)
- standalone `__cluster_dims__(2,1,1)` TU calling
  `cluster_reduce_sum_f32_dsmem` → builds (`sm_90a`, nvcc 12.0)
- `python grokking_optimizers/compile.py --self-test` → `137 passed, 1 failed`
- `ruff check grokking_optimizers/` → `All checks passed!`

**Hardware checks (deferred — 🟡, no CUDA device in this env):**
- **DSMEM/cluster metrics:** with `ENABLE_DSMEM_REDUCE=1` and a Prodigy reduce
  whose grid fits one cluster, `ncu --metrics \
  sm__ctas_launched.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
  smsp__inst_executed_op_shared_ld.sum,launch__cluster_size` — expect the
  cross-block `atomicAdd` traffic to L2 (`lts__t_sectors_op_atom*`) to **drop
  to zero** and the cluster size metric to read the launched shape; the partial
  exchange shows up as shared (DSMEM) loads, not global atomics.
- **Bit-parity-vs-atomic (tolerance, NOT bit-exact):** drive identical
  `(param, param_init, grad, d_prev)` through both the atomic kernel and the
  DSMEM kernel and compare `r_sum`/`s_sum`. The DSMEM gather sums block partials
  in **rank order** in fp32 while the atomic path completes in nondeterministic
  order, so results match to **~1 ulp** (same reorder class as the existing
  warp/block shuffle trees), NOT bit-identically. Gate: relative error
  ≤ a few ulps; downstream `d` update unchanged within optimizer tolerance.
- **≤2-cluster-in-megakernel caveat:** when this helper is later composed into
  the Stage-6 persistent megakernel, verify occupancy with `ncu
  --metrics sm__maximum_active_clusters` and cap `SG_TUNED_CLUSTER_SHAPE` volume
  at ≤2 there — confirm the persistent grid does not lose resident blocks /
  stall on cluster-slot starvation at larger cluster volumes.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 4.2 | prodigy_reduce_dsmem_kernel | DSMEM shared exchange replaces L2 atomics; cluster size metric | §Stage 4.2 ncu DSMEM |
| 4.2 | cluster_reduce_sum_f32_dsmem | DSMEM sum vs atomic sum within ~1 ulp (tolerance, not bit-exact) | §Stage 4.2 bit-parity |
| 4.2 | SG_TUNED_CLUSTER_SHAPE | ≤2-cluster cap inside persistent megakernel (slot starvation) | §Stage 4.2 megakernel caveat |

---

## Stage 5 — mamba3 AMD-native

AMD-native hand-written AMDGCN forward+backward for the gfx942 (CDNA3 / MI300X)
Mamba-3 model kernel, replacing the scalar-labeled-as-MFMA reference with REAL
MFMA + DPP + LDS-handoff scan built on the shared, compiler-verified primitives
`csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp` (`namespace amd =
sg::gfx942::amdgcn`).

**File touched (only one):** `grokking_optimizers/kernels/gfx942/mamba3_gfx942.hip.hpp`

**What was rewritten (§5 device section, `__AMDGCN__ || __HIPCC__`):**
- **§5.1 in_proj / x_proj / out_proj — REAL MFMA matmul.** The reference's
  "scalar dot-product loop labeled MFMA" is now a true tiled bf16 matmul:
  `mfma_tile_16x16` / `mfma_matmul_bf16` issue `amd::mfma_bf16_16x16x16`
  (bf16x4 = `short[4]` operands, f32[4] accumulate) over 16×16 output tiles,
  one wavefront per tile, K contracted in 16-wide steps, with
  `amd::sched_group_barrier<MFMA,1>()`/`<VMEM,2>()` interleave (§2.11).
- **§5.2 / §5.6 RMSNorm fwd+bwd — DPP reductions.** The `__shfl_xor` butterfly
  is replaced by `amd::wave_reduce_add_dpp` (§2.6); `dweight` accumulates via
  `amd::atomic_add_agent_f32` (AGENT scope, §2.13).
- **§5.3 conv1d** depthwise k=3 + SiLU, **§5.5 gate multiply** — ported scalar.
- **§5.4 / §5.7 SSM selective scan fwd+bwd — workgroup-barrier LDS handoff.**
  Per-lane sequential segment + work-efficient Blelloch monoid scan across 64
  lanes; the reference's raw `__builtin_amdgcn_fence + s_barrier` is routed
  through `amd::workgroup_barrier_release/acquire` (§2.13). Paired-RoPE on
  B/C state dims retained.
- Read-once global loads use `amd::streaming_load` (§2.7); bf16 tensors flow as
  raw `short` bit-patterns through a self-contained bf16↔f32 codec (the
  free-standing gate has no `hip_bfloat16`).
- `__global__` entry kernels (`mamba3_gfx942_{in_proj_mfma,rmsnorm_fwd,
  conv1d_fwd,ssm_fwd<S>,gate_mul,rmsnorm_bwd,ssm_bwd<S>}`) with SEQ_LEN ∈
  {4,17,128} instantiated for the grokking shapes.

**Gate-caught issues fixed (this is what compiler verification bought):**
- **MFMA operand register type.** Reused `amd::mfma_bf16_16x16x16`, which takes
  `const short[4]` (bf16x4), NOT the reference's `uint32_t[4]` — the gate
  rejects u32x4 for the `_1k` builtins. Operand fragments packed as `short[4]`.
- **Compile-time-constant intrinsic args.** DPP-ctrl, sched-group mask/size and
  fp8 byte-index are routed through the templated primitives
  (`dpp_mov<CTRL>`, `sched_group_barrier<MASK,SIZE>`), never runtime values —
  the gate rejects runtime args.
- **Math builtins under the bare gate.** `rsqrtf`/`sincosf` are absent on the
  free-standing AMDGPU target; replaced with `__builtin_amdgcn_rsqf` and
  `__builtin_sinf`/`__builtin_cosf` (valid under hipcc too).
- **Launch builtins absent on the gate.** The free-standing target lacks
  `threadIdx/blockIdx/blockDim/gridDim`, `__global__`, `__shared__` (HIP
  runtime is stubbed). Added a gate-only shim (`__AMDGCN__ && !__HIPCC__`)
  modeling them with the AMDGCN workitem/workgroup ISA builtins so the device
  bodies type-check; on a real hipcc build the runtime supplies them and the
  shim is off.
- **Host/device split.** The ATen path's `#include "csrc/common/platform.h"`
  (→ `<cuda.h>`) cannot resolve on the AMDGPU target, so the ATen orchestration
  (A) is guarded `#if !defined(__AMDGCN__)` and the device kernels (B) under
  `#if defined(__AMDGCN__) || defined(__HIPCC__)` — one header, two passes.

**Launch wiring:** documented (§5.LAUNCH note in-file). The public entry points
`sg::gfx942::models::mamba::{forward,backward,selective_scan_fwd,
selective_scan_bwd}<T>` are UNCHANGED and still route to the proven ATen +
rocBLAS path (numerics-correct, MFMA via rocBLAS), so the bindings and the host
`mamba.hip.cpp` TU resolve byte-for-byte as before. Live `hipLaunchKernelGGL`
of the §5 kernels + hipcc link is **🟡 DEFERRED** (MI300X-gated: no hipcc / no
device in this env); the device kernels are compiler-verified and ready to wire
once the model TU migrates `.hip.cpp -> .hip` on hardware.

**Verification gate (this host, run today):**
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/mamba3_gfx942.hip.hpp`
  → `AMDGCN_OK` (device code compiles for gfx942)
- `bash scripts/amdgcn_check.sh --header csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp`
  → `AMDGCN_OK` (unchanged)
- `python grokking_optimizers/compile.py --self-test` → `137 passed, 1 failed`
  (the 1 failure is the pre-existing `flag_base_superset_regression` sm_90
  flag-baseline test, untouched by this change)
- `ruff check grokking_optimizers/` → `All checks passed!`
- sm_90 CUDA build UNAFFECTED — only the one gfx942 header changed.

**Hardware checks (deferred — 🟡, no MI300X / no hipcc in this env):**
- **MFMA utilization:** `rocprof --stats` / `rocprofv2` on the §5 kernels —
  expect `SQ_INSTS_VALU_MFMA*` (matrix-core instruction counts) non-zero for
  `mamba3_gfx942_in_proj_mfma` and the projection GEMMs, and the matrix unit
  busy fraction (`GRBM_GUI_ACTIVE` vs MFMA-issue) to confirm the 16×16×16 tiles
  keep the V_MFMA pipe fed (not VALU-bound); compare vs the rocBLAS path.
- **DPP / scan correctness:** drive identical (x, w, A, B, C, dt) through the
  §5 device kernels and the ATen path; compare RMSNorm, scan-fwd y_out, and the
  bwd grads to tolerance (DPP/atomic reduction reorder → ~1 ulp class, NOT
  bit-exact). Gate: relative error ≤ a few ulps end-to-end.
- **LDS handoff:** confirm the `workgroup_barrier_release/acquire` scan produces
  the same prefixes as the serial recurrence across SEQ_LEN ∈ {4,17,128}; check
  no LDS bank-conflict stalls (`SQ_LDS_BANK_CONFLICT`) in the Blelloch sweeps.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 5 | mamba3_gfx942_in_proj_mfma | MFMA-core instr counts non-zero; matrix unit busy fraction | §Stage 5 rocprof MFMA |
| 5 | ssm_fwd/ssm_bwd LDS handoff | Blelloch prefixes == serial recurrence; bank-conflict-free | §Stage 5 LDS handoff |
| 5 | rmsnorm_fwd/bwd DPP | DPP-reduced norm/grads vs ATen within ~1 ulp (tolerance) | §Stage 5 DPP correctness |

## Stage 5 — attention AMD-native

`grokking_optimizers/kernels/gfx942/attention_gfx942.hip.hpp` was rewritten to
the same TWO-pass / ONE-header structure proven on `mamba3_gfx942.hip.hpp`:

- **HOST pass** (`#if !defined(__AMDGCN__)`): the unchanged ATen + wave-64 LDS
  attention path, exposing the public
  `sg::gfx942::models::attention::{attention_forward,attention_backward}<ActT,
  kHeadDim,kCausal>` entry points the bindings call. The thin
  `csrc/backends/hip/gfx942/models/attention.hip.{h,cpp}` shim resolves unchanged.
- **DEVICE pass** (`#if defined(__AMDGCN__) || defined(__HIPCC__)`): a REAL
  hand-written AMDGCN forward + backward built on
  `csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp` (namespace `amd`). Carries
  the gate-only workitem/launch shim (`threadIdx/blockIdx/__global__/__shared__`
  via ISA builtins) under `#if defined(__AMDGCN__) && !defined(__HIPCC__)`, off
  under real hipcc — copied verbatim from the mamba3 exemplar.

**§5 device kernels (matrix-core attention):**
- `mfma_tile_16x16` / `mfma_matmul_bf16` — `S = Q·Kᵀ` and `O = P·V` as REAL
  16×16×16 bf16 MFMA (`amd::mfma_bf16_16x16x16`), one wavefront per 16×16 output
  tile, K-contraction in 16-wide bf16 steps, `amd::streaming_load` read-once VMEM,
  MFMA/VMEM interleave via `amd::sched_group_barrier<0x008,1>` / `<0x100,2>`.
- `attention_gfx942_fwd_mfma<32,kCausal>` — QKᵀ MFMA, scale + causal mask, row
  softmax (DPP), then PV MFMA (P and Vᵀ packed to bf16 in LDS).
- `attention_gfx942_bwd_mfma<32,kCausal>` — recompute A from saved `softmax_lse`,
  then `dV = Aᵀ·dO`, `dA = dO·Vᵀ`, the softmax jacobian, `dQ = dS·K`,
  `dK = dSᵀ·Q` — all five contractions through the MFMA tile path.
- Force-instantiated for the grokking configs `<32,true>` (decoder, causal,
  seq_len=4) and `<32,false>` (ViT, non-causal, seq_len=17).

**Softmax-max DPP approach:** the primitives header only ships
`wave_reduce_add_dpp` (SUM). The softmax row-MAX is implemented as
`wave_reduce_max_dpp` — the IDENTICAL row-shift + row-broadcast butterfly shape
with `fmaxf` substituted for `+`, using the same compile-time literal DPP
controls via `amd::dpp_mov<CTRL>` (`0x111/0x112/0x114/0x118` row_shr,
`0x142/0x143` row_bcast), broadcasting the top-lane result with `readlane`. The
row-sum then uses `amd::wave_reduce_add_dpp`.

**§2.10 LDS tile sizing (64 KB CDNA3 budget):** the sm_90 FMHA reference stages
whole K/V head tiles in 228 KB SMEM, which overflows the 64 KB CDNA3 LDS. The
tiling here is sized to the grokking shapes (D=32, N∈{4,17}, one wavefront per
(batch,head)): forward `scores[N×N] f32` (≤ 32·32·4 = 4096 B) + bf16 pack scratch
`Pbf[N×N] + Vtb[D×N]` (≤ (1024+1024)·2 = 4096 B) ⇒ ≤ ~8 KB; backward
`scores[N×N] + dA[N×N] f32` + bf16 pack ⇒ ≤ ~12 KB — all ≪ 64 KB, no K/V
re-tiling needed. A flash-style streaming tile (score buffer capped at N×Bc with
Bc=64 ⇒ 8 KB) is documented for the larger-N regime the grokking shapes don't reach.

**Gate-caught fixes (the device-compile gate did its job):**
1. `fmaxf` is undeclared under the free-standing gate (no `<cmath>`) → added a
   `dfmaxf` shim over `__builtin_fmaxf` and used it throughout the device section.
2. `__host__` is undefined under the bare gate → dropped it from the
   `attention_lds_bytes_fwd` launch-sizing helper (`__device__ __forceinline__`).
The two residual `'shared' attribute ignored` warnings are benign and identical
to the mamba3 exemplar (the gate's `__shared__` shim).

**Verification (this env):**
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/attention_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `python grokking_optimizers/compile.py --self-test` → `137 passed, 1 failed`
  (the 1 failure is the pre-existing `flag_base_superset_regression` sm_90
  flag-baseline test, untouched by this change)
- `ruff check grokking_optimizers/` → `All checks passed!`
- sm_90 CUDA build UNAFFECTED — only the one gfx942 attention header changed.

**Hardware checks (deferred — 🟡, no MI300X / no hipcc in this env):**
- **MFMA utilization:** `rocprof --stats` on `attention_gfx942_fwd_mfma` /
  `_bwd_mfma` — expect non-zero `SQ_INSTS_VALU_MFMA*` for the QKᵀ / PV / dV / dA /
  dQ / dK tiles and the matrix unit kept fed (16×16×16 MFMA not VALU-bound);
  compare vs the rocBLAS / scalar-LDS path.
- **DPP softmax correctness:** drive identical (Q,K,V) through the §5 kernels and
  the ATen LDS path; confirm `wave_reduce_max_dpp` row-max + `wave_reduce_add_dpp`
  row-sum reproduce the stable softmax to ~1 ulp (reduction reorder, NOT
  bit-exact) for both causal (N=4) and non-causal (N=17).
- **Numerics:** fwd `out` and bwd `grad_{q,k,v}` vs the ATen reference within a
  few ulps end-to-end; bf16 pack/unpack round-trip error within bf16 tolerance.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 5 | attention_gfx942_fwd_mfma | QKᵀ/PV MFMA-core instr counts non-zero; matrix unit fed | §Stage 5 attention MFMA |
| 5 | attention softmax DPP | DPP max+sum softmax vs ATen within ~1 ulp (causal + non-causal) | §Stage 5 attention DPP |
| 5 | attention_gfx942_bwd_mfma | dV/dA/dS/dQ/dK MFMA tiles + jacobian grads vs ATen | §Stage 5 attention numerics |

---

## Stage 5 — transformer_decoder AMD-native

`grokking_optimizers/kernels/gfx942/transformer_decoder_gfx942.hip.hpp` rewritten
from the ~102-line thin ATen wrapper into the proven two-pass, one-header form
(exact mamba3 scaffolding): HOST pass (`!__AMDGCN__`) keeps the unchanged ATen +
rocBLAS orchestration and the public `sg::gfx942::models::decoder::{forward,
backward}` entry points (decoder.hip.h shim + decoder.hip.cpp + bindings resolve
unchanged); DEVICE pass (`__AMDGCN__ || __HIPCC__`) is REAL hand-written AMDGCN
on `csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp`, with the gate-only
workitem/launch shim under `__AMDGCN__ && !__HIPCC__`.

**Device kernels (§5):**
- `decoder_gfx942_mfma_gemm` — the one 16×16×16 bf16 MFMA driver
  (`mfma_matmul_bf16` → `mfma_tile_16x16`) used for ALL heavy matmuls: QKV
  projection, attention output projection, and the FFN up (D→4D) / down (4D→D)
  matmuls. C[M,N]=A[M,K]·Wᵀ[N,K] (the [out,in] projection-weight layout), 64
  lanes feed the ISA fragment (4 bf16/lane/16-K step = short[4]), f32[4] acc,
  MFMA/VMEM `sched_group_barrier` interleave (§2.11), read-once `streaming_load`.
- `decoder_gfx942_attention` — fused per-(batch,head) S=QKᵀ·scale → softmax →
  O=PV with the S×S score tile staged in LDS (dynamic `extern __shared__`).
- `decoder_gfx942_layernorm_fwd` — LayerNorm mean & var.
- `decoder_gfx942_gelu` — tanh-approx GELU elementwise.
- `decoder_gfx942_residual_add` — f32-accum residual + bf16 store.

**GELU approach:** tanh approximation `0.5·x·(1 + tanh(k0·(x + k1·x³)))` with
`k0 = 0.7978845608028654` (√(2/π)), `k1 = 0.044715` — the SAME constants as the
sm_90 decoder's `gelu_tanh`, so HIP numerics track the reference. The bare gate
has no libm `tanhf`, so the tanh is clang's `__builtin_tanhf` (also valid under
hipcc; verified it compiles under the gate before committing to it).

**LayerNorm approach:** the two-pass clean form needs BOTH Σx and Σx², so it does
TWO `amd::wave_reduce_add_dpp` reductions per row (one over the running sum, one
over the running sum-of-squares) — the §2.6 row-shift + row-broadcast DPP
butterfly replacing a `__shfl` tree. `var = Σx²/n − (Σx/n)²`, then
`(x−mean)·rsqrt(var+eps)·γ + β` with `__builtin_amdgcn_rsqf`.

**LDS tile sizing (§2.10, 64 KB CDNA3 cap):** only the attention softmax uses
LDS — one S×S FP32 score tile, passed as a dynamic `extern __shared__` allocation
of `S*S*sizeof(float)`. At the grokking S_max=128 this is 128·128·4 = 65536 B =
exactly the 64 KB budget (`static_assert` guards it); for the common grokking S
(4, 17) it is far smaller. The MFMA-GEMM, LayerNorm, and GELU kernels keep their
accumulators in VGPRs (the 16×16 MFMA acc spread / DPP) and use NO LDS, so they
never contend for the budget.

**Gate-caught / structural fixes during bring-up:**
- The baseline (host-only) header FAILED the gate (`cuda.h` →
  `bits/libc-header-start.h not found`) under the free-standing amdgcn target —
  confirming the `#if !defined(__AMDGCN__)` host-include guard is mandatory; the
  two-pass split fixes it (`AMDGCN_OK`).
- bf16 MFMA operands are `short[4]` (bf16x4), NOT u32x4 — used `amd::
  mfma_bf16_16x16x16(float acc[4], const short[4], const short[4])` directly.
- `sched_group_barrier` mask/size, `dpp` ctrl args are compile-time constants —
  satisfied via the templated `amd::sched_group_barrier<MASK,SIZE>()` /
  `wave_reduce_add_dpp`.
- Pre-verified `__builtin_tanhf` / `__builtin_expf` / `__builtin_amdgcn_rsqf` /
  `__builtin_fmaxf` compile under the gate before wiring them into GELU/softmax.

**Verification (this env):**
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/transformer_decoder_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `PYTHONPATH=. python grokking_optimizers/compile.py --self-test` →
  `137 passed, 1 failed` (the 1 failure is the pre-existing sm_90 flag-baseline
  regression test, untouched by this change)
- `ruff check grokking_optimizers/` → `All checks passed!`
- sm_90 CUDA build UNAFFECTED — only the one gfx942 decoder header changed.

**Hardware checks (deferred — 🟡, no MI300X / no hipcc in this env):**
- **MFMA utilization:** `rocprof --stats` on `decoder_gfx942_mfma_gemm` — expect
  non-zero `SQ_INSTS_VALU_MFMA*` across the QKV / out-proj / FFN-up / FFN-down
  tiles and the matrix unit kept fed (16×16×16 MFMA, not VALU-bound); compare vs
  the rocBLAS path.
- **DPP LayerNorm correctness:** the two-DPP mean/var vs the ATen LayerNorm to
  ~1 ulp (reduction reorder, not bit-exact) across the grokking D.
- **GELU + softmax numerics:** the tanh-approx GELU and the attention softmax vs
  the ATen reference within a few ulps end-to-end; bf16 round-trip within bf16
  tolerance.
- **Launch wiring:** the §5.LAUNCH `hipLaunchKernelGGL` sequence (one pre-LN
  layer) becomes live once the model TU migrates `.hip.cpp → .hip` on hardware.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 5 | decoder_gfx942_mfma_gemm | QKV/out/FFN MFMA-core instr counts non-zero; matrix unit fed | §Stage 5 decoder MFMA |
| 5 | decoder layernorm DPP | two-DPP mean/var vs ATen LayerNorm within ~1 ulp | §Stage 5 decoder DPP |
| 5 | decoder GELU + attention | tanh-GELU + softmax vs ATen within a few ulps | §Stage 5 decoder numerics |

---

## Stage 5 — vit AMD-native

**File:** `grokking_optimizers/kernels/gfx942/vit_gfx942.hip.hpp` (rewritten
~113-line thin-ATen header → two-pass HOST/DEVICE, ~560 lines). Same canonical
two-pass scaffolding proven on `mamba3_gfx942` / `transformer_decoder_gfx942`:
- **HOST pass** (`#if !defined(__AMDGCN__)`): unchanged ATen + rocBLAS path. The
  public `sg::gfx942::models::vit::{forward,backward,patch_project}` entry points
  are byte-for-byte intact, so `models/vit.hip.{h,cpp}` and the `bindings.cpp`
  `vit_forward`/`vit_backward` dispatch resolve unchanged.
- **DEVICE pass** (`#if defined(__AMDGCN__) || defined(__HIPCC__) ||
  GROK_HIP_DEVICE`): real hand-written AMDGCN ViT forward on
  `amdgcn_primitives.hip.hpp` (`namespace amd = sg::gfx942::amdgcn`), with the
  gate-only workitem/launch shim copied exactly under
  `#if defined(__AMDGCN__) && !defined(__HIPCC__)`.

**What the §5 device kernels implement (all matmuls via 16×16×16 bf16 MFMA):**
- `vit_gfx942_matmul_bias` — the single MFMA GEMM (`C[M,N]=A[M,K]·Wᵀ[N,K]+bias`)
  that the patch-embedding projection, QKV, attention out-proj, MLP up/down, and
  the classification head all route through. 16-wide bf16 K-steps, f32[4] acc,
  `amd::sched_group_barrier<0x008,1>` (MFMA) interleaved with `<0x100,2>` (VMEM).
- `vit_gfx942_attention` — per-(batch,head) scaled dot-product attention. S=QKᵀ
  per-lane partial dot reduced with `amd::wave_reduce_add_dpp`; numerically
  stable **online softmax** (running max/sum with exp-rescale); O=PV accumulated
  per lane. K/V staged once in LDS.
- `vit_gfx942_layernorm_fwd` — pre-norm LayerNorm (mean AND var), both via
  `amd::wave_reduce_add_dpp` (two reductions), `__builtin_amdgcn_rsqf` for invstd.
- `vit_gfx942_gelu` — element-wise tanh-approx GELU.

**Softmax row-MAX approach:** built a DPP **max butterfly** with the SAME shape
as `amd::wave_reduce_add_dpp` but `__builtin_fmaxf` instead of `+`
(`wave_reduce_max_dpp`): `dpp_mov<0x111/0x112/0x114/0x118>` (row_shr 1/2/4/8)
then `dpp_mov<0x142/0x143>` (row_bcast 15/31), folded with fmaxf, then
`readlane(63)` broadcasts the wavefront max. The attention kernel uses the
online-softmax variant (per-key running max via `fmaxf` + exp-rescale of the
running sum and the O accumulator), which is the same max algebra applied
incrementally; the standalone `wave_reduce_max_dpp` is provided for the
batched/two-pass path.

**GELU approach:** tanh approximation matching PyTorch `nn.GELU` default and the
sm_90 ViT: `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))` via `__builtin_tanhf`
(gate-verified), with `√(2/π)` folded as a literal.

**LDS tile sizing (§2.10, CDNA3 64 KB):** only the attention kernel uses LDS —
it stages this head's K and V strips plus is bounded by `kAttnMaxS=240`,
`kAttnMaxHeadDim=64`: `2·240·64·2 B + 240·4 B = 62 400 B < 65 536 B`, enforced by
a `static_assert`. The matmul / LayerNorm / GELU kernels are LDS-free (DPP
reductions + register accumulators, 0 LDS bytes). Longer sequences stream K/V in
≤240-row strips.

**Gate-caught fixes during bring-up:**
- My first LDS budget (`kAttnMaxS=256`) FAILED the `static_assert`
  (`66560 <= 65536`) — the gate evaluated the constexpr and rejected it; dropped
  to `kAttnMaxS=240` (62 400 B) to fit the 64 KB CDNA3 LDS.
- `size_t` is unavailable under the free-standing amdgcn gate (no `<cstddef>`) —
  the GELU element index now uses `unsigned long` instead.
- Pre-verified `__builtin_tanhf` / `__builtin_expf` / `__builtin_amdgcn_rsqf` /
  `__builtin_fmaxf` and the bf16x4 `short[4]` MFMA operand type via the gate
  before wiring them in.

**Verification (this env):**
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/vit_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `PYTHONPATH=. python grokking_optimizers/compile.py --self-test` →
  `137 passed, 1 failed` (the 1 failure is the pre-existing sm_90 flag-baseline
  regression test, untouched by this change)
- `ruff check grokking_optimizers/` → `All checks passed!`
- sm_90 CUDA build UNAFFECTED — only the one gfx942 vit header changed.

**Hardware checks (deferred — 🟡, no MI300X / no hipcc in this env):**
- **MFMA utilization:** `rocprof --stats` on `vit_gfx942_matmul_bias` — expect
  non-zero `SQ_INSTS_VALU_MFMA*` across patch-embed / QKV / out-proj / FFN /
  head tiles and the matrix unit kept fed (16×16×16 MFMA, not VALU-bound).
- **Attention softmax numerics:** the DPP-max + online-softmax vs the ATen
  reference within a few ulps end-to-end; bf16 round-trip within bf16 tolerance.
- **DPP LayerNorm correctness:** the two-DPP mean/var vs ATen LayerNorm to ~1 ulp.
- **Launch wiring:** the §5.LAUNCH `hipLaunchKernelGGL` sequence (one pre-LN
  encoder layer) becomes live once the model TU migrates `.hip.cpp → .hip`.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 5 | vit_gfx942_matmul_bias | patch/QKV/out/FFN/head MFMA-core instr counts non-zero; matrix unit fed | §Stage 5 vit MFMA |
| 5 | vit_gfx942_attention | DPP-max + online-softmax vs ATen within a few ulps | §Stage 5 vit softmax |
| 5 | vit layernorm DPP | two-DPP mean/var vs ATen LayerNorm within ~1 ulp | §Stage 5 vit DPP |

---

## Stage 5 — reduction-bearing optimizers AMD-native (looksam/muon/prodigy/sg11/sg15) 🟡

Five reduction-bearing gfx942 optimizer kernels now carry a REAL hand-written
AMDGCN reduction (section (B), `#if defined(__AMDGCN__) || defined(__HIPCC__)`)
alongside the UNCHANGED ATen host orchestration (section (A),
`#if !defined(__AMDGCN__)`). Each follows the proven mamba3/attention two-pass
single-header pattern: the host launchers + public entry points resolve
byte-for-byte for the thin `.hip.cpp` TUs, and the device pass compiles the
hand-written reductions built on
`csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp` (`namespace amd`). The
gate-only workitem/launch shim (`threadIdx/blockIdx/__global__/__shared__` via
the AMDGCN ISA builtins under `#if defined(__AMDGCN__) && !defined(__HIPCC__)`)
is copied verbatim from the exemplar.

Every reduction is the canonical 2-level **wave → block → grid** tree from the
task spec: each thread grid-strides its partials → `amd::wave_reduce_add_dpp`
(DPP wavefront sum, replacing `__shfl`/`__shfl_xor` butterflies) → lane 0 of
each wavefront writes to an LDS slot → `workgroup_barrier_release` → the first
wavefront reduces the slots with a second DPP reduce → one thread does
`amd::atomic_add_agent_f32` to the global accumulator (§2.13 AGENT scope — AMD
has no DSMEM, so cross-workgroup accumulation uses an XCD-visible global atomic).

**Per-file reduction kernels (the high-value AMDGCN piece; APPLY stays ATen):**
- **looksam** — `looksam_gfx942_sumsq_reduce`: the SAM gradient L2 norm
  ‖g‖ = sqrt(Σ gᵢ²) as a global sum-of-squares; host does the final
  sqrt/rsqrt for the 1/‖g‖ perturbation scale. 1 AGENT atomic. (Reference
  exemplar — already in place; re-verified.)
- **muon** — `muon_gfx942_frobenius_reduce`: the momentum-buffer Frobenius norm
  ‖M‖_F = sqrt(Σ_ij M_ij²) that normalizes the Newton-Schulz iterate (flat sum
  over numel); host does sqrt + 1/‖M‖_F. 1 AGENT atomic. Newton-Schulz GEMMs
  stay on rocBLAS MFMA.
- **prodigy** — `prodigy_gfx942_rs_reduce`: the r-sum and s-sum global
  reductions TOGETHER in one pass (mirrors the sm_90 `prodigy_reduce_kernel`):
  r = Σ gᵢ·(p_initᵢ − pᵢ)·d_prev, s = Σ d_prev²·|gᵢ|; two DPP reduces, two LDS
  trees, **2 AGENT atomics**. Host forms d_new = max(d_prev, r/(|s|+1e-12)).
- **supergrok11** — `supergrok11_gfx942_cosine_reduce`: the cosine-gate
  3-quantity reduction in one pass: num = Σ g·m, den_g = Σ g², den_m = Σ m²;
  three DPP reduces, three LDS trees, **3 AGENT atomics**. Host forms
  gate = clamp(num / sqrt(den_g·den_m + 1e-12), 0, 1).
- **supergrok15** — `supergrok15_gfx942_sharpness_reduce`: the sharpness
  reduction Σ_i sharpnessᵢ (global gate signal) as a global sum. 1 AGENT atomic.
  Host forms the sharpness-driven gate scale (e.g. mean = Σ/n).

**APPLY decision:** for ALL five, the per-element optimizer update (the
Adam/Lion/SAM/Newton-Schulz apply after the reduction) stays on the proven ATen
host path; only the reduction — the high-value AMDGCN piece — is migrated.

**Gate-caught fixes during bring-up:**
- The bare gate has no `<cmath>`, so `fabsf` is unavailable in the device pass;
  prodigy's |gᵢ| uses `__builtin_fabsf` (valid under hipcc too). Verified via the
  gate before wiring it in.
- The host-pass guard `#if !defined(__AMDGCN__)` around the ATen/torch block is
  mandatory — `torch/extension.h` pulls in `<cuda.h>`/ATen, which the
  free-standing amdgcn target cannot resolve (matches the looksam exemplar).
- DPP-ctrl/sched-mask are compile-time-constant templates (inherited from
  `amd::wave_reduce_add_dpp`); the multi-quantity reductions reuse the verified
  primitive rather than open-coding a second butterfly.
- All AGENT atomics route through `amd::atomic_add_agent_f32` (the gate maps
  `__hip_atomic_fetch_add` + `__HIP_MEMORY_SCOPE_AGENT` to the host `__atomic_*`
  stub for the free-standing compile).

**Verification (this env):**
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/looksam_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/muon_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/prodigy_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/supergrok11_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/supergrok15_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `PYTHONPATH=. python grokking_optimizers/compile.py --self-test` →
  `137 passed, 1 failed` (the 1 failure is the pre-existing sm_90 flag-baseline
  regression test, untouched by this change)
- `ruff check grokking_optimizers/` → `All checks passed!`
- sm_90 CUDA build UNAFFECTED — only the five gfx942 optimizer headers changed.

**Hardware checks (deferred — 🟡, no MI300X / no hipcc in this env):**
- **MI300X numerics:** each reduction (‖g‖, ‖M‖_F, r/s, num/den_g/den_m,
  Σ sharpness) vs the ATen reference within a few ulps end-to-end.
- **wave→block→AGENT bit-parity:** the DPP wave reduce → LDS block tree →
  AGENT-atomic grid sum reproduces the rocPRIM segmented-reduction result
  bit-for-bit (modulo the non-associative-atomic ordering tolerance).
- **Launch wiring:** the §5.LAUNCH `hipLaunchKernelGGL` sequence becomes live
  once each optimizer TU migrates `.hip.cpp → .hip`.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 5 | looksam_gfx942_sumsq_reduce | ‖g‖ sum-of-squares vs ATen `.norm()` within a few ulps | §Stage 5 reduction optimizers |
| 5 | muon_gfx942_frobenius_reduce | ‖M‖_F vs ATen `.norm()` within a few ulps | §Stage 5 reduction optimizers |
| 5 | prodigy_gfx942_rs_reduce | r/s dual sum vs sm_90 prodigy_reduce_kernel bit-parity | §Stage 5 reduction optimizers |
| 5 | supergrok11_gfx942_cosine_reduce | num/den_g/den_m triple sum vs ATen within a few ulps | §Stage 5 reduction optimizers |
| 5 | supergrok15_gfx942_sharpness_reduce | Σ sharpness vs ATen `.sum()` within a few ulps | §Stage 5 reduction optimizers |

---

## Stage 5 — elementwise optimizers AMD-native (adamw/lion/grokfast/grokadamw/neuralgrok)

🟡 **HARDWARE-GATED.** The five pure-elementwise optimizer headers now carry a
real hand-written AMDGCN grid-stride update kernel (§5, section B) alongside the
unchanged ATen host orchestration (section A). Each is a per-parameter update
(read grad + state, compute new param + state, write back) — **no reductions, no
cross-lane MFMA** — so the AMD-native deliverable is a streaming (read-once)
grid-stride `__global__` kernel built on
`csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp` (namespace `amd`):

- **Grid-stride** over `numel`, one stride of elements per workitem (§2.1).
- **Streaming loads** (§2.7): the grad is read once via `amd::streaming_load`
  (nontemporal `__builtin_nontemporal_load` — bypasses L2 for one-touch data);
  the param write-back uses `amd::streaming_store`.
- **Lean recompute** (§2.9): trivial here — the whole step lives in registers,
  state EMAs are read/written exactly once, no recompute needed.

**Per-element MATH — inlined (option a).** The vendor-neutral device step
functions in `csrc/algorithms/<opt>.h` (`adamw_step`, `lion_step`,
`grokfast_fused_step`, `grokadamw_step`, `neuralgrok_psi_forward` +
`neuralgrok_apply_step`) include `csrc/common/{types.h,utils.cuh}`, which pull
torch/cuda and cannot resolve under the bare amdgcn gate. So the tiny per-element
arithmetic is copied **verbatim** into each device kernel (numerically identical
to the algorithm header), keeping the bindings + `.hip.cpp` TUs resolving
unchanged via section A. libm calls are mapped to clang builtins under the gate:
`sqrtf→__builtin_sqrtf`, `copysignf→__builtin_copysignf`, `fabsf→__builtin_fabsf`.

**Kernels written (all `sg::gfx942::native`, fp32 param + fp32 grad instantiated):**
- `adamw_gfx942_kernel<ParamT,GradT>` — m/v EMAs + bias-corrected decoupled-WD
  apply, fused (vs 3 ATen launches).
- `lion_gfx942_kernel<ParamT,GradT>` — sign-momentum interp + decoupled-WD apply
  + momentum refresh, fused (sign via `__builtin_copysignf`, zero-interp → 0).
- `grokfast_gfx942_kernel<ParamT,GradT>` — slow-gradient EMA filter +
  amplification (`g_amp = g + lamb·ema`) + Adam apply, fused.
- `grokadamw_gfx942_kernel<ParamT,GradT>` — EMA filter + amplification + Adam
  apply, fused.
- `neuralgrok_gfx942_kernel<ParamT,GradT,H>` — per-element psi-net amplifier MLP
  (`relu(W1·|g|+b1)·W2 + b2`, H-wide, unrolled; layer-1 input is a per-element
  scalar so it is pure SIMD — no MFMA tile applies) → `g_amp = (s·alpha+beta)·g`
  → Adam apply, fused. Instantiated at `H ∈ {16,32}` (`neuralgrok_psi<H>` helper).

**Compile routing (two passes, one header):** the HOST `#if !defined(__AMDGCN__)`
guard keeps the existing ATen path + public launcher signatures byte-for-byte
(torch pulls `<cuda.h>`, invisible to the bare gate); the DEVICE
`#if defined(__AMDGCN__) || defined(__HIPCC__)` block carries section B with the
gate-only workitem/launch shim copied verbatim from the looksam/mamba3 exemplar.
Each §5.LAUNCH note documents the deferred `hipLaunchKernelGGL` wiring.

**Gate-caught fixes:** the lion/grokfast/grokadamw/neuralgrok headers had no host
guard (the bare `#include <torch/extension.h>` + namespace were unguarded) — wrapped
the entire ATen block in `#if !defined(__AMDGCN__)` so torch is invisible to the
device pass; libm `sqrtf`/`copysignf`/`fabsf` swapped to the `__builtin_*` forms the
bare gate resolves. No DPP/sched constant-arg issues (no reductions here). adamw was
already in the pattern and re-verified.

**Verification (this env):**
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/adamw_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/lion_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/grokfast_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/grokadamw_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `bash scripts/amdgcn_check.sh --header grokking_optimizers/kernels/gfx942/neuralgrok_gfx942.hip.hpp`
  → `AMDGCN_OK`
- `PYTHONPATH=. python grokking_optimizers/compile.py --self-test` →
  `137 passed, 1 failed` (the 1 failure is the pre-existing sm_90 flag-baseline
  regression test, untouched by this change)
- `ruff check grokking_optimizers/` → `All checks passed!`
- sm_90 CUDA build UNAFFECTED — only the five gfx942 elementwise optimizer
  headers changed; the section-A public launchers are byte-for-byte unchanged.

**Hardware checks (deferred — 🟡, no MI300X / no hipcc in this env):**
- **MI300X numeric parity:** each fused kernel (adamw/lion/grokfast/grokadamw/
  neuralgrok) vs its `csrc/algorithms/<opt>.h` device-step reference within a few
  ulps end-to-end (bit-identical math, modulo fp contraction/FMA ordering).
- **Launch wiring:** the §5.LAUNCH `hipLaunchKernelGGL` sequence becomes live once
  each optimizer TU migrates `.hip.cpp → .hip`.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 5 | adamw_gfx942_kernel | fused m/v + apply vs `adamw_step` reference within a few ulps | §Stage 5 elementwise optimizers |
| 5 | lion_gfx942_kernel | sign-momentum step vs `lion_step` reference within a few ulps | §Stage 5 elementwise optimizers |
| 5 | grokfast_gfx942_kernel | EMA-amplified Adam vs `grokfast_fused_step` reference | §Stage 5 elementwise optimizers |
| 5 | grokadamw_gfx942_kernel | EMA-amplified Adam vs `grokadamw_step` reference | §Stage 5 elementwise optimizers |
| 5 | neuralgrok_gfx942_kernel | psi-MLP amplifier + Adam vs `neuralgrok_{psi_forward,apply_step}` | §Stage 5 elementwise optimizers |
