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
