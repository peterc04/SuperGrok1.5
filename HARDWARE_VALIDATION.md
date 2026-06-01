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
| supergrok2   | 🟡 | 🟡 | 🟡 | CSA/HCA fwd; **bilevel backward** = Stage 1A |

## 2. Per-stage hardware checks (appended as stages complete)

### Stage 0 — compile correctness
- [ ] `FORCE_CUDA=1 WITH_CUTLASS=1 pip install -e .` links a loadable `_C`
      extension on an H100 box (the CPU-host `nvcc -c` gate cannot catch device
      link / ptxas-lowering errors).
- [ ] `cuobjdump -sass build/.../launch_adamw.o | head` shows real SASS.

<!-- Stage 1+ checks are appended below by each stage. -->

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
