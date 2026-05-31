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

---

### Deferred-check ledger (every runtime check the build could not run on CPU)
Each stage appends one line per deferred check: `STAGE | cell | what to verify | command`.

| stage | cell | deferred check | command ref |
|-------|------|----------------|-------------|
| 0 | all | device link + SASS sanity | §2 Stage 0 |
