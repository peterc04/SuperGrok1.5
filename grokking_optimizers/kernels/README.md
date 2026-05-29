# Per-Arch Kernel Reference Headers

This directory contains **per-architecture reference/specification kernels** for
all 7 optimizers and 3 model architectures across the 3-arch active set:

| Subdirectory | Architecture | Extension | Files |
|---|---|---|---|
| `sm_90/` | NVIDIA Hopper (H100/H200) | `.cuh` | 16 |
| `gfx942/` | AMD CDNA3 (MI300X/MI300A) | `.hip.hpp` | 16 |
| `tpu/` | Google TPU v5p (Pallas/JAX) | `.py` | 12 |

## Relationship to Production Kernels

**These headers are NOT compiled by the build system.** The production pipeline is:

```
csrc/algorithms/*.h          (vendor-neutral elementwise math)
    |
    v
csrc/backends/cuda/sm_90/launch_*.cu     (CUDA launch glue, namespace sg::sm90)
csrc/backends/hip/gfx942/launch_*.hip.cpp (HIP launch glue,  namespace sg::gfx942)
    |
    v
csrc/bindings/helpers.h      (SG_DISPATCH macro -> runtime arch dispatch)
    |
    v
grokking_optimizers/_ops      (pybind11 extension module)
```

These reference headers use `namespace grokking::sm90` / `grokking::gfx942` (not
the `sg::` namespace expected by `SG_DISPATCH`). They serve as:

1. **Arch-specific algorithm specifications** with NanPolicy, gradient clipping,
   and prefetch-pipelined variants that the vendor-neutral `csrc/algorithms/*.h`
   does not express.
2. **Codegen self-test targets** for `compile.py`'s autotuner and synth-GEMM
   emitter.
3. **Migration templates** for converting `csrc/backends/` launch files from
   ATen-based dispatch to hand-written `__global__` kernels.

## Kernel Header Contract

Every optimizer header follows this pattern:

### GPU (sm_90 / gfx942)

- Include guard: `#ifndef GROKKING_<OPT>_<ARCH>_{CUH,HIP_HPP}_`
- Namespace: `grokking::{sm90,gfx942}`
- `<Opt>State` struct with `num_state_tensors()` and `state_bytes_per_element()`
- Template parameters: `<typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>`
- Device functions: `<opt>_update` (scalar) + `<opt>_update_vec4` (vectorized)
- Prefetch variant: `<opt>_update_prefetch` (software-pipelined, 2 register sets)
- Global kernels: `<opt>_kernel` + `<opt>_kernel_prefetch`
- FP32 accumulators throughout; param type is templated

### TPU (Pallas/JAX)

- Uses `common_tpu.py`: `NanPolicy` enum, `apply_nan_policy()`, `to_bf16()`/`to_f32()`
- `PARAM_DTYPE = jnp.bfloat16`, `ACCUM_DTYPE = jnp.float32`
- Step functions accept JAX arrays and return updated state

## Optimizer Coverage

| Optimizer | sm_90 | gfx942 | TPU | Notes |
|---|---|---|---|---|
| AdamW | ✓ | ✓ | — | + prefetch retrofit |
| Lion | ✓ | ✓ | — | + prefetch retrofit |
| Grokfast | ✓ | ✓ | — | + prefetch retrofit |
| GrokAdamW | ✓ | ✓ | — | + prefetch retrofit |
| LookSAM | ✓ | ✓ | ✓ | New |
| Prodigy | ✓ | ✓ | ✓ | New |
| NeuralGrok | ✓ | ✓ | ✓ | New |
| SuperGrok 1.1 | ✓ | ✓ | ✓ | New |
| SuperGrok 1.5 | ✓ | ✓ | ✓ | New |
| Muon | ✓ | ✓ | ✓ | New (Newton-Schulz) |
| SuperGrok v2 | ✓ | ✓ | ✓ | CSA/HCA hybrid attention + PEER + GRU |

### SuperGrok v2 meta-model (CSA/HCA)

SuperGrok v2 is a learned optimizer: it views a parameter tensor's flattened
elements as a **sequence** and runs a small meta-net over it to produce a smart
gradient, then applies an Adam-style update. The sequence mixer is a
**DeepSeek-V4-style CSA/HCA hybrid attention** stack (replacing the previous
Mamba-3 bidirectional selective scan). Per spec SS3b, only the mixer changed --
the GRU + PEER routing + expert MLP + AdamW apply tail is kept verbatim.

| Stage | What it does |
|---|---|
| input proj | `[grad, sharpness] -> d_model` per element |
| **CSA** | Compressed Sparse Attention: KV compress `m=4` (learned pool) + lightning-indexer **top-k** selection (low-rank rank=4) + sliding window. Produces `csa_ctx` (fine/local context; was the mamba fwd scan). |
| **HCA** | Heavily Compressed Attention: heavy KV compress `m'=128` (mean pool) + **dense** attention over all compressed entries + sliding window. Produces `hca_ctx` (global coarse context; was the mamba bwd scan). |
| GRU | per-element temporal memory (input: `g, s, csa_ctx, hca_ctx`) — unchanged |
| PEER | 4-head product-key expert routing — unchanged |
| expert MLP | top-routed per-element expert (FP32 / INT8 / INT4 tiers) — unchanged |
| AdamW apply | smart-gradient blended AdamW step — unchanged |

Multi-query KV is shared across attention heads; all accumulation is FP32.
Attention is **stateless across optimizer steps**, so (unlike the carried Mamba
fwd/bwd state) only the GRU state persists — the reference `*State` structs drop
the two mamba-state tensors (`num_state_tensors()` 7 -> 5). Per-arch reference
files: `sm_90/supergrok2_sm90.cuh` (CSA/HCA via WGMMA-routed GEMMs in
production), `gfx942/supergrok2_gfx942.hip.hpp` (rocBLAS/ATen GEMMs, 64-wide
wavefront), `tpu/supergrok2_tpu.py` (reshape/mean compression, `jax.lax.top_k`
indexer, `jnp.einsum` + `jax.nn.softmax` attention).

## Model Coverage

| Model | sm_90 | gfx942 | TPU |
|---|---|---|---|
| Transformer Decoder | ✓ | ✓ | ✓ |
| ViT | ✓ | ✓ | ✓ |
| Mamba-3 | ✓ | ✓ | ✓ |

## Wiring to Production (Future)

To promote a reference header to production:

1. Change namespace from `grokking::<arch>` to `sg::<arch>`
2. Add the source file to `setup.py` source globs
3. Add `grokking_optimizers/kernels/<arch>` to `include_dirs`
4. Have the `csrc/backends/.../launch_*` files `#include` the header
5. Verify with `cuobjdump -sass` / `roc-obj` that expected instructions emit
