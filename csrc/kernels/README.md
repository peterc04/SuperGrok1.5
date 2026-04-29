# csrc/kernels/ — Specialized Kernel Tree

Per the all-specialized refactor (see `REFACTOR_PLAN.md`), each optimizer
has one fully-specialized kernel file per supported arch:

```
cuda/sm_80/<optimizer>_sm80.cu       # Ampere
cuda/sm_90/<optimizer>_sm90.cu       # Hopper (highest-value target)
cuda/sm_100/<optimizer>_sm100.cu     # Blackwell
hip/gfx942/<optimizer>_gfx942.hip.cpp # AMD CDNA3 (MI300X)
```

Each file is wrapped in `namespace sg::<arch> { ... }` so the four
translation units do not collide on kernel/launcher symbols.

## Overlay files (`*_overlay.cu`)

A handful of pre-existing arch-tuned files predate the all-specialized
refactor. They are partial specializations (cp.async paths on Ampere,
Hopper FP8 / warp-specialized paths, Blackwell TMA scaffolding, gfx942
BF16 MFMA) that used to overlay on top of the generic kernels in
`csrc/cuda/generic/`.

After the structural refactor, these overlay files are renamed
`*_overlay.cu` (or `*_overlay.hip.cpp`) and **excluded from the build**
until they are merged into the canonical per-arch kernels in a
hardware-validated tuning pass.

The overlays are kept in-tree (rather than deleted) because they
contain real arch-specific work that informs the future hand-tuned
divergence — once a future GPU-equipped session merges the overlay
into the canonical `<optimizer>_<arch>.cu`, the overlay file is
deleted.

## Cross-arch numerical agreement

Math is identical across the four arch variants of any given
optimizer. The cross-arch numerical agreement test
(`tests/test_cross_arch_agreement.py`) runs each available arch on the
same inputs and asserts elementwise agreement within FP tolerance.
Hand-tuned arch-specific divergence (cp.async double-buffering, TMA
descriptor setup, MFMA assignment, FP8 paths, warp specialization) may
change throughput but must not change the numerical output.

## Adding a new specialized kernel

1. Hand-write the sm_90 version first (highest-value target).
2. Benchmark with `benchmarks/benchmark_supergrok2.py`.
3. Port to sm_80, sm_100, gfx942 one at a time. Keep the math
   identical; only swap arch-specific primitives.
4. Re-run autotune (`autotune/tune.py`) to refresh
   `csrc/common/tuned_configs.h`.
5. Cross-arch agreement test is mandatory.

## CPU and TPU

`cpu/` is preserved for testing only — not a runtime fallback.
`tpu/v5p/` and `tpu/v6e/` host the JAX/Pallas kernels split per TPU
generation; framework-side detection picks the right Pallas kernel
from `supergrok2_jax_tpu/sharding.py`.
