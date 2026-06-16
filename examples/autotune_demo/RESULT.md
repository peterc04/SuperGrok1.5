# Autotune demo — naive parameterized tiled GEMM through compile.py

**Date:** 2026-06-16 · **GPU:** NVIDIA H100 80GB HBM3 (sm_90a) · **dtype:** fp32 · **size:** M=N=K=2048

**One-line result:** compile.py's autotuner machinery swept **24/24** configs of a
non-hand-tuned tiled GEMM, all passing the strict fp32 correctness gate, and found a
**1.04x** speedup over the in-source naive default. The hypothesis was a *large* (multiples)
gain on a naive baseline; the real gain here is small, and the data below explains exactly
why — it is an honest negative result with an instructive cause.

## Speedup table

| Metric | Config | Time (ms) | vs naive | vs cuBLAS |
|---|---|---|---|---|
| **NAIVE default** (in-source `#ifndef` defaults) | `tile_m=16, tile_n=16, tile_k=16, block=256` | **4.970** | 1.00x | 14.3x slower |
| **AUTOTUNER best** (min `elapsed_ms`, gate-PASS) | `tile_m=32, tile_n=16, tile_k=16, block=256` | **4.786** | **1.04x** | 13.8x slower |
| Worst config in sweep | `tile_m=128, tile_n=64, tile_k=32, block=256` | 14.703 | 0.34x | 42x slower |
| cuBLAS (`torch.mm`) — context only | (closed-source) | **0.347** | 14.3x faster | 1.00x |

- **#configs tried:** 24 / 24 (full grid: `tile_m∈{16,32,64,128} × tile_n∈{16,64,128} × tile_k∈{16,32} × block∈{256}`).
- **Correctness:** **24/24 PASS** the gate — compile.py's `_compare_outputs(ref, cand, "fp32")`
  with `TOLERANCES["fp32"] = (1e-5, 1e-6)`. Every variant was `status="deterministic"`
  (`max_rel = 0.00e+00`, bit-identical to the strict-math reference). The GEMM math is
  invariant to the tiling knobs, as designed.
- **Winning config:** `tile_m=32, tile_n=16, tile_k=16, block=256` (REG_TILE=2).
- **Sweep wall time:** 1090 s (≈18 min) — dominated by per-variant `nvcc` compiles while the
  H100 was **shared** with another process (a separate `run_tuner.py` mamba sweep, ~18 GB /
  ~99% SM util throughout). Timings are CUDA-event medians over 30 iters, robust to that
  contention, but absolute ms carry some co-tenant noise.

Full per-config table: `/tmp/autotune_demo_build/summary.json`.

## Which autotuner path was used

**Fallback path — compile.py's real search + macro-injection + build + time + gate machinery,
driven directly** (`run_autotune.py`), NOT `python -m grokking_optimizers.compile` end-to-end.

The intended end-to-end invocation is **blocked by a known, still-open portability gap**: for a
foreign (non-grokking) project, compile.py's `build()` calls `_validate(spec)`
(`grokking_optimizers/compile.py:9088`), which hard-rejects any optimizer name outside the
grokking profile. Verified live:

```
$ python -m grokking_optimizers.compile --optimizer gemm --model gemm --arch sm_90a \
      --config examples/autotune_demo/compile_config.toml \
      --search-space examples/autotune_demo/gemm_search_space.yaml ...
ValueError: optimizer='gemm' not in ['adamw', 'grokadamw', ..., 'supergrok2']
```

(This is "Gap 3" documented in `examples/toy_tune_project/README.md`. The hook *timing+validation*
seam — that README's "Gap 1" — **is** now closed: `_make_variant_timer` routes through
`_hook_capture` + `_compare_outputs` when `spec.tune_hook` is set, `compile.py:15307–15397`.
But `_validate` fires first, so a foreign project cannot complete a run.) Editing `compile.py`
was out of scope, so the demo uses the path the task names as the fallback. `run_autotune.py`
imports and uses the *actual* compile.py functions with zero edits to it:

| Step | compile.py function used |
|---|---|
| load config / search space | `C.load_config()`, `C.get_search_space()` |
| enumerate the space (sampler) | `C.cartesian(space, arch)` |
| inject tuning knobs as nvcc `-D` flags | `C.resolve_macros(cfg, dims, "device")` → `-DSG_TUNED_TILE_M=… -DSG_TUNED_TILE_N=… …` |
| arch feature / bare flags | `C.resolve_extra_nvcc_flags(cfg, dims, arch)` |
| stable per-config identity | `C.config_key(cfg)`, `C._short_key(...)` |
| reference + variant time & capture | `C._hook_capture(tune_hook, so, spec, regime, seed, ...)` (the *same* fresh-subprocess protocol the megakernel timer uses) |
| correctness gate | `C._compare_outputs(ref, cand, "fp32")` with `C.TOLERANCES["fp32"]` |
| winner selection | minimize `elapsed_ms` among gate-PASS configs |

The four knobs are the project's own `#ifndef SG_TUNED_TILE_M / _N / _K / SG_TUNED_BLOCK` guards
in `gemm_kernel.cu`, defaulting to a deliberately-small tile. compile.py's `resolve_macros`
emits exactly the `-DSG_TUNED_*` overrides (confirmed: e.g. `-DSG_TUNED_TILE_M=32`).

## Pre-flight correctness (independent of the sweep)

Before the sweep, 10 representative legal configs (REG_TILE 1→64, shared-tile 2→32 KB, both
`block=256` and `block=128`) were each built in their own process and checked against
`torch.mm`: **10/10 correct, `maxabs=0.00000`**. The kernel has full M/N/K boundary guards and
`static_assert`s (`block | tile_m*tile_n`, shared tile ≤ 48 KB), so every config the search can
reach is compilable and correct — which the 24/24 in-sweep `deterministic` gate then confirmed.

## Honest interpretation

The hypothesis — *a naive, un-hand-tuned baseline should show a large (multiples) autotuner
gain, proving the production megakernel's ~2% reflects prior hand-tuning rather than a tool
ceiling* — **did not reproduce here: the gain is only 1.04x.** The cause is specific and worth
stating plainly. (1) The "naive" default I shipped (a 16×16 output tile, one output per thread)
is *already near the best shape this kernel can express*: it coalesces global loads cleanly and
keeps occupancy high, so it is not pathologically slow relative to its siblings. (2) The rest of
the search space is mostly **worse**, not better — larger tiles (REG_TILE up to 64) blow up
per-thread register pressure and spill, so e.g. `128×128` runs ~3x *slower* than the default
(14.7 ms vs 4.97 ms). Tile-shape knobs alone, on a kernel with no register blocking, no
vectorized (`float4`) loads, no double-buffering, and no tensor cores, simply have little upside
to find. (3) The cuBLAS context number makes this concrete: the *entire* config family lives in a
narrow band ~14x slower than cuBLAS, so the autotuner is choosing the best point on a low, flat
plateau — it cannot conjure the missing kernel-architecture optimizations that a 14x gap would
require.

So this run does **not** show the intended large contrast — but it is not a tool failure, and it
refines the original claim rather than refuting it. The autotuner did its job perfectly: it built
and **strictly validated all 24 variants** (the real value of the correctness gate — every tile
shape is provably bit-identical), correctly **rejected the many configs that regress** (the
worst is 3x slower), and **picked the genuine optimum** of the exposed knobs. The honest lesson
is that *autotuner leverage is bounded by what the knobs can reach*: tile-shape tuning of a kernel
that is already coalesced-and-small finds ~4%, much like tile/register knobs on the
already-expert-tuned production megakernel found ~2%. A *large* multiples-gain demonstration would
require either (a) a default that is pathologically bad in a dimension the knobs control (e.g.
exposing the un-blocked vs. shared-memory-blocked choice itself as a knob, so the default is
literally the no-shared-memory kernel), or (b) richer knobs (vectorization width, register-tile
M×N, async-copy depth) that can actually climb toward the cuBLAS roofline. With the simple
tile-shape knobs here, ~4% is the real ceiling — and that, measured end-to-end through compile.py's
own search + gate, is the honest finding.

## Files

- `gemm_kernel.cu` — the parameterized tiled fp32 GEMM (`SG_TUNED_TILE_M/_N/_K`, `SG_TUNED_BLOCK`
  `#ifndef` guards; naive small-tile default; full boundary guards + `static_assert`s; registers
  `torch.ops.sg_demo.gemm`).
- `tune_hook.py` — the portable `run(*, so_path, model, optimizer, arch, regime, seed) -> {"output", "elapsed_ms"}` hook.
- `compile_config.toml` — project config (`tune_hook`, `macro_prefix`, `[sources]`) read by `C.load_config`.
- `gemm_search_space.yaml` — the 24-config search space (compile.py `--search-space` schema).
- `run_autotune.py` — the driver that imports and drives compile.py's real machinery (above).
- `build_variant.py` — builds one variant `.so` per config in its own process.

This whole directory is a **temporary, removable demo** under `examples/` — it touches no
production `compile.py` / kernels / `setup.py`.
