# `autotune/` — Offline Per-Arch Kernel Tuning

Replaces the runtime `grokking_optimizers/jit/` specializer.

`tune.py` runs a microbenchmark grid for each (kernel, arch, shape) and
writes the winners to `csrc/common/tuned_configs.h`. The header is committed
to the repo and `#include`d by every per-arch kernel for `__launch_bounds__`,
`BLOCK_M`, `BLOCK_N`, `STAGES`, etc.

## How it works

1. `grids.py` defines the parameter grid per kernel (block sizes, stages,
   warps, etc.). One Python dict per kernel.
2. `runner.py` builds a tiny test extension that compiles a single kernel
   under multiple `__launch_bounds__` and tile-size combinations, runs each
   under representative inputs, and times them with CUDA events / HIP
   events / TPU step time. Median latency wins.
3. `cutlass_profile.py` wraps the CUTLASS profiler binary for the SG2
   projection GEMMs and Muon Newton-Schulz GEMMs (the only places we use
   CUTLASS). The profiler explores tile shapes / pipeline stages / split-K
   in the standard CUTLASS way; the winning config is written into
   `tuned_configs.h` alongside the hand-rolled kernel configs.
4. `tune.py` is the CLI entry. Picks an arch, walks the grid for each
   kernel, writes results into `csrc/common/tuned_configs.h`.

## Usage

```
python autotune/tune.py --arch sm_90 --output csrc/common/tuned_configs.h
python autotune/tune.py --arch all   --output csrc/common/tuned_configs.h
python autotune/tune.py --kernel grokadamw --arch sm_90 --dry-run
```

You need a GPU of the target arch to run autotune for that arch. Run
once per arch you care about; commit the result.

## Output format

`csrc/common/tuned_configs.h` is a C++ header containing `static
constexpr LaunchConfig` tables keyed by `(kernel_id, arch_id, shape_bucket)`.
See the comments in the file itself for the exact schema.

## Adding a new kernel

1. Add an entry to `grids.py` describing the parameter axes and ranges.
2. Add the corresponding template parameters to the kernel
   (`__launch_bounds__(BLOCK_M, MIN_BLOCKS_PER_SM)`,
   `template <int BLOCK_M, int BLOCK_N, int STAGES>`).
3. Re-run `tune.py` on each arch.

## Staleness

If `csrc/common/tuned_configs.h` is missing or older than `grids.py`, the
build emits a warning and uses the default configs. A CI job verifies the
committed header is consistent with the grid definitions but does not
re-tune (re-tuning needs hardware).
