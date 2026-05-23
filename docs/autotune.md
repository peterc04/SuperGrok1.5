# Autotune guide — `grokking_optimizers.compile`

This doc covers the autotune surface in depth: the two modes, the PGO
loop, the YAML search-space schema, the runtime split, and
troubleshooting. The [optimization matrix](optimization_matrix.md)
records which §12 candidates landed enabled-by-default vs. behind-flag.

## TL;DR — typical workflows

```bash
# 1) Production: AOT + Bayesian autotune + profile, end-to-end
python -m grokking_optimizers.compile \
    -O supergrok2 -M mamba -A sm_90 \
    --cache build/.compile_cache.json \
    --bayesian-trials 500 --top-k 20

# 2) CPU host preps the AOT cache; GPU host finishes the sweep
#    (host A)
python -m grokking_optimizers.compile \
    -O supergrok2 -M mamba -A sm_90 \
    --cache build/.compile_cache.json --aot-only \
    --aot-artifact-dir build/compiled/aot_artifacts
#    (rsync the cache + artefact dir to host B)
#    (host B — has H100)
python -m grokking_optimizers.compile \
    -O supergrok2 -M mamba -A sm_90 \
    --cache build/.compile_cache.json --jit-only

# 3) PGO-flavoured AOT then JIT autotune on the PGO binary
python -m grokking_optimizers.compile \
    -O supergrok2 -M mamba -A sm_90 \
    --cache build/.compile_cache.json --pgo \
    --pgo-workload scripts/pgo_workload.py --pgo-steps 1000

# 4) Quick debug sweep (25 Bayesian trials, no PGO)
python -m grokking_optimizers.compile -O lion -M mamba -A sm_90 --quick

# 5) Exhaustive sweep (every survivor after the static pre-filter)
python -m grokking_optimizers.compile \
    -O lion -M mamba -A sm_90 --mode exhaustive
```

## The two autotune modes

### `--mode bayesian` (default)

Two-stage Optuna TPE-driven sweep.

**Stage 1 — TPE** runs `--bayesian-trials` (default 500) iterations of
the multivariate `TPESampler`. The first 10% are Latin-Hypercube
warm-up; ~3% random fraction throughout to escape local minima. Every
trial is recorded in `cache.bayesian_trials` with `stage="tpe"`.

**Stage 2 — neighbour refinement** takes the top-K (default 20)
successful trials and benchmarks every ±2-step neighbour on every dim
of the YAML search space. Dedup against the TPE-seen set is automatic.
Recorded as `stage="refine"`.

Winner = `min(timing_ms)` across both stages. Resumable: the Optuna
study is persisted to `<out>/optuna_<opt>_<model>_<arch>.db` so a
second run with the same args picks up where the first left off (TPE
re-uses its own history; refine repeats deterministically because
neighbour generation is pure).

### `--mode exhaustive`

Every config that survives the YAML pre-filter is built and timed.
The cache is written every 5 trials so Ctrl-C is recoverable.

When to use exhaustive: the search space is small enough (after
pre-filter) that running it all is cheaper than running 500 TPE
trials, or you want a ground-truth ranking for the matrix. On full
sm_90 space (~thousands of survivors) Bayesian wins on wall-time.

## Search-space YAML schema

The YAML at `configs/search_space.yaml` is the single source of truth
for the tunable dims. One top-level dict per arch:

```yaml
sm_90:
  dims:
    - name: block
      type: int
      values: [64, 128, 256, 512, 1024]
      macro: SG_TUNED_BLOCK_SIZE
      applies_to: [host, device]
    - name: maxrregcount
      type: int
      values: [128, 168, 200, 232, 255]
      macro: null               # not a -D; promoted to NVCC --maxrregcount=N
      applies_to: [device]
    - name: cluster_shape
      type: tuple
      values:
        - [1, 1, 1]
        - [2, 1, 1]
        - [2, 2, 1]
      macro: SG_TUNED_CLUSTER_SHAPE
      applies_to: [device]
    - name: warp_specialization
      type: bool
      values: [false, true]
      macro: SG_TUNED_WARP_SPECIALIZATION
      applies_to: [device]
  prefilter:
    register_pressure_max: 255
    smem_budget_bytes:    232448
    rules:
      - name: vec_block_alignment
        expr: "block % (vec * 4) == 0"
      - name: tma_requires_block
        expr: "(not tma) or block >= 128"
```

**Per-dim fields**:
- `name`: identifier used in pre-filter expressions and as the
  cache-trial key.
- `type`: `int` | `bool` | `enum` | `tuple`.
- `values`: list of literals (tuples become Python tuples post-load).
- `macro`: the `-D` name; pass `null` to skip macro emission (then it
  becomes a bare compiler flag — currently supports `maxrregcount`).
- `applies_to`: subset of `[host, device]` — controls which flag list
  picks up the macro.

**Per-arch pre-filter**:
- `rules`: list of `{name, expr}`. Each `expr` is a Python boolean
  evaluated against the config dict. False eliminates that config.
- `register_pressure_max`, `smem_budget_bytes`, `waves_per_eu_max`:
  reserved keys for future static analysers. Currently informational.

The pre-filter elimination count is logged at the start of every
autotune run (search this in the report file for "[prefilter]").

## The PGO loop (`--pgo`)

Three-pass build:

1. **Instrument** AOT-build with `-fprofile-generate=<dir>` on the
   host compiler and `-Xcompiler -fprofile-generate` on NVCC (the HIPCC
   path passes the flag directly).
2. **Collect** runs the workload script (default
   `scripts/pgo_workload.py`) which loads the instrumented `.so` and
   executes `--pgo-steps` (default 1000) optimizer steps. The driver
   sets `LLVM_PROFILE_FILE` and `GCOV_PREFIX` so all profile files
   land under `<out>/pgo_profile/`. The collector validates that at
   least one new profile file appeared.
3. **Use** AOT-rebuild with `-fprofile-use=<dir>` and
   `-fprofile-correction`. JIT autotune then runs on the PGO binary.

Invalidation: the cache stores `pgo_workload_hash = sha256(workload
contents, step count)`. Changing either invalidates the PGO entry.
The cache also records `pgo_enabled` as a freshness factor so a
non-PGO build never reuses a PGO artefact (and vice versa).

## Runtime split (`--runtime`)

Default is `--runtime both`: `main()` spawns an AOT subprocess (no GPU
init), then a JIT subprocess (full GPU env). Both processes
read/write the same on-disk cache. Use `--runtime aot` or
`--runtime jit` (or the aliases `--aot-only` / `--jit-only`) to run
only one half.

When AOT and JIT are on different machines, pass `--aot-artifact-dir
<shared-path>` on the AOT host. The .so is published into that dir
and the JIT host picks it up via the cache's
`primary_artifact.path`. If the path differs between hosts (e.g. NFS
on different mount points), the cache is the source of truth — copy
both the cache JSON and the artefact dir over.

## Optional optimizations (per `docs/optimization_matrix.md`)

The following are landed in `compile.py` from §12:

| Flag / env | Default | Effect | §12 row |
|---|---|---|---|
| `--pruner {none, median, hyperband}` | none | Optuna pruner during TPE; Hyperband enables Successive Halving brackets when timing reports intermediate values | A1 |
| `--transfer-learning` | off | Seed the TPE study with prior trials from sibling-optimizer studies on the same `(model, arch)` | A2 |
| Newer-compiler probe | on (auto) | `_torch_load` logs the nvcc/hipcc version. When nvcc ≥ 12.6 is on PATH, `--split-compile=$(nproc)` is appended automatically | C1 |
| ccache fallback | on (auto) | When `ccache` is on PATH it takes the host `CC`/`CXX` wrappers (typically 3-4.5× faster than sccache on host TUs). sccache always handles NVCC | C2 |
| Redis-backed sccache | on if env set | `SCCACHE_REDIS_ENDPOINT` propagates into child builds for cluster-wide cache sharing | I1 |

The behind-flag candidates listed in the matrix have stub argparse
entries on the roadmap but are not wired into `compile.py` yet — they
either need device-toolchain validation (BOLT-toolchain, splitting
heavy TUs) or telemetry (per-shape autotune, ensemble dispatch).

## Determinism + reproducibility

- `--seed N` seeds the TPE sampler. Same `(arch, seed, n_trials,
  search-space.yaml)` → same TPE trial order.
- Optuna study persisted to `<out>/optuna_<opt>_<model>_<arch>.db`.
  Re-running with the same study name resumes.
- Every config + timing is appended to `cache.sweep_history` (with a
  `stage` field) and `cache.bayesian_trials`.
- Cache writes are atomic (tmp + rename) so a Ctrl-C never corrupts
  the file. Reloading a corrupt cache archives it as
  `<cache>.corrupt.bak` and starts fresh.
- v2 → v3 forward migration is automatic on load; the v2 file is
  backed up as `<cache>.v2.bak`.

## Troubleshooting

**sccache miss / 0% hit rate on NVCC.** Older sccache versions had a
[CUDA hash instability bug](https://github.com/mozilla/sccache/issues/160).
Upgrade sccache to ≥ 0.8 and re-run; the cache should populate after
the second build.

**Optuna study won't resume.** Check that
`<out>/optuna_<opt>_<model>_<arch>.db` exists and is the same SQLite
file. Optuna's `study_name` includes `(opt, model, arch)` so changing
any of those starts a fresh study (intentional — different combos are
different problems).

**`.gcda` files empty after PGO collect.** The instrumented build
must export `LLVM_PROFILE_FILE` (set automatically by
`grokking_optimizers.pgo.collect_workload`). Verify the workload
actually ran the optimizer's hot path — exit code 0 with no profile
files written is a script bug. The collector explicitly reports
"no profile files appeared" in this case.

**AOT/JIT cache-key mismatch across hosts.** The `is_aot_fresh` check
factors in `(source_hash, host_cflags_hash, device_cflags_hash,
pgo_enabled, pgo_workload_hash, search_space_hash)`. If the JIT host
sees `aot_completed_at` but no artefact at `primary_artifact.path`,
copy the artefact alongside the cache (or re-run AOT on the JIT host;
the cache will pick up the new artefact path).

**Worker crash mid-sweep.** `_jit_autotune` logs `[worker time failed
for ...; restart + fallback]` and falls back to one-shot subprocess
timing for that variant before restarting the worker. The sweep keeps
going. If the worker fails repeatedly, set
`SG_DISABLE_PERSISTENT_WORKER=1` (not yet implemented; planned escape
hatch) or use `--mode exhaustive` which is more crash-tolerant.

**Transfer-learning has no effect.** Sibling trials live in
`cache.bayesian_trials` keyed by `<sibling_opt>/<model>/<arch>`. Run
the sibling optimizer first (e.g. `-O adamw`) before invoking
`--transfer-learning` on the target (e.g. `-O supergrok2`).
