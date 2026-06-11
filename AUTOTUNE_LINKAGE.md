# Autotuner → Product-Build Linkage (operator runbook)

**What this covers.** How to make `compile.py`'s autotuned launch parameters
ACTIVE in the shipped `grokking_optimizers/_ops*.so` — produce winners, rebuild
so the per-TU flags are baked in, verify they actually landed, and re-run the
parity + roofline gates against the tuned binary.

**Background — lever (b).** The `compile.py` JIT autotuner builds and times
kernel variants and picks a winner per `(optimizer, model, arch)`. Historically
that winner only reached a header that nothing on the install path included, so
the product `.so` shipped the in-header defaults — the autotuner was
decorative. The linkage now is:

```
compile.py build_jit  --(write-on-win, atomic)-->  grokking_optimizers/_kernel_tuned.json
                                                          |
pip install -e .  /  ./build.sh                           v
   setup.py TunedBuildExtension  --reads JSON, injects per-TU nvcc flags-->  _ops*.so
```

The five SAFE per-TU dims are applied: `-DSG_TUNED_BLOCK_SIZE`,
`-DSG_TUNED_VEC_WIDTH`, `-DSG_TUNED_UNROLL`, `-DSG_TUNED_ASYNC_DEPTH`, and
`--maxrregcount=N` (only when N>0). They go onto each optimizer's CUDA TUs
(`csrc/backends/cuda/sm_90/launch_<opt>.cu` and the megakernel cells
`csrc/fused/sm_90/mega_<model>_<opt>.cu`). The per-optimizer kernel header
`kernels/sm_90/<opt>_sm90.cuh` is `#include`d by `launch_<opt>.cu`, so flags on
the launcher TU reach the kernel body. Bindings (`*.cpp`), model-only TUs
(`models/<model>.cu`), and common/header TUs get **no** per-optimizer flags and
keep their in-header defaults. (`cluster_shape` and the feature macros — TMA /
WGMMA / fp8 / swizzle — are component-scoped and riskier; they are phase-2 and
are **not** emitted by this pass.)

> If `_kernel_tuned.json` is absent the build is **byte-identical to before**:
> every kernel uses its `#ifndef SG_TUNED_*` default (block 256 / vec 4 /
> unroll 1 / async_depth 2; nvcc's own register allocator). A one-line notice
> is printed.

---

## 0. Prerequisites

- A GPU host (the JIT autotuner builds + times real kernels). Steps 1 is
  GPU-bound; steps 2–4 only need a CUDA toolchain (`FORCE_CUDA=1` lets the
  build configure without a visible device).
- `ninja` on PATH (the injection hook patches torch's ninja writer; the build
  already requires `use_ninja=True`).
- The repo installed editable at least once (so the extension build tree
  exists): `FORCE_CUDA=1 pip install -e . --no-build-isolation`.

---

## 1. Produce winners → `grokking_optimizers/_kernel_tuned.json`

Run the JIT autotuner **once per optimizer** (optionally per model). Each run
read-merge-writes its winner into the canonical JSON, so 11 sequential runs
accumulate 11 entries under the arch key.

```bash
# One optimizer (sm_90 / Hopper). Repeat for each of the 11.
python -m grokking_optimizers.compile \
    --optimizer adamw --model decoder --arch sm_90 \
    --cache build/.compile_cache.json \
    --jit-only --mode bayesian --bayesian-trials 300
```

Optimizers: `adamw grokadamw grokfast lion looksam muon neuralgrok prodigy
supergrok11 supergrok15 supergrok2`. Models (short): `decoder vit mamba`.

> **One winner per `(arch, optimizer)` — model is LAST-WINS.** The JSON schema
> is `{arch: {optimizer: {...}}}` (no model dimension), so tuning the same
> optimizer on a *second* model overwrites the first, and that single winner is
> applied to ALL of that optimizer's TUs — its `launch_<opt>.cu` AND every
> `mega_<model>_<opt>` megakernel cell across all models. If you tune `decoder`
> then `vit` for `adamw`, only `vit`'s `adamw` winner survives. Pick the model
> whose launch params you want shipped (the `"model"` field in each entry
> records which sweep produced it).

A convenience loop:

```bash
for opt in adamw grokadamw grokfast lion looksam muon \
           neuralgrok prodigy supergrok11 supergrok15 supergrok2; do
  python -m grokking_optimizers.compile \
      --optimizer "$opt" --model decoder --arch sm_90 \
      --cache build/.compile_cache.json --jit-only --mode bayesian
done
```

> **Run optimizers SEQUENTIALLY, not in parallel processes.** The exporter's
> write is atomic (tmp + `os.replace`), so a reader never sees a half-written
> file — but it is read-merge-write, so two concurrent per-optimizer
> *processes* can clobber each other's just-written entry. One process at a
> time (the loop above) is safe. A failed JSON write is a loud warning in the
> compile report and **never** fails the tuning run; the winner is still in the
> compile cache and `tuned_configs.h`.

Confirm the JSON was written and looks sane:

```bash
python -c "import json;d=json.load(open('grokking_optimizers/_kernel_tuned.json'));\
import pprint;pprint.pprint({k:sorted(v) for k,v in d.items() if k!='_meta'});\
print('meta:',d.get('_meta'))"
```

Schema (canonical, short arch key):

```json
{
  "sm_90": {
    "adamw": {"block":256,"vec":4,"unroll":1,"async_depth":2,
              "maxrregcount":0,"model":"decoder"},
    "supergrok2": {"block":128,"vec":2,"unroll":4,"async_depth":3,
                   "maxrregcount":96,"model":"decoder"}
  },
  "_meta": {"timestamp":"…","source":"grokking_optimizers.compile.build_jit",
            "last_optimizer":"supergrok2","last_arch":"sm_90",
            "compile_py_version_hash":"…"}
}
```

---

## 2. Rebuild — bake the tuned flags into `_ops*.so`

A plain rebuild now consumes the JSON automatically (no special flag — the old
`./build.sh --autotune` two-pass flow was removed; see §5):

```bash
# editable rebuild (FORCE_CUDA=1 if no visible GPU on the build host)
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0a" \
    pip install -e . --no-build-isolation -v
# or:  ./build.sh
```

Watch for the build-time notice. With a JSON present you should see lines like:

```
  [tuned-inject] injected per-TU tuned flags into N CUDA TU(s) in build.ninja.
```

With no JSON you should instead see:

```
  [tuned-inject] no _kernel_tuned.json found; building with in-header kernel defaults …
```

---

## 3. Verify the flags actually landed

The autotuner's own acceptance criterion: the flags must be in `build.ninja`.

```bash
NINJA=$(ls build/temp.*/build.ninja | head -1)
grep -cE 'SG_TUNED'      "$NINJA"   # expect NONZERO now (was 0 — decorative)
grep -cE 'maxrregcount'  "$NINJA"   # nonzero iff some winner had maxrregcount>0
# See exactly which TUs were overridden (per-build-statement cuda_post_cflags):
grep -nE 'cuda_post_cflags = .*SG_TUNED' "$NINJA"
```

Expected: one indented `cuda_post_cflags = … -DSG_TUNED_… ` override line
directly beneath each `build …/launch_<opt>.o: cuda_compile …` (and each tuned
`mega_…` cell). Each override carries the **full** base flag list (arch / `-O3`
/ fast-math) **plus** the tuned extras — the base flags must still be present
(a missing `compute_90a` / `--use_fast_math` would mean the override dropped
the base; that is a bug, not expected).

Optionally confirm the emitted machine code reflects the register cap:

```bash
# Recompile one tuned TU standalone with -Xptxas -v and read "registers".
scripts/compile_to_object.sh \
    csrc/backends/cuda/sm_90/launch_supergrok2.cu \
    -DWITH_CUTLASS -Xptxas -v 2>&1 | grep -i 'registers'
```

### CPU-only sanity (no GPU, no build)

The injection logic has a CPU-only unit test that asserts the path→optimizer
mapping, the per-source flag computation, the `build.ninja` rewrite, the
export round-trip, and macro/header drift:

```bash
python -m tuning.test_build_injection      # OK — all test groups passed.
```

---

## 4. Re-run parity + roofline against the tuned binary

The tuned `.so` changes codegen (block size, vectorization, register pressure),
so re-gate correctness and re-measure performance.

```bash
# Single-step fused-vs-reference numerical PARITY (H100):
PYTHONPATH=. python3 tests/hw/parity_gate_h100.py

# Reference-parity unit tests (the non-hw half runs anywhere):
PYTHONPATH=. python3 -m pytest tests/hw/test_reference_parity.py -q

# Roofline / distance-to-roofline for every optimizer × model:
python -m tuning.roofline --models decoder,vit,mamba --steps 100
#   -> results/h100_grokking_race/roofline.{json,png}, ROOFLINE.md
```

A tuned build must still PASS parity (the autotuner's winner selection already
gates on an on-device numerical oracle, so a winner that fails parity should
not have been written — treat any parity regression as a real bug). Roofline
closeness is expected to improve or hold for the tuned optimizers.

---

## 5. Notes / gotchas

- **`./build.sh --autotune` is gone.** It called a nonexistent `autotune/tune.py`
  and staged the wrong header path. The autotuner is `grokking_optimizers.compile`;
  invoking `--autotune`/`--no-autotune` now prints this runbook's pointer and
  exits.
- **Secondary header.** `compile.py` still writes
  `csrc/algorithms/tuned_configs.h` (back-compat consumer). The **canonical**
  handoff to the product build is `_kernel_tuned.json`; the header is not on the
  install include path and is not what bakes flags into the `.so`.
- **Source of truth for the macros.** The macro names / defaults and the
  TU→optimizer mapping live in `grokking_optimizers/_tuned_inject.py`
  (`MACROS`, `optimizer_for_source`). `MACROS` is kept in lockstep with the
  header guards at `grokking_optimizers/kernels/sm_90/adamw_sm90.cuh:31-41`; the
  unit test fails if they drift.
- **Future torch.** The injection hook monkeypatches
  `torch.utils.cpp_extension._write_ninja_file` (torch 2.4.x). If a future torch
  removes/renames it, the build prints a loud warning and falls back to a stock
  (in-header-defaults) build — it never hard-fails. Re-point the hook in
  `setup.py` (`TunedBuildExtension.build_extensions`) if that happens.
- **AMD / TPU.** The same JSON schema supports a `gfx942` arch key (the build
  arch key is chosen automatically). The pass currently emits the four
  `-DSG_TUNED_*` macros + `--maxrregcount`; AMD's equivalent VGPR cap
  (`-amdgpu-max-num-vgprs`) and TPU Pallas kwargs are out of scope for this
  per-TU nvcc pass.
```
