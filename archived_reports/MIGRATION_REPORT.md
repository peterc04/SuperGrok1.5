# Migration Report — Kernel Tree Consolidation

Collapse the two parallel kernel trees into **one**. The production device logic
moved into `grokking_optimizers/kernels/` (overwriting the scaffolding
placeholders); `csrc/backends/` reduced to thin TUs + shims + bindings.

## Verification method

The environment has **nvcc 12.0** (installed via pip wheels) and **CUTLASS
v3.6.0** (cloned to `third_party/cutlass`, gitignored), but **no GPU**.

Key finding: the original production tree **does not compile in include order on
its own** — each `launch_<opt>.cu` uses `WARP_SIZE` / `GROK_CUDA` inside the
algorithm header it includes, but only `#define`s them *after* the include. So
"the build compiles cleanly today" was never literally true with a real nvcc;
the `--self-test` suite validates the Python pipeline (flags, search space, cache
schema, tree structure), **not** kernel compilation.

Therefore the correct gate for a **byte-for-byte move** is **preprocessor
equivalence**: the thin `.cu` (header `#include`) must preprocess (`nvcc -E`)
to the exact same translation unit as the original `.cu`, modulo the embedded
`__FILE__` / `__LINE__` metadata inside torch's `TORCH_CHECK` macros. This was
proven for every CUDA TU (incl. SG2 with `-DWITH_CUTLASS`).

- sm_90: **preprocessor-equivalence gate PASS** on all 11 optimizers + 3 model TUs.
- gfx942: `.hip.cpp` route through the host compiler; verified by **byte-for-byte
  text match** of header body vs original (no preprocessor step needed).
- TPU: no code moved (see A.3); reference banner added.

`--self-test`: **137 passed, 1 failed** — identical to the pre-migration baseline
(the lone failure, `flag_base_superset_regression`, is **pre-existing** and
unrelated; `compile.py` flag tables were not touched). Net regressions: **0**.
`ruff check grokking_optimizers/`: clean.

## sm_90 optimizers — `launch_<opt>.cu` → `<opt>_sm90.cuh` + thin `.cu`

| Optimizer | before `.cu` (LOC) | header `.cuh` (LOC) | thin `.cu` (LOC) | PTX blocks | CUTLASS collectives | PP-gate |
|---|---:|---:|---:|---:|---:|:--:|
| adamw | 1127 | 1145 | 5 | 10 | 0 | OK |
| lion | 1115 | 1133 | 5 | 10 | 0 | OK |
| grokfast | 1106 | 1124 | 5 | 10 | 0 | OK |
| grokadamw | 1119 | 1137 | 5 | 10 | 0 | OK |
| looksam | 1191 | 1209 | 5 | 10 | 0 | OK |
| muon | 1143 | 1161 | 5 | 10 | 0 | OK |
| neuralgrok | 1124 | 1142 | 5 | 10 | 0 | OK |
| prodigy | 1169 | 1187 | 5 | 10 | 0 | OK |
| supergrok11 | 1191 | 1209 | 5 | 10 | 0 | OK |
| supergrok15 | 1156 | 1174 | 5 | 10 | 0 | OK |
| **supergrok2** | **3619** | **3637** | **5** | **16** | **6** | **OK** |

(header LOC = original + 18-line include-guard/banner. PTX/CUTLASS counts are
exact matches of the originals — no drop.)

## sm_90 models — `models/<m>.cuh` → `<canon>_sm90.cuh` + thin shim

| Model | before (LOC) | canonical header (LOC) | shim (LOC) | PTX | CUTLASS | gate |
|---|---:|---:|---:|---:|---:|:--:|
| attention → attention | 1202 | 1219 | 4 | 5 | 4 | PP-OK (via decoder/vit TUs) |
| decoder → transformer_decoder | 1840 | 1857 | 4 | 5 | 0 | PP-OK |
| mamba → mamba3 | 2715 | 2732 | 4 | 11 | 7 | PP-OK |
| vit → vit | 2005 | 2022 | 4 | 5 | 0 | PP-OK |

The `csrc/backends/.../models/<m>.cuh` shims keep all external includers
(`bindings.cpp`, the `<m>.cu` instantiation TUs, the HIP tree) resolving
unchanged; the model `.cu` TUs (template instantiations) preprocess identically
through shim→canonical vs the original headers.

## gfx942 — `launch_<opt>.hip.cpp` → `<opt>_gfx942.hip.hpp` + thin `.hip.cpp`

All 11 optimizers + 4 model headers moved **byte-for-byte** (text-match
verified). Each header carries an `AMDGCN-asm status: NOT PRESENT` banner (the
path is ATen + rocBLAS; rocBLAS dispatches MFMA internally; native AMDGCN asm
needs `.hip.cpp`→`.hip`, roadmap item 2). The hand-written
`__builtin_amdgcn_mfma_*` reference in `mamba3_gfx942.hip.hpp` is **preserved
`#if 0`-guarded** (not deleted) as the template for that migration. The SG2
gfx942 bilevel throw (`csa_hca_bilevel_not_implemented`) is intact.

## TPU (Pallas)

No executing code moved. The authoritative, executed path is
`csrc/backends/pallas/launch_<opt>.py` (resolved by `profile.py` by hard path,
all 11 optimizers, importing the live `_pallas_kernels.py`); the
`kernels/tpu/*_tpu.py` files are a partial reference spec and are **not imported
by the production path** — so there were no duplicated executing kernel bodies
to collapse. Each `kernels/tpu/*_tpu.py` got a REFERENCE/SPEC status banner
documenting the (intentionally reversed) dependency direction. TPU has no
inline-asm concept (XLA/Pallas `BlockSpec` lowering).

## What moved vs what stayed

- **Moved to `grokking_optimizers/kernels/`**: every sm_90 device function,
  `__global__` kernel, PTX block, CUTLASS collective (optimizers + models); every
  gfx942 ATen/rocBLAS step body + model logic.
- **Stayed in `csrc/backends/`**: thin `.cu`/`.hip.cpp` TUs (one `#include`
  each), thin `models/*.cuh`/`*.hip.h` shims, the model `.cu` template
  instantiations, and bindings glue. **Zero device-function bodies remain.**
- **Unchanged (as required)**: `csrc/algorithms/*.h` math; optimizer Python
  frontends' public API; the race driver; the SG2 gfx942 bilevel throw.

## Scope reductions (to protect the build / honesty)

- **Part B perf annotations (`__launch_bounds__`, L2 persistence) NOT applied.**
  Adding/removing them changes the compiled translation unit and would break the
  preprocessor-equivalence guarantee that is the verification for a byte-for-byte
  move, with no way to validate the change without a GPU. The production kernels
  already carry their chosen `__launch_bounds__`. `__forceinline__ __device__`
  qualifiers were confirmed preserved on every device function. Recorded as
  roadmap items 4–5.
- **No standalone per-TU compilation claim.** Because the original tree does not
  compile in include order (above), compilation is validated structurally
  (preprocessor equivalence) here and must be finished on a GPU build host /
  CUDA CI runner — roadmap item 6.

## Self-test deltas

Two self-tests asserted the **deleted placeholder API** (`<opt>_update(`,
`namespace grokking`, placeholder size-helper tokens). Updated
`elementwise_headers` and `model_headers` to assert the **production contract**
(`namespace sg::sm90`, `<opt>_kernel(` launchers, `cudaError_t` model entry
points, per-element step in `csrc/algorithms/<opt>.h`). No count change.
