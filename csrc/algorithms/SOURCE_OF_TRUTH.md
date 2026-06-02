# Optimizer-math single source of truth (Phase 4 WS4)

## TL;DR
`csrc/algorithms/<opt>.h` is the **canonical** per-element optimizer math. Edit
the update math **there and only there** for the CUDA paths. The check
`python3 scripts/check_math_single_source.py` enforces the invariant in CI/local.

## Why this resolves the "two trees can drift" gap
There appeared to be two live CUDA optimizer-math trees. In fact both **derive
from one** by `#include` (no copy), so they cannot drift:

| consumer | path | how it gets the math |
|----------|------|----------------------|
| per-op kernel (the `setup.py`-compiled path) | `grokking_optimizers/kernels/sm_90/<opt>_sm90.cuh` (included by `csrc/backends/cuda/sm_90/launch_<opt>.cu`) | `#include "csrc/algorithms/<opt>.h"` → calls `<opt>_step(...)` |
| fused L3/L1 megakernel | `csrc/fused/sm_90/opt_components.cuh::apply_optimizer<OptId>` | `#include "csrc/algorithms/<opt>.h"` → calls `<opt>_step(...)` |

`scripts/check_math_single_source.py` fails if any `<opt>_sm90.cuh` stops
`#include`-ing its canonical header (i.e. someone reimplemented the math).

## Necessary RE-EXPRESSIONS (cannot `#include` the C header — keep in sync by hand)
These are different languages/toolchains, so they re-express the same math. Each
carries a cross-reference comment back to the canonical header; the check script
inventories them.

| arch | path | why it can't include the canonical header | sync obligation |
|------|------|-------------------------------------------|-----------------|
| gfx942 device | `csrc/fused/gfx942/opt_components.hip.hpp` | `csrc/algorithms/*.h` → `platform.h` `GROK_HIP` branch pulls `<thrust/...>`, which the free-standing AMDGCN clang gate can't resolve | transcribed **byte-faithfully**; comment marks each as a transcription of `csrc/algorithms/<opt>.h`. A math edit must be mirrored here. |
| gfx942 per-op | `grokking_optimizers/kernels/gfx942/<opt>_gfx942.hip.hpp` (device pass) | hand-written AMDGCN device kernels (DPP/MFMA/streaming) — a different kernel shape, not an elementwise re-include | mirror the *formula*, not the layout. |
| tpu_v5p | `csrc/backends/pallas/launch_<opt>.py` (authoritative) + `grokking_optimizers/kernels/tpu/<opt>_tpu.py` (reference; the 4 base ones re-export, WS5) | JAX/XLA, a different runtime | the JAX path is the canonical TPU math; the 4 base `kernels/tpu` shims re-export `launch_<opt>.py` (single TPU source). |

## Edit protocol
1. Change `csrc/algorithms/<opt>.h` — the CUDA per-op and fused paths update
   automatically (they `#include` it).
2. Mirror the change in `csrc/fused/gfx942/opt_components.hip.hpp` (gfx942
   transcription) and, if the formula changed, the gfx942 per-op device kernel.
3. Mirror in the TPU JAX path (`csrc/backends/pallas/launch_<opt>.py`).
4. Run `python3 scripts/check_math_single_source.py` (structural guard) and the
   `--self-test` numeric oracles.

## Status
- CUDA single-source: **enforced + verified** (`check_math_single_source.py` → OK).
- gfx942 / TPU re-expressions: **cross-referenced, manual-sync** (numeric parity
  vs the canonical reference is 🟡 — MI300X / TPU, see `HARDWARE_VALIDATION.md`).
