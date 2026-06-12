# Static register-pressure campaign — TC engines (Lane B, 2026-06-12)

Baseline: HEAD = `642e360` (pristine worktree; the shared tree was concurrently edited
by another lane). Everything in this report is **compile-time only** (no GPU process was
launched). All builds reproduce the PRODUCTION `_ops` device flag set:
`compile.py::NVCC_DEVICE_BASE` (`-O3 -std=c++17 -DWITH_CUDA --expt-relaxed-constexpr
-Xfatbin -compress-all -Xptxas --opt-level=3 -Xptxas -v -Xptxas --warn-on-spills
--extra-device-vectorization -Xcompiler -fPIC -Xcompiler -fno-strict-aliasing
--default-stream per-thread --resource-usage`) + setup.py's project flags
(`--use_fast_math -DNDEBUG -lineinfo`) + `-gencode=arch=compute_90a,code=sm_90a`
(the `compute_90` PTX-fallback gencode runs no ptxas pass, so it cannot change these
numbers). Harness: `compile_one.sh`; parser: `parse2.py`; raw logs in `logs/`.

The measured TUs are the production TC cell carriers
(`mega_{decoder,vit,mamba}_real_adamw_tc_launcher.cu`), each of which instantiates the
shared TC engine once per OptId — so one TU yields the full engine x tail matrix.
Bench layouts measured with `-DSG_DEC_BENCH_LAYOUT=1 -DSG_DEC_SCALAR_MEGAKERNEL=0` /
`-DSG_VIT_BENCH_LAYOUT=1` / `-DSG_MB_BENCH_LAYOUT=1` (the d=1024 branches; they DO
change codegen, so both are tabulated).

## Reading the numbers (load-bearing)

1. **Every TC kernel reports `Used 255 registers` in every build** — the kernels carry
   `__launch_bounds__(256)`, so ptxas fills the register file to the cap by design.
   "Used" will never drop below 255 on this launch config; **spill bytes are the
   pressure metric**, not the register count.
2. **`-Xptxas --maxrregcount` is silently IGNORED for these kernels** (launch_bounds
   takes precedence) — verified at 240/224/192: byte-identical resource output. The
   "maxrregcount hard-cap" anti-pattern is not even expressible here.
3. **Kernel-level spill numbers are incomplete**: `__noinline__` device functions
   (mamba uses them at HEAD) carry their own spill counts that ptxas reports per
   function, not on the entry line. All tables below show **TOTAL = entry + callee**
   spill stores (parse2.py attributes callee blocks to their entry section). This
   correction *changes the picture for mamba* (see Inventory).

## 1. INVENTORY (baseline, HEAD, production flags)

TOTAL spill stores / loads = entry + callee functions. smem = static per CTA.
All kernels: 255 regs, 256 threads/CTA -> 65 280 regs/CTA = the full 64K SM file ->
**theoretical occupancy 1 CTA/SM** (exactly the persistent design's pin; smem 9.7-43.9 KB
of 227 KB is never the limiter). Occupancy is identical in every variant below.

### decoder (production d=128)              | bench d=1024
| tail        | regs | TOT sp_st | TOT sp_ld | smem  | bench TOT sp_st |
|-------------|------|-----------|-----------|-------|-----------------|
| AdamW       | 255  | 0         | 0         | 43188 | 0               |
| Lion        | 255  | 0         | 0         | 43188 | 0               |
| Grokfast    | 255  | 0         | 0         | 43188 | 0               |
| GrokAdamW   | 255  | 0         | 0         | 43188 | 0               |
| NeuralGrok  | 255  | 0         | 0         | 43188 | 0               |
| Prodigy     | 255  | 0         | 0         | 43444 | 0               |
| Muon        | 255  | 0         | 0         | 43316 | 0               |
| LookSAM     | 255  | **15252** | 15552     | 43188 | 15028           |
| SuperGrok11 | 255  | **15252** | 15552     | 43828 | 15028           |
| SuperGrok15 | 255  | **15252** | 15552     | 43700 | 15028           |
| SuperGrok2  | 255  | **15424** | 15896     | 43188 | 15264           |

### vit (production d=128)                  | bench d=1024
| tail        | regs | TOT sp_st | smem  | bench TOT sp_st |
|-------------|------|-----------|-------|-----------------|
| AdamW..Prodigy (7 light) | 255 | 0-8 | 9700-9956 | 0-8 |
| LookSAM     | 255  | **15020** | 9700  | 15364           |
| SuperGrok11 | 255  | **15020** | 10340 | 15364           |
| SuperGrok15 | 255  | **15020** | 10212 | 15364           |
| SuperGrok2  | 255  | **18288** | 9704  | 19036           |

### mamba (production d=128)                | bench d=1024
mamba's engine already `__noinline__`s its tile fns at HEAD — the per-fn correction
reveals every cell pays them:
| tail        | regs | entry sp_st | callee sp_st | TOT sp_st | bench TOT |
|-------------|------|-------------|--------------|-----------|-----------|
| AdamW       | 255  | 44          | 5804         | **5848**  | 6760      |
| Lion/Grokfast/NeuralGrok | 255 | 44 | 5804      | **5848**  | ~6760     |
| GrokAdamW   | 255  | 8           | 5684         | **5692**  | 6596      |
| Prodigy     | 255  | 20          | 5684         | **5704**  | ~6600     |
| Muon        | 255  | 48          | 5804         | **5852**  | ~6760     |
| LookSAM/SG11/SG15 | 255 | 3396   | 6056         | **9452**  | 10356     |
| SuperGrok2  | 255  | 3648        | 6532         | **10180** | 11164     |

**Reconciliation with the morning report:** "vit reg spills 6452B -> 0" refers to the
production (AdamW) vit cell — confirmed still 0 at HEAD. "mamba 255-reg budget held"
was a kernel-entry-level statement; the callee-inclusive truth is ~5.8 KB on every
mamba cell. "the shared engine is at 255 regs WITH 2.6 KB spills" matches the
mamba-style per-fn footprint, and the 15 KB SAM-cell numbers above are the same
blocker amplified by the duplicated second pass.

## 2. ATTRIBUTION (probe data, decoder)

| probe (production flags + one knob) | AdamW regs | AdamW sp_st | LookSAM TOT sp_st |
|---|---|---|---|
| baseline (IL=2, N=128, S=2)            | 255 | 0   | 15252 |
| `SG_TUNED_DEC_GEMM_INTERLEAVE=1` (acc 128->64 regs) | **253** | 0 | 356 |
| `SG_TUNED_TILE_N=64` (acc 128->64 regs)             | **253** | 0 | 0   |
| `SG_TUNED_DEC_GEMM_STAGES=1`           | 255 | 0   | 13300 (smem -8 KB) |
| `--maxrregcount` 240/224/192           | 255 | 0   | 15252 (flag ignored) |

* **The wgmma accumulator array owns the margin.** `WgmmaAccum<128>` = 64 fp32
  regs/fragment; the M-atom interleave (kIL=2, the H1 win) keeps TWO live = 128 regs
  in the K-loop. Halving the accumulator total (either knob) lands the engine at 253
  regs / 0 spills for every single-pass tail — i.e. **non-accumulator demand ~= 190
  regs**; accumulator perturbations of +-64 regs decide everything. (Both knobs are
  registered tuner dims; they trade HBM traffic/MMA overlap and are NOT proposed as
  fixes — they are the attribution instrument.)
* **The 15 KB SAM-cell spills are the duplicated engine body.** LookSAM/SG11/SG15/SG2
  inline the full tile fwd+bwd + dW machinery TWICE (P1 + the P2.4 SAM second pass);
  at IL=1 (64 freed regs) the duplication costs only ~0.4 KB — the doubled body times
  the accumulator margin is exactly the 15 KB.
* **fp32 weights converted on read** (`__float2bfloat16(W[...])` in the fwd/dX staging
  accessors) are the cp.async blocker (a) — an async copy cannot convert. Isolated-
  function probes show the pure-bf16 dW path allocates spill-free where fwd/dX spill
  1.2-2.3 KB when outlined; HOWEVER an A/B of the *same probe* with the bf16 cache
  shows no measurable per-fn spill delta — the conversion web is NOT a register-
  pressure owner at compile-visible scale (claim tested and retracted; see C1 below).
* **SASS audits** (cuobjdump, sm_90a): no STL/LDL ever lands inside an
  HGMMA..WARPGROUP.DEPBAR in-flight window in ANY measured build — ptxas protects the
  wgmma window; the spill cost is bandwidth, not the mamba-documented determinism
  hazard (which only ever fired with the accumulator forced out, per the in-tree note).
* Hazard note: `__noinline__` on the big tile fns WITHOUT also outlining/handling the
  GEMM web ICE'd nvcc 12.4 cicc (segfault) in one combination (V2 probe). The shipped
  patch shapes below all compile clean; avoid that intermediate shape.

## 3. CANDIDATES (implemented; ranked; patches in this directory)

### #1 `0002-decoder-sam-scoped-outline.patch` (+ `0002s-...-standalone.patch`) and `0003-vit-sam-scoped-outline.patch` — SAM-cell-scoped out-of-line tile shims
The mamba precedent (one shared `__noinline__` frame for the twice-instantiated tile
body), **scoped via `if constexpr` so only the SAM-coupled cells take the ABI
boundary**. Single-pass cells keep the inline bodies — their SASS is opcode-identical
to HEAD (verified instruction-by-instruction on decoder AdamW; only link addresses
shift).

decoder TOTAL spill stores (prod d=128 / bench d=1024):
| cell | HEAD | + patch | delta |
|---|---|---|---|
| 7 single-pass cells | 0 / 0 | **0 / 0 (SASS-identical)** | none |
| LookSAM  | 15252 / 15028 | **8004 / 8140** | -48% |
| SG11     | 15252 / 15028 | **7932 / 8140** | -48% |
| SG15     | 15252 / 15028 | **8004 / 8140** | -47% |
| SG2      | 15424 / 15264 | **8588 / 8416** | -44% |
Entry-level (the megakernel hot body): 15.3K -> 3.4-3.7K (-76..78%).

vit TOTAL spill stores (prod / bench): LookSAM/SG11/SG15 15020/15364 -> **12408/12536**
(-17%), SG2 18288/19036 -> **12640/12832** (-31%); entry-level 15.0-18.2K -> 6.6-6.8K
(-56..63%). 7 light cells unchanged (0-8 B, same as HEAD).

Numerics: same code, same fp expression order (ascending-k preserved); the SAM cells'
instruction *scheduling* changes (any optimization does); single-pass cells bit-exact
at the SASS level. Warpgroup-uniform call; wgmma choreography inside the callee is the
already-silicon-validated mamba pattern. GPU parity + A/A/A gates still required
before shipping (designed to pass; no math change).

### #2 `0004-mamba-scope-noinline.patch` — scope mamba's existing unconditional outline
HEAD mamba taxes EVERY cell with the out-of-line frame. Scoping it to the SAM-coupled
cells returns the 7 single-pass cells to inline allocation:
| cell | HEAD TOT sp_st (prod/bench) | + patch | delta |
|---|---|---|---|
| AdamW          | 5848 / 6760 | **1032 / 996** | -82/85% |
| Lion/Grokfast/NeuralGrok/Muon | 5848-5852 | **1032** | -82% |
| GrokAdamW      | 5692 / 6596 | **1032 / 996** | -82% |
| Prodigy        | 5704 | **1044** | -82% |
| LookSAM/SG11/SG15/SG2 | 9452-10180 / 10356-11164 | **unchanged** | 0 |
SASS audit of the new inline AdamW kernel: 0 STL/LDL inside any HGMMA in-flight window
(the in-tree determinism-hazard note was specifically about the accumulator being
forced out in the double-buffer window; it does not occur). RISK FLAG: this reverts
the single-pass cells to the pre-fix inline topology (which the original fix note says
"stayed under it") — it must clear the mamba A/A/A + fp64-oracle gates first, and the
residual 1.0 KB entry spill should be watched. Production mamba currently ships ~5.8 KB
per step of callee spill traffic on its RACE cells (adamw/lion/...) — this patch is the
single biggest spill-byte reduction available per cell-count.

### #3 `0001-decoder-bf16-weight-prestage.patch` — bf16 weight cache (cp.async ring enabler)
Converts the 8 per-layer GEMM weight matrices fp32->bf16 ONCE per step into a
workspace carve (787 KB at d=128, 50 MB at d=1024; workspace single-source updated
host+kernel, both layout branches), staged by the fwd/dX GEMMs as a pure bf16 copy.
`cache[i] = __float2bfloat16(params[i])` is the identical deterministic rounding the
on-read path performed -> **operand values bit-identical by construction**; the SAM
second pass re-converts after the perturb behind one added grid barrier (SAM steps
only); element-owned writes, no atomics (A/A/A-safe shape).
* Spill table: 7 light cells 0 -> 0 at both layouts; SAM cells 15252 -> 15268 (+16 B,
  the convert sweep's tail in an already-spilling body; the +16 B vanishes when 0002
  lands on top — the intended composition, measured: `dec_C1S*` tables).
* HONESTY: the hoped-for compile-visible register relief did NOT materialize (A/B of
  isolated GEMM fns: spill profile unchanged). The candidate's value is structural:
  it **eliminates cp.async ring blocker (a)** (the dominant streamed operand is now a
  flat bf16 buffer, 16-byte-chunkable, rows 16B-aligned at both layouts) and halves
  fwd/dX weight-read HBM traffic (fp32->bf16 reads; perf claim is GPU-gated).
* Ranked #3 ONLY because its payoff is gated on the ring/GPU work; as a pressure
  patch alone it is neutral.

Composed series (`0001`+`0002`): decoder light cells 0 spills, SAM cells 7932-8588
(prod) / 8140-8416 (bench). Series + `0003` + `0004` apply cleanly to HEAD in order
(verified `git apply` + full launcher recompile of the composed tree: rc=0).

Compile-clean proof for every patch: the model's TC launcher TU (all 10-11 OptId
instantiations), the scalar sibling cell, the JIT TC cell TU, and
`decoder_tc_selftest.cu` all compile rc=0 with production flags at BOTH layout
branches (logs: `dec_C1S*`, `vit_S*`, `mb_S*`, `*_scalarcell`, `*_jitcell`,
`*_selftest`, `dec_SERIES_composed`).

## 4. RING HEADROOM VERDICT (cp.async mbarrier ring, decoder)

The morning report's three blockers, post-campaign:
* **(a) fp32-convert-on-read — RESOLVED by 0001.** The fwd/dX B-operand is a flat,
  contiguous, 16B-aligned bf16 region; cp.async (or TMA with a 1D descriptor) can
  stream it as-is. Region sizes are multiples of 8 elements at d=128 AND d=1024
  (derived from layout constants, no hardcoding).
* **(c) zero register headroom — MARGIN CREATED AND NOW MEASURABLE.** The ring's
  in-loop state is ~8-16 registers (slot index, mbarrier phase, prefetch pointers;
  mbarrier objects live in smem, descriptors are remat-cheap). Evidence the engine can
  absorb it: (i) the single-pass cells absorb a **64-register** accumulator
  perturbation (IL 1<->2) while staying at 0 spills — the allocator has remat slack an
  order of magnitude larger than the ring needs; (ii) the ring REPLACES the
  synchronous staging loop, retiring the staged-element/index registers it currently
  holds (under `--extra-device-vectorization` that window is ~8-16 wide), so the net
  K-loop delta is ~0; (iii) the SAM cells no longer flood the signal — with 0002 the
  compile gate "0 spills on single-pass cells, <=8.6 KB on SAM cells" is a clean
  regression detector for the ring patches (at HEAD the 15 KB noise made the ring's
  own spill contribution invisible — likely the "2.6 KB spills, no room" observation).
  Watch metric during ring bring-up: TOTAL spill stores per cell (parse2.py), not
  "Used".
* **(b) dW transposed-strided operands — NOT ADDRESSED (out of scope).** The dW
  GEMMs read dY/X transposed; cp.async cannot transpose. Scope the ring to fwd/dX
  first (they are the B0/P1 critical path); dW keeps synchronous staging until a
  TMA-with-transpose descriptor design lands.
* smem budget: DecTcSmem is 43.2 KB of 227 KB; the double-buffer ring slots ALREADY
  exist in smem (kDecTcStages=2); adding 2 mbarrier pairs costs 32 bytes. No smem
  blocker.

Verdict: with 0001 (+0002 for signal hygiene) applied, the decoder ring is unblocked
on (a) and (c); (b) is the remaining structural item and is avoidable by scoping.

## 5. Files
* Patches: `0001-decoder-bf16-weight-prestage.patch`,
  `0002-decoder-sam-scoped-outline.patch` (series, on top of 0001),
  `0002s-decoder-sam-scoped-outline-standalone.patch` (same change vs pristine HEAD),
  `0003-vit-sam-scoped-outline.patch`, `0004-mamba-scope-noinline.patch`.
* Harness: `compile_one.sh` (production-flag single-TU compiler),
  `parse_ptxas.py` (per-kernel), `parse2.py` (entry+callee totals).
* Raw ptxas logs for every number above: `logs/*.log` (39 builds).
* No commits were made; the shared tree was never edited (worktrees under /dev/shm).

## 6. Standing-rules checklist
* No functionality suppression: no path disabled; SAM second pass, all 11 tails intact.
* No maxrregcount caps (and they are inert here anyway — documented).
* Perf-shaped constants: none introduced; the only knobs touched in probes are
  existing SG_TUNED_* dims, restored to defaults in the patches.
* No problem-specific hardcoding: all new sizes derive from layout constants
  (verified by the d=1024 branch compiling with correct scaled sizes).
* GPU gates still owed (other lane / later): per-cell fp64-oracle parity, A/A/A
  bit-determinism, tail_gate, wiring_check, and perf measurement for 0001's traffic
  claim and 0002/0003/0004's spill-traffic savings.
