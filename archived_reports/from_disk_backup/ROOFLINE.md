# Roofline — REAL megakernel paths vs the H100 ceiling (fp32-forced)

Owner directive: **distance-to-roofline is the optimization metric**, measured on
the **REAL megakernel paths**. The previous chart wrongly measured eager+L1 for the
adamw cells — the decoder auto-default is bf16, and the L3 precision gate in
`grokking_race_v2._try_fused_train_step` declines the persistent megakernel unless
the resolved precision is fp32. This pass forces `matmul_precision=fp32` so
decoder/vit/mamba × adamw run the TRUE L3 persistent fwd+bwd+opt megakernel, AND
directly measures the validated bf16 tensor-core (TC) cells.

Data: `roofline.json`; plot: `roofline.png`; harness: `tuning/roofline.py`.

## ⚠️ Measurement-integrity incident (read first)
The task premise was "the GPU is quiet (fleet SIGSTOPped)". It was **NOT quiet**:
1. The SIGSTOPped tuner fleet (`tuning.tune_optimizers --model {vit,mamba}`, PIDs
   449297/449483) was found **resuming ~every 30 s** (an orchestrator outside
   process visibility keeps `SIGCONT`-ing them; PPID=1, no cron/visible supervisor).
   A watchdog re-`SIGSTOP`s them within ~1 s of each resume.
2. A **second `python -m tuning.roofline` (PID 481225) I did NOT launch** was found
   running concurrently, holding ~18.7 GB and ~50 % util *continuously* — no quiet
   window. It (and a concurrent agent) also edited `tuning/roofline.py` /
   `grokking_race_v2.py` and moved git HEAD during this session. **It will overwrite
   the repo `roofline.json`/`roofline.png`.** Canonical-output / provenance is for
   the owner to reconcile; **my clean results are durable in `/tmp/roofline_refined.json`
   and in the tables below.**

Contention only ever SLOWS, and it hits the cooperative-grid persistent megakernel
hard (one CTA/SM grid — cannot yield to other processes) while eager cuBLAS is
hurt much less:
- decoder adamw **L3** megakernel: **5.4 st/s contended → 11.8 st/s quiet** (~2.2×)
- decoder **TC** megakernel: **68.2 ms contended → 28.7 ms quiet** (~2.4×)
- decoder **lion** eager+L1: **57.4 contended ≈ 57.1 quiet** (robust); but
  **prodigy 49.3 contended → 61.3 quiet (+24 %)** — fast GEMM-bound eager rows ARE
  somewhat contention-sensitive.

**Handling:** the **9 megakernel rows** (adamw-L3 ×3 + TC/scalar-mega ×6) — the
REAL paths that are the point of this task — were re-measured **min-of-3 wall** on
the guarded GPU (tightly consistent reps, e.g. vit-L3 3.62/3.64/3.64; decoder-TC
28.7/28.9/28.7), so they are CLEAN. The **30 eager rows are LOWER BOUNDS** (PID
481225 made a quiet eager pass unobtainable): clean eager would be *faster*, so the
eager-vs-L3 ratios below **understate** eager's advantage — the qualitative
conclusion (eager ≥ L3 at this scale) only gets stronger clean. **The prior shipped
chart's contention state is unknown.**

## Methodology (ncu SOL container-blocked — ERR_NVGPUCTRPERM; substitutes stated)
- **achieved FLOP/s** = FLOPs/step ÷ measured wall/step. Wall: the REAL race train
  fn, fp32-forced, `use_fused=True` (so the L3/TC megakernel IS what's timed),
  discard-warmup then two run-lengths differenced (per-run setup cancels),
  CUDA-synchronized.
- **FLOP-trap fix (critical):** `torch.profiler(with_flops=True)` counts only
  registered aten GEMMs — a fused megakernel is ONE opaque kernel registering ZERO
  FLOPs. The FLOP pass therefore sets `use_fused=False` so the profiler sees the
  real eager GEMMs. FLOPs/step is path- AND dtype-independent, so this is the
  correct count for the L3 wall. Verified: decoder adamw fp32 profiled 2.23 GF/step
  with `use_fused=True` vs **42.3 GF/step** with `use_fused=False` (~19×).
- **L3-fired verification:** each row records the path that ACTUALLY executed (L3
  counter + `_FUSED_ABI_STALE`). adamw rows show `l3_fired=full`, `abi_stale=False`.
- **arithmetic intensity** = FLOPs/step ÷ analytical bytes/step. TC rows store
  activations in bf16 (2 B) → reported with a TC-faithful AI AND the fp32-model AI;
  TC points land memory-bound in this model, so their *ceiling/fraction* depend on
  the byte assumption — **achieved TF/s (the honest axis) does not.**
- **ceiling** = min(compute peak, AI × 3.35 TB/s). Engine-driven: scalar megakernel
  + eager fp32 → FP32 CUDA-core **66.9 TF**; bf16 TC → **989 TF**.
- **TC cells** (`mega_{decoder,vit,mamba}_real_adamw_tc.cu`) JIT-loaded exactly like
  their gates; both `tc_train_step` (bf16) and `scalar_train_step` (fp32) timed at
  the same B (full batch truncated to B%16, the TC dW 16-step-atom requirement),
  `ncta_cap=0` = one CTA/SM = the shipped saturating config.

## Headline — same-precision (fp32) absolute throughput
Tiny model (d=128, seq=4, ~0.42 M params) → every GEMM is small → **roofline-hostile
for every path**. Even the best eager row sits at single-digit % of the fp32 roof.
**Megakernel rows are CLEAN (min-of-3 quiet); eager rows are contended LOWER BOUNDS
(clean would be faster → ratios understate eager).** All read against the engine's
ceiling: scalar/eager fp32 → 66.9 TF; TC bf16 → 989 TF.

| model | L3-scalar (adamw, fp32, CLEAN) | TC (adamw, bf16, CLEAN) | scalar-mega (fp32, CLEAN) | eager+L1 best (fp32, lower-bnd) | eager ÷ L3 |
|---|--:|--:|--:|--:|--:|
| decoder | 11.8 st/s · 0.499 TF/s | **34.8 st/s · 1.467 TF/s** | 12.0 st/s | lion 57.4 st/s · 2.429 TF/s | ≥ 4.9× |
| vit | 3.6 st/s · 0.676 TF/s | **6.1 st/s · 1.135 TF/s** | 3.8 st/s | lion 53.9 st/s · 10.0 TF/s | ≥ 14.8× |
| mamba | 16.5 st/s · 0.855 TF/s | **29.2 st/s · 1.504 TF/s** | 15.7 st/s | lion 20.3 st/s · 1.051 TF/s | ≥ 1.2× |

**TC (bf16 tensor-core) is the FASTEST megakernel path on all three models**, and on
**mamba TC (1.504 TF/s) BEATS mamba eager (1.051 TF/s)** — a genuine megakernel win
(the scan-heavy mamba step has enough non-cuBLAS work that the fused TC path pulls
ahead). TC-vs-scalar-megakernel speedups (back-to-back, same process, contention-
robust ratio): **decoder 2.71×, mamba 1.86×, vit 1.64×** (all TC-wins).
Full data: durable `/tmp/roofline_refined.json` (repo `roofline.json` may be
overwritten by the concurrent PID 481225 run).

## Reading
1. **The L3 scalar megakernel is SLOWER than eager at this scale** (decoder eager
   ≥ 4.9× the L3-scalar megakernel at identical fp32; vit ≥ 14.8×; mamba ≥ 1.2×).
   One CTA/SM grid-strided over the full batch cannot beat cuBLAS's batched
   small-GEMM scheduling when the model is this tiny. This is the honest negative
   result the no-suppression directive explicitly accepts — the L3 path is *correct*
   (gates green, real fwd+bwd+opt in one persistent kernel) but not yet a throughput
   win here; closing the gap is a kernel redesign (batch-tiling / multi-CTA-per-
   tensor), not a tune. (Ratios are LOWER BOUNDS — clean eager is faster.)
2. **TC (bf16) is the fastest megakernel path, and beats eager on mamba.** TC beats
   the scalar megakernel everywhere (decoder 2.71×, mamba 1.86×, vit 1.64×) AND
   beats *eager* on mamba (TC 1.504 vs eager 1.051 TF/s). On decoder/vit TC still
   trails eager in absolute TF/s, but by far less than scalar does. Note TC's
   roofline *fraction* reads LOWER than scalar's despite being faster, because the
   bf16 ceiling (989 TF) is ~15× the fp32 ceiling — **rank megakernel paths by
   absolute TF/s, not fraction.**
3. **fp32-vs-old-bf16 deltas confound path + precision + ceiling.** The adamw rows
   genuinely changed PATH (eager+L1 → real L3); the other optimizers only changed
   the *ceiling label* (bf16 ceiling → fp32 ceiling), so a higher "fraction" there
   is NOT higher efficiency — achieved TF/s is unchanged. See `deltas_vs_eager`
   (`path_changed`, `ceiling_ratio`).
4. **ncta / MIN_BLOCKS (task 3):** `ncta_cap=0` (one CTA/SM) is already the maximal
   saturating config — these are cooperative grid-sync kernels, so CTAs must be
   co-resident; you cannot exceed one-per-SM without deadlocking the barrier.
   MIN_BLOCKS would require a recompile + full re-validation and would *cut*
   registers on a register-hungry kernel (likely harmful). No cheap knob closes the
   ~5× gap. The real task-3 content is the per-model scalar-vs-TC verdict above
   (mamba: scalar ≈ TC, scan-dominated, matching the prior 0.46× finding).

## Levers toward the roof (expected-impact order)
- **Batch-tile / multi-CTA-per-tensor megakernel redesign** — the single-CTA-per-SM
  grid-stride is the bottleneck; a cooperative tiling that fills SMs like cuBLAS is
  the structural fix (the eager path proves the FLOPs are reachable).
- **TC everywhere it wins** — TC already beats scalar; the gap to eager is the same
  saturation problem, not precision.
- **Run the race at a realistic model size** — at d=128/seq=4 the kernel-launch and
  occupancy overheads dominate; the megakernel's launch-elision advantage only pays
  off when the per-launch GEMMs are large enough to matter.
