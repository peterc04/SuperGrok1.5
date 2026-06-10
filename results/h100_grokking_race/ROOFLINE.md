# Roofline — every race pipeline vs the H100 ceiling

Owner directive: **distance-to-roofline is the optimization metric** (not watts),
tracked at 10-step granularity. Full data: `roofline.json`; graph: `roofline.png`;
harness: `tuning/roofline.py` (re-run after any kernel/build change).

## Methodology (ncu SOL is container-blocked — ERR_NVGPUCTRPERM; substitutes stated)
- **achieved FLOP/s** = profiler-measured FLOPs/step ÷ measured wall/step on the
  REAL race train functions: discard-warmup run, then two run lengths differenced
  (per-run setup cancels) with CUDA-synchronized walls; FLOPs from
  `torch.profiler(with_flops=True)` over 20 steps (GEMM-registered; elementwise
  ops register 0 → slight undercount, conservative for "how close to roof").
- **arithmetic intensity** = FLOPs/step ÷ analytical bytes/step (weights fwd+bwd,
  grads w+r, optimizer state r+w per per-optimizer state-tensor counts,
  activations fwd-write + bwd-read; model in `tuning/roofline.py:bytes_per_step`).
- **ceiling** = min(compute peak, AI × 3.35 TB/s HBM3). Eager fp32 matmul with
  `allow_tf32=False` (this build) → compute peak = **FP32 66.9 TF/s**; TF32
  (494.7) and BF16 (989.4) lines are drawn for reference (reachable only by
  changing matmul precision policy — a science decision, not free).
- A per-10-step wall series (the race's own tracking cadence, eval included,
  labelled) is stored per pipeline in `roofline.json`.

## Results (achieved / attainable ceiling at the pipeline's AI)

| pipeline (best→worst per model) | steps/s | GF/step | achieved TF/s | % of roof |
|---|--:|--:|--:|--:|
| **vit** — adamw | 57.8 | 243 | 14.07 | **21.0%** |
| vit — grokfast | 53.9 | 243 | 13.12 | 19.6% |
| vit — prodigy / grokadamw / lion / looksam / muon / sg15 / sg11 | 28–49 | 243–319 | 9.0–11.9 | 13.5–17.7% |
| vit — neuralgrok | 28.4 | 243 | 6.90 | 10.3% |
| vit — **supergrok2** | 5.6 | 258 | 1.45 | **2.2%** |
| **decoder** — grokadamw | 94.3 | 55.6 | 5.24 | **7.8%** |
| decoder — adamw / grokfast / prodigy / lion | 64–75 | 55.6 | 3.5–4.2 | 5.3–6.2% |
| decoder — sg15 / looksam / neuralgrok / muon / sg11 | 34–51 | 56–73 | 2.5–3.0 | 3.7–4.5% |
| decoder — **supergrok2** | 5.9 | 60.8 | 0.36 | **0.53%** |
| **mamba** — all non-SG | 16–21 | 68–81 | 1.3–1.4 | ~2.0% |
| mamba — sg11 / sg15 / neuralgrok | 12–14 | 68–89 | 0.8–1.3 | 1.2–1.9% |
| mamba — **supergrok2** | 6.6 | 72.6 | 0.48 | **0.71%** |

## Reading
1. **Every pipeline is compute-ceiling-bound at its AI** (AI 57–162 ≫ the
   FP32 ridge ≈ 20 FLOP/B) — HBM bandwidth is NOT the limiter anywhere. The
   distance to roof is **launch/latency-bound eager execution**: many small
   GEMMs (40–70 µs) + per-step optimizer kernel chains leave SMs idle between
   kernels. This is the quantitative version of the earlier power finding
   (~235 W ceiling): util gaps, not register pressure or bandwidth.
2. **Model dominates the roofline position**: vit (16-patch attention → bigger
   GEMMs) reaches 10–21%; decoder 4–8%; mamba 1–2% (sequential scan, smallest
   kernels). Optimizer choice modulates within the model band — the optimizer's
   step cost is pure overhead against the roof.
3. **SuperGrok2 is the farthest from the roof on every model** (0.53% decoder /
   2.2% vit / 0.71% mamba): its step launches ~20 small kernels (csa/hca topk ×
   3.5 ms, index_select ×306, mul ×530 …) — 10–18× slower than peers. This is
   the #1 roofline-recovery target among optimizers.
4. **Levers, in expected-impact order** (each re-measured by re-running
   `python -m tuning.roofline`):
   a. **Wire the L3 fused megakernels into the race** (task #22, owner-chosen):
      one persistent kernel per step removes the launch gaps — the largest
      structural move toward the roof; needs the C2 full-state plumbing.
   b. **Autotuner→build linkage** (task #23): tuned block/vec/unroll/maxrregcount
      currently never reach `_ops.so` (verified decorative); landing them is
      free single-kernel headroom.
   c. **SG2 kernel-chain fusion/CUDA-graphs** + the suspected kernel-race fix
      (compute-sanitizer probe pending — SG2 is also nondeterministic run-to-run).
   d. **Precision policy** (TF32/BF16 matmul) would raise the *ceiling* 7.4–14.8×
      — but changes the race's numerics; owner decision, not assumed.
