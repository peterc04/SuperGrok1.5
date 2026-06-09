# Grokking Race — NVIDIA H100 (sm_90a)

First on-silicon run of the grokking race on a real **NVIDIA H100 80GB HBM3**
(compute capability 9.0, CUDA 12.4, torch 2.4.1+cu124). All optimizers run
through the compiled `_ops` extension (per-op fused sm_90 kernels via
`get_ops()`); the model is the eager PyTorch Decoder-Transformer.

- **Task:** modular division `a ÷ b (mod 97)` — the canonical grokking task
- **Model:** 2-layer Decoder-Transformer, d=128, 4 heads (422,755 params)
- **Split:** 50/50 train/test, 15,000 step budget, eval every 200, grok = test-acc ≥ 0.95
- **Seed:** 42 · single H100 · `--no-fused` path off (eager model + fused optimizer kernels)

Reproduce:
```bash
python3 grokking_race_v2.py --optimizers adamw,lion,grokfast,grokadamw,looksam,\
prodigy,neuralgrok,muon,supergrok,supergrok15,supergrok2 \
  --num-seeds 1 --train-test-ratios 50/50 --tasks decoder \
  --early-stop-max-steps 15000 --eval-every 200 --no-status-server --output results/
```

## Results

| Rank | Optimizer | Grok step | Final test-acc | Status |
|-----:|-----------|----------:|---------------:|--------|
| 🥇 | **Grokfast**   | 2,600 | 1.000 | ✓ grokked |
| 🥈 | AdamW          | 3,000 | 1.000 | ✓ grokked |
| 🥉 | LookSAM        | 3,200 | 0.995 | ✓ grokked |
| 4 | Lion            | 4,000 | 1.000 | ✓ grokked |
| 5 | NeuralGrok      | 4,800 | 1.000 | ✓ grokked |
| 6 | GrokAdamW       | 5,000 | 1.000 | ✓ grokked |
| — | Prodigy         |   —   | 0.007 | ✗ did not grok |
| — | Muon            |   —   | 0.007 | ✗ did not grok → **fixed**, see below |
| — | SuperGrok1.1    |   —   | 0.007 | ✗ did not grok |
| — | SuperGrok1.5    |   —   | 0.218 | ✗ partial (started generalizing, ran out of budget) |
| — | SuperGrok2      |   —   |   —   | ✗ runtime error (multi-head PEER reshape) |

**6 of 11 optimizers grok**, Grokfast fastest at 2,600 steps. Every grokked run
shows the classic signature: train-acc → 1.0 within a few hundred steps
(memorization), then a delayed jump of test-acc from ~0.01 to ~1.0 (grokking).
`curves_decoder_ft50.png` plots the full train/test trajectories;
`race_decoder_ft50.png` is the grok-speed bar chart.

## DNF diagnoses (on-silicon bugs surfaced by this run)

These could only be found by running on a real GPU — they are silent on the CPU
gate (`nvcc -c`) used previously. Status as of this commit:

- **Muon — FIXED & VERIFIED.** The fused Newton-Schulz combine+update kernel
  (`muon_sm90.cuh`) applied `p·(1-decay_factor) + …` while the host passes
  `decay_factor = 1 - lr·wd`, i.e. it retained ~2% of each weight per step
  instead of 98% — inverted weight decay that destroyed the parameters. The
  non-fused path used the correct `p·decay_factor + …`. Corrected to match the
  canonical `muon.h` formula. **Post-fix re-run on H100: Muon groks at 600
  steps** (val-acc 0.95) — flat-at-random before, the *fastest* grokker after.
- **Prodigy.** The adaptive `d_lr` stays pinned at its 1e-6 init (flat at random
  for all 15k steps), so the effective LR never grows. The device d-update
  matches the canonical paper formula; the failure is a bootstrapping/numerical
  interaction that needs on-device `d_lr` logging to pin down (open).
- **SuperGrok1.1 / 1.5.** Meta-net dimension mismatches between the Python
  config and the kernel's hard-coded hidden width (e.g. `SG15_H=64` vs the
  default `meta_hidden_dim=32`) corrupt the sharpness signal (open).
- **SuperGrok2.** The CSA/HCA meta-net's PEER routing assumes a single head, but
  the Python net stacks `num_peer_heads=4` parameters → `reshape({-1, 44})` on a
  192-element tensor raises at runtime. (A separate `input_proj` bf16/fp32 dtype
  bug on this path was already fixed.) Needs a per-head loop in the kernel (open).

The full kernel audit that surfaced these (plus latent bf16-only dtype bugs in
prodigy/supergrok11 and a grokadamw Q3 OOB) is summarized in the commit history.
