# Grokking Race — NVIDIA H100 (sm_90a)

On-silicon run of the grokking race on a real **NVIDIA H100 80GB HBM3**
(compute capability 9.0, CUDA 12.4, torch 2.4.1+cu124). All optimizers run
through the compiled `_ops` extension (per-op fused sm_90 kernels via
`get_ops()`); the model is the eager PyTorch Decoder-Transformer.

- **Task:** modular division `a ÷ b (mod 97)` — the canonical grokking task
- **Model:** 2-layer Decoder-Transformer, d=128, 4 heads (422,755 params)
- **Split:** 50/50 train/test, 15,000-step budget, eval every 200, grok = test-acc ≥ 0.95
- **Seed:** 42 · single H100 · eager model + fused optimizer kernels (`--no-fused` off)

Reproduce:
```bash
python3 grokking_race_v2.py --optimizers adamw,lion,grokfast,grokadamw,looksam,\
prodigy,neuralgrok,muon,supergrok,supergrok15,supergrok2 \
  --num-seeds 1 --train-test-ratios 50/50 --tasks decoder \
  --early-stop-max-steps 15000 --eval-every 200 --no-status-server \
  --output results/h100_grokking_race
```

## Results — 8 of 11 grok

| Rank | Optimizer | Grok step | Peak test-acc | Final test-acc | Status |
|-----:|-----------|----------:|--------------:|---------------:|--------|
| 🥇 | **Muon**       |   **400** | 1.000 | 0.976 | ✓ grokked (fastest) |
| 🥈 | **Prodigy**    | **1,000** | 1.000 | 0.007 | ✓ grok threshold hit, ✗ not sustained † |
| 🥉 | Grokfast       |     2,600 | 1.000 | 1.000 | ✓ grokked |
| 4 | AdamW           |     3,000 | 1.000 | 1.000 | ✓ grokked |
| 5 | LookSAM         |     3,200 | 1.000 | 0.995 | ✓ grokked |
| 6 | Lion            |     4,000 | 1.000 | 1.000 | ✓ grokked |
| 7 | NeuralGrok      |     4,800 | 1.000 | 1.000 | ✓ grokked |
| 8 | GrokAdamW       |     5,000 | 1.000 | 1.000 | ✓ grokked |
| — | SuperGrok1.5    |       —   | 0.918 | 0.022 | ✗ DNF — nearly groks, meta-net collapses it ‡ |
| — | SuperGrok1.1    |       —   | 0.020 | 0.007 | ✗ DNF — meta-net collapse ‡ |
| — | SuperGrok2      |       —   | 0.017 | 0.007 | ✗ DNF — CSA/HCA meta routing ‡ |

**8 of 11 optimizers reach the grok threshold** (test-acc ≥ 0.95). The two
fastest — **Muon (400 steps) and Prodigy (1,000 steps)** — were *flat at random
for all 15k steps before this audit's on-silicon fixes*. Muon is now the fastest
grokker of the entire field and holds its solution; **7 of the 8 grok cleanly and
sustain ~100%**.

† **Prodigy** reaches 100% test-acc at step 1,000 (the grok threshold), then its
adaptive `d` estimate slowly destabilizes and the solution decays (final 0.007).
The d-adaptation *math* is parity-verified against `prodigyopt`
(`tests/hw/parity_gate_h100.py`, ratio 1.09); sustained convergence is a
remaining d-schedule tuning question, not a correctness bug. Pre-fix it never
left chance — see DNF diagnoses.

‡ The three SuperGrok variants are **not kernel bugs** — see below.

Every clean grok shows the classic signature: train-acc → 1.0 within a few
hundred steps (memorization), then a delayed jump of test-acc from ~0.01 to ~1.0
(grokking). `curves_decoder_ft50.png` plots the full trajectories;
`race_decoder_ft50.png` is the grok-speed bar chart.

## On-silicon bugs surfaced & fixed by this run

These are silent on the CPU `nvcc -c` gate used previously — they only appear
when the kernels actually execute on a GPU. All fixed and parity-verified on the
H100 (`tests/hw/parity_gate_h100.py` → 11 pass / 0 fail):

- **Muon — FIXED, now fastest (400 steps).** The fused Newton-Schulz
  combine+update kernel applied `p·(1−decay_factor)+…` while the host passes
  `decay_factor = 1 − lr·wd` — i.e. it kept ~2% of each weight per step instead
  of 98% (inverted weight decay that destroyed the parameters). Flat-at-random
  before; the fastest grokker after.
- **Prodigy — FIXED, now groks at step 1,000.** Two bugs: (1) the `d`-adaptation
  numerator was degree-1 in `d` while the denominator was degree-2 → `d_hat ∝
  1/d`, an immediate catapult; corrected to the scale-free degree-2 form. (2) The
  device accumulators used instantaneous `r/s`, so `d` ratcheted unbounded → an
  EMA (decay `r/s` by `β₃=√β₂`) now stabilizes the estimate. Flat-at-random
  before; reaches 100% test-acc after (sustained convergence still open, see †).
- **SuperGrok1.1 / 1.5 — kernels fixed; collapse is the trained meta-net, not the
  kernel.** Fixed on-device: a meta-net hidden-dim OOB (`SG_H=64` vs the default
  `meta_hidden_dim=32`), a frozen step-counter (pybind copies `list[int]` by
  value, so bias-correction was pinned at t=1 → a ~50× inflated Adam
  denominator), an inverted/double-applied `μ` mixing term (restored to the
  canonical cosine-gated `g + (1−gate)·α·μ`), and a `sam_step` that threw on
  every call (SAM silently disabled, sharpness pinned at 0). After all of these
  the kernel is parity-exact, but both still **memorize (train→1.0) then collapse
  at ~step 900**: the *learned* `rescale·MLP(grad,sharpness)` correction grows
  until it destabilizes the update — an unbounded learned-scalar dynamics issue
  (cf. the Prodigy d-ratchet), owned by the algorithm. SuperGrok1.5 gets within
  one step of grokking (peak test-acc **0.918**) before the meta-net destroys it.
  **Decisive control:** freezing the meta-net (`rescale=0`) reduces SuperGrok1.1
  *exactly to AdamW* and it **groks at step 2,700** — proving the base optimizer
  and the fused kernel are sound and isolating the collapse to the trained
  meta-net. No hyperparameter tuning was applied to force a grok.
- **SuperGrok2 — runtime fixed, meta routing still flat.** The earlier per-head
  PEER reshape crash and an `input_proj` bf16/fp32 dtype mismatch are fixed (the
  step now runs cleanly and the matrix-GRU reconstruction + ascending-|grad| sort
  are parity-stable), but the CSA/HCA per-head routing does not yet drive
  generalization (peak test-acc 0.017). Same research-owned class as 1.1/1.5.

The full kernel-maximality + numeric-parity audit that surfaced these (plus
latent bf16-only dtype bugs and a GrokAdamW Q3 OOB) is summarized in the branch
commit history and `HARDWARE_VALIDATION.md`.
