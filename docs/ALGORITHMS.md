# How the race's optimizers work

This is the rundown of the ten optimizers in the grokking race: what each one
does, what was wrong with it in the old code, how the new implementation is
checked, what it costs, and whether it can plausibly run at 7–10B parameters on
one H100 or at 552B on 64 H200s. SuperGrok 1.5 and SuperGrok 2 were removed; the
last commit that has them (and all the CUDA/HIP/TPU code) is `19c9d39`.

Every implementation lives in `grokking_optimizers/` as a plain
`torch.optim.Optimizer` (CPU or GPU). None of them was trusted because it runs:
each is compared, in `tests/`, against the published algorithm or the authors'
own code, and most match **bit for bit** (difference 0.0 after hundreds of
steps).

## At a glance

| | Idea | Optimizer state (fp32, bytes/param) | Extra forward+backward per step | Checked against (tests) | Old repo |
|---|---|---|---|---|---|
| **AdamW** | per-weight step size from gradient mean / RMS; decoupled weight decay | 8 | 0 | `torch.optim.AdamW`: bit-exact | formula right, but the class could not step; only a toy-shape CUDA kernel ran it |
| **Lion** | sign of a momentum blend; every weight moves by exactly `lr` | 4 | 0 | google/automl reference: bit-exact | formula right; allocated 3× the state it needed; the tuned β2 never reached it |
| **Grokfast** | add `λ ×` a slow EMA of the gradient before AdamW | 12 | 0 | authors' `gradfilter_ema` + AdamW: bit-exact | CUDA path right; AMD/TPU copies started the EMA at zero; v1 gave it weight decay 0.005 vs 1.0 for everyone else |
| **GrokAdamW** | Grokfast + per-tensor clip + per-tensor β1 decay + loss-driven EMA decay | 12 | 0 | pip `grokadamw` 0.1.2 (= QuixiAI/grokadamw): bit-exact | the CUDA kernel was a different algorithm (54% trajectory drift in 200 steps); the originals' signal switched the filter off |
| **LookSAM** | sharpness-aware gradient every k steps, its orthogonal part reused in between | 12 (+8 transient) | 1/k = 0.2 | paper Algorithm 1 transcription (< 1e-6); ρ = 0 is AdamW bit-exact; k = 1 is SAM | not LookSAM: AdamW on a stale `g + 2.33 (g_s − g)` correction |
| **Prodigy** | Adam with one global step size `d` estimated from distance travelled | 16 | 0 | `prodigyopt` 1.1.2: bit-exact (params and `d`) | not Prodigy: ℓ1 norm taken in the wrong place, ε not scaled by `d`, `lr` ignored |
| **NeuralGrok** | a small MLP reweights each gradient tensor; trained by a one-step bilevel lookahead | 8 + amplifier | ≈ +1 fwd+bwd every T = 4 steps, plus the amplifier on every gradient entry | official NeuralOptGrok code: bit-exact | not NeuralGrok: `α = 10, β = 4` were misread flags; it was AdamW with a sign-flip hazard; v1 disabled it, v2 never trained the amplifier |
| **Muon** (Kimi K3 per-head) | orthogonalize each weight matrix's momentum (Newton–Schulz); per-head blocks for attention queries | 4 (matrices), 8 (the rest, AdamW) | 0 (Newton–Schulz matmuls instead) | `torch.optim.Muon` semantics in fp32: < 1e-5 | could not step; sent embeddings and the output head through Newton–Schulz; no Nesterov |
| **Shampoo** (Meta's Distributed Shampoo) | precondition each weight matrix by the inverse 4th roots of its accumulated row and column gradient covariances, `L^(-1/4) G R^(-1/4)`; step length grafted from Adam | 8 + 2 × (rows² + cols²) per matrix block (factors and their roots) | 0 (eigendecompositions every 10 steps instead) | Meta's `distributed_shampoo` (commit `d24a149`): bit-exact in 7 configurations, fp32 and fp64 | not in the old code |
| **SuperGrok 1.1** | AdamW + a learned per-element correction φ(g, sharpness), cosine-gated, meta-trained by a one-step lookahead; layer-wise β1, clip, adaptive α | 12 + meta-net | SAM probe every 10 steps; meta step every 5 | naive transcription of the declared algorithm, all components on: < 1e-12; each component shown live; reductions to AdamW bit-exact | could only run in a CUDA kernel that dropped γ, the clip and adaptive α; its gate froze at 0.5; the learned correction collapsed training |

"Bytes/param" is optimizer state only; the parameters and gradients themselves
add 8 bytes/param in fp32.

## Race rules that apply to all of them

* **Same base.** Every Adam-family method uses the grokking recipe of Power et
  al. (2022): lr 1e-3, betas (0.9, 0.98), decoupled weight decay 1.0, full-batch
  gradient descent. Each method adds only its own mechanism, so a difference in
  the race is attributable to that mechanism. Lion (lr 3e-4, wd 3.0) and Prodigy
  (lr 1.0, which multiplies its learned `d`) keep their own step-size conventions.
* **Two splits, train and test**, exactly as in the original race scripts (a seeded
  shuffle; the first 50% trains, the rest tests). No validation split. Test data is
  only ever evaluated: no optimizer sees a test example, loss or accuracy.
* **Grokked** means **test accuracy ≥ 95% held for 50 consecutive evals** (500
  steps). A single spike does not count; the first crossing is recorded separately.
  Summaries use the median over *all* seeds, a seed that never grokked counting as
  never, so an optimizer that groks on one lucky seed out of three is a DNF.
* **Cost to grok is reported three ways:** gradient steps, training seconds
  (evaluation excluded, the same for everyone) and training FLOPs. FLOPs are
  measured with PyTorch's `FlopCounterMode` over everything an iteration runs:
  LookSAM's second pass, NeuralGrok's meta step and amplifier, Muon's
  Newton–Schulz, Shampoo's eigendecompositions (the counter scores those as
  zero; the race adds 9n³ each, Golub & Van Loan's count). It agrees with the textbook `6 × active params × tokens` to 0.6%
  for a plain step.
* **Held-out data for meta-learners.** NeuralGrok, GrokAdamW and SuperGrok 1.1 need a
  loss on examples they do not train on. They get a 10% slice carved out of *their
  own* training split and take gradients on the other 90%, as the original NeuralGrok
  race did. Every optimizer gets the same labelled examples; how it spends them is
  part of its method. `--same-train-data` makes every optimizer train on the same
  90% instead, for a data-matched comparison.
* **MoE load balancing is the same for everyone.** After each step the router's
  selection bias moves by ±0.001 toward balanced load, computed from the step's
  own training forward only (extra forwards inside SAM or meta steps are excluded).

---

## AdamW

**What it does.** Two running averages per weight: the gradient (`m`) and the
squared gradient (`v`). The step is `m̂ / (√v̂ + ε)`, so each weight moves about
`lr` times a signal-to-noise ratio of its own gradient. Weight decay is applied
separately, `θ ← θ(1 − lr·wd)`; that separation is the "W" (Loshchilov & Hutter).

```
m ← β1 m + (1−β1) g        v ← β2 v + (1−β2) g²
θ ← θ(1 − lr·wd) − lr · (m / (1−β1ᵗ)) / (√(v / (1−β2ᵗ)) + ε)
```

**For grokking** it is the baseline to beat: with wd = 1 it is exactly the
recipe that produced grokking in the first place, and weight decay drives the
"cleanup" phase (Nanda et al. 2023).

**At scale** it is the industry default (DeepSeek-V3 used it). Its only problem
is memory: 8 bytes/param of fp32 moments. A 7–10B model on one 80 GB H100 needs
bf16 moments and bf16 weights without an fp32 master copy.

**Old code.** The exported class raised `NotImplementedError` on `step()`. The
race ran a CUDA megakernel compiled for one 423K-parameter decoder, which only
read the first parameter group, silently dropped the last 0–15 training
examples, and kept its moments outside `state_dict()`.

## Lion

**What it does.** One momentum buffer. Each step moves every weight by exactly
`lr` in the direction `−sign(β1·m + (1−β1)·g)`, then updates the momentum with a
slower β2. Because every coordinate takes the same size step, it wants an lr
3–10× smaller and weight decay 3–10× larger than AdamW. With weight decay λ it
keeps every weight inside `|θ| ≤ 1/λ`.

```
c = β1 m + (1−β1) g
θ ← θ(1 − lr·λ) − lr · sign(c)
m ← β2 m + (1−β2) g
```

**For grokking**: reliable but slow in the old tuning (median 2,180 steps vs 314
for AdamW). **At scale** its value is memory: 4 bytes/param, the cheapest way to
fit fp32 state for a 7B model. LLM benchmarks show roughly AdamW quality, not a
speed-up.

**Old code.** The formula was right, but it allocated three state planes (12
bytes/param) and used one, erasing its only advantage; the tuner tuned a β2 the
race never passed.

## Grokfast

**What it does.** A filter in front of AdamW: keep a slow EMA of each gradient
and add it back amplified,

```
ema ← α·ema + (1−α)·g      (α = 0.98; seeded with the first gradient)
ĝ   = g + λ·ema            (λ = 2), then AdamW on ĝ
```

Slowly varying gradient directions get amplified up to 1 + λ = 3×; fast
alternating ones pass through almost unchanged. The paper's thesis is that the
generalizing signal lives in the slow component.

**Evidence.** The paper's "up to 50× faster" is measured against Adam with no
weight decay, which groks in ~40K steps. Against AdamW with wd = 1 most of that
speed-up is already delivered by weight decay; the old 5-seed tuning had
Grokfast 2.1× *slower* than AdamW. No evidence beyond ~1M-parameter models.

**Cost.** +4 bytes/param, no extra passes. At scale the EMA should be switched
off for sparsely touched tables (Engram, embeddings).

## GrokAdamW

**What it does** (E. Hartford's `grokadamw`, no paper). Grokfast feeding AdamW,
plus three heuristics, exactly as the published code does them:

1. each tensor's gradient is clipped to norm 1.0 on its own;
2. momentum decays by tensor index: `β1_i = β1·(1−γ)^i` with γ = 0.1, where `i`
   counts tensors with a gradient in order (a tensor index, not a layer index);
3. the EMA decay adapts to the train/held-out loss gap:
   `α = α₀·exp(−κ·max(0, eval−train)/max(eval, train))`.

**Known flaws in the published code** (reproduced by default, as it is the
published algorithm): the momentum uses `β1_i` but the bias correction uses the
global `β1`, inflating the first steps of deep tensors up to 10×
(`bias_correction1="layer"` fixes it). With ~290 tensors in the tiny DeepSeek
model, γ = 0.1 leaves `β1_i ≈ 0` for almost every tensor, so most of the model
trains with no momentum.

**Old code.** The CUDA kernel implemented a different algorithm (global clip,
per-tensor bias correction, first-gradient seed); your original race scripts
used the README's `(v−t)/t` signal, which drove α to 0 through the whole
memorization plateau and switched the Grokfast filter off.

## LookSAM

**What it does** (Liu et al., CVPR 2022). SAM steps with the gradient `g_s`
taken at the nearby worst-case point `w + ρ·g/‖g‖`, which costs a second
forward+backward every step. LookSAM pays that only every k-th step:

```
every k-th step:  g_s = ∇L(w + ρ g/‖g‖);  update with g_s;
                  cache g_v = g_s − (⟨g_s,g⟩/‖g‖²)·g      (the part orthogonal to g)
other steps:      update with g + α·(‖g‖/‖g_v‖)·g_v
```

Norms are over the whole parameter vector. ρ = 0.05, k = 5, α = 0.7, AdamW base.

**Evidence.** On ViT/ImageNet it matches full SAM (79.8 vs 74.7 top-1 for AdamW)
at 1.18× AdamW's time. No published evidence that it speeds up grokking; the old
tuning had it equal to AdamW (321 vs 314 steps) while paying 1.2× compute.

**Old code.** Not LookSAM: because Adam ignores a constant scale, the kernel's
`(1−α)·g + α·(g_s − g)` was exactly AdamW on `g + 2.33·(g_s − g)` with a stale
correction and no orthogonal projection.

## Prodigy

**What it does** (Mishchenko & Defazio, 2023). Adam whose step size `d` is
learned: it grows until the distance travelled from the initial point, weighted
by the gradients seen, says it is large enough.

```
m ← β1 m + (1−β1) d g      v ← β2 v + (1−β2) d² g²
r ← β3 r + (1−β3) d²⟨g, x0 − x⟩      s ← β3 s + (1−β3) d² g      (β3 = √β2)
d ← max(d, r/‖s‖₁)
x ← x − lr·d·m/(√v + d·ε)
```

**Evidence.** "Close to hand-tuned Adam", not better. With wd = 1 and a
constant lr (the race's regime) `d` can ratchet upward; the authors recommend
wd ≤ 0.1 with a cosine schedule. **Cost** 16 bytes/param (stores the initial
weights): it cannot fit a 7.5B model on one H100 in any regime.

**Old code.** Not Prodigy: the ℓ1 norm was taken per step instead of on the
accumulated sum, ε was not scaled by `d`, `lr` was never read, and a hard guard
could freeze `d` depending on the loss scale.

## NeuralGrok

**What it does** (Zhou, Fan, Jaggi, Fu, 2025). A small MLP, the *amplifier*,
looks at each gradient tensor's entries and reweights them:

```
p  = softmax(MLP(g)) over the entries of one tensor
g' = c·(p ⊙ g)/‖p ⊙ g‖          then global clip 1.0 and Adam
```

Every T = 4 steps the amplifier is trained by a one-step bilevel lookahead:
recompute the gradient, take a virtual SGD step through the amplifier, and
update the amplifier to lower the loss of that virtual step on held-out data.

**Your doubt is justified.** Measured by the analysis:

* the paper's evidence is one table on five toy tasks, no seeds or error bars,
  against a weak baseline (Adam + L2 1e-3); its own ablation shows plain
  per-tensor gradient normalization, *without* the learned amplifier, already
  gives most of the effect;
* the amplifier sees only a scalar (the gradient value): no tensor identity or
  history. What it learns is a 1-D reweighting curve, consistent with the
  paper's "surprisingly low transferability";
* at the paper's own scale the (128, 128) amplifier costs 2.8× the model's
  forward+backward per step;
* it does not scale as published: the meta step back-propagates through the
  amplifier applied to every gradient entry, about 2 KB per parameter, which is
  14.6 TB at 7B and 1.15 PB at 552B. An all-zero gradient (an MoE expert that
  got no tokens) gives 0/0 in the official code (fixed here).

It stays in the tiny race as a faithful reproduction. The paper's own control is
available as `amplifier_mode="identity"` (normalized Adam without the learned
part) to measure what the learned amplifier adds. It should not go to 7–10B.

**Old code.** `α = 10, β = 4` were misread command-line flags (they are
hidden-width multipliers in the official code), the amplifier was 1→16→1 on
`|g|`, and the result was AdamW except that in 17 of 200 initializations the
factor went negative (gradient *ascent*). Your v1 script set `neural_alpha = 0`
(so NeuralGrok was AdamW with wd 0.005); v2's amplifier had no gradient path and
never trained.

## Muon, Kimi K3 style

**What it does.** For each hidden weight matrix, keep a momentum buffer and
replace the update by an approximation of its polar factor `UVᵀ` (all singular
values set to ~1), computed with five Newton–Schulz iterations in bf16. The
update is rescaled so its RMS is 0.2, the same as AdamW's, so AdamW's lr and
weight decay carry over (Moonlight). Everything that is not a hidden matrix
(embeddings, output head, norms, sinks, mHC static terms, Engram tables and
gains) goes to an auxiliary AdamW.

```
M ← μ M + G                      U = G + μ M            (Nesterov)
O = NewtonSchulz(U)              per independent block
W ← W(1 − lr·wd) − lr · 0.2·√max(rows, cols) · O
```

**Kimi K3's Per-Head Muon** (arXiv 2607.24653 §2.5) orthogonalizes each
attention head's block of the Q/K/V projections separately, so a few
high-magnitude heads cannot dominate the shared update. On DeepSeek-V4.1's
attention this means:

* `wq_b` (queries) is split into its 64 heads (16 in the `h100` preset), and the
  indexer's `wq_b` into its index heads;
* K and V are one shared head (`wkv`), so there is nothing to split;
* `wo_a` is **always** split into its output groups: it stores independent
  matrices, and orthogonalizing the stack would be the wrong update (that is
  correctness, not a style choice);
* each block is scaled by its own shape; using the full matrix's shape would
  double the update RMS;
* every routed expert is orthogonalized on its own.

`per_head=False` turns the head split off for an ablation. A one-study caveat:
on GPT-2 Small, head-wise and whole-matrix Muon ended within 0.002 nats; K3 and
GLM-5 adopted it for stability at scale. QK-Clip (Kimi K2) is available as
`Muon.qk_clip` but off; it matters here because DeepSeek-V4.1 removed V4's
per-head query norm.

**Evidence.** The best-supported optimizer here: Moonlight, Kimi K2/K3, GLM-4.5/5
and DeepSeek-V4 itself were trained with it. Half of AdamW's state for matrices.
On an MoE at grokking batch sizes Newton–Schulz is a noticeable share of FLOPs,
which the race's FLOP count includes.

**Old code.** `step()` raised; the kernel had no Nesterov step and routed by
`ndim == 2`, sending the embeddings and output head through Newton–Schulz.

## Shampoo

**What it does** (Gupta, Koren, Singer, ICML 2018; made practical by Anil et al.
2020 and Shi et al. 2023). Adam gives every weight its own step size. Shampoo
instead preconditions each weight *matrix* as a whole: it keeps two running
covariances of the gradient `G`, one over rows and one over columns, and steps
along

```
L ← β2 L + (1 − β2) G Gᵀ          R ← β2 R + (1 − β2) Gᵀ G
direction = L^(-1/4) · M · R^(-1/4)          M = Adam's first moment (bias-corrected)
```

The race runs **Meta's Distributed Shampoo**, the implementation that won the
2024 MLCommons AlgoPerf training-algorithms benchmark (28% faster than the
baseline, external tuning), in its documented "replace Adam" recipe:

* **Grafting from Adam.** Each block's step keeps Shampoo's *direction* but
  takes the *length* of Adam's step for that block. With the race's shared lr,
  betas and weight decay, a difference from AdamW is therefore the
  preconditioner's doing.
* **Blocking.** Dimensions up to `max_preconditioner_dim` = 1024 are merged
  (small matrices become vectors, preconditioned by a full `d × d` matrix,
  power −1/2), and anything larger is cut into 1024-wide blocks.
* **Amortized roots.** The inverse roots come from an eigendecomposition of
  `L / (1 − β2^t) + εI` (ε = 1e-12), recomputed every 10 steps. Before step 10
  the step is Adam's.
* **Which parameters.** The hidden matrices are preconditioned: the same ones
  Muon orthogonalizes (attention and expert projections, router, mHC
  mixers). Embeddings, Engram tables, norms and other vectors take Adam's
  step, Shampoo's own grafting method. Preconditioning a hashed table's rows
  would cost a 1024 × 1024 matrix per 1024 rows and mean nothing.
  `shampoo_on="all"` preconditions everything.

**How it relates to Muon.** With no accumulation (β2 = 0) and ε → 0,
`(G Gᵀ)^(-1/4) G (Gᵀ G)^(-1/4) = U Vᵀ` for `G = U S Vᵀ`: exactly the
orthogonalized gradient that Muon approximates with Newton–Schulz. Muon is
Shampoo without memory; Shampoo whitens with statistics accumulated over many
steps. In the plots they share a hue (Shampoo is cross-hatched,
dash-dot-dot).

**Checked.** Against Meta's own code (github.com/facebookresearch/optimizers
at commit `d24a149`), step for step, in seven configurations: merged vectors,
2-D blocks, Adam-then-Shampoo with amortized roots, no weight decay, an
Adam-only group, β1 = 0, and fp64. The parameters are **bit-identical** in every
one. The test runs whenever that package is installed (CI installs it on
Python 3.12). Independently of it, the first step matches the textbook formula
computed in fp64, the Adam phase matches `torch.optim.AdamW`, grafting gives
each block Adam's step length, and a checkpoint resumes bit-exactly.

**Cost and scale.** State per matrix block of `m × n`: the two factors and
their inverse roots, `2(m² + n²)` floats, plus Adam's two moments. For
1024 × 1024 blocks that is 4 floats per parameter (Meta packs the symmetric
matrices and stores half; this port keeps them whole for clarity), so with
Adam's moments about 24 bytes/param in fp32. Compute:
* preconditioning every step, `2·2·1024³` FLOPs per 1M-parameter block, about 4
  thousand FLOPs per parameter;
* the eigendecompositions every 10 steps, `2·9·1024³` per block, about 2
  thousand per parameter per step amortized.

On the `h100` preset (8.69B parameters, 1.34B active) at the race's 18.6K-token
full batch, that is roughly 40% on top of the model's own FLOPs. It is also
slower per FLOP: eigendecompositions do not run on tensor cores. The race
counts all of it. At 7–10B on one H100, Shampoo needs the CUDA phase's
low-precision storage (bf16 or packed factors), and blocking at 1024 is what
keeps it feasible at all.

## SuperGrok 1.1

**What it is supposed to do.** Your own design: AdamW whose gradient is corrected,
element by element, by a small learned network that also sees how sharp the loss is
around each weight, with GrokAdamW's machinery around it. Per step, per tensor:

```
s   = |∇L(w + ρ·g/‖g‖) − g|          sharpness, from a SAM probe every 10 steps (and step 1)
g   = clip(g, 1.0)                     per tensor
μ   = r · φ(g, s)                      φ: 2→32→1 GELU MLP on every (g, s) pair; r a learned scale (starts at 0)
gate = 1 − sigmoid(5 · cos(g, m))      m = Adam momentum; the correction is trusted less when g agrees with m
ĝ   = g + gate · ramp · λ · α · μ      ramp: 0 for 100 steps, then linear to 1 over 100
AdamW on ĝ with β1 = 0.9 · (1 − γ)^layer
α   = 0.98 · exp(−κ · signal)          every 50 steps; signal from the train/held-out loss gap, 10 once memorized
```

Every 5 steps the meta step trains φ and r through one virtual step,
`w⁺ = w(1 − lr·wd) − lr·(g + r·φ(g, s))`, to lower the held-out loss plus the training
loss at `w⁺` (Adam, lr 1e-4).

**Why your old code ran like AdamW.** You were right. The old repository could only run
SuperGrok 1.1 inside one fused CUDA kernel, and the Python `step()` raised. That kernel
silently dropped most of the design:

* the layer-wise β1 (γ), the per-tensor clip and the whole adaptive-α path (κ, the
  thresholds, the refresh cadence) never reached the kernel, so changing them changed
  nothing (the tuner tuned four dead knobs);
* the gate's `sqrt(|g|²|m|² + 1e-12)` floor made it a constant 0.5 for every tensor
  once gradients got small, i.e. after memorization, exactly when it was meant to act;
* the meta-net was trained on `|Δ|` sharpness from the host but evaluated on a separately
  computed `Δ²` buffer in the kernel, and it was meta-trained on the early-stopping split;
* in your v1 script the meta optimizer never trained the meta-net at all (its parameters
  were not in the graph), and v2 swallowed every meta-step exception.

What remained was AdamW plus a correction that started at exactly zero (r = 0). In the one
full H100 run with a trained meta-net, the correction then grew until training collapsed
(DNF, final test accuracy 0.007).

**What the new implementation does.** `grokking_optimizers/supergrok11.py` runs *every*
component, in plain PyTorch:

* It is checked against an independent, naive transcription of the declared algorithm (the
  legacy `sam_step`, `meta_step`, `_update_alpha` and the kernel header's update), with all
  components active at once. Parameters, meta-net weights, α and sharpness agree to
  1e-12 over 30 steps (fp64).
* **Each component is shown to act on the update.** Switching it off changes the
  trajectory by this much (max parameter difference ÷ max movement, 30 steps):

  | component switched off | effect |
  |---|---|
  | cosine gate | 1.4 |
  | layer-wise β1 | 0.64 |
  | per-tensor clip | 0.16 |
  | learned correction / meta step | 0.12 |
  | adaptive α | 0.10 |
  | train term of the meta objective | 0.034 |
  | warm-up ramp | 0.001 |
  | sharpness input | 0.00002 |

  The sharpness input is live but almost invisible. φ sees `s ≈ 3e-4` through weights of
  size ~0.01, so its second input barely registers.
* Switching off the correction, γ, the clip and the probe gives `torch.optim.AdamW` bit for
  bit. That is the proof the rest is additive, not a claim.
* The race logs what every component is doing at each evaluation: α, r, φ's bias, the
  gate (mean/min/max), the size of the correction relative to the gradient, the share of
  coordinates whose sign the correction flips, and the mean sharpness.
* Two deliberate departures from the legacy meta step, so the meta-net is trained on what
  it is deployed on: it sees the *clipped* gradient (legacy: raw), and the virtual step
  applies the same zero-gradient mask (and optional cap) as the real one. Both matter
  only when a tensor's gradient norm exceeds 1 or has exact zeros.
* Race settings: the legacy race's values (λ = 1.0, the last value you raced), β1 decaying
  per **transformer block** rather than per tensor (per tensor, γ = 0.1 would leave most of
  the 290 tensors of even the tiny model with β1 < 0.1), and no correction on
  exactly-zero gradient entries (otherwise unused Engram rows drift by ~lr every step).

**The paired control.** `supergrok11_frozen` is the same optimizer with the learned
correction and the SAM probe off: AdamW + block-wise β1 + the per-tensor clip, on the same
data split. The gap between the two is what the meta-net contributes.

**What to expect: the design has a structural flaw, now reproducible.** Measured by the
analysis on a small MLP, and visible within 30 steps in the tests above (r·φ's constant
term already −0.014 and 7% of coordinates sign-flipped):

1. φ is effectively affine at these input sizes, and what it learns is a same-sign
   constant (its output bias; "constant share" 1.000 from step 250 on).
2. The meta-gradient on r and on φ's bias kept the same sign in 100% of 400 meta steps,
   even while training loss rose 22×. Adam pushes both at full rate, so their product
   grows quadratically.
3. The lookahead cannot see the damage: its virtual step (plain SGD, no gate, ramp, λ or α)
   is 69–530× smaller than the real Adam step.
4. Once the constant outgrows the post-memorization gradient, 55–65% of coordinates get
   their update sign flipped and training collapses. A live gate does not prevent it; no
   gate is worse.
5. Removing the constant (zero-mean φ) prevents the collapse, but then the meta-net learns
   nothing (correction ~4e-10 of the gradient): AdamW again.

**Seen in the race setting.** One CPU run of the race itself: DeepSeek-V4.1-Flash
`tiny`, modular division mod 23, 50% train, seed 42, 4,000 steps, with the code as it
stood before the fixes above (same mechanism). SuperGrok 1.1 memorized its training
examples by step 100.

| step | ‖correction‖ / ‖g‖ | coordinates whose sign it flips | train acc |
|---|---|---|---|
| 200 | 0.05 | 46% | 0.90 (all it trains on) |
| 1,100 | 1.02 | 45% | 0.90 |
| 1,200 | | | 0.07 (chance) |
| 4,000 | 15 | 48% | 0.07 |

Training collapsed at step ~1,150, when the correction's norm reached the
gradient's, and never recovered: the loss sits at 3.12 ≈ ln 23. The learned
constant `r · b2` grew from −1.2e-5 (step 200) to −6.2e-5 (step 1,200) and then held,
negative all along. AdamW and the
frozen control memorized and held on the same data. None of the three grokked in
4,000 steps at this small modulus; that takes the GPU race at p = 97.

Opt-in variants for experiments, none claimed to help until the race says so:
`meta_grad="first_order"` (matches the exact meta-gradient to cosine 1.0 and needs only
chunk-sized memory; required above ~1e8 parameters); `max_correction_ratio` (caps each
tensor's correction at a multiple of its gradient norm, a bound the design lacks); the
historical variants `gate_mode="atlas"`, `meta_objective="lookahead_val"|"align"`,
`sharpness_transform="square"` and `gate_eps=1e-12`.

**If you want it to actually grok faster,** the mechanism above says what has to change:
the correction needs a bound (a NeuralGrok-style per-tensor normalization or the ratio
cap); the virtual step must model the real Adam update; and the meta objective needs an
interior optimum rather than a linear one. Each is a new design, to be tested in the race
against the frozen control.

**Cost and scale.** State is 12 bytes/param (m, v, sharpness). Every 10 steps there is one
extra forward+backward (the probe); every 5 steps a meta step (two forwards and a backward
through a virtual model). The φ network runs on every gradient entry every step (~200
matmul FLOPs per parameter, counted in the race's FLOPs). The exact meta step materializes a [N, 32] activation over all
parameters, about 1.3 TB at 10B, so it cannot run at scale; the first-order form can. Not
recommended for 7–10B until the flaw above is fixed.

---

## Memory on one H100 (80 GB) for the `h100` preset

The `h100` DeepSeek-V4.1-Flash preset has 8.69B parameters (1.34B active per
token) with the race's vocabulary. Weights, gradients and optimizer state alone,
before activations:

| Recipe | bytes/param | 8.69B model |
|---|---|---|
| fp32 weights + fp32 grads, no optimizer state | 8 | 70 GB |
| + AdamW fp32 moments (the usual recipe) | 16 | 139 GB: does not fit |
| pure bf16 weights + grads, Muon or Lion fp32 momentum | 8 | 70 GB: fits, barely |
| pure bf16 weights + grads, AdamW bf16 moments | 8 | 70 GB: fits, barely |
| + Grokfast / GrokAdamW / LookSAM extra state | 12+ | 104 GB+: does not fit |
| Prodigy (4 state tensors) | 20+ | does not fit in any regime |
| Shampoo, 1024 blocks (factors and roots unpacked, plus Adam's moments) | 32 | does not fit; needs bf16 / packed factors |

So the plain fp32 PyTorch race cannot run this preset on one H100; that is the
job of the CUDA phase (bf16 or FP8 storage, stochastic rounding, fused
optimizer-in-backward, activation checkpointing). Two practical knobs until then:
the number of routed experts changes total parameters without changing
per-token compute (`--model-overrides '{"n_routed_experts": 16}'` gives 2.65B,
which fits with fp32 AdamW at 42 GB), and `--micro-batches` splits the
18.6K-token full batch (4,656 examples x 4 tokens) to bound activation memory without changing the maths.
