# Mamba-3 (SISO) Reference — Reversible Foundation

Reference implementation of the **full Llama-style Mamba-3** model for the
`mamba` model. Source paper: **arXiv 2603.15569** (ICLR 2026), *"Mamba-3:
Improved Sequence Modeling using State Space Principles"* (Lahoti, Li, Chen,
Wang, Bick, Kolter, Dao, Gu). Full text read at `/tmp/mamba3_paper.txt`.

- Code: `grokking_optimizers/mamba3_block.py` (`Mamba3Layer` mixer, `SwiGLU_MLP`,
  `Mamba3Block`, `Mamba3Model`, `RMSNorm`)
- Oracle: `tests/hw/mamba3_oracle.py` (fp64; forward + VERIFIED hand-derived
  manual backward that matches autograd to **≤4e-15** per param + input — §4, §6, §7)

The SSM mixer is the **SISO base** Mamba-3 (the MIMO variant of §3.3 is a
documented later option, §5). The **architecture is now the complete Llama-style
stack** of Section 3.4: each layer alternates a **Mamba-3 mixer** sub-block and a
**SwiGLU MLP** sub-block, both pre-norm + residual (`h = h + Mixer(RMSNorm(h))`,
then `h = h + SwiGLU(RMSNorm(h))`). With the SwiGLU blocks added, the canonical
`d=2048 / nl=24 / MLP-inner-4096` config reaches the paper's **~1.5B** (the
previous mixer-only reference was 0.661B). It is NOT wired into production, the
CUDA megakernel is untouched, and nothing is committed. The CUDA megakernel
transcribes its forward + backward line-for-line from this reference and the
verified manual backward in §6 (mixer) and §7 (SwiGLU + pre-norm/residual).

---

## 1. Exact forward equations implemented

Per token. Quantity layout (RESOLVED against paper Sec 3.3 line 501-504 +
App C line 1661-1663 — the P=`head_dim` channels in a head **share** `alpha,dt,
B,C`; B,C,θ are shared across **all** heads via multi-value attention):

| Quantity | scope | shape per token |
|---|---|---|
| `dt`, `A_real`, `lambda` | **per head** | `[n_heads]` |
| `theta`, `B`, `Bhat`, `C`, `Chat` | **head-shared** (all heads) | `[N_c]` |
| `x` (SSM input), `D` skip, `z` gate | **per channel** | `[d_inner]` |

State is **real**, size `N = state_dim`, viewed as `N_c = N/2` complex
coordinates (a 2-vector `(real, imag)` each); the full SSM state is
`[n_heads, head_dim, N_c, 2]` — within a head all `head_dim` channels evolve with
the SAME `alpha_t`, `R_t` and head-shared `Bbar_t`, differing only through `x`.

### 1.1 Exponential-trapezoidal discretization (Section 3.1, Prop 1, Eq 5–6; Table 1 last row)

The continuous LTV SSM `h'(t) = A(t)h(t) + B(t)x(t)`, `y = C(t)^T h(t)` is
discretized with the **exponential-trapezoidal** rule to the **3-term** recurrence:

```
h_t = alpha_t * h_{t-1}  +  beta_t * B_{t-1} x_{t-1}  +  gamma_t * B_t x_t      (Eq 6)
```

with coefficients (Eq 5 / Table 1):

```
alpha_t = exp(dt_t * A_t)                                  # decay
beta_t  = (1 - lambda_t) * dt_t * exp(dt_t * A_t)          # = (1-lambda_t)*dt_t*alpha_t
gamma_t = lambda_t * dt_t
```

`lambda_t ∈ [0,1]` is a **data-dependent** convex-combination gate.
`lambda_t = 1` → Mamba-2 exponential-Euler; `lambda_t = 1/2` → classical
trapezoid (Remark 2). **Default parameterization** (Appendix A.3, Table 8 — best
ppl): `lambda_t = sigmoid(u_t)` with `u_t` a learned linear projection of the
token; the `lambda_t = 1/2 + O(dt)` constraint is intentionally **not** enforced.

### 1.2 Complex state via the "RoPE trick" (Section 3.2; Prop 2, Prop 4; Eq 8, 9, 25)

The SSM is complex (Eq 8): transition `Diag(A(t) + i*theta(t))`, complex
projections `(B + i*Bhat)`, `(C + i*Chat)`. Proposition 2 (Eq 9) shows the
discretized complex SSM of complex-dim `N_c` equals a **real** SSM of state dim
`N = 2*N_c` whose transition is a scalar decay × block-diagonal of 2×2 rotations:

```
R(phi) = [[cos phi, -sin phi],
          [sin phi,  cos phi]]
R_t    = blockdiag_{i=1..N_c} R(dt_t * theta_t[i])
Bbar_t = [ B_t ; Bhat_t ]   in R^N        # real/imag stacked per complex coord
Cbar_t = [ C_t ; -Chat_t ]  in R^N
```

Combined with exponential-trapezoidal (Proposition 4, **Eq 25**, the
direct-rotation real form), the recurrence we run is:

```
h_t = alpha_t * (R_t @ h_{t-1})
      + beta_t  * (R_t @ (Bbar_{t-1} * x_{t-1}))
      + gamma_t *        (Bbar_t     * x_t)                 (Eq 25)
y_t = Cbar_t^T @ h_t
```

For complex coord `i`, the 2-vector `h_t[i]` is rotated by `R(dt_t*theta_t[i])`;
`B,C` act as scalars on `x_t`; `y_t` is the **real part** of the complex inner
product (`Cbar = [C, -Chat]` encodes the `Re(...)` of Eq 8's output). With `dt`
per-head and `theta` head-shared, the angle is `phi[h,i] = dt[h]*theta[i]`
(**per-head per-coord** `[n_heads, N_c]` — exact, no mean-over-channels approx).

> **Why Eq 25 and not Eq 11/24 (cumulative-product RoPE form)?** They are
> algebraically identical (Prop 4 proof, Appendix B.3). Eq 25 applies `R_t`
> **per step** inside the recurrence — the cleanest *sequential* form for a CUDA
> kernel to transcribe line-for-line. The cumulative-product `prod R^T` form is
> the optimization for the *parallel/chunked matmul* path; the kernel phase can
> switch to it if it implements the chunked SSD algorithm.

### 1.3 Data-dependent A (Remark 1)

All SSM params are data-dependent. Real part `A_t < 0` is a **scalar per head**
(Eq 8/9/17: "`A_t ∈ R` is a scalar so that `exp(dt·A_t)` commutes with rotations",
paper proof line ~1431; Mamba-2's scalar-identity transition), from a base
**per-head** rate `A_log` (Mamba-2 init `log(arange(1..n_heads))`) plus a
data-dependent **per-head** log-rate `A_mod`, mapped through
`A_t = -softplus(A_mod + exp(A_log))` so `alpha_t = exp(dt_t*A_t) ∈ (0,1]`, with
`dt_t = softplus(dt_proj(.)) > 0` (also per head). The rotation `R_t` (built from
`theta`) carries the per-coordinate structure; the real part stays scalar.
Imaginary part `theta_t` is a free head-shared data-dependent linear projection
(rotation frequency); per-step angle `phi_t[h,i] = dt_t[h] * theta_t[i]`.

### 1.4 BCNorm + B,C biases (Section 3.4; Appendix F, Table 10)

After the `B`,`C` projections: **RMSNorm** (BCNorm / QKNorm) over the state dim,
then add learnable, **data-independent, channel-wise** biases to **both** `B` and
`C` (added *after* the norm), **initialized to all-ones** (Appendix F default,
Table 10a). These biases inject the convolution-like data-independent component
that (with the trapezoidal implicit conv) replaces the short conv.

### 1.5 Mixer-block scaffold (Section 3.4 / Fig. 2; Appendix C)

```
residual = x
xz = in_proj(x);  x_main, z = xz.chunk(2)        # no bias
x_in = x_main                                     # NO conv1d AND NO SiLU (see §5)
(dt, A_real, theta, lambda, B, C) = x_proj(x_in)  # all data-dependent
y = mamba3_scan(x_in, dt, A_real, phi, lambda, Bbar, Cbar)   # Eq 25
y = y + x_in * D                                  # per-channel D skip
y = y * SiLU(z)                                   # gate: phi(Y,Z)=Y(.)SiLU(Z)  (App C, Eq C)
out = out_proj(y)                                 # no bias
return out (+ residual if self.residual)          # see §1.6 — OFF in Mamba3Block
```

The output gate `· SiLU(z)` is RETAINED (App C: `φ(Y,Z) := Y ⊙ SiLU(Z)`); only
the SSM-input SiLU is dropped. `Mamba3Layer` has a `residual` flag: `True`
(default) keeps the Mamba-1/-2 drop-in contract (`out + x`); the Llama-style
`Mamba3Model` constructs the mixer with `residual=False` and adds the residual
around the pre-norm at the **block** level (§1.6).

### 1.6 Llama-style architecture: alternating Mamba-3 mixer + SwiGLU (Section 3.4)

> Section 3.4: *"The overall architecture follows Llama (Grattafiori et al.
> 2024), **alternating Mamba-3 and SwiGLU blocks with pre-norm**."*

Each `Mamba3Block` is **one Llama layer** = a Mamba-3 mixer sub-block then a
SwiGLU MLP sub-block, each pre-norm + residual:

```
h = h + Mamba3Mixer( RMSNorm_mix(h) )      # mixer residual=False (added here)
h = h + SwiGLU_MLP ( RMSNorm_mlp(h) )      # SwiGLU feed-forward
```

`Mamba3Model`: `tok+pos` embeddings → `nl` × `Mamba3Block` → final `RMSNorm` →
`Linear(d, p)` head on the **last** token; CE loss. The Mamba-2 **post-gate
RMSNorm is removed** (BCNorm stabilizes; Section 3.4) — only the single final
RMSNorm before the head remains.

**Alternation convention (RESOLVED — Convention A, "n of each"):** `nl` counts
the **mixer** blocks; **each is paired with a SwiGLU MLP**, so the canonical
config has **24 Mamba-3 mixers + 24 SwiGLU blocks**. This is the Mamba-2
convention the paper explicitly follows (Appendix D: *"pretraining procedures
follow those of Dao and Gu (2024)"*), and it is the convention that makes the
canonical config (`d_model=2048`, `nl=24`, MLP inner 4096, Llama-3.1 vocab
128256, **tied** embeddings) land at **~1.5B** (measured 1.5277B; §3). The
alternative "24 total = 12 mixer + 12 SwiGLU" gives only 0.895B and is rejected.

**SwiGLU MLP** (`SwiGLU_MLP`, the standard Llama feed-forward, **no biases**):

```
SwiGLU(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
```

with three bias-free `Linear`s, inner ("MLP"/"hidden") dim `d_ff`. At the 1.5B
scale `d_ff = 4096 = 2·d_model` (Appendix C Table: *"SISO MLP dim … 1.5B:
4,096"*; Appendix D fixes the expand factor at 2). The default `mlp_ratio=2`
reproduces 4096 at d=2048. The paper states **4096 directly**, so we use it as-is
and do **not** re-apply the classic Llama `2/3` factor (it is already folded into
the stated 4096). The MIMO variant shrinks this to 3824 to parameter-match SISO
(Appendix C); SISO uses the full 4096. **Backward in §7.**

---

## 2. Key differences vs the current Mamba-1 `SelectiveSSMLayer`

| Aspect | Mamba-1 (`SelectiveSSMLayer`, grokking_race_v2.py:412) | Mamba-3 (`Mamba3Layer`) |
|---|---|---|
| **Short conv1d** | depthwise `Conv1d(k=3)` on `x_main` (NON-causal) | **DROPPED** — trapezoidal implicit conv + B,C biases subsume it (Sec 3.1.2, 4.2) |
| **SSM-input SiLU** | `SiLU` after the conv on `x_main` | **DROPPED** — conv "and its accompanying activation function" obviated (Sec 3.4) |
| **`dt` / `A` / `lambda` scope** | per-channel `dt`; `A=-exp(A_log)` per `[d_inner,N]` | **per head** (`P` channels in a head share `alpha,dt,B`; Sec 3.3, App C) |
| **Discretization** | exponential-**Euler**, 2-term: `h = exp(dt·A)h + dt·B·x` | exponential-**trapezoidal**, 3-term: `+ beta·B_{t-1}x_{t-1}` (Eq 6) |
| **State** | **real** diagonal, `A = -exp(A_log)` (real eigenvalues) | **complex**, `A = A_real + i·theta`; real form = decay × 2×2 rotations (Prop 2) |
| **State tracking** | cannot (real eigenvalues) — fails parity | can (rotations) — solves parity / modular arithmetic (Table 5b) |
| **`lambda` (trapezoid gate)** | n/a | `lambda_t = sigmoid(u_t)`, data-dependent (App A.3) |
| **B,C normalization** | none | **BCNorm** (RMSNorm) on B,C (Sec 3.4) |
| **B,C biases** | none | learnable all-ones-init channel-wise biases on B,C (App F) |
| **Per-layer norm** | post-`LayerNorm` inside the block | **Llama pre-norm**: RMSNorm before the mixer AND before the SwiGLU MLP; model-level final RMSNorm; post-gate norm removed (Sec 3.4) |
| **Norm type** | LayerNorm | RMSNorm |
| **Gate** | `(y + D·x)·SiLU(z)` | same scaffold: `(y + D·x)·SiLU(z)` (App C `phi`) |
| **MLP / feed-forward** | none (mixer-only stack) | **SwiGLU MLP** alternating with the mixer, Llama-style (Sec 3.4); `down(SiLU(gate)⊙up)`, inner 4096 at 1.5B (§1.6, §7) |
| **Model architecture** | embed → `nl` mixers → norm → head | embed → `nl`×(mixer + SwiGLU) blocks, pre-norm → norm → head (Llama, Sec 3.4) |

---

## 3. Parameter counts (measured — full Llama-style mixer + SwiGLU model)

The model is now the complete Llama-style stack (§1.6): `nl` Mamba-3 **mixer**
blocks + `nl` **SwiGLU** blocks (Convention A, "n of each") + 2 pre-norms per
block + final norm + head.

| Config | d_model | state | head_dim | n_heads | MLP inner | nl (mixers) | Params |
|---|---|---|---|---|---|---|---|
| **Toy grokking gate** | 128 | 128 | 64 | 4 | 256 | 2 | **593,713** |
| **Canonical 1.5B (small grokking vocab)** | 2048 | 128 | 64 | 64 | 4096 | **24** | **1,265,411,169 (1.265B)** |
| **Canonical 1.5B (Llama-3.1 vocab, tied embed)** | 2048 | 128 | 64 | 64 | 4096 | **24** | **1,527,677,952 (1.528B ≈ 1.5B)** ✅ |

**Toy (d=128, nl=2)** = **593,713**: mixer 370,512 + SwiGLU 196,864 + embed/head/norm 26,337
(was 396,593 mixer-only; the two SwiGLU MLPs `3·128·256` + their pre-norms add ~197k).

**Canonical 1.5B (d=2048, nl=24, MLP inner 4096):**
- **Per-`Mamba3Block`** ≈ **52.7M**: a mixer (**27,538,048**, dominated by
  `in_proj` 2·d·d_inner=16.78M, `out_proj` 8.39M, `x_proj` 576·4096=2.36M) + a
  SwiGLU MLP (**25,167,872** = `3·d·d_ff`=3·2048·4096=25.17M) + two pre-norm
  RMSNorm weights (2·2048).
- **Layer stack (vocab-independent)** = 24 mixers (**0.661B**) + 24 SwiGLU
  (**0.604B**) = **1.265B**.
- With the paper's **Llama-3.1 tokenizer vocab (128256)** and **tied embeddings**
  (`out.weight` tied to `tok.weight`, the Mamba LM convention): embedding
  128256·2048 = 262.7M counted once → **1.528B ≈ the paper's 1.5B**. (The
  small-grokking-vocab number, 1.265B, simply omits the large embedding/head; the
  layer stack is identical.)

**The ~1.5B target is now reached** — the previous reference was mixer-only and
topped out at 0.661B precisely because it omitted these SwiGLU blocks. Adding
them (Sec 3.4 "alternating Mamba-3 and SwiGLU blocks with pre-norm") supplies the
~0.6B of MLP params and the large tied embedding closes the gap to 1.5B.

The per-head `dt/A/lambda` resolution (§5) is unchanged and still governs the
mixer's internal cost (it shrank `x_proj`/`dt_proj` vs the old per-channel form —
the dominant ~33.8M/layer saving that brought the mixer stack to 0.661B).

---

## 4. Validation results (`tests/hw/mamba3_oracle.py`, CPU fp64)

```
[C] fp64 forward loss            = 4.2325235939129477e+00  (finite=True)
[B] parameters total             = 45              # +10 vs mixer-only: per block
[B] parameters w/ finite grad    = 45              #   mixer_norm + mlp_norm + 3 MLP W
[A] forward deterministic        = True            # bit-identical across 2 runs
[D] fp32 vs fp64 forward max|diff|= 1.22e-06  (rel 7.54e-07)  -> within tol
[B*] finite-diff spot check (out.bias[0]): autograd vs FD rel_err ~1.6e-08  PASS
[E] TOY param count            = 593,713   (mixer 370,512 + SwiGLU 196,864 + 26,337)
[E] 1.5B (small grokking vocab)= 1,265,411,169 (1.265B): 24 mixers 0.661B + 24 SwiGLU 0.604B
[E] 1.5B (Llama-3.1 vocab,tied)= 1,527,677,952 (1.528B ≈ 1.5B)   <-- paper target reached

(F) MANUAL backward vs torch.autograd — fp64, per-parameter (mixer + SwiGLU +
    both per-block pre-norms) + input grad:
    seed0 / shapeA (n_heads=8, N_c=4,  nl=2): WORST rel-err = 1.4e-15
    seed7 / shapeB (n_heads=8, N_c=6,  nl=3): WORST rel-err = 2.8e-15
    TOY config     (n_heads=4, N_c=64, nl=2): WORST rel-err = 3.9e-15
ORACLE RESULT: PASS
```

(The canonical 1.5B-scale model also forward-checks finite + deterministic
standalone: 1,265,411,169 params, `d_ff=4096`, `n_heads=64`.)

All criteria pass:
- **Forward** finite + deterministic; **fully differentiable** (all 45 params get
  finite gradients, incl. the new `gate/up/down_proj` and the `mixer_norm` /
  `mlp_norm` pre-norms); fp64 numerically stable; fp32 forward matches fp64 to ~1e-6.
- **(F) The headline gate**: the hand-derived MANUAL (non-autograd) backward in
  `mamba3_oracle.py` matches `torch.autograd` to **≤ 4e-15 relative** (worst over
  ALL parameters AND the input, across two random seeds / shapes + the toy config)
  — comfortably under the 1e-10 gate and the ~1e-12 target. The SwiGLU MLP, the
  two per-block pre-norms, and the block-level residuals are all covered by the
  new `swiglu_forward`/`swiglu_backward` + `block_forward`/`block_backward`. The
  manual backward is the verified reference the CUDA megakernel transcribes
  line-for-line (§6 mixer, §7 SwiGLU + pre-norm/residual). RMSNorm in the block
  computes in the input dtype (no internal `.float()` downcast) so the fp64 oracle
  is exact; an fp32 round-trip would have capped precision at ~1e-7.

No CUDA build or megakernel gate was run.

---

## 5. Ambiguities — RESOLVED

- **RESOLVED — SiLU on the SSM input is DROPPED.** Paper Sec 3.4 (line 678):
  exponential-trapezoidal discretization + B,C biases "obviate the short causal
  convolution **and its accompanying activation function** present in Mamba-2".
  So `x_main` feeds the projections and the scan **directly** (`x_in = x_main`,
  no `SiLU`, no conv). The **output gate** `· SiLU(z)` is a *different*, retained
  activation (App C, `φ(Y,Z) := Y ⊙ SiLU(Z)`) and stays.

- **RESOLVED — `dt` (and `A_real`, `lambda`) are PER HEAD.** Paper Sec 3.3
  (line 501-504): the SISO recurrence is stacked over the `P = head_dim` channels
  of a head "with `P` copies **sharing the same `alpha_t`, `dt_t` and `B_t`**";
  App C (line 1661-1663) confirms B,C are `N·R` (head-shared) and the per-head
  input `x` is `P`-dim. So `dt`, the scalar `A_real`, and `lambda` are **per
  head** (`[n_heads]`), `theta/B/C` are **head-shared** (`[N_c]`), `x/D/z` are
  **per channel** (`[d_inner]`). The rotation angle `phi[h,i] = dt[h]·theta[i]`
  is therefore **per-head per-coord** and **EXACT** — the old per-channel-mean(dt)
  approximation is gone. `dt_proj` now emits `n_heads`; `x_proj` emits `n_heads`
  for `A_mod` and `lambda` (was `d_inner`); `A_log` is `[n_heads]`.

- **RESOLVED — real-part `A` is a scalar per head.** Prop 2/Eq 9/17 keep `A_t` a
  real **scalar** (`A(t) ∈ R`, paper line 365; "scalar so that `exp(dt·A_t)`
  commutes with rotations", line ~1431); the rotation `R_t` carries the
  per-coordinate structure (Mamba-2's scalar-identity transition). Base per-head
  rate `A_log = log(arange(1..n_heads))`, plus per-head `A_mod`, then
  `A_real = -softplus(A_mod + exp(A_log))`. (Previously this was an
  unjustified per-channel mean over a `[d_inner, N_c]` spectrum; now it is the
  faithful per-head scalar.)

- **D skip term kept (low-risk choice).** Mamba-3 Fig. 2 / App C do not
  explicitly state whether the Mamba-1/-2 per-channel `D` skip (`y += D ⊙ x`) is
  retained. **KEPT `D`** because Mamba-3 "retains the overall layout of its
  predecessor" (Sec 3.4) and it is a single per-channel vector — negligible
  params, trivially zeroable if the kernel phase confirms it is dropped.

- **NOTE — B,C biases are per-coord here (head-specific in the full model).**
  App F (line 1778) says the biases are "head-specific and channel-wise". In the
  **SISO** layer B,C are head-shared (one stream of `N_c` per token), so a single
  per-`N_c` bias is the faithful instantiation; the head-specific form only
  materializes in the multi-head MIMO projection. Init all-ones (App F default).

- **NOTE: MIMO not implemented.** Per the task this is the **SISO base** only.
  The MIMO variant (Section 3.3, Appendix C: rank-`R` projections `B_t∈R^{N×R}`,
  `X_t∈R^{P×R}`, element-wise `W_X` scaling to avoid `R×` param blowup, matmul
  state update `H_t = a_t H_{t-1} + B_t X_t^T`, `Y_t = H_t^T C_t`) is a documented
  later option. A `mimo_rank` knob would slot into `x_proj`/`out_proj` and the
  scan; left out to keep the foundation clean.

- **NOTE: small-d head fallback.** When `d_inner` is not a multiple of
  `head_dim` (the tiny-d grokking gate, e.g. `d=16`), the layer falls back to a
  single head (`head_dim = d_inner`) so the toy config runs. The 1.5B config
  (`d_inner=4096`, `head_dim=64`) gives 64 heads as in the paper.

- **RESOLVED — alternation convention is "n mixers + n SwiGLU" (Convention A).**
  Sec 3.4 says only "alternating Mamba-3 and SwiGLU blocks"; the count could be
  read as "24 of each" or "24 total (12+12)". Resolved to **24 of each** because
  (a) Appendix D states the pretraining "procedures follow Dao and Gu (2024)"
  (Mamba-2), whose layer count counts the *mixer* layers, each interleaved with a
  SwiGLU when Llama-style; (b) only this convention reaches the paper's stated
  **~1.5B** at `d=2048/nl=24/MLP-4096` (measured **1.528B** with the Llama-3.1
  vocab + tied embeddings; the 12+12 reading gives only 0.895B). `nl` in
  `Mamba3Model` is the mixer-block count; each `Mamba3Block` holds one mixer +
  one SwiGLU. (§1.6, §3.)

- **RESOLVED — SwiGLU specifics: no biases, inner = 4096 = 2·d, no extra 2/3.**
  The paper "follows Llama" (Sec 3.4) → bias-free MLP projections (Llama
  convention; no MLP bias is mentioned). The SISO MLP inner dim is stated
  **directly as 4096** at 1.5B (Appendix C Table; = 2·d_model, expand 2 per
  Appendix D), so we set `d_ff=4096` as-is and do **not** re-apply the classic
  Llama `2/3·(4·d)` reduction — that factor is already baked into the stated
  4096. (`mlp_ratio=2` default; MIMO would use 3824 to param-match — not built.)

- **NOTE — embeddings tied for the 1.5B headline.** `Mamba3Model` builds an
  *untied* head (`out = Linear(d, p)` + bias) to keep the small grokking-gate
  drop-in contract. The paper's 1.5B uses the Llama-3.1 vocab (128256); the
  Mamba LM convention **ties** the head to the token embedding, so the reported
  1.528B counts one `V·d` embedding (head shared). The untied count is 1.79B.
  The *layer stack* (1.265B) is identical either way and is what the kernel
  transcribes.

---

## 6. The DERIVED backward (mixer — verified, transcribe the CUDA kernel from this)

This is the hand-derived, NON-autograd backward implemented in
`tests/hw/mamba3_oracle.py` (`scan_backward`, `layer_backward`,
`model_backward`) and **verified to match `torch.autograd` to ≤7e-15 relative**
in fp64 for every parameter and the input (§4 (F)). The CUDA megakernel
transcribes the math below line-for-line. Notation matches §1: per-head scalars
`alpha_t[h], beta_t[h], gamma_t[h]`; head-shared `theta_t[c]`; head-shared
`Bbar_t[c]=(Br,Bi)`, `Cbar_t[c]=(Cr,-Ci)` (2-vectors); per-channel input
`x_t[h,p]`. State `h_t[b,h,p,c,:]` is a 2-vector per complex coord `c`.

### 6.1 Forward (the exact recurrence the backward inverts)

```
alpha_t[h] = exp(dt[t,h] * A_real[t,h])          # in (0,1]
gamma_t[h] = lambda[t,h] * dt[t,h]
beta_t[h]  = (1 - lambda[t,h]) * dt[t,h] * alpha_t[h]
phi_t[h,c] = dt[t,h] * theta[t,c]                # per-head per-coord angle
R(phi) @ (w0,w1) = (cos*w0 - sin*w1,  sin*w0 + cos*w1)
v_t[b,h,p,c] = Bbar_t[c] * x_t[b,h,p]            # 2-vector (Br*x, Bi*x)

h_t = alpha_t * (R_t @ h_{t-1}) + beta_t * (R_t @ v_{t-1}) + gamma_t * v_t
y_t[b,h,p] = sum_{c,d} Cbar_t[c,d] * h_t[b,h,p,c,d]      # h_{-1}=v_{-1}=0
```

### 6.2 Scan backward — reverse-time recurrence (`t = L-1 … 0`)

Carry **TWO** adjoints (this is the key structure):
- `gh[b,h,p,c,:] = dL/dh_t` — flows back through `h_{t+1} = alpha R h_t + …`;
- `gv[b,h,p,c,:] = dL/dv_t` — accumulated from step `t+1`'s β-term
  `beta_{t+1} R_{t+1} v_t`. **This is the width-2 coupling** (`v_{t-1}` feeds both
  step `t-1`'s γ-term and step `t`'s β-term) — the #1 index-bug site. Both carries
  start at 0.

At step `t`, given `gh` (carried from `t+1`, = grad through future `h`'s) and
`gv` (carried from `t+1`'s β-term):

```
# (1) output: y_t = sum Cbar_t · h_t
gh          += Cbar_t * dy_t                               # now gh = full dL/dh_t
dCbar[t,c,d] = sum_{b,h,p}  h_t[b,h,p,c,d] * dy_t[b,h,p]   # Cbar head-shared

# (2) v_t total grad = (gv from t+1 β-term) + (γ-term at t)
dv_t          = gv + gamma_t * gh
dx[t,h,p]    += sum_{c,d} dv_t[...,c,d] * Bbar_t[c,d]      # x per channel
dBbar[t,c,d] += sum_{b,h,p} dv_t[...,c,d] * x_t[b,h,p]     # Bbar head-shared

# (3) coefficient + angle grads.  Rh = R_t @ h_{t-1},  Rv = R_t @ v_{t-1}
dalpha[b,t,h] += sum_{p,c,d} Rh * gh         # per head: sum over p,c,d
dbeta [b,t,h] += sum_{p,c,d} Rv * gh
dgamma[b,t,h] += sum_{p,c,d} v_t * gh
# phi via R'(phi) = [[-sin,-cos],[cos,-sin]].  For a term coef*R@w, adjoint g:
#   dphi += coef*( g0*(-sin*w0 - cos*w1) + g1*(cos*w0 - sin*w1) )
dphi[b,t,h,c] += sum_p [ alpha_t * R'term(h_{t-1}[..,c]) + beta_t * R'term(v_{t-1}[..,c]) ]

# (4) propagate to the previous step.  R orthogonal -> adjoint = R^T = R(-phi):
#   R^T @ g = (cos*g0 + sin*g1,  -sin*g0 + cos*g1)
gv  = beta_t  * (R_t^T @ gh)      # becomes dL/dv_{t-1}  -> carried to step t-1
gh  = alpha_t * (R_t^T @ gh)      # becomes dL/dh_{t-1}
```

At `t=0`, `h_{-1}=v_{-1}=0`, so `Rh=Rv=0` (their α/β/φ contributions vanish) and
the final `gh,gv` multiply into the zero initial state and are discarded.

### 6.3 Fold the per-head coefficient Jacobians

With `alpha = exp(dt·A_real)`, `beta = (1-lambda)·dt·alpha`, `gamma = lambda·dt`,
`phi = dt·theta`:

```
ddt[t,h]     = dalpha*(A_real*alpha)
             + dbeta*((1-lambda)*alpha*(1 + dt*A_real))
             + dgamma*lambda
             + sum_c  dphi[t,h,c]*theta[t,c]          # (post-softplus dt)
dlambda[t,h] = dbeta*(-dt*alpha) + dgamma*dt          # (w.r.t. lambda)
dA_real[t,h] = dalpha*(dt*alpha) + dbeta*((1-lambda)*dt^2*alpha)
dtheta[t,c]  = sum_h  dphi[t,h,c]*dt[t,h]             # theta head-shared
```

(`dalpha/d(dt)=A·alpha`, `dalpha/dA=dt·alpha`; `dbeta/d(dt)=(1-λ)·alpha·(1+dt·A)`,
`dbeta/dλ=-dt·alpha`, `dbeta/dA=(1-λ)·dt²·alpha`; `dgamma/d(dt)=λ`, `dgamma/dλ=dt`;
`dphi/d(dt)=theta`, `dphi/dtheta=dt`.)

### 6.4 Through the projections (the rest of the layer)

```
# softplus / sigmoid Jacobians (dt_pre = dt_proj output, u_lam pre-sigmoid):
ddt_pre  = ddt    * sigmoid(dt_pre)                  # dt = softplus(dt_pre)
du_lam   = dlambda * lambda*(1-lambda)               # lambda = sigmoid(u_lam)
# A_real = -softplus(A_arg), A_arg = A_mod + exp(A_log):
dA_arg   = dA_real * (-sigmoid(A_arg))
dA_mod   = dA_arg
dA_log[h]= sum_{b,t} dA_arg[b,t,h] * exp(A_log[h])   # d(exp(A_log))/dA_log = exp(A_log)

# Cbar=(Cr2,-Ci2), Bbar=(Br2,Bi2):  dCr2=dCbar[..,0], dCi2=-dCbar[..,1],
#                                    dBr2=dBbar[..,0], dBi2= dBbar[..,1]
# each B/C stream: +bias then RMSNorm  (bias grad = sum over b,l; then RMSNorm bwd)
dB_bias = sum_{b,l} dBr2 ; ... ; (dBr, dB_norm.w) = rmsnorm_bwd(dBr2, …) ; etc.

# ddt_pre -> dt_proj backward (weight+bias) -> ddt_lr
# reassemble x_proj output grad: cat[ddt_lr, dA_mod, dtheta, du_lam, dBr,dBi,dCr,dCi]
#   -> x_proj backward -> dx_in (3rd path)
# x_in fans into THREE: the scan (dx from 6.2 (2)), x_proj, and the D-skip
#   (dx_in += dy_skip * D ; dD = sum_{b,l} dy_skip * x_in)
dx_main = dx_scan + dx_xproj + dx_Dskip              # NO SiLU now: dx_in = dx_main
# xz = [x_main | z]; gate sz=silu(z): dz = (dy_gated*y_skip)*silu'(z)
#   -> in_proj backward -> dx_inproj
# in the Llama-style Mamba3Block the mixer's residual is OFF (§1.6), so the input
# grad is dx_inproj ONLY here; the residual path (dx_residual) is added at the
# BLOCK level in §7. (If self.residual: dx = dx_inproj + dx_residual as before.)
dx = dx_inproj  (+ dx_residual if self.residual)
```

`RMSNorm` backward (no fp32 downcast, `y = x·rsqrt(mean(x²)+eps)·w`, `r=rsqrt(…)`,
`xhat=x·r`): `dw = Σ_rows dy·xhat`, `dxhat = dy·w`,
`dx = r·(dxhat - xhat·(Σ_lastdim dxhat·xhat)/D)`.

---

## 7. SwiGLU MLP + Llama-style block backward (verified — §4 (F))

The new sub-blocks added around the mixer (§1.6). Implemented in
`tests/hw/mamba3_oracle.py` as `swiglu_forward`/`swiglu_backward` and
`block_forward`/`block_backward`, verified to match `torch.autograd` to **≤4e-15**
for every SwiGLU/pre-norm parameter + the input (§4 (F)). All standard; the CUDA
megakernel transcribes them line-for-line alongside §6.

### 7.1 SwiGLU MLP (no biases)

```
# forward
g_pre = x @ W_gate^T ;  u = x @ W_up^T
s     = SiLU(g_pre)  ;  prod = s ⊙ u
out   = prod @ W_down^T

# backward (upstream dout)
(dprod, dW_down) = linear_bwd(dout, prod, W_down)
ds = dprod ⊙ u ;  du = dprod ⊙ s
dg_pre = ds ⊙ SiLU'(g_pre)                  # SiLU'(x)=σ(x)(1+x(1-σ(x)))
(dx_gate, dW_gate) = linear_bwd(dg_pre, x, W_gate)
(dx_up,   dW_up)   = linear_bwd(du,     x, W_up)
dx = dx_gate + dx_up                        # x fans into gate AND up
```

### 7.2 Llama block (two pre-norm residuals)

```
# forward (residual=False mixer; RMSNorm_mix/mlp are the per-block pre-norms)
h1 = x  + Mixer ( RMSNorm_mix(x)  )
h2 = h1 + SwiGLU( RMSNorm_mlp(h1) )

# backward (upstream dh2)
# h2 = h1 + SwiGLU(RMSNorm_mlp(h1)):
dh1   = dh2.clone()                                  # residual path
dh1n  = swiglu_bwd(dh2, …)                            # through the MLP
(dh1_mlpnorm, dW_mlpnorm) = rmsnorm_bwd(dh1n, …)
dh1  += dh1_mlpnorm                                   # h1 fans: residual + mlp_norm
# h1 = x + Mixer(RMSNorm_mix(x)):
dx    = dh1.clone()                                   # residual path
dxn   = layer_bwd(mixer, dh1)                          # mixer with residual OFF
(dx_mixnorm, dW_mixnorm) = rmsnorm_bwd(dxn, …)
dx   += dx_mixnorm                                    # x fans: residual + mixer_norm
```

Each pre-norm input therefore fans into **two** paths (its residual + its
sub-block); the residual `.clone()` and the additive accumulation onto `dx`/`dh1`
are the bug-prone spots (mirrors how the mixer input fans into in_proj + the
mixer residual when `residual=True`).

The model wrapper adds: cross-entropy (`dlogits=(softmax-onehot)/B`), the head
`Linear`, the final `RMSNorm`, last-token gather (only `t=L-1` carries grad into
the block stack), and the embedding `index_add_`. All verified in §4 (F).
