# DESIGN-TC-PIPELINE.md — Tensor-core batch-tiled megakernel (R2) implementation contract

**Scope.** Branch `claude/h100-audit-maximal`, H100 sm_90, CUDA 12.4. Performance
architecture for the persistent-megakernel race cells (3 models × 11 optimizers =
33 L3 cells). Target: **maximize roofline fraction per cell** under the owner's
constraints (bf16 race precision with fp32 accum; no CUDA graphs — one persistent
kernel + in-kernel grid barriers; deterministic — no float atomics, fixed
reduction order; hang-free — occupancy ≥ 1 or refuse; maximal verified PTX).

**This is an implementation contract for a build agent.** Every optimization lands
inside a component layer (substrate / model-stage header / optimizer header /
`opt_components.cuh`) so all 33 cells inherit it through recomposition by
`megakernel_codegen.py`. No cell-specific hacks. The architecture in
`csrc/fused/COMPONENT_CONTRACT.md` is inviolate.

**Honesty rails.** Where my context notes contradict the code, the **code wins**
and is cited `file:line`. Every number is sourced or computed (arithmetic shown).
Achievable roofs are shape-honest — no "wgmma → peak" claims. Uncertainties are
flagged 🟡.

---

## 0. Corrections to the briefing (code wins)

| Briefing claim | Code reality | Source |
|---|---|---|
| decoder full-batch "~4.7k samples ≈ 4191" | **B = int(97²·0.5) = 4704** samples; the `4191` in code comments is a stale val-split number, not the train batch | `grokking_race_v2.py:238` (`frac_train 0.5`, `p 97`); verified `int(97*97*0.5)=4704` |
| decoder "vocab 97/99" | tok-embedding rows **99**; CE over **99** logits | `decoder_layout.cuh:23` `SG_DEC_VOCAB=99` |
| decoder megakernel "3 phases / 2 barriers" | current decoder cell is **4 phases / 3 grid barriers**: P0 zero → **B0** (`:124`) → P1 fwd+bwd → **B1** (`:148`) → P2 reduce → **B2** (`:178`) → P3 opt. (The file's own docstring at `:18` says "5 phases / 4 barriers" but enumerates only P0–P3 and B0–B2 — the docstring is stale; the **call sites** are the ground truth.) | `fused_decoder_megakernel.cuh:124,148,178` |
| mamba "d_inner 256, 259,425 params" | confirmed: **28 tensors, 259,425 elems**; smem **145,124 B** dynamic | `mamba3_layout.cuh:40-41,116` |
| vit "418,017 params, 17 pos" | confirmed: **32 tensors, 418,017 elems**; smem **188,080 B** dynamic | `vit_layout.cuh:46-47,137` |
| "wgmma machinery exists / warpgroup utils" | **No in-kernel wgmma anywhere.** Every model-stage GEMM is a **scalar fp32 owner-computes triple loop** (`dec_linear`, `vit_linear`, `mb_linear`). The only WGMMA path is **host-launched CUTLASS** in `mma.cuh` (not device-callable). `wgs::` has mbarrier / setmaxnreg / elect only — no MMA. cp.async + DSMEM-reduce + L2-persist exist in `primitives.cuh`; **TMA tensor-map / `cp.async.bulk.tensor` do NOT exist** | grep: only hit for `wgmma` outside `mma.cuh` is `compile.py` (a feature-detect string); `model_stage_*` scalar loops at `dec_linear` `model_stages_decoder.cuh:289`, `vit_linear:299`, `mb_linear:307` |

**Consequence of the last row** — this is the design's pivot and is developed in §1.

---

## 1. The regime, and the two architectural forks (read this first)

### 1.1 The regime split: current = memory-bound; post-R2 decoder/vit = compute-bound; mamba = memory-bound. Verified.

The current scalar cells are memory-bound (on the 223 MB partial workspace, §1.2).
After Fork B removes the partials (§3), the **decoder and vit move COMPUTE-bound**
(decoder AI ≈ 900 FLOP/B ≫ the 295 FLOP/B bf16 ridge — §10.4), so their roof is
GEMM-limited and the §3 short-K/occupancy=1 achievable-roof honesty is what governs.
**Mamba stays memory-bound** intrinsically (the selective scan, AI ≈ 0.86 FLOP/B —
§8). The compute floors below set the prize; the per-GEMM roofs (§3) set what is
actually reachable.

Decoder fwd+bwd FLOPs/step (computed — `python3`, shown):
- per-layer MACs = in_proj(S·3d·d) + scores(H·S·S·dh) + ctx(H·S·dh·S) + out_proj(S·d·d) + ff0(S·dff·d) + ff2(S·d·dff); head(1·V·d).
- fwd = L·per_layer + head = **1,593,728 MAC/sample**; bwd ≈ 2× fwd (dX + dW) ⇒ fwd+bwd = 3× fwd = **9,562,368 FLOP/sample** (MAC=2 FLOP).
- × B=4704 ⇒ **44.98 GFLOP/step** (matches the briefing "tens of GFLOP").

| precision | peak (TF/s) | decoder compute floor / step |
|---|---|---|
| bf16 TC | 989.4 | **45.5 µs** |
| tf32 TC | 494.7 | 90.9 µs |
| fp32 CUDA | 66.9 | **672 µs** |

Source peaks: `tuning/roofline.py:44-54`.

**The current scalar fp32 design is pinned near the 672 µs CUDA-core floor and in
practice far below it** (one sample per CTA, no ILP, no vectorization, no TC).
The bf16-TC floor is **14.8× below** the fp32-CUDA floor — that gap is the prize.

### 1.2 But the dW partial workspace already costs 3× the bf16 floor

The current cells use a `[nCTA × total]` per-CTA grad-partial buffer
(`fused_decoder_megakernel.cuh:112` `tok.workspace + cta*total`), reduced in P2:
- nCTA=132 (H100 SMs), total=422,755, fp32 ⇒ **223 MB**.
- P1 writes it, P2 reads it ⇒ **446 MB / step → 133 µs** at HBM3 3.35 TB/s.

**133 µs is ~3× the 45 µs bf16 compute floor.** So even a perfect TC GEMM leaves
the decoder cell memory-bound on the partials alone. **Q2 (eliminate the partials)
is co-primary with Q1 (the TC restructure), not a follow-up.** The same holds for
vit (132×418,017×4 = 221 MB) and is the dominant cost for mamba (132×259,425×4 =
137 MB; mamba's GEMMs are already memory-bound — §8).

### 1.3 The Q1/Q2 duality — you cannot eliminate both global acts AND partials

This is the load-bearing constraint the whole doc is organized around.

- The full token tensor does **not** fit smem: T·d·2B = 18,816·128·2 = **4.8 MB ≫
  227 KB**. So "activations smem-resident" is only ever achievable **per-CTA on a
  token-tile**, never globally.
- The weight-grad contraction dW = Xᵀ@dY sums over the **token dimension T**,
  which is **spread across CTAs** (each CTA owns a token slice). Therefore:
  - **Either** acts stay CTA-local in smem (a token-tile) **and** dW needs a
    **cross-CTA reduction** (→ the partials, 223 MB),
  - **or** dW is **output-stationary** — each CTA owns dW tiles and contracts the
    **full T itself** (Q2, no partials) — **but then acts must live in HBM** so the
    dW-owning CTA can read every token's activation.

You get one or the other per GEMM, chosen by traffic. Naming this duality is the
honest answer to Q1+Q2; "both at once" is impossible.

**Fork A — CTA-local token-tile (M_tile≈128), wgmma on the local tile, keep the
current recompute-in-bwd discipline.** No global acts; no new inter-layer grid
barriers (a CTA stays local through all layers, exactly as today). **But dW still
needs the cross-CTA reduction → keeps the 223 MB partials (133 µs).**

**Fork B — dW-output-stationary (Q2).** Eliminates the 223 MB partials. fwd writes
acts to HBM; bwd dW-tile CTAs stream the full T. Acts traffic (decoder, bf16,
rough): ~2 `[T,d]` tensors/layer × L = 9.6M elems × 2B × (write+read) = **39 MB →
11.5 µs** (computed). vit/mamba scale with d_inner but stay tens of MB in bf16.

**Recommendation: Fork B for all three models.** 39 MB (11.5 µs) of acts traffic
replaces 446 MB (133 µs) of partials traffic — an **11× traffic reduction on the
dominant term**, and it deletes one whole grid-barrier phase (P2 reduce, B1). The
acts round-trip is the documented cost; it is an order of magnitude smaller than
the partials it replaces. Fork A is retained only as the fallback for any GEMM
whose dW output tile is so small that output-stationary under-occupies (none in the
race — see §3).

> **Note on the briefing's Q1 phrasing.** The briefing's "layer-by-layer batched
> GEMMs across T with a grid barrier between each layer GEMM, acts in HBM" is one
> realization of Fork B, but **a global per-layer barrier is NOT mandatory** and is
> rejected: it would serialize the grid L×(#GEMMs) times per step. The chosen
> structure (§2) keeps each CTA's fwd→bwd **local and barrier-free within the
> token-tile**, with grid barriers only at the genuine cross-CTA couplings
> (embedding scatter, optimizer). Code/analysis wins over the context note.

---

## 2. The persistent-kernel phase structure (Fork B), per step

One launch, gridDim = #SMs, 256 threads/CTA (two Hopper warpgroups). Barriers are
the existing `GridBarrier::sync` / `sync_reset` (`megakernel_common.cuh:144,209`).

```
P0  zero embedding-grad accumulators (tok/pos only — the ONLY tensors needing a
    cross-CTA scatter; all weight dW are output-stationary so need no pre-zero).
--- B0 (sync_reset) ---
P1  FWD+BWD, token-tile-parallel. Each CTA owns a contiguous token-tile of M_tile
    rows (M_tile=SG_TUNED_TILE_M, default 128). For its tile it runs the full
    fwd then bwd through all layers, wgmma on bf16 smem tiles, fp32 accumulators.
      - forward activations needed by bwd are written to HBM acts[T, ...] (bf16)
        OR recomputed (the current recompute discipline still applies per tile).
      - dW: each CTA accumulates its tile's contribution into its OWNED dW output
        tiles by contracting its M_tile rows; see §3 for the split-K ownership that
        makes this a full-T contraction without partials.
      - embedding grads (tok/pos): index-scatter into the P0 accumulators by the
        deterministic owner map of §3.4.
      - loss: per-CTA fp32 NLL slot (kept — `fused_decoder_megakernel.cuh:147`).
--- B1 (sync_reset) ---
P2  OPTIMIZER tail. apply_optimizer<Opt> over the (now-complete, in-place) grad.
    For epilogue-fusable optimizers (§4) this phase is FUSED INTO P1's dW epilogue
    and B1+P2 collapse — 3 phases / 2 barriers for adamw/lion/grokfast/grokadamw/
    neuralgrok. Non-fusable optimizers keep P2 (and their own precompute phases).
```

**Barrier count per step, decoder, before vs after:**

| design | phases | grid barriers |
|---|---|---|
| current (scalar, partials) | 4 (P0,P1,P2-reduce,P3-opt) | **3** (B0,B1,B2; the fwd→bwd sync inside P1 is `__syncthreads`, not a grid barrier) |
| Fork B, non-fusable opt | 3 | **2** |
| Fork B, **epilogue-fused opt** (5 of 11) | 2 | **1 or 2** (see caveat) |

The dW-reduce phase (P2) + its barrier (B2) are **deleted** (no partials → nothing
to reduce): the baseline **3 → 2** barriers, a net saving of **one** grid barrier.
This is the second win of Fork B beyond traffic.

> **Embedding caveat on the "1 barrier" fused case.** Weight dW fuses to its
> optimizer in the P1 epilogue (0 extra barriers). But the **tok/pos** grads are a
> cross-CTA scatter (§3.4) that every CTA contributes to — their optimizer apply
> must wait for **all** CTAs to finish scattering, which needs **B1 before the
> tok/pos apply**. So a fully-epilogue-fused cell is **2 phases / 1 barrier for the
> ~28 weight tensors, but the 2 embedding tensors still require that one barrier** —
> i.e. the realistic fused floor is **1 barrier (B0 for the embedding-accumulator
> zero + the post-scatter rendezvous folded into it)**, not zero. The 5-fusable
> cells land at **1 barrier**; do not claim 0. 🟡

> 🟡 **Recompute vs acts-in-HBM is itself a per-model choice.** Decoder's bwd
> currently recomputes each layer from `layer_in` (`model_stages_decoder.cuh:498`
> `dec_recompute_layer`) to stay under 48 KB static smem. Under Fork B the dW CTA
> needs *other* CTAs' tile activations, so the cross-tile activations MUST be in
> HBM regardless; the within-tile recompute can stay. The build agent picks
> recompute-vs-store per GEMM by the smem budget table (§6/§9); both are
> bit-identical in the produced grad (same expression, different storage class).

---

## 3. Per-GEMM TC plan (M/N/K, wgmma tiling, ownership, achievable roof)

Conventions: wgmma atom `m64nNk16` for bf16 (Hopper ss-wgmma; N ∈ {8..256}). A
"CTA tile" is the M×N output block one CTA owns; the K loop iterates k-steps of 16.
Activations are bf16 in smem staged from HBM via cp.async (TMA in phase-2, §4).
**Achievable roof per GEMM is bounded by the mainloop k-step count** (short-K = few
MMA issues to amortize prologue/epilogue + occupancy=1 has no inter-CTA latency
hiding — the in-CTA producer/consumer pipeline must hide it). I give the roof class
honestly; I do **not** claim peak.

### 3.1 Decoder (d=128, dff=512, S=4, V=99, H=4, dh=32; B=4704, T=18,816)

The forward GEMMs are tall-skinny in T: **M = number of (sample×position) rows the
GEMM touches.** in_proj/out_proj/ff act on all S positions ⇒ M-rows = T = 18,816.
The head acts on the last position only ⇒ M-rows = B = 4704.

| GEMM | M (rows) | N | K | wgmma tiling | CTA tile (M×N) | acts location | roof class |
|---|---|---|---|---|---|---|---|
| **in_proj** (fwd) qkv = X·Wᵀ | 18,816 | 3d=384 | d=128 | m64n128k16, 3 N-tiles, **8 k-steps** | 128×128, ~147 M-tiles over 132 SMs | X in smem tile (bf16) | K=128 → mid. M huge → TC-saturated on issue; **~55–70%** 🟡 |
| **scores** q·kᵀ (per head) | S=4 | S=4 | dh=32 | **too small for wgmma** — 4×4×32. Keep **scalar fp32** per (head,qrow) exactly as now (`model_stages_decoder.cuh:364`) | — | smem | N/A (special-cased; <0.1% of FLOPs) |
| **ctx** attn·v (per head) | S=4 | dh=32 | S=4 | too small — scalar fp32, kept | — | smem | N/A |
| **out_proj** a = ctx·Wᵀ | 18,816 | d=128 | d=128 | m64n128k16, **8 k-steps** | 128×128 | smem tile | as in_proj, **~55–70%** 🟡 |
| **ff0** g = x1·Wᵀ | 18,816 | dff=512 | d=128 | m64n128k16, 4 N-tiles, **8 k-steps** | 128×128 | smem tile | M·N large; **~60–72%** 🟡 |
| **ff2** ff = g·Wᵀ | 18,816 | d=128 | dff=512 | m64n128k16, **32 k-steps** | 128×128 | smem tile | **best — K=512 amortizes prologue; ~72–82%** 🟡 |
| **head** logits = hn·Wᵀ | 4704 | V=99 | d=128 | m64n96k16 (pad N→128), 8 k-steps | 64×96 | smem | N=99 odd, padded; M=B large; **~45–60%** 🟡 |
| **bwd dX** (per linear) dX = dY·W | 18,816 | K_in | N_out | mirror of fwd, swap N↔K | 128×128 | acts HBM | same class as the fwd it mirrors |
| **bwd dW** (per linear) dW = dYᵀ·X | N_out | K_in | **T=18,816** | **m64nNk16 contracting full T**, T/16=**1176 k-steps** | dW output tile (e.g. 64×128 for ff0_w[512,128]) | **dY,X stream from HBM acts** | **K=T huge → the most TC-efficient GEMMs; ~80–88%** 🟡 |

**Why dW is the TC-friendliest and how Q2 lands here.** dW = Σ_{t∈[0,T)} dY[t,:]ᵀ
X[t,:]. In GEMM terms (M_g=N_out, N_g=K_in, **K_g=T**) the contraction is the full
token dimension. **Each CTA owns a fixed (M_g×N_g) output tile of dW and contracts
ALL T tokens itself** by streaming dY[*, n_out_tile] and X[*, k_in_tile] from the
HBM acts buffer in **fixed ascending-t k-step order**. No `[nCTA×total]` partial,
no cross-CTA reduce — the dW tile is written **once**, complete. This is the literal
Q2 deliverable.

- **Tile ownership map.** dW tensors are small (largest decoder weight = ff0_w/ff2_w
  = 512×128 = 65,536 elems = 8 output-tiles of 64×128). The 30 weight tensors yield
  ~**O(60) dW output tiles** total. Assign tiles to CTAs round-robin by a fixed
  `tile_id = atomic-free static partition` (tile_id → CTA = tile_id % nCTA), so
  every step the same CTA owns the same tile (determinism + L2 warmth). With ~60
  tiles and 132 SMs, ~half the SMs own a dW tile; **the rest are NOT idle** — they
  own dX tiles and the fwd GEMMs in P1 (the phases overlap within P1; dW and dX of
  a layer are independent given dY).
- **Determinism.** k-order is **ascending t**, fixed and identical on every run;
  one CTA owns each tile end-to-end (no work-steal split of a tile's K-loop). No
  float atomics. ✔ matches the owner's determinism rule.
- **Eliminated traffic (quantified).** Removes the 223 MB partial buffer and its
  446 MB/step round-trip (**133 µs**); adds ~39 MB/step acts (**11.5 µs**). Net
  **−122 µs/step** of HBM time on the decoder.

**What remains of P2 (the embedding scatter).** tok/pos grads are index-scatter,
not a GEMM: `g.tok[token_id·d + j] += dh` and `g.pos[s·d + j] += dh`
(`model_stages_decoder.cuh:981-988`). These collide across tokens/CTAs and **cannot
be output-stationary** (the output row depends on data-dependent `token_id`).
**Deterministic owner map (Q2 tail):**
  - **tok** (99 rows × 128): owner = `row j of tok-tensor → CTA (row % nCTA)`. Each
    owner CTA, after B0, loops **all T tokens in ascending t**, and for each token
    whose id maps to a row it owns, accumulates `dh[t,:]`. Fixed t-order + single
    owner per row ⇒ deterministic, atomic-free. Cost: each CTA scans T token-ids
    (cheap, T=18,816 int reads, ~75 KB) — but only accumulates for its rows.
  - **pos** (4 rows × 128): trivially 4 owner-CTAs, each sums dh[t, s_owned] over
    its tokens. (mamba pos=8, vit pos=17 — same pattern.)
  - This is the ONLY surviving cross-CTA accumulation; it needs **P0 zero + B0**
    (the embedding accumulators) and is folded into P1's tail (no extra barrier).

### 3.2 ViT (d=128, dff=512, kSeq=17, kNPatch=16, kPatch=49, V=97)

Same six GEMM families as decoder, **M-rows = T = B·kSeq = 4704·17 = 79,968**
(every position contributes — head uses CLS pos 0 only, M=B). Attention is **FULL
(no causal mask)** (`model_stage_vit.cuh:346`), so scores/ctx are S=17 square — still
too small for wgmma (17×17×32), kept scalar. **Extra GEMM: patch_proj** (per-patch
embed, Linear(49→128)) — M=B·16=75,264, N=128, K=49 → m64n128k16 with K=49 padded
to 64, 4 k-steps; low roof (~40–55% 🟡, short K, odd pad) but small FLOP share.

ViT dW for ff0_w/ff2_w (512×128) contracts T=79,968 (T/16 = **4,998 k-steps**) —
even more TC-efficient than decoder (**~82–88%** 🟡). ViT FLOPs/step ≈ decoder ×
(79,968/18,816) on the per-position GEMMs ≈ **~190 GFLOP/step** 🟡 (rough; head and
patch_proj differ) ⇒ bf16 floor ≈ **~190 µs**. ViT is the most TC-bound of the
three and benefits most from R2.

### 3.3 Mamba (d=128, d_inner=256, state=16, dt_rank=8, kSeq=8, V=97) — §8

Projection GEMMs (in_proj 128→512, x_proj 256→40, dt_proj 8→256, out_proj 256→128),
M-rows = T = B·8 = 37,632. These ARE wgmma-able (in_proj K=128, out_proj K=256).
**But the selective scan (the core) is NOT a GEMM** — it is a sequential recurrence
over seq=8 per channel, held in registers (`mb_scan_fwd:487`). §8 shows mamba is
**memory-roof-bound**, so TC on the projections is a secondary win; the primary
mamba wins are bf16 storage + Q2 + tiling the proj GEMMs around the register scan.

---

## 4. TMA + pipeline (producer/consumer warpgroups, smem budget, mbarrier)

### 4.1 What exists vs what is new (substrate work)

**Exists, reuse as-is:** `wgs::Mbarrier` (arrive_expect_tx / try_wait / parity,
`warp_specialize.cuh:72`), `wgs::warpgroup_reg_{alloc,dealloc}<N>`
(setmaxnreg, :159), `wgs::elect_one_sync` (:41), `wgs::fence_async_proxy` (:143),
`cp_async_cg_16` / commit / `wait_group` (`primitives.cuh:497`), DSMEM cluster
reduce (`cluster_reduce_sum_f32_dsmem:629`), `GridBarrier` (`megakernel_common.cuh`).

**New substrate (the genuinely hard, SASS-verify-at-instruction-level part):**
1. **In-kernel ss-wgmma wrapper** `wgs::wgmma_m64nNk16_bf16(acc[], descA, descB)` —
   the `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16` PTX + the
   `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group` discipline. Lives in a
   **new** `csrc/backends/cuda/sm_90/wgmma.cuh` (substrate, arch-guarded; gfx942
   and pre-sm_90 fall back to the scalar `*_linear`). Verified at SASS: `cuobjdump
   -sass` must show `WGMMA.*` and `wgmma.wait_group`.
2. **smem core-matrix descriptors** — the swizzled smem layout + the 64-bit matrix
   descriptor (start-addr/leading-dim/stride/swizzle-mode encode) the ss-wgmma
   operands require. This is the part with no existing analogue and is the main
   implementation risk; specify the 128-byte-swizzle layout for bf16 K-tiles.
3. **TMA (phase-2 only):** host-built `cuTensorMapEncode` descriptors +
   `cp.async.bulk.tensor.2d.shared::cluster.global` for the acts/weight staging.
   **Not a phase-1 prerequisite** — phase-1 uses the existing cp.async (`.cg.16`)
   for staging, which already overlaps load latency. TMA is a measured follow-up
   (it reduces address-gen overhead and enables the multicast for shared weights).

### 4.2 Producer/consumer structure per CTA

The 256-thread block is already split into WG0 (producer, regs→32) and WG1
(consumer, regs→200) via setmaxnreg (`fused_megakernel.cuh:257-262`,
`fused_decoder_megakernel.cuh:118-119`). **Keep that split.** New role binding:
- **WG0 (producer):** `elect_one_sync` lane issues cp.async (phase-1) / TMA
  (phase-2) of the next bf16 A/B K-tile into the smem ring buffer; `arrive_expect_tx`
  on the stage's mbarrier.
- **WG1 (consumer):** `try_wait(parity)` on the mbarrier, then issues the
  `wgmma.mma_async` over the staged tile into fp32 accumulators; flips parity.
- **One `fence_async_proxy` per stage hand-off** (the documented >10%-regression
  footgun if loose — `warp_specialize.cuh:24`).

Pipeline depth = `SG_TUNED_PIPE_DEPTH` (default 2 = double-buffer; 3 = triple if
smem allows). Each stage buffers one A K-tile + one B K-tile.

### 4.3 smem budget per model vs the 227 KB cap (the occupancy gate)

bf16 A/B K-tiles, m64n128k16, 128-row M-tile: A tile = 128×16×2B = 4 KB; B tile =
128×16×2B = 4 KB (for N=128). Per pipeline stage = ~8 KB. Depth 2 = ~16 KB; depth 3
= ~24 KB. Plus the fp32 accumulator fragments live in **registers** (not smem).

| model | activation smem (current) | + bf16 halving (§5) | + TC pipeline (depth 2) | total | vs 227 KB | occ ≥ 1? |
|---|---|---|---|---|---|---|
| decoder | ~42 KB static (`fused_decoder_megakernel.cuh:107`) | ~21 KB | +16 KB | **~37 KB** | ✔ huge margin | ✔ (also clears 48 KB static cliff → could stay static) |
| mamba | 145,124 B dynamic (`mamba3_layout.cuh:116`) | ~73 KB | +16 KB | **~89 KB** | ✔ | ✔ |
| vit | 188,080 B dynamic (`vit_layout.cuh:137`) | **~94 KB** | +16 KB | **~110 KB** | ✔ | ✔ — **but ONLY after bf16 halving** |

**vit is the binding constraint and proves the dependency chain
`Q5(bf16 acts) → Q4(pipeline fits) → Q1(TC on vit)`:** 188 KB acts + 16 KB pipeline
= 204 KB leaves only 23 KB for everything else at occ=1 — fragile. **Halving acts to
~94 KB is what makes the vit TC pipeline placeable.** bf16 activations is a
**prerequisite for vit R2, not a nicety.** The launcher's occupancy-refuse
(`cudaErrorLaunchOutOfResources`, `fused_megakernel.cuh:349`) MUST be re-verified per
model after adding pipeline buffers + the 200-reg consumer; this is a **gate in the
budget table, not a footnote**.

### 4.4 GridBarrier vs mbarrier vs clusters

- **Keep `GridBarrier` for the cross-PHASE rendezvous** (P0/P1/P2 boundaries). It
  needs no cooperative launch (scales past the cooperative CTA cap) and is the whole
  reason the design avoids CUDA graphs. The `__nanosleep` backoff
  (`megakernel_common.cuh:180`) stays.
- **mbarrier is for the intra-CTA producer/consumer hand-off only** (§4.2) — a
  different scope; they compose, not compete.
- **Clusters / DSMEM:** **do NOT use for the dW reduce** (Fork B has no reduce). The
  only candidate is sharing a *weight* K-tile across a 2-CTA cluster via TMA
  multicast (phase-2). Cap cluster ≤ 2 inside a persistent grid (large clusters
  starve resident slots — `primitives.cuh:594`). Portability: the cluster path is
  arch-gated; gfx942 substrate keeps the generic (no-cluster) reduce. ✔ contract
  allows per-arch substrate impls.

---

## 5. bf16 policy (the dtype boundary table) + parity gate

### 5.1 Op-level dtype boundary (matches torch autocast)

The owner's rule: bf16 storage + matmuls, **fp32 accumulators**, and autocast keeps
softmax/LN/CE/optimizer math fp32. Exact table:

| op | inputs | compute | output stored | rationale |
|---|---|---|---|---|
| in_proj / out_proj / ff0 / ff2 / head / patch_proj (and all dX/dW) | **bf16** | wgmma, **fp32 acc** | bf16 acts / **fp32 grad** | autocast matmul boundary; grad stays fp32 for the optimizer |
| attention scores / softmax | bf16 q,k | **fp32** (dot + softmax) | bf16 weights | autocast keeps softmax fp32; tiny S so cost negligible |
| LayerNorm (fwd+bwd) | bf16 x | **fp32** (mean/var/normalize) | bf16 y; **fp32 xhat/inv cached** | autocast LN fp32; the two mean-correction terms need fp32 |
| GELU / SiLU / softplus | bf16 | **fp32** transcendental | bf16 | autocast elementwise upcast; exact-erf GELU (`dec_gelu:99`) |
| cross-entropy / NLL | fp32 logits | **fp32** (logsumexp), **fp64** loss reduce | fp32 grad | the 1e-5 loss rel-tol is the tightest gate (`fused_decoder_megakernel.cuh:171`) |
| selective scan (mamba) | bf16 x_main, B, C | **fp32 scan state in registers** | bf16 y_scan | scan is memory-bound; fp32 state preserves the recurrence (`mb_scan_fwd:492`) |
| optimizer apply (all 11 tails) | fp32 grad + fp32 state | **fp32** | fp32 params + state | autocast never touches the optimizer; FusedOptState is fp32 |

**Storage halving (the §4.3 budgets):** every `[kSeq×width]` smem activation array
in `DecSampleSmem` / `VitSampleSmem` / `MambaSampleSmem` becomes `__nv_bfloat16`;
the **LN caches (xhat, inv_std) and the fp32 reduction scratch `red[256]` stay
fp32** (they feed fp32 LN bwd). Decoder ~42→~21 KB, vit ~188→~94 KB, mamba ~145→~73
KB (computed in §4.3). New per-model smem budgets are the "+ bf16 halving" column.

### 5.2 Parity gate (chaos-floor methodology — read the calibrated-gate comments)

The existing gate (`tests/hw/test_megakernel_vs_eager.py`) is explicit that an
**L3 loss-trajectory-vs-eager test is ill-posed** for the surrogate path (:8-16) and
that the faithful unit is the optimizer tail + a **grok smoke floor** (:27-31). For
the **real** TC cells the bf16 gate is therefore **tiered**, matching the L3-REAL
decoder gate at `:288-294`:

1. **Single-step grad parity vs eager bf16-autocast** (NOT fp32 eager): build the
   eager model under `torch.autocast(bf16)`, one fwd+bwd, compare every weight grad.
   **Tolerance: rel ≤ 2e-2** 🟡 (bf16 has ~3 mantissa bits; this is the autocast
   round-off class, far looser than the fp32 gate's `1e-4` at `:292`). Loss rel ≤
   `1e-3` (logsumexp is fp32 so it stays tight).
2. **Determinism A/B/A:** run the cell 3× (A, B, A); assert run A == run A
   **bit-identically** (fixed tile ownership + fixed k-order + no float atomics
   guarantee it). B may differ from A only if B perturbs scheduling — it must not,
   so A==B==A bit-exact. This is the real witness that §3's determinism holds.
3. **Grok-floor (the chaos-robust gate):** 3 seeds, real model + the bf16-TC fused
   train step, must reach the grokking accuracy threshold (mirrors `:294` "must
   grok"). Because bf16 + reordered fp32 reductions make the **loss trajectory**
   chaotically diverge from eager (sensitive dependence — the calibrated-gate point
   at `:12-16`), the final-accuracy floor — not the trajectory — is the correct
   acceptance criterion for the bf16 cells.

> 🟡 The 2e-2 grad tolerance is my estimate from the bf16 mantissa width; the build
> agent must **calibrate it against an eager-bf16 oracle** the same way the fp32
> gate was calibrated against fp64 (`:349-364`), and tighten if the oracle is closer.

---

## 6. R1 quick wins (pre-TC) — recommendation: do R1 first, then R2

R1 stays inside the **current scalar design**, lands in days, and **every R1 piece is
reused wholesale by R2**:

| R1 win | mechanism | expected multiplier | reused by R2? |
|---|---|---|---|
| **k-samples-per-CTA** (vs one-sample-at-a-time) | process K_samp rows per CTA inner loop, unroll the owner-computes GEMM loops over K_samp → ILP + register reuse of weights | **~3–6×** 🟡 (the current `dec_linear` does 1 row's dot at a time with zero cross-row reuse; batching K_samp amortizes the weight loads and fills the pipeline) | yes — becomes the M_tile of §3 |
| **split-K dW without TC** | the Q2 output-stationary dW (§3.1) but with scalar fp32 contraction over full T | **kills the 223 MB partials → −133 µs** | yes — same ownership map, just swap the inner loop for wgmma |
| **bf16 scalar storage** | the §5 dtype table, scalar loads/stores | **~2× on the memory-bound activation traffic**; halves smem (enables R2 placement on vit) | yes — identical dtype boundary |

**Recommendation: R1 first (≈3–5 days), then R2.** Reasoning:
- R1's split-K dW and bf16 storage are **prerequisites R2 needs anyway** (the
  ownership map, the dtype boundary, the halved smem that makes vit placeable). Doing
  them in the scalar design **de-risks them independently** of the hard wgmma
  descriptor work.
- The integration agent currently landing vit/mamba cells builds on the **current**
  design; R1 is **incremental** on that (k-samples + split-K + bf16 are localized
  edits to the existing `*_linear` / dW loops and the phase structure), whereas R2 is
  a GEMM-engine rewrite. R1 churn is low and additive; an R2-first rewrite would
  collide with in-flight integration.
- R1 alone is expected to move the decoder from "near the fp32-CUDA floor in
  practice" to **single-digit× the bf16 floor** (split-K removes the 3× partials
  overhead; bf16 halves activation traffic; k-samples fills the ALU pipe). That is a
  large fraction of R2's win for a fraction of the risk — and is measurable before
  committing to wgmma.

**Do not skip R1.** The only R1 piece R2 *replaces* is the scalar inner dot (→
wgmma); everything else R2 *keeps*.

---

## 7. Sort / SG2 (in-CTA bitonic sort + sequence-space partitioning)

### 7.1 In-CTA bitonic sort (eliminate the host round-trip)

SG2 currently sorts each tensor's flat grad by **|grad| ascending** on the **host**
(`torch.sort`, `opt_stage_supergrok2.cuh:34-42`) and feeds `perm`/`unsort` into the
kernel. **Design: a STAGE -1 in-CTA bitonic sort** so the whole step is launch-free.
- Each CTA owns a whole tensor (tensor-per-CTA, `opt_stage_supergrok2.cuh:64-66`),
  so the sort is **per-CTA over that tensor's N keys** — no cross-tensor crossing
  (the exact correctness hazard the header flags at :38-42).
- **Bitonic sorting network** — the standard nested form: outer `k` doubles 2→N
  (`ceil(log2 N)` levels), inner `j` halves k→1, giving **~½·(log₂N)²
  compare-exchange passes** total (N=512 → ~45 passes, N=65,536 → ~136 — NOT
  log₂N), each pass `__syncthreads`-separated over N/2 pairs (threads stride). Keys =
  `|grad[i]|` (fp32), payload = original index i → produces `perm`; `unsort` is the
  inverse scatter.
- **Determinism:** bitonic is a **fixed comparison network** (data-independent
  schedule) → deterministic. Ties: break by **ascending original index** (stable),
  matching the host sort's stable-sort contract. No atomics.
- **Cost/feasibility:** largest race param ~65,536 elems → ~136 compare-exchange
  passes × ~33K compares/pass = trivial vs the CSA/HCA attention that follows (still
  O(N·passes) ≪ the dense HCA's O(N·Nh·heads·head_dim)). Keys fit a per-CTA smem
  scratch (65,536×4 = 256 KB — **too large for one CTA's smem**); so for N > ~8192
  the sort is **global-memory bitonic** (compare-exchange in the tensor's HBM slice,
  `__syncthreads` per stage) — still launch-free, just not smem-resident. 🟡 For the
  race's small tensors (most are ≤512) the smem path applies. Drop-in as STAGE -1;
  the header is already written to accept it (`opt_stage_supergrok2.cuh:55-57`).

### 7.2 Sequence-space partitioning of the SG2 meta-net (replace tensor-per-CTA)

The current SG2 stage is **tensor-per-CTA** (one CTA does a whole tensor's
CSA/HCA/PEER/GRU), a conscious correctness-first divergence from design-req-#1
(`opt_stage_supergrok2.cuh:97-119`) whose **cost** is owned: with ~30 tensors and
~132 SMs, ~100 CTAs pull an empty queue and idle, and one 256-thread CTA does each
tensor's entire (dense) HCA attention serially.

**Design: sequence-space partitioning for the LARGE tensors.** The stage bodies
**already take explicit N/Nc/Nh and use CTA-cooperative grid-stride loops** (the
header author built for this — `:117-119`), so **only the outer driver changes**:
- For a large tensor, spread its N rows across **a cluster of CTAs** (or the whole
  grid for the single largest), with **one grid barrier after the compress stage**
  (so all compressed K/V are visible before attention/topk read them — the genuine
  cross-CTA coupling the header names at :100-104).
- Small tensors (which dominate the race's param *count*) stay **tensor-per-CTA**
  (one CTA, barrier-light) — a size threshold `SG_TUNED_SG2_SEQPART_MIN` routes.
- **Determinism preserved:** the compress/attention reductions are still
  owner-computes per output element in fixed order; the grid barrier only orders
  visibility, not summation order.
- This reuses the **same stage bodies** verbatim — the change is the row-ownership
  in the driver + one barrier, exactly the refactor the header was written to admit.

---

## 8. Mamba scan — confirm memory-roof-bound, design only what helps

**Arithmetic intensity of the scan (computed reasoning).** Per (sample, channel j),
the scan does, over seq=8 and state=16: ~8×16 FMAs for h-update + 8×16 for y-accum
≈ **256 MAC = 512 FLOP/channel**. It reads x_main[8], dt_pre[8], Bmat[8×16],
Cmat[8×16], A_log[16] ≈ (8+8+128+128+16) bf16 = 288×2 = **576 B/channel**, writes
y_scan[8] = 16 B. AI ≈ 512 FLOP / 592 B ≈ **0.86 FLOP/B**. The roofline ridge for
bf16-TC is 989.4e12/3.35e12 ≈ **295 FLOP/B**. **0.86 ≪ 295 → the scan is deeply
memory-roof-bound** (by ~340×). This confirms the briefing and the header's own
"memory-roof-bound" framing.

**Therefore for mamba:**
- **Do NOT** try to TC the scan (it is not a GEMM and is memory-bound anyway).
- **Keep fp32 scan state in registers** (`mb_scan_fwd:492` `h[kState]`,
  `mb_scan_bwd` `hh[kSeq+1][kState]`) — the seq=8 register exploit is correct and is
  what keeps smem off the 128 KB the naive h-in-smem would need (`model_stage_mamba3.cuh:74-88`).
- **bf16 storage** for the scan's inputs/outputs (x_main, B, C, y_scan) — halves the
  576 B/channel traffic → the direct multiplier on the memory-bound scan.
- **TC the projection GEMMs around the scan** (in_proj 128→512, out_proj 256→128,
  x_proj 256→40) — these ARE GEMMs (§3.3), M-rows=T=37,632, and tiling them with
  wgmma + Fork-B dW is the real mamba compute win. The scan sits between them as a
  register-resident barrier-free stage.
- **conv1d** (depthwise k=3, `mb_conv1d_fwd:420`) is also memory-bound (3 MACs/elem)
  → bf16 storage, kept scalar.

Net mamba story: **Q5 (bf16) + Q2 (split-K dW) + tiled proj GEMMs**; the scan itself
is left as the proven register recurrence, only its operands go bf16.

---

## 9. Autotuner dims (new `-DSG_TUNED_*`, per-TU defaults)

These ride the **existing kernel-autotuner mechanism** (`AUTOTUNE_LINKAGE.md`:
`compile.py` → `_kernel_tuned.json` → `setup.py TunedBuildExtension` injects per-TU
nvcc flags onto `launch_<opt>.cu` and `mega_<model>_<opt>.cu`). The current four
SAFE dims are block/vec/unroll/async_depth + maxrregcount (`AUTOTUNE_LINKAGE.md:21`).
The doc's note that "TMA/WGMMA/cluster macros are component-scoped, riskier, phase-2,
NOT emitted by the current pass" (`:30-31`) means these **new dims must be added to
`MACROS` in `_tuned_inject.py`** and to the per-TU pass (lockstep with the header
`#ifndef` guards — the unit test `tuning/test_build_injection` fails on drift,
`AUTOTUNE_LINKAGE.md:223-227`).

| new macro | meaning | per-TU default (`#ifndef`) | search range |
|---|---|---|---|
| `SG_TUNED_TILE_M` | token-tile rows per CTA (the M_tile of §3) | **128** | {64,128,256} |
| `SG_TUNED_TILE_N` | wgmma N (output tile width) | **128** | {64,128,256} |
| `SG_TUNED_PIPE_DEPTH` | producer/consumer ring stages (§4.2) | **2** | {2,3,4} (smem-capped) |
| `SG_TUNED_WG_COUNT` | warpgroups/CTA (producer+consumer split) | **2** | {2} fixed phase-1; {2,4} phase-2 cooperative |
| `SG_TUNED_K_SAMPLES` | samples-per-CTA in the R1 scalar path (§6) | **4** | {1,2,4,8} |
| `SG_TUNED_CLUSTER_DIM` | cluster size for TMA-multicast weights (phase-2) | **1** | {1,2} (≤2 in persistent grid, `primitives.cuh:594`) |
| `SG_TUNED_SG2_SEQPART_MIN` | min tensor N for SG2 sequence-space partition (§7.2) | **8192** | {4096,8192,16384} |

**Absent definitions yield a correct untuned kernel** (contract rule 3,
`COMPONENT_CONTRACT.md:28-31`): default M_tile=128, depth=2, etc. all compose into
a valid cell. **Do not touch the Optuna optimizer-hyperparameter tuner in `tuning/`**
— that is a separate system (the briefing and `AUTOTUNE_LINKAGE.md` both stress this;
this is the **kernel** autotuner).

---

## 10. Validation + measurement plan

### 10.1 Per-phase gates

| phase | gate |
|---|---|
| R1 split-K dW (scalar) | grad parity vs current cell **bit-identical** (same fp32 math, just output-stationary; ascending-t order matches the old ascending-CTA reduce *only if* both sum T in the same order — verify, the order changed from CTA-major to t-major so it is fp32-rel ≤ 1e-5, **not** bit-exact) 🟡 |
| R1 bf16 storage | grad parity vs eager-bf16, rel ≤ 2e-2 (§5.2); grok-floor 3 seeds |
| R2 wgmma per GEMM | **SASS audit** (§10.3) + single-step grad parity vs eager-bf16 |
| R2 full cell | determinism A/B/A bit-exact; grok-floor 3 seeds; occupancy-refuse test (force a too-large smem config, assert `cudaErrorLaunchOutOfResources`, **no hang**) |
| recomposition (every phase) | all 33 cells regenerate from `megakernel_codegen.py`; `git diff` generator-consistent (`COMPONENT_CONTRACT.md:5,37`); out-of-tree portability TU compiles (rule 4) |

### 10.2 Roofline protocol per cell (the owner's metric)

Reuse `tuning/roofline.py`'s **timed-FLOPs methodology** (ncu blocked —
ERR_NVGPUCTRPERM, `roofline.py:5`): achieved = FLOPs_per_step / wall_per_step;
wall = CUDA-synced over 100 steps after 25 warmup, **quiet-GPU window** (no other
process — the 3 grok-smoke + tuner fleet must be idle for the measurement window).
Ceiling = min(bf16_tc_peak, AI·HBM_BW); fraction = achieved/ceiling. **Per-pipeline
precision = bf16** now (the cells compute bf16), so the ceiling is the **989.4 TF/s
BF16 line** (`roofline.py:287`, `_PRECISION_PEAK['bf16']`), not tf32.

### 10.3 SASS audit checklist (per cell .so)

`cuobjdump -sass <mega_model_opt>.o | grep -E ...`:
- ✔ `WGMMA` present (the in-kernel tensor-core MMA — absence = scalar fallback shipped).
- ✔ `wgmma.wait_group` / `wgmma.commit_group` present (pipeline correctness).
- ✔ cp.async (`LDGSTS`) present (phase-1 staging) / `UTMALDG` (phase-2 TMA).
- ✔ `setmaxnreg` (`R2UR`/the warpgroup reg ops) present (producer/consumer split).
- ✔ **no `STL`/`LDL` spills** in the consumer wgmma loop (register pressure check;
  the `--maxrregcount` tuned dim exists to fix this if it appears).

### 10.4 Step-time budget table (projection; decoder, bf16)

| component | µs (projected) | basis |
|---|---|---|
| fwd+bwd GEMMs (TC) | ~60–90 µs 🟡 | 45 µs bf16 floor ÷ ~0.5–0.75 achievable roof (§3 per-GEMM classes, occ=1) |
| acts HBM round-trip (Fork B) | ~11.5 µs | 39 MB / 3.35 TB/s (computed §1.3) |
| grid barriers (2, Fork B) | ~2–4 µs 🟡 | 2 × ~1–2 µs (`__nanosleep` backoff rendezvous, no co-residency stall at occ=1) |
| elementwise (LN/GELU/softmax/CE, fp32) | ~10–20 µs 🟡 | small S; dominated by the 99-wide CE logsumexp × B |
| optimizer tail (fused, adamw) | ~5–8 µs 🟡 | 422,755 elems × (read g + r/w m,v + w param) ≈ 8.5 MB / 3.35 TB/s ≈ 2.5 µs + overhead |
| **total / step** | **~90–135 µs** 🟡 | → **~7,500–11,000 steps/s** |
| **roofline fraction** | **~35–50%** 🟡 | achieved (45 GFLOP / ~110 µs ≈ 410 GF/s) / 989 TF/s — memory-bound, so read against AI·BW: at AI≈45GFLOP/50MB≈900 FLOP/B the cell is **compute-bound** and the fraction is GEMM-roof-limited ~0.5 |

> 🟡 The whole budget is a **no-silicon projection** (no GPU runs permitted here; the
> 3 grok-smoke + tuner processes are live). The achievable-roof column is the honest
> uncertainty: short-K GEMMs at occ=1 will not hit peak. The build agent measures
> the real numbers via §10.2 and the SASS audit confirms the instructions are
> present.

---

## 11. Effort / sequencing (day-level; substrate vs per-header)

| phase | work | substrate vs header | recompose+validate | days 🟡 |
|---|---|---|---|---|
| **R1.1** | k-samples-per-CTA in `dec_/vit_/mb_linear` + dW loops | per-header (3 model headers) | regen 33 cells, parity + grok-floor | 1–2 |
| **R1.2** | split-K output-stationary dW (kills partials) + embedding owner-map | per-header (3) + phase struct in 3 `fused_*_megakernel.cuh` | bit/rel grad parity, drop B1 phase | 2 |
| **R1.3** | bf16 scalar storage (§5 dtype table) | per-header (3 SampleSmem structs) | eager-bf16 parity (calibrate tol), re-check smem budgets | 1–2 |
| **R2.1** | new `wgmma.cuh` substrate (ss-wgmma PTX + smem descriptors) + SASS verify on a microbench | **substrate** (1 new header) | SASS audit (WGMMA present) | 3–5 (the hard part) |
| **R2.2** | producer/consumer pipeline (mbarrier ring + cp.async staging) in a shared GEMM helper | **substrate** (extend `wgs::`) | per-GEMM grad parity | 2–3 |
| **R2.3** | wire wgmma GEMMs into the 3 model headers (replace scalar dots; keep scores/ctx/scan scalar) | per-header (3) | full-cell parity + determinism A/B/A + roofline | 2–3 |
| **R2.4** | TMA (host descriptors + `cp.async.bulk.tensor`) — phase-2 optional | substrate + autotuner dims | SASS (UTMALDG), roofline delta | 2–3 |
| **R2.5** | SG2 in-CTA bitonic sort (STAGE -1) + sequence-space partition driver | per-header (`opt_stage_supergrok2.cuh` driver) | SG2 parity (1e-5) + determinism | 2–3 |
| **autotuner** | add 7 `SG_TUNED_*` to `_tuned_inject.py` MACROS + pass | tooling | `tuning/test_build_injection` | 1 |

**Substrate work** (inherited by all 33 cells): `wgmma.cuh`, the `wgs::` pipeline
extension, the TMA descriptors, the new autotuner macros. **Per-header work** (lands
once per model, recomposed across all 11 optimizers): the k-samples / split-K / bf16
edits and the wgmma wiring in the 3 `model_stage_*.cuh` + 3 `fused_*_megakernel.cuh`.
**Optimizer headers** are largely untouched except §4's epilogue-fusion (§ below).

**Recomposition cost per phase** is one `megakernel_codegen.py` run + the parity
suite + a quiet-window roofline pass; it is dominated by the **per-cell CUTLASS-free
nvcc compile** (the cells are header-only; no CUTLASS in the L3 path, so the heavy
`mma.cuh` TUs are NOT rebuilt). Heavy-TU serial-compile OOM (`mma.cuh:384-396`) does
**not** apply to the L3 cells — another reason the in-kernel wgmma path is preferred
over dragging CUTLASS device-side.

---

## Appendix A — Optimizer epilogue-fusion verdict table (Q3)

The dW tile is **register-resident** at the end of P1's dW GEMM (the fp32
accumulator fragment). An optimizer whose apply is **element-local** can run **in
that epilogue**, before the fragment is stored — eliminating B1+P2 (the grad never
leaves registers for those tails). Verdict per the 11 (apply math is
`opt_components.cuh:184-254`):

| optimizer | element-local apply? | epilogue-fusable? | what it needs before apply | fallback if not fused |
|---|---|---|---|---|
| **AdamW** | yes (m,v,param all element-local, `adamw_step`) | **✔ FUSE** | bc1/bc2 (host scalars) | — |
| **Lion** | yes (`lion_step`, m only) | **✔ FUSE** | — | — |
| **Grokfast** | yes (EMA fused in apply, `opt_stages_precompute.cuh:52`) | **✔ FUSE** | — | — |
| **GrokAdamW** | yes (EMA fused, `:53`) | **✔ FUSE** | — | — |
| **NeuralGrok** | yes (psi MLP inline, `opt_components.cuh:222-229`) | **✔ FUSE** | psi-net weights (per-tensor, in `extra`) | — |
| **Prodigy** | apply is element-local BUT needs **global d** first | **✗ stage** | `d_factor` = cross-ALL-tensors r/s reduce (`opt_stages_precompute.cuh:58`) | precompute-d phase (B) **before** P2; order: P1 dW → B → d-reduce → B → apply |
| **Muon** | NO — needs Newton–Schulz on the **full matrix** | **✗ NOT fusable** | NS orth direction (5 matmul iters, `:59`) | keep staged: P1 dW → B → NS phase → B → apply (`muon_update_step`) |
| **SuperGrok11** | apply element-local; needs **per-tensor gate** | **✗ stage (light)** | one per-tensor cosine reduction (`:60`, single-drain, no grid barrier) | per-tensor mu+gate stage before apply (CTA-local, barrier-free) |
| **SuperGrok15** | apply element-local; gate is a **host scalar** | **partial** — mu is element-local, gate host-set | mu meta-MLP forward (`:62`); gate via `FusedScalars.gate` | mu stage before apply (gate already a scalar → no reduce) |
| **SuperGrok2** | apply is Adam on `smart_grad` | **✗ separate tail** | the whole CSA/HCA/PEER/GRU meta-net (§7) produces `smart_grad` | the SG2 in-kernel meta stages (own phase), then Adam apply |

**Summary:** **5 of 11 fully epilogue-fusable** (adamw/lion/grokfast/grokadamw/
neuralgrok → 2 phases / 1 barrier). Prodigy/SG11/SG15 need **one reduction phase
before apply** (3 phases / 2 barriers). **Muon and SG2 are NOT fusable** — they keep
their multi-phase staged structure (NS orthogonalization / meta-net), which is
correct and faithful to the per-op pipelines (`opt_components.cuh:19-27`). The
fusion is a per-optimizer codegen branch in the cell template — it lands in
`fused_megakernel.cuh`'s phase selection, inherited by all model×opt combos.

## Appendix B — What is explicitly NOT done (road not taken)

- **Host-CUTLASS GEMMs (`mma.cuh`) for the L3 cells: REJECTED.** CUTLASS collectives
  are not device-callable (own their TMA mainloop, `mma.cuh:145-153`), and ~15–30
  host launches/step **without CUDA graphs** is pure launch overhead that swamps the
  45 µs compute floor. The owner forbids graphs and mandates one persistent kernel.
  ∴ in-kernel wgmma is the only consistent path. (`mma.cuh` stays for the L1 per-op
  Muon/SG2 paths it already serves.)
- **Global per-layer grid barrier (the briefing's literal Q1 phrasing): REJECTED** —
  it serializes the grid L×#GEMMs/step. Fork B keeps fwd→bwd CTA-local.
- **DSMEM cluster reduce for dW: NOT USED** — Fork B has no dW reduce.
- **TF32 path: NOT the race precision** — bf16 is (owner decision); TF32 stays only
  as the `mma.cuh` fp32-accuracy fallback for L1.
- **FP8: out of scope** — not in the owner's precision decision; the roofline tracks
  it only as a drawn ceiling (`roofline.py:51`).
