# RING — decoder cp.async double-buffered ring, fwd/dX (Lane D, 2026-06-12)

Patch: `0005-decoder-cpasync-ring-fwddx.patch` — applies cleanly on
`642e360` + `0001-decoder-bf16-weight-prestage.patch` +
`0002-decoder-sam-scoped-outline.patch` (verified: fresh worktree, apply all
three, `diff -r` vs the working tree = identical). CPU-only campaign: no GPU
process was launched; every number below is compile-time (production `_ops`
flag set via Lane B's `compile_one.sh`, sm_90a). Raw logs: `logs/ring_*.log`;
objects for the GPU lane: `/dev/shm/tmp/objs/ring_{prod,bench1024,selftest,
jitcell,scalarcell}.o` (SASS dumps: `/dev/shm/ring_{prod,bench1024}.sass`).

## 1. WHAT THE PATCH DOES

Files: `csrc/fused/sm_90/model_stage_decoder_tc.cuh`,
`csrc/fused/sm_90/fused_decoder_megakernel.cuh` (headers only; no generated
cell touched; vit/mamba engines untouched).

1. **cp.async ring in the GEMM engine** (`tc_gemm_block_unpipelined`): a
   compile-time-selected branch (`if constexpr (kRingAsync)`) replaces the
   synchronous per-element staging TRANSPORT with 16-byte
   `cp.async.cg.shared.global` copies (the silicon-validated
   `primitives.cuh::cp_async_cg_16` + commit/wait + `fence_async_proxy`
   handshake — the tile_pipeline contract). The slot ring (`kDecTcStages=2`),
   the M-atom-interleave group structure (627d134/a12c376: kIL A-tiles + ONE
   shared B-tile per k-step), the ascending-k wgmma issue order, and the
   `__syncthreads` choreography are preserved VERBATIM — per k-step: wg0
   issues the group's wgmmas on slot k%S, ALL 256 threads fire tile k+1's
   copies into the other slot (2 LDGSTS/thread instead of ~16 dependent
   2-byte LDG→reg→STS pairs), then wgmma-wait + per-thread
   `cp_async_wait_group<0>` + ONE `fence.proxy.async` + the existing barrier.
   No mbarriers added (all 256 threads both produce and consume — the
   collective barrier IS the correct signal at prefetch distance 1; the
   warp-specialized mbarrier producer/consumer split is the documented next
   step, not this patch).
2. **Ring selection by source type**: the fwd/dX wrappers now pass POD
   gmem-row sources (`DecGmemTileSrcA/B`: base, ld, rows_valid) instead of
   lambdas — identical accessor semantics (same pad guard) — and a trait
   gates the ring. The dW path keeps its lambda sources → compiles the
   UNCHANGED synchronous engine (transposed-strided acts reads = ring
   blocker (b), needs TMA-with-transpose; deliberately out of scope). S=1
   (`SG_TUNED_DEC_GEMM_STAGES=1`, tuner dim) also compiles the ring OUT
   (serial path bit-for-bit, unchanged contract). Ragged N-tiles (never
   occur at either layout: TILE_N=128 divides every fwd/dX Nout/Kin) take a
   predicated zero-fill `STS.128 [..], RZ` — pad rows are never READ from
   gmem (same guard semantics as the old lambda).
3. **C1-T: transposed section of the bf16 weight cache**. The dX B-operand
   reads W TRANSPOSED — from the row-major C1 cache that staging is a
   k-stride 2-byte gather no 4/8/16B cp.async can express (the same
   structural shape as blocker (b)). The prestage cache now carries a second
   section with the 8 matrices stored as W^T [Kin,Nout]; `dectc_wbf_convert`
   fills both sections (element-owned, one writer per index, deterministic;
   the SAM second pass re-converts both via the SAME call — no new barrier).
   WT[r,c] == W-cache[c,r] BIT-IDENTICALLY (same fp32 element through the
   same `__float2bfloat16`), so dX staging streams K-contiguous rows with
   UNCHANGED operand values. Cache cost: 786,432 B → 1,572,864 B at d=128;
   50,331,648 B → 100,663,296 B at d=1024 (workspace deltas, sized via the
   single-source `dec_wbf_floats()`; both fit with large margin — the bench
   workspace is ~6 GB dominated by acts).
4. **Alignment plumbing**: `alignas(16)` on `DecTcSmem::sA/sB` (no member
   offset changes — all block sizes are 16B multiples; pins the LDGSTS smem
   requirement); 16B-align bump for the cache base in both layout branches
   (+3 floats slack added to `dec_tc_workspace_floats`, host+kernel single
   source). All gmem chunk addresses are 16B-aligned by construction (every
   operand base/ld/g0/kbase is a multiple of 8 bf16 at BOTH layouts; the
   d=1024 branch compiles the same derivations — no problem-specific
   hardcoding).

NUMERICS (the fp64-oracle/A-A-A design argument): the ring changes WHEN and
HOW operand bytes move, never WHAT they are — identical bf16 values land at
identical smem offsets in the same ascending-k order, and the wgmma sequence
is untouched (SASS-verified below) → bit-identical accumulation by
construction. No path disabled, no math reordered, all 11 tails intact.

## 2. REG/SPILL REGRESSION GATE (parse2.py TOTALs, entry+callees)

Baseline = `dec_C1S*` (642e360+0001+0002, Lane B). Ring = `ring_*` (this patch).

production d=128 (`ring_prod.log`):
| cell | base TOT sp_st | ring TOT sp_st | delta | regs |
|---|---|---|---|---|
| AdamW/Lion/Grokfast/GrokAdamW/NeuralGrok/Prodigy/Muon | 0 | **0** | none | 255 |
| LookSAM  | 8004 | **7884** | -120 | 255 |
| SuperGrok11 | 7932 | **7884** | -48 | 255 |
| SuperGrok15 | 8004 | **7884** | -120 | 255 |
| SuperGrok2  | 8588 | **8608** | +20 (0.2%, noise) | 255 |

bench d=1024 (`ring_bench1024.log`):
| cell | base | ring | delta |
|---|---|---|---|
| 7 single-pass | 0 | **0** | none |
| LookSAM/SG11/SG15 | 8140 | **8160** | +20 (noise) |
| SuperGrok2 | 8416 | **8380** | -36 |

**GATE: every single-pass cell stays 0-spill at BOTH layouts; SAM cells move
within ±120 B (≤1.5%) on a 7.9–8.6 KB base.** Spill loads move the same way
(e.g. prod LookSAM 8408→8128). Stack frames: +64 B uniformly (736→800
single-pass, 1888→1936 / 2208→2256 SAM — the by-value source structs; zero
spill consequence). The ring's in-loop register state priced at ~8–16 regs
was absorbed exactly as Lane B's margin analysis predicted (it RETIRES the
synchronous staging window's registers; net allocation unchanged at 255).

## 3. SASS AUDIT (cuobjdump -sass; scanner = Lane B windows + opcode census)

Counts per kernel, baseline → ring (prod; bench identical in structure):
| metric | AdamW base | AdamW ring | LookSAM base | LookSAM ring |
|---|---|---|---|---|
| LDGSTS.E.BYPASS.128 | 0 | **32** | 0 | **32** |
| LDGDEPBAR (cp commit) | 0 | **16** | 0 | **16** |
| DEPBAR.LE SB0 (cp wait) | 0 | **16** | 0 | **16** |
| FENCE.VIEW.ASYNC.S | 0 | **16** | 0 | **16** |
| STS.U16 (2-byte staging stores) | 52 | **4** | 56 | **8** |
| HGMMA.64x128x16 / 64x8x16 | 36/9 | **36/9** | 40/10 | **40/10** |
| WARPGROUP.ARRIVE / DEPBAR | 45/45 | **45/45** | 50/50 | **50/50** |
| BAR.SYNC | 141 | **141** | 167 | **167** |
| STL/LDL inside HGMMA..WARPGROUP.DEPBAR windows | 0 | **0** | 0 | **0** |

Reconciliation: 16 commit sites = 8 fwd/dX GEMM call sites × (prologue +
steady-state stager); 32 LDGSTS = 2 per stager body (A-chunk + B-chunk, the
rolled 2-iteration loop). The 4 residual STS.U16 are the dW path's two
rolled `stage_kmajor_tile` loops (out of scope by design); fwd/dX 2-byte
staging is GONE (52→4 static; SAM cells 56→8, their outlined shims carry the
ring too). **The wgmma web is instruction-count-identical** (HGMMA, ARRIVE,
DEPBAR, BAR.SYNC all unchanged) — the engine's H1 choreography was not
perturbed.

Placement (AdamW prod, steady-state k-loop block, addresses from the dump):
```
/*5910*/ ..wg0 branch..
/*5970*/ WARPGROUP.ARRIVE
/*5980*/ HGMMA.64x128x16.F32.BF16 R88, gdesc[UR20], R88, gsb0    ; atom 0 (acc)
/*5990*/ WARPGROUP.DEPBAR.LE gsb0, 0x0
/*5a10*/ HGMMA.64x128x16.F32.BF16 R88, gdesc[UR20], RZ, !UPT, .. ; atom 0 (k=0)
/*5af0*/ HGMMA.64x128x16.F32.BF16 R24, gdesc[UR20], R24, gsb0    ; atom 1 — kIL=2 interleave
/*5bc0*/ HGMMA.64x8x16.F16 RZ, gdesc[URZ], ..                    ; commit-group dummy
/*5c10*/ @P2 BRA ..        ; k+1 < k_steps guard
/*5e70*/ LDGSTS.E.BYPASS.128 [R197+0x2000], desc[UR14][R12.64]   ; A-tile 16B chunk
/*6000*/ LDGSTS.E.BYPASS.128 [R197],        desc[UR14][R12.64]   ; B-tile 16B chunk
/*6040*/ @!P2 BRA 0x5ce0   ; chunk loop (2 iters/thread)
/*6060*/ LDGDEPBAR                                               ; cp.async.commit_group
/*60a0*/ DEPBAR.LE SB0, 0x0                                      ; cp.async.wait_group 0
/*6100*/ FENCE.VIEW.ASYNC.S                                      ; fence.proxy.async
/*6120*/ BAR.SYNC.DEFER_BLOCKING 0x0                             ; __syncthreads
/*6140*/ @!P2 BRA 0x5830   ; next k
```
The staging block sits between the wgmma issue block and the iteration's
closing barrier — the exact window the synchronous STS.U16 staging occupied
in the baseline (verified: baseline block layout at 0x4e60–0x57b0 is the
same shape with element loops in place of the LDGSTS cluster), and the
per-HGMMA `WARPGROUP.ARRIVE/DEPBAR.LE gsb0` adjacency is the PRE-EXISTING
H1 lowering, byte-pattern-identical in both builds. Note the scanner's
"in-window LDGSTS" linear metric reads 0 because the staging loop is its own
basic block (branch target), not linearly between HGMMA and DEPBAR —
placement is therefore evidenced by the CFG excerpt above, not the linear
scan. STL/LDL-in-window stays 0 everywhere (Lane B's determinism-hazard
audit, clean).

## 4. SMEM ACCOUNTING

Static smem per cell, ptxas: **byte-identical to baseline at BOTH layouts**
(prod AND bench): 7 cells @ 43188 B, Muon 43316, Prodigy 43444, SG15 43700,
SG11 43828. The ring reuses the existing `kDecTcStages=2` slots (DecTcSmem
sA 8192 B + sB 8192 B unchanged; `alignas(16)` adds no padding) and adds NO
mbarriers → +0 smem. Headroom vs the 227 KB carveout: ≥183 KB untouched.

## 5. EXPECTED-WIN MECHANISM (and what makes it a wash) — honest, falsifiable

Removed stall: the synchronous staging window's REGISTER-MEDIATED 2-byte
element loop. Per k-step a CTA stages 8 KB (2 A-tiles + 1 B-tile); the old
transport was ~16 dependent LDG(2B)→reg→STS(2B) pairs per thread whose
long-scoreboard latency the allocator could only overlap ~8–16 loads deep
(Lane B's measured window); the ring issues 2 fire-and-forget
LDGSTS.E.BYPASS.128 per thread — the WHOLE tile is in flight concurrently,
zero register data path, ~94% fewer staging instructions (52→4 static
2-byte ops), and the copy flies while the wgmma group on the previous tile
executes. Secondary win: dX's weight staging was an uncoalesced k-strided
2-byte gather of W; from C1-T it is dense 16B-sector reads.

Wash risks (each falsifies on the GPU gate):
1. **d=128 production**: GEMM staging is a minority of step time (scalar
   LN/attention/CE + dW dominate); a 2× faster staging window may move
   step time <1%. ROOFLINE arbitrates keep-if-better.
2. **L2-resident weights**: the 0.8–1.5 MB cache sits in L2 after first
   touch, so the latency the old path exposed was L2-hit latency partially
   hidden by the wgmma window already — the ring's gain shrinks toward
   issue-count savings only.
3. **Prefetch distance stays 1 k-step**: `DEPBAR.LE SB0,0x0` still exposes
   (miss latency − wgmma window) on true-HBM streams (the acts A-operands at
   bench T=65536). Fixing that needs S>2 with `wait_group<S-2>` multi-group
   tracking or the warp-specialized mbarrier ring — the documented next
   step; this patch deliberately keeps the validated barrier structure.
4. **.cg bypasses L1**: A-tiles are re-staged once per N-tile (up to 4× at
   Nout=dff) and the old .ca LDGs may have earned L1 hits on those re-reads;
   BYPASS shifts them to L2. If this dominates, per-operand transport choice
   (A via .ca?) is a one-line follow-up tuned by measurement.
5. dW (P2) is untouched by design — overall step-time win is bounded by the
   fwd/dX (B0/P1 + SAM second pass) share.

## 6. COMPILE-CLEAN PROOF + GATES OWED

All five TUs rc=0 with production flags: `ring_prod` (TC launcher, ALL 11
OptId instantiations), `ring_bench1024` (d=1024 branch, all 11),
`ring_selftest` (decoder_tc_selftest.cu — exercises the lambda/synchronous
engine path, untouched by the trait), `ring_jitcell`
(mega_decoder_real_adamw_tc.cu), `ring_scalarcell` (mega_decoder_real_adamw
.cu — scalar path, no ring code included; warning sets byte-identical to
baseline modulo worktree path). No new warnings anywhere; the nvcc 12.4
cicc tile-fn-noinline hazard shape was avoided (the ring lives INSIDE the
0002-outlined tile bodies; composition compiles clean at both layouts).

Standing rules: no functionality suppressed (all 11 tails + SAM second pass
intact; dW path fully functional on its existing transport); no new
perf-shaped constants (ring depth = existing SG_TUNED_DEC_GEMM_STAGES;
ring auto-disabled at S=1); sizes derive from layout constants (bench branch
compiles the scaled derivations). GPU gates owed before ship (other lane):
per-cell fp64-oracle parity, A/A/A bit-determinism, tail_gate, wiring 33/33,
and the ROOFLINE keep-if-better measurement for the ring itself (mechanism
+ wash risks above are the falsifiable claims to test). Last-partial-tile
behavior (T % kTileM ≠ 0) is unchanged: the ring reads the same
in-workspace-bounds rows the scalar staging read; pad-row outputs remain
unconsumed.
