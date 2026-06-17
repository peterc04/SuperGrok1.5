# Phase-1 status audit (workflow wcrdmvuvq, 2026-06-17) — refute-by-default

8 agents (5 investigate / 3 adversarial-verify), read-only. Numbers reproduced by re-running
each bench's FLOP formula; narrative re-anchored to the freshest committed artifacts.

## Roofline (derived-from-measured-time; NOT formally scored — #24 pending)
Peaks: bf16 TC ~989 TF/s dense; HBM3 ~3.35 TB/s; ridge ~295 FLOP/byte.

| model | config | ms/step (full fused step) | achieved TF/s | % of 989 peak | AI (FLOP/B) | bound |
|---|---|---|---|---|---|---|
| decoder | d2048 B16384 | 618.5 (fresh 616.2) | 64.0 | **6.48%** | 12239 | latency/serialization |
| vit | d2048 B1024 | 5759 baseline → **~1434 (S=8, 4.02x)** | 1.83→~7.3 | 0.185→~0.74% | 3252 | B1 load-imbalance (fixed) |
| mamba3 | d128 B4096 (cannot scale; smem cap ~d142) | 221.6 | 0.30 | 0.0303% | 3492 | scan-dominated (GEMM-only numerator near-meaningless) |

Key honesty notes (verifier-confirmed):
- The "618 ms / 6.466%" gate figure = full fwd+bwd+AdamW-tail wall (one persistent kernel), and 6.466% = 64 TF/s / 989. PER-STEP, not a sum.
- The FLOP numerator counts ONLY dense GEMMs; the step is mostly non-GEMM fp32 CUDA-core work (attn/LN/GELU/embed/CE) → roofline % UNDERSTATES true utilization at these tiny-d/short-seq grokking configs.
- NONE is compute- or bandwidth-bound at the whole-step level (AI 3k–12k >> 295 ridge; HBM 0.003–0.16% of 3.35 TB/s). Real bound = LATENCY/serialization: grid-barriers ~20%, phase serialization, staging-bound dW (16.5%), ViT load-imbalance.
- VERIFIER REFUTE: decoder fwd/dX is NO LONGER drain-bound — the deeper-ring KEEP fixed it; fresh fwd-fine re-profile = WGMMA-compute-bound (WAIT 9%/6% vs WGMMA 47%/41%) → S>4 buffering measured-unlikely to help.

## Optimizer readiness (11) — FUSION maxed, KERNELS not perf-maxed
Architecture: ALL 33 cells run fwd+bwd+optimizer in ONE persistent `__global__` kernel, ONE launch,
ZERO intermediate launches (the single-binary single-launch design — that IS realized/maxed).
But the optimizer TAIL (P3) is **5.9% of the decoder critical path** (~38ms/618ms), not <1% — real headroom.

| opt | correctness | perf_maxed | ready | note |
|---|---|---|---|---|
| adamw | green | yes(thin) | Y | toy-scale (d128) gate only; RG4/RG6 |
| lion | green | yes(thin) | Y | dead v+extra state (VRAM) |
| grokfast | green | yes(thin) | Y | toy-scale gate |
| grokadamw | caveat | yes | N | multi-step gate MISSING for vit/mamba |
| prodigy | caveat | no | N | d-adaptation not CI-guarded |
| muon | green | no | N | BUG-04 (mamba OOM d>=1024) |
| neuralgrok | green | no | N | BUG-04 |
| looksam | caveat | no | N | BUG-04 + earned-tol |
| supergrok11 | caveat | no | N | warm-up gate CLI-only (not CI-collected) |
| supergrok15 | caveat | no | N | NO warm-up gate at all |
| supergrok2 | caveat | no | N | CSA oracle co-wrong (HIGH) + non-voting probe |

VERIFIER net: "perf_maxed:yes is THIN — rests on a stale '<1% of step' framing that is actually 5.9%; the
autotune that would give real optimizer perf data FAILED (zero winners). Caveats are mostly MISSING/toy-scale
GATES, not active math drift (adamw/lion/grokfast pass the committed 33/33x3-seed pytest gate at d=128)."

## GPU continuous-utilization
- Persistent megakernel CONFIRMED: 1 CTA/SM, hand-built sense-reversing GridBarrier (no cooperative launch,
  no CUDA graph), ENTIRE step in one launch. Training data fully VRAM-resident.
- Only host bubble: ONE mandatory `.item()` D2H sync/step (dispatch.py:2005) — negligible at 618ms/step.
- Verdict: "mostly saturated". The ~20% grid-barrier + phase serialization is the real utilization gap, NOT host bubbles or H2D.

## Front-load incident (this window)
The 12h autotune front-load FAILED in 8 min: (1) `-M transformer_decoder` rejected (CLI wants `decoder`);
(2) `g++-cached` not on PATH → torch ABI probe killed every --jit-only build. BOTH fixed at source
(run_12h_frontload.sh model token; .fast_build_env.sh symlinks the wrapper onto PATH). Fix smoke-test verified.
