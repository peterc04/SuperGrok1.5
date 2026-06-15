# A5 Audit — CORRECTED findings (durable)

## CRITICAL CORRECTIONS made during audit (initial reads were wrong):
1. test_megakernel_vs_eager.py is 687 lines (first Read truncated to 395, stale pyc). It HAS test_decoder_oracle_matches_autograd (L340), test_decoder_kernel_mirror_matches_oracle (L368), test_decoder_layout (L393) — all CPU, un-hw-marked, RUN IN CI, PASS. Oracle/mirror are NOT orphaned. RETRACTED.
2. CI inline jobs functional_metanets (L849) + functional_smoke (L561) DO run the Python meta-net forwards (sg11/sg15 SharpnessMetaNet, neuralgrok _Amplifier, sg2 CSAHCAMetaNet) on CPU — checkpoint-parity (gc on==off <1e-4). NOT zero coverage; but transparency-only, not correctness-vs-oracle.
3. Fresh CPU collection = 37 tests (not 34).

## Confirmed findings (solid):
- F1 [HIGH]: parity_gate_h100.py NOT in any CI workflow; not pytest-collectable (module-level assert cuda L43 + section* naming). SG2 §4/SG11 §3b teeth non-vacuous by inspection (gate_ok/lamb_ok guards) but NEVER executed in automation.
- F2 [HIGH]: test_kernel_matches_reference_gpu real fp64 allclose ONLY adamw (L702); other 10 = isfinite+moved (L713). And it's hw-gated => 0 in CI.
- F3 [HIGH]: profile_maximal Tier D descent probes reimplement Adam/Lion inline in jax (L514-518, L533-535) — vacuous for kernel/header. trace_check (L552) is real-but-lowering-only.
- F4 [MED]: lamb_eff composition in bindings.cpp (L1374-81 SG11, L1502 SG15) OUTSIDE algorithms/*.h => manifest blind spot.
- F5 [MED]: check_math _REINLINE_PATTERNS Adam-moment only; opt_components.cuh structural greps only "adamw.h" + `if fused and` guard => non-Adam fused (GRU/PEER/slow) structurally UNGUARDED + missing file silent-skips.
- F6 [MED]: _LazyOps catches ImportError only (L488); no ABI/schema check => stale .so imports OK, has_kernels()=True, TypeError at call.
- F7 [MED-latent]: ref_sg_phi_forward (L200) uses TANH; kernel uses GELU (supergrok11.h L50). Two disagreeing meta-net refs; only self-tested; trap if wired as oracle.
- F8 [LOW]: SG2 §4b expert_out=mu_buf/(1-gru_decay) recovered from kernel output (L460) => mu_new circular; teeth=dm/dv/dslow only. PEER/CSA/HCA forward smoke-only (§4a isfinite L404).
- F9 [LOW]: SG15 full param-update never numerically gated (§3=mu only, §3b=SG11, §4=SG2, CPU neutral-gate=off). gate_global*alpha*mu mixing unchecked on kernel.
- F10 [LOW]: grokadamw/grokfast/neuralgrok/looksam real-kernel nontrivial-regime numeric = NONE; CPU tests are degenerate-reduction-to-AdamW; hw = smoke.
- F11 [LOW]: decoder_oracle.py L15 stale cross-ref "item 5a" (tests now A/A2/B).
- F12 [LOW]: LookSAM sam_step + SG2 bilevel adjoint = zero coverage.

## Existing good machinery (report as-is):
- Codegen-gate EXISTS + runs in CI: verify_all 4d (in-memory re-emit all 99, rstrip-compare) + 4e tier-comment + 5a 3 dispatch tables. CPU phase 1/4/5.
- drift_guard runs check_math in CI. Manifest currently in sync.
- MODELS/OPTIMIZERS tuples => codegen deterministic.
- build.sh --autotune resolved (exit 2 + message).

## Specs to write: build-stamp (.so schema version asserted in _LazyOps._resolve); manifest-ext (hash bindings.cpp lamb_eff region); codegen-gate residual (write_all + git-diff worktree-clean, since 4d is in-memory only).
