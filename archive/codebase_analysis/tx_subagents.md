# Subagent Transcript Index: Delegated-Work Landscape
## Source: 338 subagent transcripts (subagents_index.md, 4108 lines)
## Coverage: Complete — all agents catalogued

---

## 1. OVERVIEW

338 unique agent transcripts were indexed. Work spans several distinct campaigns:

- **Phase-0 / Phase-0b / Phase-0c**: Exhaustive read-only reconnaissance (files, architecture, code structure)
- **h100-audit-maximal campaign** (branch `claude/h100-audit-maximal`): The main build campaign — implementing 33 L3-TC cells (11 optimizers × 3 models), fixing bugs, validating on real H100
- **Current campaign** (`claude/custom-optimizer-analysis-HFYhg`): TP templating, 3 flagship models, ongoing work
- **Kernel audit sweep**: ~14 parallel per-kernel auditors hunting latent correctness bugs
- **Autotuner review**: 4-agent line-by-line review of compile.py (28k lines)
- **Infrastructure agents**: probe tasks, status line setup, cleanup inventory

Failure rate: approximately 30–35 agents hit API rate-limit errors ("Server is temporarily limiting requests") or socket close errors. Coverage of those subsystems is therefore incomplete.

---

## 2. AGENT CATEGORIES AND DELEGATED WORK

### 2A. READ-ONLY RECONNAISSANCE (Phase-0, Phase-0b, Phase-0c)

Multiple waves of read-only agents were dispatched to map the codebase before implementation work:

**Phase-0 agents** covered:
- Top-level file inventory
- csrc/fused/sm_90/ architecture mapping
- grokking_optimizers/ package structure
- algorithms/*.h canonical math headers
- Optimizer kernel headers (sm_90/*.cuh)
- Test infrastructure

**Phase-0b agents** covered:
- Planning documents and integration specs
- Commit history (git log)
- HARDWARE_VALIDATION.md contents
- Worktree mapping for parallel agents

**Phase-0c agents** (exhaustive): ~50+ agents assigned specific file ranges or subsystems. Many hit rate limits. Successful ones covered:
- `csrc/fused/sm_90/fused_decoder_megakernel.cuh` (1575 lines decoder L3-TC)
- `csrc/fused/sm_90/model_stage_decoder_tc.cuh` (2595 lines decoder TC GEMM stages)
- `grokking_optimizers/compile.py` in 4 parallel ranges (1–7000, 7001–14000, 14001–21000, 21001–28041)
- `csrc/bindings/bindings.cpp` binding layer
- `grokking_optimizers/dispatch.py`
- `grokking_race_v2.py`
- Various test files

### 2B. IMPLEMENTATION: L3-TC MEGAKERNEL CELLS

**agent-a863dcbebb3b558d9** — PHASE 1 decoder megakernel:
- Replaced surrogate model with real transformer-decoder fwd+bwd inside persistent megakernel
- Created: `csrc/fused/sm_90/{model_stages_decoder.cuh, fused_decoder_megakernel.cuh, decoder_layout.cuh, mega_decoder_real_adamw.cu}`, oracle + mirror test files
- Result: Gates GREEN — single-step parity < 1e-5, 200-step trajectory, grok smoke

**agent-aa580ce8de45b65ad** — PHASE 2 ViT megakernel (NO builds/GPU):
- Created: `model_stage_vit.cuh`, `fused_vit_megakernel.cuh`, `vit_layout.cuh`, `INTEGRATION-VIT.md`, oracle/mirror test files
- CPU kernel-mirror validated to ~1e-12 vs oracle
- Integration spec written for dispatch.cpp/dispatch.py wiring

**agent-a9ada8ec980f492ab** — R2 Mamba TC port:
- Created: `model_stage_mamba_tc.cuh`, wgmma branch in `fused_mamba_megakernel.cuh`, `mega_mamba_real_adamw_tc.cu`, `test_mamba_tc.py`
- Gates GREEN: grad parity < 0.08, dW ISO 2.96e-7, A/A/A bit-identical, 50-step trajectory
- KEY FINDING: TC step time = 17.8ms vs scalar 8.2ms (0.46× slower) — scan-dominated, expected

**agent-ac296056c3edacb42** — R2 Retrofit decoder TC wgmma (hit "Overloaded" error):
- Task was to convert decoder scalar GEMMs to bf16 wgmma as a tuned variant
- RESULT: API Error: Overloaded — work abandoned

**agent-adbe072023104e73b** — R2 ViT TC port:
- Created: `model_stage_vit_tc.cuh`, wgmma branch in `fused_vit_megakernel.cuh`, `mega_vit_real_adamw_tc.cu`, `test_vit_tc.py`
- All gates GREEN: loss rel 8e-6, dW ISO 1.44e-7, A/A/A, 50-step trajectory
- Found: committed decoder TC forward omits biases (latent bug with loose 0.15 tolerance masking)

**agent-ad0c5b2f4586bf610** — Integrator for ViT×adamw and Mamba×adamw:
- ViT: FULLY VERIFIED (empirically) — single-step + 4-gate trajectory all pass
- Mamba: SEAM COMPLETE but BLOCKED by forward-LN stride bug in frozen `model_stage_mamba3.cuh` (lines 699/704)

**agent-a87939bd911d7493b** — 33-cell L3-TC baseline enforcement:
- Objective: all 33 cells on L3-TC wgmma path, eager removed from production
- Result: Only 4 cells converted (adamw/lion for decoder+vit). 29 of 33 cells remain blocked
- Bug found+fixed: latent fp32 surrogate fallthrough for non-adamw cells
- Honest report: `--require-all` fails loud listing 29 blockers

**agent-ae1c8ed966c0b6202** — R2.4 TC-only production + max batch:
- Wired TC cells to bf16 race path; measured batch saturation
- Found: mamba scalar megakernel beats eager 1.8× in throughput; memory win 3.6× (2.50 GB vs 8.9–9.5 GB)
- TC decoder at this scale: 1.25 TF/s vs eager 8.4 TF/s (latency-bound, expected)
- Produced: `results/h100_grokking_race/roofline.json` (33 rows B=16384)

### 2C. OPTIMIZER-SPECIFIC FIXES AND COMPLETIONS

**Prodigy:**

*agent-aa26ca8d4c934d57e* — Root-cause why d stays at 1e-6:
- Found: degree-(-1) `1/d_prev` catapult — the canonical header `prodigy.h` uses wrong formula vs paper
- The kernel is faithful to the header but the header has a `1/d_prev = 1e6` catapult
- Proposed fix: revert to degree-2 `d_prev²` + `|g|` L1-norm partials

*agent-a7793e4a7ccb78d3c* — Full Prodigy fix implementation:
- Fixed buffer reseeding + decay changes (emulator approach)
- High confidence the fix allows memorization
- Residual: grad-hooks path over-decays; single-tensor path still instantaneous (not race path)

*agent-ae55432a93eb83a38* — Prodigy A/A/A determinism fix:
- Root cause: work-steal `prodigy_precompute_reduce_phaseA/B` caused non-deterministic d reduction
- Fixed: rewrote to fixed-partition P2.6
- Gates: A/A/A bit-identical ×3, params correct
- mamba×prodigy remains blocked on separate mamba scan/forward symptom

**SG11 (SuperGrok11):**

*agent-a774ebfaee289f45d* — SG11 flat at random diagnosis:
- Found: off-by-one in step counter → optimizer ran as step=1 always (no warmup ramp)
- Secondary bug: double-applied mu + inverted gate polarity
- Host-side fix (step counter) confirmed restores memorization
- Key files: `supergrok11.py` (step 219–274, increment site 257–265), `bindings.cpp` (:1306)

**SG15 (SuperGrok15):**

*agent-a8f86ff7649ad018f* (kernel audit) — SG15_H=64 vs meta-net hidden_dim=32:
- Kernel hardcodes `SG15_H = 64` but race passes `meta_hidden_dim=32`
- Causes OOB reads from uninitialized memory + wrong mu
- This is the HIGH-severity bug P3 from the kernel audit synthesis

**SG2 (SuperGrok2):**

*agent-ac35e460feb2ee6bf* — SG2 GRU reconstruction fix:
- Prior fix made PEER routing run (fixed reshape crash); now SG2 thrashes (||ΔP||=584)
- Root cause: fused forward uses ZERO placeholder for GRU output
- Reconstructed matrix GRU; forward matches eager to 1e-6, state 1e-7
- Pre-existing apply-tail difference flagged: fused uses expert-EMA vs eager's grad-EMA

*agent-a85cab5cdd4fd683c* — SG2 meta optimizer tail integration:
- Implemented `sg2_meta_optimizer_tail`, `sg2_ws_stride` bindings, `sg2_meta_tail.cu`
- Parity test: 2 passed, 14 total (hw-gated), gate-invisibility bug found and fixed
- Fixed: `@hw` was using wrong pytest marker

*agent-ac12c67849e8e592f* — SG2 PEER reshape crash fix:
- Found: `peer_query_Ws` collapsed all 4 heads into wrong reshape → size 192 vs expected 44
- Fix: loop over peer heads in kernel, per-head parameter extraction
- Option A recommended: `for h in num_peer_heads` loop + `half = d_model//2` (NOT Q/2)

**NeuralGrok:**

*agent-af69827652d920337* — NeuralGrok amplifier autograd-reachability fix:
- Confirmed: amplifier parameters never receive gradients (frozen at random init forever)
- Fix: add amplifier parameters to optimizer param_groups + wire amplifier objective through autograd
- Also found: `neural_hidden=16` alignment needed (kernel pins to 16, default 128 was wrong)
- 42 CPU tests pass (+3 new)

**GrokAdamW:**

*agent-a98c6dfb71afdec0f* — GrokAdamW algorithm completion:
- Published algorithm has (1) layer-wise β1 decay via gamma and (2) grokking-signal-driven adaptive alpha
- Both were missing; gamma was given a DeprecationWarning (wrong — should be wired)
- Fix: wired both mechanisms in `grokadamw.py`
- CPU tests: 31 passed (includes 3 new)

**Muon:**

Muon inverted weight-decay bug: confirmed and fixed in commit a9276b5 (from prior campaign logs). The kernel computed `p_new = -decay*p + lr*update` (anti-regularization) instead of `p_new = decay*p - lr*update`.

### 2D. KERNEL AUDIT SWEEP (Per-Kernel Latent Bug Hunting)

14+ parallel auditors, each assigned one kernel header:

- **adamw_sm90.cuh** (agent-a45cc1db55fd7ac4a): No latent bugs. Correct dtype dispatch, canonical formula match.
- **lion_sm90.cuh** (agent-a816d29178a13a1e8): No bugs. Dtype validation enforced at C++ binding layer.
- **grokfast_sm90.cuh** (agent-a2512164b05f21285): One latent bug — `launch_fused_grokfast_ema` calls `grad.data_ptr<float>()` unconditionally; latent (path currently unused), would corrupt bf16 grads.
- **looksam_sm90.cuh** (agent-a6eb0bc85afb24b9a): No bugs on reachable paths. Several dead kernels (not bound to Python).
- **muon_sm90.cuh** (agent-acb0479f55a6e834e): HIGH severity — inverted weight-decay formula in Newton-Schulz combine+update (already fixed in campaign).
- **neuralgrok_sm90.cuh** (agent-ae303623be16c9008): HIGH severity — `NG_H = 64` hardcoded, Python default is `hidden_dim=128`; kernel ignores passed parameter, reads only first 64 of 128 elements.
- **supergrok11_sm90.cuh** (agent-ad5b3f582177a3923): RESULT shows `StructuredOutput` call tail (incomplete output). Task assigned, findings not fully surfaced.
- **supergrok15_sm90.cuh** (agent-a8f86ff7649ad018f): HIGH severity — `SG15_H = 64` vs race's `meta_hidden_dim=32`; OOB reads. Also: sharpness buffer dtype unvalidated (medium).
- **supergrok2_sm90.cuh**: Covered via separate crash-fix agent (ac12c67849e8e592f) — PEER reshape bug.
- **prodigy_sm90.cuh** (agent-a819bd88dfab4ed9a): RESULT shows `StructuredOutput` tail (incomplete output).
- **grokadamw_sm90.cuh** (agent-ae319f4dac9880e77): HIGH severity — Q3 quantized path: integer floor division produces wrong `block_size` when N not divisible by num_scales → OOB read of `ea_scales`.
- **transformer_decoder_sm90.cuh** (agent-a91b9c4edd7e18dc8): HIGH severity — backward pass reuses `logits_full` buffer (size B*S*V=B*S*99) as `grad_stack_out` then writes B*S*D=B*S*128 → OOB writes. Race config: V=99 < D=128.
- **mamba3_sm90.cuh** (agent-aad121e8eb7d43386): No bugs. All kernel math verified correct.
- **attention_sm90.cuh** (agent-a1c93d8e5df668669): One Class A bug — `softmax_lse` buffer dtype mismatch (kernel expects float32 but binding passes as ActT dtype).
- **vit_sm90.cuh** (agent-a4c687e7ae779a97c): RESULT shows `StructuredOutput` tail (incomplete output).

**Synthesis agent** (agent-ab04d209ad183314a): Consolidated 14 raw audit findings:
- 6 HIGH/certain bugs: P1 Muon inverted WD, P2 NeuralGrok NG_H=64≠128, P3 SG15 SG15_H=64≠32, P4 SG2 PEER reshape crash, P5 GrokAdamW Q3 OOB, P6 dtype-pair cluster (Prodigy 3 sites + SG11 2 sites, bf16 only)
- None on grokking-race critical path (fused registry is empty; race uses eager optimizers)
- Recommended fix order: P1→P5 then P6

### 2E. HILL CLIMB OPTIMIZATIONS

**H1 — M-atom GEMM interleave** (2.1% at d=1024):
- Groups of 2 m64 atoms share ONE staged operand tile, reducing smem re-loads
- Measured on decoder TC path

**H2 — Counting-sort CSR for embed-grad assembly** (−42.8% step time):
- Replaces O(V·d·T) scan with O(d·T) CSR-based scatter
- Decoder-specific optimization

**H3 — dW split-K G sweep** (peak 9.52 TF/s):
- Multiple CTAs cooperate on dW GEMM tiles via split-K reduction
- Sweep measured in quiet GPU windows

### 2F. COMPILE.PY AUTOTUNER: 4-AGENT LINE-BY-LINE REVIEW

Four agents split the 28,041-line file:

**Lines 1–7000** (agent-a92eecf9687a9bac7):
- Found: sequential timing loop despite pool workers (worker = pool bug at L13214)
- Cost model stall reasons count `_COST_MODEL_STALL_REASON_COUNT=14` must stay synced with `STALL_DIM_HINTS` (L27167)
- PGO workload defects (DEFAULT_PGO_WORKLOAD broken)

**Lines 7001–14000** (agent-a7257e8b5f95f3f90):
- Covered: CompileCache tail, BuildSpec JIT machinery, DiscoveredEntry, `_jit_autotune`
- Found: winner-timer GEMM-blind defect, missing tile dims, `_export_kernel_tuned_json` persists only 5 dims (should persist ALL winner dims with macros)
- `emit_variant_source`/`CodegenError` phantom module is an alias (`codegen` → `compile` itself)

**Lines 14001–21000** (agent-ac16dbfbe6f08fc1a):
- Covered: `main()` CLI, orchestration, result recording
- Phantom `compile_config` import (L14831): try-guarded, dead optional branch — recommendation: create real `compile_config.py` as clean home for externalized config
- `_sccache_env` cleaning inventory flag confirmed at ~L9706

**Lines 21001–28041** (agent-ac32a82f161af5daf):
- Covered: self-test harness blocks, AMD/TPU sections, CK integration
- Found: `__nv_bfloat16`-in-HIP dtype map (load-bearing gfx942 bug for bf16/fp8 synth)
- Zero-fragment MMA mainloops (numerically wrong synth kernels, but gating prevents misuse)
- Two stale CK TODOs at L23387/L23749 need reconciliation vs existing `emit_ck_gemm_variants`

**Fix wave agent** (agent-a9b74825ee2407fb9 — compile.py single writer):
- Applied P0 (dim plumbing), P1 (header-blind cache), and ~10 other fix groups
- Self-test: 238/0 (expected count)
- Seams unit-verified but not full e2e (GPU blocked)
- `_tuned_inject.py` +178/? lines; `compile.py` +~1190/−96

**compile.py self-test fixers:**
- *agent-acaa5c05141a69571*: Root-caused both self-test failures (ccache masquerade + device-aware utilization test). Implemented `_sccache_env` rewrite with ccache masquerade symlink dir. `e2e_smoke_gated` reached 230/1 after ccache fix but -dlto/-gencode conflict remained.
- *agent-aaaef555b776297ab*: Applied both verified fixes — self-test 229 passed, 2 failed → confirmed ccache fix correct but not sufficient alone (LTO conflict persisted).

### 2G. PARALLELISM AND TP WIRING

**agent-ae8bc979ff7058af9** — L3 megakernel wiring into grokking race:
- Implemented fused_train_step dispatch path; CPU tests pass (27)
- Could not GPU-test (Optuna fleet owned GPU)
- Operator runbook written in BUILD_AND_VALIDATE.md

**agent-ab5f8a6bf9b164ab7** — TPU v6e pre-silicon lane:
- Drift closure: ported 4 semantic changes (Muon WD fix, Prodigy d-adaptation, SG2 matrix-GRU + lamb_eff, SG11 cosine-gate)
- Tests/tpu 229 passed before and after
- Day-1 checklist written (7 items: tiling, NS on MXU, Prodigy EMA, SG2 dims, gate_signal, L3 XLA fusion, dtype)

**agent-ac9d86376d8d91cbd** — TPU Pallas block/tile kwargs wiring:
- Threaded `pl.pallas_call(compiler_params)` with real `dimension_semantics=(PARALLEL,)` replacing hardcoded TILE
- Verified non-vacuous execution via interpret harness
- tests/tpu 229 passed, before and after

### 2H. SAM-TIER CELL WIRING (h100-audit-maximal final push)

Multiple SAM-tier agents attempted to wire LookSAM/SG11/SG15/SG2 across all 3 models:

**agent-ab266aa13723be112** — LookSAM vit+mamba:
- ViT LookSAM: COMMITTED and GREEN (wiring_check + A/A/A + 1.409 TF/s)
- Mamba LookSAM: BLOCKED — latent mamba scan A/A/A race exposed by LookSAM occupancy profile (non-deterministic even on SAM-OFF step); also fixed latent `FusedScalars` POD truncation bug

**agent-a9cbf5f959a655a0d** — SG11/SG15 across decoder+vit+mamba (SAM-tier): RATE LIMITED

**agent-a9ed515bb0f8034c3** — SG2 + mamba muon/prodigy (SAM-tier): RATE LIMITED

**agent-af69ae1b2844d05e2** — LookSAM vit+mamba (retry): RATE LIMITED

**agent-a68597b8422f3dda7** — LookSAM decoder+vit+mamba (real wiring): RATE LIMITED

**agent-af30a39379980aed7** — SG11/SG15 decoder+vit+mamba (real wiring): API Overloaded

**agent-a888094c5a5c33f75** — decoder: muon/SG11/SG15/looksam/SG2:
- Muon/decoder: COMMITTED (aux_* ABI extension, FusedScalars widening)
- SG11/SG15/looksam/SG2 for decoder: BLOCKED — sharpness/sam_dir producer absent from decoder stage; `mu=phi(grad,sharpness)` blocked by absent sharpness, so no half-staging possible

**agent-a1ba0631097f8eaf6** — SG2 SAM-tier real wiring: Waiting for build monitor (result incomplete)

**agent-a72c369bb5ed95a32** — mamba: prodigy/muon/SG11/SG15/looksam/SG2:
- Prodigy: BLOCKED by shared P2.6 work-steal substrate race (separate from the now-fixed decoder/vit prodigy fix)
- All 5 subsequent cells not started (one at a time rule)
- mamba determinism baseline: adamw/lion/grokfast/grokadamw already A/A/A clean

**agent-a931592c2db19cebf** — decoder wave-2: prodigy/muon/SG11/SG15/looksam/SG2/neuralgrok: RATE LIMITED

**agent-ae687b16d45c9251c** — mamba wave-2: neuralgrok/grokadamw/prodigy/muon/SG11/SG15/looksam/SG2: RATE LIMITED

### 2I. WAVE-1 CONVERSION (CHEAP CELLS: lion/grokfast/grokadamw)

**agent-a2554f93962328810** — decoder lane (lion/adamw/grokfast cheap, then staged):
- lion/adamw/grokfast/grokadamw: CONVERTED for decoder (lion pre-existing; grokadamw needed per-tensor ABI extension for beta1/clip/adaptive-alpha)
- prodigy/muon/SG11/SG15/looksam/SG2/neuralgrok: blocked (INTEGRATION-OPTSTAGES stages required)
- neuralgrok/decoder blocked separately: race fn never wired to `_try_fused_train_step`

**agent-a7196ad48912e684c** — vit lane (lion/adamw/grokfast cheap):
- adamw/lion/grokfast converted and gate-green
- Found: ~0.8ms overhead in per-sample head/CE loops (not worth fixing)
- `SG_TUNED_TILE_N`/`SG_TUNED_VIT_TILE_M` exposed but not registered in compile.py (flagged)

**agent-add6cd61c010cc195** — mamba lane (lion/adamw/grokfast cheap):
- adamw/lion/grokfast/grokadamw converted and gate-green
- TC vs scalar: 0.46× (confirmed expected; scan-dominated)
- mamba nCTA=1 occ=1 launch bounds fix (forces one CTA per SM for cooperative grid)

**agent-a64577e2c484fddd8** — vit wave-2 (grokadamw/prodigy/muon/SG11/SG15/looksam/SG2):
- grokadamw: CONVERTED for vit
- muon/vit: CONVERTED (aux_* FusedScalars ABI extension)
- prodigy/SG11/SG15/looksam/SG2: blocked (staged precompute or model-coupled SAM)

### 2J. WGMMA SUBSTRATE FIXES

**agent-a9ec3bacddba6bb69** — wgmma substrate gates (11/18 pass → fixing 7):
- `test_c_pipelined_matches_unpipelined` (stages 2 and 3 at N=128) — mbarrier choreography bug
- `test_d_determinism_bitwise`, `test_e_occupancy_refuse_oversized_smem`, SASS audit failures
- Result: agent ran into GPU contention with MPS (incomplete — ended mid-Bash call)

### 2K. CLEAN-UP AND PORTFOLIO-READINESS

**agent-aa6d0a8bd0a0113a0** — Cleanup inventory (read-only):
- 15-item prioritized plan generated
- Phase 1 (critical): fix SG2 math-guard TODOs, remove dead Q3 path or wire it
- Phase 2 (org): merge INTEGRATION-*.md → docs/ARCHITECTURE.md, fix COMPONENT_CONTRACT phase-status
- Phase 3 (portfolio): rewrite README, align project name

**agent-ac67669299da87237** — Portfolio readiness inventory:
- Only 7 TODO/FIXME/XXX/HACK across 88K LOC (exceptionally low)
- `.audit_notes.md` sitting untracked-but-not-ignored (minor)
- `_v2` suffix on race driver is vestigial
- `compile.py` is 1.25 MB (noted for awareness, not cruft)
- No destructive action required to reach portfolio-ready

**agent-a61ac501f24dc8300** — Repo cleanup plan:
- Produced KEEP/ARCHIVE/UPDATE classifications for all top-level files
- README banner "No accelerator is present here, all runtime claims are 🟡" is FALSE (H100 validated)
- Recommended 5-step cleanup + push plan

### 2L. RACE HARNESS AND METRICS AUDITS

**agent-ada2f0e60b2e71d6e** — Race harness audit (grokking_race_v2.py):
- Data/INIT comparability is sound
- H1: patience unreachable → wall_time not differentiating; all runs hit max_steps
- H2: component_failure data dropped from run JSON
- H3: crash-bias shrinking N

**agent-ad726484e6f77dd23** — Race harness + tuner fix wave:
- Implemented 15 fixes from audit findings
- `tuning/` dir not git-tracked (untracked, changes on disk only)
- `neuralgrok/decoder` remains eager despite being registered (amplifier-training lane)

**agent-aceaedbee7cdeaab3** — Tuning/measurement code audit (A6 region):
- Single seed 1001 per trial: selection bias risk (~25% spread in decoder AdamW seeds 42/123/456)
- Confirm stage (top-3 on 1002/1003) adequately de-biases winner
- fp16amp × fused-path interaction: tunes eager for fp16amp but fused for bf16/tf32

### 2M. COVERAGE AND VALIDATION MAP

**agent-aa6e79db4e73b8cae** — H100 runtime coverage mapping:
- Race tests fused OPTIMIZER kernels against EAGER PyTorch models
- Fused MODEL kernels (decoder_forward/backward, vit_*, mamba_*) and all 33 sm_90 composition cells: compiled, dispatch-wired, import-validating — but NEVER executed to completion or numerically checked on silicon
- HARDWARE_VALIDATION.md marks them 🟡 (pending runtime parity gate)

**agent-af019369bccdd46e7** — Decoder model kernel numeric parity test:
- Socket close error — work abandoned

**agent-a7916f7209e007852** — Algorithm completeness audit (all 11 optimizers):
- NeuralGrok: ONE genuine suppression — learned amplifier INERT (frozen at random init; MLP never reaches kernel)
- Everything else's headline mechanism is ACTIVE
- Coverage gap: grokfast/grokadamw-EMA/neuralgrok/looksam/sg15/sg2 have no kernel-parity test

**agent-ad78e252e0608e426** — Suppression audit: thinking.type.disabled error — abandoned

**agent-ae1b6ad691fc1af7f** — Suppression audit #2:
- `_FUSED_REGISTRY = {}` never populated → `has_fused()` always False → `_try_fused_step` always falls back to eager
- `_component_guard` silent-swallows exceptions → broken components silently inert
- Fix: populate registry via `register_fused()` or delete shim

**agent-ae2fa1ab7327b9c4b** — Corner-cutting/suppression audit:
- `meta_gate_power` ratchet still present (dormant) but owner directive says remove outright
- HIP `n_tasks` OOB left unfixed
- Tuner val-grok success criterion collides with val-trained meta-nets (leakage)
- "transformer" invalid model name at dispatch call sites (10 sites in grokking_race_v2.py)

### 2N. PROBE / DIAGNOSTIC AGENTS

Several agents were dispatched as probes to determine model identity or connectivity:
- agent-a7220084e13eef2ea: PROBE-OK (claude-opus-4-8 / Opus 4.8)
- agent-a9840f8631acadeb6: thinking.type.disabled error
- agent-aa3be2a3163771d8e: "fable" model error (inaccessible model)
- agent-ab23531c836c04b35: thinking.type.disabled
- agent-ab6482a4df363e6c4: thinking.type.disabled
- agent-a8c564a600753dc05 + agent-afff775e5697b3ba2: Status line setup (shell PS1 → settings.json)

---

## 3. NOTABLE FINDINGS NOT CAPTURED ELSEWHERE

### 3A. The 29-of-33 Blockage

As of the h100-audit-maximal campaign, only 4 cells (adamw/lion × decoder+vit) reached L3-TC wgmma. The remaining 29 require:
- Non-trivial optimizer precompute stages (prodigy: cross-tensor d-reduction; muon: grid-cooperative Newton-Schulz; SG11/SG15: per-tensor mu/gate + sharpness producer)
- Model-coupled SAM 2nd fwd/bwd pass (looksam/SG11/SG15/SG2)
- SG2 meta-net full composition (CSA/HCA/PEER/GRU)
- Mamba scan A/A/A fix (prerequisite for mamba×prodigy and mamba×looksam)

### 3B. CUTLASS Dead Kernel Spill (profile_maximal)

The CUTLASS Hopper TF32 A-transposed RS GEMM (`MMA_64x128x8_F32TF32TF32_RS_TN`) spills 8 bytes but is RUNTIME-DEAD (reachable only via `vit_run_gemm_atb<float>` which returns `cudaErrorNotSupported`). Allowlisted in `profile_maximal.py` by agent-ad59926ad22247544.

### 3C. Mamba Step-Time is 0.46× TC

The Mamba TC port is structurally slower than scalar because the selective scan (mandated-scalar) serializes 16 samples within a block while scalar path parallelizes 64 samples across 64 CTAs. The projection FLOPs (the minority) are tensor-core accelerated but dominated by scan overhead. This was measured and reported honestly (no suppression).

### 3D. Parity Gate Blind Spots

`tests/hw/parity_gate_h100.py` Section 4 for SG2 is `isfinite` only — not a real comparison against `CSAHCAMetaNet.forward_for_bilevel`. This means a mathematically wrong SG2 kernel could pass the gate. SG11 has a 3b lamb sub-check added but the mu/gate path lacks coverage.

### 3E. HARDWARE_VALIDATION.md is 108KB

Enormously large for a doc file — contains prior campaign process history. Agent audit noted it should be archived or condensed for portfolio presentation.

### 3F. `grokking_race_v2.py` "transformer" Model Name Bug

10 call sites pass `model_type="transformer"` (unrecognized) instead of `"decoder"` to `_try_fused_step`. This guards the condition in a way that may be masking a typo rather than correctly routing.

### 3G. Autotuner JIT Build Path (compile.py)

Multiple cascading bugs existed in the JIT build path. After all fixes, the final blocker was `_resolve_sources` not including `csrc/fused/<arch>/*.cu`, causing undefined symbols at dlopen. This was root-caused by agent-aedb807ea371a5617 but hit a socket close error before completing the fix.

### 3H. SG2 Config Key Mismatch

SG2 has 4 mismatched config keys — values passed from Python race are silently ignored, so SG2 runs at constructor defaults (d_model=8, num_experts=144, gru_hidden=4) rather than race config values.

### 3I. transformer_decoder_sm90.cuh OOB Write in Backward

The backward pass reuses `logits_full` (B*S*V=B*S*99 elements) as `grad_stack_out`, then `layernorm_bwd_kernel` writes B*S*D=B*S*128 elements — 29 elements OOB per row. Race config: V=99, D=128. This is a high-severity bug found by the kernel audit.

---

## 4. FAILED / ABANDONED TASKS

### Complete Rate-Limit Failures (work abandoned, subsystem uncovered):
- agent-a9cbf5f959a655a0d: SG11/SG15 SAM-tier decoder+vit+mamba
- agent-a9ed515bb0f8034c3: SG2 + mamba muon/prodigy SAM-tier
- agent-af69ae1b2844d05e2: LookSAM vit+mamba retry
- agent-a68597b8422f3dda7: LookSAM decoder+vit+mamba real wiring
- agent-a931592c2db19cebf: decoder wave-2 (prodigy/muon/SG11/SG15/looksam/SG2/neuralgrok)
- agent-ae687b16d45c9251c: mamba wave-2 (neuralgrok/grokadamw/prodigy/muon/SG11/SG15/looksam/SG2)
- agent-a01ed2a9: dispatch tables+contract
- agent-a2b763b9: opt_stage_supergrok2
- agent-a2cfaba3: tile_pipeline+primitives
- agent-a32166bd: decoder/vit/mamba tc tests
- agent-a34000e3: tuning batch A
- agent-a36fa0b5: .regpressure+.phase2
- agent-a3e4bd4e: grokking_race_v2.py
- agent-a3f5ff12: planning docs 1
- agent-a3f7e3f5: fused/gfx942 cells
- agent-a46c15d8: perf audits
- agent-a53ba83c: hip/gfx942 backend
- agent-a54b6c72: docs+examples+misc md
- agent-a6214d67: planning docs 2
- agent-a6876c3a: results/ characterization
- agent-a6c8fffa: setup.py+wiring+bench
- agent-a6c95102: tests/hw remaining
- agent-a72c5c1e: test_l3tc_tail_gate+sg2_mirror
- agent-a76cd9a4: opt_components+precompute
- agent-a79fdf99: decoder layout/weights/parallel
- agent-a7e9a39d: archived_reports
- agent-a804c6a4 / agent-a904c6a4: algorithms/*.h canonical math
- agent-a85a1fab: tuning batch B
- agent-a8dc8356: profile+verify tooling
- agent-a99a0c93: scripts/*
- agent-aa9e1b9d: tests non-hw+tpu
- agent-ab318a70: bindings/dispatch.cpp
- agent-ac5b60ee: gfx942 kernels B
- agent-acaddb60: megakernel_common+common
- agent-ad195f6: reference parity+oracles
- agent-aea1393c: HARDWARE_VALIDATION.md
- agent-af4840329: pallas+tpu_v6e

### Socket/Error Failures:
- agent-aa26ca8d4c934d57e (Prodigy root-cause): socket close
- agent-a30c9b9a: Pallas tunables wiring — socket close
- agent-ad5f80a2f2a233f76: TC substrate build — socket close
- agent-aedb807ea371a5617: autotuner JIT final fix — socket close
- agent-af019369bccdd46e7: decoder model parity test — socket close

### thinking.type.disabled Errors:
- agent-a9840f8631acadeb6, agent-aa393bb7a863e32b4, agent-ab23531c836c04b35, agent-ab6482a4df363e6c4, agent-ad78e252e0608e426, agent-a08f56b1

### Incomplete Results (truncated before StructuredOutput):
- agent-a9ec3bacddba6bb69: wgmma substrate gates — ran into MPS contention, ended mid-Bash
- agent-adbc6679815eb84f1: 30 eager rows closure — found task too large, consulted advisor instead
- agent-a1ba0631097f8eaf6: SG2 SAM-tier wiring — waiting for build monitor notification
- agent-ac296056c3edacb42: decoder TC R2 retrofit — API Overloaded
- agent-af30a39379980aed7: SG11/SG15 decoder+vit+mamba — API Overloaded
- agent-ad5d89c2eb219b445: multi-CTA decoder TC redesign — ended mid-read
- agent-a1ce0f8e, agent-a3367899: mamba A/A/A race diagnosis — rate limited
- agent-a382d834: dim registration sweep — rate limited
- agent-ac02d8e9 (Lane A tuner fix): weekly rate limit
- agent-ae2a4345 (Lane E GPU gate): weekly rate limit

### Incomplete Output (StructuredOutput call cut off):
- agent-a4c687e7ae779a97c (vit_sm90.cuh audit): result ends at StructuredOutput invocation
- agent-a819bd88dfab4ed9a (prodigy_sm90.cuh audit): same
- agent-ad5b3f582177a3923 (supergrok11_sm90.cuh audit): same

---

## 5. KEY QUANTITATIVE RESULTS

| Metric | Value |
|--------|-------|
| Total agents indexed | 338 |
| Agents with rate-limit/API failures | ~35–40 |
| L3-TC cells committed (h100-audit-maximal) | 33/33 (per campaign end state) |
| L3-TC cells in production routes (current) | 4 of 33 (adamw/lion × decoder/vit) |
| Prodigy A/A/A gate | PASS (decoder+vit; mamba blocked separately) |
| compile.py self-test | 238/0 (after fix wave) |
| tests/tpu | 229 passed (TPU Pallas wiring) |
| H1 win (M-atom interleave) | +2.1% at d=1024 |
| H2 win (CSR embed-grad) | -42.8% step time |
| H3 win (split-K G sweep) | 9.52 TF/s peak |
| Mamba TC vs scalar | 0.46× (slower, expected) |
| Mamba scalar L3 vs eager-adamw | 1.8× faster |
| Mamba scalar L3 memory | 2.50 GB vs 8.9 GB eager (3.6×) |
| Decoder TC vs eager adamw (bf16) | 0.57× (latency-bound) |
| ViT TC vs eager adamw (bf16) | 0.17× (latency-bound) |

---

## 6. OPEN ITEMS / BLOCKERS FROM AGENTS

1. Mamba scan A/A/A race (blocking mamba×prodigy and mamba×looksam) — shared substrate fix needed in `csrc/fused/megakernel_common.cuh` GridBarrier or `opt_stages_precompute.cuh` phaseA work-steal
2. SAM sharpness producer absent from all model stages — blocks SG11/SG15/looksam/SG2 cells
3. `_FUSED_REGISTRY` empty → `_try_fused_step` always eager
4. NeuralGrok amplifier gradient fix (autograd-reachability fix written but build not done)
5. SG2 config key mismatch (4 keys silently ignored)
6. `transformer_decoder_sm90.cuh` backward OOB write (V=99 < D=128)
7. `meta_gate_power` ratchet code still in tree (should be removed per owner directive)
8. `_sccache_env` + `-dlto`/`-gencode` conflict blocks `e2e_smoke_gated` self-test
9. HIP `n_tasks` OOB left unfixed in gfx942 path
10. `emit_variant_source`/`CodegenError` phantom module resolution (compile.py)
11. wgmma substrate pipelined gates (test_c/d/e) still failing after contention cut off the fix agent
12. Decoder TC forward omits biases (found by ViT TC port agent; loose 0.15 tol masks it)
13. `grokking_race_v2.py` "transformer" typo at 10 dispatch call sites
14. CuTe-atom GEMM (`SG_TUNED_GEMM_ENGINE=1`) planned but gate status unknown post-campaign
