---
name: supergrok-working-prefs
description: "How to work on the SuperGrok1.5/SuperGrok2 H100 stack — L3-TC-only, prebuilt binaries, caching, exhaustive reading, parallel agents, minimize GPU hours"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

User's directives for the SuperGrok1.5 (a.k.a. SuperGrok2) grokking-optimizer H100 stack at
github.com/peterc04/SuperGrok1.5 (work on /workspace, branch claude/custom-optimizer-analysis-HFYhg):

- **L3-TC kernels ONLY.** Never use the scalar/naive megakernel path. The production path is the
  persistent fused wgmma megakernel (`fused_<model>_megakernel_tc<Opt>`, `gemm_impl="wgmma"`,
  `-DSG_TUNED_GEMM_IMPL=1`). There is an **upgraded** version = the BAKED perf levers (decoder
  `SG_TUNED_DEC_FWD_PIPE=1`/`FWD_STAGES=4` +1.49×, `DW_STAGE=1` +2.05×; ViT `VIT_P1_SUBTILE_S=8`
  4.02×) — already the default `#ifndef` values — plus the `_dectc_codegen/{deep_s3,deep_s4,postedit}`
  PTX-edit experiments. Use these, not the gated-off scalar fallback.
- **Use the PREBUILT binaries from prior compiles** instead of recompiling: `_ops*.so`,
  `tune11_out/*/*.so` (tuned), `task11_bench_build/{A_sk4,B_sk4,C_sk2}/*.so` (nvcc-vs-compile-file
  3-point), `nvcc_baseline_build/`, and `_dectc_codegen/*/*.{cubin,ptx,fatbin}`. Disassemble cubins
  statically (cuobjdump/nvdisasm) — no GPU needed.
- **Use the compile-file caching for fast recompiles:** source `.fast_build_env.sh`
  (PYTORCH_NVCC=.build_tools/nvcc-cached→sccache, CXX=.build_tools/g++-cached→ccache, caches under
  `.build_cache/{sccache,ccache}`); the warm `.build_cache` (1.3G) is committed.
- **Read the codebase LITERALLY exhaustively** — do NOT grep-skim. Read full files. The authoritative
  reference is `CODEBASE_EXPLAINED.md`; live state in `SESSION_STATE.md`/`PLANNING_INPUT.md`/
  `.perf/phase1_status_audit.md`; the commit history + `claude_session_archive/` hold key context.
- **Parallelize aggressively** with agents/workflows (ultracode). Get things done FAST and with the
  **least GPU hours** (prefer prebuilt artifacts + static analysis; batch any GPU work 8-wide).

**Why:** the user caught me grep-skimming Phase 0 and missing the prebuilt binaries + upgraded kernels.
**How to apply:** before any recompile, look for a prebuilt artifact; default to the L3-TC path; fan
out reading across parallel agents instructed to read files in full. See [[ncu-blocked-runpod]].
