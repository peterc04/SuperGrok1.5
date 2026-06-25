# AREA: deadcode_source — provably-dead code in the TRUE SOURCE (APPLY-READY removal list)

Scope: `grokking_optimizers/*.py`, `csrc/**`, `tests/**`, `tuning/**`, root `*.py`.
Method: reachability-checked. A symbol is removed ONLY if it has zero reachable
callers (no `opt_id`/pybind/test/CI/codegen path). Conservative: if reachable via
any of those, it STAYS.

Production path that MUST stay (verified untouched by this spec):
- L3-TC persistent wgmma megakernel: the 6 sm_90 `_tc` cells/launchers
  (`mega_{decoder,mamba,vit}_real_adamw_tc{,_launcher}.cu`) — KEEP.
- The prebuilt `_ops` (csrc/bindings + the globbed backend TUs) — KEEP.
- The 3 model layouts (decoder/mamba3/vit `*_layout.cuh`, `*_flagship_layout.cuh`) — KEEP.
- The gfx942 tree (`csrc/backends/hip/gfx942/**`, `csrc/fused/gfx942/**`) — KEEP.
- The tpu tree (`csrc/backends/pallas/**`, `csrc/fused/tpu_v6e/**`) — KEEP.
- math-drift guard + parity/determinism gates — KEEP (one obsolete, always-throwing
  mamba gate is removed; see below — it is NOT a parity/determinism/math-drift gate).

=============================================================================
## SUMMARY OF FINDINGS (what is removable vs what the task flagged that ISN'T)
=============================================================================

REMOVABLE (provably dead, reachability-checked): exactly ONE path.
  (1) The obsolete `tc_dump_outproj_operands` on the Mamba scalar-TC cell — the
      C++ function (a pure `TORCH_CHECK(false, …)` stub), its pybind `.def`, AND
      the sole caller test `test_tc_proj_dw_exact_on_own_operands` (which can
      ONLY RuntimeError when it runs on hardware). See REMOVAL 1.

NOT REMOVABLE — task candidates that are NOT dead / do NOT exist (KEEP, evidence below):
  (2a) `MambaModel` / `SelectiveSSMLayer` in grokking_race_v2.py — LIVE. Imported
       + instantiated by `tests/hw/test_mamba_megakernel.py:39` (`g.MambaModel(...)`)
       and transcribed by `tests/hw/mamba_oracle.py` / `mamba3_oracle.py`. KEEP.
  (2b) `_maybe_wrap_cuda_graph` no-ops — DO NOT EXIST in the current tree
       (`grep -c _maybe_wrap_cuda_graph grokking_race_v2.py` → 0). The
       datasets.md / datasets_v2.md spec OWNS grokking_race_v2.py dead-code and
       lists that symbol; this AREA does NOT touch grokking_race_v2.py to avoid a
       double-removal / merge conflict. NOTHING to do here.
  (3a) Dead generated `mega_<model>_<opt>.cu` cells that `#include` the removed
       `fused_megakernel.cuh` — DO NOT EXIST. The 33 sm_90 per-cell `.cu`
       (`mega_mamba3_adamw.cu`, …) are ABSENT on disk; only the 6 `_tc` cells exist
       (the REAL path). They were already deleted; there is no file to remove.
       The gfx942 `mega_*.hip` cells `#include "csrc/fused/gfx942/fused_megakernel.hip.hpp"`
       which EXISTS (gfx942 tree STAYS). NOTHING to remove.
  (3b) `launch_<opt>.cu` / `csrc/backends/cuda/sm_90/models/*.cu` shims
       "referenced by verify_all/profile but absent-from-build" — these PATHS are
       absent on disk (no `launch_*.cu`, no `models/` subdir under cuda/sm_90).
       They are referenced ONLY by `grokking_optimizers/verify_all.py` — a LIVE CI
       gate (`.github/workflows/ci.yml:1244` runs `verify_all --phase 1 4 5`). The
       references are the maximality harness DETECTING the archive gap, not dead
       code. Removing them would alter a live gate. KEEP verify_all.py verbatim.
  (4)  Unreferenced helpers / dead `#if` / commented-out blocks — none found that
       are provably dead. `_maybe_checkpoint` is duplicated across 4 optimizer
       files but each is used in-file (reachable). The only `#if 0` hit is a prose
       comment reference (amdgcn_primitives.hip.hpp), not a dead branch. The
       decoder/vit `tc_dump_ff2_operands` are REAL, WORKING functions driven by
       LIVE tests (test_decoder_tc.py:662, test_vit_tc.py:591) — KEEP. The mamba
       sibling is the ONLY dead one (it is a stub; the Mamba-3 scalar path stores
       no bf16 acts). Root scratch probes (`_probe1.py`, `_sg_realsg_probe.py`,
       `_fg_runner.py.disabled`) have zero importers but are operator scratch
       tooling, not build/gate code — NOT listed (conservative; out of "provably
       dead production code" intent, and removing operator scratch is risky).

removable_lines (conservative source-dead total): 56
  = cu function block (19) + cu pybind .def (2) + test function block (34) +
    1 of the 2 leading blank separator lines before the test (net 1).
  (Detail per removal below. The stale gate-(2) line in the test module docstring
   is OPTIONAL prose cleanup, NOT counted.)

=============================================================================
## REMOVAL 1 — obsolete `tc_dump_outproj_operands` on the Mamba scalar-TC cell
=============================================================================

REACHABILITY EVIDENCE (zero LIVE callers beyond the one always-failing test):

  $ grep -rn tc_dump_outproj_operands  (over *.py *.cu *.cuh *.cpp *.h *.hpp *.inc)
    tests/hw/test_mamba_tc.py:551:    dY, X = mod.tc_dump_outproj_operands(params, B)
    csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu:148:  static … tc_dump_outproj_operands(…)
    csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu:155:  TORCH_CHECK(false, "tc_dump_outproj_operands: obsolete …")
    csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu:245:  mm.def("tc_dump_outproj_operands", &tc_dump_outproj_operands, …)
  (No other matches. Hits in `claude_session_archive/**` are archived JSONL
   transcripts, NOT source — out of scope.)

WHY DEAD (provable):
  - The C++ function body is unconditionally `TORCH_CHECK(false, "… obsolete on the
    Mamba-3 scalar path …")` — it can never return; its own comment declares it
    OBSOLETE (no stored bf16 projection acts on the Mamba-3 scalar-per-sample path).
  - The ONLY caller is `test_tc_proj_dw_exact_on_own_operands` (test_mamba_tc.py),
    which calls it at line 551. On a non-sm_90 runner the test is SKIPPED
    (`_GATE = pytest.mark.skipif(not _sm90a_available())`); on real sm_90 hardware
    it RUNS and RuntimeErrors at line 551 (the TORCH_CHECK fires). So the test is a
    broken/obsolete gate with no green outcome on hardware. It is NOT selected by
    name in CI (no `-k proj_dw`), not xfail-marked (conftest.py / pyproject have no
    xfail), and is not part of the parity/determinism/keystone gates (the mamba TC
    grad parity is covered by `test_tc_grad_parity_*`/keystone + the determinism
    A/A/A gate, which remain).
  - The pybind `.def` at lines 245-246 is the only registration of the symbol.

NOTE — siblings are LIVE, DO NOT touch them: decoder/vit `tc_dump_ff2_operands`
  are real working functions (mega_decoder_real_adamw_tc.cu:147 / mega_vit_real_adamw_tc.cu:277,
  registered at :310 / :415) driven by LIVE gates (test_decoder_tc.py:662,
  test_vit_tc.py:591). Only the MAMBA `tc_dump_outproj_operands` is the dead stub.

-----------------------------------------------------------------------------
### 1A. DELETE the C++ function + its calibration-hook comment
FILE: /workspace/SuperGrok1.5/csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu
Remove the ENTIRE block below VERBATIM (the leading blank line 139 stays; this
removes lines 140-158, the comment + function. Leave the `}` of the prior function
on line 138 and the blank line on line 139 intact. After 158 there is a blank line
159 — keep exactly one blank line between the prior function and the next section).

OLD (delete verbatim):
```
// CALIBRATION HOOK (test_mamba_tc.py::test_tc_proj_dw_exact_on_own_operands):
// slice the kernel's OWN stored bf16 acts dY_dyout[L1] [T,d] and X_ygated[L1]
// [T,d_inner] from the workspace, returned as fp32 CPU tensors. The gate
// contracts them in fp32 ascending-t and compares to the kernel's
// out_proj.weight grad slice — isolating the output-stationary dW GEMM (K=T)
// from the operand-chain bf16 divergence. A ~1e-6 match proves the dW GEMM is
// bit-exact on its own operands, calibrating the per-tensor bf16 tol as headroom
// over a GEMM-exact floor. Reuses the cached workspace (call AFTER tc_train_step).
static std::vector<torch::Tensor> tc_dump_outproj_operands(torch::Tensor params, int64_t B) {
    // OBSOLETE on the Mamba-3 scalar-per-sample path: there is no stored bf16 acts
    // region (the dW is accumulated scalar-style into the per-CTA full-grad partial,
    // not output-stationary on wgmma). The Mamba-1 calibration hook this served is
    // not part of the fp64 parity gate; the grad parity is validated end-to-end by
    // the keystone grad-parity test + the looksam/SG sam_dir/sharpness gates.
    (void)params; (void)B;
    TORCH_CHECK(false, "tc_dump_outproj_operands: obsolete on the Mamba-3 scalar path "
                       "(no stored bf16 projection acts; dW is in the full-grad partial).");
    return {};
}
```
(19 lines removed. The blank line that followed it — line 159 — becomes the single
blank separating the prior `}` from the `#if SG_MB_SCALAR_MEGAKERNEL` block.)

-----------------------------------------------------------------------------
### 1B. DELETE the pybind registration of the removed symbol
FILE: /workspace/SuperGrok1.5/csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu
Inside `PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm)`, remove the two-line `.def`
(lines 245-246). The `tc_train_step` `.def` above (ending `pybind11::arg("ncta_cap") = 0);`)
and the `#if SG_MB_SCALAR_MEGAKERNEL` below stay.

OLD (delete verbatim):
```
    mm.def("tc_dump_outproj_operands", &tc_dump_outproj_operands,
           "gate-only: dump the kernel's stored dY_dyout[L1], X_ygated[L1] (fp32 CPU)");
```
(2 lines removed. This snippet is unique in the file.)

-----------------------------------------------------------------------------
### 1C. DELETE the sole-caller test (the only consumer; always RuntimeErrors)
FILE: /workspace/SuperGrok1.5/tests/hw/test_mamba_tc.py
Remove the entire `test_tc_proj_dw_exact_on_own_operands` function INCLUDING its
`@_GATE` decorator and ONE of the two blank separator lines above it (lines
529-563; keep the blank line 528 after the prior test, and keep the two blank
lines 564-565 before `test_tc_determinism`). Net: delete the decorator+function
(34 lines: 530-563) plus one separator blank (529).

OLD (delete verbatim — the decorator + the whole function):
```
@_GATE
def test_tc_proj_dw_exact_on_own_operands():
    """(2) The output-stationary dW GEMM (K=T) is bit-exact on the kernel's OWN
    stored bf16 acts: dump dY_dyout[L1], X_ygated[L1], contract fp32 ascending-t,
    compare to the kernel's out_proj.weight[L1] grad slice. ~1e-6 isolates the dW
    GEMM from the operand-chain bf16 divergence (calibrates the per-tensor tol)."""
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    named = _eager_mamba3_named(seed=7)
    B = 128
    g = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g)
    targets = torch.randint(0, P_HEAD, (B,), generator=g)
    params = _flat(named).to(dev)
    total = int(mod.TOTAL)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    _, kgrad = _run_tc_step(mod, params.clone(), tokens.to(dev), targets.to(dev),
                            state, lr=1e-3, betas=(0.9, 0.98), wd=0.0, eps=1e-8, step=1)
    kgrad = kgrad.cpu().double()
    # dump reads the CACHED GPU workspace; params is used only for device+nCTA → pass the GPU tensor.
    dY, X = mod.tc_dump_outproj_operands(params, B)   # [T*d], [T*d_inner] fp32 (CPU)
    T = B * SEQ
    dY = dY.double().reshape(T, D_MODEL)
    X = X.double().reshape(T, D_INNER)
    ref = dY.t() @ X    # [d, d_inner] — out_proj.weight[L1] grad, fp32 over the kernel's bf16 acts
    lay = mamba3_param_layout()
    idx = lay["names"].index("layers.1.mixer.out_proj.weight")
    off, sz, shape = lay["offsets"][idx], lay["sizes"][idx], lay["shapes"][idx]
    kg = kgrad[off:off + sz].reshape(shape)
    err = (kg - ref).abs().max().item(); den = ref.abs().max().item() + 1e-30
    rel = err / den
    print(f"[mbtc] ISO out_proj.weight[L1] dW: kernel vs fp32(own bf16 acts) max|err|={err:.3e} rel={rel:.3e}")
    assert rel < 5e-3, f"proj-dW ISO rel {rel:.2e} — the dW GEMM is NOT bit-exact on its own operands (a real bug)"
```
After removal, the file goes `…assert worst <= 1.0, …` (end of the prior test) →
ONE blank line is removed so exactly two blanks separate it from the next `@_GATE`
`def test_tc_determinism():`. (34 fn lines + 1 separator blank = 35 lines from the
file; counted as 34 fn + net 1 in the total.)

-----------------------------------------------------------------------------
### 1D. (OPTIONAL — prose only, NOT counted) stale gate-(2) line in the docstring
FILE: /workspace/SuperGrok1.5/tests/hw/test_mamba_tc.py  (module docstring, ~lines 14-15)
After removing the test, the top docstring's "(2) proj-dW ISO …" bullet is stale.
This is prose, not code; updating it is OPTIONAL and risk-free. If applied:
OLD:
```
  (2) proj-dW ISO: kernel's OWN bf16 acts → fp32 contraction == kernel out_proj
      dW (~1e-6; the dW GEMM bit-exactness, isolated from the operand chain).
```
(Leave the (1)/(3)/(4)/(5) bullets. Renumbering is unnecessary — they are prose.)

=============================================================================
## VERIFY AFTER APPLY (reachability stays clean; gates stay green)
=============================================================================
  $ grep -rn tc_dump_outproj_operands csrc tests   # → NO matches
  # The mamba TC TU still JIT-compiles (it owns its own PYBIND11_MODULE and is
  # auto-excluded from _ops by setup.py:_owns_extension_module — unchanged).
  # Remaining mamba TC gates stay: test_tc_grad_parity*/keystone, test_tc_determinism
  # (A/A/A), test_tc_short_trajectory, test_tc_step_time_vs_scalar — none reference
  # the removed symbol.
  # NO change to: the 6 _tc cells' core paths, _ops, the 3 layouts, gfx942/tpu trees,
  # verify_all.py (live CI gate), grokking_race_v2.py (owned by datasets_v2.md).
