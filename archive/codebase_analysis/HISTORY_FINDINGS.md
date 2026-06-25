# Git-history findings (first-hand, lead) — "the most up-to-date versions"

## How the campaign was built
The 2026-06-25 campaign on `claude/custom-optimizer-analysis-HFYhg` was built by **parallel git-worktree
"tracks"** (one per spec: vit_flagship, datasets_v2, tp_remainder, memory_strategy, ep_size, attention_shard,
host_bringup, mamba smem redesign, nvshmem_pybind, mamba_test_fix, staged_opt_plumbing) that were **cherry-picked
into the mainline** (reflog shows the alternating `commit`/`cherry-pick` pattern). 14 worktree branches survive.

## Current integrated state
- **HEAD = `e69df73`** (closure commit). It is **1 commit ahead of `origin/main`** (`03bd3f0`); the closure
  commit is the only unpushed one. The campaign work IS on origin/main.
- The remote tracking branch `origin/claude/custom-optimizer-analysis-HFYhg` is frozen at `c29ed4e` (the 06-22
  clone point) — that's why `LEDGER.json`/`PROGRESS.md` header cite c29ed4e. **Use git HEAD, not the ledgers.**
- HEAD is the **most up-to-date integrated line**: every worktree tip is either (a) already cherry-picked in, or
  (b) BEHIND HEAD (their `git diff HEAD <tip>` is deletion-dominant — HEAD has files they lack: tuning/_tp8_*,
  flagship_distributed.py, the nvshmem bring-up files, the mamba smem redesign).

## The ONE piece of un-integrated NEWER work  ⚠️
**`3df7ee9` (worktree wf_89f9a418-f9d-3, "finish: mamba_test_fix") was NEVER cherry-picked into HEAD.**
It rewrites `tests/hw/test_mamba_tc.py` (+86/-12). HEAD still has the OLD version. The fix resolves the
"Mamba 3/5 failing" status the docs report, and it is NOT a loosening — it is a rigorous recalibration:
- `test_tc_single_step_grad_parity`: adds a per-tensor **calibrated** tolerance. The 2 "failures" are NOT bugs —
  they are high-cancellation B*S-token-reduced grads (the complex SSM B/C stream biases + BCNorm weights from the
  exponential-trapezoidal scan backward, the token-embedding scatter, the SwiGLU down_proj dW) whose IRREDUCIBLE
  bf16 floor exceeds the 0.08 base tol (measured worst: B_bias floor 0.1546, C_norm.weight 0.1415, tok.weight
  0.0994, down_proj 0.0875). Elevated tol = 0.30 (~1.94x the max floor, within the 1.5-3x band) for that group,
  0.08 for the rest, PLUS a per-tensor no-suppression witness (k-vs-bf16 must ride within 3x its OWN bf16-vs-fp64
  floor; a real bug trips it even under the loose tol). This is the Mamba-3 analogue of the decoder's 0.15 tier.
- `test_tc_proj_dw_exact_on_own_operands`: **skipped with a documented reason** — the wgmma-operand-dump witness
  doesn't apply to Mamba-3's SCALAR SSM path (the SSM grad accumulates into the per-CTA full-grad partial, not
  output-stationary on wgmma; the binding `tc_dump_outproj_operands` is hard-obsoleted `TORCH_CHECK(false)` at
  `mega_mamba_real_adamw_tc.cu:148`). The 4 tensor-core GEMMs (in/x/dt/out_proj) are validated end-to-end elsewhere.

### Action implication
To get Mamba to a clean gate, cherry-pick/apply `3df7ee9`'s `test_mamba_tc.py` (or `git checkout 3df7ee9 --
tests/hw/test_mamba_tc.py`). This is "the most up-to-date version" of that test. Decide whether to integrate it
before relying on the Mamba gate.

## `fd80883` (staged_opt_plumbing) — empty
Tree-identical to its base (5e084ca); a no-op finish commit. No missed source. (The phase6/staged_opt_plumbing/
scratch dir is workflow working files, not committed source.)
