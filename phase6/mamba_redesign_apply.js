export const meta = {
  name: 'supergrok-mamba-redesign-apply',
  description: 'Apply the Mamba flagship smem redesign (Level A layer-streaming + Level B stream the per-sample scratch to HBM) so the flagship Mamba TC megakernel launches under the 227KB cap; gated byte-identical at small size',
  phases: [{ title: 'Redesign', detail: 'implement + gate the Mamba smem redesign' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'level_a_done', 'level_b_done', 'launches_at_flagship', 'dyn_smem_kb', 'small_byte_identical', 'commit_sha', 'remaining', 'summary'],
  properties: {
    area: { type: 'string' },
    level_a_done: { type: 'boolean' }, level_b_done: { type: 'boolean' },
    launches_at_flagship: { type: 'boolean', description: 'does the flagship Mamba TC kernel actually launch (dyn_smem < 227KB)?' },
    dyn_smem_kb: { type: 'number', description: 'the flagship dyn_smem request in KB after your redesign' },
    small_byte_identical: { type: 'boolean', description: 'test_mamba_tc.py still green (small size byte-identical)?' },
    commit_sha: { type: 'string' }, remaining: { type: 'array', items: { type: 'string' } }, summary: { type: 'string' },
  },
}
const PROMPT = [
  'ISOLATED GIT WORKTREE of /workspace/SuperGrok1.5 (Edit/Write/Bash/build/run freely). GOAL: make the FLAGSHIP Mamba-3 TC megakernel (d=2048, L=24) actually LAUNCH on H100 — currently unlaunchable: fused_mamba_megakernel_tc reinterprets dynamic smem as MambaSampleSmem = 19.56 MB/block (88x over the 227 KB cap) -> cudaErrorInvalidValue.',
  'SPEC: /workspace/impl_diffs/mamba_smem_redesign.md — read IN FULL. It gives Level A (exact VERBATIM OLD/NEW: gate kMbStreamSmem, the one-LayerAct + kMbActsRing=2 ring struct, cross-layer acts moved to a per-CTA HBM MbActsHbm region bound by mb_acts_bind, the fwd/bwd per-layer spill/refill restructure, the workspace + dyn_smem changes) and Level B (structure + exact formulas: also stream the big per-sample SEQ x {d_inner,d_ff} scratch to HBM / tile over d_inner so resident smem drops to ~121 KB < 227 KB). Apply Level A verbatim, then IMPLEMENT Level B from its structure+formulas (this is the part that actually clears 227 KB).',
  'HARD GATE — byte-identical at SMALL size: everything behind if-constexpr(kMbStreamSmem) which is FALSE at d=128/d=1024 (the all-layers struct is <227KB there), so the production/test TU folds to the shipped code. RUN: CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_mamba_tc.py -q -> MUST stay green (note: 2 pre-existing failures B_bias-tol + obsolete proj_dw may be present from the mamba_test_fix track; if so, confirm YOUR change does not ADD failures vs the pre-change baseline — run the baseline first). Then build the FLAGSHIP Mamba TC (mirror flagship_train.py: -include mamba flagship layout + -DSG_MB_SCALAR_MEGAKERNEL=0 etc.) and VERIFY it LAUNCHES: actually run one fwd-bwd-opt step at d=2048/L24 (ncta_cap=8, tiny B) and confirm no cudaErrorInvalidValue + finite loss. Report dyn_smem_kb (must be < 227).',
  'When done: restore dirty *.pyc, git add -A, git commit -m "mamba flagship smem redesign (layer-stream + scratch-to-HBM)", report sha. BE HONEST: if Level B is too large to fully land, land Level A + as much of B as compiles+gates, report launches_at_flagship=false + the exact remaining scratch that must move, and dyn_smem_kb achieved. gfx942/tpu untouched. NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem.',
].join('\n')

phase('Redesign')
const r = await agent(PROMPT, { label: 'Mamba flagship smem redesign', phase: 'Redesign', schema: SCHEMA, isolation: 'worktree' })
log('Mamba redesign: A=' + (r&&r.level_a_done) + ' B=' + (r&&r.level_b_done) + ' launches=' + (r&&r.launches_at_flagship) + ' smem=' + (r&&r.dyn_smem_kb) + 'KB')
return r || { area:'mamba', level_a_done:false, level_b_done:false, launches_at_flagship:false, dyn_smem_kb:0, small_byte_identical:false, commit_sha:'', remaining:['agent died'], summary:'null' }
