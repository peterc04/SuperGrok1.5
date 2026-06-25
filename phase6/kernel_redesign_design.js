export const meta = {
  name: 'supergrok-kernel-redesign-design',
  description: 'READ-ONLY design specs for the two flagship kernel blockers: Mamba smem layer-streaming (unlaunchable 19.56MB/block) and ViT Fork-B grad-partial elimination (memory-bound nCTA*total)',
  phases: [{ title: 'Design', detail: '2 read-only agents emit apply-ready redesign specs' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'spec_path', 'feasible', 'confidence', 'risks', 'summary'],
  properties: {
    area: { type: 'string' }, spec_path: { type: 'string' },
    feasible: { type: 'boolean', description: 'is the redesign tractable as scoped (byte-identical-safe at small size)?' },
    confidence: { type: 'string' }, risks: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}
const COMMON = 'READ-ONLY on /workspace/SuperGrok1.5 (Read/Grep/Glob; do NOT edit/build/run). Write an apply-ready spec to /workspace/impl_diffs/<AREA>.md (VERBATIM OLD snippets + NEW). PROJECT: production path = the L3-TC PERSISTENT wgmma megakernel (1 CTA/SM, GridBarrier, fwd-bwd-opt fused). HARD GATE: the change MUST be byte-identical at the SMALL (production/test) size — the model pytest (tests/hw/test_<model>_tc.py) stays green; the redesign only changes behavior at FLAGSHIP scale. Reference: the DECODER TC is the clean template — its DecTcSmem is layer-INDEPENDENT (ring×tile, not ×kLayers) and its Fork-B path eliminated the per-CTA grad-partial (reuses the workspace as the bf16 acts buffer). gfx942/tpu untouched. Be honest about feasibility — these are deep kernel changes.'

const agents = [
  { name: 'mamba_smem_redesign', label: 'Mamba smem layer-streaming redesign', prompt: COMMON + '\n\n' + [
    'AREA: csrc/fused/sm_90/{model_stage_mamba3.cuh, fused_mamba_megakernel.cuh, model_stage_mamba_tc.cuh}. PROBLEM (confirmed live): the flagship Mamba TC megakernel reinterprets its DYNAMIC smem as MambaSampleSmem (model_stage_mamba3.cuh:~142) which caches ALL layers per sample — layer_in[kLayers][...][d] + kLayers× LayerAct (x_in/g_pre/u_mlp[...][dff]) → sizeof = kMambaSmemBytes ≈ 19.56MB at d=2048/L24, requested as dyn_smem in launch_fused_mamba_megakernel_tc → 88× over the H100 227KB cap → cudaErrorInvalidValue, UNLAUNCHABLE.',
    'GOAL: spec a LAYER-STREAMING smem redesign so MambaSampleSmem is sized by ONE layer (+ a small ring) instead of all kLayers — i.e. make the per-sample smem LAYER-INDEPENDENT like the decoder DecTcSmem. The acts that must persist across layers go to the HBM workspace (DecActs-style), not smem. Study HOW the decoder TC keeps DecTcSmem layer-independent (ring+tile in smem; per-layer acts in the HBM acts workspace bound by dec_acts_bind) and mirror that structure for Mamba. KEY: byte-identical at the SMALL size (test_mamba_tc.py L=2/small-d must stay green — at small L the streamed smem must equal the current all-layers smem behavior, or be #if/if-constexpr-gated so small is unchanged). Spell out: the new MambaSampleSmem layout (one-layer + ring), where the cross-layer acts move in the HBM workspace (+ the workspace-size formula), the per-layer loop restructure, and the launch dyn_smem request (now < 227KB). HONEST feasibility: this is a real architectural change; if a full streamed rewrite is too large for an exact spec, give the precise STRUCTURE + the exact smem/workspace formulas + the gated insertion points. gate after apply: test_mamba_tc.py green (small byte-identical) + the flagship Mamba TC compiles AND launches (dyn_smem < 227KB). spec -> /workspace/impl_diffs/mamba_smem_redesign.md.',
  ].join('\n') },

  { name: 'vit_forkb', label: 'ViT Fork-B grad-partial elimination', prompt: COMMON + '\n\n' + [
    'AREA: csrc/fused/sm_90/{model_stage_vit_tc.cuh, fused_vit_megakernel.cuh, mega_vit_real_adamw_tc_launcher.cu}. PROBLEM (confirmed): the ViT TC megakernel still carries the legacy per-CTA grad-partial workspace = nCTA*total floats (8*1.596B*4 ≈ 51GB at ncta_cap=8 flagship) → OOMs at full occupancy, forced to ncta_cap=4 (3% of 132 SMs). The DECODER TC ELIMINATED this (its Fork-B path reuses the workspace as the bf16 acts buffer + does split-K dW reduction into the workspace instead of per-CTA grad partials).',
    'GOAL: spec porting the decoder Fork-B grad-partial elimination to ViT. Study the decoder TC: how dec_tc_workspace_floats avoids nCTA*total (the Fork-B dW: split-K partials + the acts-buffer reuse, NOT nCTA*total grad partials), and the dW-spec/split-K machinery in model_stage_decoder_tc.cuh. Mirror it for ViT: replace the ViT nCTA*total grad-partial with the decoder-style split-K dW reduction into the reused workspace, so the ViT workspace drops to ~acts+wbf+dW-transpose (no nCTA*total term) → flagship ViT runs at ncta_cap=8 (full occupancy) within 80GB. KEY: byte-identical at the SMALL ViT size (test_vit_tc.py stays green — the Fork-B path must produce the SAME grad as the current path; the decoder Fork-B is parity-proven, mirror its exactness). Spell out the new vit_tc_workspace_floats (no nCTA*total), the ViT dW split-K reduction, and the launcher change. HONEST: if the ViT dW structure differs enough from the decoder that a full port is large, give the precise mapping + the exact workspace formula + insertion points. gate: test_vit_tc.py green + flagship ViT TC compiles + the workspace fits ncta_cap=8 at flagship (no OOM). spec -> /workspace/impl_diffs/vit_forkb.md.',
  ].join('\n') },
]

phase('Design')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'Design', schema: SCHEMA })
    .then(r => ({ name: a.name, ...(r || { area: a.name, spec_path: '', feasible: false, confidence: 'AGENT NULL', risks: ['agent died'], summary: 'null' }) }))))
log('Kernel-redesign specs: ' + results.filter(r => r && r.spec_path).length + '/' + agents.length + ' (feasible: ' + results.filter(r=>r&&r.feasible).length + ')')
return results
