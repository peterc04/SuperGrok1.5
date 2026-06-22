export const meta = {
  name: 'maxed-cells',
  description: 'Maximize each megakernel cell to roofline, per-model lanes',
  phases: [{ title: 'Maximize', detail: '3 model lanes, cells maxed+gated+measured' }],
}
phase('Maximize')
const lanes = ['mamba', 'decoder', 'vit']
const R = await Promise.all(lanes.map(model => agent(
  `In /workspace/SuperGrok1.5: maximize L3-TC megakernel cells for model=${model}, one cell finished at a time, each maxed BEFORE moving on. First (mamba lane only): profile + FIX the TC-vs-scalar 0.46x bottleneck (per-sample scan serialization; multi-CTA tiling). Per cell: optimize until roofline-verified flat (warp-occupancy, tile shapes, dW ownership; eager/scalar are test refs only); gates = tests/hw/test_l3tc_tail_gate.py + wiring_check.py per cell (no suppression; blocked cells loud + cited); measure quiet (CUDA_MPS_PIPE_DIRECTORY=/nonexistent), B=16384. Order opts: lion,adamw,grokfast first (cheap), then grokadamw(3 mechanisms), prodigy/muon/SG11/SG15/looksam/neuralgrok/SG2 per their INTEGRATION specs. EDIT ONLY ${model} stage/launcher files + shared opt_components via single-writer rule: lane edits opt headers ONLY for ${model}-needed tails, append-only. No commit. Report per-cell TF/s + frac.`,
  { label: `max:${model}` })))
return R.map((r,i)=>({lane: lanes[i], report: r?.slice?.(0,2000)}))