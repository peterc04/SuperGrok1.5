export const meta = {
  name: 'wave2-finish',
  description: 'Finish decoder+mamba staged cells (rate-limited lanes)',
  phases: [{ title: 'Finish' }],
}
phase('Finish')
const lanes = [
  ['decoder','muon,supergrok11,supergrok15,looksam,supergrok2'],
  ['mamba','prodigy,muon,supergrok11,supergrok15,looksam,supergrok2 + fix mamba A/A/A determinism fails'],
]
const R = await Promise.all(lanes.map(([m,opts]) => agent(
  `/workspace/SuperGrok1.5 finish lane model=${m}: convert+max cells ${opts}, one at a time maxed-before-next, carrying committed campaign optimizations (single accumulator, partitioned-K bias, split-K dW, P2.5 global clip). STAGED tails per INTEGRATION-OPTSTAGES (prodigy global-d, muon grid-cooperative NS, SG11/15 SAM sharpness side-channel via 2nd in-kernel fwd/bwd, looksam sam_dir), SG2 via parity-proven sg2_meta_tail. Per cell: tail_gate(state+A/A/A bit-identical — re-run A/A/A 3x if NaN, NaN=real bug to fix not ignore)+wiring_check+roofline quiet B=16384. Edit only ${m} stage/launcher + needed opt headers append-only. Blocked=loud+cited. Commit per validated cell with git. Report per-cell TF/s+verdict.`,
  { label: `fin:${m}` })))
return R.map((r,i)=>({lane: lanes[i][0], report: r?.slice?.(0,1500)}))