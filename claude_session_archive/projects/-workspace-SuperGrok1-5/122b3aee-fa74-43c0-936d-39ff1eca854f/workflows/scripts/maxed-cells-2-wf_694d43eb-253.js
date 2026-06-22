export const meta = {
  name: 'maxed-cells-2',
  description: 'Wave 2: convert+max remaining cells per lane',
  phases: [{ title: 'Wave2' }],
}
phase('Wave2')
const lanes = [
  ['decoder','prodigy,muon,supergrok,supergrok15,looksam,supergrok2,neuralgrok(mamba-port-skip)'],
  ['vit','grokadamw,prodigy,muon,supergrok,supergrok15,looksam,supergrok2'],
  ['mamba','neuralgrok,grokadamw,prodigy,muon,supergrok,supergrok15,looksam,supergrok2 + fix mamba determinism gate fails'],
]
const R = await Promise.all(lanes.map(([m,opts]) => agent(
  `/workspace/SuperGrok1.5 wave-2 lane model=${m}: convert+max cells for opts ${opts}, one at a time, maxed before next (carry the wave-1 vit/decoder optimizations: single accumulator, partitioned bias K, split-K). STAGED tails per INTEGRATION-OPTSTAGES (prodigy global d, muon NS grid-cooperative, SG11/15 sharpness side-channel via SAM 2nd fwd/bwd in-kernel), SG2 via the parity-proven sg2_meta_tail. Per cell: tail_gate(state+A/A/A)+wiring_check+roofline quiet (B=16384). Edit only ${m} stage/launcher + needed opt headers append-only. Blocked = loud+cited. No commit. Report per-cell TF/s table.`,
  { label: `w2:${m}` })))
return R.map((r,i)=>({lane: lanes[i][0], report: r?.slice?.(0,1500)}))