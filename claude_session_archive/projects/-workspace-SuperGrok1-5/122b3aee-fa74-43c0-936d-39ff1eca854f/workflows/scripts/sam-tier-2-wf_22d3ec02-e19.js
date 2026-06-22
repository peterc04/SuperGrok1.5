export const meta = {
  name: 'sam-tier-2',
  description: 'Carry looksam/decoder template to remaining SAM + sg2 + mamba cells',
  phases: [{ title: 'SAM2' }],
}
phase('SAM2')
const lanes = [
  ['looksam_rest', 'looksam across vit+mamba — carry the COMMITTED looksam/decoder template (bab8914): SAM 2nd in-kernel backward sam_dir = g(w+rho*g/|g|)-g'],
  ['sg11_15', 'supergrok11 + supergrok15 across decoder+vit+mamba — SAM sharpness side-channel (2nd in-kernel fwd/bwd), sg11 per-tensor cosine gate, sg15 host gate scalar; reuse looksam/decoder SAM-perturb plumbing'],
  ['sg2_mamba', 'supergrok2 across decoder+vit+mamba via parity-proven sg2_meta_tail (48e4364) composed into each cell route; PLUS mamba muon (P2.7 NS port) + mamba prodigy (diagnose scan-forward A/A/A, fix-or-cite)'],
]
const R = await Promise.all(lanes.map(([lbl,desc]) => agent(
  `/workspace/SuperGrok1.5 SAM-tier-2 lane: ${desc}. Reference the WORKING committed looksam/decoder cell (bab8914) for the in-kernel SAM 2nd-backward pattern. HARD RULE (non-negotiable): a cell is DONE and committed ONLY when, on a FRESH FORCE_CUDA=1 ./build.sh, wiring_check.py prints path=L3-TC-megakernel(wgmma) for it AND tail_gate -k <cell> passes state+A/A/A bit-identical (NaN=real bug, retry 3x, fix don't ignore). NEVER commit source-written-but-eager — that already caused a false 23-vs-16 drift. A cell that genuinely cannot route wgmma stays eager with a written cited reason (loud block > false done). Per cell maxed (carry committed opts: split-K, single-accum, partitioned-bias). Edit only your opt headers + needed model launcher append-only. git commit per fresh-build-verified cell. Report per cell: the exact wiring_check line + A/A/A verdict + TF/s, OR the block citation.`,
  { label: lbl })))
return R.map((r,i)=>({lane: lanes[i][0], report: r?.slice?.(0,1200)}))