export const meta = {
  name: 'sam-tier',
  description: 'Wire SAM-coupled cells (sg11/15/looksam) + sg2 across models, fresh-build verified',
  phases: [{ title: 'SAM' }],
}
phase('SAM')
// 3 lanes by optimizer-family (file-disjoint: each owns its opt header + touches model launchers append-only).
// HARD RULE baked into every prompt: commit a cell ONLY when wiring_check shows wgmma on a FORCE_CUDA fresh build.
const lanes = [
  ['sam11_15', 'supergrok11,supergrok15 across decoder+vit+mamba — SAM 2nd in-kernel fwd/bwd for sharpness side-channel; sg11 per-tensor cosine gate, sg15 host gate scalar'],
  ['looksam', 'looksam across decoder+vit+mamba — sam_dir = g(w+rho*g/|g|) - g via 2nd in-kernel backward, cached every-k'],
  ['sg2_compose', 'supergrok2 across decoder+vit+mamba — compose the PARITY-PROVEN sg2_meta_tail (commit 48e4364) into each model TC cell route; mamba prodigy+muon also (mamba scan-forward symptom: fix or cite)'],
]
const R = await Promise.all(lanes.map(([lbl,desc]) => agent(
  `/workspace/SuperGrok1.5 SAM-tier lane: ${desc}. These were FALSELY marked done before (source committed but wiring_check showed EAGER) — do the REAL wiring this time. HARD RULE: a cell is DONE only when, on a FRESH FORCE_CUDA=1 ./build.sh, wiring_check.py shows path=L3-TC-megakernel(wgmma) for it AND tail_gate (state+A/A/A bit-identical, NaN=real bug retry-3x) passes. Do NOT commit on source-written-but-eager. Per cell maxed (carry committed opts: split-K, single-accum, partitioned bias). Blocked=loud+cited (a cell that genuinely cannot wire stays eager with a written reason — better than a false-done). Edit only your opt headers + needed model launcher append-only. Commit per fresh-build-verified cell. Report per-cell: wiring_check line + A/A/A verdict + TF/s.`,
  { label: lbl })))
return R.map((r,i)=>({lane: lanes[i][0], report: r?.slice?.(0,1200)}))