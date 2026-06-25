export const meta = {
  name: 'supergrok-apply-tracks-parallel',
  description: 'Parallel WORKTREE write-agents: each applies one disjoint spec (ViT flagship / Mamba flagship / datasets) + builds + gates + commits to its branch, reports the commit sha for the lead to cherry-pick',
  phases: [{ title: 'Apply', detail: '3 isolated write-agents apply+build+gate+commit one spec each' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['spec', 'applied_clean', 'gate_pass', 'commit_sha', 'branch', 'deviations', 'gate_output_tail', 'diffstat', 'summary'],
  properties: {
    spec: { type: 'string' },
    applied_clean: { type: 'boolean' },
    gate_pass: { type: 'boolean' },
    commit_sha: { type: 'string', description: 'full sha of the commit you made in your worktree (empty if you did not commit)' },
    branch: { type: 'string', description: 'the branch your worktree is on (git branch --show-current)' },
    deviations: { type: 'array', items: { type: 'string' } },
    gate_output_tail: { type: 'string' },
    diffstat: { type: 'string' },
    summary: { type: 'string' },
  },
}
const COMMON = `You operate in an ISOLATED GIT WORKTREE copy of /workspace/SuperGrok1.5 (branched from the full current session state — it INCLUDES the CuTe atoms, the flagship decoder layout emitter, and the decoder dW-generalization). You may freely Edit/Write/Bash/build. Your changes are isolated from other agents.
PROCEDURE: (1) Read your spec file IN FULL. (2) Apply EVERY OLD/NEW block with exact-match Edits (OLD blocks were copied verbatim from the live files; if one does not match, the file drifted — find the true current text, apply the equivalent change, RECORD it in deviations). Write NEW files in full. (3) Run the spec's gate_commands EXACTLY. (4) If a gate fails on a mechanical apply error (mismatched brace, missing include), fix the real cause, re-run, record in deviations — NEVER loosen a correctness gate to force a pass. (5) When the gates pass (or you have done your best), commit ALL your changes: \`cd <your worktree root> && git add -A && git commit -m "track: <spec name> applied + gated"\`, then report the full sha (\`git rev-parse HEAD\`) and branch (\`git branch --show-current\`). (6) Report the schema fields HONESTLY — a truthful gate_pass=false with the error tail is more useful than a forced pass.
PROJECT RULES: production path = L3-TC persistent wgmma megakernel; the flagship work MUST keep the existing (non-flagship) layout BYTE-IDENTICAL (regenerate the current-layout header and diff vs the committed one — MUST be empty) and keep the current-L parity/determinism gate green; new flagship layouts go to a SEPARATE header; gfx942/tpu untouched. scripts/compile_to_object.sh is at the repo root; CUTLASS at third_party/cutlass/include. Be exhaustive and precise — you are trusted to land this track end to end.`

const specs = [
  { name: 'vit_flagship', file: '/workspace/impl_diffs/vit_flagship.md', label: 'apply ViT flagship' },
  { name: 'mamba_flagship', file: '/workspace/impl_diffs/mamba_flagship.md', label: 'apply Mamba flagship' },
  { name: 'datasets', file: '/workspace/impl_diffs/datasets_v2.md', label: 'apply datasets Layer-A' },
]

phase('Apply')
const results = await parallel(specs.map(s => () =>
  agent(`${COMMON}\n\nYOUR SPEC: ${s.file}\nApply it, build+gate per its gate_commands, commit your worktree, and report the sha + result.`,
    { label: s.label, phase: 'Apply', schema: SCHEMA, isolation: 'worktree' })
    .then(r => ({ name: s.name, ...(r || { spec: s.name, applied_clean: false, gate_pass: false, commit_sha: '', branch: '', deviations: ['agent died'], gate_output_tail: '', diffstat: '', summary: 'null' }) }))))
log(`Applied: ${results.filter(r => r && r.gate_pass).length}/${specs.length} gates passed; shas: ${results.map(r => r && r.commit_sha ? r.commit_sha.slice(0,8) : 'none').join(', ')}`)
return results
