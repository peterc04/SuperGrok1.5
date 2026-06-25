export const meta = {
  name: 'supergrok-apply-flagships',
  description: 'Parallel WORKTREE-ISOLATED apply+build+gate of the ViT-flagship, Mamba-flagship, and datasets specs — each agent applies its spec to an isolated copy, runs its gate, reports pass + exact deviations',
  phases: [{ title: 'Apply', detail: '3 isolated agents apply+build+gate one spec each' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['spec', 'applied_clean', 'gate_pass', 'deviations', 'gate_output_tail', 'diffstat', 'summary'],
  properties: {
    spec: { type: 'string' },
    applied_clean: { type: 'boolean', description: 'true if every OLD block in the spec matched the live file and applied' },
    gate_pass: { type: 'boolean', description: 'true if ALL the spec gate_commands passed' },
    deviations: { type: 'array', items: { type: 'string' }, description: 'any OLD-block mismatch, fix you had to make, or gate adjustment — empty if none' },
    gate_output_tail: { type: 'string', description: 'last ~15 lines of the decisive gate command output' },
    diffstat: { type: 'string', description: 'git diff --stat of the changes you made' },
    summary: { type: 'string' },
  },
}
const COMMON = `You operate in an ISOLATED GIT WORKTREE copy of /workspace/SuperGrok1.5 — you may freely Edit/Write/Bash/build in it; your changes do NOT touch other agents. Your job: APPLY one apply-ready spec, then BUILD+GATE it, and report the result precisely.
PROCEDURE: (1) Read the spec file IN FULL. (2) For EACH OLD/NEW block, apply the edit with an exact-match Edit (the OLD blocks were copied verbatim from the live file — if one does NOT match, the live file drifted: find the true current text, apply the equivalent change, and RECORD it in deviations). (3) For NEW files, write them in full. (4) Run the spec's gate_commands EXACTLY. (5) If a gate fails, READ the error, fix the real cause if it is a mechanical apply error (mismatched brace, missing include), re-run, and record what you did in deviations — do NOT loosen a correctness gate to force a pass. (6) Report the schema fields honestly: applied_clean (did every OLD match), gate_pass (did the gates pass), deviations, the decisive gate output tail, and git diff --stat.
PROJECT RULES: production path = L3-TC persistent wgmma megakernel; the flagship work MUST keep the existing (non-flagship) layout BYTE-IDENTICAL (a diff of the regenerated current-layout header vs the committed one must be empty) and keep the current-L parity/determinism gate green; new layouts are emitted to a separate header; gfx942/tpu untouched. Build env: the repo root has scripts/compile_to_object.sh; CUTLASS is at third_party/cutlass/include. Be exhaustive and precise.`

const specs = [
  { name: 'vit_flagship', file: '/workspace/impl_diffs/vit_flagship.md', label: 'apply ViT flagship' },
  { name: 'mamba_flagship', file: '/workspace/impl_diffs/mamba_flagship.md', label: 'apply Mamba flagship' },
  { name: 'datasets', file: '/workspace/impl_diffs/datasets_v2.md', label: 'apply datasets Layer-A' },
]

phase('Apply')
const results = await parallel(specs.map(s => () =>
  agent(`${COMMON}\n\nYOUR SPEC: ${s.file}\nApply it, build+gate per its gate_commands, and report. Be honest about any deviation or gate failure — a truthful FAIL is more useful than a forced pass.`,
    { label: s.label, phase: 'Apply', schema: SCHEMA, isolation: 'worktree' })
    .then(r => ({ name: s.name, ...(r || { spec: s.name, applied_clean: false, gate_pass: false, deviations: ['agent died'], gate_output_tail: '', diffstat: '', summary: 'null' }) }))))
log(`Applied: ${results.filter(r => r && r.gate_pass).length}/${specs.length} gates passed`)
return results
