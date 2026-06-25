export const meta = {
  name: 'supergrok-mamba-rerun',
  description: 'Re-run the Mamba-flagship write-agent in an isolated worktree (its first attempt died unchanged); apply + build + gate + commit to its branch',
  phases: [{ title: 'Apply', detail: 'one isolated write-agent applies the mamba_flagship spec' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['spec', 'applied_clean', 'gate_pass', 'commit_sha', 'branch', 'deviations', 'gate_output_tail', 'diffstat', 'summary'],
  properties: {
    spec: { type: 'string' }, applied_clean: { type: 'boolean' }, gate_pass: { type: 'boolean' },
    commit_sha: { type: 'string' }, branch: { type: 'string' },
    deviations: { type: 'array', items: { type: 'string' } },
    gate_output_tail: { type: 'string' }, diffstat: { type: 'string' }, summary: { type: 'string' },
  },
}
const PROMPT = [
  'You operate in an ISOLATED GIT WORKTREE copy of /workspace/SuperGrok1.5 (branched from the full current session state — includes the CuTe atoms, flagship decoder + ViT, and the TP foundation). Freely Edit/Write/Bash/build.',
  'YOUR SPEC: /workspace/impl_diffs/mamba_flagship.md — read it IN FULL, then apply EVERY OLD/NEW block with exact-match Edits (if an OLD block drifted, find the true current text, apply the equivalent change, RECORD in deviations); write NEW files in full.',
  'Then run the spec gate_commands EXACTLY. NOTE the Mamba analogue of the scalar gate: the Mamba megakernel has its own SG_MB_SCALAR_MEGAKERNEL (or equivalent) — mirror how the decoder used -DSG_DEC_SCALAR_MEGAKERNEL=0 and ViT used -DSG_VIT_SCALAR_MEGAKERNEL=0 + the *_BENCH_LAYOUT flag to elide the legacy scalar/staged-opt path at flagship width; GREP for the exact Mamba flag name in fused_mamba_megakernel.cuh and use it in the flagship compile gate. If a gate fails on a mechanical apply error, fix the real cause and re-run; NEVER loosen a correctness gate.',
  'When done, commit ALL changes: git add -A && git commit -m "track: mamba_flagship applied + gated", then report git rev-parse HEAD + git branch --show-current + the schema fields HONESTLY.',
  'RULES: keep the existing (non-flagship) Mamba layout BYTE-IDENTICAL (regen + diff vs committed = empty); keep the current-L parity/determinism gate green; flagship layout to a SEPARATE header; gfx942/tpu untouched. scripts/compile_to_object.sh at repo root; CUTLASS at third_party/cutlass/include.',
].join('\n')

phase('Apply')
const r = await agent(PROMPT, { label: 'apply Mamba flagship (re-run)', phase: 'Apply', schema: SCHEMA, isolation: 'worktree' })
log('Mamba re-run: gate_pass=' + (r && r.gate_pass) + ' sha=' + (r && r.commit_sha ? r.commit_sha.slice(0,8) : 'none'))
return r || { spec: 'mamba_flagship', applied_clean: false, gate_pass: false, commit_sha: '', branch: '', deviations: ['agent died again'], gate_output_tail: '', diffstat: '', summary: 'null' }
