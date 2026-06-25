export const meta = {
  name: 'supergrok-deadcode-analysis',
  description: 'READ-ONLY deep dead-code analysis: confirm the removable build/scan/session artifacts, reachability-check the true source for provably-dead code, and produce the apply-ready removal spec + the post-cleanup LOC/language report (deliverable #2)',
  phases: [{ title: 'Analyze', detail: '2 read-only agents emit the removal spec + LOC report' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'spec_path', 'removable_lines', 'confidence', 'risks', 'summary'],
  properties: {
    area: { type: 'string' }, spec_path: { type: 'string' },
    removable_lines: { type: 'integer' },
    confidence: { type: 'string' }, risks: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}
const COMMON = 'READ-ONLY on /workspace/SuperGrok1.5 (Read/Grep/Glob; do NOT edit/build/run). Write your spec to /workspace/impl_diffs/<AREA>.md as an apply-ready removal list (exact paths to delete + exact in-file OLD snippets to remove, VERBATIM). PROJECT: the production path is the L3-TC persistent wgmma megakernel; the 33 _tc cells + the prebuilt _ops + the 3 model layouts + the gfx942/tpu trees must STAY; only PROVABLY-dead (reachability-checked) code/artifacts get removed; the math-drift guard + the parity/determinism gates must stay green. Be precise + conservative — a wrong removal breaks the build.'

const agents = [
  { name: 'deadcode_artifacts', label: 'removable artifacts audit + LOC report', prompt: COMMON + '\n\n' + [
    'AREA: the committed NON-SOURCE artifacts + the final LOC report. ALREADY IDENTIFIED (verify each is unreferenced by any .py/.cu/.cuh/.cpp/.sh/.md/setup/CMake/pyproject — grep, and confirm no test/build imports it): (a) _dectc_codegen/ (64 files, ~7.95M lines of nvcc .cudafe1.cpp/.ptx intermediate dumps — referenced ONLY by _scan_prep.sh); (b) _scan/ (scan output txt dumps ~60K lines); (c) _scan_prep.sh + any _scan* helper; (d) claude_session_archive/ (committed Claude session tool-result dumps — confirm nothing imports/reads it). For EACH: confirm removable (zero source/build/test refs) or flag why not. ALSO decide results/ (benchmark result JSONs — KEEP if they are reference results, flag for the user). Produce: the exact rm list + a recomputed LOC/language report (lines + files + %) for (i) the WHOLE repo as-is and (ii) AFTER removing the confirmed artifacts (the TRUE source). Use a cloc-style by-extension count excluding third_party/.git/pycache. spec -> /workspace/impl_diffs/deadcode_artifacts.md. removable_lines = total artifact lines confirmed removable.',
  ].join('\n') },

  { name: 'deadcode_source', label: 'source reachability dead-code (line-by-line)', prompt: COMMON + '\n\n' + [
    'AREA: the TRUE SOURCE (grokking_optimizers/*.py, csrc/**, tests/**, tuning/**, root *.py) — reachability-check for PROVABLY-dead code. Known candidates to verify+spec (each: confirm zero reachable callers before listing): (1) the obsolete tc_dump_outproj_operands path on the Mamba scalar TC (the test that hit RuntimeError — the dump fn is obsolete); (2) any in-file dead model classes (e.g. a duplicate MambaModel/SelectiveSSMLayer in grokking_race_v2.py if the real model is elsewhere) + _maybe_wrap_cuda_graph no-ops (the datasets spec flagged these); (3) the archive-gap dead generated cells (mega_<model>_<opt>.cu that #include the removed fused_megakernel.cuh / launch_fused_megakernel — they do NOT compile, confirmed earlier; the REAL path is the _tc cells) + any launch_<opt>.cu/models/*.cu shims referenced by verify_all/profile but absent-from-build; (4) unreferenced helper functions / dead #if branches / commented-out blocks across the kernels + Python. For EACH, give the exact file + verbatim snippet (or whole-file path) + the reachability evidence (grep showing no caller). Be CONSERVATIVE: if a symbol is reachable via opt_id dispatch / pybind / a test, KEEP it. spec -> /workspace/impl_diffs/deadcode_source.md. removable_lines = your conservative source-dead total.',
  ].join('\n') },
]

phase('Analyze')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'Analyze', schema: SCHEMA })
    .then(r => ({ name: a.name, ...(r || { area: a.name, spec_path: '', removable_lines: 0, confidence: 'AGENT NULL', risks: ['agent died'], summary: 'null' }) }))))
log('Dead-code analysis: ' + results.map(r => (r && r.removable_lines || 0)).reduce((a,b)=>a+b,0) + ' removable lines identified')
return results
