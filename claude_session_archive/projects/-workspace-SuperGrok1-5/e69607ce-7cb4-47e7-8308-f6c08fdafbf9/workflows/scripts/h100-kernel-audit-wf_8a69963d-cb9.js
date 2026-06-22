export const meta = {
  name: 'h100-kernel-audit',
  description: 'Parallel static audit of sm_90 kernels for on-device correctness bugs + SG2 shape-bug trace + repo cleanup plan',
  phases: [
    { title: 'Audit', detail: 'one agent per sm_90 kernel: dtype/shape/count contract + math vs canonical' },
    { title: 'Synthesize', detail: 'dedupe + prioritize findings into a verification worklist' },
  ],
}

// Each entry: kernel header + the canonical optimizer math (if any) to check against.
const KERNELS = [
  { name: 'adamw',        canonical: 'csrc/algorithms/adamw.h' },
  { name: 'lion',         canonical: 'csrc/algorithms/lion.h' },
  { name: 'grokfast',     canonical: 'csrc/algorithms/grokfast.h' },
  { name: 'grokadamw',    canonical: 'csrc/algorithms/grokadamw.h' },
  { name: 'looksam',      canonical: 'csrc/algorithms/looksam.h' },
  { name: 'prodigy',      canonical: 'csrc/algorithms/prodigy.h' },
  { name: 'neuralgrok',   canonical: 'csrc/algorithms/neuralgrok.h' },
  { name: 'muon',         canonical: 'csrc/algorithms/muon.h' },
  { name: 'supergrok11',  canonical: 'csrc/algorithms/supergrok11.h' },
  { name: 'supergrok15',  canonical: 'csrc/algorithms/supergrok15.h' },
  { name: 'attention',    canonical: '(none — model kernel)' },
  { name: 'mamba3',       canonical: '(none — model kernel)' },
  { name: 'transformer_decoder', canonical: '(none — model kernel)' },
  { name: 'vit',          canonical: '(none — model kernel)' },
]

const PATTERNS = `
KNOWN BUG CLASSES already confirmed on this H100 (look for MORE of the same):
 (A) Mixed-dtype paired tensors into a single-template kernel: the kernel
     dispatches on ONE tensor's scalar_type() (e.g. weight) then reads a PAIRED
     tensor (e.g. its bias) with data_ptr<SAME_T>(). If Python can pass them in
     different dtypes (weight bf16 via projection_precision='bf16', bias left
     fp32) -> "expected BFloat16 but found Float" at runtime. (Confirmed: SG2
     input_proj_W bf16 vs input_proj_b fp32.)
 (B) Wrong count/loop-bound: a kernel loops 't < n_tasks' or indexes arr[t]
     where n_tasks / the size were set to the ELEMENT count instead of the
     TASK/entry count, walking a small array out of bounds. (Confirmed: fused_step
     n_tasks=n.)
 (C) at:: host ops mixing bf16 weights with fp32 activations without a .to()
     coercion (e.g. a.matmul(b) where a is bf16 and b fp32).
 (D) reshape/view({-1, K}) where the tensor numel may not be divisible by K
     (dimension-wiring mismatch between the passed config int and the actual
     tensor shape). (Confirmed: SG2 meta-net '[-1,44]' vs numel 192.)
 (E) Missing __syncthreads()/__syncwarp() between dependent shared-memory phases,
     or a grid/block index that can exceed the allocated buffer.
 (F) Numeric divergence from the canonical algorithm header (wrong formula,
     swapped operands, missing bias-correction, wrong beta, etc.).
`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['kernel', 'findings'],
  properties: {
    kernel: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'bug_class', 'file_line', 'description', 'evidence', 'why_fails_or_wrong', 'suggested_fix', 'confidence'],
        properties: {
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          bug_class: { type: 'string', enum: ['A-dtype-pair', 'B-count', 'C-host-mixdtype', 'D-reshape', 'E-sync-oob', 'F-math', 'other'] },
          file_line: { type: 'string', description: 'path:line' },
          description: { type: 'string' },
          evidence: { type: 'string', description: 'the exact code snippet + why it is reachable' },
          why_fails_or_wrong: { type: 'string' },
          suggested_fix: { type: 'string' },
          confidence: { type: 'string', enum: ['certain', 'likely', 'speculative'] },
        },
      },
    },
  },
}

phase('Audit')

const auditPromise = parallel(KERNELS.map((k) => () =>
  agent(
    `You are auditing ONE CUDA kernel header on a real NVIDIA H100 (sm_90a) build of the SuperGrok1.5 repo (cwd /workspace/SuperGrok1.5). The extension already builds, links, imports, and 10/11 optimizers run; you are hunting LATENT on-device correctness bugs (especially on code paths the grokking race does not exercise, and numeric errors).

Files to read:
 - grokking_optimizers/kernels/sm_90/${k.name}_sm90.cuh   (THE kernel)
 - its canonical optimizer math: ${k.canonical}            (compare the device math against this, if applicable)
 - the binding that calls it: grep csrc/bindings/bindings.cpp for the relevant launch_/op name to see the Python->C++ dtype/shape contract.
 - Optionally the Python optimizer grokking_optimizers/optimizers/${k.name}.py to see what dtypes/shapes it actually passes.

${PATTERNS}

Method: read the kernel fully. For every kernel that dispatches on a tensor dtype, check that ALL tensors it reads with data_ptr<T>() under that branch are guaranteed that dtype. For every reshape/index, check the dimension is derivable and divides. For optimizer kernels, verify the per-element update matches the canonical header formula EXACTLY (operand order, bias-correction, eps placement). 

Report ONLY findings you can back with a specific code snippet and a concrete reachability argument. Prefer precision over recall: a 'certain' high-severity finding is worth far more than ten speculative ones. If the kernel looks correct, return an empty findings array — that is a valid, valuable result. Return the structured object.`,
    { label: `audit:${k.name}`, phase: 'Audit', schema: FINDINGS_SCHEMA, agentType: 'Explore' }
  )
))

// Dedicated SG2 shape-bug tracer (independent, runs concurrently with the audit).
const sg2Promise = agent(
  `Trace and FIX a specific runtime bug on a real H100 in /workspace/SuperGrok1.5. Running the SuperGrok2 optimizer in grokking_race_v2.py crashes with:
   RuntimeError: shape '[-1, 44]' is invalid for input of size 192
raised inside grokking_optimizers/optimizers/supergrok2.py line ~1647 (the call _ops.supergrok2_prepare_and_batched_step(...)), i.e. INSIDE the C++ kernel launch_csa_hca_batched_step in grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh.

Already fixed (do NOT re-report): input_proj_W bf16 vs input_proj_b fp32 dtype mismatch. This shape bug is the NEXT failure after that fix.

Candidate reshape sites in supergrok2_sm90.cuh: line ~1453 peer_query_Ws.reshape({-1, d_model}); ~1458 prod_keys_A.reshape({-1, half}); ~1465 prod_keys_B.reshape({-1, Q-half}); ~1426-1429 expert_W*.reshape({num_experts,-1}). The meta-net config (race): sg2_meta_d_model=8, sg2_num_peer_experts=1024 (pk_dim=sqrt=32), num_peer_heads default 4, sg2_expert_hidden=4, sg2_recurrent_dim=8. '44' is not a clean config value — find which reshape's K becomes 44 and which tensor has numel 192, then explain the dimension-wiring mismatch (what d_model/half/Q the kernel computed vs the actual tensor shape built by the Python CSAHCAMetaNet in supergrok2.py).

Read supergrok2_sm90.cuh around those lines, the CSAHCAMetaNet construction + the peer_query_Ws/product_keys parameter shapes in grokking_optimizers/optimizers/supergrok2.py, and the d_model/pk_dim/num_heads args passed into supergrok2_prepare_and_batched_step (bindings.cpp ~2118 + the python call ~1647-1694). Identify the EXACT root cause (file:line) and the minimal correct fix. Do NOT edit files — report the diagnosis + proposed fix precisely so the main agent can apply and GPU-verify it.`,
  { label: 'sg2-shape-trace', phase: 'Audit', agentType: 'Explore' }
)

// Repo cleanup planner (non-destructive: produce a PLAN, classify, never delete).
const cleanupPromise = agent(
  `Produce a REPO CLEANUP PLAN for /workspace/SuperGrok1.5 to make it production/portfolio-ready. This is the transition phase. Be NON-DESTRUCTIVE: classify and propose, do not delete anything.

Inventory the top-level and notable files. Classify each into: KEEP (core: README, LICENSE, setup.py, build.sh, csrc/, grokking_optimizers/, tests/, scripts/, grokking_race_v2.py, third_party/), ARCHIVE (stale process docs: BUILD_REPORT.md, FIX2_REPORT.md, MANDATE_REPORT.md, MIGRATION_REPORT.md, PHASE3..7_REPORT.md, RESTRUCTURE_PLAN.md, HARDWARE_VALIDATION.md, MANIFEST.in — judge each), or UPDATE (docs that must change to reflect that the code now builds+runs+groks on a real H100: README.md status banner, HARDWARE_VALIDATION.md 🟡->✅).

Specifically read README.md and note its 'No accelerator is present here, all runtime claims are 🟡' banner — that is now FALSE (the extension builds, imports, runs, and the grokking race groks on H100). List the exact lines/sections that must be rewritten.

Also scan for obvious dead weight: empty files, __pycache__ checked in, duplicate/orphaned files, TODO/FIXME/XXX clusters that signal unfinished work. Return a concrete, file-by-file plan with a recommended 'archive/' directory layout and the specific README/HARDWARE_VALIDATION edits needed. Do NOT edit anything.`,
  { label: 'cleanup-plan', phase: 'Audit', agentType: 'Explore' }
)

const [auditResults, sg2Trace, cleanupPlan] = await Promise.all([auditPromise, sg2Promise, cleanupPromise])

phase('Synthesize')

const allFindings = (auditResults || []).filter(Boolean).flatMap(r => (r.findings || []).map(f => ({ ...f, kernel: r.kernel })))
const high = allFindings.filter(f => f.severity === 'high')
const certain = allFindings.filter(f => f.confidence === 'certain')

const synthesis = await agent(
  `You are the synthesis stage of an H100 kernel audit. Below are raw findings (JSON) from ${KERNELS.length} per-kernel auditors, plus a SuperGrok2 shape-bug trace and a repo cleanup plan. Produce a SINGLE prioritized verification worklist for the main engineer (who has a real H100 and will GPU-verify each item before fixing).

Dedupe overlapping findings. Drop the speculative/low-confidence noise. Rank by (severity, confidence, blast-radius). For each surviving item give: file:line, one-line issue, the concrete fix, and the exact command/test to VERIFY it on the H100 (e.g. a tiny python repro calling the op, or a recompile+import). Then give a 5-line executive summary: how many high/certain bugs, which optimizers/models are affected, and whether any are on the grokking-race critical path.

RAW AUDIT FINDINGS:
${JSON.stringify(allFindings, null, 1).slice(0, 18000)}

SG2 SHAPE TRACE:
${(sg2Trace || 'none').slice(0, 6000)}

CLEANUP PLAN:
${(cleanupPlan || 'none').slice(0, 6000)}`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return {
  counts: { total: allFindings.length, high: high.length, certain: certain.length },
  high_certain: allFindings.filter(f => f.severity === 'high' && f.confidence !== 'speculative'),
  worklist: synthesis,
  sg2_trace: sg2Trace,
  cleanup_plan: cleanupPlan,
}
