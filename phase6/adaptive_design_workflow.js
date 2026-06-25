export const meta = {
  name: 'supergrok-adaptive-design',
  description: 'Two parallel READ-ONLY design specs: 3D-5D adaptive parallelism with Expert Parallelism as the 5th axis (degree inferred from front-end params), and size/config-adaptive kernel specialization (CTA-tiling) via if-constexpr config templating',
  phases: [{ title: 'Design', detail: '2 agents emit apply-ready specs to /workspace/impl_diffs/' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'spec_path', 'files_changed', 'gate_commands', 'confidence', 'risks', 'summary'],
  properties: {
    area: { type: 'string' }, spec_path: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    gate_commands: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string' }, risks: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}
const COMMON = [
  'You are READ-ONLY on /workspace/SuperGrok1.5 (Read/Grep/Glob; do NOT edit/build/run). Write /workspace/impl_diffs/<AREA>.md as APPLY-READY edits (VERBATIM OLD copied from the live file + NEW replacement; new files in full).',
  'VERIFIED CONTEXT (this session): production path = the L3-TC PERSISTENT wgmma megakernel (1 CTA/SM, hand GridBarrier, fwd-bwd-opt fused in ONE __global__ launch). It is now TEMPLATED on parallelism config: the decoder megakernel signature is template<OptId Opt, class Par=SingleGPU> with a trailing CommCtx comm arg (applied + gated this session, 19/19). ParConfig with axes DP,TP,PP,SP,Z lives in csrc/fused/sm_90/parallel_config.cuh with par::SingleGPU; CommCtx carries the comm wiring; tp_transport.cuh has the in-kernel device-NVSHMEM all-reduce (NVSHMEM 3.7.0 installed). The TP reduce is gated by if constexpr (Par::kTPComm) so the SingleGPU instantiation is byte-identical. THIS if-constexpr-on-config gating is the canonical specialization mechanism.',
  'USER DESIGN DIRECTIVE the spec MUST honor: the megakernel is templated on its deployment config and if constexpr folds in EXACTLY the machinery that config needs - distributed builds the all-reduce/parallelism, single-GPU builds none of it (byte-identical), large size builds CTA-tiling, small size does not. Parallelism DEGREE is auto-selected 3D-5D from FRONT-END params: base 3D = DP x TP x PP; +SP (4th) if a sequence model; +EP (5th, EXPERT PARALLELISM) if the model uses experts (MoE). Every NEW config branch MUST be byte-identical when its config is OFF (the SingleGPU / no-EP / small-size default folds to todays kernel). gfx942/tpu untouched. Read the cited files plus /workspace/impl_diffs/tp_kernel.md, dist_step.md, run_harness.md IN FULL first.',
].join('\n')

const agents = [
  { name: 'adaptive_parallelism', label: '3D-5D adaptive + Expert Parallelism (EP)', prompt: COMMON + '\n\n' + [
    'AREA: csrc/fused/sm_90/parallel_config.cuh + grokking_optimizers/parallel/* + the front-end model config (grokking_race_v2.py MODEL_SCALES_BY_MODEL and the model definitions). GOAL: design (apply-ready) the upgrade from a fixed 4D mesh to ADAPTIVE 3D-5D.',
    '(1) Add EP (Expert Parallelism) as the 5th axis to ParConfig (add an EP degree + a kEPComm gate, mirroring exactly how TP is represented: kTPComm + CommCtx tp_* fields + a team). EP engages ONLY when the model declares experts (MoE); for a non-MoE model kEPComm is false and ALL EP machinery folds away (byte-identical, like SingleGPU). Spell out how the in-kernel expert all-to-all/dispatch would gate under if constexpr (Par::kEPComm), and SCOPE the kernel-side expert routing honestly (the ParConfig + CommCtx + team + gate points now; the expert-dispatch megakernel body as a flagged follow-on).',
    '(2) Design the FRONT-END to ParConfig INFERENCE: a Python function (in grokking_optimizers/parallel/ or distributed.py) that reads the model config and picks the degree - base 3D (DP x TP x PP); +SP iff the model is a sequence model (decoder / ViT-patches / Mamba are sequence, so SP-eligible); +EP iff the model has experts (num_experts > 1 or an MoE flag). It returns the ParConfig + per-axis degrees, honoring the device count (8 H100s) and the run_harness.md mesh math, and maps to which template instantiation the launcher dispatches.',
    '(3) BYTE-IDENTICAL when OFF: a 3D or 4D model never instantiates the EP branch; the current 4D flagship path is unchanged.',
    'gate_commands: ["python -c \\"import grokking_optimizers.parallel\\"", "python -m pytest tests/ -k \\"parallel or config\\" -q", "bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu"]. Spec -> /workspace/impl_diffs/adaptive_parallelism.md.',
  ].join('\n') },

  { name: 'size_adaptive', label: 'size/config-adaptive kernel specialization (CTA-tiling)', prompt: COMMON + '\n\n' + [
    'AREA: grokking_optimizers/megakernel_codegen.py + the megakernel tile/occupancy config (csrc/fused/sm_90/*.cuh tile knobs such as SG_TUNED_TILE_N, the ring stages, the CTA/occupancy config) + the autotuner (tuning/ and compile.py autotune). GOAL: design (apply-ready) SIZE/CONFIG-ADAPTIVE kernel specialization - the self-designing megakernel made concrete.',
    '(1) The codegen/launcher selects knobs (notably CTA-TILING - multiple CTAs per output tile / clusters / occupancy>1, vs the current 1-CTA/SM persistent shape) by WORKLOAD SIZE: CTA-tiling ON for LARGE configs (more SMs to fill - this is bottleneck LEVER 2, the measured 20% grid-barrier idle at d=2048 = cross-CTA load imbalance), OFF for SMALL configs (the persistent 1-CTA/SM shape wins - overhead dominates). Find where the current build picks tile/occupancy config; design a size-thresholded selector (a function of d, layers, T/B, number-of-SMs) that emits the right knobs, the SAME way distributed-vs-single gates the all-reduce.',
    '(2) Config-gated + byte-identical at the current default: the existing flagship / d=2048 / d=128 builds KEEP their current knobs (so todays gates stay green); only new size-tiers select new knobs. Honor the user framing: if distributed the kernel builds all-reduce/parallelism machinery, if single-GPU it does not - CTA-tiling is the same idea by size. So the selector COMPOSES with ParConfig (distributed+large -> CTA-tiling + all-reduce; single+small -> neither).',
    '(3) Be honest about scope: actually implementing a CTA-tiled (occupancy>1) megakernel variant is a real kernel change (the GridBarrier assumes 1 CTA/SM). Deliver the SELECTOR + config plumbing + the codegen knob surface now, and a PRECISE scoping of the CTA-tiled kernel variant (what changes: the barrier, tile ownership, smem) as the follow-on.',
    'gate_commands: ["python -m grokking_optimizers.megakernel_codegen --decoder-layout > /tmp/d.cuh && diff /tmp/d.cuh csrc/fused/sm_90/decoder_layout.cuh", "python -c \\"import grokking_optimizers.megakernel_codegen\\""]. Spec -> /workspace/impl_diffs/size_adaptive.md.',
  ].join('\n') },
]

phase('Design')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'Design', schema: SCHEMA })
    .then(r => ({ name: a.name, ...(r || { area: a.name, spec_path: '', files_changed: [], gate_commands: [], confidence: 'AGENT NULL', risks: ['agent died'], summary: 'null' }) }))))
log('Adaptive design specs: ' + results.filter(r => r && r.spec_path).length + '/' + agents.length)
return results
