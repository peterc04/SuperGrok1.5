export const meta = {
  name: 'supergrok-resource-fit',
  description: 'Design the ROBUST resource-fit planner (workload x hardware -> parallelism degree + memory strategy + kernel knobs), plus a survey/scoping of the memory-strategy machinery (offload/recompute/streaming) needed for large-model-on-few-GPUs (e.g. 10B on 1 GPU)',
  phases: [{ title: 'Plan', detail: '2 agents emit apply-ready specs to /workspace/impl_diffs/' }],
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
  'You are READ-ONLY on /workspace/SuperGrok1.5 (Read/Grep/Glob; do NOT edit/build/run). Write /workspace/impl_diffs/<AREA>.md as APPLY-READY edits (VERBATIM OLD from the live file + NEW; new files in full).',
  'CONTEXT: SuperGrok2 is a portable, SELF-ADAPTING training stack over a persistent fused L3-TC wgmma megakernel (1 CTA/SM, fused fwd-bwd-opt). It is templated on a parallelism config (ParConfig DP,TP,PP,SP,Z + par::SingleGPU; CommCtx) and the in-kernel device-NVSHMEM all-reduce is gated by if constexpr (Par::kTPComm) so SingleGPU is byte-identical. NVSHMEM 3.7.0 is installed. The flagship 1.5B decoder runs single-GPU (acts live in an HBM workspace dec_tc_workspace_floats; AdamW state = 3x params).',
  'USER DIRECTIVE the spec MUST honor: strategy decisions must NOT key on GPU count. A SINGLE GPU can host a 10B+ model even for TRAINING, which needs heavy machinery (optimizer/activation OFFLOAD to host, RECOMPUTE / gradient checkpointing, LAYER STREAMING, possibly CTA-tiling). So the stack is a ROBUST PLANNER: from (model size/shape + hardware: #GPUs, HBM/GPU, host RAM, interconnect) it decides the FULL execution config - parallelism degree (3D-5D), memory strategy (in-HBM | ZeRO-offload | activation-recompute | layer-streaming | host-offload), and kernel knobs (CTA-tiling, ring depth, occupancy). 10M-on-1GPU -> trivial; 10B-on-1GPU -> offload+recompute+streaming; 1.5B-on-8GPU -> 4D+ZeRO-3. Driver = memory-fit + compute-shape vs hardware, never a naive GPU-count switch. Same if-constexpr/config-templating emits exactly the chosen machinery (byte-identical when a strategy is OFF). gfx942/tpu untouched. Read /workspace/impl_diffs/{run_harness,dist_step,adaptive_parallelism,size_adaptive}.md if present, plus the cited code, IN FULL first.',
].join('\n')

const agents = [
  { name: 'resource_fit_planner', label: 'robust workload x hardware planner', prompt: COMMON + '\n\n' + [
    'AREA: a NEW Python planner module (e.g. grokking_optimizers/parallel/resource_planner.py) + the hook into grokking_optimizers/distributed.py / the launcher dispatch. GOAL: design (apply-ready, give the new file in full) a ROBUST PLANNER function plan_execution(model_cfg, hw_cfg) -> ExecutionPlan that, from the front-end model config (d, layers, seq, vocab, num_experts, is_sequence) and the hardware (num_gpus, hbm_bytes_per_gpu, host_ram_bytes, nvlink/pcie), computes: (a) the parallelism mesh (DP,TP,PP,SP,EP degrees) - reuse the run_harness.md mesh math + the adaptive_parallelism.md 3D-5D inference; (b) the MEMORY STRATEGY flags (need_zero_offload, need_activation_recompute, need_layer_streaming, need_param_offload) from a MEMORY-FIT estimate (params + opt-state + activations + the dec_tc workspace incl. the staged-opt SG2 scratch, vs hbm_bytes_per_gpu); (c) the kernel knob tier (cta_tiling on/off, ring depth) by compute size. Give the EXACT memory-fit arithmetic (mirror the live dec_tc_*_floats formulas + the SG2 ~91*Nmax/TP fact). Show how ExecutionPlan maps to the kernel template instantiation + the compile flags. Worked examples in the spec: 10M/1GPU, 1.5B/8GPU, 10B/1GPU, 10B/8GPU. gate_commands: ["python -c \\"import grokking_optimizers.parallel\\"", "python -m pytest tests/ -k \\"plan or resource or parallel\\" -q"]. Spec -> /workspace/impl_diffs/resource_fit_planner.md.',
  ].join('\n') },

  { name: 'memory_strategy', label: 'offload/recompute/streaming machinery survey + scope', prompt: COMMON + '\n\n' + [
    'AREA: survey the EXISTING memory machinery + scope what is needed for large-model-on-few-GPUs. Read zero3.py (does it support offload to host?), the dec_tc workspace carve (fused_decoder_megakernel.cuh dec_tc_*_floats - acts/wbf/dW-transpose/staged-opt all in HBM), any gradient-checkpoint/recompute path, the launcher allocation (mega_decoder_real_adamw_tc_launcher.cu cudaMalloc workspace), and grokking_optimizers/distributed.py. GOAL: produce a spec that (1) INVENTORIES what exists today (ZeRO-3 shards params+state across DP; acts are recomputed? or stored full? - determine from the code; is there any host-offload or activation-recompute?), and (2) SCOPES the additions needed so a 10B model TRAINS on 1 GPU: optimizer-state host-offload (ZeRO-offload style), activation RECOMPUTE (checkpoint the layer boundaries; the persistent megakernel already streams layers - relate to that), and LAYER STREAMING of weights from host. Be honest and precise: which are tractable config flags vs which are real kernel/launcher changes. Give whatever is APPLY-READY now (e.g. a recompute or offload flag in the planner/launcher) as exact edits, and a precise scoping of the rest. Tie each to the if-constexpr/config-templating mechanism (byte-identical when OFF). gate_commands: ["python -c \\"import grokking_optimizers.parallel.zero3\\"", "bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1"]. Spec -> /workspace/impl_diffs/memory_strategy.md.',
  ].join('\n') },
]

phase('Plan')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'Plan', schema: SCHEMA })
    .then(r => ({ name: a.name, ...(r || { area: a.name, spec_path: '', files_changed: [], gate_commands: [], confidence: 'AGENT NULL', risks: ['agent died'], summary: 'null' }) }))))
log('Resource-fit specs: ' + results.filter(r => r && r.spec_path).length + '/' + agents.length)
return results
