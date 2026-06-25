export const meta = {
  name: 'supergrok-apply-remaining',
  description: 'Three parallel WORKTREE apply-agents on DISJOINT file sets: TP remainder (C.3 loop + D reduce-points + E launcher heap), memory-strategy seam (mem_config + planner), and EP + size-adaptive seams; each applies+builds+gates+commits, reports sha to cherry-pick',
  phases: [{ title: 'Apply', detail: '3 isolated write-agents apply disjoint specs' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'applied_clean', 'gate_pass', 'commit_sha', 'deviations', 'gate_output_tail', 'diffstat', 'summary'],
  properties: {
    area: { type: 'string' }, applied_clean: { type: 'boolean' }, gate_pass: { type: 'boolean' },
    commit_sha: { type: 'string', description: 'full sha of your worktree commit (git rev-parse HEAD), empty if none' },
    deviations: { type: 'array', items: { type: 'string' } },
    gate_output_tail: { type: 'string' }, diffstat: { type: 'string' }, summary: { type: 'string' },
  },
}
const COMMON = [
  'You operate in an ISOLATED GIT WORKTREE copy of /workspace/SuperGrok1.5, branched from the FULL current state (3 flagships decoder/ViT/Mamba + datasets + CuTe atoms + TP FOUNDATION already present: tp_transport.cuh has the device-NVSHMEM all-reduce, parallel_config.cuh has CommCtx, fused_decoder_megakernel.cuh is template<OptId Opt, class Par=SingleGPU>). Freely Edit/Write/Bash/build.',
  'PROCEDURE: (1) Read your spec(s) IN FULL. (2) Apply EVERY OLD/NEW block with exact-match Edits; if an OLD block drifted, find the true current text and apply the equivalent change, RECORD in deviations. Write NEW files in full. (3) Run the spec gate_commands EXACTLY. For any CUDA flagship/TC build, mirror the scalar-gate convention: decoder uses -DSG_DEC_SCALAR_MEGAKERNEL=0, ViT -DSG_VIT_SCALAR_MEGAKERNEL=0, Mamba its own SG_MB_*; grep the right flag. (4) If a gate fails on a mechanical apply error, fix the real cause and re-run; NEVER loosen a correctness gate. (5) COMMIT: restore any dirty *.pyc first (git checkout -- on pyc paths), then git add -A and git commit -m "track: <area>"; report git rev-parse HEAD as commit_sha. (6) Report schema fields HONESTLY — a truthful gate_pass=false with the error tail beats a forced pass.',
  'RULES: keep existing non-flagship layouts BYTE-IDENTICAL; keep the SingleGPU / default path byte-identical (every new config branch if-constexpr/#if-gated OFF); keep current gates green; gfx942/tpu untouched. scripts/compile_to_object.sh at repo root; CUTLASS at third_party/cutlass/include; NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem (compile NVSHMEM TUs with -DSG_HAS_NVSHMEM=1 -rdc=true -I$NVSHMEM_HOME/include).',
].join('\n')

const agents = [
  { name: 'tp_remainder', label: 'TP remainder (C.3 loop + D reduce-pts + E launcher heap)', prompt: COMMON + '\n\n' + [
    'AREA (you OWN these files exclusively this round): csrc/fused/sm_90/{model_stage_decoder_tc.cuh, fused_decoder_megakernel.cuh, mega_decoder_real_adamw_tc_launcher.cu, tp_layer.cuh}. SPEC: /workspace/impl_diffs/tp_kernel.md — apply EDIT C.3 (the grid-lockstep P1 restructure: the SingleGPU branch is the literal current loop, byte-identical; the kTPComm branch is the lockstep variant), EDIT D (thread <class Par, class Transport> into dectc_forward_tile/backward_tile via _impl bodies + back-compat forwarders + the _tp wrappers, and the FOUR reduce-point if-constexpr(Par::kTPComm) inserts — the spec spells out the forward in full and says MIRROR for the backward: derive the backward dX reduce points the same way), and EDIT E (the launcher: nvshmem_malloc symmetric tp_sym_heap split out of the cudaMalloc workspace + CommCtx population + the TP-degree dispatch allow-list {1,8}). The helpers tp_rowparallel_fwd_partial_tile / tp_allreduce_sum_fixed_order / tp_tile_slot_floats already exist in tp_layer.cuh (verified). GATES: decoder pytest 19/19 byte-identical (SingleGPU unchanged) is MANDATORY; plus the tp_loopback NVSHMEM RDC compile; plus a default decoder TC compile. This is the HARDEST track — if you cannot fully land E (the 8-GPU launcher/NVSHMEM bootstrap), land C.3+D (kernel-side, gated, byte-identical OFF) and the loopback build, and HONESTLY scope what remains. gate_commands: ["CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_decoder_tc.py -q", "bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1"]. area=tp_remainder.',
  ].join('\n') },

  { name: 'memory_strategy', label: 'memory-strategy seam (offload/recompute/streaming planner + gate)', prompt: COMMON + '\n\n' + [
    'AREA (you OWN these — all NEW/disjoint): a NEW grokking_optimizers/parallel/mem_strategy.py (the plan_memory_strategy planner — escalate in-HBM -> nCTA-cap -> recompute -> offload -> stream purely by memory-fit, NEVER by GPU count) + a NEW csrc/fused/sm_90/mem_config.cuh (MemConfig<OffloadOpt,RecomputeActs,StreamLayers,StreamDepth> POD, default InHbm = byte-identical sibling to SingleGPU) + a CPU test. SPEC: /workspace/impl_diffs/memory_strategy.md — apply ONLY item A (the APPLY-READY planner + the mem_config.cuh gate POD + any harness wiring that EMITS -DSG_MEM_* macros). Do NOT attempt items C/D/E (the real offload/recompute/streaming KERNEL bodies — those are scoped follow-ons that touch the launcher/decoder which another agent owns this round). Keep everything byte-identical when OFF (default InHbm, no -DSG_MEM_*). gate_commands: ["python -c \\"import grokking_optimizers.parallel.mem_strategy\\"", "bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1"]. area=memory_strategy.',
  ].join('\n') },

  { name: 'ep_size', label: 'EP 5th axis + size-adaptive CTA-tiling seams', prompt: COMMON + '\n\n' + [
    'AREA (you OWN these): csrc/fused/sm_90/parallel_config.cuh (add EP axis + SizeConfig POD), grokking_optimizers/parallel/auto_config.py (NEW, the 3D-5D inference), grokking_optimizers/megakernel_codegen.py (the decoder_knobs_for_size selector + --decoder-knobs), tests/ (the instantiation allow-list). SPECS: apply BOTH /workspace/impl_diffs/adaptive_parallelism.md (EP as the trailing defaulted 6th ParConfig template param + kEP/kEPComm + ep_* CommCtx fields + auto_config.py infer_parallel_config: base 3D, +SP if sequence, +EP if experts) AND /workspace/impl_diffs/size_adaptive.md (SizeConfig<CtaTile,CtasPerTile,ClusterDim,TileN> + decoder_knobs_for_size + DEC_SIZE_TIERS + --decoder-knobs). Apply the APPLY-NOW Python/POD parts; the real CTA-tiled occupancy>1 kernel body is a scoped follow-on (do NOT author it). CRITICAL byte-identity: every existing ParConfig<5-arg> site must still resolve (EP defaults to 1, kEPComm=false, all EP/Size branches fold); the decoder/vit/mamba layout emitters MUST stay byte-identical (regen + diff). gate_commands: ["python -c \\"import grokking_optimizers.parallel\\"", "python -m grokking_optimizers.megakernel_codegen --decoder-layout > /tmp/d.cuh && diff /tmp/d.cuh csrc/fused/sm_90/decoder_layout.cuh", "bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu", "python -m pytest tests/ -k \\"parallel or config\\" -q"]. area=ep_size.',
  ].join('\n') },
]

phase('Apply')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'Apply', schema: SCHEMA, isolation: 'worktree' })
    .then(r => ({ name: a.name, ...(r || { area: a.name, applied_clean: false, gate_pass: false, commit_sha: '', deviations: ['agent died'], gate_output_tail: '', diffstat: '', summary: 'null' }) }))))
log('Apply-remaining: ' + results.filter(r => r && r.gate_pass).length + '/' + agents.length + ' gates passed; shas ' + results.map(r => (r && r.commit_sha ? r.commit_sha.slice(0,8) : 'none')).join(' '))
return results
