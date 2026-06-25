export const meta = {
  name: 'supergrok-vit-mamba-tp',
  description: 'Two parallel worktree agents: mirror the DONE decoder TP track (foundation + C.3 loop + D reduce-points + E sym-heap launcher + attention head-shard) onto ViT and Mamba, gated SingleGPU byte-identical + loopback/NVSHMEM compile',
  phases: [{ title: 'TP', detail: 'mirror decoder TP onto ViT and Mamba' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'applied_clean', 'gate_pass', 'commit_sha', 'deviations', 'gate_output_tail', 'diffstat', 'summary'],
  properties: {
    area: { type: 'string' }, applied_clean: { type: 'boolean' }, gate_pass: { type: 'boolean' },
    commit_sha: { type: 'string' }, deviations: { type: 'array', items: { type: 'string' } },
    gate_output_tail: { type: 'string' }, diffstat: { type: 'string' }, summary: { type: 'string' },
  },
}
const COMMON = [
  'You operate in an ISOLATED GIT WORKTREE of /workspace/SuperGrok1.5, branched from the FULL merged state where the DECODER TP TRACK IS COMPLETE and is your WORKED REFERENCE. Study how the decoder did it, then MIRROR it onto your model.',
  'DECODER TP REFERENCE (read these to learn the exact pattern): csrc/fused/sm_90/parallel_config.cuh (ParConfig/SingleGPU/CommCtx/SizeConfig); csrc/fused/sm_90/tp_transport.cuh (LoopbackTransport + NvshmemTransport + make_transport_from_comm); csrc/fused/sm_90/tp_layer.cuh (tp_rowparallel_fwd_partial_tile / tp_colparallel_dx_partial_tile / tp_allreduce_sum_fixed_order, templated on weight dtype); csrc/fused/sm_90/fused_decoder_megakernel.cuh (the template<OptId Opt, class Par=SingleGPU> kernel + the if constexpr(!Par::kTPComm) grid-stride loop vs the kTPComm grid-lockstep n_rounds/active loop + tr construction + launcher templated on Par + nvshmem_malloc sym-heap dec_tc_ensure_tp_sym_heap + CommCtx population + {1,8} dispatch); csrc/fused/sm_90/model_stage_decoder_tc.cuh (the dectc_*_tile_impl<Par,Transport> two-body shape + the FOUR if constexpr(Par::kTPComm) reduce points at out_proj-fwd/ff2-fwd/ff0-dX/in_proj-dX + the attention head-localization H_loc=kHeads/TP); csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu (EDIT E). The specs /workspace/impl_diffs/{tp_kernel,dist_step}.md describe the decoder version.',
  'PROCEDURE: mirror EVERY decoder TP edit onto your model files (same column/row Megatron shard of the attn in_proj + ff weights, same 4 reduce points, same grid-lockstep loop, same launcher sym-heap, same attention head-shard). Apply with exact-match Edits; build; gate. Then restore dirty *.pyc and git add -A and git commit -m "vit/mamba TP track"; report git rev-parse HEAD.',
  'RULES (MANDATORY): the SingleGPU/default path MUST stay BYTE-IDENTICAL — your model pytest gate must pass unchanged (every new branch if constexpr(Par::kTPComm)-gated OFF; ParConfig EP defaults preserve all sites). gfx942/tpu untouched. NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem. Be HONEST in gate_pass — a truthful partial with the error tail beats a forced pass. If a piece (e.g. the model lacks attention, like Mamba) does not map, scope it precisely.',
].join('\n')

const agents = [
  { name: 'vit_tp', label: 'ViT TP (mirror decoder track)', prompt: COMMON + '\n\n' + [
    'YOUR MODEL: ViT. Files: csrc/fused/sm_90/{model_stage_vit_tc.cuh, fused_vit_megakernel.cuh, mega_vit_real_adamw_tc_launcher.cu (if present; else the ViT launcher)} + reuse the SHARED tp_transport.cuh/tp_layer.cuh/parallel_config.cuh (do NOT duplicate them — they are model-agnostic). Mirror the decoder TP track: template the ViT megakernel on <OptId Opt, class Par=SingleGPU> + CommCtx; the grid-lockstep loop; the reduce points at the ViT analogues (attn out_proj fwd, ff2 fwd, ff0 dX, in_proj dX); the attention head-shard H_loc=vit::kHeads/TP; the launcher sym-heap + {1,8} dispatch. GATES: ViT pytest (find it: tests/hw/test_vit_tc.py or equivalent) must stay byte-identical (SingleGPU); plus a default ViT TC compile; plus the ViT NVSHMEM RDC compile (-DSG_VIT_SCALAR_MEGAKERNEL=0 -DSG_HAS_NVSHMEM=1 -rdc=true -I$NVSHMEM_HOME/include). gate_commands: ["CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_vit_tc.py -q || ls tests/hw | grep -i vit", "bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_vit_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 -DSG_VIT_SCALAR_MEGAKERNEL=0"]. area=vit_tp.',
  ].join('\n') },

  { name: 'mamba_tp', label: 'Mamba TP (mirror decoder track)', prompt: COMMON + '\n\n' + [
    'YOUR MODEL: Mamba-3. Files: csrc/fused/sm_90/{model_stage_mamba_tc.cuh, fused_mamba_megakernel.cuh, the Mamba launcher} + reuse the SHARED tp_transport.cuh/tp_layer.cuh/parallel_config.cuh. Mirror the decoder TP track. Mamba has NO standard attention — instead TP-shard the SSM/projection weights (the in_proj / out_proj / the SSM A,B,C,D + conv along the appropriate dimension; column/row Megatron split) and place the all-reduce at the analogous recombination points (after the row-parallel out projection). Identify the Mamba reduce points from the model_stage_mamba_tc.cuh structure (mirror the decoder logic: a row-parallel GEMM that produces a partial -> publish -> fixed-order reduce). The attention head-shard does NOT apply (Mamba has no heads) — scope that precisely; do the SSM/projection shard instead. GATES: Mamba pytest byte-identical (SingleGPU) + a default Mamba TC compile + the Mamba NVSHMEM RDC compile. gate_commands: ["CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_mamba_tc.py -q || ls tests/hw | grep -i mamba", "bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1"]. area=mamba_tp.',
  ].join('\n') },
]

phase('TP')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'TP', schema: SCHEMA, isolation: 'worktree' })
    .then(r => ({ name: a.name, ...(r || { area: a.name, applied_clean: false, gate_pass: false, commit_sha: '', deviations: ['agent died'], gate_output_tail: '', diffstat: '', summary: 'null' }) }))))
log('ViT/Mamba TP: ' + results.filter(r => r && r.gate_pass).length + '/' + agents.length + ' gates passed; shas ' + results.map(r => (r && r.commit_sha ? r.commit_sha.slice(0,8) : 'none')).join(' '))
return results
