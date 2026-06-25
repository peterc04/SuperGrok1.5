export const meta = {
  name: 'supergrok-8gpu-bringup',
  description: 'Two parallel worktree agents for the 8-GPU flagship bring-up: TP attention head-localization (kernel) and the host NVSHMEM team bootstrap + weight-shard + torchrun harness (host glue)',
  phases: [{ title: 'Bringup', detail: '2 isolated agents apply the 8-GPU bring-up pieces' }],
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
  'You operate in an ISOLATED GIT WORKTREE of /workspace/SuperGrok1.5, branched from the FULL merged state: 3 flagships, CuTe, the COMPLETE TP track (foundation + EDIT C.3 grid-lockstep loop + D four reduce-points + E nvshmem_malloc symmetric-heap launcher with the {1,8} TP dispatch), resource planner, EP 5th axis, size-adaptive selector, datasets. NVSHMEM 3.7.0 at NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem; the decoder TC TU already compiles with -DSG_HAS_NVSHMEM=1 -rdc=true -I$NVSHMEM_HOME/include AND the ParTP8+NvshmemTransport launcher RDC compiles. Freely Edit/Write/Bash/build.',
  'PROCEDURE: read your spec(s) IN FULL; apply with exact-match Edits (record drift in deviations); run the gate_commands; fix mechanical apply errors and re-run (NEVER loosen a correctness gate); then restore dirty *.pyc (git checkout -- on pyc) and git add -A and git commit -m "bringup: <area>"; report git rev-parse HEAD as commit_sha and the schema fields HONESTLY (a truthful gate_pass=false with the error tail beats a forced pass).',
  'RULES: the SingleGPU/default path MUST stay byte-identical (decoder pytest tests/hw/test_decoder_tc.py 19/19 is the MANDATORY gate for any kernel edit — every new branch if constexpr(Par::kTPComm)-gated OFF); gfx942/tpu untouched. scripts/compile_to_object.sh at repo root.',
].join('\n')

const agents = [
  { name: 'attention_shard', label: 'TP attention head-localization (kernel)', prompt: COMMON + '\n\n' + [
    'AREA (you OWN this): csrc/fused/sm_90/model_stage_decoder_tc.cuh (the attention fwd/bwd tile fns dectc_attn_fwd_tile / dectc_attn_bwd_tile and their callers inside dectc_forward_tile_impl / dectc_backward_tile_impl). GOAL: implement the TP ATTENTION HEAD-LOCALIZATION that tp_kernel.md scoped out (its NOTE in §6/§12): on the kTPComm path, with column-parallel in_proj each TP rank owns H_loc = dec::kHeads / Par::kTP heads, so the attention (qkv split, per-head softmax, context) must run over the LOCAL head shard (local qkv stride 3*dec::kD/TP, H_loc heads) instead of the full dec::kHeads / 3*dec::kD. Wrap the change if constexpr(Par::kTPComm) so the SingleGPU path is BYTE-IDENTICAL (kHeads/1 = kHeads, full stride). The out_proj forward all-reduce (reduce point 1, already applied) recombines the per-rank context to full width, so downstream is unchanged. Verify the formulas: at TP=1, H_loc==kHeads and the strides equal the current literals exactly. GATES: decoder pytest 19/19 (SingleGPU byte-identical) MANDATORY; plus the ParTP8 RDC compile: NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem; bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 -DSG_HAS_NVSHMEM=1 -rdc=true -I$NVSHMEM_HOME/include. If kHeads%TP!=0 for some TP, restrict to the {1,8} dispatch set (kHeads divisible by 8 — check dec::kHeads). gate_commands: ["CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_decoder_tc.py -q"]. area=attention_shard.',
  ].join('\n') },

  { name: 'host_bringup', label: 'host NVSHMEM bootstrap + weight-shard + torchrun harness', prompt: COMMON + '\n\n' + [
    'AREA (you OWN these): grokking_optimizers/distributed.py (the TP weight-shard) + a NEW host bootstrap helper + a NEW tuning/flagship_distributed.py torchrun harness. SPECS: /workspace/impl_diffs/dist_step.md §6.C.5 (partition_tensor_parallel: the per-rank Megatron column/row weight-shard of in_proj 3d x d and ff 4d x d so per-rank Nmax = kDecMaxTensorNumel/TP) + /workspace/impl_diffs/run_harness.md (the flagship_distributed.py torchrun harness + the per-rank memory math + the TP8.ZeRO-3 mesh). GOAL: (1) the host weight-shard that feeds per-rank sharded params/grad/state to the launcher; (2) the host NVSHMEM TP-team bootstrap: nvshmem_init + nvshmem_team_split_strided for the TP group + the COLLECTIVE nvshmem_malloc of the symmetric tp_sym_heap (matching dec_tc_ensure_tp_sym_heap in the launcher EDIT E), with the team id stored into CommCtx.tp_comm_handle as the void*-cast int32; (3) the torchrun harness tuning/flagship_distributed.py that launches the flagship decoder under TP8.ZeRO-3 across 8 GPUs, seeds identically, and verifies cross-rank loss agreement + A/A/A. BE HONEST: NVSHMEM multi-GPU bootstrap on this box may need MPI/PMI or the nvshmem bootstrap plugin; if the actual 8-GPU launch cannot be brought up here, deliver the apply-ready host code + harness + a --dry-run that validates the plan/mesh/shard math on CPU, and scope the live-run blocker precisely. gate_commands: ["python -c \\"import grokking_optimizers.distributed\\"", "python tuning/flagship_distributed.py --help", "python tuning/flagship_distributed.py --dry-run --model decoder --gpus 8"]. area=host_bringup.',
  ].join('\n') },
]

phase('Bringup')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'Bringup', schema: SCHEMA, isolation: 'worktree' })
    .then(r => ({ name: a.name, ...(r || { area: a.name, applied_clean: false, gate_pass: false, commit_sha: '', deviations: ['agent died'], gate_output_tail: '', diffstat: '', summary: 'null' }) }))))
log('Bringup: ' + results.filter(r => r && r.gate_pass).length + '/' + agents.length + ' gates passed; shas ' + results.map(r => (r && r.commit_sha ? r.commit_sha.slice(0,8) : 'none')).join(' '))
return results
