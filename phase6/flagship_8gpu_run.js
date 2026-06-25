export const meta = {
  name: 'supergrok-flagship-8gpu-run',
  description: 'The north-star run: ONE flagship 1.5B decoder across all 8 H100s via TP8 + in-kernel device-NVSHMEM all-reduce (NvshmemTransport), verifying cross-rank loss agreement + finite descending loss',
  phases: [{ title: 'Run', detail: 'distributed flagship decoder across 8 GPUs' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ran_8gpu', 'tp_allreduce_in_kernel', 'cross_rank_agree', 'loss_descends', 'steps', 'per_rank_gib', 'blockers', 'summary'],
  properties: {
    ran_8gpu: { type: 'boolean' },
    tp_allreduce_in_kernel: { type: 'boolean', description: 'did the kernel use the device NvshmemTransport (not loopback)?' },
    cross_rank_agree: { type: 'boolean' }, loss_descends: { type: 'boolean' },
    steps: { type: 'integer' }, per_rank_gib: { type: 'number' },
    blockers: { type: 'array', items: { type: 'string' } }, summary: { type: 'string' },
  },
}
const PROMPT = [
  'You are on the 8x H100 box at /workspace/SuperGrok1.5 (Edit/Write/Bash/build/run; you may make NON-committed scratch wiring like the multiopt runners, but do NOT edit committed kernel source). All 8 GPUs are FREE. GOAL: run ONE flagship 1.5B decoder ACROSS ALL 8 GPUs via TENSOR PARALLELISM (TP8) + ZeRO-3, using the validated IN-KERNEL device-NVSHMEM all-reduce — the north-star one-model-across-8 run.',
  'WHAT IS ALREADY VALIDATED + AVAILABLE: (1) tests/hw/nvshmem_smoke.py + grokking_optimizers/nvshmem_bringup_ext.py + csrc/fused/sm_90/nvshmem_bringup_pybind.cpp — the NVSHMEM UID bootstrap + team_split + symmetric malloc + cross-rank all-reduce, PASSING on 2/4/8 GPUs (study how the smoke inits NVSHMEM under torchrun + sets NVSHMEM_DISABLE_NVLS=1/NCCL_NVLS_ENABLE=0). (2) The decoder TC megakernel + launcher compile under ParConfig<8,8,1,1,Z3>+NvshmemTransport RDC (the {1,8} TP dispatch in mega_decoder_real_adamw_tc_launcher.cu, the sym-heap dec_tc_ensure_tp_sym_heap, CommCtx population). (3) tuning/flagship_distributed.py (the harness: dry-run + the TPBootstrap, now backed by the real pybind) + grokking_optimizers/distributed.py partition_tensor_parallel (the TP weight-shard). (4) The flagship layout decoder_flagship_layout.cuh.',
  'DO: wire torchrun --nproc_per_node=8 so each rank: inits NVSHMEM via the pybind (UID over torch.distributed) + team_split for TP8 + the collective sym-heap malloc; builds/loads the flagship decoder TC with -DSG_HAS_NVSHMEM=1 -rdc=true + ParTP8; TP-shards the 1.476B weights (partition_tensor_parallel: in_proj 3d/8, ff 4d/8 per rank → per-rank Nmax=1.28M) + ZeRO-3 shards the AdamW state; runs the megakernel ~20-50 steps on a fixed batch with the IN-KERNEL tp_allreduce firing at the 4 reduce points (NvshmemTransport, NOT loopback). NOTE: attention runs full-width per rank on the kTPComm path (the head-shard is scoped out; that is fine — the 4 linear-projection reduces are what shard the workspace). VERIFY: ran_8gpu, the kernel used the device NvshmemTransport (not loopback), cross-rank loss AGREES (all 8 ranks same loss, since TP is one logical model), loss DESCENDS over steps, per-rank GiB. BE HONEST about any blocker (if the full megakernel TP path has a runtime bug the loopback/compile did not catch, scope it precisely — this is the first time the in-kernel device all-reduce drives the real megakernel end-to-end).',
  'Set NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem, NVSHMEM_DISABLE_NVLS=1, NCCL_NVLS_ENABLE=0. Report the schema fields.',
].join('\n')

phase('Run')
const r = await agent(PROMPT, { label: 'flagship one-model-across-8 run', phase: 'Run', schema: SCHEMA })
log('8-GPU flagship: ran=' + (r&&r.ran_8gpu) + ' in-kernel-nvshmem=' + (r&&r.tp_allreduce_in_kernel) + ' agree=' + (r&&r.cross_rank_agree) + ' descends=' + (r&&r.loss_descends))
return r || { ran_8gpu:false, tp_allreduce_in_kernel:false, cross_rank_agree:false, loss_descends:false, steps:0, per_rank_gib:0, blockers:['agent died'], summary:'null' }
