export const meta = {
  name: 'supergrok-lever3-overlap',
  description: 'Lever 3: start overlapping flagship-scale GPU compute NOW on the 8 idle H100s — build the flagship decoder TUs for the single-GPU-fitting optimizers and launch concurrent flagship trainings, banking loss-vs-step ranking trajectories while the 8-GPU TP impl lands',
  phases: [{ title: 'Overlap', detail: 'saturate the 8 GPUs with flagship-scale decoder training + bank trajectories' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ran', 'optimizers_run', 'gpus_used', 'trajectories', 'blockers', 'summary'],
  properties: {
    ran: { type: 'boolean' },
    optimizers_run: { type: 'array', items: { type: 'string' } },
    gpus_used: { type: 'integer' },
    trajectories: { type: 'string', description: 'compact loss-vs-step per optimizer (the early ranking signal)' },
    blockers: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}
const PROMPT = [
  'You are on the 8x H100 box at /workspace/SuperGrok1.5 (you may Edit/Write/Bash/build/run — NOT in a worktree, work in the main tree but do NOT commit). GOAL (Lever 3 = overlap the benchmark run with implementation): SATURATE the 8 idle H100s NOW with FLAGSHIP-SCALE (d=1600, L=48, 1.476B-param) DECODER training, banking loss-vs-step trajectories — the early optimizer-ranking signal — while other agents finish the 8-GPU TP path.',
  'CONTEXT: the flagship decoder builds + runs + trains single-GPU TODAY. Reference runner: /workspace/phase1/flagship_train.py (builds mega_decoder_real_adamw_tc against the flagship layout with -DSG_DEC_BENCH_LAYOUT=1 -DSG_DEC_SCALAR_MEGAKERNEL=0 -DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 -include csrc/fused/sm_90/decoder_flagship_layout.cuh, ncta_cap=8, B=16, runs tc_train_step; AdamW loss descended 4.59->2.69). The resource planner grokking_optimizers/parallel/resource_planner.py tells which optimizers FIT single-GPU at decoder-flagship (the elementwise opts fit; SG2 does not at full occupancy — its per-CTA scratch is the KNOWN DEEP LIMIT).',
  'DO THIS: (1) Determine which of the 11 optimizers are runnable at flagship single-GPU. The launcher mega_decoder_real_adamw_tc_launcher.cu dispatches multiple OptIds via opt_id — check whether tc_train_step (or a sibling entry) accepts an opt_id so ONE flagship build runs several optimizers; if only AdamW is exposed, that is fine. (2) Launch CONCURRENT flagship trainings across as many of the 8 GPUs as you can (CUDA_VISIBLE_DEVICES=k each), one optimizer (or one seed/config) per GPU, ~100-300 steps each on a FIXED batch (overfit) OR the mod-97 stream, banking loss every ~10 steps. At minimum run AdamW on all 8 GPUs with varied seeds (saturation + stability). Ideally run the distinct elementwise optimizers you can build (Lion, GrokFast, GrokAdamW, NeuralGrok) one per GPU. (3) Collect the loss-vs-step trajectories into a compact table (the early ranking). (4) Be HONEST about what you could build/run vs blockers (per-opt flagship TU build issues, memory).',
  'CONSTRAINTS: do NOT edit committed source (build via -D flags + -include only, like flagship_train.py); do NOT touch the worktrees the other agents use; keep each run within ~50 GiB/GPU (ncta_cap=8, B<=16). CONTENTION GUARD: other implementation agents run pytest GATES on GPU 0 (CUDA_VISIBLE_DEVICES=0) — so use ONLY GPUs 1-7 for your trainings (leave GPU 0 free). Report ran / optimizers_run / gpus_used / trajectories / blockers / summary.',
].join('\n')

phase('Overlap')
const r = await agent(PROMPT, { label: 'Lever 3: overlap flagship GPU compute', phase: 'Overlap', schema: SCHEMA })
log('Lever 3: ran=' + (r && r.ran) + ' gpus=' + (r && r.gpus_used) + ' opts=' + (r && r.optimizers_run ? r.optimizers_run.join(',') : 'none'))
return r || { ran: false, optimizers_run: [], gpus_used: 0, trajectories: '', blockers: ['agent died'], summary: 'null' }
