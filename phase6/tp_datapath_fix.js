export const meta = {
  name: 'supergrok-tp-datapath-fix',
  description: 'Fix the 3 committed-source megakernel TP-data-path bugs (IMA + missing per-rank weight offset + head divisibility) so the one-model-across-8 flagship decoder run trains correctly; gated SingleGPU byte-identical + 8-GPU cross-rank agreement',
  phases: [{ title: 'Fix', detail: 'debug+fix the kTPComm megakernel data path, re-run 8-GPU' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['singlegpu_byte_identical', 'ima_cleared', 'cross_rank_agree', 'loss_descends', 'commit_sha', 'remaining', 'summary'],
  properties: {
    singlegpu_byte_identical: { type: 'boolean', description: 'decoder pytest 19/19 (kTPComm changes fold away on SingleGPU)' },
    ima_cleared: { type: 'boolean', description: 'compute-sanitizer clean on the 8-GPU kTPComm run' },
    cross_rank_agree: { type: 'boolean' }, loss_descends: { type: 'boolean' },
    commit_sha: { type: 'string' }, remaining: { type: 'array', items: { type: 'string' } }, summary: { type: 'string' },
  },
}
const PROMPT = [
  'ISOLATED GIT WORKTREE of /workspace/SuperGrok1.5 — you MAY edit committed kernel source (csrc/fused/sm_90/). Build/run/debug freely on the 8 H100s. GOAL: fix the 3 bugs that crash the one-model-across-8 flagship decoder run on the in-kernel device-NVSHMEM TP path, so it trains correctly across 8 GPUs.',
  'CONTEXT (validated): the NVSHMEM transport + bring-up + launcher sym-heap + ParTP8 codegen all WORK (8-GPU all-reduce smoke bit-exact). The crash is in the megakernel TP DATA PATH, only exercised when the device NvshmemTransport drives the real flagship megakernel (d=1600/L48). compute-sanitizer log: /tmp/claude-0/-/6354dc07-b50f-40a0-8748-5189102539d3/scratchpad/san_rank0.log (87 invalid 4-byte global writes in fused_decoder_megakernel_tc<AdamW,ParConfig<8,8,1,1,Z3>>+0x1dc30, wild ptr ~35GiB below the NVSHMEM heap). Re-run wiring (non-committed, reuse it): tuning/_tp8_scratch_pybind.cu (calls the launcher tp_size=8 arm), tuning/_tp8_build.sh (the manual -rdc -dlink build torch JIT omits), tuning/_tp8_run.py (the torchrun driver). NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem; export NVSHMEM_DISABLE_NVLS=1 NCCL_NVLS_ENABLE=0.',
  'THE 3 BUGS TO FIX (in csrc/fused/sm_90/{model_stage_decoder_tc.cuh, fused_decoder_megakernel.cuh, mega_decoder_real_adamw_tc_launcher.cu}):',
  '(A) PER-RANK WEIGHT SHARD OFFSET (the correctness bug): dectc_wbf_convert + dec_bind read the FULL weight matrices with full kDecOffsets identically on every rank — comm.tp_rank is never used. For the column/row-parallel TP GEMMs, each rank must cache/use ITS shard: for column-parallel weights (in_proj 3d, ff0 dff) rank r owns columns [r*Nout/P,(r+1)*Nout/P); for row-parallel (out_proj, ff2) rank r owns the input-rows slice [r*Kin/P,...]. Thread comm.tp_rank/tp_size into the wbf cache + the GEMM source pointers so each rank computes its DISTINCT partial (then the existing in-kernel all-reduce sums the full-width result correctly). This is what the tp_layer.cuh helpers expect (a rank-distinct Wshard ptr).',
  '(B) ATTENTION at kHeads%TP!=0: flagship kHeads=25, TP=8. The simplest CORRECT fix: on the kTPComm path run attention FULL-WIDTH REPLICATED (each rank computes all 25 heads from the full qkv — the head-shard is a memory optimization, not required for correctness; the out_proj all-reduce already recombines). Make the kTPComm attention call the SAME full-width attention as SingleGPU (Hloc=kHeads, Dloc=kD), NOT the partial Hloc=kHeads/TP path, when kHeads%TP!=0 (or just always full-width on kTPComm for now). This removes the Dloc==Hloc*kDhead invariant violation. NOTE: if attention is full-width, the in_proj for attention must NOT be column-sharded the way (A) shards ff — keep the attention qkv full-width (replicated) and only TP-shard the FFN weights + out_proj; reconcile (A) so attention weights stay full and only ff0/ff2 + out_proj shard. Be careful + consistent.',
  '(C) THE IMA is most likely a consequence of (A)/(B) (wrong offsets / inconsistent widths writing out of bounds). After fixing A+B, re-run under compute-sanitizer and confirm 0 errors; if a distinct IMA remains, debug it (the +0x1dc30 offset, the dW/acts/heap indexing on the kTPComm path).',
  'GATES (MANDATORY): (1) decoder pytest CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_decoder_tc.py -q -> 19/19 (your kTPComm edits MUST be if-constexpr(Par::kTPComm)-gated so SingleGPU is byte-identical). (2) the 8-GPU run (tuning/_tp8_run.py via torchrun --nproc_per_node=8): compute-sanitizer CLEAN, all 8 ranks same loss (cross_rank_agree), loss DESCENDS over ~20 steps. Commit your committed-source fixes (git add -A && git commit -m "fix: megakernel TP data-path (per-rank weight shard + full-width attn on kTPComm)"), report sha. BE HONEST if a bug resists — scope it precisely. gfx942/tpu untouched.',
].join('\n')

phase('Fix')
const r = await agent(PROMPT, { label: 'TP data-path fix + 8-GPU re-run', phase: 'Fix', schema: SCHEMA, isolation: 'worktree' })
log('TP data-path fix: singlegpu=' + (r&&r.singlegpu_byte_identical) + ' ima_clear=' + (r&&r.ima_cleared) + ' agree=' + (r&&r.cross_rank_agree) + ' descends=' + (r&&r.loss_descends) + ' sha=' + (r&&r.commit_sha?r.commit_sha.slice(0,8):'none'))
return r || { singlegpu_byte_identical:false, ima_cleared:false, cross_rank_agree:false, loss_descends:false, commit_sha:'', remaining:['agent died'], summary:'null' }
