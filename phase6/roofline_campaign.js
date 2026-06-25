export const meta = {
  name: 'supergrok-roofline-campaign',
  description: 'Sustained GPU-saturating roofline campaign (deliverable #1): build + measure achieved TF/s (wallclock + analytical FLOPs) and analytical arithmetic intensity for the flagship cells across GPUs 1-7, then plot the roofline graph vs the H100 bf16 ceiling',
  phases: [{ title: 'Roofline', detail: 'build+measure flagship cells, produce the roofline PNG' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['cells_measured', 'graph_path', 'csv_path', 'peak_pct_range', 'blockers', 'summary'],
  properties: {
    cells_measured: { type: 'integer' },
    graph_path: { type: 'string' }, csv_path: { type: 'string' },
    peak_pct_range: { type: 'string', description: 'min..max % of the 989 TF/s bf16 ceiling across measured cells' },
    blockers: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}
const PROMPT = [
  'You are on the 8x H100 box at /workspace/SuperGrok1.5 (Edit/Write/Bash/build/run; do NOT commit, do NOT touch the .claude/worktrees other agents use). GOAL: a ROOFLINE CEILING test of the FLAGSHIP cells (deliverable #1) — and keep GPUs 1-7 BUSY the whole time (a sustained job, not short bursts). ncu HW counters are DENIED, so do it ncu-FREE.',
  'MEASUREMENT TOOL — USE nsys (NOT CUDA-event wallclock, NOT ncu): the user explicitly wants nsys, which is what we use (ncu HW counters are DENIED in-container; nsys works). nsys at /opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys (or `which nsys`). The L3-TC megakernel is ONE __global__ launch per step, so the nsys CUDA-kernel summary duration IS the authoritative per-step GPU time.',
  'METHOD per cell (model x optimizer at flagship): (1) build the flagship cell (decoder d=1600/L48, ViT d=1664/L48, mamba d=2048/L24) like /workspace/phase1/flagship_train.py does (-include flagship layout + -DSG_*_SCALAR_MEGAKERNEL=0 + the bench-layout staged-opt elision + ncta_cap=8); REUSE cached builds (decoder multiopt .so already built last run — the cells #23 decoder set was measured; keep its builds). (2) run the cell under nsys: `nsys profile -t cuda -o /workspace/phase6/nsys/<model>_<opt> --force-overwrite=true python <runner with ~30 steps>` then `nsys stats --report cuda_gpu_kern_sum <out>.nsys-rep` (or --report gpukernsum), and EXTRACT the fused_<model>_megakernel_tc kernel total/avg GPU ns and the launch count → per-step kernel GPU time. (3) ACHIEVED TF/s = (analytical GEMM FLOPs/step) / (per-step kernel GPU seconds) — reuse the decoder_bench.py FLOP formula (it prints GEMM FLOPs/step; mirror for ViT/Mamba). (4) analytical ARITHMETIC INTENSITY (FLOP/byte) = GEMM FLOPs / bytes-moved (params+acts read/written per step, bf16). Record (intensity, achieved TF/s, %of 989 TF/s peak, nsys-kernel-ns) per cell to a CSV. RE-MEASURE the decoder cells with nsys too (for a consistent all-nsys graph; the prior run used wallclock).',
  'COVERAGE: start with the 3 models x AdamW (the readily-buildable cells; decoder flagship build is cached at /workspace/flagship_build/mega_decoder_flagship). Then expand to the other optimizers per model IF the launcher exposes an opt_id (check mega_*_real_adamw_tc_launcher.cu — it dispatches multiple OptIds; if tc_train_step takes an opt_id you get many cells from ONE build). Measure as many of the 33 cells as you can; HONESTLY report which are pending + why. Spread builds/runs across GPUs 1-7 (NOT GPU 0 — reserved for other agents gates) and keep several running concurrently so the GPUs stay saturated.',
  'DELIVERABLE: a matplotlib roofline PNG at /workspace/phase6/roofline_flagship.png — x=arithmetic intensity (FLOP/byte, log), y=achieved TF/s (log), the H100 bf16 roofline (ridge at 989 TF/s, the HBM3 ~3.35 TB/s bandwidth slope), and each measured cell as a point labelled model:opt. Also a CSV /workspace/phase6/roofline_flagship.csv. Report cells_measured, graph_path, csv_path, peak_pct_range, blockers, summary.',
].join('\n')

phase('Roofline')
const r = await agent(PROMPT, { label: 'roofline campaign (flagship cells)', phase: 'Roofline', schema: SCHEMA })
log('Roofline: ' + (r && r.cells_measured) + ' cells; graph ' + (r && r.graph_path) + '; peak% ' + (r && r.peak_pct_range))
return r || { cells_measured: 0, graph_path: '', csv_path: '', peak_pct_range: '', blockers: ['agent died'], summary: 'null' }
