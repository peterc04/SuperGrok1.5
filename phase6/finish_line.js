export const meta = {
  name: 'supergrok-finish-line',
  description: 'Parallel finish-line: staged-opt state plumbing (all 11 opts at flagship), launcher NVSHMEM-init pybind (cross-GPU one-model-across-8), and the pre-existing Mamba test recalibration',
  phases: [{ title: 'Finish', detail: '3 worktree agents close the finish-line gaps' }],
}
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'done', 'gate_pass', 'commit_sha', 'deviations', 'gate_output_tail', 'summary'],
  properties: {
    area: { type: 'string' }, done: { type: 'boolean' }, gate_pass: { type: 'boolean' },
    commit_sha: { type: 'string' }, deviations: { type: 'array', items: { type: 'string' } },
    gate_output_tail: { type: 'string' }, summary: { type: 'string' },
  },
}
const COMMON = [
  'ISOLATED GIT WORKTREE of /workspace/SuperGrok1.5, branched from the FULL state (3 flagships + full TP for all 3 models + resource planner + EP + size-adaptive + memory-strategy + datasets + host bring-up). Freely Edit/Write/Bash/build/run.',
  'PROCEDURE: read the cited code/specs IN FULL; implement; build+gate; restore dirty *.pyc (git checkout -- on pyc), git add -A, git commit -m "finish: <area>"; report git rev-parse HEAD. Be HONEST in gate_pass.',
  'RULES: keep the SingleGPU/default path byte-identical (the 3 model pytests are the byte-identity gates); gfx942/tpu untouched. NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem. scripts/compile_to_object.sh at repo root.',
].join('\n')

const agents = [
  { name: 'staged_opt_plumbing', label: 'staged-opt state plumbing → all 11 flagship opts', prompt: COMMON + '\n\n' + [
    'GOAL: make ALL 11 optimizers runnable at flagship single-GPU (the elementwise 5 already run via the Lever-3 multiopt driver at /tmp/claude-0/-/6354dc07-b50f-40a0-8748-5189102539d3/scratchpad/mega_decoder_multiopt_tc.cu — study it). Extend a multiopt flagship driver (a scratchpad JIT TU, build-via-include only, NO committed-source edit, mirroring mega_decoder_real_adamw_tc_launcher.cu opt_id dispatch) to the STAGED optimizers Muon/Prodigy/LookSAM/SuperGrok11/SuperGrok15: allocate their EXTENDED state buffers + device weight/meta packs that the launcher binds host-side (Prodigy param_init/persist; SG11/15 sharpness+phi packs; LookSAM SAM scratch; Muon NS scratch). The resource planner confirms they FIT single-GPU at full occupancy (33.5-44.5 GiB) EXCEPT SuperGrok2 (ncta=1 deep limit — run SG2 at ncta=1 or document it FITS-but-slow). Then RUN each of the 11 opts on the flagship decoder (B=16, fixed batch, ~100 steps), banking the loss-vs-step trajectories → the COMPLETE 11-optimizer flagship ranking. Write the trajectories to /workspace/phase6/flagship_11opt_ranking.json + a readable table. Use GPUs 1-7 (GPU 0 reserved). gate: all 11 produce finite descending loss. area=staged_opt_plumbing.',
  ].join('\n') },

  { name: 'nvshmem_pybind', label: 'launcher NVSHMEM-init pybind + 2-GPU smoke', prompt: COMMON + '\n\n' + [
    'GOAL: enable the ACTUAL cross-GPU one-model-across-8 run (the capability track). The kernel+launcher already compile under ParTP8+NvshmemTransport RDC; the missing piece is the host NVSHMEM bring-up, which canNOT be driven from Python (no nvshmem python binding). Add a small C++ pybind (a new TU, or extend the launcher build) exposing: nvshmem_init via the UID bootstrap (nvshmemx_get_uniqueid + nvshmemx_init_attr with the UID broadcast over torch.distributed/NCCL — MPI/PMI are absent; the UID plugin nvshmem_bootstrap_uid IS present), nvshmem_team_split_strided for the TP team, and the collective nvshmem_malloc of the symmetric tp_sym_heap (matching dec_tc_ensure_tp_sym_heap). Wire it into tuning/flagship_distributed.py (the harness has the dry-run + TPBootstrapBlocked stub — replace the stub with the real pybind call). Then a 2-GPU SMOKE: torchrun --nproc_per_node=2 a minimal nvshmem_init+team+malloc+a tiny all-reduce, asserting cross-rank correctness. BE HONEST: if the multi-GPU torchrun+NVSHMEM bring-up hits a runpod/container blocker, deliver the apply-ready pybind + the smoke script + scope the exact blocker (this is the genuine wildcard). gate: the 2-GPU nvshmem smoke runs OR the blocker is precisely scoped. area=nvshmem_pybind.',
  ].join('\n') },

  { name: 'mamba_test_fix', label: 'Mamba test recalibration (2 pre-existing failures)', prompt: COMMON + '\n\n' + [
    'GOAL: fix the 2 PRE-EXISTING Mamba test failures (NOT TP regressions — confirmed: the kernel matches the bf16 oracle). (1) tests/hw/test_mamba_tc.py::test_tc_single_step_grad_parity fails because layers.0.mixer.B_bias has an irreducible bf16 noise floor of ~0.155 (k-vs-bf16 0.1547 == bf16-vs-fp64 0.1546, i.e. the kernel IS correct) but _TC_GRAD_REL=0.08 is too tight for the SSM B-projection biases. RECALIBRATE exactly as the decoder did (test_decoder_tc.py uses per-tensor calibrated tols with a documented bf16-floor witness): set a higher tol for the high-cancellation SSM bias group (B_bias/Bhat_bias/C_bias/etc) to its MEASURED floor with ~1.5-3x headroom, keeping the discipline (gate to the measured floor + the structural witness layer0~=layer1, NOT a loose pass-everything number) — a real bug would show k-vs-bf16 >> bf16-vs-fp64, which the recalibrated test still CATCHES. (2) test_tc_proj_dw_exact_on_own_operands fails with RuntimeError: tc_dump_outproj_operands obsolete on the Mamba-3 scalar path. Either skip it with @pytest.mark.skip(reason=...) on the scalar path (the wgmma-operand-dump witness does not apply to the scalar SSM path) OR adapt it to the scalar full-grad-partial dW witness if feasible. gate: CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_mamba_tc.py -q → all pass (no correctness suppressed — only floor-calibrated tols + an honest skip). area=mamba_test_fix.',
  ].join('\n') },
]

phase('Finish')
const results = await parallel(agents.map(a => () =>
  agent(a.prompt, { label: a.label, phase: 'Finish', schema: SCHEMA, isolation: 'worktree' })
    .then(r => ({ name: a.name, ...(r || { area: a.name, done: false, gate_pass: false, commit_sha: '', deviations: ['agent died'], gate_output_tail: '', summary: 'null' }) }))))
log('Finish-line: ' + results.filter(r => r && r.gate_pass).length + '/' + agents.length + ' gates passed; shas ' + results.map(r => (r && r.commit_sha ? r.commit_sha.slice(0,8) : 'none')).join(' '))
return results
