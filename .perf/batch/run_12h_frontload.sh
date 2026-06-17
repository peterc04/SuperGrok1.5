#!/usr/bin/env bash
# ============================================================================
# 12h AUTONOMOUS GPU FRONT-LOAD — runs unattended during the usage-reset gap.
# Keeps the H100 productive while the operator (+ the orchestrating model) are
# away. RECORDS verdicts/winners to .perf/batch/ for REVIEW — does NOT commit
# or auto-ship anything (the keep/revert ratchet stays the operator's call).
#
# Launch ONLY after the GPU is free (no other timing job) — it owns the GPU.
# Robust: every step is timeout-bounded + `|| true` + logged; one failure never
# kills the run.
# ============================================================================
cd /workspace/SuperGrok1.5 || exit 1
mkdir -p .perf/batch
exec >> .perf/batch/run.log 2>&1
set +e
echo "==================== FRONT-LOAD START $(date -u) ===================="
# shellcheck disable=SC1091
source .fast_build_env.sh 2>/dev/null || true

# Abort-guard: if compile.py is broken CPU-side, don't waste the GPU window.
echo "--- preflight: compile.py --self-test (CPU) ---"
if ! CUDA_VISIBLE_DEVICES="" timeout 10m python -m grokking_optimizers.compile --self-test > .perf/batch/preflight_selftest.log 2>&1; then
  echo "!!! preflight self-test FAILED (rc=$?) — see preflight_selftest.log; ABORTING front-load to avoid wasting the GPU window."
  echo "ABORTED-PREFLIGHT $(date -u)" > .perf/batch/DONE
  exit 1
fi
echo "--- preflight OK: $(tail -1 .perf/batch/preflight_selftest.log) ---"

# ---- GPU-WAIT GUARD: self-launch when the GPU is free (the ViT B1 gate done) ----
# The ViT B1 gate's build phases idle the GPU for <10min; 10 consecutive low
# readings (= 10min sustained idle) means it is truly done. The other in-flight
# workflows are read-only/worktree-isolated (no GPU), so the GPU's only user is
# the ViT B1 gate. Cap at 2h so a stuck gate never blocks the window forever.
echo "--- waiting for GPU to free (ViT B1 gate) @ $(date -u) ---"
low=0; waited=0
while [ "$low" -lt 10 ] && [ "$waited" -lt 7200 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$used" ] && [ "$used" -lt 4000 ]; then low=$((low+1)); else low=0; fi
  sleep 60; waited=$((waited+60))
done
echo "--- GPU free (low=$low, waited=${waited}s) @ $(date -u) ---"

# ---- opportunistically apply the autotuner new-knob registration so the sweep
#      explores the NEW perf levers (ViT-B1 sub-tile, decoder ring depth/pipe),
#      not just the converged knobs. Guarded: apply only if it applies clean AND
#      --self-test still passes; else revert + sweep the existing knobs only. ----
if [ -f .perf/autotuner_new_knobs.diff ]; then
  if git apply --check .perf/autotuner_new_knobs.diff 2>/dev/null && git apply .perf/autotuner_new_knobs.diff 2>/dev/null; then
    if CUDA_VISIBLE_DEVICES="" timeout 10m python -m grokking_optimizers.compile --self-test >/dev/null 2>&1; then
      echo "--- applied autotuner_new_knobs.diff (sweep WILL explore the new perf levers) ---"
    else
      git checkout -- grokking_optimizers/compile.py grokking_optimizers/_tuned_inject.py 2>/dev/null
      echo "--- autotuner_new_knobs.diff self-test FAILED -> reverted; sweeping existing knobs only ---"
    fi
  else
    echo "--- autotuner_new_knobs.diff did not apply cleanly -> skip; sweeping existing knobs only ---"
  fi
fi

# Enable the autotuner (operator kill-switch; removed for this real tuning run).
rm -f .STOP_TUNING && echo "--- removed .STOP_TUNING (autotuner enabled) ---"

# ---- 1. Decoder RE-PROFILE @ the shipped 618ms PIPE=1/STAGES=4 build ----
# Gate-free diagnostic: re-grounds the STALE relevance gate (the 56.5% fwd/dX +
# 19.2% B1 shares were measured at the superseded 920ms build). Read the fresh
# phase table from these on return.
echo "--- decoder re-profile (coarse 5-slot) @ $(date -u) ---"
timeout 30m python tuning/decoder_bench.py --profile --d 2048 --B 16384 > .perf/batch/decoder_reprofile_coarse.txt 2>&1 || echo "  reprofile-coarse rc=$?"
echo "--- decoder re-profile (--fwd-fine) @ $(date -u) ---"
timeout 30m python tuning/decoder_bench.py --profile --fwd-fine --d 2048 --B 16384 > .perf/batch/decoder_reprofile_fwdfine.txt 2>&1 || echo "  reprofile-fwdfine rc=$?"

# ---- 2. AUTOTUNER SWEEP across the 33 cells (the long autonomous tail) ----
# compile.py JIT-builds + fp64-gates + times the flag/macro/structural-variant
# space per cell (Optuna-TPE + learned cost-model pruning, RESUMABLE), writing
# winners to grokking_optimizers/_kernel_tuned.json + populating the cost-model
# cache. ~24m/cell x 33 ~= 13h. NOTE on review: the in-loop correctness gate
# runs at d=128 (RG4) — winners must be RE-GATED at d=2048 before shipping; this
# run RECORDS them for that review, it does not ship.
# NOTE: mamba3 is SKIPPED here — its staged-opt scratch is un-gated (BUG-04, scale_correctness_audit.json)
# so it OOMs at the d=2048 autotune timing (~199 GiB) AND can't place at d=2048 (M1). Run mamba autotune
# on return AFTER the byte-identical kMbStagedOptScratch gate (BUG-04) + M1 land. Decoder/ViT gate it at
# bench width, so they autotune cleanly at d=2048.
MODELS="transformer_decoder vit"
# adamw first (the production default / highest value), then the rest.
OPTS="adamw lion grokfast grokadamw prodigy muon neuralgrok looksam supergrok11 supergrok15 supergrok2"
for model in $MODELS; do
  for opt in $OPTS; do
    echo "--- autotune  $model / $opt  @ $(date -u) ---"
    timeout 32m python -m grokking_optimizers.compile -M "$model" -O "$opt" --arch sm_90 --jit-only \
        > ".perf/batch/autotune_${model}_${opt}.log" 2>&1
    echo "    -> rc=$? ; winners now in grokking_optimizers/_kernel_tuned.json"
  done
done

echo "==================== FRONT-LOAD DONE $(date -u) ===================="
echo "DONE $(date -u)" > .perf/batch/DONE
