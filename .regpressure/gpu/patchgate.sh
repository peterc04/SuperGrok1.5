#!/bin/bash
# patchgate.sh TAG MODELS BENCHES  — rebuild + wiring + tailgate + retime.
# (Patch application + any seam edits happen BEFORE this script is invoked.)
set -e
cd /workspace/SuperGrok1.5
source .regpressure/env.sh
TAG="${1:?tag}"
MODELS="${2:-decoder}"
BENCHES="${3:-decoder}"

echo "=== [$TAG] build (_ops incremental, full parallelism) ==="
./build.sh > ".regpressure/gpu/buildrun_${TAG}.log" 2>&1 || { echo "BUILD_FAIL"; tail -40 ".regpressure/gpu/buildrun_${TAG}.log"; exit 2; }
cp build.log ".regpressure/gpu/build_${TAG}.log" 2>/dev/null || true
grep -E "done in" ".regpressure/gpu/buildrun_${TAG}.log" | tail -1

echo "=== [$TAG] wiring_check --require-all ==="
python wiring_check.py --require-all > ".regpressure/gpu/wiring_${TAG}.log" 2>&1 \
  && echo "WIRING_OK" || { echo "WIRING_FAIL"; tail -15 ".regpressure/gpu/wiring_${TAG}.log"; exit 3; }

echo "=== [$TAG] full 33-cell tail gate ==="
python -m pytest tests/hw/test_l3tc_tail_gate.py -m hw -q -s > ".regpressure/gpu/tailgate_${TAG}.log" 2>&1 \
  && echo "TAILGATE_OK $(grep -E 'passed' ".regpressure/gpu/tailgate_${TAG}.log" | tail -1)" \
  || { echo "TAILGATE_FAIL"; grep -E "FAILED|failed|Error" ".regpressure/gpu/tailgate_${TAG}.log" | head -20; exit 4; }

export MAX_JOBS=24
IFS=',' read -ra MS <<< "$MODELS"
for m in "${MS[@]}"; do
  [ -z "$m" ] && continue
  echo "=== [$TAG] prodtime $m ==="
  python .regpressure/gpu/prodtime.py --models "$m" --reps 3 \
      --out ".regpressure/gpu/prod_${m}_${TAG}.json"
done
IFS=',' read -ra BS <<< "$BENCHES"
for b in "${BS[@]}"; do
  [ -z "$b" ] && continue
  case "$b" in
    decoder)  B=4096;  D=1024; SCRIPT=tuning/decoder_bench.py ;;
    vit)      B=16384; D=128;  SCRIPT=tuning/vit_bench.py ;;
    mamba)    B=4096;  D=128;  SCRIPT=tuning/mamba_bench.py ;;
  esac
  echo "=== [$TAG] bench $b d=$D B=$B ==="
  python "$SCRIPT" --d "$D" --B "$B" --reps 5 --iters 10 \
      > ".regpressure/gpu/bench_${b}_d${D}_${TAG}.log" 2>&1 \
      && grep "wall/step" ".regpressure/gpu/bench_${b}_d${D}_${TAG}.log" \
      || { echo "BENCH_FAIL $b"; tail -5 ".regpressure/gpu/bench_${b}_d${D}_${TAG}.log"; exit 5; }
done
echo "PATCHGATE_DONE [$TAG]"
