#!/bin/bash
# Baseline timing chain (one GPU process at a time, sequential).
set -e
cd /workspace/SuperGrok1.5
source .regpressure/env.sh
export MAX_JOBS=24
TAG="${1:?tag required (e.g. BASE, P1, ...)}"
MODELS="${2:-decoder,vit,mamba}"
BENCHES="${3:-decoder,vit,mamba}"
IFS=',' read -ra MS <<< "$MODELS"
for m in "${MS[@]}"; do
  echo "=== prodtime $m [$TAG] ==="
  python .regpressure/gpu/prodtime.py --models "$m" --reps 3 \
      --out ".regpressure/gpu/prod_${m}_${TAG}.json"
done
IFS=',' read -ra BS <<< "$BENCHES"
for b in "${BS[@]}"; do
  case "$b" in
    decoder) B=4096;  SCRIPT=tuning/decoder_bench.py ;;
    vit)     B=16384; SCRIPT=tuning/vit_bench.py ;;
    mamba)   B=4096;  SCRIPT=tuning/mamba_bench.py ;;
  esac
  echo "=== bench $b d=1024 B=$B [$TAG] ==="
  python "$SCRIPT" --d 1024 --B "$B" --reps 5 --iters 10 \
      2>&1 | tee ".regpressure/gpu/bench_${b}_d1024_${TAG}.log"
done
echo "CHAIN_DONE [$TAG]"
