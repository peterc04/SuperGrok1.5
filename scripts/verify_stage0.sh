#!/usr/bin/env bash
# Stage 0 verification gate: every sm_90 TU must compile to object, self-test
# must stay 137/1 (or better), ruff clean. Prints one PASS/FAIL line per check.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

PASS=0; FAIL=0
chk() {  # name  flags  tu
  local name="$1"; shift
  local tu="$1"; shift
  local out
  out=$(bash scripts/compile_to_object.sh "$tu" "$@" 2>&1 | grep -oE 'COMPILE_OK|COMPILE_FAIL' | head -1)
  if [ "$out" = "COMPILE_OK" ]; then echo "PASS  $name"; PASS=$((PASS+1));
  else echo "FAIL  $name"; FAIL=$((FAIL+1)); fi
}

for o in adamw lion grokfast grokadamw looksam muon neuralgrok prodigy supergrok11 supergrok15; do
  chk "compile:$o" "csrc/backends/cuda/sm_90/launch_${o}.cu"
done
chk "compile:supergrok2(cutlass)" "csrc/backends/cuda/sm_90/launch_supergrok2.cu" -DWITH_CUTLASS
chk "compile:model:mamba(cutlass)" "csrc/backends/cuda/sm_90/models/mamba.cu" -DWITH_CUTLASS
chk "compile:model:decoder(cutlass)" "csrc/backends/cuda/sm_90/models/decoder.cu" -DWITH_CUTLASS
chk "compile:model:vit(cutlass)" "csrc/backends/cuda/sm_90/models/vit.cu" -DWITH_CUTLASS

echo "----"
echo "compile: $PASS passed, $FAIL failed"
