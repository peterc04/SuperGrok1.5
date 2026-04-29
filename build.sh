#!/usr/bin/env bash
# build.sh — top-level build wrapper for grokking-optimizers.
#
# Usage:
#   ./build.sh                    # default: plain release build
#   ./build.sh --autotune         # two-pass build (stub -> tune.py -> rebuild)
#   ./build.sh --no-autotune      # explicit single-pass (default behavior)
#   ./build.sh --debug            # -G -O0 -lineinfo, fast-math off (CUDA_DEBUG=1)
#   ./build.sh --profile          # release build, then ncu --set full
#
# Env:
#   MAX_JOBS   parallel ninja jobs (default: $(nproc))
#
# The ninja invocation by torch's BuildExtension prints "[N/M] ..." progress
# lines on stderr. We tee everything to build.log and post-process with a
# tqdm filter to render percent + ETA. Non-progress lines pass through to
# the terminal as-is.

set -euo pipefail

cd "$(dirname "$0")"

AUTOTUNE=0
DEBUG=0
PROFILE=0

for arg in "$@"; do
  case "$arg" in
    --autotune)    AUTOTUNE=1 ;;
    --no-autotune) AUTOTUNE=0 ;;
    --debug)       DEBUG=1 ;;
    --profile)     PROFILE=1 ;;
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *)
      echo "build.sh: unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

: "${MAX_JOBS:=$(nproc)}"
export MAX_JOBS
echo "build.sh: MAX_JOBS=${MAX_JOBS}"

if [[ "$DEBUG" == "1" ]]; then
  export CUDA_DEBUG=1
  echo "build.sh: CUDA_DEBUG=1 (debug build, fast-math disabled)"
else
  unset CUDA_DEBUG || true
fi

# tqdm-aware progress filter for ninja's "[N/M] ..." lines.
# Non-progress lines are passed through unchanged. If tqdm or python is
# missing we fall back to plain tee.
PROGRESS_FILTER=$(cat <<'PYEOF'
import re, sys
try:
    from tqdm import tqdm
except ImportError:
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
    sys.exit(0)

pat = re.compile(r"^\s*\[(\d+)/(\d+)\]")
bar = None
last_total = None
for line in sys.stdin:
    m = pat.match(line)
    if m:
        cur, total = int(m.group(1)), int(m.group(2))
        if bar is None or total != last_total:
            if bar is not None:
                bar.close()
            bar = tqdm(total=total, unit="obj", dynamic_ncols=True,
                       desc="ninja", leave=True)
            last_total = total
            bar.n = cur
        else:
            bar.n = cur
        bar.refresh()
        # also keep the raw line in build.log via tee upstream
    else:
        if bar is not None:
            bar.clear()
        sys.stdout.write(line)
        sys.stdout.flush()
        if bar is not None:
            bar.refresh()
if bar is not None:
    bar.close()
PYEOF
)

run_build() {
  local label="$1"
  echo "build.sh: ${label}"
  # tee raw output to build.log; pipe a copy through the tqdm filter.
  set +e
  pip install -e . --no-build-isolation -v 2>&1 \
    | tee build.log \
    | python -c "$PROGRESS_FILTER"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" != "0" ]]; then
    echo "build.sh: pip install failed (rc=$rc); see build.log" >&2
    exit "$rc"
  fi
}

START=$SECONDS

if [[ "$AUTOTUNE" == "1" ]]; then
  echo "build.sh: --autotune two-pass build"
  AUTOTUNE_PASS=1 run_build "pass 1: stub configs"
  echo "build.sh: running autotune/tune.py to write tuned_configs.h"
  python autotune/tune.py
  unset AUTOTUNE_PASS
  run_build "pass 2: tuned configs"
else
  run_build "single-pass build"
fi

if [[ "$PROFILE" == "1" ]]; then
  echo "build.sh: --profile running ncu"
  mkdir -p profile_output
  ncu --set full --target-processes all -o profile_output/baseline \
      python benchmarks/profile_smoke.py
fi

ELAPSED=$((SECONDS - START))
printf 'build.sh: done in %dm%02ds\n' $((ELAPSED/60)) $((ELAPSED%60))
