#!/usr/bin/env bash
# build.sh — top-level build wrapper for grokking-optimizers.
#
# Usage:
#   ./build.sh                    # default: plain release build
#   ./build.sh --autotune         # two-pass build (stub -> tune.py -> rebuild)
#   ./build.sh --no-autotune      # explicit single-pass (default behavior)
#   ./build.sh --debug            # -G -O0 -lineinfo, fast-math off (CUDA_DEBUG=1)
#   ./build.sh --profile          # release build, then ncu --set full
#   ./build.sh --package          # build + stage redistributable dist/ tree
#   ./build.sh --package-tarball  # --package + supergrok2-VERSION-SHA.tar.gz
#
# Env:
#   MAX_JOBS   parallel ninja jobs (default: $(nproc))
#
# --package combines well with --autotune, --debug, --no-autotune. Combining
# with --debug skips the strip step so the extensions stay debuggable.
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
PACKAGE=0
TARBALL=0

for arg in "$@"; do
  case "$arg" in
    --autotune)        AUTOTUNE=1 ;;
    --no-autotune)     AUTOTUNE=0 ;;
    --debug)           DEBUG=1 ;;
    --profile)         PROFILE=1 ;;
    --package)         PACKAGE=1 ;;
    --package-tarball) PACKAGE=1; TARBALL=1 ;;
    -h|--help)
      sed -n '2,21p' "$0"
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

# ---------------------------------------------------------------------------
# --package: stage a redistributable dist/ tree mirroring the runtime layout.
#
# Output layout (matches what Python sees after `pip install -e .`):
#   dist/
#   ├── grokking_optimizers/        Python sources + compiled _ops.<...>.so
#   ├── csrc/backends/pallas/       Pallas/JAX TPU kernels (Python)
#   ├── csrc/common/tuned_configs.h post-autotune values (if AUTOTUNE ran)
#   ├── LICENSE
#   ├── pyproject.toml              build manifest (version pin)
#   ├── README.md
#   └── INSTALL.md                  auto-generated install guide (if template)
#
# Stage-2 fix (Phase 8): earlier revisions staged `supergrok2_jax_tpu/` and
# `csrc/kernels/tpu/` — neither path exists in this tree. The JAX/TPU Pallas
# kernels actually live under `csrc/backends/pallas/`; tuned_configs.h is
# only present after `--autotune`. Staging is now guarded on existence so a
# plain `--package` no longer creates empty directories or skips real files.
#
# Compiled extensions: torch's BuildExtension produces a single multi-arch
# fatbin .so (one file with cubin/PTX for every arch listed in setup.py's
# -gencode + --offload-arch flags). The driver picks the right cubin at
# load time. If WITH_CUTLASS=1 was set, that .so links CUTLASS in.
# Debug symbols are stripped unless --debug was set.
# ---------------------------------------------------------------------------
package_dist() {
  local dist_dir="dist"
  echo "build.sh: --package staging into ${dist_dir}/"
  rm -rf "${dist_dir}"
  mkdir -p "${dist_dir}/grokking_optimizers"

  # Python sources (preserve subpackage layout, including kernel headers).
  cp -r grokking_optimizers/*.py "${dist_dir}/grokking_optimizers/"
  # Per-arch in-package kernel headers (*.cuh / *.hip.hpp) and any package
  # data live under grokking_optimizers/kernels/ — copy the whole subpackage.
  if [[ -d grokking_optimizers/kernels ]]; then
    cp -r grokking_optimizers/kernels "${dist_dir}/grokking_optimizers/"
  fi
  if [[ -d grokking_optimizers/optimizers ]]; then
    cp -r grokking_optimizers/optimizers "${dist_dir}/grokking_optimizers/"
  fi

  # JAX/TPU Pallas kernels (Python). These live under csrc/backends/pallas/,
  # not the long-gone csrc/kernels/tpu/ or supergrok2_jax_tpu/ paths.
  if [[ -d csrc/backends/pallas ]]; then
    mkdir -p "${dist_dir}/csrc/backends/pallas"
    cp -r csrc/backends/pallas/* "${dist_dir}/csrc/backends/pallas/" 2>/dev/null || true
  fi

  # Math manifest consumed by the autotuner / kernel emitter.
  if [[ -f scripts/optimizer_math_manifest.json ]]; then
    mkdir -p "${dist_dir}/scripts"
    cp scripts/optimizer_math_manifest.json "${dist_dir}/scripts/"
  fi

  # Compiled extension(s). pip install -e . places the .so directly under
  # grokking_optimizers/ (in-place editable install).
  shopt -s nullglob
  local ops_files=( grokking_optimizers/_ops*.so grokking_optimizers/_ops*.pyd )
  shopt -u nullglob
  if [[ ${#ops_files[@]} -eq 0 ]]; then
    echo "build.sh: --package: no compiled _ops.* extension found" >&2
    echo "  Expected grokking_optimizers/_ops*.{so,pyd} after a successful build." >&2
    exit 3
  fi
  for f in "${ops_files[@]}"; do
    cp "$f" "${dist_dir}/grokking_optimizers/"
  done
  echo "build.sh: --package staged ${#ops_files[@]} _ops extension(s)"

  # Post-autotune tuned_configs.h (only present after `--autotune`; on a
  # plain build it does not exist and is simply skipped — the consumer can
  # re-tune later). Guarded so we never mkdir an empty csrc/common/.
  if [[ -f csrc/common/tuned_configs.h ]]; then
    mkdir -p "${dist_dir}/csrc/common"
    cp csrc/common/tuned_configs.h "${dist_dir}/csrc/common/"
  fi

  # Top-level metadata. (REFRESH.md does not exist in this tree; LICENSE is
  # required by the package metadata, so it must ship.)
  for f in pyproject.toml README.md LICENSE; do
    [[ -f "$f" ]] && cp "$f" "${dist_dir}/"
  done

  # Strip debug symbols unless --debug.
  if [[ "$DEBUG" != "1" ]]; then
    for f in "${dist_dir}/grokking_optimizers"/_ops*.so; do
      [[ -f "$f" ]] || continue
      strip --strip-unneeded "$f" 2>/dev/null || true
    done
    echo "build.sh: --package stripped debug symbols (run with --debug to keep)"
  else
    echo "build.sh: --package kept debug symbols (--debug was set)"
  fi

  # Generate INSTALL.md from the template.
  if [[ -f scripts/INSTALL.md.template ]]; then
    cp scripts/INSTALL.md.template "${dist_dir}/INSTALL.md"
  fi

  # Size report.
  local total_bytes
  total_bytes=$(du -sb "${dist_dir}" | awk '{print $1}')
  local total_mib=$(( total_bytes / 1024 / 1024 ))
  echo "build.sh: --package complete. dist/ size: ${total_mib} MiB"
  du -sh "${dist_dir}/grokking_optimizers"/_ops*.so 2>/dev/null | while read -r sz f; do
    echo "  ${f}: ${sz}"
  done
}

package_tarball() {
  local sha
  sha=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  local version
  version=$(grep -E '^version *=' pyproject.toml | head -1 | sed 's/.*"\([^"]*\)".*/\1/')
  : "${version:=3.0.0}"
  local tarball="supergrok2-${version}-${sha}.tar.gz"
  echo "build.sh: --package-tarball creating ${tarball}"
  tar -czf "${tarball}" -C dist .
  local sz
  sz=$(du -sh "${tarball}" | awk '{print $1}')
  echo "build.sh: --package-tarball ${tarball} (${sz})"
}

if [[ "$PACKAGE" == "1" ]]; then
  package_dist
  if [[ "$TARBALL" == "1" ]]; then
    package_tarball
  fi
fi

ELAPSED=$((SECONDS - START))
printf 'build.sh: done in %dm%02ds\n' $((ELAPSED/60)) $((ELAPSED%60))
