#!/usr/bin/env bash
# One-shot restore of the full working env on a FRESH runpod instance.
#
# The /workspace NFS volume is PERSISTENT; the container root (/usr, pip
# site-packages, /usr/local/bin, /dev/shm) is EPHEMERAL and wiped on restart.
# This script rebuilds the ephemeral pieces FROM the volume so a restart is one
# command back to working. Idempotent: warm-volume steps are no-ops on re-run.
#
#   usage:  bash /workspace/SuperGrok1.5/scripts/bootstrap_env.sh
#           then open a new shell (or: source ~/.supergrok_env)
set -euo pipefail

VENV=/workspace/venv
PROJ=/workspace/SuperGrok1.5
SCCACHE_BIN_DIR=/workspace/.local/bin
export PIP_CACHE_DIR=/workspace/.cache/pip
mkdir -p "$PIP_CACHE_DIR" "$SCCACHE_BIN_DIR"

# (1) Persistent venv on the volume, inheriting the image's torch/CUDA/numpy.
#     --system-site-packages keeps the image torch 2.4.1+cu124 that _ops.so is
#     ABI-linked against; new deps install into the venv (on the volume).
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "bootstrap: creating venv (--system-site-packages) at $VENV"
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

# (2) Sanity: inherited torch MUST be the image's CUDA build, untouched.
python - <<'PY'
import torch, numpy
assert torch.__version__.startswith("2.4.1+cu124"), torch.__version__
assert numpy.__version__ == "1.26.3", numpy.__version__
print("inherited torch/cuda/numpy OK:", torch.__version__, "| cuda", torch.cuda.is_available(), "| numpy", numpy.__version__)
PY

# (3) Reinstall pip-added deps INTO the venv (fast: warm pip cache on the volume).
bash "$PROJ/scripts/install_deps.sh"

# (4) sccache binary on the volume (compile.py's _sccache_env wraps nvcc + host CC).
if ! command -v sccache >/dev/null 2>&1 && [[ ! -x "$SCCACHE_BIN_DIR/sccache" ]]; then
  echo "bootstrap: fetching sccache to $SCCACHE_BIN_DIR"
  ver=v0.8.2; pkg=sccache-$ver-x86_64-unknown-linux-musl
  curl -fsSL "https://github.com/mozilla/sccache/releases/download/$ver/$pkg.tar.gz" \
    | tar xz -C /tmp && install -m755 "/tmp/$pkg/sccache" "$SCCACHE_BIN_DIR/sccache"
fi
export PATH="$SCCACHE_BIN_DIR:$PATH"

# (5) Build/runtime env vars (recreates ephemeral /dev/shm scratch dirs).
#     SCCACHE_DIR stays on /dev/shm by design (fast; sccache-on-NFS locking is
#     unsafe). Warm incremental rebuilds come from TORCH_EXTENSIONS_DIR (on volume).
source "$PROJ/.regpressure/env.sh"

# (6) Ensure the extension imports; rebuild in-place ONLY if the .so is gone.
#     (build/ + _ops.so persist on the volume, so this is normally a no-op.)
cd "$PROJ"
if ! python -c "from grokking_optimizers import _ops" 2>/dev/null; then
  echo "bootstrap: _ops missing -> rebuilding in-place (pip install -e .)"
  pip install -e . --no-build-isolation
fi

# (7) Convenience: auto-activate in new shells (re-established each bootstrap;
#     ~/.bashrc is ephemeral, so the authoritative copy is this script on the volume).
cat > ~/.supergrok_env <<EOF
source $VENV/bin/activate
export PATH="$SCCACHE_BIN_DIR:\$PATH"
source $PROJ/.regpressure/env.sh
EOF
grep -q supergrok_env ~/.bashrc 2>/dev/null || echo "source ~/.supergrok_env" >> ~/.bashrc

echo "bootstrap complete: venv active, deps present, sccache on PATH, _ops importable."
