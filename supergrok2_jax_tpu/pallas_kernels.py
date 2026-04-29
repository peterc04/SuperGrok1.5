"""Compatibility shim — pallas kernels moved to ``csrc/kernels/tpu/``.

In the all-specialized refactor, the Pallas kernels were colocated with the
PyTorch arch-specific kernels under ``csrc/kernels/tpu/{v5p,v6e}/``. The
shared implementation lives in ``csrc/kernels/tpu/_pallas_kernels.py``.

This module re-exports the public surface for backwards compatibility with
imports of the form ``from supergrok2_jax_tpu.pallas_kernels import ...``.
New code should import from ``csrc.kernels.tpu`` (and pick a per-version
submodule via ``csrc.kernels.tpu.get_kernels()``).
"""

# Re-export everything that was previously importable from this module.
# The shared implementation is at csrc/kernels/tpu/_pallas_kernels.py.
import sys
from pathlib import Path

# Make csrc/kernels/tpu importable as a top-level package without forcing
# users to pip-install csrc/. The repo root is the parent of this file's
# parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from csrc.kernels.tpu._pallas_kernels import *  # noqa: F401,F403
from csrc.kernels.tpu._pallas_kernels import _associative_combine, _HAS_PALLAS  # noqa: F401

# New canonical entry point: pick the right per-version kernel module.
from csrc.kernels.tpu import get_kernels, detect_tpu_version  # noqa: F401
