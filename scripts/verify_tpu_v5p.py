#!/usr/bin/env python3
"""Standalone verification of the TPU v5p JAX/Pallas bridge.

Performs five checks and exits 0 on success, non-zero otherwise:

    1. Print ``jax`` and ``jaxlib`` versions.
    2. Print ``jax.devices()`` and the version returned by
       ``csrc.kernels.tpu._pallas_kernels.detect_tpu_version()``.
    3. Import every TPU kernel/model module and print the number of
       public symbols loaded from each (as a structural sanity check).
    4. Run a single small forward pass through
       ``mamba3_scan_pallas_tile128`` (or its pure-JAX fallback when
       Pallas isn't available) with deterministic input and print the
       output shape plus the first three values of the bias output.
    5. Cross-check that the v5p registry's model kernels are the very
       objects exported by ``_pallas_models`` (re-export, not wrapper).

Run with::

    python scripts/verify_tpu_v5p.py

The script is safe to run on hosts without a TPU: it falls back to the
pure-JAX path and still exits 0 when the bridge is structurally healthy.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import Any

# Make the repo root importable when this script is launched directly:
#   python scripts/verify_tpu_v5p.py
# The repo root is the parent of the directory containing this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_FAIL = "FAIL"
_OK = "ok"


def _print_header(title: str) -> None:
    print()
    print(f"── {title} " + "─" * max(0, 60 - len(title) - 4))


def _check(label: str, fn) -> bool:
    """Run ``fn``; print ``ok`` / ``FAIL``; return True on success."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - top-level reporter
        print(f"  [{_FAIL}] {label}: {exc}")
        traceback.print_exc()
        return False
    print(f"  [{_OK}] {label}")
    return True


def step_1_versions(state: dict[str, Any]) -> None:
    """Print ``jax`` / ``jaxlib`` versions."""
    import jax
    import jaxlib

    print(f"  jax     = {jax.__version__}")
    print(f"  jaxlib  = {jaxlib.__version__}")
    state["jax"] = jax


def step_2_devices(state: dict[str, Any]) -> None:
    """Print the visible devices and the detected TPU version tag."""
    jax = state["jax"]
    devices = jax.devices()
    print(f"  jax.devices() = {devices}")
    print(f"  default_backend = {jax.default_backend()}")

    pallas_kernels = importlib.import_module("csrc.kernels.tpu._pallas_kernels")
    state["pallas_kernels"] = pallas_kernels

    version = pallas_kernels.detect_tpu_version()
    print(f"  detect_tpu_version() = {version!r}")
    if version not in {"v4", "v5e", "v5p", "v6e"}:
        raise AssertionError(
            f"detect_tpu_version() returned unknown tag {version!r}"
        )


def step_3_symbol_counts(state: dict[str, Any]) -> None:
    """Import every TPU kernel/model module and print their symbol counts."""
    targets = [
        "csrc.kernels.tpu._pallas_kernels",
        "csrc.kernels.tpu._pallas_models",
        "csrc.kernels.tpu.v5p",
        "csrc.device.models.tpu_v5p.transformer_tpu_v5p",
        "csrc.device.models.tpu_v5p.vit_tpu_v5p",
        "csrc.device.models.tpu_v5p.mamba_tpu_v5p",
    ]
    for name in targets:
        mod = importlib.import_module(name)
        public = [s for s in dir(mod) if not s.startswith("_")]
        # If the module declares __all__, use that as ground truth.
        all_list = getattr(mod, "__all__", None)
        if all_list is not None:
            print(f"  {name}: {len(public)} public symbols "
                  f"(__all__={len(all_list)})")
        else:
            print(f"  {name}: {len(public)} public symbols")
    # Stash for later steps.
    state["pallas_models"] = importlib.import_module(
        "csrc.kernels.tpu._pallas_models"
    )
    state["v5p"] = importlib.import_module("csrc.kernels.tpu.v5p")


def step_4_scan_forward(state: dict[str, Any]) -> None:
    """Run a small deterministic forward pass through the tile-128 scan."""
    import jax.numpy as jnp

    pallas_kernels = state["pallas_kernels"]

    # Deterministic input: a sequence of slightly perturbed identity
    # affine matrices and zero biases. Result shape is invariant to the
    # exact perturbation; we just need a non-trivial-but-stable input.
    N = 64
    eye = jnp.eye(2, dtype=jnp.float32)[None]
    Ms = jnp.tile(eye, (N, 1, 1))
    # Use a deterministic ramp on the bias to make the first three output
    # values informative without depending on PRNG.
    bs = jnp.stack([jnp.arange(N, dtype=jnp.float32),
                    jnp.zeros(N, dtype=jnp.float32)], axis=-1)

    Ms_out, bs_out = pallas_kernels.mamba3_scan_pallas_tile128(Ms, bs)
    print(f"  Ms_out.shape = {tuple(Ms_out.shape)}, dtype={Ms_out.dtype}")
    print(f"  bs_out.shape = {tuple(bs_out.shape)}, dtype={bs_out.dtype}")
    first3 = bs_out[:3].tolist()
    print(f"  bs_out[:3] = {first3}")

    if Ms_out.shape != (N, 2, 2) or bs_out.shape != (N, 2):
        raise AssertionError(
            f"Unexpected output shapes: Ms_out={Ms_out.shape} "
            f"bs_out={bs_out.shape}"
        )


def step_5_registry_identity(state: dict[str, Any]) -> None:
    """Verify v5p model kernels are the _pallas_models objects."""
    v5p = state["v5p"]
    pm = state["pallas_models"]
    expected = {
        "attention_forward", "attention_backward",
        "decoder_forward", "decoder_backward",
        "vit_forward", "vit_backward", "vit_patch_project",
        "mamba_forward", "mamba_backward",
        "mamba_layer_forward", "mamba_selective_scan",
    }
    have = set(v5p._MODEL_KERNELS)
    if have != expected:
        raise AssertionError(
            f"v5p _MODEL_KERNELS keys differ.\n"
            f"  expected: {sorted(expected)}\n"
            f"  found:    {sorted(have)}"
        )
    for name in expected:
        if v5p._MODEL_KERNELS[name] is not getattr(pm, name):
            raise AssertionError(
                f"v5p _MODEL_KERNELS[{name!r}] is not "
                f"_pallas_models.{name} (re-export broken)"
            )


def main() -> int:
    state: dict[str, Any] = {}
    all_ok = True

    print("SuperGrok1.5 :: TPU v5p JAX bridge verification")

    _print_header("1. JAX / jaxlib versions")
    all_ok &= _check("versions", lambda: step_1_versions(state))

    if not all_ok:
        # Without JAX nothing else can run.
        return 1

    _print_header("2. Devices + detect_tpu_version()")
    all_ok &= _check("devices", lambda: step_2_devices(state))

    _print_header("3. Module imports + symbol counts")
    all_ok &= _check("imports", lambda: step_3_symbol_counts(state))

    _print_header("4. mamba3_scan_pallas_tile128 forward pass")
    all_ok &= _check("scan", lambda: step_4_scan_forward(state))

    _print_header("5. v5p registry identity")
    all_ok &= _check("registry", lambda: step_5_registry_identity(state))

    print()
    if all_ok:
        print("All checks passed.")
        return 0
    print("One or more checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
