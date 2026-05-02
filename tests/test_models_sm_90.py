#!/usr/bin/env python3
"""Smoke tests for the sm_90 model bindings (`_ops.models.*`).

This file does NOT exercise the kernel itself — that requires a real
Hopper GPU and is covered elsewhere. Here we only verify that:

  1. ``grokking_optimizers._ops`` can be imported.
  2. The ``_ops.models`` submodule exists.
  3. The expected entry points are registered (decoder_forward,
     decoder_backward, vit_forward, vit_backward, vit_patch_project,
     mamba_forward, mamba_backward, mamba_layer_forward,
     mamba_selective_scan_forward, mamba_selective_scan_backward,
     plus the *_attention_* component test surfaces).
  4. Each binding is callable (i.e. a pybind11 builtin function),
     not a stub or a placeholder ``None``.

Skipped when CUDA is not available — the C++ extension that owns these
bindings is only built on hosts with a CUDA toolkit.

Usage:
    pytest tests/test_models_sm_90.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Skip the entire module if CUDA isn't present. The bindings live in the
# compiled extension, which is only built when the CUDA / HIP toolchain
# is available; CPU-only CI runs do not have the shared object.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="sm_90 model bindings require a CUDA-capable host build of _ops",
)


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def ops():
    """Load the compiled extension module via the strict loader."""
    from grokking_optimizers._ops_loader import get_ops
    try:
        return get_ops()
    except RuntimeError as exc:
        pytest.skip(f"_ops extension not built: {exc}")


# ═══════════════════════════════════════════════════════════════════════
#  Namespace presence
# ═══════════════════════════════════════════════════════════════════════


def test_ops_module_importable(ops):
    """The compiled extension itself must import cleanly."""
    assert ops is not None


def test_ops_models_namespace_exists(ops):
    """`_ops.models` is the submodule registered by models_module.cpp."""
    assert hasattr(ops, "models"), (
        "_ops.models submodule is missing — register_model_bindings() "
        "is not being called from csrc/bindings/module.cpp"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Entry-point registration
# ═══════════════════════════════════════════════════════════════════════

# Mirrors csrc/bindings/models_module.cpp::register_model_bindings.
EXPECTED_DECODER_ENTRIES = (
    "decoder_forward",
    "decoder_backward",
    "decoder_attention_forward",
    "decoder_attention_backward",
)

EXPECTED_VIT_ENTRIES = (
    "vit_forward",
    "vit_backward",
    "vit_attention_forward",
    "vit_attention_backward",
    "vit_patch_project",
)

EXPECTED_MAMBA_ENTRIES = (
    "mamba_forward",
    "mamba_backward",
    "mamba_layer_forward",
    "mamba_selective_scan_forward",
    "mamba_selective_scan_backward",
)

ALL_EXPECTED = (
    EXPECTED_DECODER_ENTRIES
    + EXPECTED_VIT_ENTRIES
    + EXPECTED_MAMBA_ENTRIES
)


@pytest.mark.parametrize("name", EXPECTED_DECODER_ENTRIES)
def test_decoder_entry_registered(ops, name):
    """Each Decoder Transformer entry point is bound."""
    assert hasattr(ops.models, name), (
        f"_ops.models.{name} is not registered (Decoder Transformer)"
    )


@pytest.mark.parametrize("name", EXPECTED_VIT_ENTRIES)
def test_vit_entry_registered(ops, name):
    """Each ViT entry point is bound."""
    assert hasattr(ops.models, name), (
        f"_ops.models.{name} is not registered (Vision Transformer)"
    )


@pytest.mark.parametrize("name", EXPECTED_MAMBA_ENTRIES)
def test_mamba_entry_registered(ops, name):
    """Each Mamba entry point is bound."""
    assert hasattr(ops.models, name), (
        f"_ops.models.{name} is not registered (Mamba SSM)"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Callability
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", ALL_EXPECTED)
def test_entry_is_callable(ops, name):
    """A registered binding must be a Python callable, not a stub.

    pybind11 builtin functions have a real C-level callable; we don't
    invoke the kernel here because that requires correctly shaped GPU
    tensors — this is a smoke test, not a parity test.
    """
    fn = getattr(ops.models, name, None)
    assert fn is not None, f"_ops.models.{name} is missing"
    assert callable(fn), (
        f"_ops.models.{name} is registered but not callable "
        f"(got {type(fn).__name__})"
    )


def test_namespace_no_unexpected_entries(ops):
    """Every public entry on _ops.models is one of the expected ones.

    Regression guard: catches typos in models_module.cpp where a new
    binding ships under the wrong name. We allow dunder attributes
    (pybind11 sets a bunch of those) and pure submodule attrs.
    """
    public_attrs = {
        a for a in dir(ops.models)
        if not a.startswith("_")
    }
    unexpected = public_attrs - set(ALL_EXPECTED)
    # Future model component test surfaces may extend this list — fail
    # only if a registered entry is wholly outside the expected groups.
    assert not unexpected, (
        f"Unexpected entries on _ops.models: {sorted(unexpected)}. "
        f"Update EXPECTED_*_ENTRIES in this file or remove the binding."
    )


# ═══════════════════════════════════════════════════════════════════════
#  Doc-string sanity
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", ALL_EXPECTED)
def test_entry_has_docstring(ops, name):
    """Each binding registers with a non-empty docstring (helps `help()`)."""
    fn = getattr(ops.models, name)
    doc = (getattr(fn, "__doc__", "") or "").strip()
    assert doc, f"_ops.models.{name} has no docstring — see models_module.cpp"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
