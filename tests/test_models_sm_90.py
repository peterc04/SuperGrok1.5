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


# ═══════════════════════════════════════════════════════════════════════
#  Reference parity tests (kernel ↔ PyTorch oracle)
# ═══════════════════════════════════════════════════════════════════════
#
# These tests build a small reference model (pure PyTorch) from
# ``tests.reference.models.<model>_ref``, run a forward pass to
# capture the oracle output, then attempt to invoke the corresponding
# fused-kernel binding on the same input + the model's weights and
# assert ``allclose`` agreement.
#
# Why they may skip even with a GPU present:
# The kernel headers document a packed weight layout that is *not* a
# trivial flatten of the reference ``nn.Module``'s parameters. Until a
# packing helper exists in the binding (or a test-side packer is
# written from the kernel-header specs), the actual kernel call is
# guarded by ``pytest.skip("weight packing TBD")``. The structural
# part — model build, oracle forward, dtype/shape contract — runs and
# protects against drift in the reference.
#
# Tolerances:
#   BF16 → rtol=1e-3, atol=1e-3
#   FP32 → rtol=1e-5, atol=1e-5
# Match the kernel's documented numerical envelope.
# ═══════════════════════════════════════════════════════════════════════


# Per-dtype tolerances used by every parity test below. Keep in sync
# with the kernel's documented numerical envelope (see
# csrc/models/*.h). BF16 has ~7-bit mantissa so 1e-3 is roughly the
# tightest reasonable bound; FP32 must agree to ~1 ULP at this scale.
_TOLERANCES = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.bfloat16: dict(rtol=1e-3, atol=1e-3),
}


def _device_for_kernel():
    """Resolve the CUDA device the kernel will execute on.

    All parity tests run on cuda:0 — the bindings dispatch on the
    current device, and the test harness already skips when CUDA is
    not available.
    """
    return torch.device("cuda:0")


def _seeded(seed: int = 0):
    """Manual-seed both CPU and CUDA RNGs for reproducible inits."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Decoder parity ───────────────────────────────────────────────────


def test_decoder_forward_parity(ops):
    """Reference Decoder Transformer ↔ ``_ops.models.decoder_forward``.

    Builds a 2-layer, d=128, h=4, ntok=99, seq=4 model. Oracle runs
    on GPU in FP32. Skipped past oracle stage until weight packing is
    available — see module docstring above.
    """
    if not hasattr(ops.models, "decoder_forward"):
        pytest.skip("decoder_forward binding not registered")

    from tests.reference.models.decoder_ref import make_decoder

    _seeded(0)
    device = _device_for_kernel()
    model = make_decoder(nl=2, d=128, h=4, ntok=99, seq=4).to(device).eval()

    batch, seq = 2, 4
    x = torch.randint(0, 99, (batch, seq), device=device, dtype=torch.long)

    with torch.no_grad():
        ref_out = model(x)
    assert ref_out.shape == (batch, 99), (
        f"reference decoder produced unexpected shape {ref_out.shape}"
    )
    assert ref_out.dtype == torch.float32

    pytest.skip(
        "weight packing TBD — _ops.models.decoder_forward uses an opaque "
        "packed layout (see csrc/models/decoder_sm_90.h). Enable once a "
        "packer is wired up."
    )

    # ── live path (executes when the skip above is removed) ──────────
    # tol = _TOLERANCES[ref_out.dtype]
    # packed = _pack_decoder_weights(model)  # TODO: define packer
    # kernel_out = ops.models.decoder_forward(x, *packed)
    # assert kernel_out.shape == ref_out.shape
    # assert torch.allclose(kernel_out, ref_out, **tol), (
    #     f"decoder kernel disagrees with PyTorch oracle: "
    #     f"max |Δ| = {(kernel_out - ref_out).abs().max().item():.3e}"
    # )


# ── ViT parity ───────────────────────────────────────────────────────


def test_vit_forward_parity(ops):
    """Reference ViT ↔ ``_ops.models.vit_forward``.

    Builds a p=97, patch_dim=49, num_patches=16, d=128, h=4, nl=2
    model; oracle on GPU FP32. Skipped past oracle stage until weight
    packing is available.
    """
    if not hasattr(ops.models, "vit_forward"):
        pytest.skip("vit_forward binding not registered")

    from tests.reference.models.vit_ref import make_vit

    _seeded(0)
    device = _device_for_kernel()
    model = make_vit(p=97, patch_dim=49, num_patches=16,
                     d=128, h=4, nl=2).to(device).eval()

    batch, num_patches, patch_dim = 2, 16, 49
    x = torch.randn(batch, num_patches, patch_dim, device=device)

    with torch.no_grad():
        ref_out = model(x)
    assert ref_out.shape == (batch, 97), (
        f"reference ViT produced unexpected shape {ref_out.shape}"
    )
    assert ref_out.dtype == torch.float32

    pytest.skip(
        "weight packing TBD — _ops.models.vit_forward uses an opaque "
        "packed layout (see csrc/models/vit_sm_90.h). Enable once a "
        "packer is wired up."
    )

    # ── live path ─────────────────────────────────────────────────────
    # tol = _TOLERANCES[ref_out.dtype]
    # packed = _pack_vit_weights(model)  # TODO
    # kernel_out = ops.models.vit_forward(x, *packed)
    # assert torch.allclose(kernel_out, ref_out, **tol)


# ── Mamba parity ─────────────────────────────────────────────────────


def test_mamba_forward_parity(ops):
    """Reference Mamba SSM ↔ ``_ops.models.mamba_forward``.

    Builds a p=97, ntok=99, seq_len=8, d=128, nl=2 model; oracle on
    GPU FP32 with the Python selective-scan fallback. Skipped past
    oracle stage until weight packing is available.
    """
    if not hasattr(ops.models, "mamba_forward"):
        pytest.skip("mamba_forward binding not registered")

    from tests.reference.models.mamba_ref import make_mamba

    _seeded(0)
    device = _device_for_kernel()
    model = make_mamba(p=97, ntok=99, seq_len=8, d=128, nl=2).to(device).eval()

    batch, seq_len = 2, 8
    x = torch.randint(0, 99, (batch, seq_len), device=device, dtype=torch.long)

    with torch.no_grad():
        ref_out = model(x)
    assert ref_out.shape == (batch, 97), (
        f"reference Mamba produced unexpected shape {ref_out.shape}"
    )
    assert ref_out.dtype == torch.float32

    pytest.skip(
        "weight packing TBD — _ops.models.mamba_forward uses an opaque "
        "packed layout (see csrc/models/mamba_sm_90.h). Enable once a "
        "packer is wired up."
    )

    # ── live path ─────────────────────────────────────────────────────
    # tol = _TOLERANCES[ref_out.dtype]
    # packed = _pack_mamba_weights(model)  # TODO
    # kernel_out = ops.models.mamba_forward(x, *packed)
    # assert torch.allclose(kernel_out, ref_out, **tol)


# ── BF16 parity scaffolds ────────────────────────────────────────────
# Same structure as above but cast both oracle and kernel inputs to
# BF16 once the live path is enabled. They currently exercise only the
# oracle for shape/dtype contract.


@pytest.mark.parametrize("model_name", ["decoder", "vit", "mamba"])
def test_model_forward_parity_bf16(ops, model_name):
    """BF16 parity scaffold — same shapes as FP32 tests, BF16 dtype.

    Currently runs only the reference forward to confirm the oracle is
    BF16-compatible. The kernel call is gated on the weight packer
    being available, identical to the FP32 tests.
    """
    if not hasattr(ops.models, f"{model_name}_forward"):
        pytest.skip(f"{model_name}_forward binding not registered")

    _seeded(0)
    device = _device_for_kernel()

    if model_name == "decoder":
        from tests.reference.models.decoder_ref import make_decoder
        model = make_decoder().to(device=device, dtype=torch.bfloat16).eval()
        # Token IDs stay int64; only float weights cast to BF16.
        x = torch.randint(0, 99, (2, 4), device=device, dtype=torch.long)
        expected_shape = (2, 99)
    elif model_name == "vit":
        from tests.reference.models.vit_ref import make_vit
        model = make_vit().to(device=device, dtype=torch.bfloat16).eval()
        x = torch.randn(2, 16, 49, device=device, dtype=torch.bfloat16)
        expected_shape = (2, 97)
    elif model_name == "mamba":
        from tests.reference.models.mamba_ref import make_mamba
        model = make_mamba().to(device=device, dtype=torch.bfloat16).eval()
        x = torch.randint(0, 99, (2, 8), device=device, dtype=torch.long)
        expected_shape = (2, 97)
    else:
        pytest.fail(f"unknown model_name {model_name!r}")

    with torch.no_grad():
        ref_out = model(x)
    assert ref_out.shape == expected_shape
    assert ref_out.dtype == torch.bfloat16

    pytest.skip(
        f"weight packing TBD — _ops.models.{model_name}_forward uses an "
        f"opaque packed layout. Enable once a packer is wired up."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
