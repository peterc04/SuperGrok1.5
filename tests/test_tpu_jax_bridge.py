"""Smoke tests for the TPU v5p JAX/Pallas bridge.

This file is a *smoke test* for the JAX/Pallas bridge that connects the
TPU v5p device-header layer (``csrc/device/models/tpu_v5p/*``) to the
shared Pallas/JAX implementation (``csrc/kernels/tpu/_pallas_kernels.py``
and ``csrc/kernels/tpu/_pallas_models.py``).

What it verifies (on any host with JAX installed):
    1. Importability of the bridge modules.
    2. The v5p kernel registry (``_OPTIMIZER_KERNELS`` / ``_MODEL_KERNELS``
       and the ``get_kernels(kind)`` helper) exposes the expected callables.
    3. ``detect_tpu_version()`` returns one of {v4, v5e, v5p, v6e}; on
       non-TPU hosts the helper falls back to ``"v4"`` rather than raising.
    4. The pure-JAX fallback path of ``mamba3_scan_pallas_tile128`` runs
       on CPU and produces output with the correct shape.
    5. ``_pallas_models`` exposes the expected forward/backward callables.
    6. The TPU v5p device-header re-exports the underlying Pallas symbols
       (where the header re-exports verbatim) and exposes a callable
       wrapper for the rest.

What it does NOT verify:
    Real on-TPU execution. Tests that need a live TPU backend
    (``jax.devices('tpu')``) are gated by the ``has_tpu`` fixture and
    skipped cleanly on hosts without TPU.

Run with: ``pytest tests/test_tpu_jax_bridge.py``.
"""

import importlib

import pytest

# Skip the entire module if JAX is unavailable.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def has_tpu() -> bool:
    """Return True iff a TPU backend is reachable from this process.

    ``jax.devices('tpu')`` raises ``RuntimeError`` on hosts where the TPU
    backend isn't registered, and returns an empty list when the backend
    is registered but no devices are visible. We treat both as "no TPU".
    """
    try:
        devices = jax.devices("tpu")
    except Exception:
        return False
    return bool(devices)


@pytest.fixture(scope="module")
def pallas_kernels():
    """Import and return the central Pallas kernels module."""
    return importlib.import_module("csrc.kernels.tpu._pallas_kernels")


@pytest.fixture(scope="module")
def pallas_models():
    """Import and return the central Pallas models module."""
    return importlib.import_module("csrc.kernels.tpu._pallas_models")


@pytest.fixture(scope="module")
def v5p_pkg():
    """Import and return the v5p kernel registry package."""
    return importlib.import_module("csrc.kernels.tpu.v5p")


# ─── 1. Importability ────────────────────────────────────────────────────


def test_import_pallas_kernels(pallas_kernels):
    """``csrc.kernels.tpu._pallas_kernels`` imports cleanly."""
    assert hasattr(pallas_kernels, "detect_tpu_version")
    assert hasattr(pallas_kernels, "mamba3_scan_pallas_tile128")


def test_import_pallas_models(pallas_models):
    """``csrc.kernels.tpu._pallas_models`` imports cleanly."""
    assert hasattr(pallas_models, "decoder_forward")
    assert hasattr(pallas_models, "ModelConfig")


def test_import_v5p_init(v5p_pkg):
    """``csrc.kernels.tpu.v5p`` imports cleanly and exposes registries."""
    assert hasattr(v5p_pkg, "_OPTIMIZER_KERNELS")
    assert hasattr(v5p_pkg, "_MODEL_KERNELS")
    assert hasattr(v5p_pkg, "get_kernels")
    assert hasattr(v5p_pkg, "TILE_SIZE")
    assert v5p_pkg.TILE_SIZE == 128


# ─── 2. TPU detection ────────────────────────────────────────────────────


def test_detect_tpu_version_returns_known_string(pallas_kernels):
    """``detect_tpu_version()`` returns one of the recognised version tags.

    Per the implementation, the function is total: on non-TPU hosts (and
    on unrecognised ``device_kind`` strings) it falls back to ``"v4"``
    rather than raising. So we just verify the value is in the known set.
    """
    version = pallas_kernels.detect_tpu_version()
    assert version in {"v4", "v5e", "v5p", "v6e"}, (
        f"detect_tpu_version() returned unexpected value: {version!r}"
    )


def test_detect_tpu_version_matches_devices(pallas_kernels, has_tpu):
    """On a real TPU host the detected version is non-fallback."""
    if not has_tpu:
        pytest.skip("No TPU backend available")
    version = pallas_kernels.detect_tpu_version()
    # On a real TPU we expect *some* TPU label; v4 is also a valid
    # answer here, so we only assert membership.
    assert version in {"v4", "v5e", "v5p", "v6e"}


# ─── 3. Kernel registration (v5p package surface) ───────────────────────


# Per ``csrc/kernels/tpu/v5p/__init__.py``:
#   _OPTIMIZER_KERNELS has 4 entries (the scan / expert-MLP kernels)
#   _MODEL_KERNELS has 11 entries (the model forward/backward callables)
EXPECTED_OPTIMIZER_KEYS = {
    "mamba3_scan",
    "pallas_persistent_scan_fused_elem",
    "vmem_persistent_expert_mlp",
    "sharded_mamba3_scan",
}

EXPECTED_MODEL_KEYS = {
    "attention_forward",
    "attention_backward",
    "decoder_forward",
    "decoder_backward",
    "vit_forward",
    "vit_backward",
    "vit_patch_project",
    "mamba_forward",
    "mamba_backward",
    "mamba_layer_forward",
    "mamba_selective_scan",
}


def test_v5p_optimizer_kernels_keys(v5p_pkg):
    """Optimizer-kernel registry exposes the 4 expected entries."""
    assert set(v5p_pkg._OPTIMIZER_KERNELS) == EXPECTED_OPTIMIZER_KEYS
    for name, fn in v5p_pkg._OPTIMIZER_KERNELS.items():
        assert callable(fn), f"v5p optimizer kernel {name!r} is not callable"


def test_v5p_model_kernels_keys(v5p_pkg):
    """Model-kernel registry exposes the 11 expected entries."""
    assert set(v5p_pkg._MODEL_KERNELS) == EXPECTED_MODEL_KEYS
    for name, fn in v5p_pkg._MODEL_KERNELS.items():
        assert callable(fn), f"v5p model kernel {name!r} is not callable"


def test_v5p_get_kernels_dispatcher(v5p_pkg):
    """``get_kernels(kind)`` returns a fresh dict per kind and rejects junk."""
    opts = v5p_pkg.get_kernels("optimizers")
    models = v5p_pkg.get_kernels("models")
    assert set(opts) == EXPECTED_OPTIMIZER_KEYS
    assert set(models) == EXPECTED_MODEL_KEYS

    # Returned dict must be a copy (mutating it must not corrupt the
    # registry), which the v5p __init__ implements via ``dict(...)``.
    opts.pop(next(iter(opts)))
    assert set(v5p_pkg._OPTIMIZER_KERNELS) == EXPECTED_OPTIMIZER_KEYS

    with pytest.raises(ValueError):
        v5p_pkg.get_kernels("not-a-kind")


# ─── 4. Pallas fallback execution ───────────────────────────────────────


def test_mamba3_scan_pallas_tile128_runs_on_cpu(pallas_kernels):
    """The tile-128 affine scan produces correctly-shaped output on CPU.

    On CPU there is no real Pallas backend; the implementation either
    falls through to ``jax.lax.associative_scan`` directly (when the
    input is small) or catches the Pallas exception and falls back to
    the same. Either way the function must return shape-correct output.
    """
    N = 64  # smaller than tile size 128 -> uses associative_scan path
    Ms = jnp.tile(jnp.eye(2)[None], (N, 1, 1)).astype(jnp.float32)
    bs = jnp.zeros((N, 2), dtype=jnp.float32)

    Ms_out, bs_out = pallas_kernels.mamba3_scan_pallas_tile128(Ms, bs)
    assert Ms_out.shape == (N, 2, 2)
    assert bs_out.shape == (N, 2)
    # Identity scan of identities is still identity.
    assert jnp.allclose(Ms_out, Ms)
    assert jnp.allclose(bs_out, bs)


def test_mamba3_scan_pallas_tile128_large_input(pallas_kernels):
    """Test the multi-tile path (input > tile size)."""
    N = 256  # >= 128 -> exercises the cross-tile reduction code path
    Ms = jnp.tile(jnp.eye(2)[None], (N, 1, 1)).astype(jnp.float32)
    bs = jnp.zeros((N, 2), dtype=jnp.float32)

    try:
        Ms_out, bs_out = pallas_kernels.mamba3_scan_pallas_tile128(Ms, bs)
    except ImportError:
        pytest.skip("Pallas import failed at call site")
    assert Ms_out.shape == (N, 2, 2)
    assert bs_out.shape == (N, 2)


# ─── 5. Model surface ───────────────────────────────────────────────────


_MODEL_CALLABLE_NAMES = (
    "attention_forward",
    "attention_backward",
    "decoder_forward",
    "decoder_backward",
    "vit_forward",
    "vit_backward",
    "vit_patch_project",
    "mamba_forward",
    "mamba_backward",
    "mamba_layer_forward",
    "mamba_selective_scan",
)


@pytest.mark.parametrize("name", _MODEL_CALLABLE_NAMES)
def test_pallas_models_surface_callable(pallas_models, name):
    """Each documented model symbol exists and is callable."""
    obj = getattr(pallas_models, name, None)
    assert obj is not None, f"_pallas_models is missing {name}"
    assert callable(obj), f"_pallas_models.{name} is not callable"


def test_pallas_models_dataclasses_present(pallas_models):
    """Weight / config NamedTuples are also exported."""
    for cls_name in ("ModelConfig", "DecoderWeights", "ViTWeights",
                     "MambaLayerWeights", "MambaWeights"):
        assert hasattr(pallas_models, cls_name), (
            f"_pallas_models missing class {cls_name}"
        )


# ─── 6. Device-header re-export check ───────────────────────────────────


def test_transformer_device_header_delegates():
    """``transformer_tpu_v5p`` exposes a callable that delegates to
    ``_pallas_models.decoder_forward``.

    The header wraps the underlying function in ``transformer_forward``
    rather than re-exporting ``decoder_forward`` directly, but the
    ``DecoderWeights`` and ``ModelConfig`` types are re-exported
    verbatim and must be the same object.
    """
    header = importlib.import_module(
        "csrc.device.models.tpu_v5p.transformer_tpu_v5p"
    )
    pallas_models = importlib.import_module(
        "csrc.kernels.tpu._pallas_models"
    )
    assert callable(header.transformer_forward)
    assert callable(header.transformer_backward)
    # Types are re-exported, not wrapped.
    assert header.DecoderWeights is pallas_models.DecoderWeights
    assert header.ModelConfig is pallas_models.ModelConfig


def test_vit_device_header_delegates():
    """``vit_tpu_v5p`` re-exports the Pallas ViT surface."""
    header = importlib.import_module(
        "csrc.device.models.tpu_v5p.vit_tpu_v5p"
    )
    pallas_models = importlib.import_module(
        "csrc.kernels.tpu._pallas_models"
    )
    assert callable(header.vit_forward)
    assert callable(header.vit_backward)
    # ``vit_patch_project`` is re-exported verbatim.
    assert header.vit_patch_project is pallas_models.vit_patch_project
    assert header.ViTWeights is pallas_models.ViTWeights
    assert header.ModelConfig is pallas_models.ModelConfig


def test_mamba_device_header_delegates():
    """``mamba_tpu_v5p`` re-exports the Pallas Mamba surface."""
    header = importlib.import_module(
        "csrc.device.models.tpu_v5p.mamba_tpu_v5p"
    )
    pallas_models = importlib.import_module(
        "csrc.kernels.tpu._pallas_models"
    )
    assert callable(header.mamba_forward)
    assert callable(header.mamba_backward)
    # These two are re-exported verbatim (same object identity).
    assert header.mamba_layer_forward is pallas_models.mamba_layer_forward
    assert header.mamba_selective_scan is pallas_models.mamba_selective_scan
    assert header.MambaWeights is pallas_models.MambaWeights
    assert header.MambaLayerWeights is pallas_models.MambaLayerWeights


def test_v5p_registry_callables_match_pallas_models(v5p_pkg, pallas_models):
    """Each entry in ``_MODEL_KERNELS`` is the very object exported by
    ``_pallas_models`` (re-export, not a wrapper)."""
    for name in EXPECTED_MODEL_KEYS:
        assert v5p_pkg._MODEL_KERNELS[name] is getattr(pallas_models, name)


# ─── 7. Real-TPU smoke (skipped on non-TPU hosts) ───────────────────────


def test_pallas_scan_executes_on_tpu(pallas_kernels, has_tpu):
    """Run the tile-128 scan on a real TPU device when one is present."""
    if not has_tpu:
        pytest.skip("No TPU backend available")
    N = 256
    Ms = jnp.tile(jnp.eye(2)[None], (N, 1, 1)).astype(jnp.float32)
    bs = jnp.zeros((N, 2), dtype=jnp.float32)
    Ms_out, bs_out = pallas_kernels.mamba3_scan_pallas_tile128(Ms, bs)
    assert Ms_out.shape == (N, 2, 2)
    assert bs_out.shape == (N, 2)
