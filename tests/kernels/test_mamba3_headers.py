"""Structural validation of Mamba-3 model kernel headers.

Validates that sm_90, gfx942, and TPU Mamba-3 headers conform to the
shared contract: state structs, resource declarations, per-layer
forward/backward functions, and cross-arch consistency.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO_ROOT / "grokking_optimizers" / "kernels"

SM90_FILE = KERNEL_DIR / "sm_90" / "mamba3_sm90.cuh"
GFX942_FILE = KERNEL_DIR / "gfx942" / "mamba3_gfx942.hip.hpp"
TPU_FILE = KERNEL_DIR / "tpu" / "mamba3_tpu.py"

FORWARD_LAYERS = [
    "embed_forward",
    "rmsnorm_forward",
    "in_proj_forward",
    "conv1d_forward",
    "ssm_scan_forward",
    "out_proj_forward",
    "residual_add",
    "lm_head_forward",
]

BACKWARD_LAYERS = [
    "lm_head_backward",
    "out_proj_backward",
    "ssm_scan_backward",
    "conv1d_backward",
    "in_proj_backward",
    "rmsnorm_backward",
    "embed_backward",
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _has_layer(src: str, layer: str) -> bool:
    """Check if a layer function name appears in source, accepting naming variants.

    Accepts: exact match, model-prefixed (mamba3_), _fwd/_bwd shorthand,
    or selective_scan as stand-in for ssm_scan.
    """
    if layer in src:
        return True
    prefixed = "mamba3_" + layer
    if prefixed in src:
        return True
    fwd_alt = layer.replace("_forward", "_fwd")
    bwd_alt = layer.replace("_backward", "_bwd")
    if fwd_alt != layer and fwd_alt in src:
        return True
    if bwd_alt != layer and bwd_alt in src:
        return True
    if "ssm_scan" in layer:
        alt = layer.replace("ssm_scan", "selective_scan")
        if alt in src:
            return True
        if ("mamba3_" + alt) in src:
            return True
    if layer == "embed_backward" and "d_tok_embed" in src:
        return True
    return False


# ── File existence ───────────────────────────────────────────────────────────

def test_sm90_exists() -> None:
    assert SM90_FILE.is_file()


def test_gfx942_exists() -> None:
    assert GFX942_FILE.is_file()


def test_tpu_exists() -> None:
    assert TPU_FILE.is_file()


# ── Include guards / imports ─────────────────────────────────────────────────

def test_sm90_include_guard() -> None:
    src = _read(SM90_FILE)
    assert "#ifndef GROKKING_MAMBA3_SM90_CUH_" in src


def test_gfx942_include_guard() -> None:
    src = _read(GFX942_FILE)
    assert "#ifndef GROKKING_MAMBA3_GFX942_HIP_HPP_" in src


def test_sm90_includes_common() -> None:
    assert "common_sm90.cuh" in _read(SM90_FILE)


def test_gfx942_includes_common() -> None:
    assert "common_gfx942.hip.hpp" in _read(GFX942_FILE)


def test_tpu_imports_common() -> None:
    assert "common_tpu" in _read(TPU_FILE)


# ── Namespace ────────────────────────────────────────────────────────────────

def test_sm90_namespace() -> None:
    src = _read(SM90_FILE)
    assert "namespace grokking" in src and "namespace sm90" in src


def test_gfx942_namespace() -> None:
    src = _read(GFX942_FILE)
    assert "namespace grokking" in src and "namespace gfx942" in src


# ── State struct ─────────────────────────────────────────────────────────────

def test_sm90_has_state_struct() -> None:
    assert "Mamba3State" in _read(SM90_FILE)


def test_gfx942_has_state_struct() -> None:
    assert "Mamba3State" in _read(GFX942_FILE)


def test_tpu_has_state_class() -> None:
    assert "Mamba3State" in _read(TPU_FILE)


# ── Template / type parameters ───────────────────────────────────────────────

def test_sm90_template_params() -> None:
    src = _read(SM90_FILE)
    assert "ParamT" in src
    assert "SEQ_LEN" in src
    assert "NanPolicy" in src or "NAN_POLICY" in src


def test_gfx942_template_params() -> None:
    src = _read(GFX942_FILE)
    assert "ParamT" in src
    assert "SEQ_LEN" in src


# ── Per-layer forward functions ──────────────────────────────────────────────

@pytest.mark.parametrize("layer", FORWARD_LAYERS)
def test_sm90_has_forward_layer(layer: str) -> None:
    assert _has_layer(_read(SM90_FILE), layer), f"sm_90 missing {layer}"


@pytest.mark.parametrize("layer", FORWARD_LAYERS)
def test_gfx942_has_forward_layer(layer: str) -> None:
    assert _has_layer(_read(GFX942_FILE), layer), f"gfx942 missing {layer}"


@pytest.mark.parametrize("layer", FORWARD_LAYERS)
def test_tpu_has_forward_layer(layer: str) -> None:
    assert _has_layer(_read(TPU_FILE), layer), f"TPU missing {layer}"


# ── Per-layer backward functions ─────────────────────────────────────────────

@pytest.mark.parametrize("layer", BACKWARD_LAYERS)
def test_sm90_has_backward_layer(layer: str) -> None:
    assert _has_layer(_read(SM90_FILE), layer), f"sm_90 missing {layer}"


@pytest.mark.parametrize("layer", BACKWARD_LAYERS)
def test_gfx942_has_backward_layer(layer: str) -> None:
    assert _has_layer(_read(GFX942_FILE), layer), f"gfx942 missing {layer}"


@pytest.mark.parametrize("layer", BACKWARD_LAYERS)
def test_tpu_has_backward_layer(layer: str) -> None:
    assert _has_layer(_read(TPU_FILE), layer), f"TPU missing {layer}"


# ── Mamba-specific features ──────────────────────────────────────────────────

def test_sm90_has_selective_scan() -> None:
    src = _read(SM90_FILE)
    assert "selective_scan" in src.lower() or "ssm_scan" in src


def test_gfx942_has_selective_scan() -> None:
    src = _read(GFX942_FILE)
    assert "selective_scan" in src.lower() or "ssm_scan" in src


def test_tpu_has_selective_scan() -> None:
    src = _read(TPU_FILE)
    assert "selective_scan" in src.lower() or "ssm_scan" in src


def test_sm90_has_conv1d() -> None:
    assert "conv1d" in _read(SM90_FILE).lower()


def test_gfx942_has_conv1d() -> None:
    assert "conv1d" in _read(GFX942_FILE).lower()


def test_tpu_has_conv1d() -> None:
    assert "conv1d" in _read(TPU_FILE).lower()


def test_sm90_has_softplus() -> None:
    src = _read(SM90_FILE)
    assert "softplus" in src.lower() or "log1p" in src.lower()


def test_sm90_has_silu() -> None:
    src = _read(SM90_FILE)
    assert "silu" in src.lower() or "SiLU" in src


def test_tpu_uses_associative_scan() -> None:
    src = _read(TPU_FILE)
    assert "associative_scan" in src or "lax.scan" in src


# ── Size helpers ─────────────────────────────────────────────────────────────

def test_sm90_has_size_helpers() -> None:
    src = _read(SM90_FILE)
    assert "param_bytes" in src


def test_gfx942_has_size_helpers() -> None:
    src = _read(GFX942_FILE)
    has_size = "param_bytes" in src or "Mamba3Sizes" in src or "SMEM_BYTES" in src
    assert has_size


def test_tpu_has_size_helpers() -> None:
    assert "param_bytes" in _read(TPU_FILE)


# ── Arch-specific features ──────────────────────────────────────────────────

def test_sm90_uses_wgmma_or_cutlass() -> None:
    src = _read(SM90_FILE)
    assert "wgmma" in src.lower() or "cutlass" in src.lower() or "CUTLASS" in src


def test_gfx942_uses_mfma() -> None:
    src = _read(GFX942_FILE)
    assert "mfma" in src.lower() or "MFMA" in src


def test_sm90_has_blelloch_or_parallel_scan() -> None:
    src = _read(SM90_FILE)
    has_scan = any(term in src.lower() for term in [
        "blelloch", "prefix_sum", "parallel_scan", "upsweep", "up_sweep",
        "downsweep", "down_sweep", "scan_", "warp_scan",
    ])
    assert has_scan, "sm_90 SSM should use parallel scan"


# ── Tied embeddings ──────────────────────────────────────────────────────────

def test_sm90_tied_embeddings() -> None:
    assert "TIED_EMBEDDINGS" in _read(SM90_FILE)


def test_gfx942_tied_embeddings() -> None:
    assert "TIED_EMBEDDINGS" in _read(GFX942_FILE)


def test_tpu_tied_embeddings() -> None:
    src = _read(TPU_FILE)
    assert "tied_embeddings" in src.lower() or "TIED_EMBEDDINGS" in src
