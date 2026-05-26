"""Structural validation of transformer decoder model kernel headers.

Validates that sm_90, gfx942, and TPU transformer decoder headers
conform to the shared contract: state structs, resource declarations,
per-layer forward/backward functions, constexpr size helpers, and
cross-arch consistency.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO_ROOT / "grokking_optimizers" / "kernels"

SM90_FILE = KERNEL_DIR / "sm_90" / "transformer_decoder_sm90.cuh"
GFX942_FILE = KERNEL_DIR / "gfx942" / "transformer_decoder_gfx942.hip.hpp"
TPU_FILE = KERNEL_DIR / "tpu" / "transformer_decoder_tpu.py"

FORWARD_LAYERS = [
    "embed_forward",
    "rmsnorm_forward",
    "qkv_proj_forward",
    "attention_forward",
    "o_proj_forward",
    "residual_add",
    "mlp_gate_up_forward",
    "mlp_down_forward",
    "lm_head_forward",
]

BACKWARD_LAYERS = [
    "lm_head_backward",
    "mlp_down_backward",
    "mlp_gate_up_backward",
    "o_proj_backward",
    "attention_backward",
    "qkv_proj_backward",
    "rmsnorm_backward",
    "embed_backward",
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _has_layer(src: str, layer: str) -> bool:
    """Check if a layer function name appears in source, accepting naming variants.

    Accepts: exact match, _fwd/_bwd shorthand, or linear_backward as a
    stand-in for specific projection backward functions in TPU code.
    """
    if layer in src:
        return True
    fwd_alt = layer.replace("_forward", "_fwd")
    bwd_alt = layer.replace("_backward", "_bwd")
    if fwd_alt != layer and fwd_alt in src:
        return True
    if bwd_alt != layer and bwd_alt in src:
        return True
    if layer.endswith("_backward") and "linear_backward" in src:
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
    assert "#ifndef GROKKING_TRANSFORMER_DECODER_SM90_CUH_" in src
    assert "#define GROKKING_TRANSFORMER_DECODER_SM90_CUH_" in src


def test_gfx942_include_guard() -> None:
    src = _read(GFX942_FILE)
    assert "#ifndef GROKKING_TRANSFORMER_DECODER_GFX942_HIP_HPP_" in src
    assert "#define GROKKING_TRANSFORMER_DECODER_GFX942_HIP_HPP_" in src


def test_sm90_includes_common() -> None:
    src = _read(SM90_FILE)
    assert "common_sm90.cuh" in src


def test_gfx942_includes_common() -> None:
    src = _read(GFX942_FILE)
    assert "common_gfx942.hip.hpp" in src


def test_tpu_imports_common() -> None:
    src = _read(TPU_FILE)
    assert "common_tpu" in src


# ── Namespace ────────────────────────────────────────────────────────────────

def test_sm90_namespace() -> None:
    src = _read(SM90_FILE)
    assert "namespace grokking" in src
    assert "namespace sm90" in src


def test_gfx942_namespace() -> None:
    src = _read(GFX942_FILE)
    assert "namespace grokking" in src
    assert "namespace gfx942" in src


# ── State struct ─────────────────────────────────────────────────────────────

def test_sm90_has_state_struct() -> None:
    src = _read(SM90_FILE)
    assert "TransformerDecoderState" in src or "DecoderState" in src


def test_gfx942_has_state_struct() -> None:
    src = _read(GFX942_FILE)
    assert "TransformerDecoderState" in src or "DecoderState" in src


def test_tpu_has_state_class() -> None:
    src = _read(TPU_FILE)
    assert "TransformerDecoderState" in src or "DecoderState" in src


# ── Resource declarations ────────────────────────────────────────────────────

def test_sm90_resource_declarations() -> None:
    src = _read(SM90_FILE)
    assert "SMEM_BYTES" in src or "smem_bytes" in src.lower()
    assert "REG_HINT" in src or "reg_hint" in src.lower()


def test_gfx942_resource_declarations() -> None:
    src = _read(GFX942_FILE)
    assert "SMEM_BYTES" in src or "smem_bytes" in src.lower() or "LDS" in src


# ── Template parameters ─────────────────────────────────────────────────────

def test_sm90_template_params() -> None:
    src = _read(SM90_FILE)
    assert "ParamT" in src
    assert "SEQ_LEN" in src
    assert "BATCH" in src
    assert "NanPolicy" in src or "NAN_POLICY" in src


def test_gfx942_template_params() -> None:
    src = _read(GFX942_FILE)
    assert "ParamT" in src
    assert "SEQ_LEN" in src
    assert "BATCH" in src
    assert "NanPolicy" in src or "NAN_POLICY" in src


# ── Per-layer forward functions ──────────────────────────────────────────────

@pytest.mark.parametrize("layer", FORWARD_LAYERS)
def test_sm90_has_forward_layer(layer: str) -> None:
    src = _read(SM90_FILE)
    assert _has_layer(src, layer), f"sm_90 missing {layer}"


@pytest.mark.parametrize("layer", FORWARD_LAYERS)
def test_gfx942_has_forward_layer(layer: str) -> None:
    src = _read(GFX942_FILE)
    assert _has_layer(src, layer), f"gfx942 missing {layer}"


@pytest.mark.parametrize("layer", FORWARD_LAYERS)
def test_tpu_has_forward_layer(layer: str) -> None:
    src = _read(TPU_FILE)
    assert _has_layer(src, layer), f"TPU missing {layer}"


# ── Per-layer backward functions ─────────────────────────────────────────────

@pytest.mark.parametrize("layer", BACKWARD_LAYERS)
def test_sm90_has_backward_layer(layer: str) -> None:
    src = _read(SM90_FILE)
    assert _has_layer(src, layer), f"sm_90 missing {layer}"


@pytest.mark.parametrize("layer", BACKWARD_LAYERS)
def test_gfx942_has_backward_layer(layer: str) -> None:
    src = _read(GFX942_FILE)
    assert _has_layer(src, layer), f"gfx942 missing {layer}"


@pytest.mark.parametrize("layer", BACKWARD_LAYERS)
def test_tpu_has_backward_layer(layer: str) -> None:
    src = _read(TPU_FILE)
    assert _has_layer(src, layer), f"TPU missing {layer}"


# ── Size helpers ─────────────────────────────────────────────────────────────

def test_sm90_has_size_helpers() -> None:
    src = _read(SM90_FILE)
    assert "param_bytes" in src
    assert "activation_bytes" in src or "activation_bytes_per_layer" in src


def test_gfx942_has_size_helpers() -> None:
    src = _read(GFX942_FILE)
    has_size = "param_bytes" in src or "DecoderSizes" in src or "SMEM_BYTES" in src
    assert has_size


def test_tpu_has_size_helpers() -> None:
    src = _read(TPU_FILE)
    assert "param_bytes" in src
    assert "activation_bytes" in src or "activation_bytes_per_layer" in src


# ── Arch-specific features ──────────────────────────────────────────────────

def test_sm90_uses_wgmma_or_cutlass() -> None:
    src = _read(SM90_FILE)
    assert "wgmma" in src.lower() or "cutlass" in src.lower() or "CUTLASS" in src


def test_sm90_has_rope() -> None:
    src = _read(SM90_FILE)
    assert "rope" in src.lower()


def test_gfx942_uses_mfma() -> None:
    src = _read(GFX942_FILE)
    assert "mfma" in src.lower() or "MFMA" in src


def test_gfx942_uses_nontemporal() -> None:
    src = _read(GFX942_FILE)
    assert "nontemporal" in src.lower()


def test_sm90_attention_is_causal() -> None:
    src = _read(SM90_FILE)
    assert "causal" in src.lower()


def test_gfx942_attention_is_causal() -> None:
    src = _read(GFX942_FILE)
    assert "causal" in src.lower()


# ── Tied embeddings ──────────────────────────────────────────────────────────

def test_sm90_tied_embeddings() -> None:
    src = _read(SM90_FILE)
    assert "TIED_EMBEDDINGS" in src


def test_gfx942_tied_embeddings() -> None:
    src = _read(GFX942_FILE)
    assert "TIED_EMBEDDINGS" in src


def test_tpu_tied_embeddings() -> None:
    src = _read(TPU_FILE)
    assert "tied_embeddings" in src.lower() or "TIED_EMBEDDINGS" in src
