"""Structural validation of the elementwise kernel headers.

We cannot compile CUDA/HIP on this host, so these tests validate the
header files' textual contract: include guards, required symbols,
matching API signatures between arch variants, and state struct
consistency.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO_ROOT / "grokking_optimizers" / "kernels"

SM90_DIR = KERNEL_DIR / "sm_90"
GFX942_DIR = KERNEL_DIR / "gfx942"

OPTIMIZERS = ["adamw", "lion", "grokfast", "grokadamw"]

SM90_HEADERS = {opt: SM90_DIR / f"{opt}_sm90.cuh" for opt in OPTIMIZERS}
GFX942_HEADERS = {opt: GFX942_DIR / f"{opt}_gfx942.hip.hpp" for opt in OPTIMIZERS}

EXPECTED_STATE_TENSORS = {
    "adamw": 2,
    "lion": 1,
    "grokfast": 3,
    "grokadamw": 3,
}

EXPECTED_STATE_BYTES = {
    "adamw": 8,
    "lion": 4,
    "grokfast": 12,
    "grokadamw": 12,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── File existence ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_header_exists(opt: str) -> None:
    assert SM90_HEADERS[opt].is_file(), f"Missing {SM90_HEADERS[opt]}"


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_header_exists(opt: str) -> None:
    assert GFX942_HEADERS[opt].is_file(), f"Missing {GFX942_HEADERS[opt]}"


def test_common_sm90_exists() -> None:
    assert (SM90_DIR / "common_sm90.cuh").is_file()


def test_common_gfx942_exists() -> None:
    assert (GFX942_DIR / "common_gfx942.hip.hpp").is_file()


# ── Include guards ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_include_guard(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    guard = f"GROKKING_{opt.upper()}_SM90_CUH_"
    assert f"#ifndef {guard}" in src
    assert f"#define {guard}" in src
    assert f"#endif" in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_include_guard(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    guard = f"GROKKING_{opt.upper()}_GFX942_HIP_HPP_"
    assert f"#ifndef {guard}" in src
    assert f"#define {guard}" in src


# ── Common header inclusion ──────────────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_includes_common(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    assert '#include "common_sm90.cuh"' in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_includes_common(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    assert '#include "common_gfx942.hip.hpp"' in src


# ── Namespace ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_namespace(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    assert "namespace grokking" in src
    assert "namespace sm90" in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_namespace(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    assert "namespace grokking" in src
    assert "namespace gfx942" in src


# ── State structs ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_state_tensors(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    expected = EXPECTED_STATE_TENSORS[opt]
    match = re.search(r"num_state_tensors\(\)\s*\{\s*return\s+(\d+)", src)
    assert match, f"num_state_tensors() not found in {opt} sm_90"
    assert int(match.group(1)) == expected


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_state_tensors(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    expected = EXPECTED_STATE_TENSORS[opt]
    match = re.search(r"num_state_tensors\(\)\s*\{\s*return\s+(\d+)", src)
    assert match, f"num_state_tensors() not found in {opt} gfx942"
    assert int(match.group(1)) == expected


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_state_bytes(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    expected = EXPECTED_STATE_BYTES[opt]
    match = re.search(
        r"state_bytes_per_element\(\)\s*\{\s*return\s+(\d+)\s*\*\s*sizeof\(float\)",
        src,
    )
    if match:
        assert int(match.group(1)) * 4 == expected
    else:
        match = re.search(
            r"state_bytes_per_element\(\)\s*\{\s*return\s+sizeof\(float\)", src
        )
        assert match and expected == 4, f"state_bytes mismatch for {opt} sm_90"


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_state_bytes(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    expected = EXPECTED_STATE_BYTES[opt]
    match = re.search(
        r"state_bytes_per_element\(\)\s*\{\s*return\s+(\d+)\s*\*\s*sizeof\(float\)",
        src,
    )
    if match:
        assert int(match.group(1)) * 4 == expected
    else:
        match = re.search(
            r"state_bytes_per_element\(\)\s*\{\s*return\s+sizeof\(float\)", src
        )
        assert match and expected == 4, f"state_bytes mismatch for {opt} gfx942"


# ── Update + kernel function presence ────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_has_update_function(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    assert f"{opt}_update(" in src, f"{opt}_update not found in sm_90 header"


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_has_update_function(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    assert f"{opt}_update(" in src, f"{opt}_update not found in gfx942 header"


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_has_kernel(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    assert f"{opt}_kernel(" in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_has_kernel(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    assert f"{opt}_kernel(" in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_has_vec4(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    assert f"{opt}_update_vec4(" in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_has_vec4(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    assert f"{opt}_update_vec4(" in src


# ── Template signature ───────────────────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_template_params(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    assert "typename ParamT" in src
    assert "NanPolicy NAN_POLICY" in src
    assert "bool ENABLE_CLIP" in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_template_params(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    assert "typename ParamT" in src
    assert "NanPolicy NAN_POLICY" in src
    assert "bool ENABLE_CLIP" in src


# ── Arch-specific intrinsics ────────────────────────────────────────────────

def test_sm90_common_uses_cuda_intrinsics() -> None:
    src = _read(SM90_DIR / "common_sm90.cuh")
    assert "__half2float" in src
    assert "__bfloat162float" in src
    assert "__float2half_rn" in src
    assert "__float2bfloat16_rn" in src


def test_gfx942_common_uses_hip_types() -> None:
    src = _read(GFX942_DIR / "common_gfx942.hip.hpp")
    assert "hip_bfloat16" in src
    assert "hip/hip_runtime.h" in src


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_sm90_uses_ldg(opt: str) -> None:
    src = _read(SM90_HEADERS[opt])
    assert "__ldg" in src, f"sm_90 {opt} should use __ldg for read-only cache"


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_gfx942_uses_nontemporal(opt: str) -> None:
    src = _read(GFX942_HEADERS[opt])
    assert "__builtin_nontemporal_load" in src or "nontemporal" in src.lower(), \
        f"gfx942 {opt} should use non-temporal loads"


# ── GrokAdamW-specific: always-on grad_clip ──────────────────────────────────

def test_sm90_grokadamw_has_grad_clip_param() -> None:
    src = _read(SM90_HEADERS["grokadamw"])
    assert "float grad_clip" in src


def test_gfx942_grokadamw_has_grad_clip_param() -> None:
    src = _read(GFX942_HEADERS["grokadamw"])
    assert "float grad_clip" in src


# ── Lion-specific: no bias correction / eps ──────────────────────────────────

def test_sm90_lion_no_bias_correction() -> None:
    src = _read(SM90_HEADERS["lion"])
    lines = [l for l in src.splitlines()
             if "bias_correction" in l and not l.strip().startswith("//")]
    assert len(lines) == 0, "Lion should have no bias_correction parameter"


def test_sm90_lion_no_eps() -> None:
    src = _read(SM90_HEADERS["lion"])
    fn_body = src[src.index("lion_update("):]
    fn_sig_end = fn_body.index(")")
    fn_sig = fn_body[:fn_sig_end]
    assert "float eps" not in fn_sig, "Lion should have no eps parameter"


# ── Cross-arch consistency ───────────────────────────────────────────────────

@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_state_tensor_count_matches_across_arch(opt: str) -> None:
    sm90_src = _read(SM90_HEADERS[opt])
    gfx942_src = _read(GFX942_HEADERS[opt])
    sm90_match = re.search(r"num_state_tensors\(\)\s*\{\s*return\s+(\d+)", sm90_src)
    gfx942_match = re.search(r"num_state_tensors\(\)\s*\{\s*return\s+(\d+)", gfx942_src)
    assert sm90_match and gfx942_match
    assert sm90_match.group(1) == gfx942_match.group(1), \
        f"{opt}: state tensor count differs between sm_90 and gfx942"
