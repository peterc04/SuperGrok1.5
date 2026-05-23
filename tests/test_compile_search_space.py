"""tests/test_compile_search_space.py — YAML loader + pre-filter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grokking_optimizers import search_space as ss  # noqa: E402


def test_load_yaml_validates_shape(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("sm_90:\n  dims:\n    - name: block\n")  # missing type
    with pytest.raises(ss.SearchSpaceError):
        ss.load_yaml(yaml_path)


def test_load_yaml_rejects_unknown_target(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        "sm_90:\n"
        "  dims:\n"
        "    - name: block\n"
        "      type: int\n"
        "      values: [64]\n"
        "      macro: SG_TUNED_BLOCK_SIZE\n"
        "      applies_to: [host, gibberish]\n"
    )
    with pytest.raises(ss.SearchSpaceError):
        ss.load_yaml(yaml_path)


def test_load_yaml_rejects_duplicate_dim(tmp_path: Path) -> None:
    yaml_path = tmp_path / "dup.yaml"
    yaml_path.write_text(
        "sm_90:\n"
        "  dims:\n"
        "    - {name: block, type: int, values: [64]}\n"
        "    - {name: block, type: int, values: [128]}\n"
    )
    with pytest.raises(ss.SearchSpaceError):
        ss.load_yaml(yaml_path)


def test_real_yaml_loads() -> None:
    space = ss.load_yaml(REPO_ROOT / "configs" / "search_space.yaml")
    assert "sm_90" in space
    assert "gfx942" in space
    sm = space["sm_90"]
    assert any(d["name"] == "block" for d in sm["dims"])
    assert any(d["name"] == "cluster_shape" for d in sm["dims"])
    assert any(d["name"] == "tma" for d in sm["dims"])
    gfx = space["gfx942"]
    assert any(d["name"] == "waves_per_eu" for d in gfx["dims"])


def test_cartesian_counts() -> None:
    space = ss.load_yaml(REPO_ROOT / "configs" / "search_space.yaml")
    configs = ss.cartesian(space, "sm_90")
    # 5 block × 3 vec × 5 unroll × 4 num_stages × 5 maxrregcount ×
    # 3 cluster × 3 swizzle × 2 warpspec × 2 tma × 4 async_depth = 720000
    assert len(configs) == 5 * 3 * 5 * 4 * 5 * 3 * 3 * 2 * 2 * 4


def test_prefilter_eliminates() -> None:
    space = ss.load_yaml(REPO_ROOT / "configs" / "search_space.yaml")
    configs = ss.cartesian(space, "sm_90")
    survivors, eliminated = ss.prefilter(
        configs, space["sm_90"]["prefilter"])
    # The Hopper rules (alignment + cluster volume + tma/warpspec >= 128)
    # cut the space by roughly half. Sanity: should drop something.
    assert eliminated > 0
    assert len(survivors) + eliminated == len(configs)
    # And every survivor should satisfy the rules manually
    for cfg in survivors[:50]:
        assert cfg["block"] % (cfg["vec"] * 4) == 0
        if cfg["tma"]:
            assert cfg["block"] >= 128


def test_resolve_macros_filters_by_target() -> None:
    space = ss.load_yaml(REPO_ROOT / "configs" / "search_space.yaml")
    dims = space["sm_90"]["dims"]
    cfg = {"block": 256, "vec": 4, "unroll": 8, "num_stages": 3,
           "maxrregcount": 168, "cluster_shape": (2, 1, 1),
           "swizzle": "xor4", "warp_specialization": True, "tma": True,
           "async_depth": 4}
    host = ss.resolve_macros(cfg, dims, "host")
    device = ss.resolve_macros(cfg, dims, "device")
    # block / vec / unroll apply to both
    assert any(m.startswith("-DSG_TUNED_BLOCK_SIZE=") for m in host)
    assert any(m.startswith("-DSG_TUNED_BLOCK_SIZE=") for m in device)
    # tma is device-only
    assert any(m.startswith("-DSG_TUNED_TMA=") for m in device)
    assert not any(m.startswith("-DSG_TUNED_TMA=") for m in host)
    # maxrregcount has macro=null → no -D
    assert not any("maxrregcount" in m.lower() for m in host + device)


def test_resolve_extra_nvcc_flags() -> None:
    space = ss.load_yaml(REPO_ROOT / "configs" / "search_space.yaml")
    dims = space["sm_90"]["dims"]
    cfg = {"maxrregcount": 168, "block": 256}
    flags = ss.resolve_extra_nvcc_flags(cfg, dims)
    assert "--maxrregcount=168" in flags


def test_hash_space_stable_and_diff() -> None:
    space = ss.load_yaml(REPO_ROOT / "configs" / "search_space.yaml")
    h1 = ss.hash_space(space, "sm_90")
    h2 = ss.hash_space(space, "sm_90")
    assert h1 == h2
    assert h1 != ss.hash_space(space, "gfx942")


def test_config_key_deterministic() -> None:
    cfg1 = {"block": 256, "vec": 4, "unroll": 8}
    cfg2 = {"unroll": 8, "block": 256, "vec": 4}  # reordered
    assert ss.config_key(cfg1) == ss.config_key(cfg2)
