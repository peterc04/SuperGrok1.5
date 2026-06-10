"""CPU-only unit test for lever (b): per-TU tuned-flag build injection.

Runs WITHOUT a GPU and WITHOUT building anything. It exercises the pure logic
in ``grokking_optimizers/_tuned_inject.py`` — the same functions setup.py's
``TunedBuildExtension`` calls — so the autotuner→product-build linkage can be
verified in CI / on a laptop.

Run:  python -m tuning.test_build_injection
      (or: python tuning/test_build_injection.py)

Covers:
  1. TU path -> optimizer mapping, incl. the ambiguous-TU policy
     (launcher / mega cell -> opt; bindings / model TUs / common -> NONE).
  2. Per-source nvcc flag computation (block/vec/unroll/async_depth +
     maxrregcount; maxrregcount==0 omitted; absent/ambiguous -> []).
  3. build.ninja rewrite: the right CUDA edges get a per-statement
     cuda_post_cflags override carrying the FULL flag list (base preserved),
     and unrelated edges (.cpp, model .cu) get nothing.
  4. export_winner read-merge-write round-trip (atomic JSON, only the safe
     per-TU dims persisted, sibling winners survive).
  5. Drift guard: the MACROS source-of-truth macro names + defaults match the
     committed kernel header (adamw_sm90.cuh #ifndef SG_TUNED_*).

These assertions are the executable contract; setup.py is NOT imported (it
runs setup() at import time and probes torch/CUDA — not importable here).
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_INJECT_PATH = os.path.join(_REPO_ROOT, "grokking_optimizers", "_tuned_inject.py")


def _load_inject():
    """Load _tuned_inject.py by path (no package __init__, no torch)."""
    spec = importlib.util.spec_from_file_location(
        "grokking_optimizers_tuned_inject_test", _INJECT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ti = _load_inject()


# --------------------------------------------------------------------------
# Tiny assert harness (no pytest dependency required).
# --------------------------------------------------------------------------
_FAILURES = []


def check(cond, msg):
    if cond:
        return
    _FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def eq(got, exp, msg):
    check(got == exp, f"{msg}: got {got!r}, expected {exp!r}")


# --------------------------------------------------------------------------
# 1. TU path -> optimizer mapping (incl. ambiguous-TU policy).
# --------------------------------------------------------------------------
def test_source_to_optimizer():
    cases = {
        # per-optimizer launcher TUs -> the optimizer.
        "csrc/backends/cuda/sm_90/launch_adamw.cu": "adamw",
        "csrc/backends/cuda/sm_90/launch_supergrok2.cu": "supergrok2",
        "csrc/backends/cuda/sm_90/launch_supergrok15.cu": "supergrok15",
        "csrc/backends/cuda/sm_90/launch_muon.cu": "muon",
        # megakernel cells: mega_<model>_<opt> -> the opt (greedy on opt).
        "csrc/fused/sm_90/mega_transformer_decoder_supergrok2.cu": "supergrok2",
        "csrc/fused/sm_90/mega_mamba3_adamw.cu": "adamw",
        "csrc/fused/sm_90/mega_vit_supergrok11.cu": "supergrok11",
        "csrc/fused/sm_90/mega_vit_muon.cu": "muon",
        # AMBIGUOUS / non-opt TUs -> None (keep in-header defaults).
        "csrc/bindings/dispatch.cpp": None,        # not even CUDA
        "csrc/bindings/bindings.cpp": None,
        "csrc/backends/cuda/sm_90/models/transformer_decoder.cu": None,  # model-only
        "csrc/backends/cuda/sm_90/models/vit.cu": None,
        "csrc/backends/cuda/sm_90/primitives.cuh": None,   # header, not a TU
        "csrc/backends/cuda/sm_90/mma.cuh": None,
        "csrc/fused/sm_90/fused_megakernel.cuh": None,
        # a .cu that names no known optimizer -> None.
        "csrc/backends/cuda/sm_90/launch_unknownopt.cu": None,
        "csrc/fused/sm_90/mega_vit_notanopt.cu": None,
    }
    for path, exp in cases.items():
        eq(ti.optimizer_for_source(path), exp, f"optimizer_for_source({path})")


# --------------------------------------------------------------------------
# 2. Per-source nvcc flag computation.
# --------------------------------------------------------------------------
def _sample_tuned():
    return {
        "sm_90": {
            "adamw": {"block": 128, "vec": 2, "unroll": 4,
                      "async_depth": 3, "maxrregcount": 64, "model": "decoder"},
            "muon": {"block": 256, "vec": 4, "unroll": 1,
                     "async_depth": 2, "maxrregcount": 0, "model": "vit"},
            "supergrok2": {"block": 512, "vec": 8, "unroll": 2,
                           "async_depth": 4, "maxrregcount": 128},
        },
        "_meta": {"source": "test"},
    }


def test_flag_computation():
    tuned = _sample_tuned()
    # adamw: all five dims, maxrregcount>0 emitted. arch passed as internal
    # "sm_90a" must still resolve via canonicalization to the "sm_90" key.
    eq(ti.source_extra_nvcc_flags("adamw", tuned, "sm_90a"),
       ["-DSG_TUNED_BLOCK_SIZE=128", "-DSG_TUNED_VEC_WIDTH=2",
        "-DSG_TUNED_UNROLL=4", "-DSG_TUNED_ASYNC_DEPTH=3",
        "--maxrregcount=64"],
       "adamw flags (sm_90a canonicalizes to sm_90)")
    # muon: maxrregcount==0 -> OMITTED (nvcc default register allocator).
    eq(ti.source_extra_nvcc_flags("muon", tuned, "sm_90"),
       ["-DSG_TUNED_BLOCK_SIZE=256", "-DSG_TUNED_VEC_WIDTH=4",
        "-DSG_TUNED_UNROLL=1", "-DSG_TUNED_ASYNC_DEPTH=2"],
       "muon flags (maxrregcount=0 omitted)")
    # ambiguous TU (optimizer None) -> [].
    eq(ti.source_extra_nvcc_flags(None, tuned, "sm_90"), [],
       "None optimizer -> []")
    # optimizer with no recorded winner -> [].
    eq(ti.source_extra_nvcc_flags("lion", tuned, "sm_90"), [],
       "untuned optimizer -> []")
    # absent arch -> [].
    eq(ti.source_extra_nvcc_flags("adamw", tuned, "gfx942"), [],
       "absent arch -> []")
    # empty / None JSON -> [].
    eq(ti.source_extra_nvcc_flags("adamw", None, "sm_90"), [],
       "None tuned -> []")
    eq(ti.source_extra_nvcc_flags("adamw", {}, "sm_90"), [],
       "empty tuned -> []")


def test_compute_source_flags_mapping():
    tuned = _sample_tuned()
    srcs = [
        "csrc/backends/cuda/sm_90/launch_adamw.cu",       # adamw -> tuned
        "csrc/bindings/dispatch.cpp",                      # none
        "csrc/fused/sm_90/mega_vit_supergrok2.cu",        # supergrok2 -> tuned
        "csrc/backends/cuda/sm_90/models/vit.cu",          # model -> none
        "csrc/backends/cuda/sm_90/launch_lion.cu",        # lion: not in JSON
    ]
    m = ti.compute_source_flags(srcs, tuned, "sm_90")
    # adamw launcher gets adamw's block.
    eq(m.get("csrc/backends/cuda/sm_90/launch_adamw.cu", [None])[0],
       "-DSG_TUNED_BLOCK_SIZE=128", "adamw launcher -> adamw block")
    # supergrok2 mega cell gets supergrok2's block (512).
    eq(m.get("csrc/fused/sm_90/mega_vit_supergrok2.cu", [None])[0],
       "-DSG_TUNED_BLOCK_SIZE=512", "supergrok2 mega cell -> supergrok2 block")
    # unrelated .cpp, model .cu, and untuned lion are absent from the map.
    check("csrc/bindings/dispatch.cpp" not in m,
          "dispatch.cpp must get NO flags")
    check("csrc/backends/cuda/sm_90/models/vit.cu" not in m,
          "model TU must get NO flags")
    check("csrc/backends/cuda/sm_90/launch_lion.cu" not in m,
          "untuned lion launcher must get NO flags")


# --------------------------------------------------------------------------
# 3. build.ninja rewrite.
# --------------------------------------------------------------------------
def _esc(s):
    return s.replace(":", "$:").replace(" ", "$ ")


def _write_fake_ninja(directory, edges, base_cuda_post):
    """edges: list of (src, obj, rule) where rule in {compile, cuda_compile}."""
    path = os.path.join(directory, "build.ninja")
    lines = [
        "ninja_required_version = 1.3",
        "cxx = c++", "nvcc = nvcc",
        "cflags = -O3", "post_cflags = -std=c++17",
        "cuda_cflags = -I/repo",
        f"cuda_post_cflags = {base_cuda_post}",
        "cuda_dlink_post_cflags = ", "ldflags = -shared",
        "rule compile",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "rule cuda_compile",
        "  command = $nvcc $cuda_cflags -c $in -o $out $cuda_post_cflags",
    ]
    for src, obj, rule in edges:
        lines.append(f"build {_esc(obj)}: {rule} {_esc(src)}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def test_ninja_rewrite():
    tuned = _sample_tuned()
    d = tempfile.mkdtemp()
    adamw = ("/repo/csrc/backends/cuda/sm_90/launch_adamw.cu",
             os.path.join(d, "launch_adamw.o"), "cuda_compile")
    binding = ("/repo/csrc/bindings/dispatch.cpp",
               os.path.join(d, "dispatch.o"), "compile")
    mega = ("/repo/csrc/fused/sm_90/mega_vit_muon.cu",
            os.path.join(d, "mega_vit_muon.o"), "cuda_compile")
    model = ("/repo/csrc/backends/cuda/sm_90/models/vit.cu",
             os.path.join(d, "vit.o"), "cuda_compile")
    base = "-O3 --use_fast_math -gencode=arch=compute_90a,code=sm_90a"
    ninja = _write_fake_ninja(d, [adamw, binding, mega, model], base)

    sources = [e[0] for e in (adamw, binding, mega, model)]
    objects = [e[1] for e in (adamw, binding, mega, model)]
    n = ti.inject_overrides_into_ninja(
        ninja, sources, objects, base.split(), tuned, "sm_90",
        quote=shlex.quote)
    eq(n, 2, "exactly 2 CUDA edges (adamw launcher + muon mega) get overrides")

    text = open(ninja, encoding="utf-8").read()

    # The override lines (indented cuda_post_cflags) the rewrite added.
    overrides = [ln for ln in text.splitlines()
                 if ln.strip().startswith("cuda_post_cflags =")
                 and "SG_TUNED" in ln]
    eq(len(overrides), 2, "2 per-statement override lines added")
    for ln in overrides:
        # base flags MUST be preserved in every override (no silent drop).
        check("--use_fast_math" in ln and "compute_90a" in ln,
              f"base flags preserved in override: {ln!r}")

    # adamw override present with its values + maxrregcount.
    check(re.search(r"SG_TUNED_BLOCK_SIZE=128.*--maxrregcount=64", text)
          is not None, "adamw override has block=128 and --maxrregcount=64")
    # muon override present, NO maxrregcount (0 omitted).
    muon_ov = [ln for ln in overrides if "SG_TUNED_BLOCK_SIZE=256" in ln]
    eq(len(muon_ov), 1, "muon override present")
    check("--maxrregcount" not in muon_ov[0],
          "muon override omits --maxrregcount (was 0)")

    # The .cpp binding edge and the model .cu edge must have NO override line
    # directly beneath them.
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("build ") and "dispatch.o" in ln:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            check(not nxt.strip().startswith("cuda_post_cflags ="),
                  "dispatch.cpp edge must NOT be followed by an override")
        if ln.startswith("build ") and os.path.join(d, "vit.o").replace(
                ":", "$:").replace(" ", "$ ") in ln:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            check(not nxt.strip().startswith("cuda_post_cflags ="),
                  "model vit.cu edge must NOT be followed by an override")


def test_ninja_rewrite_noops_without_json():
    """No JSON / empty tuned -> ninja file is left byte-identical."""
    d = tempfile.mkdtemp()
    adamw = ("/repo/csrc/backends/cuda/sm_90/launch_adamw.cu",
             os.path.join(d, "launch_adamw.o"), "cuda_compile")
    base = "-O3 -gencode=arch=compute_90a,code=sm_90a"
    ninja = _write_fake_ninja(d, [adamw], base)
    before = open(ninja, encoding="utf-8").read()
    n = ti.inject_overrides_into_ninja(
        ninja, [adamw[0]], [adamw[1]], base.split(), None, "sm_90")
    eq(n, 0, "no-JSON injection touches 0 edges")
    eq(open(ninja, encoding="utf-8").read(), before,
       "ninja file unchanged when there is nothing tuned")


# --------------------------------------------------------------------------
# 4. export_winner read-merge-write round-trip.
# --------------------------------------------------------------------------
def test_export_merge_roundtrip():
    import json
    d = tempfile.mkdtemp()
    jp = os.path.join(d, "_kernel_tuned.json")

    ok1 = ti.export_winner(
        "adamw", "decoder", "sm_90a",
        {"block": 128, "vec": 2, "unroll": 4, "async_depth": 3,
         "maxrregcount": 64, "timing_ms": 1.23, "config_key": "deadbeef",
         "cluster_shape": (2, 1, 1)},  # tuple dim must NOT be persisted.
        path=jp)
    check(ok1, "export_winner adamw returned True")
    ok2 = ti.export_winner(
        "muon", "vit", "sm_90",
        {"block": 256, "vec": 4, "unroll": 1, "async_depth": 2,
         "maxrregcount": 0},
        path=jp)
    check(ok2, "export_winner muon returned True")

    data = json.load(open(jp, encoding="utf-8"))
    # both winners survive the merge under the short "sm_90" key.
    check("sm_90" in data, "sm_90 arch key present")
    check("adamw" in data["sm_90"] and "muon" in data["sm_90"],
          "both adamw and muon winners present after merge")
    # only the safe per-TU dims (+ model provenance) persisted; internal
    # autotuner keys and tuple dims dropped.
    a = data["sm_90"]["adamw"]
    eq(a["block"], 128, "adamw block persisted")
    eq(a["maxrregcount"], 64, "adamw maxrregcount persisted")
    eq(a.get("model"), "decoder", "adamw model provenance persisted")
    check("timing_ms" not in a, "timing_ms NOT persisted")
    check("config_key" not in a, "config_key NOT persisted")
    check("cluster_shape" not in a, "cluster_shape (tuple) NOT persisted")
    # _meta present.
    check("_meta" in data and "timestamp" in data["_meta"],
          "_meta with timestamp present")

    # load_tuned reads it back identically.
    rt = ti.load_tuned(jp)
    eq(rt["sm_90"]["muon"]["vec"], 4, "load_tuned round-trips muon vec")

    # export_winner must NEVER raise / must return False on a bad path,
    # never propagate (fail-closed). Use a path whose "directory" is actually
    # an existing FILE, so os.makedirs() raises NotADirectoryError — a case
    # the exporter must catch and return False for (a failed persist can never
    # break a tuning run).
    afile = os.path.join(d, "a_regular_file")
    with open(afile, "w", encoding="utf-8") as fh:
        fh.write("x")
    bad = ti.export_winner("adamw", "decoder", "sm_90", {"block": 1},
                           path=os.path.join(afile, "sub", "_k.json"))
    check(bad is False, "export_winner on unwritable path returns False")


# --------------------------------------------------------------------------
# 5. Drift guard: MACROS source-of-truth vs the committed kernel header.
# --------------------------------------------------------------------------
def test_macro_drift_against_header():
    header = os.path.join(_REPO_ROOT, "grokking_optimizers", "kernels",
                          "sm_90", "adamw_sm90.cuh")
    if not os.path.isfile(header):
        print("  SKIP: adamw_sm90.cuh not found (drift check skipped)")
        return
    text = open(header, encoding="utf-8").read()
    # Parse `#define SG_TUNED_X <int>` guarded by `#ifndef SG_TUNED_X`.
    parsed = {}
    for m in re.finditer(r"#define\s+(SG_TUNED_[A-Z0-9_]+)\s+(\d+)", text):
        parsed[m.group(1)] = int(m.group(2))
    expected = ti.header_default_macros()  # {macro: default}
    for macro, default in expected.items():
        check(macro in parsed,
              f"macro {macro} from MACROS is #defined in adamw_sm90.cuh")
        if macro in parsed:
            eq(parsed[macro], default,
               f"{macro} default in header matches MACROS table")


# --------------------------------------------------------------------------
# 6. torch-INTEGRATION test (skipped if torch is unavailable).
#
# The pure-logic tests above run the rewrite on a HAND-WRITTEN build.ninja.
# This one drives torch's REAL `_write_ninja_file_and_compile_objects` through
# the SAME module-level _write_ninja_file monkeypatch setup.py installs, so it
# proves the two things the isolation tests structurally cannot:
#   (a) torch's bare-name call to _write_ninja_file (cpp_extension.py:1771)
#       actually resolves to OUR patched function, and
#   (b) our ninja parser matches torch's ACTUAL `build <obj>: cuda_compile
#       <src>` emission/escaping — not our synthetic copy of it.
# It compiles NOTHING: _run_ninja_build (and the hermeticity helpers) are
# stubbed, and build.ninja is written BEFORE _run_ninja_build is called.
# --------------------------------------------------------------------------
def test_torch_integration_real_ninja_writer():
    try:
        import torch  # noqa: F401
        import torch.utils.cpp_extension as cpp
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP: torch unavailable ({type(exc).__name__}); "
              "torch-integration test skipped (pure-logic tests cover the rest)")
        return

    import shlex as _shlex

    d = tempfile.mkdtemp()
    # Mixed real-format sources: an adamw launcher (.cu), a binding (.cpp),
    # and a muon mega cell (.cu). Absolute object paths (torch abspaths them).
    src_adamw = os.path.join(d, "launch_adamw.cu")
    src_bind = os.path.join(d, "dispatch.cpp")
    src_mega = os.path.join(d, "mega_vit_muon.cu")
    for p in (src_adamw, src_bind, src_mega):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("// stub\n")
    obj_adamw = os.path.join(d, "launch_adamw.o")
    obj_bind = os.path.join(d, "dispatch.o")
    obj_mega = os.path.join(d, "mega_vit_muon.o")
    sources = [src_adamw, src_bind, src_mega]
    objects = [obj_adamw, obj_bind, obj_mega]

    tuned = _sample_tuned()  # adamw + muon have winners under "sm_90"

    # The SAME wrapper setup.py's TunedBuildExtension installs.
    orig_writer = cpp._write_ninja_file

    def patched_writer(*args, **kwargs):
        orig_writer(*args, **kwargs)
        ti.inject_overrides_into_ninja(
            kwargs.get("path"), list(kwargs.get("sources")),
            list(kwargs.get("objects")),
            list(kwargs.get("cuda_post_cflags") or []),
            tuned, "sm_90", quote=_shlex.quote)

    # Hermetic stubs so nothing touches a real toolchain / runs ninja.
    saved = {
        "_write_ninja_file": cpp._write_ninja_file,
        "_run_ninja_build": cpp._run_ninja_build,
        "verify_ninja_availability": cpp.verify_ninja_availability,
        "get_compiler_abi_compatibility_and_version":
            cpp.get_compiler_abi_compatibility_and_version,
        "get_cxx_compiler": cpp.get_cxx_compiler,
    }
    try:
        cpp._write_ninja_file = patched_writer
        cpp._run_ninja_build = lambda *a, **k: None         # no compile
        cpp.verify_ninja_availability = lambda *a, **k: None
        cpp.get_compiler_abi_compatibility_and_version = \
            lambda *a, **k: (True, "0.0")
        cpp.get_cxx_compiler = lambda *a, **k: "c++"
        cpp._write_ninja_file_and_compile_objects(
            sources=sources, objects=objects,
            cflags=["-O3"], post_cflags=["-std=c++17"],
            cuda_cflags=["-I/repo"],
            cuda_post_cflags=["-O3", "--use_fast_math",
                              "-gencode=arch=compute_90a,code=sm_90a"],
            cuda_dlink_post_cflags=None,
            build_directory=d, verbose=False, with_cuda=True)
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append(f"torch-integration call raised: {exc!r}")
        return
    finally:
        for k, v in saved.items():
            setattr(cpp, k, v)

    ninja_path = os.path.join(d, "build.ninja")
    check(os.path.isfile(ninja_path),
          "torch wrote build.ninja in the build_directory")
    if not os.path.isfile(ninja_path):
        return
    text = open(ninja_path, encoding="utf-8").read()

    # Our parser matched torch's REAL emission: adamw + muon edges have
    # overrides; the .cpp edge does not.
    overrides = [ln for ln in text.splitlines()
                 if ln.strip().startswith("cuda_post_cflags =")
                 and "SG_TUNED" in ln]
    eq(len(overrides), 2,
       "2 override lines injected into torch's real build.ninja "
       "(adamw + muon .cu edges)")
    check(any("SG_TUNED_BLOCK_SIZE=128" in ln for ln in overrides),
          "adamw override present in torch's real build.ninja")
    check(any("SG_TUNED_BLOCK_SIZE=256" in ln for ln in overrides),
          "muon override present in torch's real build.ninja")
    for ln in overrides:
        check("--use_fast_math" in ln and "compute_90a" in ln,
              f"base flags preserved in real-ninja override: {ln!r}")
    # the binding object must not be followed by an override line.
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("build ") and "dispatch.o" in ln:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            check(not nxt.strip().startswith("cuda_post_cflags ="),
                  "dispatch.cpp edge has NO override in torch's real build.ninja")


# --------------------------------------------------------------------------
def main():
    tests = [
        test_source_to_optimizer,
        test_flag_computation,
        test_compute_source_flags_mapping,
        test_ninja_rewrite,
        test_ninja_rewrite_noops_without_json,
        test_export_merge_roundtrip,
        test_macro_drift_against_header,
        test_torch_integration_real_ninja_writer,
    ]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    if _FAILURES:
        print(f"\nFAILED ({len(_FAILURES)} assertion failure(s)):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\nOK — all {len(tests)} test groups passed.")


if __name__ == "__main__":
    main()
