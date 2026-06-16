"""examples/autotune_demo/run_autotune.py — drive compile.py's REAL autotune
machinery over the naive tiled-GEMM demo.

WHY A DRIVER (which compile.py path this uses)
----------------------------------------------
The intended end-to-end path — ``python -m grokking_optimizers.compile
--optimizer gemm --model gemm ... --config compile_config.toml
--search-space gemm_search_space.yaml`` — does NOT work for a foreign
(non-grokking) project today: compile.py's ``build()`` calls ``_validate(spec)``
which hard-rejects any optimizer name outside the grokking profile
(``ValueError: optimizer='gemm' not in ['adamw', ...]`` — the long-documented
"Gap 3" in examples/toy_tune_project/README.md; confirmed still open against
the current compile.py). compile.py's tune-hook *timing+validation* seam is
otherwise fully wired (Gap 1 is closed: _make_variant_timer routes through
_hook_capture + _compare_outputs when spec.tune_hook is set).

So this driver takes the path the task names as the fallback: it drives the
SAME search via compile.py's flag/macro-injection + build + time + correctness
machinery DIRECTLY, importing the actual functions from
``grokking_optimizers.compile`` with ZERO edits to it:

  * config + space loaders : C.load_config(), C.get_search_space()
  * search enumeration      : C.cartesian(space, arch)           [real sampler]
  * macro injection         : C.resolve_macros(cfg, dims, "device")
                              C.resolve_extra_nvcc_flags(cfg, dims, arch)
  * stable config identity  : C.config_key(cfg)
  * variant timing+capture  : C._hook_capture(...)   [real subprocess protocol,
                              same one _make_variant_timer uses for the megakernel]
  * correctness gate        : C._compare_outputs(ref, cand, "fp32")  with
                              C.TOLERANCES["fp32"] == (1e-5, 1e-6)
  * winner selection        : minimize elapsed_ms among configs that pass the gate

Each variant .so is BUILT in its own subprocess (build_variant.py) and TIMED in
another fresh subprocess (compile.py's _hook_capture) — mandatory because the
GEMM op self-registers TORCH_LIBRARY(sg_demo, ...), and a TORCH_LIBRARY
namespace can be registered only once per process.

The reference is the strict-math (fast-math OFF) build of the NAIVE DEFAULT
config (16/16/16/256) — exactly what compile.py captures as its oracle — and
every variant's hook output is compared against it.

Run from the repo root:
    python -m examples.autotune_demo.run_autotune
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from grokking_optimizers import compile as C

HERE = Path(__file__).resolve().parent
REPO_ROOT = C.REPO_ROOT
KERNEL_CU = HERE / "gemm_kernel.cu"
BUILD_HELPER = HERE / "build_variant.py"
ARCH = "sm_90a"
DTYPE = "fp32"
TUNE_HOOK = "examples.autotune_demo.tune_hook:run"
OUT_DIR = Path("/tmp/autotune_demo_build")

# The in-source NAIVE DEFAULT (matches gemm_kernel.cu's #ifndef defaults). This
# is the reference config AND the baseline whose time we contrast with the
# winner. It is also the first member of the search-space Cartesian product.
NAIVE_DEFAULT = {"tile_m": 16, "tile_n": 16, "tile_k": 16, "block": 256}


def build_variant(macros, *, fastmath: bool, tag: str) -> Path:
    """Build one GEMM variant .so in a FRESH subprocess (build_variant.py) with
    the given -DSG_TUNED_* macro flags. fastmath=False is the STRICT-MATH build
    compile.py uses for the reference oracle. Returns the .so path."""
    cflags = ["-O3"] + list(macros)
    if fastmath:
        cflags += ["--use_fast_math"]
    cmd = [sys.executable, str(BUILD_HELPER), "--name", tag,
           "--cflags", json.dumps(cflags)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT),
                       timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"variant build {tag} failed (rc={r.returncode}): "
                           f"{r.stderr[-800:]}")
    so_line = [ln for ln in r.stdout.splitlines() if ln.startswith("SO_PATH ")]
    if not so_line:
        raise RuntimeError(f"variant build {tag} produced no SO_PATH: "
                           f"stdout={r.stdout[-400:]} stderr={r.stderr[-400:]}")
    return Path(so_line[-1][len("SO_PATH "):].strip())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants_dir = OUT_DIR / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    # --- Load the demo config + search space through compile.py's OWN loaders.
    cfg = C.load_config(HERE / "compile_config.toml")
    space = C.get_search_space(HERE / "gemm_search_space.yaml")
    dims = space[ARCH]["dims"]
    total = C.cartesian_count(space, ARCH)
    print(f"[demo] loaded config (tune_hook={cfg['project'].get('tune_hook')}) "
          f"+ search space: {total} configs over dims "
          f"{[d['name'] for d in dims]}")

    # A genuine compile.py BuildSpec — _hook_capture reads spec.model /
    # spec.optimizer / spec.arch from it. (opaque labels for a GEMM.)
    spec = C.BuildSpec(optimizer="gemm", model="gemm", arch=ARCH, out_dir=OUT_DIR)

    def macros_for(config):
        m = C.resolve_macros(config, dims, "device")
        m += C.resolve_extra_nvcc_flags(config, dims, ARCH)
        return m

    # --- 1) Reference oracle: STRICT-MATH (fast-math OFF) build of the NAIVE
    #        default config, captured via compile.py's _hook_capture (regime
    #        "normal", seed 0) — exactly compile.py's oracle posture.
    print("[demo] building STRICT-MATH reference (naive default "
          f"{NAIVE_DEFAULT}) ...")
    ref_so = build_variant(macros_for(NAIVE_DEFAULT), fastmath=False,
                           tag="ref_strict")
    ref_npy, ref_ms = C._hook_capture(
        TUNE_HOOK, str(ref_so), spec, regime="normal", seed=0,
        out_npy=variants_dir / "ref.npy")
    print(f"[demo] reference captured: output={ref_npy.name} "
          f"strict_ms={ref_ms:.4f}")

    # --- 2) Sweep every config (compile.py's cartesian enumerator). Build each
    #        variant (fast-math ON, the perf build), time+capture via
    #        _hook_capture, gate via _compare_outputs.
    results = []  # list of dicts
    naive_key = C.config_key(NAIVE_DEFAULT)
    t0 = time.monotonic()
    for i, config in enumerate(C.cartesian(space, ARCH)):
        ckey = C.config_key(config)
        tag = "v_" + C._short_key(ckey)
        try:
            so = build_variant(macros_for(config), fastmath=True, tag=tag)
            cand_npy, ms = C._hook_capture(
                TUNE_HOOK, str(so), spec, regime="normal", seed=0,
                out_npy=variants_dir / f"cand_{C._short_key(ckey)}.npy")
            status, max_rel = C._compare_outputs(ref_npy, cand_npy, DTYPE)
        except Exception as exc:
            print(f"  [{i+1}/{total}] {ckey}: BUILD/RUN FAIL {str(exc)[:160]}")
            results.append({"config": config, "key": ckey, "status": "build_fail",
                            "elapsed_ms": None, "max_rel": None})
            continue
        passed = status in ("ok", "deterministic")
        results.append({"config": config, "key": ckey, "status": status,
                        "elapsed_ms": ms, "max_rel": max_rel, "passed": passed})
        flag = "PASS" if passed else f"GATE-FAIL({status})"
        print(f"  [{i+1}/{total}] {ckey}: {ms:.4f} ms  {flag} "
              f"(max_rel={max_rel:.2e})")

    elapsed_total = time.monotonic() - t0

    # --- 3) Winner = min elapsed_ms among gate-PASS configs (compile.py's
    #        objective + correctness gate).
    passed = [r for r in results if r.get("passed")]
    if not passed:
        print("[demo] ERROR: no config passed the correctness gate")
        return 1
    winner = min(passed, key=lambda r: r["elapsed_ms"])
    naive = next((r for r in results if r["key"] == naive_key), None)
    naive_ms = naive["elapsed_ms"] if naive and naive.get("elapsed_ms") else None
    speedup = (naive_ms / winner["elapsed_ms"]) if naive_ms else None

    summary = {
        "n_configs_total": total,
        "n_tried": len(results),
        "n_passed_gate": len(passed),
        "n_build_fail": sum(1 for r in results if r["status"] == "build_fail"),
        "naive_default": NAIVE_DEFAULT,
        "naive_ms": naive_ms,
        "winner_config": winner["config"],
        "winner_ms": winner["elapsed_ms"],
        "speedup_naive_over_best": speedup,
        "tolerances_fp32": C.TOLERANCES.get("fp32"),
        "sweep_seconds": round(elapsed_total, 1),
        "all_results": [
            {k: v for k, v in r.items() if k != "config"} | {"config": r["config"]}
            for r in sorted(results, key=lambda r: (r["elapsed_ms"] is None,
                                                    r["elapsed_ms"] or 0))
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n========== AUTOTUNE DEMO SUMMARY ==========")
    print(f"configs tried        : {len(results)}/{total} "
          f"({len(passed)} passed gate, {summary['n_build_fail']} build-fail)")
    print(f"NAIVE default        : {NAIVE_DEFAULT}  ->  {naive_ms:.4f} ms")
    print(f"AUTOTUNER best       : {winner['config']}  ->  "
          f"{winner['elapsed_ms']:.4f} ms")
    if speedup:
        print(f"SPEEDUP (naive/best) : {speedup:.2f}x")
    print(f"correctness gate     : compile.py _compare_outputs, "
          f"fp32 TOLERANCES={C.TOLERANCES.get('fp32')} -> winner status "
          f"'{winner['status']}'")
    print(f"sweep wall time      : {elapsed_total:.1f}s")
    print(f"summary written      : {OUT_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
