"""grokking_optimizers.compile — Targeted per-(optimizer, model, arch) build.

Dev-time companion to ``setup.py``. Where ``pip install -e .`` builds the
full ``grokking_optimizers._ops`` extension with the default arch flags,
this module compiles a focused, maximally-optimised build for a single
``(optimizer, model, arch)`` triple. It owns the **build + autotune**
half of the dev pipeline; the **profile** half lives in the sibling
module ``grokking_optimizers.profile`` (which can also be invoked
standalone against any pre-built artefact).

What it does, in order:

  1. **Resolve sources.** For the chosen arch, gather:
       - ``csrc/bindings/*.cpp``                 (dispatch + bindings)
       - ``csrc/backends/<vendor>/<arch>/launch_*.{cu,hip.cpp,hip}``
       - ``csrc/backends/<vendor>/<arch>/models/*.{cu,hip.cpp}``
     The full launcher + model set for the arch is compiled so the bindings
     link cleanly; the selected ``(optimizer, model)`` is signalled to the
     algorithm headers through ``SG_BUILD_*`` macros so the headers can
     ``#if`` out unused specialisations.

  2. **Metaprogramming hooks.** Injects:
       -DSG_BUILD_OPTIMIZER_<UPPER>=1
       -DSG_BUILD_MODEL_<UPPER>=1
       -DSG_BUILD_ARCH_<SM90|GFX942|TPU_V5P>=1
       -DSG_VERBOSE=1

  3. **Build with Ninja, ``-j$(nproc)``.** Drives
     ``torch.utils.cpp_extension.load`` with ``MAX_JOBS=NCPUS`` so the
     underlying ninja invocation parallelises over every available core.

  4. **Arch-tuned codegen + LTO.**
       sm_90  -> -O3 --use_fast_math -lineinfo -Xptxas=-O3,-v,--warn-on-spills
                 --maxrregcount=255 -gencode arch=compute_90,code=sm_90
                 -gencode arch=compute_90,code=compute_90 -dlto
                 -Xcompiler -flto       (CUTLASS auto-enabled if present)
       gfx942 -> -O3 -ffast-math --offload-arch=gfx942
                 -Rpass-analysis=kernel-resource-usage -ggdb -flto

  5. **Optional autotune pass.** Sets ``AUTOTUNE_PASS=1`` + builds with
     stub configs, then rewrites ``csrc/algorithms/tuned_configs.h`` so a
     subsequent build picks up the tuned constants.

  6. **Optional profile pass.** Delegates to
     ``grokking_optimizers.profile.profile`` (ncu / rocprof / jax.profiler).
     Pass ``--no-profile`` to skip this step and invoke ``profile`` later
     against the produced .so.

CLI:
    python -m grokking_optimizers.compile \\
        --optimizer supergrok2 --model mamba --arch sm_90 \\
        [--no-autotune] [--no-profile] [--report build_report.txt]

Importable:
    from grokking_optimizers.compile import build
    so_path = build(optimizer="supergrok2", model="mamba", arch="sm_90")

The compile pass is dev-only — the production race driver continues to
load the full ``grokking_optimizers._ops`` built by ``setup.py``.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from grokking_optimizers.profile import (
    ARCH_INFO,
    ARCHES,
    MODELS,
    NCPUS,
    OPTIMIZERS,
    OPT_CLASS,  # noqa: F401  (re-exported for back-compat)
    REPO_ROOT,
    _dispatch_profile,
    env_overlay,
    make_progress,
)


@dataclass
class BuildSpec:
    optimizer: str
    model: str
    arch: str
    out_dir: Path
    autotune: bool = True
    profile: bool = True
    verbose: bool = False
    extra_macros: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source / flag resolution
# ---------------------------------------------------------------------------

def _validate(spec: BuildSpec) -> None:
    if spec.optimizer not in OPTIMIZERS:
        raise ValueError(
            f"optimizer={spec.optimizer!r} not in {list(OPTIMIZERS)}")
    if spec.model not in MODELS:
        raise ValueError(f"model={spec.model!r} not in {list(MODELS)}")
    if spec.arch not in ARCHES:
        raise ValueError(f"arch={spec.arch!r} not in {list(ARCHES)}")


def _resolve_sources(spec: BuildSpec) -> List[Path]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "pallas":
        return []
    backend = REPO_ROOT / "csrc/backends" / info["subdir"]
    bindings = sorted((REPO_ROOT / "csrc/bindings").glob("*.cpp"))
    launchers: List[Path] = []
    for g in info["launcher_glob"]:
        launchers.extend(sorted(backend.glob(g)))
    models: List[Path] = []
    for g in info["model_glob"]:
        models.extend(sorted((backend / "models").glob(g)))
    return bindings + launchers + models


def _build_macros(spec: BuildSpec) -> List[str]:
    macros = [
        f"-DSG_BUILD_OPTIMIZER_{spec.optimizer.upper()}=1",
        f"-DSG_BUILD_MODEL_{spec.model.upper()}=1",
        f"-D{ARCH_INFO[spec.arch]['macro']}=1",
        "-DSG_VERBOSE=1",
    ]
    return macros + list(spec.extra_macros)


def _host_cflags(spec: BuildSpec) -> List[str]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "pallas":
        return []
    base = [
        "-O3", "-std=c++17", f"-D{info['host_define']}",
        "-ffast-math", "-funroll-loops", "-fPIC",
        "-flto",
    ]
    return base + _build_macros(spec)


def _device_cflags(spec: BuildSpec) -> List[str]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "cuda":
        base = [
            "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
            "--expt-relaxed-constexpr", "-lineinfo",
            "-Xptxas", "-O3",
            "-Xptxas", "-v",
            "-Xptxas", "--warn-on-spills",
            "-Xcompiler", "-fPIC",
            "-Xcompiler", "-flto",
            "--resource-usage",
            "--generate-line-info",
            "--maxrregcount=255",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
            "-dlto",
        ]
        if (REPO_ROOT / "third_party/cutlass/include").exists():
            base += ["-DWITH_CUTLASS", "-DCUTLASS_NVCC_ARCHS=90a"]
        return base + _build_macros(spec)
    if info["vendor"] == "hip":
        return [
            "-O3", "-std=c++17", "-DWITH_HIP",
            "-ffast-math", "-fPIC",
            "--offload-arch=gfx942",
            "-Rpass-analysis=kernel-resource-usage",
            "-ggdb",
            "-flto",
        ] + _build_macros(spec)
    return []


def _ldflags(spec: BuildSpec) -> List[str]:
    if ARCH_INFO[spec.arch]["vendor"] == "pallas":
        return []
    return ["-flto", "-Wl,--as-needed"]


def _include_paths() -> List[str]:
    return [
        str(REPO_ROOT / "csrc/bindings"),
        str(REPO_ROOT),
    ]


# ---------------------------------------------------------------------------
# Build (torch.utils.cpp_extension.load with ninja)
# ---------------------------------------------------------------------------

def _torch_load(spec: BuildSpec, sources: List[Path],
                host_cflags: List[str], device_cflags: List[str],
                ldflags: List[str], report) -> Optional[Path]:
    """Compile via torch.utils.cpp_extension.load — ninja + nproc."""
    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        report.write(f"  [load] torch not importable: {exc}\n")
        return None

    module_name = (
        f"grokking_compiled_{spec.optimizer}_{spec.model}_{spec.arch}"
    )
    build_dir = spec.out_dir / module_name
    build_dir.mkdir(parents=True, exist_ok=True)

    report.write(f"  module:    {module_name}\n")
    report.write(f"  build dir: {build_dir}\n")
    report.write(f"  sources:   {len(sources)} files\n")
    for s in sources:
        report.write(f"             {s.relative_to(REPO_ROOT)}\n")
    report.write(f"  host:      {' '.join(host_cflags)}\n")
    report.write(f"  device:    {' '.join(device_cflags)}\n")
    report.write(f"  ldflags:   {' '.join(ldflags)}\n")
    report.flush()

    with_cuda = ARCH_INFO[spec.arch]["vendor"] in ("cuda", "hip")
    with env_overlay(
        MAX_JOBS=NCPUS,
        NINJA_STATUS="[%f/%t %es] ",
        TORCH_CUDA_VERBOSE_BUILD="1",
        CMAKE_BUILD_PARALLEL_LEVEL=NCPUS,
    ):
        t0 = time.monotonic()
        try:
            load(
                name=module_name,
                sources=[str(s) for s in sources],
                extra_cflags=host_cflags,
                extra_cuda_cflags=device_cflags,
                extra_ldflags=ldflags,
                extra_include_paths=_include_paths(),
                build_directory=str(build_dir),
                verbose=spec.verbose,
                with_cuda=with_cuda,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            report.write(f"\n[build FAILED after {elapsed:.1f}s]\n{exc}\n")
            return None

    elapsed = time.monotonic() - t0
    so = next(build_dir.glob(f"{module_name}*.so"), None)
    report.write(f"\n[build OK in {elapsed:.1f}s] -> {so}\n")
    return so


# ---------------------------------------------------------------------------
# Autotune pass
# ---------------------------------------------------------------------------

def _autotune_pass(spec: BuildSpec, sources: List[Path],
                   host_cflags: List[str], device_cflags: List[str],
                   ldflags: List[str], report) -> bool:
    """First-pass build with stub configs + write tuned_configs.h.

    The real codebase autotune would micro-bench across {block_size,
    vec_width, unroll} and rewrite this header with the winning values.
    Here we run the AUTOTUNE_PASS build to validate that the
    metaprogramming guards compile, then emit a tuned_configs.h with
    sensible per-arch defaults so the final pass can pick them up.
    """
    report.write("\n--- AUTOTUNE PASS 1 (AUTOTUNE_PASS=1, stub configs) ---\n")
    with env_overlay(AUTOTUNE_PASS=1):
        so = _torch_load(
            spec, sources,
            host_cflags + ["-DAUTOTUNE_PASS=1"],
            device_cflags + ["-DAUTOTUNE_PASS=1"],
            ldflags, report,
        )
    if so is None:
        report.write("  [autotune] pass-1 build failed; skipping tune step.\n")
        return False

    tuned_h = REPO_ROOT / "csrc/algorithms/tuned_configs.h"
    # sm_90 vs gfx942: warp width differs (32 vs 64), so block-size defaults
    # diverge. These are starting points; a real autotune sweep would
    # overwrite them per-(optimizer, model) combo.
    block = 256 if spec.arch == "sm_90" else 512
    vec_width = 4 if spec.arch == "sm_90" else 4
    unroll = 8
    tuned_h.write_text(textwrap.dedent(f"""\
        // Auto-generated by grokking_optimizers.compile autotune pass.
        // Do not edit by hand — re-run `python -m grokking_optimizers.compile`.
        //
        // Combo: optimizer={spec.optimizer} model={spec.model} arch={spec.arch}
        // Host cores at tune time: {NCPUS}
        // Generated: {datetime.datetime.now().isoformat()}
        #pragma once
        #ifndef SG_TUNED_BLOCK_SIZE
        #define SG_TUNED_BLOCK_SIZE {block}
        #endif
        #ifndef SG_TUNED_VEC_WIDTH
        #define SG_TUNED_VEC_WIDTH {vec_width}
        #endif
        #ifndef SG_TUNED_UNROLL
        #define SG_TUNED_UNROLL {unroll}
        #endif
    """))
    report.write(f"  [autotune] wrote {tuned_h.relative_to(REPO_ROOT)} "
                 f"(block={block} vec={vec_width} unroll={unroll})\n")
    return True


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def build(
    optimizer: str,
    model: str,
    arch: str,
    *,
    out_dir: Optional[Path] = None,
    autotune: bool = True,
    profile: bool = True,
    report_path: Optional[Path] = None,
    verbose: bool = False,
    extra_macros: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Compile the (optimizer × model × arch) pipeline.

    Returns the path to the built .so (or ``None`` for pallas / on failure).
    Full diagnostics — sources, flags, ninja/nvcc/hipcc output, and the
    optional profile capture — are written to ``report_path``. The profile
    capture is delegated to ``grokking_optimizers.profile``; pass
    ``profile=False`` to skip it and run ``python -m grokking_optimizers.profile``
    separately.
    """
    spec = BuildSpec(
        optimizer=optimizer,
        model=model,
        arch=arch,
        out_dir=(out_dir or (REPO_ROOT / "build" / "compiled")).resolve(),
        autotune=autotune,
        profile=profile,
        verbose=verbose,
        extra_macros=list(extra_macros or []),
    )
    _validate(spec)
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_path or (
        spec.out_dir / f"compile_{optimizer}_{model}_{arch}.txt"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    info = ARCH_INFO[arch]
    phases: List[str] = ["resolve"]
    if info["vendor"] != "pallas":
        if autotune:
            phases += ["build:autotune", "build:final"]
        else:
            phases += ["build:final"]
    if profile:
        phases += ["profile"]

    step, close = make_progress(len(phases), f"{optimizer}/{model}/{arch}")

    so_path: Optional[Path] = None
    try:
        with open(report_path, "w") as report:
            report.write("# grokking_optimizers.compile — targeted build\n")
            report.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
            report.write(f"# Optimizer: {optimizer}\n")
            report.write(f"# Model:     {model}\n")
            report.write(f"# Arch:      {arch} (vendor={info['vendor']})\n")
            report.write(f"# CPU cores: {NCPUS} (ninja -j)\n")
            report.write(f"# Out dir:   {spec.out_dir}\n")
            report.write(f"# Autotune:  {autotune}\n")
            report.write(f"# Profile:   {profile}\n")

            sources = _resolve_sources(spec)
            host_cflags = _host_cflags(spec)
            device_cflags = _device_cflags(spec)
            ldflags = _ldflags(spec)
            report.write(f"\n[resolve] {len(sources)} source files for arch={arch}\n")
            step("resolve")

            if info["vendor"] == "pallas":
                launcher = (REPO_ROOT / "csrc/backends/pallas"
                            / f"launch_{optimizer}.py")
                report.write("\n[pallas] no C++ compile; Python launcher only\n")
                report.write(f"  launcher: {launcher}\n")
                report.write(f"  exists:   {launcher.exists()}\n")
            else:
                if autotune:
                    _autotune_pass(
                        spec, sources, host_cflags, device_cflags,
                        ldflags, report,
                    )
                    step("autotune")
                report.write("\n--- FINAL BUILD PASS ---\n")
                so_path = _torch_load(
                    spec, sources, host_cflags, device_cflags, ldflags, report,
                )
                step("build")

            if profile:
                report.write("\n--- PROFILE PASS "
                             "(via grokking_optimizers.profile) ---\n")
                _dispatch_profile(optimizer, model, arch, report)
                step("profile")

            report.write(f"\n# Final .so: {so_path}\n")
    finally:
        close()

    return so_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grokking_optimizers.compile",
        description="Targeted per-(optimizer, model, arch) build pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--optimizer", "-O", required=True, choices=OPTIMIZERS,
                        help="Optimizer name (csrc/algorithms/<name>.h)")
    parser.add_argument("--model", "-M", required=True, choices=MODELS,
                        help="Model name (csrc/backends/*/models/<name>.*)")
    parser.add_argument("--arch", "-A", required=True, choices=ARCHES,
                        help="Target arch")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "build" / "compiled",
                        help="Build artifact directory (default: build/compiled)")
    parser.add_argument("--report", type=Path, default=None,
                        help="Report file path (default: <out>/compile_<O>_<M>_<A>.txt)")
    parser.add_argument("--no-autotune", action="store_true",
                        help="Skip the AUTOTUNE_PASS pre-build")
    parser.add_argument("--no-profile", action="store_true",
                        help="Skip the ncu/rocprof/jax.profiler pass "
                             "(run `python -m grokking_optimizers.profile` "
                             "later against the produced .so)")
    parser.add_argument("-D", dest="extra_macros", action="append", default=[],
                        help="Extra -D<MACRO[=VALUE]> to inject into both "
                             "host and device cflags. Repeatable.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose torch cpp_extension output")
    args = parser.parse_args(argv)

    extra = [
        m if m.startswith("-D") else f"-D{m}"
        for m in args.extra_macros
    ]

    so = build(
        optimizer=args.optimizer, model=args.model, arch=args.arch,
        out_dir=args.out,
        autotune=not args.no_autotune,
        profile=not args.no_profile,
        report_path=args.report,
        verbose=args.verbose,
        extra_macros=extra,
    )

    report = args.report or (
        args.out / f"compile_{args.optimizer}_{args.model}_{args.arch}.txt"
    )
    sys.stdout.write(f"{report}\n")
    return 0 if (so is not None or ARCH_INFO[args.arch]["vendor"] == "pallas") else 1


if __name__ == "__main__":
    raise SystemExit(main())
