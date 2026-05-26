"""Runtime-specialized kernel JIT via NVRTC / hipRTC.

Activated when ``build(..., enable_runtime_specialization=True)`` or the
CLI is invoked with ``--enable-runtime-specialization``.

At runtime, hot kernels can request specialization for a specific
``(arch, dtype, shape_class)``. The registry checks its CUBIN cache;
on miss, calls NVRTC / hipRTC with ``constexpr``-baked constants and
caches the result on disk. In-process module handles are memoised in a
thread-safe dict so dispatch on cache hit is sub-microsecond.

Why bucket shapes instead of keying on exact shape?
    NVRTC compilation costs ~50-500ms per kernel; if we keyed the cache
    on the exact (D0, D1, ..., Dn) tuple we'd recompile on every minor
    shape change. The four-bucket scheme (tiny / small / medium / large
    / huge) preserves the wins from constant-folded loop bounds without
    blowing up the cache.

Graceful degradation:
    * If cuda-python / hip-python isn't installed, ``dispatch`` raises
      ``RegistryError`` — never crashes the host build.
    * ``initialize_registry`` swallows that error and writes a single
      ``[nvrtc] disabled: ...`` line to the build report, so a CPU-only
      build host can still complete ``build()`` even with the flag on.

Pallas archs are rejected at construction time — Pallas owns its own
JIT path (XLA) and runtime specialization belongs there, not here.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


class RegistryError(RuntimeError):
    """Raised when runtime specialization cannot proceed.

    Common causes: unknown arch, Pallas vendor, missing cuda-python /
    hip-python binding, NVRTC/hipRTC compile failure.
    """


# ---------------------------------------------------------------------------
# Shape bucketing
# ---------------------------------------------------------------------------

# Threshold table (inclusive upper bounds, in element count):
#   tiny    < 1 Ki        (warm-up / scalar reductions)
#   small   < 64 Ki       (typical attention head dims)
#   medium  < 4 Mi        (4096x4096 matmul tile)
#   large   < 256 Mi      (full transformer hidden layer)
#   huge    everything else (KV cache, activation snapshots)
_SHAPE_BUCKETS: Tuple[Tuple[int, str], ...] = (
    (1024,        "tiny"),
    (65536,       "small"),
    (4_194_304,   "medium"),
    (268_435_456, "large"),
)

# Representative SIZE_HINT to pass into the kernel template per bucket.
# Templates can use this for constexpr unroll / shared-mem sizing.
_SHAPE_HINT_SIZE: Dict[str, int] = {
    "tiny":   256,
    "small":  4_096,
    "medium": 65_536,
    "large":  1_048_576,
    "huge":   16_777_216,
}


def _shape_class(shape: Tuple[int, ...]) -> str:
    """Bucket a shape into one of {tiny, small, medium, large, huge}.

    Empty shape == scalar == "tiny".
    """
    n = 1
    for d in shape:
        n *= int(d)
    for upper, label in _SHAPE_BUCKETS:
        if n < upper:
            return label
    return "huge"


def _shape_class_size(cls: str) -> int:
    """Return the representative element count for a shape bucket."""
    return _SHAPE_HINT_SIZE.get(cls, 4_096)


# ---------------------------------------------------------------------------
# Default kernel source template
# ---------------------------------------------------------------------------

# Map dtype mnemonics → actual C++ types that NVRTC/hipRTC understand.
# Defaults to the input string so callers can pass a raw C++ type if they
# like (e.g. dtype="float").
_DTYPE_C_TYPE: Dict[str, str] = {
    "fp32":  "float",
    "f32":   "float",
    "float": "float",
    "fp16":  "__half",
    "f16":   "__half",
    "half":  "__half",
    "bf16":  "__nv_bfloat16",
    "bfloat16": "__nv_bfloat16",
    "fp64":  "double",
    "f64":   "double",
    "double": "double",
}


def _resolve_ctype(dtype: str) -> str:
    """Translate a dtype mnemonic to a real C++ type name."""
    return _DTYPE_C_TYPE.get(dtype, dtype)


def _default_template_provider(op: str, arch: str) -> str:
    """Return a NVRTC/hipRTC source template with ``{DTYPE}``,
    ``{SHAPE_DIMS}``, ``{SIZE_HINT}`` placeholders for ``.format()``.

    The default template is a trivial element-wise copy kernel — enough
    to validate that NVRTC / hipRTC can compile *something*. Real callers
    should pass their own ``template_provider`` to KernelRegistry.

    No headers are included so the template compiles cleanly even on
    minimal hosts where ``cuda_fp16.h`` / ``cuda_bf16.h`` aren't on the
    NVRTC include path. Callers that need ``__half`` / ``__nv_bfloat16``
    must supply their own ``template_provider`` (and arrange for the
    headers to be reachable).
    """
    return r"""
extern "C" __global__ void specialized_{OP}_kernel({DTYPE}* out, const {DTYPE}* in, int n) {{
    constexpr int kSizeHint = {SIZE_HINT};
    constexpr int kShapeDims = {SHAPE_DIMS};
    (void)kSizeHint; (void)kShapeDims;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}}
""".replace("{OP}", op)


# ---------------------------------------------------------------------------
# KernelRegistry — per-arch dispatcher
# ---------------------------------------------------------------------------

class KernelRegistry:
    """Sub-microsecond runtime-specialized kernel lookup.

    Usage::

        reg = KernelRegistry(arch="sm_90a",
                              cache_dir=Path("~/.cache/sg/nvrtc"))
        handle = reg.dispatch("adamw", dtype="fp32", shape=(4096, 4096))
        handle(out_ptr, in_ptr, n)   # live invocation (GPU-only)

    Construction does NOT require cuda-python / hip-python — only
    ``dispatch`` does, and only on cache miss. This lets a CPU-only
    AOT host construct the registry and pre-warm a few entries without
    importing GPU libraries.
    """

    def __init__(self, arch: str, cache_dir: Path,
                 template_provider: Optional[Callable[[str, str], str]] = None):
        # Defer the ARCH_TABLE import to avoid a circular dependency at
        # module import time (compile.py imports kernel_registry late).
        from grokking_optimizers.compile import ARCH_TABLE
        if arch not in ARCH_TABLE:
            raise RegistryError(f"unknown arch {arch!r}")
        entry = ARCH_TABLE[arch]
        if entry.vendor == "pallas":
            raise RegistryError(
                f"KernelRegistry not applicable to Pallas arch {arch!r}; "
                "use JAX/XLA jit directly.")
        self.arch = arch
        self.vendor = entry.vendor
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._handle_cache: Dict[str, Any] = {}
        self._template_provider = template_provider or _default_template_provider

    # ------------------------------------------------------------------
    # Public dispatch entry point
    # ------------------------------------------------------------------

    def _key(self, op: str, dtype: str, shape_cls: str) -> str:
        """Stable 24-hex-char cache key (sha256 truncated)."""
        return hashlib.sha256(
            f"{self.arch}|{op}|{dtype}|{shape_cls}".encode()
        ).hexdigest()[:24]

    def dispatch(self, op: str, dtype: str, shape: Tuple[int, ...]):
        """Return a callable kernel handle. Sub-µs on in-process cache hit.

        First call for a (op, dtype, shape_class) tuple performs the
        NVRTC/hipRTC compile (~50-500ms); subsequent calls hit the
        in-process module dict.
        """
        shape_cls = _shape_class(tuple(shape))
        key = self._key(op, dtype, shape_cls)
        with self._lock:
            cached = self._handle_cache.get(key)
            if cached is not None:
                return cached
        cubin_path = self.cache_dir / f"{key}.cubin"
        if cubin_path.exists() and cubin_path.stat().st_size > 0:
            handle = self._load_cubin(cubin_path, op)
        else:
            cubin = self._compile(op, dtype, shape_cls, tuple(shape))
            # Write atomically: write to .tmp then rename so a crash mid-write
            # doesn't poison the cache.
            tmp_path = cubin_path.with_suffix(".cubin.tmp")
            tmp_path.write_bytes(cubin)
            os.replace(tmp_path, cubin_path)
            handle = self._load_cubin(cubin_path, op)
        with self._lock:
            self._handle_cache[key] = handle
        return handle

    # ------------------------------------------------------------------
    # Internal — compile + load
    # ------------------------------------------------------------------

    def _compile(self, op: str, dtype: str, shape_cls: str,
                 example_shape: Tuple[int, ...]) -> bytes:
        src = self._template_provider(op, self.arch).format(
            DTYPE=_resolve_ctype(dtype),
            SHAPE_DIMS=len(example_shape),
            SIZE_HINT=_shape_class_size(shape_cls),
        )
        if self.vendor == "cuda":
            return self._nvrtc_compile(src)
        if self.vendor == "hip":
            return self._hiprtc_compile(src)
        # Should be unreachable — __init__ rejects pallas.
        raise RegistryError(f"no rt-compile path for vendor {self.vendor!r}")

    def _nvrtc_compile(self, src: str) -> bytes:
        """Compile a CUDA source string to CUBIN (or PTX fallback) via NVRTC.

        Handles both the ``cuda.bindings.nvrtc`` (cuda-python >= 12.x) and
        the legacy ``cuda.nvrtc`` import paths. Different cuda-python
        versions return ``(err, value)`` tuples vs. raw values from each
        nvrtc* call; we normalise via ``_first_payload``.
        """
        try:
            from cuda.bindings import nvrtc  # type: ignore  # cuda-python >= 12.x
        except ImportError:
            try:
                from cuda import nvrtc  # type: ignore  # legacy
            except ImportError as exc:
                raise RegistryError(
                    "NVRTC requires cuda-python (pip install cuda-python)"
                ) from exc

        from grokking_optimizers.compile import ARCH_TABLE
        entry = ARCH_TABLE[self.arch]
        compute = f"compute_{entry.cutlass_arch}{entry.arch_suffix}"

        create_res = nvrtc.nvrtcCreateProgram(
            src.encode(), b"kernel.cu", 0, [], [])
        prog = _first_payload(create_res)
        if prog is None:
            raise RegistryError(f"nvrtcCreateProgram failed: {create_res!r}")

        opts = [f"-arch={compute}".encode(),
                b"-default-device",
                b"--std=c++17"]
        compile_res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if _err_value(compile_res) != 0:
            # Try to fetch the log for the user.
            log = ""
            try:
                log_sz = _first_payload(nvrtc.nvrtcGetProgramLogSize(prog)) or 0
                if log_sz:
                    buf = bytearray(log_sz)
                    nvrtc.nvrtcGetProgramLog(prog, buf)
                    log = buf.decode("utf-8", "replace")
            except Exception:
                pass
            raise RegistryError(f"NVRTC compile failed: {compile_res!r}\n{log}")

        # Prefer CUBIN (binary, ready to load) when available; fall back to
        # PTX (text, needs a driver JIT step) otherwise. Some hosts return
        # a 0-byte CUBIN (e.g. CPU-only NVRTC with no SASS backend for the
        # target arch); treat that the same as "CUBIN unsupported" and use
        # PTX instead so the cubin cache file isn't empty.
        cubin_ok = (hasattr(nvrtc, "nvrtcGetCUBINSize")
                    and hasattr(nvrtc, "nvrtcGetCUBIN"))
        if cubin_ok:
            size = _first_payload(nvrtc.nvrtcGetCUBINSize(prog)) or 0
            if size > 0:
                buf = bytearray(size)
                nvrtc.nvrtcGetCUBIN(prog, buf)
                return bytes(buf)
        size = _first_payload(nvrtc.nvrtcGetPTXSize(prog)) or 0
        buf = bytearray(size)
        nvrtc.nvrtcGetPTX(prog, buf)
        return bytes(buf)

    def _hiprtc_compile(self, src: str) -> bytes:
        """Compile a HIP source string to code-object via hipRTC."""
        try:
            from hip import hiprtc  # type: ignore  # hip-python
        except ImportError as exc:
            raise RegistryError(
                "hipRTC requires hip-python (pip install hip-python)"
            ) from exc

        from grokking_optimizers.compile import ARCH_TABLE
        offload = ARCH_TABLE[self.arch].hipcc_offload_arch
        prog = _first_payload(
            hiprtc.hiprtcCreateProgram(src.encode(), b"kernel.hip", 0, [], []))
        if prog is None:
            raise RegistryError("hiprtcCreateProgram failed")
        opts = [f"--offload-arch={offload}".encode()]
        compile_res = hiprtc.hiprtcCompileProgram(prog, len(opts), opts)
        if _err_value(compile_res) != 0:
            raise RegistryError(f"hipRTC compile failed: {compile_res!r}")
        size = _first_payload(hiprtc.hiprtcGetCodeSize(prog)) or 0
        buf = bytearray(size)
        hiprtc.hiprtcGetCode(prog, buf)
        return bytes(buf)

    def _load_cubin(self, cubin_path: Path, op: str):
        """Return a callable kernel handle backed by ``cubin_path``.

        A real GPU implementation would call cuModuleLoadData /
        hipModuleLoadData here and resolve the entry-point function.
        For host-side use (and the self-test harness) we return a thin
        stub that exposes the cubin path and raises NotImplementedError
        when actually invoked.
        """
        return _LoadedKernel(cubin_path, op)


# ---------------------------------------------------------------------------
# cuda-python return-value normalization helpers
# ---------------------------------------------------------------------------
#
# cuda-python's nvrtc bindings return either ``(err, payload)`` tuples or
# a raw payload (depending on version and which function). hip-python is
# similar. These helpers paper over both shapes so the compile path
# stays readable.

def _first_payload(res):
    """Extract the payload from an ``(err, value)`` tuple or pass through.

    Returns ``None`` if the structure isn't recognized.
    """
    if isinstance(res, tuple):
        if len(res) >= 2:
            return res[1]
        return None
    return res


def _err_value(res) -> int:
    """Return the integer error code from an ``(err, ...)`` tuple, else 0."""
    if isinstance(res, tuple) and res:
        err = res[0]
        v = getattr(err, "value", err)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    return 0


# ---------------------------------------------------------------------------
# Loaded-kernel stub
# ---------------------------------------------------------------------------

class _LoadedKernel:
    """Thin placeholder for a loaded CUBIN module.

    Production usage would wrap a ``CUmodule`` / ``hipModule_t`` and
    drive ``cuLaunchKernel`` / ``hipModuleLaunchKernel``. For the test
    harness and for host-side metadata inspection, we just keep the
    cubin path and the op name.
    """

    __slots__ = ("cubin_path", "op")

    def __init__(self, cubin_path: Path, op: str):
        self.cubin_path = cubin_path
        self.op = op

    def __repr__(self) -> str:  # pragma: no cover  (debug aid)
        return f"<Kernel op={self.op} cubin={self.cubin_path.name}>"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "live kernel invocation requires a GPU + cuda-python context; "
            "this is the host-side stub. Build a real launcher around "
            "self.cubin_path with cuModuleLoadData / hipModuleLoadData.")


# ---------------------------------------------------------------------------
# Public API — process-global registry table
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, KernelRegistry] = {}
_REGISTRY_LOCK = threading.Lock()


def get_registry(arch: str,
                 cache_dir: Optional[Path] = None) -> KernelRegistry:
    """Get-or-create a process-global ``KernelRegistry`` for ``arch``.

    Subsequent calls for the same arch return the cached instance. The
    ``cache_dir`` is only honoured on first creation; later calls keep
    the original directory.
    """
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(arch)
        if existing is not None:
            return existing
        cd = cache_dir or (
            Path(os.environ.get("HOME", "/tmp"))
            / ".cache" / "supergrok" / "nvrtc")
        reg = KernelRegistry(arch, cd)
        _REGISTRY[arch] = reg
        return reg


def initialize_registry(spec, report=None) -> Optional[KernelRegistry]:
    """Pre-warm the registry from inside ``build()``.

    Called when ``spec.enable_runtime_specialization`` is True. Attempts
    one trivial dispatch per common dtype to populate the on-disk cubin
    cache. Any RegistryError (missing cuda-python, Pallas arch, NVRTC
    compile failure) is caught and logged — never propagated, so the
    surrounding build doesn't fail because of an opt-in feature.
    """
    if not getattr(spec, "enable_runtime_specialization", False):
        return None
    try:
        cache_dir = Path(spec.out_dir) / "nvrtc_cache"
        reg = get_registry(spec.arch, cache_dir=cache_dir)
    except RegistryError as exc:
        if report is not None:
            report.write(f"[nvrtc] disabled: {exc}\n")
        return None

    op = getattr(spec, "optimizer", "kernel")
    # Only prewarm with dtypes the default template can compile without
    # external headers. Callers wiring in their own template_provider that
    # pulls in cuda_fp16.h / cuda_bf16.h should prewarm those dtypes
    # themselves after constructing the registry.
    for dtype in ("fp32", "fp64"):
        try:
            reg.dispatch(op, dtype, (1024,))
        except RegistryError as exc:
            if report is not None:
                report.write(f"[nvrtc] {dtype} prewarm skipped: {exc}\n")
    if report is not None:
        report.write(f"[nvrtc] registry initialized for {spec.arch} "
                     f"(cache={reg.cache_dir})\n")
    return reg
