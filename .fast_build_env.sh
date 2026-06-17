# Fast-iteration build env for the SuperGrok1.5 perf ratchet: caching + auto-threaded nvcc + ninja.
# Source this BEFORE any build/gate/roofline_bench/compile command:   source .fast_build_env.sh
#
# WHAT IT FIXES (each verified 2026-06-16):
#  1. DEVICE-COMPILE CACHE — the expensive cu->cubin (~200s/TU). torch called nvcc DIRECTLY, so
#     sccache never saw it (0 requests). We route nvcc through sccache via torch's PYTORCH_NVCC
#     hook (cpp_extension.py L2308 — used by BOTH the JIT bench path and the AOT _ops build).
#     The wrapper .build_tools/nvcc-cached execs `sccache nvcc` => CUDA-native cache (verified
#     100% hit on a recompiled config, "Cache hits (CUDA)").
#  2. HOST-COMPILE CACHE — the C++ bindings/stubs. CXX="ccache g++"; torch's COLLECT_GCC probe
#     (L322) sees through the wrapper so its ABI check still passes.
#  3. AUTO-THREADED nvcc — the wrapper adds `--threads 0`, so nvcc auto-detects the machine's
#     CPU count (parallel gencode/compile phases). Replaces the old FIXED, IGNORED NVCC_THREADS=8
#     (torch 2.4.1 never forwarded NVCC_THREADS).
#  4. NINJA + MAX_JOBS — torch uses ninja by default; MAX_JOBS=$(nproc) parallelizes TUs.
#  Together (1)+(2) = the FULL compile is cached (device + host), not part of it.
#
# The gendeps unblock is still required: torch's --generate-dependencies-with-compile makes the
# nvcc command non-cacheable for sccache; TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1 strips it.
# Cache-hit CORRECTNESS is unaffected: caches hash the PREPROCESSED source, so any header/macro
# change forces a real recompile (a stale hit is impossible).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PATH="/workspace/.local/bin:$PATH"                 # sccache lives here
# FIX (2026-06-17): torch's check_compiler_ok_for_platform (cpp_extension.py L320) runs
# `<basename CXX> -v`, i.e. the BARE name `g++-cached`, NOT the full $CXX path. The wrapper is
# at $ROOT/.build_tools/g++-cached which is on no PATH dir, so the autotuner's --jit-only build
# died at the ABI probe ("g++-cached: command not found") for EVERY cell. (The bench/load() path
# tolerated the full path, which is why the re-profile built fine.) Symlink the wrapper into the
# PATH dir so the basename resolves too — keeps full nvcc+ccache caching intact.
mkdir -p /workspace/.local/bin 2>/dev/null
ln -sf "$ROOT/.build_tools/g++-cached" /workspace/.local/bin/g++-cached 2>/dev/null || true
export TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1        # unblock: strip the gendeps flag
export PYTORCH_NVCC="$ROOT/.build_tools/nvcc-cached"       # nvcc -> sccache + --threads 0 (device)
export CXX="$ROOT/.build_tools/g++-cached"                 # g++ -> ccache (host stubs); single-token path for torch's which()
export SCCACHE_DIR="$ROOT/.build_cache/sccache"   # PERSISTENT (was /dev/shm ramdisk -> wiped on instance close). Volume-backed so fast-recompile survives across sessions.
export SCCACHE_CACHE_SIZE=20G
export CCACHE_DIR="$ROOT/.build_cache/ccache"     # PERSISTENT (volume-backed, survives instance close)
export CCACHE_MAXSIZE=10G
export CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime
export TMPDIR=/dev/shm/tmp                                 # nvcc cicc/ptxas scratch on ramdisk
export MAX_JOBS="$(nproc)"                                 # ninja TU parallelism (auto-detect)
unset NVCC_THREADS 2>/dev/null || true                    # superseded by --threads 0 in the wrapper
mkdir -p /dev/shm/tmp "$ROOT/.build_cache/sccache" "$ROOT/.build_cache/ccache" 2>/dev/null
command -v sccache >/dev/null 2>&1 && sccache --start-server >/dev/null 2>&1 || true
echo "[fast-build-env] caching ON: nvcc->sccache(CUDA) + g++->ccache | nvcc --threads 0 (auto=$(nproc)) | MAX_JOBS=$(nproc) | ninja"
