"""CPU JIT Specializer — generates SIMD-specific optimized kernels.

Detects CPU capabilities:
  - x86_64: AVX-512 (native vs decoded), AVX2
  - aarch64: NEON, SVE, Apple AMX
  - Cache hierarchy: L1/L2/L3 sizes
  - NUMA topology
  - Physical core count

Generates C++ source with:
  - SIMD-width-matched vectorization pragmas
  - L3-aware tile sizes for scan+elem fusion
  - NUMA-aware thread pinning hints
  - Cache-line-aligned prefetch distances
"""

import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from .specializer import KernelSpecializer, ModelConfig


@dataclass
class CPUConfig:
    """Detected CPU hardware properties."""
    arch: str = 'x86_64'
    has_avx512: bool = False
    has_avx512_native: bool = False  # False on AMD (decoded as 2x256)
    has_avx2: bool = True
    has_sve: bool = False
    has_neon: bool = False
    has_amx: bool = False  # Apple AMX
    l1_size: int = 48 * 1024
    l2_size: int = 1024 * 1024
    l3_size: int = 32 * 1024 * 1024
    numa_nodes: int = 1
    physical_cores: int = 4


class CPUSpecializer(KernelSpecializer):
    """Generate CPU-specific optimized kernels.

    Produces C++ source with compile-time constants for SIMD width,
    cache tile sizes, and prefetch distances based on the detected
    CPU hardware.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.cpu = self._detect_cpu()

    def _detect_cpu(self) -> CPUConfig:
        """Probe CPU capabilities from /proc/cpuinfo or sysctl."""
        info = CPUConfig()
        info.arch = platform.machine()

        # Detect SIMD capabilities
        try:
            with open('/proc/cpuinfo', 'r') as f:
                flags = f.read()
            info.has_avx512 = 'avx512f' in flags
            # AMD Zen4+ has AVX-512 but decodes as 2x256-bit ops
            info.has_avx512_native = 'avx512f' in flags and 'AMD' not in flags
            info.has_avx2 = 'avx2' in flags
            info.has_sve = 'sve' in flags
        except FileNotFoundError:
            # macOS or other non-Linux
            info.has_neon = info.arch in ('arm64', 'aarch64')
            try:
                sysctl = subprocess.check_output(
                    ['sysctl', '-n', 'hw.optional.amx_version'],
                    stderr=subprocess.DEVNULL
                ).strip()
                info.has_amx = int(sysctl) > 0
            except Exception:
                info.has_amx = False

        # Cache sizes
        try:
            l1_str = open('/sys/devices/system/cpu/cpu0/cache/index0/size').read().strip()
            info.l1_size = _parse_cache_size(l1_str)
        except Exception:
            pass

        try:
            l2_str = open('/sys/devices/system/cpu/cpu0/cache/index2/size').read().strip()
            info.l2_size = _parse_cache_size(l2_str)
        except Exception:
            pass

        try:
            l3_str = open('/sys/devices/system/cpu/cpu0/cache/index3/size').read().strip()
            info.l3_size = _parse_cache_size(l3_str)
        except Exception:
            pass

        # NUMA
        try:
            import os as _os
            info.numa_nodes = len(_os.listdir('/sys/devices/system/node/'))
        except Exception:
            info.numa_nodes = 1

        # Core count (exclude hyperthreads)
        try:
            import os as _os
            total = _os.cpu_count() or 2
            info.physical_cores = max(1, total // 2)
        except Exception:
            info.physical_cores = 4

        return info

    def _cache_key(self) -> str:
        cpu = self.cpu
        cfg = self.config
        simd = self._select_simd()
        raw = (
            f"cpu_{cpu.arch}_{simd}_l3{cpu.l3_size}_"
            f"d{cfg.d_inner}_s{cfg.d_state}_e{cfg.num_experts}_"
            f"eh{cfg.expert_hidden}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def _select_simd(self) -> str:
        """Select the best SIMD ISA for this CPU."""
        if self.cpu.has_avx512_native:
            return 'avx512'
        elif self.cpu.has_avx2:
            return 'avx2'
        elif self.cpu.has_sve:
            return 'sve'
        elif self.cpu.has_neon:
            return 'neon'
        return 'scalar'

    def _generate_source(self) -> str:
        """Generate C++ source from Jinja2 templates with CPU-specific constants."""
        try:
            import jinja2
        except ImportError:
            raise RuntimeError(
                "JIT compilation requires jinja2. Install with: pip install jinja2"
            )

        simd = self._select_simd()

        # Compute L3-aware tile size for the fused scan+elem loop
        state_bytes_per_elem = 5 * 4  # ea, eas, mu, sharpness, gru_state * 4 bytes
        l3_tile = max(64, self.cpu.l3_size // state_bytes_per_elem // 2)

        # Prefetch distance: how many cache lines ahead to prefetch
        prefetch_distance = max(1, self.cpu.l2_size // (64 * 4))

        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'cpu')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            undefined=jinja2.StrictUndefined,
        )

        template_vars = dict(
            SIMD=simd,
            D_INNER=self.config.d_inner,
            D_STATE=self.config.d_state,
            NUM_EXPERTS=self.config.num_experts,
            EXPERT_HIDDEN=self.config.expert_hidden,
            GRU_HIDDEN=self.config.gru_hidden,
            L3_TILE_SIZE=l3_tile,
            PREFETCH_DISTANCE=prefetch_distance,
            OMP_THREADS=self.cpu.physical_cores,
            NUMA_AWARE=self.cpu.numa_nodes > 1,
            HAS_AVX512=self.cpu.has_avx512_native,
            HAS_AVX2=self.cpu.has_avx2,
            HAS_SVE=self.cpu.has_sve,
            HAS_NEON=self.cpu.has_neon,
        )

        try:
            template = env.get_template('fused_scan_elem.cpp.j2')
            return template.render(**template_vars)
        except Exception:
            return ''

    def _compile_source(self, source: str, key: str):
        """Compile C++ source via torch.utils.cpp_extension.load_inline."""
        if not source.strip():
            return None

        import torch
        simd = self._select_simd()

        extra_flags = ['-O3', '-march=native', '-fopenmp']
        if simd == 'avx512':
            extra_flags.append('-mavx512f')
        elif simd == 'avx2':
            extra_flags.append('-mavx2')

        return torch.utils.cpp_extension.load_inline(
            name=f'sg2_jit_cpu_{key}',
            cpp_sources=[source],
            build_directory=str(self.cache_dir),
            extra_cflags=extra_flags,
        )


def _parse_cache_size(s: str) -> int:
    """Parse cache size strings like '48K', '1024K', '32M'."""
    s = s.strip().upper()
    if s.endswith('K'):
        return int(s[:-1]) * 1024
    elif s.endswith('M'):
        return int(s[:-1]) * 1024 * 1024
    elif s.endswith('G'):
        return int(s[:-1]) * 1024 * 1024 * 1024
    return int(s)
