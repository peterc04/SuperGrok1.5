"""Base class for hardware-specific kernel JIT compilation.

All specializers inherit from KernelSpecializer and implement:
  - _cache_key(): unique key for this hardware + model combination
  - _generate_source(): produce platform-specific kernel source code
  - _compile_source(): compile source into a loadable module
"""

import hashlib
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture parameters that affect kernel specialization."""
    d_inner: int = 16
    d_state: int = 16
    num_experts: int = 144
    expert_hidden: int = 8
    gru_hidden: int = 16
    num_heads: int = 4
    pk_dim: int = 8
    param_sizes: List[int] = field(default_factory=list)
    total_params: int = 0


class KernelSpecializer:
    """Base class for hardware-specific kernel JIT compilation.

    Subclasses generate optimized kernel source for a specific backend
    (CUDA, HIP, CPU, TPU) and compile it at runtime. Compiled modules
    are cached on disk keyed by hardware + model config hash.
    """

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.cache_dir = Path.home() / '.cache' / 'supergrok2'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self) -> str:
        """Unique key for this hardware + model configuration."""
        raise NotImplementedError

    def compile(self):
        """Compile specialized kernels. Returns a module with kernel functions.

        Checks disk cache first. If a cached .so exists, loads it directly.
        Otherwise generates source, compiles, and caches the result.
        """
        key = self._cache_key()
        cache_path = self.cache_dir / f'{key}.so'

        if cache_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location(f'sg2_jit_{key}', str(cache_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

        source = self._generate_source()
        module = self._compile_source(source, key)
        return module

    def _generate_source(self) -> str:
        """Generate platform-specific kernel source code."""
        raise NotImplementedError

    def _compile_source(self, source: str, key: str):
        """Compile source code into a loadable module."""
        raise NotImplementedError
