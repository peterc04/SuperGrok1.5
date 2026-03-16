"""JIT Specialization System — Hardware-specific kernel compilation.

Generates and compiles optimized kernels tailored to the exact GPU/CPU
hardware and model configuration at runtime. Caches compiled modules
in ~/.cache/supergrok2/ for instant reuse.

Supported backends:
  - CUDA (NVIDIA): sm_75 (Turing) through sm_100 (Blackwell)
  - HIP (AMD): CDNA2 through CDNA4
  - TPU: v4, v5e, v5p, v6e (via Pallas)
  - CPU: x86_64 (AVX2/AVX-512), aarch64 (NEON/SVE)
"""

from .specializer import KernelSpecializer, ModelConfig
from .cuda_specializer import CUDASpecializer, GPUConfig
from .hip_specializer import HIPSpecializer
from .tpu_specializer import TPUSpecializer
from .cpu_specializer import CPUSpecializer


def create_specializer(device, model_config):
    """Create the appropriate JIT specializer for the given device.

    Args:
        device: torch.device — the target device.
        model_config: ModelConfig — model architecture parameters.

    Returns:
        A KernelSpecializer subclass for the detected hardware.

    Raises:
        RuntimeError: If the device type is not supported (no fallback).
    """
    if device.type == 'cuda':
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return HIPSpecializer(model_config)
        return CUDASpecializer(model_config)
    elif device.type == 'cpu':
        return CPUSpecializer(model_config)
    else:
        raise RuntimeError(f"Unsupported device: {device}. No fallback.")


__all__ = [
    'create_specializer',
    'KernelSpecializer', 'ModelConfig',
    'CUDASpecializer', 'GPUConfig',
    'HIPSpecializer',
    'TPUSpecializer',
    'CPUSpecializer',
]
