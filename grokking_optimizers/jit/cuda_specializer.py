"""CUDA JIT Specializer — generates sm-version-specific kernels.

Detects GPU properties (SM version, SM count, register file size,
shared memory capacity, L2 cache size) and generates kernels with:
  - Compile-time constants for d_inner, d_state, expert_hidden
  - Hardware-tuned block sizes and launch bounds
  - SM-version-specific PTX for hot inner loops
  - Optimal shared memory padding to avoid bank conflicts
"""

import hashlib
import torch
from dataclasses import dataclass
from typing import Dict

from .specializer import KernelSpecializer, ModelConfig
from .ptx_scheduler import PTXScheduler
from .block_size_optimizer import compute_optimal_block_sizes
from .smem_layout import compute_smem_padding


@dataclass
class GPUConfig:
    """Detected GPU hardware properties."""
    sm_version: int = 80
    sm_count: int = 108
    max_threads_per_sm: int = 2048
    max_regs_per_sm: int = 65536
    max_smem_per_sm: int = 163840
    l2_cache_size: int = 40 * 1024 * 1024
    memory_bandwidth: int = 0
    warp_size: int = 32


class CUDASpecializer(KernelSpecializer):
    """Generate CUDA kernels specialized for the detected NVIDIA GPU.

    Produces kernels with compile-time constants derived from both the
    GPU hardware (SM version, SM count, cache sizes) and the model
    architecture (d_inner, d_state, expert_hidden).
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.gpu = self._detect_gpu()
        self.ptx = PTXScheduler(self.gpu.sm_version)

    def _detect_gpu(self) -> GPUConfig:
        """Probe GPU hardware properties via CUDA runtime."""
        if not torch.cuda.is_available():
            return GPUConfig()  # Defaults for offline code generation

        props = torch.cuda.get_device_properties(0)
        sm_version = props.major * 10 + props.minor

        # Query max registers per SM via cudaDeviceGetAttribute
        try:
            max_regs = torch.cuda.get_device_properties(0).max_threads_per_multi_processor
        except Exception:
            max_regs = 65536

        return GPUConfig(
            sm_version=sm_version,
            sm_count=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            max_regs_per_sm=max_regs,
            max_smem_per_sm=props.max_shared_memory_per_block_optin
            if hasattr(props, 'max_shared_memory_per_block_optin')
            else props.max_shared_memory_per_block,
            l2_cache_size=props.l2_cache_size
            if hasattr(props, 'l2_cache_size') else 40 * 1024 * 1024,
            memory_bandwidth=props.total_memory,
            warp_size=32,
        )

    def _cache_key(self) -> str:
        """Hash of GPU + model config for cache lookup."""
        gpu = self.gpu
        cfg = self.config
        raw = (
            f"cuda_sm{gpu.sm_version}_sms{gpu.sm_count}_"
            f"d{cfg.d_inner}_s{cfg.d_state}_e{cfg.num_experts}_"
            f"eh{cfg.expert_hidden}_gh{cfg.gru_hidden}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def _compute_launch_bounds(self) -> Dict[str, int]:
        """Compute optimal launch bounds for each kernel type.

        Returns min_blocks for __launch_bounds__(max_threads, min_blocks).
        """
        sm = self.gpu.sm_version
        # Hopper+ has 4 sub-partitions; target 2 blocks/SM for fused_elem
        # and 8 blocks/SM for element-wise kernels
        if sm >= 90:
            return {'scan': 8, 'elem': 2, 'persistent': 1, 'distributed': 2}
        elif sm >= 80:
            return {'scan': 8, 'elem': 2, 'persistent': 1, 'distributed': 2}
        else:
            return {'scan': 4, 'elem': 2, 'persistent': 1, 'distributed': 2}

    def _generate_source(self) -> str:
        """Generate CUDA source from Jinja2 templates.

        Templates are filled with compile-time constants derived from
        the GPU hardware and model configuration.
        """
        try:
            import jinja2
        except ImportError:
            raise RuntimeError(
                "JIT compilation requires jinja2. Install with: pip install jinja2"
            )

        import os
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'cuda')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            undefined=jinja2.StrictUndefined,
        )

        block_sizes = compute_optimal_block_sizes(
            param_sizes=self.config.param_sizes,
            sm_count=self.gpu.sm_count,
            max_threads=self.gpu.max_threads_per_sm,
        )

        launch_bounds = self._compute_launch_bounds()
        smem_padding = compute_smem_padding(
            expert_hidden=self.config.expert_hidden,
            num_experts=self.config.num_experts,
            bank_count=32, bank_width=4,
        )

        # Hardware-specific PTX for inner loops
        ptx_affine = self.ptx.generate_affine_combine()
        ptx_stochastic = self.ptx.generate_stochastic_round()
        ptx_expert_mlp = self.ptx.generate_expert_mlp(self.config.expert_hidden)

        template_vars = dict(
            D_INNER=self.config.d_inner,
            D_STATE=self.config.d_state,
            NUM_EXPERTS=self.config.num_experts,
            EXPERT_HIDDEN=self.config.expert_hidden,
            GRU_HIDDEN=self.config.gru_hidden,
            SCAN_BLOCK_SIZE=block_sizes['scan'],
            ELEM_BLOCK_SIZE=block_sizes['elem'],
            LAUNCH_MIN_BLOCKS_SCAN=launch_bounds['scan'],
            LAUNCH_MIN_BLOCKS_ELEM=launch_bounds['elem'],
            SMEM_PADDING=smem_padding,
            PTX_AFFINE_COMBINE=ptx_affine,
            PTX_STOCHASTIC_ROUND=ptx_stochastic,
            PTX_EXPERT_MLP=ptx_expert_mlp,
            SM_VERSION=self.gpu.sm_version,
            L2_CACHE_SIZE=self.gpu.l2_cache_size,
        )

        sources = []
        for template_name in ['scan_kernel.cu.j2', 'fused_elem_kernel.cu.j2',
                              'persistent_kernel.cu.j2', 'distributed_kernel.cu.j2']:
            try:
                template = env.get_template(template_name)
                sources.append(template.render(**template_vars))
            except jinja2.TemplateNotFound:
                # Template not yet created — skip gracefully during development
                pass

        return '\n'.join(sources)

    def _compile_source(self, source: str, key: str):
        """Compile CUDA source via torch.utils.cpp_extension.load_inline."""
        if not source.strip():
            return None

        return torch.utils.cpp_extension.load_inline(
            name=f'sg2_jit_{key}',
            cuda_sources=[source],
            build_directory=str(self.cache_dir),
            extra_cuda_cflags=[
                f'-arch=sm_{self.gpu.sm_version}',
                '-O3', '--use_fast_math',
                '-lineinfo',
            ],
        )
