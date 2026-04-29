"""HIP JIT Specializer — generates CDNA-version-specific kernels for AMD GPUs.

Detects CDNA version (CDNA2/MI250, CDNA3/MI300, CDNA4/MI350) and generates
kernels with architecture-specific optimizations:
  - FP4/FP6 MFMA on CDNA4
  - BF16 MFMA on CDNA2+
  - Wavefront-64 aware scan operations
  - LDS capacity-tuned tiling
"""

import hashlib
import os
import torch
from typing import Dict

from .specializer import KernelSpecializer, ModelConfig
from .gcn_scheduler import GCNScheduler


class HIPSpecializer(KernelSpecializer):
    """Generate HIP kernels specialized for the detected AMD GPU."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.cdna_version = self._detect_cdna()
        self.gcn = GCNScheduler(self.cdna_version)

    def _detect_cdna(self) -> str:
        """Detect CDNA generation from GPU architecture."""
        if not torch.cuda.is_available():
            return 'cdna2'  # Default for offline code generation

        # On ROCm, torch.cuda.get_device_capability returns (major, minor)
        # gfx90a -> (9, 0) -> cdna2
        # gfx942 -> (9, 4) -> cdna3
        # gfx950 -> (9, 5) -> cdna4
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor

        if arch >= 95:
            return 'cdna4'
        elif arch >= 94:
            return 'cdna3'
        elif arch >= 90:
            return 'cdna2'
        return 'generic'

    def _cache_key(self) -> str:
        cfg = self.config
        raw = (
            f"hip_{self.cdna_version}_"
            f"d{cfg.d_inner}_s{cfg.d_state}_e{cfg.num_experts}_"
            f"eh{cfg.expert_hidden}_gh{cfg.gru_hidden}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def _generate_source(self) -> str:
        """Generate HIP source from Jinja2 templates."""
        try:
            import jinja2
        except ImportError:
            raise RuntimeError(
                "JIT compilation requires jinja2. Install with: pip install jinja2"
            )

        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'hip')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            undefined=jinja2.StrictUndefined,
        )

        expert_load = self.gcn.generate_expert_load()
        state_pack = self.gcn.generate_state_pack_unpack()
        lds_load = self.gcn.generate_lds_load()
        scan_reduction = self.gcn.generate_scan_reduction()

        template_vars = dict(
            D_INNER=self.config.d_inner,
            D_STATE=self.config.d_state,
            NUM_EXPERTS=self.config.num_experts,
            EXPERT_HIDDEN=self.config.expert_hidden,
            GRU_HIDDEN=self.config.gru_hidden,
            WAVEFRONT_SIZE=self.gcn.config['wavefront'],
            LDS_SIZE=self.gcn.config['lds_size'],
            HAS_FP4=self.gcn.config['has_fp4'],
            HAS_FP6=self.gcn.config['has_fp6'],
            HAS_BF16_MFMA=self.gcn.config['has_bf16_mfma'],
            GCN_EXPERT_LOAD=expert_load,
            GCN_STATE_PACK=state_pack,
            GCN_LDS_LOAD=lds_load,
            GCN_SCAN_REDUCTION=scan_reduction,
            CDNA_VERSION=self.cdna_version,
        )

        sources = []
        for template_name in ['scan_kernel.hip.j2', 'fused_elem_kernel.hip.j2']:
            try:
                template = env.get_template(template_name)
                sources.append(template.render(**template_vars))
            except Exception:
                pass

        return '\n'.join(sources)

    def _compile_source(self, source: str, key: str):
        """Compile HIP source via torch.utils.cpp_extension.load_inline."""
        if not source.strip():
            return None

        return torch.utils.cpp_extension.load_inline(
            name=f'sg2_jit_hip_{key}',
            cuda_sources=[source],
            build_directory=str(self.cache_dir),
            extra_cuda_cflags=[
                '-O3', '--offload-arch=gfx942',
            ],
        )
