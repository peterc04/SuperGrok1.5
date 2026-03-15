"""Quantization utilities for SuperGrok v2.

Handles weight conversion, precision selection, and calibration for
multi-precision kernel dispatch.

Quantization applies to PROJECTION GEMMs only. Scan state accumulation
stays FP32 (numerical necessity for 65K-step recurrences). Expert
weights stay FP32 (already in smem, bandwidth not the bottleneck).
"""

import torch
from .dispatch import get_gpu_arch, supports_bf16, supports_fp8


class PrecisionConfig:
    """Configures precision for different components of the optimizer."""

    def __init__(self, projection_precision='auto', scan_precision='fp32'):
        """
        Args:
            projection_precision: 'fp32', 'tf32', 'bf16', 'fp8', or 'auto'
                'auto' selects the best available for the current GPU.
            scan_precision: Always 'fp32' (scan state accumulation).
                Exposed for documentation, not changeable.
        """
        self.scan_precision = 'fp32'  # non-negotiable

        if projection_precision == 'auto':
            arch = get_gpu_arch()
            if arch >= 90:
                self.projection_precision = 'fp8'
            elif arch >= 80:
                self.projection_precision = 'bf16'
            else:
                self.projection_precision = 'fp32'
        else:
            self.projection_precision = projection_precision

    def convert_projection_weights(self, w):
        """Convert a projection weight matrix to the target precision.

        Returns:
            For fp32/tf32: (tensor, None) — TF32 is transparent via cuBLAS
            For bf16: (tensor_bf16, None)
            For fp8: (tensor_fp8, scale) — scale needed for cuBLAS alpha
        """
        if self.projection_precision in ('fp32', 'tf32'):
            return w.float().contiguous(), None
        elif self.projection_precision == 'bf16':
            if not supports_bf16():
                return w.float().contiguous(), None
            return w.bfloat16().contiguous(), None
        elif self.projection_precision == 'fp8':
            if not supports_fp8():
                # Fallback to bf16 if FP8 not available
                if supports_bf16():
                    return w.bfloat16().contiguous(), None
                return w.float().contiguous(), None
            # FP8 requires per-tensor absmax scaling
            scale = w.abs().max().clamp(min=1e-12)
            w_scaled = w.float().div(scale)
            w_fp8 = w_scaled.to(torch.float8_e4m3fn)
            return w_fp8, scale
        else:
            raise ValueError(f"Unknown precision: {self.projection_precision}")

    def __repr__(self):
        return (f"PrecisionConfig(projection={self.projection_precision}, "
                f"scan={self.scan_precision})")
