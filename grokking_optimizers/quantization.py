"""Quantization utilities for SuperGrok v2.

Handles weight conversion, precision selection, and calibration for
multi-precision kernel dispatch.

Supported precision modes:
  Projections: fp32, tf32, bf16, fp8, mxfp4
  Expert weights: fp32, int8, int4
  Scan state: always fp32 (numerical necessity)

Dynamic precision selection (Unsloth-style): monitors training stability
and progressively lowers precision as training stabilizes.
"""

import torch
import math
from .dispatch import (
    get_gpu_arch, get_gpu_vendor,
    supports_bf16, supports_fp8, supports_tf32,
)


class PrecisionConfig:
    """Configures precision for different components of the optimizer."""

    # Valid precision modes by component
    PROJECTION_MODES = ('fp32', 'tf32', 'bf16', 'fp8', 'mxfp4', 'auto')
    EXPERT_MODES = ('fp32', 'int8', 'int4', 'auto')

    def __init__(
        self,
        projection_precision='auto',
        expert_precision='fp32',
        scan_precision='fp32',
        dynamic=False,
    ):
        """
        Args:
            projection_precision: 'fp32', 'tf32', 'bf16', 'fp8', 'mxfp4', or 'auto'.
                'auto' selects the best available for the current GPU.
            expert_precision: 'fp32', 'int8', 'int4', or 'auto'.
                Controls expert MLP weight quantization.
            scan_precision: Always 'fp32' (scan state accumulation).
                Exposed for documentation, not changeable.
            dynamic: If True, enables dynamic precision selection that
                progressively lowers precision as training stabilizes.
        """
        self.scan_precision = 'fp32'  # non-negotiable
        self.dynamic = dynamic

        # Resolve projection precision
        if projection_precision == 'auto':
            arch = get_gpu_arch()
            vendor = get_gpu_vendor()
            if vendor == 'amd':
                self.projection_precision = 'bf16' if supports_bf16() else 'fp32'
            elif arch >= 90:
                self.projection_precision = 'fp8'
            elif arch >= 80:
                self.projection_precision = 'bf16'
            else:
                self.projection_precision = 'fp32'
        else:
            if projection_precision not in self.PROJECTION_MODES:
                raise ValueError(
                    f"Unknown projection precision: {projection_precision}. "
                    f"Must be one of {self.PROJECTION_MODES}"
                )
            self.projection_precision = projection_precision

        # Resolve expert precision
        if expert_precision == 'auto':
            self.expert_precision = 'int8'
        else:
            if expert_precision not in self.EXPERT_MODES:
                raise ValueError(
                    f"Unknown expert precision: {expert_precision}. "
                    f"Must be one of {self.EXPERT_MODES}"
                )
            self.expert_precision = expert_precision

        # Dynamic precision state
        self._step_count = 0
        self._grad_norm_ema = None
        self._grad_norm_var_ema = None
        self._precision_tier = 0  # 0=highest precision, increases as we lower

    def convert_projection_weights(self, w):
        """Convert a projection weight matrix to the target precision.

        Returns:
            For fp32/tf32: (tensor, None) — TF32 is transparent via cuBLAS
            For bf16: (tensor_bf16, None)
            For fp8: (tensor_fp8, scale)
            For mxfp4: (packed_tensor, shared_exponents)
        """
        if self.projection_precision in ('fp32', 'tf32'):
            return w.float().contiguous(), None
        elif self.projection_precision == 'bf16':
            if not supports_bf16():
                return w.float().contiguous(), None
            return w.bfloat16().contiguous(), None
        elif self.projection_precision == 'fp8':
            if not supports_fp8():
                if supports_bf16():
                    return w.bfloat16().contiguous(), None
                return w.float().contiguous(), None
            scale = w.abs().max().clamp(min=1e-12)
            w_scaled = w.float().div(scale)
            w_fp8 = w_scaled.to(torch.float8_e4m3fn)
            return w_fp8, scale
        elif self.projection_precision == 'mxfp4':
            return self._quantize_mxfp4(w)
        else:
            raise ValueError(f"Unknown precision: {self.projection_precision}")

    def convert_expert_weights(self, w1, b1, w2, b2):
        """Convert expert MLP weights to target precision.

        Args:
            w1: [num_experts, expert_hidden] — first layer weights
            b1: [num_experts, expert_hidden] — first layer biases
            w2: [num_experts, expert_hidden] — second layer weights
            b2: [num_experts] — second layer biases

        Returns:
            dict with keys matching the precision mode:
            - 'fp32': original tensors as float
            - 'int8': {'w1_q', 'w1_s', 'b1', 'w2_q', 'w2_s', 'b2'}
            - 'int4': {'w1_packed', 'w1_scales', 'w1_zeros', 'b1', ...}
        """
        if self.expert_precision == 'fp32':
            return {
                'w1': w1.float().contiguous(),
                'b1': b1.float().contiguous(),
                'w2': w2.float().contiguous(),
                'b2': b2.float().contiguous(),
                'mode': 'fp32',
            }
        elif self.expert_precision == 'int8':
            return self._quantize_expert_int8(w1, b1, w2, b2)
        elif self.expert_precision == 'int4':
            return self._quantize_expert_int4(w1, b1, w2, b2)
        else:
            raise ValueError(f"Unknown expert precision: {self.expert_precision}")

    def _quantize_expert_int8(self, w1, b1, w2, b2):
        """Symmetric INT8 quantization for expert weights."""
        def sym_quant(w):
            absmax = w.abs().max().clamp(min=1e-12)
            scale = absmax / 127.0
            q = (w / scale).round().clamp(-127, 127).to(torch.int8)
            return q, scale

        w1_q, w1_s = sym_quant(w1.float())
        w2_q, w2_s = sym_quant(w2.float())
        return {
            'w1_q': w1_q.contiguous(), 'w1_s': w1_s,
            'b1': b1.float().contiguous(),
            'w2_q': w2_q.contiguous(), 'w2_s': w2_s,
            'b2': b2.float().contiguous(),
            'mode': 'int8',
        }

    def _quantize_expert_int4(self, w1, b1, w2, b2):
        """GPTQ-style INT4 quantization for expert weights."""
        def int4_quant(w, group_size=32):
            w_flat = w.reshape(-1).float()
            N = w_flat.numel()
            N_padded = ((N + 1) // 2) * 2
            if N_padded > N:
                w_flat = torch.nn.functional.pad(w_flat, (0, N_padded - N))

            num_groups = (N_padded + group_size - 1) // group_size
            actual_gs = N_padded // num_groups
            w_grouped = w_flat.reshape(num_groups, actual_gs)

            gmax = w_grouped.max(dim=1).values
            gmin = w_grouped.min(dim=1).values
            scales = ((gmax - gmin) / 15.0).clamp(min=1e-12)
            zeros = gmin

            q = ((w_grouped - zeros.unsqueeze(1)) / scales.unsqueeze(1))
            q = q.round().clamp(0, 15).to(torch.uint8).reshape(-1)

            # Pack pairs
            even = q[0::2]
            odd = q[1::2]
            packed = even | (odd << 4)
            return packed, scales, zeros

        w1_p, w1_s, w1_z = int4_quant(w1)
        w2_p, w2_s, w2_z = int4_quant(w2)
        return {
            'w1_packed': w1_p.contiguous(), 'w1_scales': w1_s, 'w1_zeros': w1_z,
            'b1': b1.float().contiguous(),
            'w2_packed': w2_p.contiguous(), 'w2_scales': w2_s, 'w2_zeros': w2_z,
            'b2': b2.float().contiguous(),
            'mode': 'int4',
        }

    def _quantize_mxfp4(self, w, block_size=32):
        """Microscaling FP4 (E2M1) quantization for projection weights."""
        w_flat = w.reshape(-1).float()
        N = w_flat.numel()
        N_padded = ((N + block_size - 1) // block_size) * block_size
        if N_padded > N:
            w_flat = torch.nn.functional.pad(w_flat, (0, N_padded - N))

        num_blocks = N_padded // block_size
        w_blocked = w_flat.reshape(num_blocks, block_size)

        # Shared exponent per block
        block_max = w_blocked.abs().max(dim=1).values.clamp(min=1e-12)
        shared_exp_f = torch.floor(torch.log2(block_max)) + 127.0
        shared_exp_f = shared_exp_f.clamp(0.0, 255.0)
        shared_exp = shared_exp_f.to(torch.uint8)

        # Scale each block by shared exponent
        block_scale = torch.exp2(shared_exp_f - 127.0).unsqueeze(1)
        w_scaled = w_blocked / block_scale

        # FP4 E2M1 magnitude values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        fp4_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        signs = (w_scaled < 0).to(torch.uint8)
        magnitudes = w_scaled.abs()

        # Nearest-value quantization
        diffs = (magnitudes.unsqueeze(-1) - fp4_vals.unsqueeze(0).unsqueeze(0)).abs()
        indices = diffs.argmin(-1).to(torch.uint8)
        encoded = (signs << 3) | indices
        encoded = encoded.reshape(-1)

        # Pack pairs into uint8
        even = encoded[0::2]
        odd = encoded[1::2]
        packed = even | (odd << 4)

        return packed.contiguous(), shared_exp.contiguous()

    # ═══════════════════════════════════════════════════════════════════
    #  Dynamic Precision Selection
    # ═══════════════════════════════════════════════════════════════════

    def update_dynamic(self, grad_norm):
        """Update dynamic precision state based on gradient norm stability.

        Call this once per optimizer step with the current gradient norm.
        When training stabilizes (low variance in grad norms), precision
        is progressively lowered.

        Args:
            grad_norm: float, the current global gradient norm.

        Returns:
            True if precision was changed, False otherwise.
        """
        if not self.dynamic:
            return False

        self._step_count += 1
        alpha = 0.01  # EMA smoothing factor

        if self._grad_norm_ema is None:
            self._grad_norm_ema = grad_norm
            self._grad_norm_var_ema = 0.0
            return False

        # Update EMA of gradient norm and its variance
        self._grad_norm_ema = (1 - alpha) * self._grad_norm_ema + alpha * grad_norm
        deviation = (grad_norm - self._grad_norm_ema) ** 2
        self._grad_norm_var_ema = (1 - alpha) * self._grad_norm_var_ema + alpha * deviation

        # Coefficient of variation (stability metric)
        cv = math.sqrt(self._grad_norm_var_ema) / max(self._grad_norm_ema, 1e-12)

        # Only consider lowering after warmup
        if self._step_count < 500:
            return False

        # Stability thresholds for progressive precision lowering
        # Lower CV = more stable = can use lower precision
        changed = False

        if cv < 0.05 and self._precision_tier < 3:
            # Very stable — lower to most aggressive
            self._precision_tier = 3
            changed = True
        elif cv < 0.10 and self._precision_tier < 2:
            # Stable — lower to medium
            self._precision_tier = 2
            changed = True
        elif cv < 0.20 and self._precision_tier < 1:
            # Somewhat stable — lower one tier
            self._precision_tier = 1
            changed = True
        elif cv > 0.30 and self._precision_tier > 0:
            # Unstable — raise precision back
            self._precision_tier = 0
            changed = True

        if changed:
            self._apply_dynamic_tier()

        return changed

    def _apply_dynamic_tier(self):
        """Apply the current dynamic precision tier."""
        vendor = get_gpu_vendor()

        # Projection precision tiers (most → least precise)
        if vendor == 'nvidia':
            proj_tiers = ['fp32', 'tf32', 'bf16', 'fp8']
        else:
            proj_tiers = ['fp32', 'fp32', 'bf16', 'bf16']

        # Expert precision tiers
        expert_tiers = ['fp32', 'fp32', 'int8', 'int4']

        tier = min(self._precision_tier, len(proj_tiers) - 1)

        # Only lower if hardware supports it
        new_proj = proj_tiers[tier]
        if new_proj == 'fp8' and not supports_fp8():
            new_proj = 'bf16'
        if new_proj == 'bf16' and not supports_bf16():
            new_proj = 'fp32'
        if new_proj == 'tf32' and not supports_tf32():
            new_proj = 'fp32'
        self.projection_precision = new_proj

        self.expert_precision = expert_tiers[min(tier, len(expert_tiers) - 1)]

    @property
    def stability_cv(self):
        """Current coefficient of variation of gradient norms."""
        if self._grad_norm_ema is None or self._grad_norm_var_ema is None:
            return float('inf')
        return math.sqrt(self._grad_norm_var_ema) / max(self._grad_norm_ema, 1e-12)

    def __repr__(self):
        parts = [
            f"projection={self.projection_precision}",
            f"expert={self.expert_precision}",
            f"scan={self.scan_precision}",
        ]
        if self.dynamic:
            parts.append(f"dynamic=True (tier={self._precision_tier})")
        return f"PrecisionConfig({', '.join(parts)})"
