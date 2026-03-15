"""Quantization utilities for SuperGrok v2.

Handles weight conversion, precision selection, and calibration for
multi-precision kernel dispatch.

Supported precision modes:
  Projections: fp32, tf32, bf16, fp8, mxfp4, nvfp4
  Expert weights: fp32, int8, int4
  Scan state: always fp32 (numerical necessity)

Dynamic precision selection (Unsloth Dynamic 2.0-style): monitors training
stability and per-component sensitivity to select the lowest safe precision.
"""

import torch
import math
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from .dispatch import (
    get_gpu_arch, get_gpu_vendor,
    supports_bf16, supports_fp8, supports_tf32, supports_nvfp4,
)


class PrecisionConfig:
    """Configures precision for different components of the optimizer."""

    # Valid precision modes by component
    PROJECTION_MODES = ('fp32', 'tf32', 'bf16', 'fp8', 'mxfp4', 'nvfp4', 'auto')
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
        elif self.projection_precision == 'nvfp4':
            if not supports_nvfp4():
                # Fall back through chain: nvfp4 -> mxfp4 -> fp8 -> bf16 -> fp32
                if supports_fp8():
                    scale = w.abs().max().clamp(min=1e-12)
                    w_scaled = w.float().div(scale)
                    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
                    return w_fp8, scale
                if supports_bf16():
                    return w.bfloat16().contiguous(), None
                return w.float().contiguous(), None
            return self._quantize_nvfp4(w)
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

    def _quantize_nvfp4(self, w, block_size=16):
        """NVFP4 quantization with per-block scaling.

        On sm_100 (Blackwell): cuBLAS accepts NVFP4 inputs natively.
        On sm_90 and below: requires software dequantization (same as MXFP4).

        NVFP4 uses 16-element blocks (vs 32 for MXFP4).
        Same E2M1 codebook: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
        """
        w_flat = w.reshape(-1).float()
        N = w_flat.numel()
        N_padded = ((N + block_size - 1) // block_size) * block_size
        if N_padded > N:
            w_flat = torch.nn.functional.pad(w_flat, (0, N_padded - N))

        num_blocks = N_padded // block_size
        w_blocked = w_flat.reshape(num_blocks, block_size)

        # Per-block scale: max representable is 6.0
        block_max = w_blocked.abs().max(dim=1).values.clamp(min=1e-12)
        scales = block_max / 6.0

        # Quantize to nearest NVFP4 value
        w_scaled = w_blocked / scales.unsqueeze(1)
        fp4_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        signs = (w_scaled < 0).to(torch.uint8)
        magnitudes = w_scaled.abs()

        diffs = (magnitudes.unsqueeze(-1) - fp4_vals.unsqueeze(0).unsqueeze(0)).abs()
        indices = diffs.argmin(-1).to(torch.uint8)
        encoded = (signs << 3) | indices
        encoded = encoded.reshape(-1)

        # Pack pairs
        even = encoded[0::2]
        odd = encoded[1::2]
        packed = even | (odd << 4)

        return packed.contiguous(), scales.contiguous()

    def report_memory(self, num_experts=144, expert_hidden=16,
                      d_model=8, d_inner=16, d_state=16, N=65536):
        """Print memory usage breakdown by precision.

        Args:
            num_experts: number of expert MLPs
            expert_hidden: expert MLP hidden dim
            d_model: meta-net model dim
            d_inner: mamba inner dim
            d_state: mamba state dim
            N: number of optimizer parameters
        """
        # Projection weights memory
        proj_elems = (
            d_model * 2  # input_proj
            + 2 * (2 * d_inner * d_model + d_inner * d_inner + d_inner
                    + d_state * d_inner * 2  # B_proj, C_proj
                    + d_inner * d_state  # A_log
                    + d_inner  # D
                    + d_inner * (d_state // 2)  # rope
                    + d_model * d_inner)  # out_proj
        )
        proj_bits = {'fp32': 32, 'tf32': 32, 'bf16': 16, 'fp8': 8,
                     'mxfp4': 4, 'nvfp4': 4}.get(self.projection_precision, 32)
        proj_bytes = proj_elems * proj_bits // 8

        # Expert weights memory
        expert_elems = num_experts * (expert_hidden * 1 + expert_hidden
                                       + 1 * expert_hidden + 1)
        expert_bits = {'fp32': 32, 'int8': 8, 'int4': 4}.get(self.expert_precision, 32)
        expert_bytes = expert_elems * expert_bits // 8

        # Optimizer state (always FP32): exp_avg, exp_avg_sq, mu, sharpness, gru
        state_bytes = N * 4 * 4  # 4 arrays × 4 bytes

        fp32_proj = proj_elems * 4
        fp32_expert = expert_elems * 4
        total_fp32 = fp32_proj + fp32_expert + state_bytes
        total_now = proj_bytes + expert_bytes + state_bytes
        savings = 100.0 * (1.0 - total_now / total_fp32) if total_fp32 > 0 else 0.0

        print(f"SuperGrok v2 Memory Report "
              f"(projection={self.projection_precision}, "
              f"expert={self.expert_precision})")
        print(f"  Projection weights: {proj_bytes:,} bytes "
              f"({proj_bytes / 1024:.1f} KB)")
        print(f"  Expert weights:     {expert_bytes:,} bytes "
              f"({expert_bytes / 1024:.1f} KB)")
        print(f"  Optimizer states:   {state_bytes:,} bytes "
              f"({state_bytes / 1024 / 1024:.1f} MB)")
        print(f"  Meta-net total:     {proj_bytes + expert_bytes:,} bytes "
              f"({(proj_bytes + expert_bytes) / 1024:.1f} KB)")
        print(f"  Savings vs FP32:    {savings:.1f}%")

    def __repr__(self):
        parts = [
            f"projection={self.projection_precision}",
            f"expert={self.expert_precision}",
            f"scan={self.scan_precision}",
        ]
        if self.dynamic:
            parts.append(f"dynamic=True (tier={self._precision_tier})")
        return f"PrecisionConfig({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════
#  Unsloth Dynamic 2.0-Style Adaptive Precision
#
#  Per-component adaptive selection monitoring each component's numerical
#  sensitivity: scan output variance for projections, routing entropy
#  for expert weights.
# ═══════════════════════════════════════════════════════════════════════

class UnslothDynamicPrecision:
    """Per-component adaptive precision selection.

    Unlike static precision (same format for all components) or simple
    dynamic (time-based schedule), this monitors each component's numerical
    sensitivity and selects the lowest safe precision independently.

    Components:
    - Projections: sensitivity = scan output variance
    - Expert weights: sensitivity = routing entropy
    - Scan state: ALWAYS FP32 (non-negotiable)
    """

    def __init__(self, warmup_steps=200, sensitivity_window=50):
        self.warmup_steps = warmup_steps
        self.sensitivity_window = sensitivity_window
        self._scan_output_variance_history = []
        self._routing_entropy_history = []
        self._current_proj_precision = 'fp32'
        self._current_expert_precision = 'fp32'
        self._step = 0

    def update(self, step, scan_output_var, routing_entropy):
        """Update precision based on current training dynamics.

        Args:
            step: current optimizer step
            scan_output_var: variance of scan output (projection sensitivity)
            routing_entropy: entropy of expert routing distribution
        """
        self._step = step
        self._scan_output_variance_history.append(float(scan_output_var))
        self._routing_entropy_history.append(float(routing_entropy))

        if step < self.warmup_steps:
            self._current_proj_precision = 'fp32'
            self._current_expert_precision = 'fp32'
            return

        # Compute stability metrics over recent window
        recent_var = self._scan_output_variance_history[-self.sensitivity_window:]
        if len(recent_var) < 2:
            return

        var_min = min(recent_var)
        var_max = max(recent_var)
        var_stability = var_max / (var_min + 1e-12)

        recent_entropy = self._routing_entropy_history[-self.sensitivity_window:]
        avg_entropy = sum(recent_entropy) / len(recent_entropy)

        # Projection precision: lower when scan output is stable
        if var_stability < 1.5:
            self._current_proj_precision = self._best_available_low_precision()
        elif var_stability < 3.0:
            self._current_proj_precision = 'bf16' if supports_bf16() else 'tf32'
        else:
            self._current_proj_precision = 'fp32'

        # Expert precision: lower when routing entropy is high
        if avg_entropy > 3.0:
            self._current_expert_precision = 'int4'
        elif avg_entropy > 2.0:
            self._current_expert_precision = 'int8'
        else:
            self._current_expert_precision = 'fp32'

    def _best_available_low_precision(self):
        if supports_fp8():
            return 'fp8'
        if supports_bf16():
            return 'bf16'
        if supports_tf32():
            return 'tf32'
        return 'fp32'

    @property
    def projection_precision(self):
        return self._current_proj_precision

    @property
    def expert_precision(self):
        return self._current_expert_precision

    def __repr__(self):
        return (f"UnslothDynamicPrecision(step={self._step}, "
                f"proj={self._current_proj_precision}, "
                f"expert={self._current_expert_precision})")


# ═══════════════════════════════════════════════════════════════════════
#  Unified Quantization Registry
#
#  Enumerates all supported quantization formats with auto-fallback
#  chains based on hardware capability.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QuantFormat:
    """Describes a single quantization format with hardware requirements."""
    name: str
    bits: int
    component: str      # 'projection' or 'expert'
    min_sm: int         # minimum SM architecture (0 = all)
    native_sm: int      # SM with native hardware support (0 = none)
    fallback: Optional[str]  # next-best format key if unavailable

    def __repr__(self):
        return f"QuantFormat({self.name}, {self.bits}b, min_sm={self.min_sm})"


QUANT_REGISTRY: Dict[str, QuantFormat] = {
    # Projection formats (ordered by aggressiveness)
    'projection_fp32':  QuantFormat('FP32',  32, 'projection', 0,   0,   None),
    'projection_tf32':  QuantFormat('TF32',  32, 'projection', 80,  80,  'projection_fp32'),
    'projection_bf16':  QuantFormat('BF16',  16, 'projection', 80,  80,  'projection_tf32'),
    'projection_fp8':   QuantFormat('FP8',   8,  'projection', 89,  90,  'projection_bf16'),
    'projection_mxfp4': QuantFormat('MXFP4', 4,  'projection', 0,   100, 'projection_fp8'),
    'projection_nvfp4': QuantFormat('NVFP4', 4,  'projection', 100, 100, 'projection_mxfp4'),

    # Expert weight formats
    'expert_fp32': QuantFormat('FP32', 32, 'expert', 0, 0, None),
    'expert_int8': QuantFormat('INT8', 8,  'expert', 0, 0, 'expert_fp32'),
    'expert_int4': QuantFormat('INT4', 4,  'expert', 0, 0, 'expert_int8'),
}


def resolve_format(requested: str, component: str) -> QuantFormat:
    """Resolve a requested format to the best available on current hardware.

    Walks the fallback chain until finding a format supported by current GPU.

    Args:
        requested: format name (e.g. 'fp8', 'int8', 'nvfp4')
        component: 'projection' or 'expert'

    Returns:
        QuantFormat that is supported on current hardware

    Raises:
        ValueError: if requested format is unknown
        RuntimeError: if no supported format found (shouldn't happen — FP32 is always available)
    """
    key = f"{component}_{requested}"
    if key not in QUANT_REGISTRY:
        raise ValueError(
            f"Unknown quantization format: '{requested}' for component '{component}'. "
            f"Available: {[k.split('_', 1)[1] for k in QUANT_REGISTRY if k.startswith(component)]}"
        )

    arch = get_gpu_arch()
    fmt = QUANT_REGISTRY[key]

    while fmt is not None and fmt.min_sm > arch:
        if fmt.fallback is None:
            raise RuntimeError(
                f"No supported format for {component} on sm_{arch}. "
                f"This should not happen — FP32 has min_sm=0."
            )
        fmt = QUANT_REGISTRY[fmt.fallback]

    return fmt
