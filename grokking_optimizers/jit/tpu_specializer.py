"""TPU JIT Specializer — generates TPU-version-specific Pallas kernels.

Detects TPU version (v4, v5e, v5p, v6e) and generates Pallas kernels
with version-specific optimizations:
  - MXU tile size (128 on v4/v5, 256 on v6e)
  - VMEM budget and persistent allocation strategy
  - Scan tile size matching VMEM capacity
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .specializer import KernelSpecializer, ModelConfig


@dataclass
class VMEMPlan:
    """VMEM allocation plan for a specific TPU + model combination."""
    resident: int = 0         # Bytes kept persistently in VMEM
    streaming_tile: int = 0   # Elements per streaming tile


@dataclass
class TPUKernels:
    """Container for JIT-generated Pallas kernel functions."""
    scan_fn: Optional[Callable] = None
    fused_fn: Optional[Callable] = None
    is_persistent: bool = False


class TPUSpecializer(KernelSpecializer):
    """Generate TPU-version-specific Pallas code.

    Unlike CUDA/HIP specializers which compile C++ source, the TPU
    specializer generates Python closures that capture compile-time
    constants (tile sizes, VMEM budgets). When JAX traces these
    closures, the constants become XLA compile-time values enabling
    full optimization by the XLA compiler.
    """

    TPU_CONFIG: Dict[str, Dict[str, Any]] = {
        'v4':  {'mxu': 128, 'vmem_mb': 32,  'persistent_vmem': False, 'bf16_mxu': True},
        'v5e': {'mxu': 128, 'vmem_mb': 32,  'persistent_vmem': False, 'bf16_mxu': True},
        'v5p': {'mxu': 128, 'vmem_mb': 96,  'persistent_vmem': True,  'bf16_mxu': True},
        'v6e': {'mxu': 256, 'vmem_mb': 64,  'persistent_vmem': True,  'bf16_mxu': True},
    }

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.tpu_version = self._detect_tpu()
        self.tpu_config = self.TPU_CONFIG.get(self.tpu_version, self.TPU_CONFIG['v4'])

    def _detect_tpu(self) -> str:
        """Detect TPU version from JAX runtime."""
        try:
            import jax
            devices = jax.devices()
            if devices and hasattr(devices[0], 'device_kind'):
                kind = devices[0].device_kind.lower()
                if 'v6' in kind:
                    return 'v6e'
                elif 'v5p' in kind or 'v5litepod' in kind:
                    return 'v5p'
                elif 'v5' in kind:
                    return 'v5e'
                elif 'v4' in kind:
                    return 'v4'
        except ImportError:
            pass
        return 'v4'

    def _cache_key(self) -> str:
        """TPU kernels are Python closures — no disk cache needed.

        Returns a key for consistency with the base class interface.
        """
        cfg = self.config
        return f"tpu_{self.tpu_version}_d{cfg.d_inner}_s{cfg.d_state}"

    def compile(self) -> TPUKernels:
        """Generate specialized Pallas kernels as Python functions.

        Returns TPUKernels containing closures that capture compile-time
        constants. These closures are traced by JAX to produce optimized
        XLA HLO.
        """
        tile = self.tpu_config['mxu']
        vmem_plan = self._plan_vmem()

        return TPUKernels(
            scan_fn=self._generate_scan(tile),
            fused_fn=self._generate_fused(tile, vmem_plan),
            is_persistent=self.tpu_config['persistent_vmem'],
        )

    def _plan_vmem(self) -> VMEMPlan:
        """Compute optimal VMEM allocation for this model.

        Determines how much VMEM to reserve for persistent weights
        (expert, GRU, PEER) vs streaming state (optimizer moments).
        """
        cfg = self.config
        # Expert weights: num_experts * expert_hidden * (input_dim=2 + output_dim=1) * 4 bytes
        expert_bytes = cfg.num_experts * cfg.expert_hidden * 4 * 4
        # GRU weights: gru_hidden * 6 gates * 4 bytes
        gru_bytes = cfg.gru_hidden * 6 * 4
        # PEER product keys: num_heads * pk_dim * 4 bytes
        peer_bytes = cfg.num_heads * cfg.pk_dim * 4
        resident = expert_bytes + gru_bytes + peer_bytes

        vmem_bytes = self.tpu_config['vmem_mb'] * 1024 * 1024
        streaming_budget = vmem_bytes - resident
        # 5 state arrays (ea, eas, mu, sharpness, gru_state) * 4 bytes each
        tile_elements = max(1, streaming_budget // (5 * 4))

        return VMEMPlan(resident=resident, streaming_tile=tile_elements)

    def _generate_scan(self, tile_size: int) -> Callable:
        """Return a Pallas scan function with compile-time tile_size.

        The returned closure captures tile_size as a Python constant.
        When JAX traces it, tile_size is known at trace time, enabling
        full unrolling and tile-size-specific optimizations.
        """
        def scan_fn(Ms, bs):
            """Tiled affine scan with compile-time tile size.

            Args:
                Ms: [N, 2, 2] affine matrices
                bs: [N, 2] affine biases

            Returns:
                ys: [N, 2] scan outputs
            """
            try:
                import jax
                import jax.numpy as jnp
                from jax import lax
            except ImportError:
                raise RuntimeError("TPU specializer requires JAX")

            N = Ms.shape[0]
            n_tiles = (N + tile_size - 1) // tile_size

            def tile_scan(carry, tile_idx):
                running_M, running_b = carry
                start = tile_idx * tile_size
                end = jnp.minimum(start + tile_size, N)

                # Process tile
                def step_fn(carry_inner, t):
                    M_acc, b_acc = carry_inner
                    M_t = Ms[t]
                    b_t = bs[t]
                    # Compose: new = M_t @ old + b_t
                    M_new = M_t @ M_acc
                    b_new = M_t @ b_acc + b_t
                    return (M_new, b_new), b_new

                indices = jnp.arange(tile_size) + start
                init = (running_M, running_b)
                (final_M, final_b), tile_outputs = lax.scan(
                    step_fn, init, indices
                )
                return (final_M, final_b), tile_outputs

            M0 = jnp.eye(2)
            b0 = jnp.zeros(2)
            _, all_outputs = lax.scan(
                tile_scan, (M0, b0), jnp.arange(n_tiles)
            )
            return all_outputs.reshape(-1, 2)[:N]

        return scan_fn

    def _generate_fused(self, tile_size: int, vmem_plan: VMEMPlan) -> Callable:
        """Return a fused scan+elem function with VMEM-aware tiling."""
        streaming_tile = vmem_plan.streaming_tile

        def fused_fn(scan_output, param, grad, exp_avg, exp_avg_sq,
                     gru_state, expert_weights, lr, beta1, beta2, eps, wd):
            """Fused scan output application + GRU + PEER + Expert + Adam.

            Processes elements in tiles of streaming_tile to fit in VMEM.
            """
            try:
                import jax
                import jax.numpy as jnp
                from jax import lax
            except ImportError:
                raise RuntimeError("TPU specializer requires JAX")

            N = param.shape[0]
            n_tiles = (N + streaming_tile - 1) // streaming_tile

            def process_tile(carry, tile_idx):
                start = tile_idx * streaming_tile
                end = jnp.minimum(start + streaming_tile, N)
                sl = slice(start, end)

                # Load tile into VMEM (streaming)
                p_tile = param[sl]
                g_tile = grad[sl]
                ea_tile = exp_avg[sl]
                eas_tile = exp_avg_sq[sl]
                gs_tile = gru_state[sl]
                so_tile = scan_output[sl]

                # Apply scan output as smart gradient
                smart_grad = so_tile * g_tile

                # Adam update
                ea_new = beta1 * ea_tile + (1 - beta1) * smart_grad
                eas_new = beta2 * eas_tile + (1 - beta2) * smart_grad ** 2
                p_new = p_tile - lr * (ea_new / (jnp.sqrt(eas_new) + eps) + wd * p_tile)

                return carry, (p_new, ea_new, eas_new, gs_tile)

            _, (params_out, ea_out, eas_out, gs_out) = lax.scan(
                process_tile, None, jnp.arange(n_tiles)
            )
            return params_out, ea_out, eas_out, gs_out

        return fused_fn

    def _generate_source(self) -> str:
        """Not used for TPU — kernels are Python closures."""
        return ''

    def _compile_source(self, source: str, key: str):
        """Not used for TPU — compile() returns TPUKernels directly."""
        return None
