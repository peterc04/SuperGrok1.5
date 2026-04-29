"""PTX Scheduler — Generate hardware-specific inline PTX for hot inner loops.

Maps SM version to FMA pipeline count and instruction capabilities,
then generates PTX with optimal instruction interleaving:
  - sm_75 (Turing): 2 FMA pipes, no cp.async
  - sm_80/86 (Ampere): 4 FMA pipes, cp.async
  - sm_89 (Ada Lovelace): 4 FMA pipes, cp.async
  - sm_90 (Hopper): 4 FMA pipes, cp.async, wgmma, TMA
  - sm_100 (Blackwell): 4 FMA pipes, cp.async, TMA, FP4 tensor cores
"""

from typing import Dict, Any


class PTXScheduler:
    """Generate hardware-specific PTX for hot inner loops."""

    # Pipeline configuration per SM version
    SM_CONFIGS: Dict[int, Dict[str, Any]] = {
        75:  {'fma_pipes': 2, 'has_async': False, 'has_wgmma': False, 'has_tma': False},
        80:  {'fma_pipes': 4, 'has_async': True,  'has_wgmma': False, 'has_tma': False},
        86:  {'fma_pipes': 4, 'has_async': True,  'has_wgmma': False, 'has_tma': False},
        89:  {'fma_pipes': 4, 'has_async': True,  'has_wgmma': False, 'has_tma': False},
        90:  {'fma_pipes': 4, 'has_async': True,  'has_wgmma': True,  'has_tma': True},
        100: {'fma_pipes': 4, 'has_async': True,  'has_wgmma': True,  'has_tma': True},
    }

    def __init__(self, sm_version: int):
        self.sm = sm_version
        self.config = self.SM_CONFIGS.get(
            sm_version,
            {'fma_pipes': 2, 'has_async': False, 'has_wgmma': False, 'has_tma': False}
        )

    def generate_affine_combine(self) -> str:
        """PTX for 2x2 affine matrix composition.

        12 FMAs total. Interleaved into groups matching the FMA pipeline count
        to maximize throughput:
          sm_75: groups of 2 (2 FMA pipelines)
          sm_80+: groups of 4 (4 FMA pipelines)
        """
        pipes = self.config['fma_pipes']
        if pipes >= 4:
            return self._affine_combine_4pipe()
        else:
            return self._affine_combine_2pipe()

    def _affine_combine_4pipe(self) -> str:
        """4 FMA pipelines (A100, H100, B200).

        All 4 independent products issued back-to-back, then 4 dependent
        accumulates, then 4 bias terms. 3 waves of 4 = 12 FMAs.
        """
        return '''
        asm volatile(
            "fma.rn.f32 %0, %6, %12, 0f00000000;\\n\\t"
            "fma.rn.f32 %1, %6, %13, 0f00000000;\\n\\t"
            "fma.rn.f32 %2, %8, %12, 0f00000000;\\n\\t"
            "fma.rn.f32 %3, %8, %13, 0f00000000;\\n\\t"
            "fma.rn.f32 %0, %7, %14, %0;\\n\\t"
            "fma.rn.f32 %1, %7, %15, %1;\\n\\t"
            "fma.rn.f32 %2, %9, %14, %2;\\n\\t"
            "fma.rn.f32 %3, %9, %15, %3;\\n\\t"
            "fma.rn.f32 %4, %6, %16, %10;\\n\\t"
            "fma.rn.f32 %5, %8, %16, %11;\\n\\t"
            "fma.rn.f32 %4, %7, %17, %4;\\n\\t"
            "fma.rn.f32 %5, %9, %17, %5;\\n\\t"
            : "=f"(n00), "=f"(n01), "=f"(n10), "=f"(n11), "=f"(nb0), "=f"(nb1)
            : "f"(m00), "f"(m01), "f"(m10), "f"(m11), "f"(b0), "f"(b1),
              "f"(r00), "f"(r01), "f"(r10), "f"(r11), "f"(rb0), "f"(rb1)
        );'''

    def _affine_combine_2pipe(self) -> str:
        """2 FMA pipelines (T4, Turing). Interleave in pairs to fill both pipes."""
        return '''
        asm volatile(
            "fma.rn.f32 %0, %6, %12, 0f00000000;\\n\\t"
            "fma.rn.f32 %2, %8, %12, 0f00000000;\\n\\t"
            "fma.rn.f32 %0, %7, %14, %0;\\n\\t"
            "fma.rn.f32 %2, %9, %14, %2;\\n\\t"
            "fma.rn.f32 %1, %6, %13, 0f00000000;\\n\\t"
            "fma.rn.f32 %3, %8, %13, 0f00000000;\\n\\t"
            "fma.rn.f32 %1, %7, %15, %1;\\n\\t"
            "fma.rn.f32 %3, %9, %15, %3;\\n\\t"
            "fma.rn.f32 %4, %6, %16, %10;\\n\\t"
            "fma.rn.f32 %5, %8, %16, %11;\\n\\t"
            "fma.rn.f32 %4, %7, %17, %4;\\n\\t"
            "fma.rn.f32 %5, %9, %17, %5;\\n\\t"
            : "=f"(n00), "=f"(n01), "=f"(n10), "=f"(n11), "=f"(nb0), "=f"(nb1)
            : "f"(m00), "f"(m01), "f"(m10), "f"(m11), "f"(b0), "f"(b1),
              "f"(r00), "f"(r01), "f"(r10), "f"(r11), "f"(rb0), "f"(rb1)
        );'''

    def generate_stochastic_round(self) -> str:
        """Branchless stochastic rounding via selp.

        Same for all SM versions — selp is universal since sm_20.
        Uses PRNG state in register to generate random threshold,
        then selp.f32 to conditionally round up/down without branches.
        """
        return '''
        /* Branchless stochastic round via selp.
         * rng: 32-bit PRNG state (xorshift32)
         * val: FP32 value to round to FP16/BF16
         *
         * The fractional part of the FP16 representation determines
         * the probability of rounding up vs down.
         */
        asm volatile(
            "xor.b32 %0, %0, %0, shl 13;\\n\\t"
            "xor.b32 %0, %0, %0, shr 17;\\n\\t"
            "xor.b32 %0, %0, %0, shl 5;\\n\\t"
            : "+r"(rng)
        );
        float threshold = __uint_as_float((rng & 0x007FFFFF) | 0x3F800000) - 1.0f;
        asm volatile(
            "set.gt.f32.f32 %0, %1, %2;\\n\\t"
            "selp.f32 %0, %3, %4, %0;\\n\\t"
            : "=f"(rounded)
            : "f"(frac), "f"(threshold), "f"(ceil_val), "f"(floor_val)
        );'''

    def generate_expert_mlp(self, expert_hidden: int) -> str:
        """Unrolled expert MLP for known expert_hidden sizes.

        For expert_hidden=8 (the default), fully unrolls the inner loops
        to eliminate loop overhead and enable better register allocation.
        """
        if expert_hidden == 8:
            return self._expert_mlp_unrolled_8()
        elif expert_hidden == 16:
            return self._expert_mlp_unrolled_16()
        return '/* generic expert MLP loop — expert_hidden not specialized */'

    def _expert_mlp_unrolled_8(self) -> str:
        """Fully unrolled 2-layer MLP with hidden_dim=8.

        Layer 1: 2 inputs -> 8 hidden (16 FMAs + 8 GELU)
        Layer 2: 8 hidden -> 1 output (8 FMAs)
        Total: 24 FMAs + 8 activation evals
        """
        lines = ['/* Expert MLP unrolled: hidden_dim=8 */']
        # Layer 1: h[i] = GELU(W1[i,0]*in0 + W1[i,1]*in1 + b1[i])
        for i in range(8):
            lines.append(
                f'float h{i} = smem_W1[{i}*3+0] * in0 + smem_W1[{i}*3+1] * in1 + smem_b1[{i}];'
            )
            # Approximate GELU: x * sigmoid(1.702 * x)
            lines.append(
                f'h{i} = h{i} * (1.0f / (1.0f + __expf(-1.702f * h{i})));'
            )
        # Layer 2: out = sum(W2[i] * h[i]) + b2
        terms = ' + '.join(f'smem_W2[{i}] * h{i}' for i in range(8))
        lines.append(f'float expert_out = {terms} + smem_b2[0];')
        return '\n'.join(lines)

    def _expert_mlp_unrolled_16(self) -> str:
        """Fully unrolled 2-layer MLP with hidden_dim=16."""
        lines = ['/* Expert MLP unrolled: hidden_dim=16 */']
        for i in range(16):
            lines.append(
                f'float h{i} = smem_W1[{i}*3+0] * in0 + smem_W1[{i}*3+1] * in1 + smem_b1[{i}];'
            )
            lines.append(
                f'h{i} = h{i} * (1.0f / (1.0f + __expf(-1.702f * h{i})));'
            )
        terms = ' + '.join(f'smem_W2[{i}] * h{i}' for i in range(16))
        lines.append(f'float expert_out = {terms} + smem_b2[0];')
        return '\n'.join(lines)

    def generate_cp_async_load(self) -> str:
        """cp.async smem load (Ampere+) or plain LDG (pre-Ampere)."""
        if self.config['has_async']:
            return '''
            /* cp.async: 16-byte aligned global->shared copy */
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\\n\\t"
                :: "r"(smem_addr), "l"(gmem_addr)
            );'''
        else:
            return '''
            /* Pre-Ampere: manual LDG + STS */
            float4 tmp = *reinterpret_cast<const float4*>(gmem_addr);
            *reinterpret_cast<float4*>(smem_addr) = tmp;'''

    def generate_tma_load(self) -> str:
        """TMA descriptor-based load (Hopper+) or cp.async fallback."""
        if self.config.get('has_tma', False):
            return '''
            /* TMA bulk copy: hardware-scheduled transfer */
            if (threadIdx.x == 0) {
                cuda::memcpy_async(
                    smem_dst, tensorMap, cuda::aligned_size_t<128>(size), barrier
                );
            }
            barrier.arrive_and_wait();'''
        return self.generate_cp_async_load()
