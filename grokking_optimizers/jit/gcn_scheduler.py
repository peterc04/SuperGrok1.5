"""GCN Scheduler — Generate AMD GCN ISA-specific code for CDNA architectures.

Maps CDNA version to instruction capabilities and generates
architecture-specific code for:
  - Expert weight loading (FP4 MFMA on CDNA4, BF16 MFMA on CDNA2/3)
  - State pack/unpack (FP6 native on CDNA4, INT8/BF16 software on earlier)
  - Wavefront-aware scan operations
  - Matrix core utilization for expert MLP
"""

from typing import Dict, Any


class GCNScheduler:
    """Generate AMD GCN ISA-specific code for CDNA architectures."""

    CDNA_CONFIGS: Dict[str, Dict[str, Any]] = {
        'cdna2': {
            'has_bf16_mfma': True,
            'has_fp8_mfma': False,
            'has_fp4': False,
            'has_fp6': False,
            'wavefront': 64,
            'mfma_size': 16,
            'lds_size': 65536,
        },
        'cdna3': {
            'has_bf16_mfma': True,
            'has_fp8_mfma': True,
            'has_fp4': False,
            'has_fp6': False,
            'wavefront': 64,
            'mfma_size': 16,
            'lds_size': 65536,
        },
        'cdna4': {
            'has_bf16_mfma': True,
            'has_fp8_mfma': True,
            'has_fp4': True,
            'has_fp6': True,
            'wavefront': 64,
            'mfma_size': 32,
            'lds_size': 131072,
        },
    }

    def __init__(self, cdna_version: str):
        self.cdna = cdna_version
        self.config = self.CDNA_CONFIGS.get(
            cdna_version,
            {'has_bf16_mfma': False, 'has_fp8_mfma': False,
             'has_fp4': False, 'has_fp6': False,
             'wavefront': 64, 'mfma_size': 16, 'lds_size': 65536}
        )

    def generate_expert_load(self) -> str:
        """Generate expert weight loading code.

        CDNA4: FP4 MFMA — hardware handles dequantization during matrix multiply.
        CDNA2/3: BF16 MFMA — software dequant if weights are quantized.
        Pre-CDNA2: FP32 — no matrix core acceleration.
        """
        if self.config['has_fp4']:
            return '''/* FP4 MFMA dequant — hardware handles conversion during V_MFMA.
             * Weights stored as packed FP4 E2M1 (2 values per byte).
             * The MFMA instruction accepts FP4 inputs directly and
             * produces FP32 accumulator outputs.
             *
             * v_mfma_f32_16x16x128_fp4 d[0:3], s[0:7], s[8:15], d[0:3]
             */
            __builtin_amdgcn_mfma_f32_16x16x128_fp4(
                expert_W1_fp4_packed, input_packed, acc);'''
        elif self.config['has_bf16_mfma']:
            return '''/* BF16 MFMA — software dequant if needed.
             * v_mfma_f32_16x16x16_bf16 d[0:3], s[0:1], s[2:3], d[0:3]
             */
            __builtin_amdgcn_mfma_f32_16x16x16_bf16(
                expert_W1_bf16, input_bf16, acc);'''
        return '''/* FP32 — no matrix core acceleration.
         * v_fmac_f32 used for expert MLP matmul.
         */'''

    def generate_state_pack_unpack(self) -> str:
        """Generate optimizer state packing/unpacking code.

        CDNA4: FP6 E3M2 — native hardware support for 6-bit storage.
        Earlier: INT8/BF16 software pack/unpack.
        """
        if self.config['has_fp6']:
            return '''/* FP6 E3M2 pack/unpack — native on CDNA4.
             * 3 FP6 values packed into 18 bits = 2.25 bytes.
             * Hardware provides dedicated pack/unpack instructions.
             *
             * Pack: v_cvt_pk_fp6_f32 — convert 3 FP32 to packed FP6
             * Unpack: v_cvt_f32_pk_fp6 — convert packed FP6 to 3 FP32
             */
            packed = __builtin_amdgcn_cvt_pk_fp6_f32(val0, val1, val2);
            /* Unpack: */
            __builtin_amdgcn_cvt_f32_pk_fp6(packed, &out0, &out1, &out2);'''
        elif self.config['has_bf16_mfma']:
            return '''/* BF16 pack/unpack — software conversion.
             * Pack: round FP32 to BF16 (truncate mantissa)
             * Unpack: zero-extend BF16 mantissa to FP32
             */
            unsigned short packed = __float2bfloat16(val);
            float unpacked = __bfloat162float(packed);'''
        return '''/* INT8 symmetric quantization — software.
         * scale = max(abs(tensor)) / 127.0f
         * packed = (int8_t)roundf(val / scale)
         * unpacked = (float)packed * scale
         */'''

    def generate_lds_load(self) -> str:
        """Generate LDS (Local Data Share) load pattern for this CDNA tier."""
        if self.config['lds_size'] >= 131072:
            return '''/* CDNA4: 128KB LDS — can fit all expert weights + scan state.
             * Use ds_read_b128 for 128-bit aligned loads.
             */'''
        return '''/* CDNA2/3: 64KB LDS — must tile expert weights.
         * Use ds_read_b64 for 64-bit loads within tile.
         */'''

    def generate_scan_reduction(self) -> str:
        """Generate wavefront-aware scan reduction.

        All CDNA architectures use wavefront-64 (64 threads execute in lockstep).
        The scan reduction uses DPP (Data Parallel Primitives) for intra-wavefront
        communication without LDS.
        """
        return '''/* DPP-based intra-wavefront scan (wavefront-64).
         * Uses v_mov_b32 with DPP modifiers for shift-and-combine.
         * 6 DPP rounds for 64-element scan: shift by 1, 2, 4, 8, 16, 32.
         */
        float scan_val = element_val;
        // Round 1: shift by 1
        scan_val += __builtin_amdgcn_ds_swizzle(scan_val, 0x041F);
        // Round 2: shift by 2
        scan_val += __builtin_amdgcn_ds_swizzle(scan_val, 0x021F);
        // Round 3: shift by 4
        scan_val += __builtin_amdgcn_ds_swizzle(scan_val, 0x011F);
        // Round 4: shift by 8
        scan_val += __builtin_amdgcn_ds_swizzle(scan_val, 0x001F);
        // Round 5: shift by 16 (cross-half-wavefront)
        scan_val += __builtin_amdgcn_readlane(scan_val, lane_id ^ 16);
        // Round 6: shift by 32 (cross-half-wavefront)
        scan_val += __builtin_amdgcn_readlane(scan_val, lane_id ^ 32);'''
