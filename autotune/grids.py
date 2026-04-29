"""Per-kernel autotune grids.

Each entry maps a kernel name to a list of parameter tuples. The runner
in ``runner.py`` walks each tuple, compiles the kernel template under
those parameters, runs the microbench, and picks the median-fastest.

Schema for each kernel:
    {
        "shape_buckets": [N_small, N_medium, N_large, ...],
        "axes": {
            "BLOCK_SIZE": [128, 256, 512],
            "STAGES":     [2, 3, 4],
            ...
        },
        "constraints": optional callable(axis_dict) -> bool,
    }

The `axes` cartesian product is the search space. `constraints` filters out
illegal combinations (e.g. shared-memory budget exceeded).

Shape buckets define the input sizes the kernel is profiled over. The
winning config per bucket is recorded; runtime dispatch picks by nearest
bucket. See csrc/common/tuned_configs.h for the layout the runner emits.
"""

from __future__ import annotations


# Common power-of-two N buckets used by per-element kernels.
ELEM_BUCKETS = [4096, 65536, 524288, 4194304, 33554432]


GRIDS = {
    # ─── Per-element optimizers (no GEMM) ────────────────────────────────
    "grokadamw_step": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 192, 256, 384, 512],
            "MIN_BLOCKS_PER_SM":   [1, 2, 4, 6, 8],
            "VEC4":                [False, True],
        },
    },
    "grokfast_adam": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256, 512],
            "MIN_BLOCKS_PER_SM":   [2, 4, 8],
            "VEC4":                [False, True],
        },
    },
    "lion_step": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256, 512],
            "MIN_BLOCKS_PER_SM":   [2, 4, 8],
            "VEC4":                [False, True],
        },
    },
    "prodigy_step": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256, 512],
            "MIN_BLOCKS_PER_SM":   [2, 4, 8],
        },
    },
    "neuralgrok_full_step": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256, 512],
            "AMPLIFIER_HIDDEN":    [16, 32, 64, 128],
        },
    },
    "looksam_direction_adjust_fused": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256, 512],
            "MIN_BLOCKS_PER_SM":   [2, 4, 8],
        },
    },
    # ─── Meta-net optimizers (small per-element MLP) ─────────────────────
    "sg15_full_step": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256],
            "META_HIDDEN":         [16, 32, 64, 128],
        },
    },
    "sg11_full_step": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256],
            "META_HIDDEN":         [16, 32, 64, 128],
        },
    },
    # ─── Reductions ──────────────────────────────────────────────────────
    "sg11_cosine_gate_reduce": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [256, 512, 1024],
            "GRID_STRIDE":         [True],
        },
    },
    "prodigy_dlr_reduce": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [256, 512, 1024],
        },
    },
    # ─── Muon NS GEMMs (delegated to CUTLASS profiler) ───────────────────
    # See cutlass_profile.py.
    "muon_ns_gemm": {
        "shape_buckets": [(64, 64), (256, 256), (1024, 1024), (4096, 4096)],
        "axes": "cutlass_profiler",
    },
    # ─── SG2 projection GEMMs (delegated to CUTLASS profiler) ────────────
    "sg2_in_proj_gemm": {
        "shape_buckets": [(1024, 8, 16), (4096, 8, 16), (16384, 8, 16),
                          (65536, 8, 16)],
        "axes": "cutlass_profiler",
    },
    "sg2_dt_proj_gemm_softplus": {
        "shape_buckets": [(1024, 16, 16), (4096, 16, 16), (16384, 16, 16)],
        "axes": "cutlass_profiler_with_epilogue",
        "epilogue": "softplus_bias",
    },
    "sg2_B_proj_gemm": {
        "shape_buckets": [(1024, 16, 16), (4096, 16, 16), (16384, 16, 16)],
        "axes": "cutlass_profiler",
    },
    "sg2_C_proj_gemm": {
        "shape_buckets": [(1024, 16, 16), (4096, 16, 16), (16384, 16, 16)],
        "axes": "cutlass_profiler",
    },
    # ─── SG2 fused element-wise step (per-element, large kernel) ─────────
    "sg2_fused_elem_step": {
        "shape_buckets": ELEM_BUCKETS,
        "axes": {
            "BLOCK_SIZE":          [128, 256, 384, 512],
            "MIN_BLOCKS_PER_SM":   [1, 2],
        },
    },
    # ─── SG2 parallel scan (Blelloch) ────────────────────────────────────
    "sg2_parallel_scan": {
        "shape_buckets": [256, 1024, 4096, 16384, 65536],
        "axes": {
            # PSCAN_BLOCK is currently a constexpr in csrc/common/types.h
            # (=512). When the autotuner is wired into the build, this
            # constexpr becomes a template parameter so the autotuner can
            # explore it.
            "PSCAN_BLOCK":         [256, 512, 1024],
            "WARPS_PER_BLOCK":     [4, 8, 16],
        },
    },
}
