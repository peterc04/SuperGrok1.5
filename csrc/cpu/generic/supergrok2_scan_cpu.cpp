/*
 * SuperGrok v2 — CPU Mamba-3 Scan Utilities
 *
 * Additional scan helpers for the CPU path.
 * The main mamba3_scan_cpu implementation lives in cpu_kernels.cpp.
 *
 * This file provides:
 *   - SIMD-accelerated inner loops for the scan projection steps
 *   - Batched scan dispatch for multi-parameter steps
 */

#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD dispatch (defined in avx512/ or neon/)
extern bool simd_available();
extern void simd_matvec(const float* W, const float* x, float* out,
                        int rows, int cols);

// ═══════════════════════════════════════════════════════════════════
//  Optimized projection helpers for scan inner loops
// ═══════════════════════════════════════════════════════════════════

void scan_input_proj_cpu(
    const float* x_sorted,   // [N, d_model]
    const float* in_proj_W,  // [2*d_inner, d_model]
    float* x_branch,         // [d_inner]
    float* z_branch,         // [d_inner]
    int d_inner, int d_model, int elem_idx
) {
    const float* inp = x_sorted + elem_idx * d_model;

    if (simd_available() && d_model >= 4) {
        // Use SIMD for x_branch projection
        simd_matvec(in_proj_W, inp, x_branch, d_inner, d_model);
        simd_matvec(in_proj_W + d_inner * d_model, inp, z_branch, d_inner, d_model);
    } else {
        for (int j = 0; j < d_inner; j++) {
            float xv = 0.0f, zv = 0.0f;
            for (int d = 0; d < d_model; d++) {
                xv += in_proj_W[j * d_model + d] * inp[d];
                zv += in_proj_W[(j + d_inner) * d_model + d] * inp[d];
            }
            x_branch[j] = xv;
            z_branch[j] = zv;
        }
    }
}

void scan_dt_proj_cpu(
    const float* dt_proj_W,  // [d_inner, d_inner]
    const float* dt_proj_b,  // [d_inner]
    const float* x_branch,   // [d_inner]
    float* dt_val,           // [d_inner]
    int d_inner
) {
    if (simd_available() && d_inner >= 4) {
        simd_matvec(dt_proj_W, x_branch, dt_val, d_inner, d_inner);
        for (int j = 0; j < d_inner; j++) {
            float raw = dt_val[j] + dt_proj_b[j];
            dt_val[j] = (raw > 20.0f) ? raw : std::log(1.0f + std::exp(raw));
        }
    } else {
        for (int j = 0; j < d_inner; j++) {
            float raw = dt_proj_b[j];
            for (int k = 0; k < d_inner; k++)
                raw += dt_proj_W[j * d_inner + k] * x_branch[k];
            dt_val[j] = (raw > 20.0f) ? raw : std::log(1.0f + std::exp(raw));
        }
    }
}
