/*
 * PTX Intrinsics for SuperGrok v2
 *
 * Hot-path intrinsics that replace multi-cycle standard library calls with
 * single-cycle PTX instructions:
 *
 *   affine_combine_ptx  — 12-FMA parallel prefix scan composition
 *   softplus_ptx        — log(1+exp(x)) via ex2.approx + lg2.approx (2 cycles vs ~16)
 *   fast_exp_ptx        — exp(x) via ex2.approx (1 cycle vs ~8)
 *   stochastic_round_ptx— branchless stochastic rounding for Config4 quantization
 *   gru_gates_ptx       — interleaved sigmoid pair for GRU z/r gates
 *
 * On HIP (AMD), all intrinsics fall back to standard math functions.
 */

#pragma once

#include "types.h"

#if defined(__CUDACC__) || defined(GROK_CUDA)

__device__ __forceinline__ Affine2x2 affine_combine_ptx(
    const Affine2x2& left, const Affine2x2& right
) {
    Affine2x2 out;
    // 12-FMA inline PTX for composing two Affine2x2 transforms.
    // Computes: M_out = M_right * M_left, b_out = M_right * b_left + b_right
    //
    // Wave 0 (cycle 0): 4 independent partial products for M_out
    // Wave 1 (cycle 4): 4 dependent accumulations for M_out + 2 bias starts
    // Wave 2 (cycle 8): 2 final bias accumulations
    asm volatile(
        // Wave 0: 4 independent partial products
        "fma.rn.f32 %0, %6, %12, 0f00000000;\n\t"   // m00  = r.m00 * l.m00
        "fma.rn.f32 %1, %6, %13, 0f00000000;\n\t"   // m01  = r.m00 * l.m01
        "fma.rn.f32 %2, %8, %12, 0f00000000;\n\t"   // m10  = r.m10 * l.m00
        "fma.rn.f32 %3, %8, %13, 0f00000000;\n\t"   // m11  = r.m10 * l.m01
        // Wave 1: accumulate cross-terms + begin bias
        "fma.rn.f32 %0, %7, %14, %0;\n\t"            // m00 += r.m01 * l.m10
        "fma.rn.f32 %1, %7, %15, %1;\n\t"            // m01 += r.m01 * l.m11
        "fma.rn.f32 %2, %9, %14, %2;\n\t"            // m10 += r.m11 * l.m10
        "fma.rn.f32 %3, %9, %15, %3;\n\t"            // m11 += r.m11 * l.m11
        "fma.rn.f32 %4, %6, %16, %10;\n\t"           // b0   = r.m00 * l.b0 + r.b0
        "fma.rn.f32 %5, %8, %16, %11;\n\t"           // b1   = r.m10 * l.b0 + r.b1
        // Wave 2: final bias accumulations
        "fma.rn.f32 %4, %7, %17, %4;\n\t"            // b0  += r.m01 * l.b1
        "fma.rn.f32 %5, %9, %17, %5;\n\t"            // b1  += r.m11 * l.b1
        : "=f"(out.m00), "=f"(out.m01), "=f"(out.m10), "=f"(out.m11),
          "=f"(out.b0), "=f"(out.b1)
        : "f"(right.m00), "f"(right.m01), "f"(right.m10), "f"(right.m11),
          "f"(right.b0), "f"(right.b1),
          "f"(left.m00), "f"(left.m01), "f"(left.m10), "f"(left.m11),
          "f"(left.b0), "f"(left.b1)
    );
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
//  softplus_ptx: log(1 + exp(x)) in ~2 cycles
//
//  Replaces logf(1.0f + expf(x)) (~16 cycles) with ex2.approx + lg2.approx.
//  Branchless saturation at x > 20 via selp.
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ float softplus_ptx(float x) {
    float result;
    asm volatile(
        "{\n\t"
        ".reg .f32 t, ex, ep1, lg;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 t, %1, 0f3FB8AA3B;\n\t"      // x * log2(e)
        "ex2.approx.f32 ex, t;\n\t"             // exp(x)
        "add.f32 ep1, ex, 0f3F800000;\n\t"      // 1 + exp(x)
        "lg2.approx.f32 lg, ep1;\n\t"           // log2(1+exp(x))
        "mul.f32 lg, lg, 0f3F317218;\n\t"       // * ln(2)
        "setp.gt.f32 p, %1, 0f41A00000;\n\t"    // x > 20.0?
        "selp.f32 %0, %1, lg, p;\n\t"           // branchless select
        "}\n\t"
        : "=f"(result) : "f"(x)
    );
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
//  fast_exp_ptx: exp(x) in 1 cycle via ex2.approx
//
//  Replaces __expf(A * dt) in scan. A_bar is always in (0,1).
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ float fast_exp_ptx(float x) {
    float result;
    asm volatile(
        "{\n\t"
        ".reg .f32 t;\n\t"
        "mul.f32 t, %1, 0f3FB8AA3B;\n\t"   // x * log2(e)
        "ex2.approx.f32 %0, t;\n\t"         // 2^t = exp(x)
        "}\n\t"
        : "=f"(result) : "f"(x)
    );
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
//  stochastic_round_ptx: branchless stochastic rounding for Config4
//
//  Replaces floor + branch + comparison with cvt.rmi + selp.
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ int stochastic_round_ptx(float x, unsigned rand_bits) {
    int result;
    asm volatile(
        "{\n\t"
        ".reg .f32 fl, frac, r;\n\t"
        ".reg .s32 ifl, up;\n\t"
        ".reg .pred p;\n\t"
        "cvt.rmi.f32.f32 fl, %1;\n\t"
        "sub.f32 frac, %1, fl;\n\t"
        "cvt.rn.f32.u32 r, %2;\n\t"
        "mul.f32 r, r, 0f2F800000;\n\t"
        "setp.lt.f32 p, r, frac;\n\t"
        "cvt.rzi.s32.f32 ifl, fl;\n\t"
        "selp.s32 up, 1, 0, p;\n\t"
        "add.s32 %0, ifl, up;\n\t"
        "}\n\t"
        : "=r"(result) : "f"(x), "r"(rand_bits)
    );
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
//  gru_gates_ptx: interleaved sigmoid pair for GRU z/r gates
//
//  Two independent sigmoid(wx + b) computations fill both FMA pipelines.
//  Uses rcp.approx for 1/(1+exp(-x)) instead of fdividef.
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void gru_gates_ptx(
    float wx_z, float bz, float wx_r, float br,
    float& z_out, float& r_out
) {
    asm volatile(
        "{\n\t"
        ".reg .f32 nz, nr, tz, tr, ez, er, dz, dr;\n\t"
        "add.f32 nz, %2, %3;\n\t"
        "add.f32 nr, %4, %5;\n\t"
        "neg.f32 nz, nz;\n\t"
        "neg.f32 nr, nr;\n\t"
        "mul.f32 tz, nz, 0f3FB8AA3B;\n\t"
        "mul.f32 tr, nr, 0f3FB8AA3B;\n\t"
        "ex2.approx.f32 ez, tz;\n\t"
        "ex2.approx.f32 er, tr;\n\t"
        "add.f32 dz, ez, 0f3F800000;\n\t"
        "add.f32 dr, er, 0f3F800000;\n\t"
        "rcp.approx.f32 %0, dz;\n\t"
        "rcp.approx.f32 %1, dr;\n\t"
        "}\n\t"
        : "=f"(z_out), "=f"(r_out)
        : "f"(wx_z), "f"(bz), "f"(wx_r), "f"(br)
    );
}

#elif defined(__HIP_DEVICE_COMPILE__) || defined(GROK_HIP)

__device__ __forceinline__ Affine2x2 affine_combine_ptx(
    const Affine2x2& left, const Affine2x2& right
) {
    return affine_combine(left, right);
}

__device__ __forceinline__ float softplus_ptx(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__device__ __forceinline__ float fast_exp_ptx(float x) {
    return expf(x);
}

__device__ __forceinline__ int stochastic_round_ptx(float x, unsigned rand_bits) {
    float fl = floorf(x);
    float frac = x - fl;
    float r = (float)rand_bits * (1.0f / 4294967296.0f);
    return (int)fl + (r < frac ? 1 : 0);
}

__device__ __forceinline__ void gru_gates_ptx(
    float wx_z, float bz, float wx_r, float br,
    float& z_out, float& r_out
) {
    z_out = 1.0f / (1.0f + expf(-(wx_z + bz)));
    r_out = 1.0f / (1.0f + expf(-(wx_r + br)));
}

#endif
