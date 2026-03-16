/*
 * PTX Intrinsics for SuperGrok v2 Blelloch Scan
 *
 * Provides affine_combine_ptx() — the critical inner-loop operation of
 * the parallel prefix scan.  Uses raw PTX FMA instructions to ensure
 * optimal ILP: the 12 FMAs are scheduled in 3 waves of 4 independent
 * instructions each, maximizing throughput on NVIDIA SMs.
 *
 * On HIP (AMD), falls back to fmaf()-based implementation.
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

#elif defined(__HIP_DEVICE_COMPILE__) || defined(GROK_HIP)

__device__ __forceinline__ Affine2x2 affine_combine_ptx(
    const Affine2x2& left, const Affine2x2& right
) {
    return affine_combine(left, right);
}

#endif
