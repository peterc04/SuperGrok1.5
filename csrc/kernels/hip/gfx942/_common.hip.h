// =====================================================================
//  csrc/kernels/hip/gfx942/_common.hip.h
//
//  Shared infrastructure for all gfx942 optimizer launchers.
//  - HIP runtime + dtype includes
//  - Common dispatch helpers and error checking
//  - Block-size constant
//
//  This header is included by every <opt>.hip.cpp in this directory.
//  It is INTERNAL — not exposed via the bindings forward decls.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sg { namespace gfx942 { namespace common {

// CDNA3 wave-64; 256 threads = 4 waves per workgroup. Suitable for the
// elementwise optimizers in this directory. Performance optimization
// (MFMA, larger waves_per_eu hints, etc.) is future work.
constexpr int kBlockSize = 256;

inline void check_hip(hipError_t e, const char* where) {
    if (e != hipSuccess) {
        throw std::runtime_error(
            std::string(where) + ": " + hipGetErrorString(e));
    }
}

// FP8 dtypes are not supported on gfx942 in this port. Param/grad
// dispatch is over {FP32, BF16, FP16}. State is always FP32 (matching
// the Python optimizer convention).
template <typename Functor, typename... Args>
inline void dispatch_param_grad_match(at::ScalarType dtype, Args&&... args) {
    switch (dtype) {
        case at::ScalarType::Float:
            Functor::template run<float>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::BFloat16:
            Functor::template run<__hip_bfloat16>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::Half:
            Functor::template run<__half>(std::forward<Args>(args)...);
            break;
        default:
            throw std::runtime_error(
                std::string("gfx942: unsupported param/grad dtype ") +
                c10::toString(dtype));
    }
}

}}} // namespace sg::gfx942::common
