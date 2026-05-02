// =====================================================================
//  csrc/kernels/hip/gfx942/_common.hip.h
//
//  Shared host-compiler-only includes for the gfx942 optimizer launcher
//  TUs in this directory.
//
//  IMPORTANT: PyTorch's HIP build routes file extensions like this:
//    .hip / .cu / .cuh  -> hipcc / nvcc (device-side compiler)
//    .cpp / .hip.cpp    -> host compiler (g++/clang++)
//
//  All files in this directory are .hip.cpp and therefore go through
//  the HOST compiler. They CANNOT contain `__global__` kernels or
//  CUDA/HIP launch syntax (`<<<...>>>`). Instead, the optimizer math is
//  expressed using ATen tensor ops (which dispatch under the hood to
//  rocBLAS / hipFFT / etc.).
//
//  Performance notes: this is a "compile + link correctly" port. The
//  per-element math is functional but goes through ATen's general
//  reduction/elementwise paths (~50-70% of an MFMA-tuned port). Future
//  work: rename to .hip and add a hipLaunchKernelGGL fast path for the
//  hot elementwise kernels (Lion, Adam, Prodigy).
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
