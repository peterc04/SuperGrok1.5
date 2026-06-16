// examples/toy_tune_project/toy_kernel.cu
// ─────────────────────────────────────────────────────────────────────────
// A deliberately TRIVIAL, NON-grokking CUDA project used to PROVE that the
// SuperGrok1.5 autotuner (grokking_optimizers/compile.py) is portable: it can
// tune a completely different project's kernel with ZERO edits to compile.py,
// driven only by a TOML config + a Python project hook.
//
// One elementwise op (SAXPY: y = a*x + b) over a 1-D float tensor, registered
// as a torch custom op so it loads via torch.ops.load_library (the simplest
// "torch_op"-style extension — exactly what compile.py's discovery probe and
// the tune_hook both expect).
//
// THE TUNABLE KNOB is the launch block size, exposed as a -D macro:
//
//     TOY_TUNED_BLOCK     (default 256)
//
// The autotuner compiles variants with -DTOY_TUNED_BLOCK=128 / 256 / 512 /
// 1024 and times each. The op is NUMERICALLY IDENTICAL for every value of
// TOY_TUNED_BLOCK (y[i] = a*x[i] + b) — that is the whole point: the knob
// changes only the launch geometry / codegen / occupancy (performance), never
// the result. So a portable autotuner can pick the fastest block size while
// the strict-math oracle confirms every variant matches bit-for-bit-tolerantly.
//
// Build (standalone, for verification):
//   torch.utils.cpp_extension.load(name='toy_saxpy',
//       sources=['toy_kernel.cu'],
//       extra_cuda_cflags=['-DTOY_TUNED_BLOCK=256'])
//
// Build (via the autotuner): see README.md — compile.py discovers this single
// .cu through its zero-config Tier-1 auto-glob fallback (no launch_*.cu /
// models/ layout required) and threads -DTOY_TUNED_BLOCK=<v> from the search
// space onto the nvcc command line.
// ─────────────────────────────────────────────────────────────────────────
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// ── The one tunable knob ──────────────────────────────────────────────────
// Overridden on the nvcc command line by the autotuner (-DTOY_TUNED_BLOCK=N)
// or by the standalone verification build. The #ifndef guard keeps the file
// compilable with no -D at all (default 256), so it builds anywhere.
#ifndef TOY_TUNED_BLOCK
#define TOY_TUNED_BLOCK 256
#endif

// Compile-time sanity: a CUDA block may have at most 1024 threads. Catching a
// bad tuned value here turns a runtime launch failure into a clear compile
// error (the autotuner treats a failed compile as an infeasible config).
static_assert(TOY_TUNED_BLOCK >= 1 && TOY_TUNED_BLOCK <= 1024,
              "TOY_TUNED_BLOCK must be in [1, 1024] (CUDA max threads/block)");

namespace {

// Grid-stride SAXPY: y[i] = a*x[i] + b. Correct for ANY blockDim/gridDim,
// which is exactly why the block size is a pure performance knob.
__global__ void toy_saxpy_kernel(float* __restrict__ y,
                                 const float* __restrict__ x,
                                 float a, float b, int64_t n) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        y[i] = a * x[i] + b;
    }
}

// In-place op: y is the mutable output (Tensor(a!) in the schema), x the input.
// Writing y = a*x + b. The schema marks y as aliased/mutable so compile.py's
// oracle snapshots y as the buffer to compare across variants.
void toy_saxpy(at::Tensor y, const at::Tensor& x, double a, double b) {
    TORCH_CHECK(x.is_cuda(), "toy::saxpy: x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "toy::saxpy: y must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "toy::saxpy: x must be float32");
    TORCH_CHECK(y.scalar_type() == at::kFloat,
                "toy::saxpy: y must be float32");
    TORCH_CHECK(x.numel() == y.numel(),
                "toy::saxpy: x and y must have the same number of elements");

    auto xc = x.contiguous();
    auto yc = y.contiguous();
    const int64_t n = xc.numel();
    if (n == 0) {
        return;
    }

    const int block = TOY_TUNED_BLOCK;                 // <-- the tuned knob
    int64_t grid64 = (n + block - 1) / block;
    // Cap the grid so the grid-stride loop covers the rest; keeps launch legal
    // on any size and keeps occupancy reasonable for the tiny toy workload.
    const int64_t kMaxGrid = 65535;
    const int grid = static_cast<int>(grid64 < kMaxGrid ? grid64 : kMaxGrid);

    auto stream = at::cuda::getCurrentCUDAStream();
    toy_saxpy_kernel<<<grid, block, 0, stream>>>(
        yc.data_ptr<float>(), xc.data_ptr<float>(),
        static_cast<float>(a), static_cast<float>(b), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // If the input/output weren't already contiguous, copy the result back.
    if (!y.is_contiguous()) {
        y.copy_(yc);
    }
}

}  // namespace

// Register as a torch custom op so the artifact loads via
// torch.ops.load_library(<.so>) and the op is callable as
// torch.ops.toy.saxpy(y, x, a, b). The "(a!)" alias annotation tells torch
// (and compile.py's schema parser) that y is mutated in place — compile.py's
// _parse_torch_schema picks y as the snapshot buffer for numerical validation.
TORCH_LIBRARY(toy, m) {
    m.def("saxpy(Tensor(a!) y, Tensor x, float a, float b) -> ()");
}

TORCH_LIBRARY_IMPL(toy, CUDA, m) {
    m.impl("saxpy", TORCH_FN(toy_saxpy));
}
