// examples/autotune_demo/gemm_kernel.cu
// ===========================================================================
// A DELIBERATELY-NAIVE-by-default, parameterized tiled fp32 GEMM, built to
// demonstrate compile.py's autotuner leverage on a kernel that has NOT been
// hand-tuned. C[M,N] = A[M,K] * B[K,N], row-major, fp32.
//
// The four tuning knobs are exposed as compile.py-style `#ifndef SG_TUNED_*`
// guards so a `-DSG_TUNED_*` flag (emitted by compile.py's resolve_macros)
// overrides the in-source default. The defaults are chosen to be CORRECT but
// SLOW (a tiny 16x16 output tile with a 16-deep K strip and only 256 threads
// — essentially the textbook "shared-memory tile" at its least-effective
// shape, with no register blocking), so the autotuner has real headroom.
//
//   SG_TUNED_TILE_M   rows of C computed per block      (default 16)
//   SG_TUNED_TILE_N   cols of C computed per block       (default 16)
//   SG_TUNED_TILE_K   K-strip depth per main-loop step   (default 16)
//   SG_TUNED_BLOCK    threads per block                  (default 256)
//
// CORRECTNESS CONTRACT (holds for EVERY legal config the search explores):
//   * The kernel is a single shared-memory-tiled GEMM with FULL boundary
//     guards on M, N, and K, so it is numerically correct for ANY M, N, K and
//     ANY (TILE_M, TILE_N, TILE_K, BLOCK) satisfying the static_asserts below.
//   * Each thread computes exactly REG_TILE = (TILE_M*TILE_N)/BLOCK output
//     elements (a register micro-tile). We require BLOCK to divide
//     TILE_M*TILE_N (static_assert) so the work partitions evenly; the search
//     space (gemm_search_space.yaml) only ever offers legal combinations.
//   * Shared-memory footprint is (TILE_M*TILE_K + TILE_K*TILE_N) floats; the
//     largest legal combo in the search space stays well under 48 KB.
//
// The op is registered as a torch custom op `torch.ops.sg_demo.gemm(C, A, B)`
// via TORCH_LIBRARY, so the tune_hook can drive it with torch.ops.load_library
// (exactly like examples/toy_tune_project). The MATH is identical for every
// config (only tiling/launch geometry changes), so every variant must match
// the strict reference build's output — only elapsed_ms varies, which is what
// the autotuner minimizes.
// ===========================================================================
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef SG_TUNED_TILE_M
#define SG_TUNED_TILE_M 16
#endif
#ifndef SG_TUNED_TILE_N
#define SG_TUNED_TILE_N 16
#endif
#ifndef SG_TUNED_TILE_K
#define SG_TUNED_TILE_K 16
#endif
#ifndef SG_TUNED_BLOCK
#define SG_TUNED_BLOCK 256
#endif

namespace {

constexpr int TILE_M = SG_TUNED_TILE_M;
constexpr int TILE_N = SG_TUNED_TILE_N;
constexpr int TILE_K = SG_TUNED_TILE_K;
constexpr int BLOCK  = SG_TUNED_BLOCK;

// Per-thread register micro-tile: each thread owns REG_TILE outputs of the
// TILE_M x TILE_N block tile. Must partition evenly.
constexpr int OUT_PER_TILE = TILE_M * TILE_N;
static_assert(OUT_PER_TILE % BLOCK == 0,
              "SG_TUNED_BLOCK must divide SG_TUNED_TILE_M*SG_TUNED_TILE_N "
              "(each thread computes an integer number of outputs)");
constexpr int REG_TILE = OUT_PER_TILE / BLOCK;

// Shared-tile load also partitions over BLOCK threads (grid-stride guarded, so
// exact divisibility is NOT required, but keep tiles sane).
static_assert(TILE_M > 0 && TILE_N > 0 && TILE_K > 0 && BLOCK > 0,
              "all tuned tile/block dims must be positive");
static_assert(BLOCK <= 1024, "SG_TUNED_BLOCK exceeds the 1024 threads/block HW limit");
// 48 KB static smem ceiling (we do not opt into the dynamic >48KB attribute).
static_assert((TILE_M * TILE_K + TILE_K * TILE_N) * (int)sizeof(float) <= 48 * 1024,
              "tuned shared tile exceeds 48KB static smem");

__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    const int block_row = blockIdx.y * TILE_M;  // first C row this block owns
    const int block_col = blockIdx.x * TILE_N;  // first C col this block owns
    const int tid = threadIdx.x;

    // This thread owns REG_TILE elements of the block tile, laid out as a
    // contiguous strip starting at linear index (tid*REG_TILE) in row-major
    // (TILE_M x TILE_N) order. acc holds the running fp32 accumulators.
    float acc[REG_TILE];
#pragma unroll
    for (int r = 0; r < REG_TILE; ++r) acc[r] = 0.0f;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * TILE_K;

        // Cooperative, boundary-guarded load of As (TILE_M x TILE_K) and
        // Bs (TILE_K x TILE_N). Grid-stride over BLOCK threads so any BLOCK
        // legally fills the tile regardless of divisibility.
        for (int i = tid; i < TILE_M * TILE_K; i += BLOCK) {
            const int r = i / TILE_K;
            const int c = i % TILE_K;
            const int gr = block_row + r;
            const int gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int i = tid; i < TILE_K * TILE_N; i += BLOCK) {
            const int r = i / TILE_N;
            const int c = i % TILE_N;
            const int gr = k0 + r;
            const int gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        // Multiply the loaded strip into each of this thread's REG_TILE outputs.
#pragma unroll
        for (int r = 0; r < REG_TILE; ++r) {
            const int lin = tid * REG_TILE + r;   // linear idx in block tile
            const int tr = lin / TILE_N;          // row within tile
            const int tc = lin % TILE_N;          // col within tile
            float s = acc[r];
#pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                s += As[tr][kk] * Bs[kk][tc];
            }
            acc[r] = s;
        }
        __syncthreads();
    }

    // Write back, boundary-guarded.
#pragma unroll
    for (int r = 0; r < REG_TILE; ++r) {
        const int lin = tid * REG_TILE + r;
        const int tr = lin / TILE_N;
        const int tc = lin % TILE_N;
        const int gr = block_row + tr;
        const int gc = block_col + tc;
        if (gr < M && gc < N) {
            C[gr * N + gc] = acc[r];
        }
    }
}

// torch custom-op entry: C = A @ B. All three are 2-D contiguous fp32 CUDA
// tensors; C is pre-allocated [M,N] by the caller (the tune_hook).
void gemm(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                B.scalar_type() == torch::kFloat32 &&
                C.scalar_type() == torch::kFloat32, "all tensors must be fp32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "A,B,C must be 2-D");
    A = A.contiguous(); B = B.contiguous(); C = C.contiguous();
    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(1);
    TORCH_CHECK(B.size(0) == K, "inner dims must match: A is [M,K], B must be [K,N]");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C must be [M,N]");

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(BLOCK);
    gemm_tiled<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                C.data_ptr<float>(), M, N, K);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm launch failed: ", cudaGetErrorString(err));
}

}  // namespace

TORCH_LIBRARY(sg_demo, m) {
    m.def("gemm(Tensor C, Tensor A, Tensor B) -> ()");
    m.impl("gemm", torch::kCUDA, &gemm);
}
