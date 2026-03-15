# AMD HIP / ROCm Port — Architecture Notes

## Wavefront-64 vs Warp-32

AMD CDNA GPUs (MI100, MI200, MI300X) use **wavefront-64**: 64 threads execute
in lockstep, vs NVIDIA's warp of 32 threads.

### Impact on Kernels

1. **Warp shuffle reductions**: `__shfl_down` operates over 64 lanes.
   Reduction loops need `WARP_SIZE/2 = 32` as the starting offset (not 16).
   Handled by `platform.h` macros (`WARP_SIZE`, `SHFL_DOWN`).

2. **Shared memory reductions**: `NUM_WARPS = BLOCK_SIZE / WARP_SIZE`.
   With `BLOCK_SIZE=256` and `WARP_SIZE=64`, we get 4 warps instead of 8.
   The shared memory arrays for warp-level partial sums must be sized to
   `NUM_WARPS`, not hardcoded to 8.

3. **Occupancy**: Each wavefront uses 64 VGPR slots. Higher register pressure
   per wavefront means fewer concurrent wavefronts per Compute Unit.
   The Mamba scan kernels (d_inner=16 threads) run at <1 wavefront per CU
   on CDNA — same occupancy problem as on NVIDIA.

4. **No explicit sync masks**: HIP's `__shfl_down` does not take a mask
   parameter. All 64 lanes in a wavefront are always synchronized.
   The `SHFL_DOWN_SYNC(mask, val, offset)` macro ignores the mask on HIP.

5. **`__syncthreads()`**: Identical between CUDA and HIP.

6. **`atomicAdd`**: Identical between CUDA and HIP for float/int types.

## CDNA Matrix Cores

MI200 (gfx90a) and MI300X (gfx942) have Matrix Cores that accelerate:
- FP16 GEMM with FP32 accumulation
- BF16 GEMM with FP32 accumulation
- FP64 GEMM (MI200+)
- INT8 GEMM (MI300X)

These are accessed via rocBLAS (through PyTorch's ATen layer) — no kernel
changes needed for GEMM operations.

## Key Differences from CUDA

| Feature | CUDA | HIP (CDNA) |
|---------|------|------------|
| Warp/wavefront size | 32 | 64 |
| `__sincosf` | Device intrinsic | Use `sincosf` |
| `__ldg(ptr)` | L1 cache hint | Direct dereference |
| Shuffle mask | Required | Ignored |
| Shared memory bank width | 4 bytes (32 banks) | 4 bytes (32 banks) |
| L1/L2 cache | Explicit control | Compiler-managed |
| Thrust | CUDA Thrust | rocThrust |
| CUB | NVIDIA CUB | hipCUB |
| cuBLAS | cuBLAS | rocBLAS (via ATen) |
| TF32 | sm_80+ | Not available |
| FP8 | sm_89+/sm_90+ | Not available (MI300X has INT8) |

## Build Requirements

- ROCm 5.4+ (recommended 6.0+)
- PyTorch built with ROCm support
- hipcc compiler

```bash
# Install PyTorch with ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Build the extension
pip install -e .
```

## Known Limitations

1. **No TF32/FP8**: AMD does not have TF32 or FP8 Tensor Core equivalents.
   Projection GEMMs use FP32 or BF16 on AMD.
2. **No Ampere/Hopper tier kernels**: AMD uses GENERIC tier only.
   The sm_80 and sm_90 specialized kernels are not compiled for HIP.
3. **Thrust sort**: rocThrust provides the same API but may have different
   performance characteristics for small arrays.
