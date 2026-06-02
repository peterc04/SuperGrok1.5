// Thin shim — canonical 'vit' sm_90 device kernels live in the kernel tree.
// Kept at this path so existing includers (bindings.cpp, vit.cu, HIP refs)
// resolve unchanged. See the header for the migration rationale.
#include "grokking_optimizers/kernels/sm_90/vit_sm90.cuh"
