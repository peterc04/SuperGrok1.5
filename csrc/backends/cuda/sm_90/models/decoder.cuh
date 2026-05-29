// Thin shim — canonical 'decoder' sm_90 device kernels live in the kernel tree.
// Kept at this path so existing includers (bindings.cpp, decoder.cu, HIP refs)
// resolve unchanged. See the header for the migration rationale.
#include "grokking_optimizers/kernels/sm_90/transformer_decoder_sm90.cuh"
