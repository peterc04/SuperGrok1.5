// Thin shim — canonical gfx942 'attention' model logic lives in the kernel tree.
// Kept at this path so attention.hip.cpp resolves unchanged. See header for rationale
// and AMDGCN-asm status.
#include "grokking_optimizers/kernels/gfx942/attention_gfx942.hip.hpp"
