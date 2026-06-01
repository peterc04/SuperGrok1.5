// HIP gfx942 thin TU for 'grokfast'. The ATen + rocBLAS step logic lives in the
// canonical header below (see its AMDGCN-asm-status banner); this TU exists so
// setup.py's *.hip.cpp glob still has a compilation unit exposing the host
// launchers. namespace sg::gfx942 preserved for bindings.
#include "grokking_optimizers/kernels/gfx942/grokfast_gfx942.hip.hpp"
