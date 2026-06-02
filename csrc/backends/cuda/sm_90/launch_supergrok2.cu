// CUDA sm_90 thin TU for 'supergrok2'. All device logic — kernels, __device__
// helpers, inline PTX, and (muon/sg2) CUTLASS collectives — lives in the
// canonical header below; this TU exists so setup.py's *.cu glob still has a
// compilation unit that instantiates them and exposes the host launchers.
#include "grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh"
