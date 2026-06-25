#include "csrc/fused/sm_90/mem_config.cuh"
int main(){ using M=sg::fused::mem::InHbm; static_assert(!M::kAnyOffHbm,"default off"); sg::fused::mem::MemRuntime r; (void)r; return (int)M::kStreamDepth; }
