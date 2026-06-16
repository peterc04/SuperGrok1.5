export CUDA_HOME=/usr/local/cuda
# sccache compile cache on the PERSISTENT volume (not /dev/shm RAM, which is wiped
# on every instance cut — that is why post-teleport rebuilds were all cold misses).
# Surviving cuts means a recompile after a teleport reuses the cached objects.
export SCCACHE_DIR=/workspace/.sccache
# Cache nvcc compiles that carry multiple -gencode targets (otherwise sccache
# forwards them uncached). Harmless for the single sm_90a target.
export SCCACHE_CACHE_MULTIARCH=1
# Generous cache ceiling so a full _ops object set is not evicted between builds.
export SCCACHE_CACHE_SIZE=20G
export TMPDIR=/dev/shm/tmp
export TORCH_EXTENSIONS_DIR=/workspace/SuperGrok1.5/build/torch_ext
export CUDA_MPS_PIPE_DIRECTORY=/nonexistent
mkdir -p /dev/shm/tmp /workspace/.sccache
