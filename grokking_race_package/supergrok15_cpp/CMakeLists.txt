cmake_minimum_required(VERSION 3.18)
project(supergrok_bench LANGUAGES CXX CUDA)

# ── Find LibTorch ─────────────────────────────────────────────────────
# Set CMAKE_PREFIX_PATH to your LibTorch installation:
#   cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── CUDA Setup ────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target architectures (same as setup.py)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# ── Compiler Flags ────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -funroll-loops")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr -lineinfo")

# ── Source Files ──────────────────────────────────────────────────────
set(KERNEL_SOURCES
    csrc/kernels.cu
    csrc/grokfast_kernels.cu
    csrc/lion_kernels.cu
    csrc/looksam_kernels.cu
    csrc/muon_kernels.cu
)

set(CPP_SOURCES
    csrc/ops.cpp
    csrc/graph_compiler.cpp
    csrc/stream_scheduler.cpp
    csrc/train.cpp
)

set(HEADERS
    csrc/ops.h
    csrc/models.h
    csrc/graph_compiler.h
    csrc/stream_scheduler.h
)

# ── Standalone Training Binary ────────────────────────────────────────
add_executable(supergrok_bench
    ${KERNEL_SOURCES}
    ${CPP_SOURCES}
)

target_compile_definitions(supergrok_bench PRIVATE WITH_CUDA)
target_include_directories(supergrok_bench PRIVATE csrc)
target_link_libraries(supergrok_bench PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

# Ensure proper RPATH for finding libtorch at runtime
set_target_properties(supergrok_bench PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ── Optional: Shared library for Python extension ─────────────────────
# This builds the same kernels as a shared library that can be loaded
# via torch.utils.cpp_extension (alternative to setup.py build)
add_library(supergrok15_ops SHARED
    ${KERNEL_SOURCES}
    csrc/ops.cpp
    csrc/graph_compiler.cpp
    csrc/stream_scheduler.cpp
)

target_compile_definitions(supergrok15_ops PRIVATE WITH_CUDA)
target_include_directories(supergrok15_ops PRIVATE csrc)
target_link_libraries(supergrok15_ops PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

# ── Install targets ───────────────────────────────────────────────────
install(TARGETS supergrok_bench DESTINATION bin)
install(TARGETS supergrok15_ops DESTINATION lib)

# ── Print configuration ──────────────────────────────────────────────
message(STATUS "LibTorch: ${TORCH_LIBRARIES}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
