// =====================================================================
// dispatch.cpp — single-source arch detection + fused_step dispatch
//
// 3-arch active set: sm_90, gfx942, tpu_v5p.
// Detects exactly one of {sm_90, gfx942} on GPU and raises on anything
// else. No fallback. TPU arch is handled in Python via JAX.
// =====================================================================

#include "helpers.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(WITH_CUDA) && !defined(WITH_HIP)
 #include <cuda_runtime.h>
#elif defined(WITH_HIP)
 #include <hip/hip_runtime.h>
#endif

namespace sg {

namespace {

int detect_arch_from_env() {
 const char* force = std::getenv("FORCE_ARCH");
 if (!force) return -1;
 std::string s(force);
 // Numeric forms.
 if (s == "90") return 90;
 if (s == "942") return 942;
 // Symbolic forms.
 if (s == "sm_90") return 90;
 if (s == "gfx942") return 942;
 // tpu_v5p is handled in Python; not valid for C++ dispatch.
 if (s == "tpu_v5p") {
 throw std::runtime_error(
 "FORCE_ARCH=tpu_v5p is not a GPU arch; TPU dispatch is handled "
 "in Python via JAX.");
 }
 throw std::runtime_error(
 "FORCE_ARCH=" + s +
 " not in supported 3-arch active set {90, 942, tpu_v5p}. "
 "Only sm_90 (Hopper) and gfx942 (MI300X/MI300A) are supported on GPU.");
}

int detect_arch_from_device() {
#if defined(WITH_CUDA) && !defined(WITH_HIP)
 int dev = 0;
 cudaError_t err = cudaGetDevice(&dev);
 if (err != cudaSuccess) {
 throw std::runtime_error(
 std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
 }
 cudaDeviceProp prop;
 err = cudaGetDeviceProperties(&prop, dev);
 if (err != cudaSuccess) {
 throw std::runtime_error(
 std::string("cudaGetDeviceProperties failed: ") +
 cudaGetErrorString(err));
 }
 int sm = prop.major * 10 + prop.minor;
 if (sm == 90) return 90; // Hopper (H100/H200)
 throw std::runtime_error(
 "Detected sm_" + std::to_string(sm) +
 "; only sm_90 (Hopper) is in the active set.");
#elif defined(WITH_HIP)
 int dev = 0;
 hipError_t err = hipGetDevice(&dev);
 if (err != hipSuccess) {
 throw std::runtime_error(
 std::string("hipGetDevice failed: ") + hipGetErrorString(err));
 }
 hipDeviceProp_t prop;
 err = hipGetDeviceProperties(&prop, dev);
 if (err != hipSuccess) {
 throw std::runtime_error(
 std::string("hipGetDeviceProperties failed: ") +
 hipGetErrorString(err));
 }
 std::string arch_name = prop.gcnArchName; // e.g. "gfx942:sramecc+:xnack-"
 auto colon = arch_name.find(':');
 if (colon != std::string::npos) arch_name = arch_name.substr(0, colon);
 if (arch_name == "gfx942") return 942;
 throw std::runtime_error(
 "Detected " + arch_name +
 "; only gfx942 (MI300X/MI300A) is in the active set.");
#else
 throw std::runtime_error(
 "No GPU backend compiled in. Build with WITH_CUDA or WITH_HIP. "
 "3-arch active set: sm_90, gfx942, tpu_v5p.");
#endif
}

} // anonymous namespace

int detect_arch() {
 int env_arch = detect_arch_from_env();
 if (env_arch >= 0) return env_arch;
 return detect_arch_from_device();
}

// =====================================================================
// fused_step — placeholder for future fused (model, optimizer, arch) TUs
//
// The csrc/fused/ stub directory was removed; this function exists only
// to keep the pybind11 surface stable for callers that probed for it.
// Re-implementing it requires landing real fused TUs first.
// =====================================================================

void fused_step(const std::string& model, const std::string& optimizer,
 torch::Tensor params, torch::Tensor input,
 torch::Tensor grad, torch::Tensor state, float lr) {
 int arch = detect_arch();
 std::string arch_str = (arch == 90) ? "sm_90" : "gfx942";
 throw std::runtime_error(
 "fused kernel not yet compiled for (" + model + ", " + optimizer +
 ", " + arch_str + ")");
}

} // namespace sg
