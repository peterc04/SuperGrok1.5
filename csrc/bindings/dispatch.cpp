// =====================================================================
// dispatch.cpp — single-source arch detection + fused_step dispatch
//
// VENDOR-level dispatch. The kernel source is single-per-vendor (one CUDA
// tree compiled as sg::sm90, one HIP tree compiled as sg::gfx942) and is
// gencode/offload-compiled by setup.py for the full arch picture (every
// NVIDIA CC / AMD gfx the toolchain accepts). So detection normalises to a
// vendor selector: ANY NVIDIA device -> 90 (the sm90 impl), ANY AMD device
// -> 942 (the gfx942 impl). The right per-SM/-gfx SASS is selected by the
// driver from the fat binary; the host only needs to pick the vendor impl.
// TPU arch is handled in Python via JAX.
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
 // TPU is handled in Python; not valid for C++ dispatch.
 if (s.rfind("tpu", 0) == 0) {
 throw std::runtime_error(
 "FORCE_ARCH=" + s + " is not a GPU arch; TPU dispatch is handled "
 "in Python via JAX.");
 }
 // AMD: any gfx target (and the bare 942 form) -> the gfx942 impl.
 if (s == "942" || s.rfind("gfx", 0) == 0) return 942;
 // NVIDIA: any sm_* / smXX form -> the sm90 impl.
 if (s.rfind("sm_", 0) == 0 || s.rfind("sm", 0) == 0) return 90;
 // Bare numeric compute capability (e.g. "70", "90", "120") -> NVIDIA.
 if (!s.empty() && s.find_first_not_of("0123456789") == std::string::npos)
 return 90;
 throw std::runtime_error(
 "FORCE_ARCH=" + s +
 " not recognized. Use an NVIDIA arch (sm_70..sm_120 or a numeric "
 "compute capability), an AMD arch (gfx906..gfx1201), or run TPU via "
 "Python/JAX.");
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
 // Any modern NVIDIA device routes to the sm90 impl; the driver loads the
 // matching per-SM SASS (or JITs from the embedded PTX) out of the fat binary.
 if (sm >= 70) return 90;
 throw std::runtime_error(
 "Detected sm_" + std::to_string(sm) +
 " (compute capability < 7.0); below the minimum supported NVIDIA arch.");
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
 // Any AMD gfx device routes to the gfx942 impl; the offload-compiled fat
 // binary carries the matching per-gfx code object.
 if (arch_name.rfind("gfx", 0) == 0) return 942;
 throw std::runtime_error(
 "Detected " + arch_name +
 "; not a recognized AMD gfx target.");
#else
 throw std::runtime_error(
 "No GPU backend compiled in. Build with WITH_CUDA or WITH_HIP.");
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
