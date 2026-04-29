// =====================================================================
//  dispatch.cpp — single-source arch detection
//
//  Replaces csrc/common/dispatch.h's tier fallback chain. Detects exactly
//  one of the supported arches {sm_80, sm_90, sm_100, gfx942} and raises
//  on anything else. No fallback.
// =====================================================================

#include "bindings.h"

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
    if (s == "80")  return 80;
    if (s == "89")  return 89;
    if (s == "90")  return 90;
    if (s == "100") return 100;
    if (s == "103") return 103;
    if (s == "120") return 120;
    if (s == "942") return 942;
    if (s == "950") return 950;
    // Symbolic forms.
    if (s == "sm_80")  return 80;
    if (s == "sm_89")  return 89;
    if (s == "sm_90")  return 90;
    if (s == "sm_100") return 100;
    if (s == "sm_103") return 103;
    if (s == "sm_120") return 120;
    if (s == "gfx942") return 942;
    if (s == "gfx950") return 950;
    throw std::runtime_error(
        "FORCE_ARCH=" + s +
        " not in supported set {80, 89, 90, 100, 103, 120, 942, 950}");
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
    if (sm == 80 || sm == 86)   return 80;   // A100/A30/A10/RTX 30 → sm_80 binding
    if (sm == 89)               return 89;   // Ada (RTX 40, L40, L40S)
    if (sm == 90)               return 90;   // Hopper (H100/H200)
    if (sm == 100)              return 100;  // Datacenter Blackwell (B100/B200/GB200)
    if (sm == 103)              return 103;  // Blackwell Ultra (B300, GB300 NVL72)
    if (sm == 120 || sm > 120)  return 120;  // Consumer Blackwell (RTX 50, RTX PRO 6000)
    throw std::runtime_error(
        "Detected sm_" + std::to_string(sm) +
        " is not in supported set "
        "{sm_80, sm_89, sm_90, sm_100, sm_103, sm_120}. "
        "Build with FORCE_CUDA=1 or upgrade hardware.");
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
    std::string arch_name = prop.gcnArchName;  // e.g. "gfx942:sramecc+:xnack-"
    auto colon = arch_name.find(':');
    if (colon != std::string::npos) arch_name = arch_name.substr(0, colon);
    if (arch_name == "gfx942") return 942;
    if (arch_name == "gfx950") return 950;
    throw std::runtime_error(
        "Detected " + arch_name +
        " is not in supported set {gfx942, gfx950}. "
        "Use MI300X / MI350X / MI355X or upgrade.");
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

} // namespace sg
