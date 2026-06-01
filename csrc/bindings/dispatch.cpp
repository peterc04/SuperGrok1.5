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
// fused_step — Stage 6 L3 persistent-megakernel dispatch (§1.12)
//
// PREFERS a fused TU when one is present for (model, optimizer, arch):
// the generated csrc/fused/<arch>/ megakernels register a host launcher per
// solver-chosen cell. fused_step detects the arch, looks up the cell, and —
// if a real fused TU is wired for it — dispatches there; otherwise it throws a
// descriptive NotImplemented so the caller falls through to the existing
// per-op path (which remains the default and is UNTOUCHED).
//
// HONESTY: only the cells whose megakernel was actually instantiated in
// csrc/fused/sm_90/megakernel_demo.cu are dispatched here. Every other cell
// throws "no fused TU; use per-op path". The set of wired cells is the
// SG_FUSED_CELLS table below — it is the C++ mirror of the cells the generator
// (grokking_optimizers/megakernel_codegen.py) emits to disk.
// =====================================================================

#if defined(WITH_CUDA) && !defined(WITH_HIP)
namespace fused { namespace sm90 {
// The persistent-megakernel scratch the host zero-initializes (mirror of
// csrc/fused/megakernel_common.cuh::PersistentContext — kept layout-identical).
struct PersistentContext {
 int* g_next_task;
 unsigned* g_arrived;
 unsigned* g_generation;
 int n_tasks;
 unsigned n_ctas;
};
// Host launchers exported by megakernel_demo.cu for the wired cells.
cudaError_t mega_mamba3_adamw(
 PersistentContext, float*, const float*, float*, float*, float*, float*,
 const int*, const int*, float, int, cudaStream_t);
cudaError_t mega_decoder_lion(
 PersistentContext, float*, const float*, float*, float*, float*, float*,
 const int*, const int*, float, int, cudaStream_t);
cudaError_t mega_vit_supergrok15(
 PersistentContext, float*, const float*, float*, float*, float*, float*,
 const int*, const int*, float, int, cudaStream_t);
}} // namespace fused::sm90
#endif

namespace {

// Canonical name of the wired fused cell, or empty if none. This is the single
// source of truth in C++ for "which (model, optimizer, arch) has a real fused
// TU"; it is kept in sync with megakernel_codegen.py's wired-cell set.
std::string wired_fused_cell(const std::string& model,
 const std::string& optimizer, int arch) {
 const std::string a = (arch == 90) ? "sm_90"
 : (arch == 942) ? "gfx942" : "tpu_v5p";
 // The three representative L3 cells instantiated in the demo TUs (§1.5/§1.6).
 if (model == "mamba3" && optimizer == "adamw")
 return "l3:mamba3+adamw:" + a;
 if (model == "transformer_decoder" && optimizer == "lion")
 return "l3:transformer_decoder+lion:" + a;
 if (model == "vit" && optimizer == "supergrok15")
 return "l3:vit+supergrok15:" + a;
 return "";
}

} // anonymous namespace

void fused_step(const std::string& model, const std::string& optimizer,
 torch::Tensor params, torch::Tensor input,
 torch::Tensor grad, torch::Tensor state, float lr) {
 int arch = detect_arch();
 std::string arch_str = (arch == 90) ? "sm_90"
 : (arch == 942) ? "gfx942" : "tpu_v5p";

 const std::string cell = wired_fused_cell(model, optimizer, arch);
 if (cell.empty()) {
 // §1.12 fall-through: no fused TU for this cell. The caller uses the
 // existing per-op dispatch (the per-optimizer *_fused_step bindings),
 // which is untouched. We surface a clear, non-silent signal.
 throw std::runtime_error(
 "no fused TU for (" + model + ", " + optimizer + ", " + arch_str +
 "); use the per-op path. Wired L3 cells: (mamba3,adamw), "
 "(transformer_decoder,lion), (vit,supergrok15).");
 }

#if defined(WITH_CUDA) && !defined(WITH_HIP)
 if (arch == 90) {
 // Prepare the megakernel inputs from the tensors. The persistent scratch
 // (task counter + barrier state) is device-allocated and zero-initialized
 // here; the host launcher pins one CTA per SM. `state` carries the
 // optimizer state slices (m, v) concatenated; sizes/offsets describe the
 // per-parameter-tensor layout (here the single flattened tensor).
 auto opts_i = torch::TensorOptions().device(params.device()).dtype(torch::kInt32);
 auto opts_u = torch::TensorOptions().device(params.device()).dtype(torch::kInt32);
 torch::Tensor g_next = torch::zeros({1}, opts_i);
 torch::Tensor g_arrived = torch::zeros({1}, opts_u);
 torch::Tensor g_generation = torch::zeros({1}, opts_u);

 const int n = static_cast<int>(params.numel());
 torch::Tensor sizes = torch::full({1}, n, opts_i);
 torch::Tensor offsets = torch::zeros({1}, opts_i);

 // acts proxy + the m/v halves of state.
 torch::Tensor acts = torch::zeros_like(params);
 auto p = params.data_ptr<float>();
 auto in = input.numel() ? input.data_ptr<float>() : p;
 auto gr = grad.numel() ? grad.data_ptr<float>() : acts.data_ptr<float>();
 // state holds [m | v]; split it (fallback to zeros if undersized).
 torch::Tensor mv = state;
 if (mv.numel() < 2 * n) mv = torch::zeros({2 * n}, params.options());
 float* m = mv.data_ptr<float>();
 float* v = m + n;

 fused::sm90::PersistentContext ctx{
 g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(g_generation.data_ptr<int>()),
 n, 0u};

 cudaError_t err = cudaSuccess;
 if (model == "mamba3" && optimizer == "adamw")
 err = fused::sm90::mega_mamba3_adamw(
 ctx, p, in, acts.data_ptr<float>(), gr, m, v,
 sizes.data_ptr<int>(), offsets.data_ptr<int>(), lr, /*step=*/1, 0);
 else if (model == "transformer_decoder" && optimizer == "lion")
 err = fused::sm90::mega_decoder_lion(
 ctx, p, in, acts.data_ptr<float>(), gr, m, v,
 sizes.data_ptr<int>(), offsets.data_ptr<int>(), lr, 1, 0);
 else if (model == "vit" && optimizer == "supergrok15")
 err = fused::sm90::mega_vit_supergrok15(
 ctx, p, in, acts.data_ptr<float>(), gr, m, v,
 sizes.data_ptr<int>(), offsets.data_ptr<int>(), lr, 1, 0);

 if (err != cudaSuccess)
 throw std::runtime_error(
 std::string("fused megakernel launch failed for ") + cell + ": " +
 cudaGetErrorString(err));
 return;
 }
#endif

 // The cell is "wired" in the manifest but its TU is not compiled into THIS
 // build (e.g. the gfx942 megakernel is a header pending a hipcc .hip TU, or
 // a CPU/host build). Surface it honestly rather than silently no-op.
 throw std::runtime_error(
 "fused TU for " + cell + " is manifest-registered but not compiled into "
 "this build (gfx942 megakernel is hardware-gated 🟡; CPU build has no "
 "fused TU). Use the per-op path.");
}

} // namespace sg
