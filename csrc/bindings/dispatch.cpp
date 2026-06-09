// =====================================================================
// dispatch.cpp — single-source arch detection + fused_step dispatch
//
// Arch-honest LOUD-GATE dispatch (Phase 8). The kernel source has real bodies
// only for sm_90/sm_90a (one CUDA tree, sg::sm90) and gfx942 (one HIP tree,
// sg::gfx942). detect_arch_from_device() therefore returns 90 for NVIDIA
// major==9 and 942 for gfx942, and THROWS for any other CC/gfx (e.g. an A100
// sm_80, which has no TMA/WGMMA path) rather than silently running Hopper SASS
// on the wrong architecture. FORCE_ARCH is the explicit operator override.
// TPU arch is handled in Python via JAX.
// =====================================================================

#include "helpers.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#if defined(WITH_CUDA) && !defined(WITH_HIP)
 #include <cuda_runtime.h>
 #include <c10/cuda/CUDAStream.h>
#elif defined(WITH_HIP)
 #include <hip/hip_runtime.h>
 #include <c10/hip/HIPStream.h>
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
 // Query only the compute-capability attributes (cheap device-attribute
 // reads) instead of the full cudaGetDeviceProperties struct copy.
 int major = 0, minor = 0;
 err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
 if (err != cudaSuccess) {
 throw std::runtime_error(
 std::string("cudaDeviceGetAttribute(ComputeCapabilityMajor) failed: ") +
 cudaGetErrorString(err));
 }
 err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
 if (err != cudaSuccess) {
 throw std::runtime_error(
 std::string("cudaDeviceGetAttribute(ComputeCapabilityMinor) failed: ") +
 cudaGetErrorString(err));
 }
 int sm = major * 10 + minor;
 // ARCH-HONEST DISPATCH (Stage 4): route per the REAL compute capability,
 // not "any NVIDIA -> 90". The shipped device code has ONE NVIDIA kernel
 // BODY: the sm_90a Hopper impl (TMA / wgmma — instructions that do not
 // exist on Ampere/Ada). setup.py emits SASS for every CC the toolchain
 // accepts plus a PTX fallback, but that PTX is lowered from sm_90a-gated
 // source, so it cannot JIT-forward to sm_80/sm_86/sm_89 — an A100 routed
 // to the sm_90a path would hit an illegal-instruction fault (or worse,
 // undefined behavior), not a clean error.
 //
 // POLICY: LOUD-GATE. We accept the Hopper family (sm_90 / sm_90a, i.e.
 // major == 9) and route it to impl 90. Every other NVIDIA CC fails loudly
 // here. This is the truthful option because only sm_90a has a real kernel
 // body — there is no sm_80/sm_86/sm_89 impl to route to. If a per-CC body
 // is ever added, give it its own selector value and a case here.
 if (major == 9) return 90;  // Hopper (sm_90 / sm_90a)
 throw std::runtime_error(
 "Detected sm_" + std::to_string(sm) +
 ": no kernel body for this NVIDIA compute capability. The shipped "
 "device code has only the sm_90a (Hopper) body, whose TMA/wgmma "
 "instructions are not valid on this arch and cannot be JIT-forwarded "
 "from the embedded PTX. Supported: sm_90a (NVIDIA), gfx942 (AMD). "
 "Set FORCE_ARCH=sm_90 only on a real Hopper device.");
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
 // ARCH-HONEST DISPATCH (Stage 4), AMD twin of the NVIDIA loud-gate above.
 // The shipped AMD device code has ONE body: the gfx942 (CDNA3) impl. A
 // gfx906/gfx908/gfx90a/gfx1100 device routed to the gfx942 code object
 // would run an incompatible ISA. setup.py offload-compiles for every gfx
 // the toolchain accepts, but the kernel BODY is gfx942-specific, so route
 // only a real gfx942 to impl 942 and fail loudly otherwise.
 // POLICY: LOUD-GATE (only gfx942 has a body).
 if (arch_name == "gfx942") return 942;
 throw std::runtime_error(
 "Detected " + arch_name +
 ": no kernel body for this AMD gfx target. The shipped device code "
 "has only the gfx942 (CDNA3 / MI300X) body. Supported: sm_90a "
 "(NVIDIA), gfx942 (AMD).");
#else
 throw std::runtime_error(
 "No GPU backend compiled in. Build with WITH_CUDA or WITH_HIP.");
#endif
}

} // anonymous namespace

int detect_arch() {
 // The GPU arch (and FORCE_ARCH) cannot change mid-process, so resolve it
 // exactly once. SG_DISPATCH invokes this up to n_params×ns_steps per step;
 // memoizing turns the per-dispatch getenv + device query into a single
 // C++11-guaranteed-thread-safe lazy init that subsequent calls read for free.
 static const int cached_arch = [] {
 int env_arch = detect_arch_from_env();
 if (env_arch >= 0) return env_arch;
 return detect_arch_from_device();
 }();
 return cached_arch;
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
namespace fused {
// The persistent-megakernel scratch the host zero-initializes (mirror of
// csrc/fused/megakernel_common.cuh::PersistentContext — kept layout-identical).
//
// This MUST live in sg::fused (NOT sg::fused::sm90): the real struct in
// megakernel_common.cuh is declared in sg::fused, so every generated cell .cu
// (which opens sg::fused::sm90 and names PersistentContext unqualified) binds
// sg::fused::PersistentContext via enclosing-namespace lookup. The dispatch
// table .inc below opens sg::fused::sm90 too, so it resolves this same shadow
// by the identical lookup — making the call-site mangling match the .cu
// definitions. A shadow in sg::fused::sm90 would mangle the cell parameter as
// sg::fused::sm90::PersistentContext and fail to link against the .cu symbols.
struct PersistentContext {
 int* g_next_task;
 unsigned* g_arrived;
 unsigned* g_generation;
 int n_tasks;
 unsigned n_ctas;
};
} // namespace fused
// Phase 3 Stage 5: all 33 sm_90 cells are real component compositions
// (csrc/fused/sm_90/mega_<model>_<opt>.cu → fused_megakernel.cuh →
// opt_components.cuh/model_stages.cuh). This generated table declares every
// cell launcher and routes (model, optimizer) → the real symbol. It replaces
// the 3 hard-coded demo routes (the toy megakernel_demo.cu was deleted).
#include "csrc/fused/sm_90/fused_dispatch_table.inc"
#endif

// gfx942 twin (🟡 hipcc/MI300X only). Faithful mirror of the sm_90 block: the
// 33 gfx942 cells are real component compositions
// (csrc/fused/gfx942/mega_<m>_<o>.hip → fused_megakernel.hip.hpp →
// opt_components.hip.hpp/model_stages.hip.hpp), AMDGCN-gate-verified; their host
// hipLaunchKernelGGL launchers + this routing compile only under hipcc. The
// WITH_CUDA build #if-excludes this entire block.
#if defined(WITH_HIP)
#include <hip/hip_runtime.h>
namespace fused { namespace gfx942_mega {
struct PersistentContext {   // mirror of megakernel_common_hip.hip.hpp
 int* g_next_task;
 unsigned* g_arrived;
 unsigned* g_generation;
 int n_tasks;
 unsigned n_ctas;
};
}} // namespace fused::gfx942_mega
#include "csrc/fused/gfx942/fused_dispatch_table.inc"
#endif

namespace {

// Canonical name of the wired fused cell, or empty if none. This is the single
// source of truth in C++ for "which (model, optimizer, arch) has a real fused
// TU"; it is kept in sync with megakernel_codegen.py's wired-cell set.
// Phase 2: expanded to cover all 99 cells (3 models × 11 optimizers × 3 archs).
// Generated by: megakernel_codegen.py --dispatch-table
// wired_fused_cell — NOW generator-emitted (WS4): the C++ tier-tag map is
// derived from the same solver enumeration that emits the 99 cells, so it
// cannot hand-sync-drift. Regenerate via:
//   python -m grokking_optimizers.megakernel_codegen --dispatch-table > \
//     csrc/fused/fused_wired_cells.inc
#include "csrc/fused/fused_wired_cells.inc"

#if defined(WITH_CUDA) || defined(WITH_HIP)
// Per-(device, n) reusable scratch for fused_step. The barrier counters and the
// `acts` proxy were previously reallocated (and `acts` memset over the full
// param size) on EVERY dispatch; sizes/offsets are constant for a given n.
// We persist them keyed by (device_index, n) and only reset the mutable parts
// (zero the 3 barrier counters + the acts buffer) per step. Behavior is
// identical — same zero-initialized scratch handed to the launcher — but the
// allocations/the full-size memset of sizes/offsets happen once per shape.
struct FusedScratch {
 torch::Tensor g_next;       // int32 [1]
 torch::Tensor g_arrived;    // int32 [1] (reinterpreted as unsigned)
 torch::Tensor g_generation; // int32 [1] (reinterpreted as unsigned)
 torch::Tensor sizes;        // int32 [1] = n  (constant per n)
 torch::Tensor offsets;      // int32 [1] = 0  (constant)
 torch::Tensor acts;         // float [numel(params)] proxy buffer
};

// Keyed by (device_index, n). n disambiguates differing param shapes on the
// same device; acts is sized to params.numel() (== n here, single flat tensor).
inline FusedScratch& fused_scratch_for(const torch::Tensor& params, int64_t n) {
 static std::unordered_map<long long, FusedScratch> cache;
 const long long dev_idx =
 params.device().is_cuda() ? static_cast<long long>(params.device().index())
 : -1LL;
 const long long key = (dev_idx << 40) ^ static_cast<long long>(n);
 auto it = cache.find(key);
 if (it == cache.end()) {
 auto opts_i =
 torch::TensorOptions().device(params.device()).dtype(torch::kInt32);
 FusedScratch s;
 s.g_next = torch::zeros({1}, opts_i);
 s.g_arrived = torch::zeros({1}, opts_i);
 s.g_generation = torch::zeros({1}, opts_i);
 s.sizes = torch::full({1}, n, opts_i); // constant for this n
 s.offsets = torch::zeros({1}, opts_i); // constant
 s.acts = torch::zeros_like(params);
 it = cache.emplace(key, std::move(s)).first;
 } else {
 // Reset only the mutable scratch; sizes/offsets are invariant for this n.
 it->second.g_next.zero_();
 it->second.g_arrived.zero_();
 it->second.g_generation.zero_();
 it->second.acts.zero_();
 }
 return it->second;
}
#endif

} // anonymous namespace

void fused_step(const std::string& model, const std::string& optimizer,
 torch::Tensor params, torch::Tensor input,
 torch::Tensor grad, torch::Tensor state, float lr) {
 int arch = detect_arch();
 std::string arch_str = (arch == 90) ? "sm_90"
 : (arch == 942) ? "gfx942" : "tpu_v6e";

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
 //
 // Keep the element count in int64 at this boundary: a large flat param
 // ( > 2^31-1 elements ) truncated to `int` would wrap negative, making
 // the `3*n` state-size check pass spuriously and the `m + n` / `m + 2*n`
 // pointer offsets index out of bounds (silent OOB corruption). The
 // numel-derived state check and pointer math below all use this int64 n.
 const int64_t n = params.numel();

 // Reusable per-(device,n) scratch: counters/acts reset, sizes/offsets fixed.
 FusedScratch& sc = fused_scratch_for(params, n);

 // acts proxy + the m/v halves of state.
 auto p = params.data_ptr<float>();
 auto in = input.numel() ? input.data_ptr<float>() : p;
 auto gr = grad.numel() ? grad.data_ptr<float>() : sc.acts.data_ptr<float>();
 // state holds [m | v | extra]; split it. `extra` is the per-optimizer third
 // per-element buffer (ema/sam_dir/s_track/mu/orth/smart_grad); each cell binds
 // it to its FusedOptState field (adamw/lion/neuralgrok ignore it). Sized 3n.
 // An undersized state tensor was previously silently replaced by a fresh
 // zero buffer that was DISCARDED after the call (the optimizer state never
 // persisted) and reallocated every step — a latent correctness bug plus a
 // hot-path realloc. Surface it instead of silently corrupting state.
 if (state.numel() < 3 * n) {
 throw std::runtime_error(
 "fused_step: state tensor for (" + model + ", " + optimizer +
 ") has " + std::to_string(state.numel()) + " elements but needs at "
 "least 3*n = " + std::to_string(3 * n) +
 " (m|v|extra). Pass a persistent, correctly-sized state tensor.");
 }
 float* m = state.data_ptr<float>();
 float* v = m + n;
 float* extra = m + 2 * n;

 // The megakernel ABI (PersistentContext.n_tasks, the int32 sizes/offsets
 // tensors) is 32-bit per the layout-identical mirror struct and the
 // generated launchers. The fused indices are being widened by the fused
 // agent; until the device-side N is int64, fail loudly rather than feed a
 // truncated count into the kernel (which would silently process the wrong
 // task range). This gate fires only for params with > 2^31-1 elements.
 TORCH_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int>::max()),
 "fused_step: param has ", n, " elements, exceeding the int32 "
 "megakernel ABI (PersistentContext.n_tasks / sizes / offsets). "
 "Use the per-op path for params with > 2^31-1 elements.");

 fused::PersistentContext ctx{
 sc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()),
 static_cast<int>(n), 0u};

 // Use the current CUDA stream so the megakernel orders against the rest of
 // the model's work and is capturable in a CUDA graph (the legacy default
 // stream 0 serialized against all work and blocked graph capture).
 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

 // Route to the real composed cell launcher (all 33 sm_90 cells). `found`
 // is set false only if no cell matches (then we fall through to the honest
 // not-compiled signal below).
 bool found = false;
 cudaError_t err = fused::sm90::dispatch_sm90_cell(
 model, optimizer, ctx, p, in, sc.acts.data_ptr<float>(), gr, m, v, extra,
 sc.sizes.data_ptr<int>(), sc.offsets.data_ptr<int>(), lr, /*step=*/1,
 stream, &found);

 if (found) {
 if (err != cudaSuccess)
 throw std::runtime_error(
 std::string("fused megakernel launch failed for ") + cell + ": " +
 cudaGetErrorString(err));
 return;
 }
 }
#endif

#if defined(WITH_HIP)
 if (arch == 942) {
 // gfx942 real composition route (hipcc/MI300X). Mirror of the sm_90 path.
 // int64 element count at the boundary (see the sm_90 path for rationale):
 // a truncated `int` numel would wrap negative and make the 3*n check and
 // m+n/m+2*n pointer math index out of bounds.
 const int64_t n = params.numel();
 FusedScratch& sc = fused_scratch_for(params, n);
 auto p = params.data_ptr<float>();
 auto in = input.numel() ? input.data_ptr<float>() : p;
 auto gr = grad.numel() ? grad.data_ptr<float>() : sc.acts.data_ptr<float>();
 if (state.numel() < 3 * n) {
 throw std::runtime_error(
 "fused_step: state tensor for (" + model + ", " + optimizer +
 ") has " + std::to_string(state.numel()) + " elements but needs at "
 "least 3*n = " + std::to_string(3 * n) +
 " (m|v|extra). Pass a persistent, correctly-sized state tensor.");
 }
 float* m = state.data_ptr<float>();
 float* v = m + n; float* extra = m + 2 * n;
 // int32 megakernel ABI gate (mirror of the sm_90 path).
 TORCH_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int>::max()),
 "fused_step: param has ", n, " elements, exceeding the int32 "
 "megakernel ABI. Use the per-op path for > 2^31-1 elements.");
 fused::gfx942_mega::PersistentContext ctx{
 sc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()),
 static_cast<int>(n), 0u};
 // Current HIP stream for graph capture / proper ordering (was default 0).
 hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
 bool found = false;
 hipError_t herr = fused::gfx942_mega::dispatch_gfx942_cell(
 model, optimizer, ctx, p, in, sc.acts.data_ptr<float>(), gr, m, v, extra,
 sc.sizes.data_ptr<int>(), sc.offsets.data_ptr<int>(), lr, 1, stream, &found);
 if (found) {
 if (herr != hipSuccess)
 throw std::runtime_error(
 std::string("gfx942 fused megakernel launch failed for ") + cell +
 ": " + hipGetErrorString(herr));
 return;
 }
 }
#endif

 // The cell is "wired" in the manifest but its TU is not compiled into THIS
 // build (CPU/host build has no fused TU). Surface it honestly.
 throw std::runtime_error(
 "fused TU for " + cell + " is manifest-registered but not compiled into "
 "this build (CPU build has no fused TU). Use the per-op path.");
}

} // namespace sg
