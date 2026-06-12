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
// FusedScalars mirror — layout-identical to csrc/fused/sm_90/opt_components.cuh
// ::FusedScalars (the FULL runtime scalar set the host binds: lr, beta1, beta2,
// eps, wd, bc1, bc2, alpha, beta, lamb, alpha_max, gate, d_factor, neg_lr_scale,
// decay_factor — 15 floats). This MUST live in sg::fused::sm90 (NOT sg::fused):
// the generated cell launchers take `const FusedScalars&` resolved by enclosing-
// namespace lookup to sg::fused::sm90::FusedScalars; the dispatch table .inc
// (opened in sg::fused::sm90) resolves the same name the same way, so the call-
// site parameter mangling matches the .cu definitions. A mirror in sg::fused
// would mangle as sg::fused::FusedScalars and fail to link. Keep field order /
// types byte-identical to the real struct — fused_step memcpy-fills this and the
// cell reads it positionally via apply_scalars().
namespace sm90 {
struct FusedScalars {
 float lr, beta1, beta2, eps, wd, bc1, bc2, alpha, beta, lamb, alpha_max,
       gate, d_factor, neg_lr_scale, decay_factor,
       // GrokAdamW append-only widening (decoder L3-TC) — keep in lock-step with
       // the real fused::sm90::FusedScalars (opt_components.cuh). Inert defaults
       // (0.0) for every non-GrokAdamW caller; layout stays byte-identical.
       gamma, grad_clip,
       // Prodigy append-only widening (decoder L3-TC, STAGED global-d) — same
       // lock-step discipline. d0/d_coef/beta3 reach the kernel's P2.6 reduction;
       // inert/eager defaults (d_coef=1, beta3=0) for every non-Prodigy caller.
       d0, d_coef, beta3,
       // Muon append-only widening (ViT L3-TC, STAGED NS) — KEEP IN LOCK-STEP with
       // the real fused::sm90::FusedScalars (opt_components.cuh): same 3 trailing
       // fields, same order. The eager Muon's 1D-weight AdamW group has INDEPENDENT
       // hyperparameters (adamw_lr/adamw_betas, muon.py:115-125); these carry them
       // to the kernel's Muon P3 1D tail. Defaults = eager Muon adamw_* defaults;
       // every non-Muon caller leaves them untouched (only OptId::Muon's P3 reads).
       aux_lr, aux_beta1, aux_beta2,
       // LookSAM append-only widening (decoder/vit/mamba L3-TC, MODEL-COUPLED SAM 2nd
       // backward) — KEEP IN LOCK-STEP with the real fused::sm90::FusedScalars
       // (opt_components.cuh): same 2 trailing fields, same order. rho = SAM
       // perturbation radius; looksam_sam = the every-k SAM-step gate (1.0 ⇒ run the
       // in-kernel perturb→2nd fwd+bwd→sam_dir=g_sam−g; 0.0 ⇒ reuse the cached
       // sam_dir). Inert defaults (0.0) for every non-LookSAM caller; byte-identical.
       rho, looksam_sam,
       // SuperGrok11/15 append-only widening (decoder/vit L3-TC, meta-net mu) — KEEP
       // IN LOCK-STEP with the real fused::sm90::FusedScalars (opt_components.cuh): two
       // trailing fields. sg_rescale = the SharpnessMetaNet rescale (mu = rescale·phi);
       // gate_temp = the SG11 cosine-gate temperature (gate = sigmoid(gate_temp·cos),
       // sg11_finalize_gate). The phi weights + sharpness are DEVICE pointers the cell
       // binds directly; only these scalars flow through the POD. Inert defaults
       // (sg_rescale=0.0, gate_temp=1.0) for every non-SG caller.
       sg_rescale, gate_temp;
};
} // namespace sm90
} // namespace fused
// Phase 3 Stage 5: all 33 sm_90 cells are real component compositions
// (csrc/fused/sm_90/mega_<model>_<opt>.cu → fused_megakernel.cuh →
// opt_components.cuh/model_stages.cuh). This generated table declares every
// cell launcher and routes (model, optimizer) → the real symbol. It replaces
// the 3 hard-coded demo routes (the toy megakernel_demo.cu was deleted).
#include "csrc/fused/sm_90/fused_dispatch_table.inc"

// PHASE 1 — extern decl of the TRUE L3 fused decoder launcher. Its DEFINITION
// lives in the nvcc-compiled csrc/fused/sm_90/mega_decoder_real_adamw.cu (which
// owns all the <<<>>> / __global__ / device-intrinsic code — dispatch.cpp is
// HOST-compiled and cannot host any of it). The boundary signature is decomposed
// to plain pointers/ints + the FusedScalars mirror (NO header-only types like
// DecoderTokenCtx/FusedOptState), and uses the same sg::fused::PersistentContext
// + sg::fused::sm90::FusedScalars mirror types the 33 surrogate cells already use
// — so the FQN/layout/mangling match the .cu definition (the existing cell cheat).
// kDecTotalElems / kDecNumTensors are NOT visible here (they live in the device
// header); fused_step's decoder branch passes the element count it already knows.
namespace fused { namespace sm90 {
cudaError_t mega_decoder_real_adamw(
    ::sg::fused::PersistentContext ctx, float* params,
    const int* tokens, const int* targets, int B,
    float* state, float* grad, float* workspace, float* loss_out,
    const int* sizes, const int* offsets,
    float lr, int step, const FusedScalars& scalars, cudaStream_t stream);
}}  // namespace fused::sm90

// PHASE 2 — extern decls of the TRUE L3 fused ViT + Mamba launchers. Definitions
// live in the nvcc-compiled csrc/fused/sm_90/mega_{vit,mamba}_real_adamw.cu (which
// own all the <<<>>> / __global__ / device-intrinsic code). Boundary signatures
// are plain pointers/ints + the FusedScalars mirror (NO header-only ViTInputCtx /
// MambaTokenCtx / FusedOptState), using the same sg::fused::PersistentContext +
// sg::fused::sm90::FusedScalars mirror types the cells already use — so FQN /
// layout / mangling match the .cu definitions. UNLIKE the decoder launcher these
// take NO sizes/offsets (the kernels read kVitSizes/kMambaSizes __constant__
// tables directly), matching the .cu definitions exactly — a mismatch is a loud
// link error, not silent. kVitTotalElems / kMambaTotalElems are NOT visible here
// (they live in the device headers); the branches below pass the element count
// they already know. ViT takes FLOAT patches; Mamba takes int tokens (the .cu
// signatures differ in the first input-pointer type).
namespace fused { namespace sm90 {
cudaError_t mega_vit_real_adamw(
    ::sg::fused::PersistentContext ctx, float* params,
    const float* patches, const int* targets, int B,
    float* state, float* grad, float* workspace, float* loss_out,
    float lr, int step, const FusedScalars& scalars, cudaStream_t stream);
cudaError_t mega_mamba_real_adamw(
    ::sg::fused::PersistentContext ctx, float* params,
    const int* tokens, const int* targets, int B,
    float* state, float* grad, float* workspace, float* loss_out,
    float lr, int step, const FusedScalars& scalars, cudaStream_t stream);

// R2.4 — extern decls of the WIRED tensor-core (bf16 wgmma) launchers. Their
// definitions live in csrc/fused/sm_90/mega_{decoder,vit}_real_adamw_tc_launcher.cu
// (compiled -DSG_TUNED_GEMM_IMPL=1, globbed into _ops; NO own pybind module — they
// call ONLY launch_*_megakernel_tc, never the scalar launcher template, so they
// co-reside with the scalar TUs without an ODR/duplicate-symbol clash — proven by
// the two-TU link probe at landing). The boundary mirrors the scalar launchers
// EXCEPT: (a) a trailing `int ncta_cap` (0 = one CTA/SM = full saturation; the race
// passes 0), and (b) the `workspace` pointer is UNUSED — the TC activations/grad
// scratch is a different size (model_*_tc dims), so each launcher TU owns its own
// cached device scratch internally; dispatch passes nullptr for it. `loss_out`
// still points at state[3*total] (read back by the Python wrapper). Mamba NOW has a
// TC launcher too (cycle-2 directive (c)): mega_mamba_real_adamw_tc_launcher.cu wires
// launch_fused_mamba_megakernel_tc into _ops. The 0.46× scalar-wins carve-out is a
// PERFORMANCE fact (scan-dominated) the roofline surfaces, not a correctness reason —
// the mamba TC kernel is 5/5-validated (test_mamba_tc.py). fp32 mamba still routes to
// the scalar megakernel; bf16 mamba×{adamw,lion,grokfast} now route to the TC path.
// opt_id (owner baseline directive): the OptId int selecting the in-kernel optimizer
// TAIL. The wgmma fwd+bwd is optimizer-independent; opt_id picks apply_optimizer<Opt>.
// 0=AdamW,1=Lion,2=Grokfast,3=GrokAdamW,6=NeuralGrok are the single-launch TC tails;
// any other value the launcher rejects with cudaErrorInvalidValue (→ LOUD throw here).
cudaError_t mega_decoder_real_adamw_tc(
    ::sg::fused::PersistentContext ctx, float* params,
    const int* tokens, const int* targets, int B,
    float* state, float* grad, float* workspace, float* loss_out,
    const int* sizes, const int* offsets,
    float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
    int ncta_cap, int opt_id);
cudaError_t mega_vit_real_adamw_tc(
    ::sg::fused::PersistentContext ctx, float* params,
    const float* patches, const int* targets, int B,
    float* state, float* grad, float* workspace, float* loss_out,
    float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
    int ncta_cap, int opt_id);   // opt_id: OptId int for the in-kernel tail
// Mamba TC launcher (cycle-2): same boundary as the decoder (int tokens), NO
// sizes/offsets (the kernel reads kMambaSizes/kMambaOffsets __constant__ tables),
// NO workspace (the launcher owns its own TC-sized scratch). opt_id ∈ {0,1,2}
// (adamw/lion/grokfast) for mamba's single-launch TC tails.
cudaError_t mega_mamba_real_adamw_tc(
    ::sg::fused::PersistentContext ctx, float* params,
    const int* tokens, const int* targets, int B,
    float* state, float* grad, float* workspace, float* loss_out,
    float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
    int ncta_cap, int opt_id);
// SuperGrok2 DEDICATED TC launchers (decoder/vit). SG2 needs the FULL CSA/HCA/PEER/
// GRU meta-net weight bundle (26 HBM ptrs) + per-tensor scalar ARRAYS (6, length P)
// + the meta-net state, none of which fit the FusedScalars POD or the generic
// mega_*_real_adamw_tc boundary — so these are PARALLEL entries (the generic
// launcher + the 28 byte-identical cells are UNTOUCHED). The state buffer carries
// [m|v|mu|loss|sharpness|slow|gru_state]; perm/unsort are built in-kernel.
cudaError_t mega_decoder_sg2_tc(
    ::sg::fused::PersistentContext ctx, float* params,
    const int* tokens, const int* targets, int B,
    float* state, float* grad, float* loss_out,
    const float* input_proj_W, const float* input_proj_b,
    const float* csa_q_W, const float* csa_k_W, const float* csa_v_W, const float* csa_out_W,
    const float* csa_compress_w, const float* csa_idx_DQ, const float* csa_idx_K,
    const float* hca_q_W, const float* hca_k_W, const float* hca_v_W, const float* hca_out_W,
    const float* gru_Wz, const float* gru_bz, const float* gru_Wr, const float* gru_br,
    const float* gru_Wh, const float* gru_bh,
    const float* peer_query_Ws, const float* prod_keys_A, const float* prod_keys_B,
    const float* expert_W1, const float* expert_b1, const float* expert_W2, const float* expert_b2,
    const float* sc_alpha, const float* sc_gru_decay, const float* sc_lamb_eff,
    const float* sc_beta1, const float* sc_bc1, const float* sc_bc2,
    float rescale, float beta2, float lr, float wd, float eps,
    float rho, float sam_on,
    int step, cudaStream_t stream, int ncta_cap);
cudaError_t mega_vit_sg2_tc(
    ::sg::fused::PersistentContext ctx, float* params,
    const float* patches, const int* targets, int B,
    float* state, float* grad, float* loss_out,
    const float* input_proj_W, const float* input_proj_b,
    const float* csa_q_W, const float* csa_k_W, const float* csa_v_W, const float* csa_out_W,
    const float* csa_compress_w, const float* csa_idx_DQ, const float* csa_idx_K,
    const float* hca_q_W, const float* hca_k_W, const float* hca_v_W, const float* hca_out_W,
    const float* gru_Wz, const float* gru_bz, const float* gru_Wr, const float* gru_br,
    const float* gru_Wh, const float* gru_bh,
    const float* peer_query_Ws, const float* prod_keys_A, const float* prod_keys_B,
    const float* expert_W1, const float* expert_b1, const float* expert_W2, const float* expert_b2,
    const float* sc_alpha, const float* sc_gru_decay, const float* sc_lamb_eff,
    const float* sc_beta1, const float* sc_bc1, const float* sc_bc2,
    float rescale, float beta2, float lr, float wd, float eps,
    float rho, float sam_on,
    int step, cudaStream_t stream, int ncta_cap);
// SuperGrok2 DEDICATED mamba TC launcher (the mamba twin of mega_{decoder,vit}_sg2_tc).
// int32 tokens (like the decoder), the same SG2 meta-net bundle + per-tensor scalar
// arrays + state layout. The mamba megakernel's P3-SG2 phase runs sg2_meta_stages.
cudaError_t mega_mamba_sg2_tc(
    ::sg::fused::PersistentContext ctx, float* params,
    const int* tokens, const int* targets, int B,
    float* state, float* grad, float* loss_out,
    const float* input_proj_W, const float* input_proj_b,
    const float* csa_q_W, const float* csa_k_W, const float* csa_v_W, const float* csa_out_W,
    const float* csa_compress_w, const float* csa_idx_DQ, const float* csa_idx_K,
    const float* hca_q_W, const float* hca_k_W, const float* hca_v_W, const float* hca_out_W,
    const float* gru_Wz, const float* gru_bz, const float* gru_Wr, const float* gru_br,
    const float* gru_Wh, const float* gru_bh,
    const float* peer_query_Ws, const float* prod_keys_A, const float* prod_keys_B,
    const float* expert_W1, const float* expert_b1, const float* expert_W2, const float* expert_b2,
    const float* sc_alpha, const float* sc_gru_decay, const float* sc_lamb_eff,
    const float* sc_beta1, const float* sc_bc1, const float* sc_bc2,
    float rescale, float beta2, float lr, float wd, float eps,
    float rho, float sam_on,
    int step, cudaStream_t stream, int ncta_cap);
}}  // namespace fused::sm90
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
// FusedScalars mirror — layout-identical to gfx942/opt_components.hip.hpp
// ::FusedScalars (same 15 floats). In sg::fused::gfx942_mega so the dispatch
// table's `const FusedScalars&` resolves to the same type the .hip cells define.
struct FusedScalars {
 float lr, beta1, beta2, eps, wd, bc1, bc2, alpha, beta, lamb, alpha_max,
       gate, d_factor, neg_lr_scale, decay_factor;
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

#if defined(WITH_CUDA) && !defined(WITH_HIP)
// PHASE-1 decoder L3-REAL: the flat param element count for the small decoder
// (vocab=99,d=128,heads=4,layers=2,seq=4). MUST equal
// sg::fused::sm90::kDecTotalElems — that device-side constant is NOT visible in
// this HOST-compiled TU (it lives in the device-only decoder_layout.cuh), so we
// mirror the literal here and cross-check it against params.numel() at the call
// site (a mismatch fails loudly). The .cu's static_asserts guard the device side.
constexpr int64_t kDecoderTotalElems = 422755;
constexpr int     kDecoderSeq        = 4;   // mirror of SG_DEC_SEQ

// PHASE-1 decoder L3-REAL scratch: per-CTA grad-partial workspace + barrier
// counters + a reduced-grad buffer. The workspace is nCTA*total grad partials +
// nCTA loss partials; sized for nCTA = #SMs (the launcher pins one CTA per SM).
// The per-tensor sizes/offsets are NOT built here — the kernel reads the
// generated __constant__ kDecSizes/kDecOffsets tables directly. Persisted + reset
// per step (the big workspace is zeroed IN-KERNEL by phase P0; the host only
// resets the barrier/task counters here).
struct DecoderScratch {
 torch::Tensor g_next;        // int32 [1]
 torch::Tensor g_arrived;     // int32 [1] (unsigned-reinterpreted)
 torch::Tensor g_generation;  // int32 [1] (unsigned-reinterpreted)
 torch::Tensor workspace;     // float [nCTA*total + nCTA] partials + loss slots
 int n_ctas = 0;
};

inline DecoderScratch& decoder_scratch_for(const torch::Tensor& params) {
 static std::unordered_map<long long, DecoderScratch> cache;
 const long long dev_idx =
 params.device().is_cuda() ? static_cast<long long>(params.device().index())
 : -1LL;
 const long long key = dev_idx;   // one decoder shape per device (fixed total)
 auto it = cache.find(key);
 const int64_t total = kDecoderTotalElems;
 if (it == cache.end()) {
 int dev = params.device().is_cuda() ? params.device().index() : 0;
 int n_sms = 1;
 cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
 auto opts_i =
 torch::TensorOptions().device(params.device()).dtype(torch::kInt32);
 auto opts_f =
 torch::TensorOptions().device(params.device()).dtype(torch::kFloat32);
 DecoderScratch s;
 s.g_next = torch::zeros({1}, opts_i);
 s.g_arrived = torch::zeros({1}, opts_i);
 s.g_generation = torch::zeros({1}, opts_i);
 // The reduced grad is now routed through the ABI `grad` tensor (exposed to the
 // caller for per-tensor parity); no internal grad scratch needed.
 // workspace: nCTA*total partials + nCTA loss slots. Zeroed in-kernel (P0);
 // allocate (not zero) — the in-kernel P0 + the loss reduce overwrite it.
 s.workspace = torch::empty({(int64_t)n_sms * total + n_sms}, opts_f);
 s.n_ctas = n_sms;
 it = cache.emplace(key, std::move(s)).first;
 } else {
 it->second.g_next.zero_();
 it->second.g_arrived.zero_();
 it->second.g_generation.zero_();
 }
 return it->second;
}

// PHASE-2 ViT L3-REAL: the flat param element count for the small ViT
// (p=97,d=128,heads=4,layers=2,patch=49,npatch=16). MUST equal
// sg::fused::sm90::kVitTotalElems (the device-side constant in vit_layout.cuh,
// NOT visible in this HOST-compiled TU); mirrored here and cross-checked against
// params.numel() at the call site (a mismatch fails loudly). The .cu's
// static_asserts guard the device side. ViT carries FLOAT patches (NOT int
// tokens), so its packing differs from the decoder/mamba — see the branch.
constexpr int64_t kVitTotalElems = 418017;
constexpr int     kVitPatchElems = 16 * 49;   // 784 (kNPatch*kPatch)

// PHASE-2 ViT L3-REAL scratch: per-CTA grad-partial workspace + barrier counters.
// Modeled BYTE-FOR-BYTE on DecoderScratch/decoder_scratch_for; only the total
// changes. The per-tensor sizes/offsets are NOT built here — the kernel reads the
// generated __constant__ kVitSizes/kVitOffsets tables directly.
struct ViTScratch {
 torch::Tensor g_next;        // int32 [1]
 torch::Tensor g_arrived;     // int32 [1] (unsigned-reinterpreted)
 torch::Tensor g_generation;  // int32 [1] (unsigned-reinterpreted)
 torch::Tensor workspace;     // float [nCTA*total + nCTA] partials + loss slots
 int n_ctas = 0;
};

inline ViTScratch& vit_scratch_for(const torch::Tensor& params) {
 static std::unordered_map<long long, ViTScratch> cache;
 const long long dev_idx =
 params.device().is_cuda() ? static_cast<long long>(params.device().index())
 : -1LL;
 const long long key = dev_idx;   // one ViT shape per device (fixed total)
 auto it = cache.find(key);
 const int64_t total = kVitTotalElems;
 if (it == cache.end()) {
 int dev = params.device().is_cuda() ? params.device().index() : 0;
 int n_sms = 1;
 cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
 auto opts_i =
 torch::TensorOptions().device(params.device()).dtype(torch::kInt32);
 auto opts_f =
 torch::TensorOptions().device(params.device()).dtype(torch::kFloat32);
 ViTScratch s;
 s.g_next = torch::zeros({1}, opts_i);
 s.g_arrived = torch::zeros({1}, opts_i);
 s.g_generation = torch::zeros({1}, opts_i);
 // workspace: nCTA*total partials + nCTA loss slots. Zeroed in-kernel (P0);
 // allocate (not zero) — the in-kernel P0 + the loss reduce overwrite it.
 s.workspace = torch::empty({(int64_t)n_sms * total + n_sms}, opts_f);
 s.n_ctas = n_sms;
 it = cache.emplace(key, std::move(s)).first;
 } else {
 it->second.g_next.zero_();
 it->second.g_arrived.zero_();
 it->second.g_generation.zero_();
 }
 return it->second;
}

// PHASE-2 Mamba L3-REAL: the flat param element count for the small Mamba
// (ntok=99,p=97,d=128,layers=2,seq=8). MUST equal sg::fused::sm90::
// kMambaTotalElems (the device-side constant in mamba3_layout.cuh). Mamba carries
// int32 tokens like the decoder (NOT ViT's float patches) — see the branch.
constexpr int64_t kMambaTotalElems = 259425;
constexpr int     kMambaSeq        = 8;   // mirror of SG_MB_SEQ

// PHASE-2 Mamba L3-REAL scratch: modeled on DecoderScratch/decoder_scratch_for.
struct MambaScratch {
 torch::Tensor g_next;        // int32 [1]
 torch::Tensor g_arrived;     // int32 [1] (unsigned-reinterpreted)
 torch::Tensor g_generation;  // int32 [1] (unsigned-reinterpreted)
 torch::Tensor workspace;     // float [nCTA*total + nCTA] partials + loss slots
 int n_ctas = 0;
};

inline MambaScratch& mamba_scratch_for(const torch::Tensor& params) {
 static std::unordered_map<long long, MambaScratch> cache;
 const long long dev_idx =
 params.device().is_cuda() ? static_cast<long long>(params.device().index())
 : -1LL;
 const long long key = dev_idx;   // one Mamba shape per device (fixed total)
 auto it = cache.find(key);
 const int64_t total = kMambaTotalElems;
 if (it == cache.end()) {
 int dev = params.device().is_cuda() ? params.device().index() : 0;
 int n_sms = 1;
 cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
 auto opts_i =
 torch::TensorOptions().device(params.device()).dtype(torch::kInt32);
 auto opts_f =
 torch::TensorOptions().device(params.device()).dtype(torch::kFloat32);
 MambaScratch s;
 s.g_next = torch::zeros({1}, opts_i);
 s.g_arrived = torch::zeros({1}, opts_i);
 s.g_generation = torch::zeros({1}, opts_i);
 // workspace: nCTA*total partials + nCTA loss slots. Zeroed in-kernel (P0).
 s.workspace = torch::empty({(int64_t)n_sms * total + n_sms}, opts_f);
 s.n_ctas = n_sms;
 it = cache.emplace(key, std::move(s)).first;
 } else {
 it->second.g_next.zero_();
 it->second.g_arrived.zero_();
 it->second.g_generation.zero_();
 }
 return it->second;
}
#endif

// OptId int for the in-kernel optimizer TAIL, MIRRORING
// csrc/fused/sm_90/opt_components.cuh::OptId (kept as a local int map so this TU
// stays free of the device header — same discipline as the FusedScalars mirror).
// Returns the OptId for a SINGLE-LAUNCH wgmma-capable tail (no precompute stage, no
// 2nd backward, no sharpness ABI), or -1 for an optimizer whose L3-TC path needs more
// than the fwd+bwd+tail single launch (SG11/SG15/SG2). LookSAM IS now wired: its SAM
// 2nd backward is an IN-KERNEL phase (P2.4) — a perturb→2nd fwd+bwd→sam_dir=g_sam−g
// loop between B2 and P3, gated to every-k SAM steps — so it remains a SINGLE
// persistent launch (the same shape as Prodigy's P2.6 / Muon's P2.7 in-kernel stages).
// The owner-baseline directive routes exactly the -1!=... set onto the TC driver; the
// rest FAIL LOUD (here / in the Python gate) with their cited blocker.
static int wgmma_tail_opt_id(const std::string& optimizer) {
    if (optimizer == "adamw")      return 0;   // OptId::AdamW
    if (optimizer == "lion")       return 1;   // OptId::Lion
    if (optimizer == "grokfast")   return 2;   // OptId::Grokfast
    if (optimizer == "grokadamw")  return 3;   // OptId::GrokAdamW
    if (optimizer == "looksam")    return 4;   // OptId::LookSAM (MODEL-COUPLED SAM 2nd
                                               // backward — in-kernel P2.4 perturb→2nd
                                               // fwd+bwd→sam_dir, SINGLE persistent launch)
    if (optimizer == "prodigy")    return 5;   // OptId::Prodigy (STAGED global-d,
                                               // in-kernel P2.6 reduction — still a
                                               // SINGLE persistent launch)
    if (optimizer == "neuralgrok") return 6;   // OptId::NeuralGrok
    if (optimizer == "muon")       return 7;   // OptId::Muon (STAGED grid-cooperative
                                               // Newton-Schulz — in-kernel P2.7, still a
                                               // SINGLE persistent launch; vit wired)
    if (optimizer == "supergrok11") return 8;  // OptId::SuperGrok11 (MODEL-COUPLED SAM 2nd
                                               // backward → sharpness=(g_sam−g)² in P2.4 +
                                               // per-tensor meta-net mu/gate precompute in
                                               // P2.45; SINGLE persistent launch)
    if (optimizer == "supergrok15") return 9;  // OptId::SuperGrok15 (same SAM 2nd backward +
                                               // per-tensor mu precompute; gate is a host
                                               // scalar — SINGLE persistent launch)
    if (optimizer == "supergrok2")  return 10; // OptId::SuperGrok2 (FULL CSA/HCA/PEER/GRU
                                               // meta-net as the optimizer phase: in-kernel
                                               // SEGMENTED SORT (STAGE -1) + SAM 2nd backward
                                               // → sharpness + sg2_meta_stages; routed via the
                                               // DEDICATED mega_*_sg2_tc entry, NOT the generic
                                               // single-launch tail switch). Decoder/vit only;
                                               // mamba stays -1 (code-absent + A/A/A race).
    return -1;                                 // unroutable on the single-launch TC path
}

} // anonymous namespace

// NOTE: the default argument VALUES live on the declaration in helpers.h (which
// bindings.cpp sees for the py::arg defaults). C++ forbids repeating a default
// argument on both the declaration and the definition, so the definition below
// lists the parameters WITHOUT defaults. Keep the two signatures in lock-step:
// (model, optimizer, params, input, grad, state, lr, beta1, beta2, eps,
//  weight_decay, alpha, lamb, gamma, gate, d_factor, bc1, bc2, neg_lr_scale,
//  decay_factor, beta, alpha_max, step, opt_only).
void fused_step(const std::string& model, const std::string& optimizer,
 torch::Tensor params, torch::Tensor input,
 torch::Tensor grad, torch::Tensor state, float lr,
 // FULL scalar set (C2-gap fix). apply_optimizer<> reads every one of
 // these; before this widening only `lr` was bound and the rest sat at
 // FusedOptState defaults (bc1/bc2/gate/d_factor == 1.0 → NO bias
 // correction, SG gating + Prodigy d-adaptation inert). bc1/bc2 are
 // un-inverted (= 1 - beta^step); the caller computes them from the shared
 // step counter (valid because all race params step together — the
 // megakernel treats the flat param as one task with one bc pair).
 float beta1, float beta2, float eps,
 float weight_decay, float alpha, float lamb,
 float gamma, float gate, float d_factor,
 float bc1, float bc2,
 float neg_lr_scale, float decay_factor,
 float beta, float alpha_max,
 int64_t step,
 // Tier selector. opt_only=true → L1: fused optimizer TAIL only; the kernel
 // reads the REAL grad from `grad` (framework-computed) and applies the
 // canonical update — the numerically-faithful path the race uses.
 // opt_only=false → L3: the kernel also runs the SURROGATE element-local
 // model fwd/bwd (model_stages.*), whose loss does NOT match the real
 // Transformer/ViT/Mamba graph (see BUILD_AND_VALIDATE.md).
 bool opt_only,
 // GEMM-engine selector for the L3-REAL path (owner directive task 1, the
 // GEMM_IMPL=wgmma wiring). "scalar" (default) → the shipped fp32
 // owner-computes megakernel (mega_*_real_adamw.cu). "wgmma" → the bf16
 // tensor-core cell (mega_*_real_adamw_tc_launcher.cu, HGMMA in SASS). The
 // Python caller (dispatch.fused_train_step) picks it per (model, precision):
 // decoder/vit at bf16 → "wgmma"; mamba (and every fp32 race) → "scalar".
 // IGNORED on the L1 tail (opt_only=true) and by the surrogate cells — only the
 // three L3-REAL blocks below read it. A defaulted arg keeps the pybind ABI
 // back-compatible: an un-rebuilt _ops still accepts the old call shape, and a
 // stale _ops that predates this arg trips the caller's one-shot TypeError latch
 // (→ loud degrade to eager, never silent).
 const std::string& gemm_impl,
 // GrokAdamW GLOBAL grad-norm clip threshold (decoder L3-TC, mechanism (ii)).
 // Trailing defaulted arg (same back-compat pattern as gemm_impl): ≤0 ⇒ no clip
 // (the inert default for every non-GrokAdamW cell). Flows into FusedScalars
 // below → the kernel's P2.5 global-norm reduction. A stale _ops without this
 // arg trips the caller's TypeError latch (loud degrade, never silent).
 float grad_clip,
 // Prodigy estimator scalars (decoder L3-TC, STAGED global-d). Trailing defaulted
 // args (same back-compat pattern): eager/inert defaults (d_coef=1, beta3=0) for
 // every non-Prodigy cell. Flow into FusedScalars → the kernel's P2.6 d-reduction.
 float d0, float d_coef, float beta3,
 // Muon 1D-group AdamW hyperparameters (ViT L3-TC, STAGED NS). Trailing defaulted
 // args (same back-compat pattern): eager Muon adamw_* defaults for every non-Muon
 // cell. Flow into FusedScalars → the kernel's Muon P3 1D tail (which device-computes
 // aux_bc1/bc2 = 1-aux_beta^step). The default args live on the helpers.h declaration.
 float aux_lr, float aux_beta1, float aux_beta2,
 // LookSAM scalars (decoder/vit/mamba L3-TC, MODEL-COUPLED SAM 2nd backward).
 // Trailing defaulted args (same back-compat pattern): inert defaults (0.0) for every
 // non-LookSAM cell. rho = SAM perturbation radius; looksam_sam = the every-k SAM-step
 // gate (1.0 ⇒ run the in-kernel perturb→2nd fwd+bwd→sam_dir=g_sam−g phase). Flow into
 // FusedScalars → the kernel's P2.4 SAM 2nd-backward phase. A stale _ops without these
 // args trips the caller's TypeError latch (loud degrade to eager, never silent).
 float rho, float looksam_sam,
 // SuperGrok11/15 scalars (decoder/vit L3-TC, meta-net mu). Trailing defaulted arg
 // (same back-compat pattern): inert default (0.0) for every non-SG cell. sg_rescale =
 // the SharpnessMetaNet rescale (mu = rescale·phi(g, sharpness)). The phi weights +
 // sharpness buffer are carried in the STATE buffer (the cell scatters/binds them);
 // only this scalar flows through the ABI. Flows into FusedScalars → the kernel's P2.45
 // meta-net mu precompute. SG also reuses `rho` (its SAM perturbation radius) + `alpha`
 // (the meta-net strength). A stale _ops without this arg trips the TypeError latch.
 float sg_rescale,
 // SuperGrok11 cosine-gate temperature (decoder/vit/mamba L3-TC). gate =
 // sigmoid(gate_temp · cos(grad, mu)) in the P2.45 finalizer (sg11_finalize_gate).
 // Trailing defaulted arg (back-compat); inert default 1.0 for every non-SG11 cell.
 float gate_temp) {
 int arch = detect_arch();
 // Resolve the GEMM engine ONCE. Unknown tokens FAIL LOUD (no silent scalar
 // fallback — the owner no-suppression rule): a typo'd "wgma" must not quietly
 // run scalar under a TC-requested run and corrupt the roofline fractions.
 const bool want_wgmma = (gemm_impl == "wgmma");
 TORCH_CHECK(gemm_impl == "scalar" || gemm_impl == "wgmma",
 "fused_step: unknown gemm_impl '", gemm_impl,
 "' (expected 'scalar' or 'wgmma').");
 std::string arch_str = (arch == 90) ? "sm_90"
 : (arch == 942) ? "gfx942" : "tpu_v6e";
 // `gamma` is NOW LIVE for the decoder L3-TC GrokAdamW cell: it is the
 // layer-wise β1 decay rate (β1_i = β1·(1-γ)^layer), consumed in the kernel's
 // P3 tail (fused_decoder_megakernel_tc, if constexpr GrokAdamW). For every
 // OTHER cell it stays inert — γ defaults to 0.0 (a single global β1), and no
 // other apply_optimizer<> branch reads st.gamma. (SG15's gamma_alpha still
 // arrives pre-baked into `alpha`, untouched.) `grad_clip` is likewise live
 // ONLY for the decoder GrokAdamW global grad-norm clip (P2.5); ≤0 ⇒ no clip.
 // Both flow to the kernel via the FusedScalars POD below → apply_scalars.

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
 // ── PHASE 1: the TRUE L3 fused decoder megakernel (real fwd+bwd+adamw). ──
 // Fires for (transformer_decoder, adamw, !opt_only): ONE persistent kernel,
 // no surrogate, no intermediate launches. The token path is carried through
 // the EXISTING fused_step arity (the pybind py::arg list pins it) by reading
 // tokens/targets from the int32-reinterpreted `input` tensor:
 //   input = int32 [B*(kSeq+1)]  (first B*kSeq tokens row-major, last B targets)
 // and `state` = float [3*total + 1] ([m|v|extra] + a trailing loss slot the
 // kernel writes the mean loss into, which the Python wrapper reads back). The
 // workspace (nCTA*total grad partials + nCTA loss partials) is device scratch
 // allocated here — it never crosses the ABI. S/vocab/d are compile-time
 // (kSeq/kVocab/kD); B is derived from input.numel(). This leaves the 33
 // generated surrogate cells untouched (their surrogate-L3 path is now
 // unreachable dead code; see BUILD_AND_VALIDATE.md PHASE-1).
 // OPTID-GENERIC GATE (owner baseline directive — all 33 cells on L3-TC): the
 // decoder real fwd+bwd+opt megakernel fires for adamw (scalar OR wgmma) AND, on
 // the wgmma path only, for the single-launch optimizer tails (lion/grokfast/
 // grokadamw/neuralgrok). The scalar real decoder kernel exists ONLY for adamw, so
 // a non-adamw scalar request is rejected inside (no silent adamw fallback). The
 // STAGED/coupled optimizers never reach here (wgmma_tail_opt_id < 0 → eager).
 const int dec_opt_id = wgmma_tail_opt_id(optimizer);
 // SuperGrok2 routes through the DEDICATED ops.sg2_fused_step entry (it needs the
 // meta-net weight bundle + per-tensor scalar arrays this generic block cannot carry),
 // so EXCLUDE it here even though wgmma_tail_opt_id("supergrok2")==10.
 const bool dec_l3_real = ((optimizer == "adamw")
                          || (want_wgmma && dec_opt_id >= 0))
                          && optimizer != "supergrok2";
 if (arch == 90 && model == "transformer_decoder" && dec_l3_real && !opt_only) {
 const int64_t total = kDecoderTotalElems;
 TORCH_CHECK(params.numel() == total,
 "fused decoder megakernel: params has ", params.numel(),
 " elements but the small decoder has ", total,
 ". Pass the flat concat of named_parameters() in order.");
 TORCH_CHECK(params.scalar_type() == torch::kFloat32 && params.is_contiguous(),
 "fused decoder megakernel: params must be contiguous fp32.");
 // input = int32 tokens(B*kSeq) ++ targets(B). B = numel/(kSeq+1).
 TORCH_CHECK(input.scalar_type() == torch::kInt32 && input.is_contiguous(),
 "fused decoder megakernel: input must be contiguous int32 "
 "(packed tokens[B*kSeq]++targets[B]).");
 const int64_t in_n = input.numel();
 TORCH_CHECK(in_n % (kDecoderSeq + 1) == 0 && in_n > 0,
 "fused decoder megakernel: input.numel() (", in_n,
 ") must be B*(kSeq+1) = B*", kDecoderSeq + 1, ".");
 const int B = (int)(in_n / (kDecoderSeq + 1));
 // state = [m|v|extra] (3*total) + 1 loss slot. PRODIGY (STAGED global-d) needs a
 // LARGER buffer: [m|v|extra/s_track | loss | param_init(total) | r_ema|s_ema|d_lr]
 // = 4*total + 4 (the TC launcher carves param_init at loss_slot+1, the 3 persisted
 // estimator scalars after it). SuperGrok11/15 (meta-net mu + SAM 2nd backward) need
 // [m|v|mu | loss | sharpness(total) | phi_pack(4*32+1)] = 4*total + 1 + kSgPhiPack
 // (the TC launcher carves sharpness at loss_slot+1 and the phi pack after it). The
 // host (fused_train_step) sizes it per-cell.
 const bool is_sg = (optimizer == "supergrok11" || optimizer == "supergrok15");
 const int64_t min_state = (optimizer == "prodigy") ? (4 * total + 4)
                         : is_sg ? (4 * total + 1 + (int64_t)(4 * 32 + 1))
                         : (3 * total + 1);
 TORCH_CHECK(state.numel() >= min_state &&
 state.scalar_type() == torch::kFloat32 && state.is_contiguous(),
 "fused decoder megakernel: state must be contiguous fp32 with >= ", min_state,
 " elements (", (optimizer == "prodigy")
 ? "[m|v|s_track|loss|param_init|r_ema|s_ema|d_lr] for Prodigy"
 : is_sg ? "[m|v|mu|loss|sharpness|phi_pack] for SuperGrok11/15"
 : "[m|v|extra]+loss", ").");
 // grad = the REDUCED weight-grad OUTPUT [total]: the kernel writes the
 // deterministically-reduced grad here (P2) and consumes it in the optimizer
 // tail (P3), which does NOT overwrite it — so after the call this buffer holds
 // exactly the grad the AdamW step used. Routing it through the ABI `grad`
 // tensor (instead of an internal scratch) EXPOSES the kernel's grads to the
 // caller so the parity test can compare every weight grad per-tensor against
 // the oracle (the keystone validation of the hand-written backward).
 TORCH_CHECK(grad.numel() == total &&
 grad.scalar_type() == torch::kFloat32 && grad.is_contiguous(),
 "fused decoder megakernel: grad must be contiguous fp32 with total = ",
 total, " elements (the reduced weight-grad output buffer).");

 DecoderScratch& dsc = decoder_scratch_for(params);
 // PersistentContext built from the mirror struct (same layout/FQN as the .cu).
 // n_tasks = #parameter tensors (30) for the reduce + optimizer phases — the
 // kernel reads the per-tensor numel/offset from its __constant__ tables.
 fused::PersistentContext ctx{
 dsc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(dsc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(dsc.g_generation.data_ptr<int>()),
 30,                    // kDecNumTensors (mirror; the .cu static-asserts it)
 0u};                   // n_ctas overwritten by the launcher (one CTA/SM)

 float* m = state.data_ptr<float>();
 float* loss_slot = m + 3 * total;          // the trailing loss slot

 // FULL scalar set (un-frozen bc1/bc2/...). Field order MUST match the mirror
 // fused::sm90::FusedScalars (== the real struct the .cu reads positionally).
 // gamma/grad_clip appended (decoder GrokAdamW): inert (0.0) for all other cells.
 fused::sm90::FusedScalars scalars{
 lr, beta1, beta2, eps, weight_decay, bc1, bc2, alpha, beta, lamb,
 alpha_max, gate, d_factor, neg_lr_scale, decay_factor, gamma, grad_clip,
 d0, d_coef, beta3,    // Prodigy estimator scalars (inert for non-Prodigy cells)
 aux_lr, aux_beta1, aux_beta2,   // Muon 1D-group AdamW hyperparams (decoder L3-TC
 rho, looksam_sam,     // LookSAM SAM 2nd-backward scalars (inert for non-LookSAM cells)
 sg_rescale,           // SuperGrok11/15 meta-net rescale (inert for non-SG cells)
 gate_temp};           // SuperGrok11 cosine-gate temperature (sigmoid(gate_temp·cos))
 // STAGED NS). MUST be passed here — the mirror FusedScalars (line 204) declares
 // aux_* with NO in-class default, so omitting them aggregate-inits to 0.0 (→ the
 // 1D AdamW tail would run β1=β2=0: m=g/v=g², the (1b) state FAIL this fixes).
 // Inert for every non-Muon decoder cell (only OptId::Muon's P3 reads aux_*).
 // Lock-step with the vit decoder-twin POD below.

 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
 // GEMM-engine branch (owner directive task 1). want_wgmma → the bf16 tensor-core
 // launcher (mega_decoder_real_adamw_tc_launcher.cu: HGMMA in SASS, its own
 // TC-sized workspace, so we pass nullptr for the scalar `workspace`); else the
 // shipped fp32 scalar launcher (mega_decoder_real_adamw.cu, scalar dsc.workspace).
 // Both run the REAL decoder fwd+bwd+AdamW as ONE persistent kernel. NO silent
 // fallback: a wgmma request runs wgmma or the launcher returns a cuda error that
 // throws below (the no-suppression rule — a TC-requested run never secretly runs
 // scalar). All <<<>>>/__global__/device code lives in the launcher TUs; this host
 // TU passes only plain pointers + the FusedScalars POD.
 cudaError_t err;
 if (want_wgmma) {
 // opt_id selects the in-kernel tail (apply_optimizer<Opt>); the fwd+bwd is
 // optimizer-independent. dec_opt_id is >=0 here (the gate required it on the
 // wgmma path), so adamw/lion/grokfast/grokadamw/neuralgrok route to their own
 // tail instantiation; an unsupported id returns cudaErrorInvalidValue → throw.
 err = fused::sm90::mega_decoder_real_adamw_tc(
 ctx, params.data_ptr<float>(),
 input.data_ptr<int>(),                              // tokens
 input.data_ptr<int>() + (int64_t)B * kDecoderSeq,   // targets
 B, m, grad.data_ptr<float>(),    // reduced-grad OUTPUT (exposed to caller)
 /*workspace=*/nullptr, loss_slot,  // TC launcher owns its own TC-sized scratch
 /*sizes=*/nullptr, /*offsets=*/nullptr,
 lr, static_cast<int>(step), scalars, stream, /*ncta_cap=*/0, dec_opt_id);
 } else {
 // The SCALAR real decoder kernel exists only for adamw. A non-adamw scalar
 // request has no real fwd+bwd+opt scalar TU → FAIL LOUD (no adamw fallback).
 TORCH_CHECK(optimizer == "adamw",
 "fused decoder megakernel: the scalar (fp32) real fwd+bwd+opt path is "
 "wired for adamw only; optimizer '", optimizer, "' has a real L3-TC path "
 "via gemm_impl='wgmma' (single-launch tail) but no scalar one. Use bf16.");
 err = fused::sm90::mega_decoder_real_adamw(
 ctx, params.data_ptr<float>(),
 input.data_ptr<int>(),                              // tokens
 input.data_ptr<int>() + (int64_t)B * kDecoderSeq,   // targets
 B, m, grad.data_ptr<float>(),    // reduced-grad OUTPUT (exposed to caller)
 dsc.workspace.data_ptr<float>(), loss_slot,
 /*sizes=*/nullptr, /*offsets=*/nullptr,
 lr, static_cast<int>(step), scalars, stream);
 }
 if (err != cudaSuccess)
 throw std::runtime_error(
 std::string(want_wgmma ? "fused decoder TC (wgmma) megakernel launch failed: "
 : "fused decoder megakernel launch failed: ") +
 cudaGetErrorString(err));
 return;
 }

 // ── PHASE 2: the TRUE L3 fused ViT megakernel (real fwd+bwd+adamw). ──
 // Fires for (vit, adamw, !opt_only): ONE persistent kernel, no surrogate, no
 // intermediate launches. UNLIKE the decoder/mamba (int tokens), ViT's input is
 // FLOAT image patches with int targets BIT-REINTERPRETED into trailing float
 // slots (fused_step's arity has one input tensor; the symmetric move to the
 // decoder's int pack):
 //   input = float32 [B*16*49 + B]  (B*784 patch pixels row-major [B][16][49],
 //                                   then B targets bit-cast to float)
 // dispatch reads patches as input.data_ptr<float>() and targets as
 // reinterpret_cast<const int*>(input.data_ptr<float>() + B*784) — a BIT
 // REINTERPRET (NOT a value cast), lossless for ALL int32. `state` = float
 // [3*total + 1] ([m|v|extra] + a trailing loss slot the kernel writes the mean
 // loss into). The workspace is device scratch (never crosses the ABI). Placed
 // BEFORE the surrogate route so the real path wins.
 // OPTID-GENERIC GATE (owner baseline directive — twin of the decoder gate): the
 // vit real fwd+bwd+opt megakernel fires for adamw (scalar OR wgmma) AND, on the
 // wgmma path only, for the single-launch tails (lion/grokfast/grokadamw/neuralgrok).
 const int vit_opt_id = wgmma_tail_opt_id(optimizer);
 // SuperGrok2 routes through the DEDICATED ops.sg2_fused_step entry (see the decoder
 // block) — EXCLUDE it from the generic vit block.
 const bool vit_l3_real = ((optimizer == "adamw")
                          || (want_wgmma && vit_opt_id >= 0))
                          && optimizer != "supergrok2";
 if (arch == 90 && model == "vit" && vit_l3_real && !opt_only) {
 const int64_t total = kVitTotalElems;
 TORCH_CHECK(params.numel() == total,
 "fused ViT megakernel: params has ", params.numel(),
 " elements but the small ViT has ", total,
 ". Pass the flat concat of named_parameters() in order.");
 TORCH_CHECK(params.scalar_type() == torch::kFloat32 && params.is_contiguous(),
 "fused ViT megakernel: params must be contiguous fp32.");
 // input = float32 patches(B*784) ++ int-target-bits(B). B = numel/(784+1).
 TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.is_contiguous(),
 "fused ViT megakernel: input must be contiguous fp32 (packed patch "
 "pixels[B*16*49] ++ int-target-bits[B] bit-reinterpreted into floats).");
 const int64_t in_n = input.numel();
 TORCH_CHECK(in_n % (kVitPatchElems + 1) == 0 && in_n > 0,
 "fused ViT megakernel: input.numel() (", in_n,
 ") must be B*(16*49+1) = B*", kVitPatchElems + 1, ".");
 const int B = (int)(in_n / (kVitPatchElems + 1));
 // state = [m|v|extra] (3*total) + 1 loss slot. PRODIGY needs 4*total+4; SuperGrok11/15
 // (meta-net mu + SAM 2nd backward) need [m|v|mu|loss|sharpness(total)|phi_pack(4*32+1)]
 // = 4*total + 1 + kSgPhiPack. The host (fused_train_step) sizes it per-cell.
 const bool vit_is_sg = (optimizer == "supergrok11" || optimizer == "supergrok15");
 const int64_t vit_min_state = (optimizer == "prodigy") ? (4 * total + 4)
                             : vit_is_sg ? (4 * total + 1 + (int64_t)(4 * 32 + 1))
                             : (3 * total + 1);
 TORCH_CHECK(state.numel() >= vit_min_state &&
 state.scalar_type() == torch::kFloat32 && state.is_contiguous(),
 "fused ViT megakernel: state must be contiguous fp32 with "
 ">= ", vit_min_state, " elements (", vit_is_sg
 ? "[m|v|mu|loss|sharpness|phi_pack] for SuperGrok11/15"
 : "[m|v|extra]+loss", ").");
 // grad = the REDUCED weight-grad OUTPUT [total] (same contract as the
 // decoder): the kernel writes the deterministically-reduced grad here (P2) and
 // consumes it in the optimizer tail (P3) WITHOUT overwriting it, so after the
 // call this buffer holds exactly the grad the AdamW step used — the parity test
 // slices it per-tensor against the oracle (the keystone check).
 TORCH_CHECK(grad.numel() == total &&
 grad.scalar_type() == torch::kFloat32 && grad.is_contiguous(),
 "fused ViT megakernel: grad must be contiguous fp32 with total = ",
 total, " elements (the reduced weight-grad output buffer).");

 ViTScratch& vsc = vit_scratch_for(params);
 fused::PersistentContext ctx{
 vsc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(vsc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(vsc.g_generation.data_ptr<int>()),
 32,                    // kVitNumTensors (mirror; the .cu static-asserts it)
 0u};                   // n_ctas overwritten by the launcher (one CTA/SM)

 float* m = state.data_ptr<float>();
 float* loss_slot = m + 3 * total;          // the trailing loss slot

 fused::sm90::FusedScalars scalars{
 lr, beta1, beta2, eps, weight_decay, bc1, bc2, alpha, beta, lamb,
 alpha_max, gate, d_factor, neg_lr_scale, decay_factor, gamma, grad_clip,
 d0, d_coef, beta3,    // Prodigy estimator scalars (inert for non-Prodigy cells)
 aux_lr, aux_beta1, aux_beta2,   // Muon 1D-group AdamW hyperparams (inert for non-Muon)
 rho, looksam_sam,     // LookSAM SAM 2nd-backward scalars (inert for non-LookSAM cells)
 sg_rescale,           // SuperGrok11/15 meta-net rescale (inert for non-SG cells)
 gate_temp};           // SuperGrok11 cosine-gate temperature (sigmoid(gate_temp·cos))

 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
 // GEMM-engine branch (task 1). want_wgmma → bf16 tensor-core launcher
 // (mega_vit_real_adamw_tc_launcher.cu, its own TC-sized scratch → nullptr
 // workspace); else the fp32 scalar launcher. ViT input is FLOAT patches with the
 // int targets bit-reinterpreted out of the trailing float slots (same pointer
 // arithmetic both ways). NO silent fallback (no-suppression rule).
 const float* vit_patches = input.data_ptr<float>();
 const int* vit_targets = reinterpret_cast<const int*>(
 input.data_ptr<float>() + (int64_t)B * kVitPatchElems);
 cudaError_t err;
 if (want_wgmma) {
 // opt_id selects the in-kernel tail; vit_opt_id is >=0 here (the gate required
 // it on the wgmma path). adamw/lion/grokfast/grokadamw/neuralgrok route to their
 // own tail; an unsupported id returns cudaErrorInvalidValue → throw below.
 err = fused::sm90::mega_vit_real_adamw_tc(
 ctx, params.data_ptr<float>(),
 vit_patches, vit_targets,
 B, m, grad.data_ptr<float>(),
 /*workspace=*/nullptr, loss_slot,  // TC launcher owns its own TC-sized scratch
 lr, static_cast<int>(step), scalars, stream, /*ncta_cap=*/0, vit_opt_id);
 } else {
 // The SCALAR real vit kernel exists only for adamw — fail loud otherwise.
 TORCH_CHECK(optimizer == "adamw",
 "fused ViT megakernel: the scalar (fp32) real fwd+bwd+opt path is wired "
 "for adamw only; optimizer '", optimizer, "' has a real L3-TC path via "
 "gemm_impl='wgmma' (single-launch tail) but no scalar one. Use bf16.");
 err = fused::sm90::mega_vit_real_adamw(
 ctx, params.data_ptr<float>(),
 vit_patches, vit_targets,
 B, m, grad.data_ptr<float>(),    // reduced-grad OUTPUT (exposed to caller)
 vsc.workspace.data_ptr<float>(), loss_slot,
 lr, static_cast<int>(step), scalars, stream);
 }
 if (err != cudaSuccess)
 throw std::runtime_error(
 std::string(want_wgmma ? "fused ViT TC (wgmma) megakernel launch failed: "
 : "fused ViT megakernel launch failed: ") +
 cudaGetErrorString(err));
 return;
 }

 // ── PHASE 2: the TRUE L3 fused Mamba megakernel (real fwd+bwd+opt). ──
 // Fires for (mamba3, !opt_only) on the scalar path for adamw, AND on the wgmma
 // path for the single-launch tails {adamw,lion,grokfast} (cycle-2 directive (c)):
 // ONE persistent kernel, no surrogate, no intermediate launches. CANONICAL model
 // name is "mamba3" (NOT "mamba" — the Python wrapper canonicalizes before calling,
 // and the wired-cell table emits mamba3; matching "mamba" here would never fire and
 // fall through to the surrogate path). Mamba's input is int32 tokens like the
 // decoder (NOT ViT's float pack):
 //   input = int32 [B*(8+1)]  (tokens[B*8] row-major [B][8], then targets[B])
 // `state` = float [3*total + 1] ([m|v|extra] + a trailing loss slot). The
 // workspace is device scratch (never crosses the ABI). Placed BEFORE the
 // surrogate route so the real path wins.
 // OPTID-GENERIC GATE (mirrors the decoder): mamba's scalar real megakernel exists
 // ONLY for adamw; the wgmma TC launcher production-routes the single-launch tails
 // {adamw,lion,grokfast,grokadamw,neuralgrok,muon} (opt_id 0/1/2/3/6/7) PLUS the now-
 // A/A/A-FIXED Prodigy (opt_id 5, in-kernel P2.6 d-estimate) and MODEL-COUPLED LookSAM
 // (opt_id 4, in-kernel P2.4 SAM 2nd backward). mb_tc_tail is the single source of the
 // routed set, so this gate cannot drift from the launcher's switch.
 const int mb_opt_id = wgmma_tail_opt_id(optimizer);
 // mamba×prodigy + mamba×looksam: the A/A/A determinism race that previously BLOCKED these
 // (non-deterministic grad, intermittent NaN, ~1e-3 loss/grad drift across bit-identical
 // re-runs on the production path) is FIXED and the carve-out is LIFTED. ROOT CAUSE: a
 // register-pressure-induced wgmma-accumulator spill in the mamba TC backward — the
 // selective-scan backward (mb_scan_bwd, model_stage_mamba3.cuh) cached a large per-thread
 // frame (dB_loc/dC_loc[kSeq][kState] + a_save[kSeq][kState], ~2 KB/thread) that spilled to
 // local memory; combined with the EXTRA register pressure of prodigy's P2.6 state OR
 // LookSAM's P2.4 SAM block (which inlines the heavy fwd+bwd a SECOND time), the whole-kernel
 // frame crossed the spill threshold and ptxas ALSO spilled the 64-register projection-GEMM
 // accumulator inside the wgmma double-buffer issue→stage→wait window (the ptxas C7515
 // hazard: "non-wgmma instructions defining accumulator registers between start and end of
 // the pipeline stage"). A spilled fp32 accumulator reloaded while wgmma.mma_async is
 // concurrently writing it is NON-DETERMINISTIC → denormal/NaN grad drift. The basic single-
 // pass tails (adamw/lion/grokfast/grokadamw/neuralgrok) + muon (single fwd + NS precompute)
 // stayed UNDER the threshold, which is why ONLY they were deterministic. FIX (no suppression
 // — toward canonical, deterministic by construction): (1) fused per-timestep dB/dC block-
 // reduce in mb_scan_bwd (drops the [kSeq][kState] arrays — same addends, same ascending
 // order); (2) dropped a_save (recompute adec=exp(dt·A) in the reverse loop, bit-identical);
 // (3) __noinline__ on mbtc_forward_tile/mbtc_backward_tile (model_stage_mamba_tc.cuh) so the
 // heavy model frame lives in ONE out-of-line frame instead of being inlined (TWICE for the
 // SAM cells) into the megakernel — the megakernel's own frame stays small and the GEMM
 // accumulator is register-resident for EVERY instantiation. mamba×prodigy AND mamba×looksam
 // now PASS test_l3tc_tail_gate A/A/A bit-exact on the production fused_train_step path
 // (loss/grad/param maxd=0 ×4; looksam on a SAM step AND a SAM-off step). SG_MAMBA_PRODIGY_
 // PROBE / SG_MAMBA_LOOKSAM_PROBE diagnostic overrides are retired (the cells route in
 // production now). The 6 basic mamba cells stay byte-identical (the scan math is unchanged;
 // __noinline__ does not alter results — re-gated green).
 // SuperGrok11/15 are now CONVERTED — the FEATURE PORT landed (the mamba megakernel SAM
 // block extends to SG11/15/SG2: sharpness=(g_sam−g)² in P2.4 + the per-tensor meta-net mu
 // precompute in P2.45/P3; the launcher switch gained the OptId::SuperGrok11/15 cases). The
 // now-deterministic LookSAM mamba instantiation proved the SAM 2nd-pass path is race-free,
 // so SG11/15 ride the SAME single-launch TC tail (opt_id 8/9). The gate re-verifies A/A/A
 // for each; if either trips the race it is landed DORMANT (gated out of _L3_WGMMA_CELLS).
 // SuperGrok2 stays OUT of the generic single-launch tail — it routes via the DEDICATED
 // sg2_fused_step → mega_mamba_sg2_tc entry (the FULL meta-net weight bundle + per-tensor
 // scalar ARRAYS don't fit the generic FusedScalars/boundary). So mb_is_sg gates SG2 only.
 const bool mb_is_sg2 = (optimizer == "supergrok2");
 const bool mb_tc_tail = (mb_opt_id >= 0 && !mb_is_sg2);  // {adamw,lion,grokfast,grokadamw,neuralgrok,muon,prodigy,looksam,supergrok11,supergrok15}; SG2 = dedicated entry (sg2_fused_step).
 const bool mamba_l3_real = (optimizer == "adamw")
                            || (want_wgmma && mb_tc_tail);
 if (arch == 90 && model == "mamba3" && mamba_l3_real && !opt_only) {
 // A wgmma request for a NON-TC-tail mamba optimizer (the SAM/SG2 set) is a loud
 // error rather than a silent scalar run under a wgmma label (no-suppression). The
 // scalar real kernel covers only adamw, so a non-adamw scalar request is also
 // rejected (no silent adamw fallback).
 TORCH_CHECK(want_wgmma || optimizer == "adamw",
 "fused_step: mamba3 scalar L3 path exists ONLY for adamw (got '", optimizer,
 "'); the non-adamw mamba tails are wgmma-only. Use gemm_impl='wgmma'.");
 TORCH_CHECK(!want_wgmma || mb_tc_tail,
 "fused_step: gemm_impl='wgmma' requested for mamba3 with optimizer '", optimizer,
 "', but the mamba TC launcher generic single-launch tail production-routes "
 "{adamw,lion,grokfast,grokadamw,neuralgrok,muon,prodigy,looksam,supergrok11,supergrok15} "
 "(opt_id 0/1/2/3/6/7/5/4/8/9). SuperGrok2 routes via the DEDICATED sg2_fused_step entry "
 "(the FULL meta-net weight bundle + per-tensor scalar arrays don't fit fused_step's ABI), "
 "NOT this generic path — call ops.sg2_fused_step (dispatch.fused_train_step does so).");
 const int64_t total = kMambaTotalElems;
 TORCH_CHECK(params.numel() == total,
 "fused Mamba megakernel: params has ", params.numel(),
 " elements but the small Mamba has ", total,
 ". Pass the flat concat of named_parameters() in order.");
 TORCH_CHECK(params.scalar_type() == torch::kFloat32 && params.is_contiguous(),
 "fused Mamba megakernel: params must be contiguous fp32.");
 // input = int32 tokens(B*kSeq) ++ targets(B). B = numel/(kSeq+1).
 TORCH_CHECK(input.scalar_type() == torch::kInt32 && input.is_contiguous(),
 "fused Mamba megakernel: input must be contiguous int32 "
 "(packed tokens[B*kSeq]++targets[B]).");
 const int64_t in_n = input.numel();
 TORCH_CHECK(in_n % (kMambaSeq + 1) == 0 && in_n > 0,
 "fused Mamba megakernel: input.numel() (", in_n,
 ") must be B*(kSeq+1) = B*", kMambaSeq + 1, ".");
 const int B = (int)(in_n / (kMambaSeq + 1));
 // state = [m|v|extra] (3*total) + 1 loss slot. PRODIGY (STAGED global-d) needs a
 // LARGER buffer: [m|v|s_track | loss | param_init(total) | r_ema|s_ema|d_lr] =
 // 4*total + 4 (the TC launcher carves param_init at loss_slot+1, the 3 persisted
 // estimator scalars after it — identical to the decoder/vit prodigy state). SuperGrok11/15
 // need [m|v|mu | loss | sharpness(total) | phi_pack(4H+1)] = 4*total + 1 + (4*32+1) (the
 // launcher carves sharpness at loss_slot+1, the phi pack after it — identical to the
 // decoder/vit SG state). The host (fused_train_step) sizes it per-cell (model-agnostic).
 const int64_t min_state =
     (optimizer == "prodigy") ? (4 * total + 4)
   : (optimizer == "supergrok11" || optimizer == "supergrok15") ? (4 * total + 1 + (4 * 32 + 1))
   : (3 * total + 1);
 TORCH_CHECK(state.numel() >= min_state &&
 state.scalar_type() == torch::kFloat32 && state.is_contiguous(),
 "fused Mamba megakernel: state must be contiguous fp32 with >= ", min_state,
 " elements (", (optimizer == "prodigy")
 ? "[m|v|s_track|loss|param_init|r_ema|s_ema|d_lr] for Prodigy"
 : (optimizer == "supergrok11" || optimizer == "supergrok15")
 ? "[m|v|mu|loss|sharpness|phi_pack] for SuperGrok11/15"
 : "[m|v|extra]+loss", ").");
 // grad = the REDUCED weight-grad OUTPUT [total] (same contract as the
 // decoder): the only check that exercises the hand-written selective-scan
 // backward's magnitudes per-tensor against the oracle (the keystone).
 TORCH_CHECK(grad.numel() == total &&
 grad.scalar_type() == torch::kFloat32 && grad.is_contiguous(),
 "fused Mamba megakernel: grad must be contiguous fp32 with total = ",
 total, " elements (the reduced weight-grad output buffer).");

 MambaScratch& msc = mamba_scratch_for(params);
 fused::PersistentContext ctx{
 msc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(msc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(msc.g_generation.data_ptr<int>()),
 28,                    // kMambaNumTensors (mirror; the .cu static-asserts it)
 0u};                   // n_ctas overwritten by the launcher (one CTA/SM)

 float* m = state.data_ptr<float>();
 float* loss_slot = m + 3 * total;          // the trailing loss slot

 fused::sm90::FusedScalars scalars{
 lr, beta1, beta2, eps, weight_decay, bc1, bc2, alpha, beta, lamb,
 alpha_max, gate, d_factor, neg_lr_scale, decay_factor, gamma, grad_clip,
 d0, d_coef, beta3,    // Prodigy estimator scalars (inert for non-Prodigy cells)
 aux_lr, aux_beta1, aux_beta2,   // Muon 1D-group AdamW hyperparams (inert for non-Muon)
 rho, looksam_sam,     // LookSAM SAM 2nd-backward scalars (inert for non-LookSAM cells).
 sg_rescale,           // SuperGrok11/15 meta-net rescale (live for the mamba SG cells).
 gate_temp};           // SuperGrok11 cosine-gate temperature (sigmoid(gate_temp·cos))
 // MUST be passed (the mirror FusedScalars declares all 26 fields with NO in-class
 // defaults — omitting these aggregate-inits them to 0.0, which for LookSAM means
 // rho=0/looksam_sam=0 → the P2.4 SAM phase is INERT and sam_dir never gets written).
 // Lock-step with the decoder/vit PODs above.

 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
 // GEMM-engine branch (mirrors the decoder/vit). want_wgmma → the bf16 tensor-core
 // launcher (mega_mamba_real_adamw_tc_launcher.cu: HGMMA in SASS for the 4 projection
 // GEMMs, scan/conv scalar by design; owns its own TC-sized workspace, so we pass
 // nullptr for the scalar `workspace`); else the shipped fp32 scalar launcher
 // (mega_mamba_real_adamw.cu, scalar msc.workspace). Both run the REAL mamba fwd+bwd
 // +opt as ONE persistent kernel. NO silent fallback: a wgmma request runs wgmma or
 // the launcher returns a cuda error that throws below (the no-suppression rule).
 cudaError_t err;
 if (want_wgmma) {
 // opt_id (mb_opt_id ∈ {0,1,2} — the gate required it on the wgmma path) selects the
 // in-kernel tail (apply_optimizer<Opt>); the fwd+bwd is optimizer-independent.
 err = fused::sm90::mega_mamba_real_adamw_tc(
 ctx, params.data_ptr<float>(),
 input.data_ptr<int>(),                              // tokens
 input.data_ptr<int>() + (int64_t)B * kMambaSeq,     // targets
 B, m, grad.data_ptr<float>(),    // reduced-grad OUTPUT (exposed to caller)
 /*workspace=*/nullptr, loss_slot,  // TC launcher owns its own TC-sized scratch
 lr, static_cast<int>(step), scalars, stream, /*ncta_cap=*/0, mb_opt_id);
 } else {
 err = fused::sm90::mega_mamba_real_adamw(
 ctx, params.data_ptr<float>(),
 input.data_ptr<int>(),                              // tokens
 input.data_ptr<int>() + (int64_t)B * kMambaSeq,     // targets
 B, m, grad.data_ptr<float>(),    // reduced-grad OUTPUT (exposed to caller)
 msc.workspace.data_ptr<float>(), loss_slot,
 lr, static_cast<int>(step), scalars, stream);
 }
 if (err != cudaSuccess)
 throw std::runtime_error(
 std::string(want_wgmma ? "fused Mamba TC (wgmma) megakernel launch failed: "
 : "fused Mamba megakernel launch failed: ") +
 cudaGetErrorString(err));
 return;
 }

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
 // n_tasks = number of work tasks = #entries in sizes/offsets (one slab per
 // parameter tensor; a single flat tensor here → 1). It is NOT the element
 // count: the megakernel loops `t < n_tasks` reading sizes[t]/offsets[t], so
 // n_tasks=n would index both 1-element arrays ~n entries out of bounds and
 // feed garbage offsets into params[] (a multi-GB OOB read). n_ctas (0 here)
 // is overwritten by the launcher with one persistent CTA per SM.
 static_cast<int>(sc.sizes.numel()), 0u};

 // Use the current CUDA stream so the megakernel orders against the rest of
 // the model's work and is capturable in a CUDA graph (the legacy default
 // stream 0 serialized against all work and blocked graph capture).
 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

 // Pack the FULL scalar set for this cell. apply_optimizer<> reads every one
 // (the C2-gap fix); the cell's apply_scalars() copies them into FusedOptState.
 // Field order MUST match fused::sm90::FusedScalars (lr, beta1, beta2, eps, wd,
 // bc1, bc2, alpha, beta, lamb, alpha_max, gate, d_factor, neg_lr_scale,
 // decay_factor). bc1/bc2 are un-inverted (= 1 - beta^step) — ONE pair per call
 // is correct because the megakernel processes the flat param as a single task
 // and all race params share `step`.
 fused::sm90::FusedScalars scalars{
 lr, beta1, beta2, eps, weight_decay, bc1, bc2, alpha, beta, lamb,
 alpha_max, gate, d_factor, neg_lr_scale, decay_factor, gamma, grad_clip,
 d0, d_coef, beta3};   // Prodigy estimator scalars (inert for non-Prodigy cells)

 // Route to the real composed cell launcher (all 33 sm_90 cells). `found`
 // is set false only if no cell matches (then we fall through to the honest
 // not-compiled signal below). `opt_only` selects L1 (faithful real-grad tail)
 // vs L3 (surrogate-model fwd+bwd+opt).
 bool found = false;
 cudaError_t err = fused::sm90::dispatch_sm90_cell(
 model, optimizer, ctx, p, in, sc.acts.data_ptr<float>(), gr, m, v, extra,
 sc.sizes.data_ptr<int>(), sc.offsets.data_ptr<int>(), lr,
 static_cast<int>(step), scalars, opt_only, stream, &found);

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
 // n_tasks = TASK count (sizes/offsets entries), NOT the element count n —
 // the megakernel loops t < n_tasks reading sizes[t]/offsets[t]; passing n
 // here was the same multi-GB OOB the sm_90 path fixed (47d8007), left
 // unfixed on this HIP twin until now.
 static_cast<int>(sc.sizes.numel()), 0u};
 // Current HIP stream for graph capture / proper ordering (was default 0).
 hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
 // FULL scalar set (mirror of the sm_90 path); field order matches
 // fused::gfx942_mega::FusedScalars. bc1/bc2 un-inverted (= 1 - beta^step).
 fused::gfx942_mega::FusedScalars scalars{
 lr, beta1, beta2, eps, weight_decay, bc1, bc2, alpha, beta, lamb,
 alpha_max, gate, d_factor, neg_lr_scale, decay_factor};
 bool found = false;
 hipError_t herr = fused::gfx942_mega::dispatch_gfx942_cell(
 model, optimizer, ctx, p, in, sc.acts.data_ptr<float>(), gr, m, v, extra,
 sc.sizes.data_ptr<int>(), sc.offsets.data_ptr<int>(), lr,
 static_cast<int>(step), scalars, opt_only, stream, &found);
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

// ── sg2_fused_step — the DEDICATED SuperGrok2 L3-TC entry (decoder/vit). SG2's
//    optimizer phase is the FULL CSA/HCA/PEER/GRU meta-net (in-kernel segmented sort
//    + SAM 2nd backward → sharpness + sg2_meta_stages), which needs the meta-net
//    weight bundle (26 HBM ptrs) + per-tensor scalar arrays (6, length P) + the
//    meta-net state — none representable in fused_step's FusedScalars POD. So this is
//    a PARALLEL host entry (fused_step + the 28 byte-identical cells are UNTOUCHED).
//    The Python caller (dispatch.fused_train_step's supergrok2 branch) packs `input`
//    (tokens/patches) IDENTICALLY to fused_step and supplies the bundle + scalar
//    arrays. The reduced grad is written to `grad` (return_grad parity), the loss to
//    state[3*total]. wgmma-only (the SG2 meta-net rides the bf16 TC fwd+bwd). NO
//    silent fallback: an unsupported model/shape throws.
void sg2_fused_step(
 const std::string& model,
 torch::Tensor params, torch::Tensor input, torch::Tensor grad, torch::Tensor state,
 // SG2 meta-net weight bundle (fp32, contiguous), in SG2Weights order.
 torch::Tensor input_proj_W, torch::Tensor input_proj_b,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, torch::Tensor csa_out_W,
 torch::Tensor csa_compress_w, torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_K,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, torch::Tensor hca_out_W,
 torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, torch::Tensor gru_br,
 torch::Tensor gru_Wh, torch::Tensor gru_bh,
 torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
 torch::Tensor expert_W1, torch::Tensor expert_b1, torch::Tensor expert_W2, torch::Tensor expert_b2,
 // per-tensor scalar arrays (length P = #param tensors), named_parameters() order.
 torch::Tensor sc_alpha, torch::Tensor sc_gru_decay, torch::Tensor sc_lamb_eff,
 torch::Tensor sc_beta1, torch::Tensor sc_bc1, torch::Tensor sc_bc2,
 // shared scalars + SAM gate.
 double rescale, double beta2, double lr, double wd, double eps,
 double rho, double sam_on, int64_t step) {
#if defined(WITH_CUDA) && !defined(WITH_HIP)
 int arch = detect_arch();
 TORCH_CHECK(arch == 90, "sg2_fused_step: SuperGrok2 L3-TC is sm_90 only.");
 TORCH_CHECK(params.scalar_type() == torch::kFloat32 && params.is_contiguous(),
 "sg2_fused_step: params must be contiguous fp32 (flat named_parameters()).");
 TORCH_CHECK(state.scalar_type() == torch::kFloat32 && state.is_contiguous(),
 "sg2_fused_step: state must be contiguous fp32.");
 TORCH_CHECK(grad.scalar_type() == torch::kFloat32 && grad.is_contiguous(),
 "sg2_fused_step: grad must be contiguous fp32 (reduced weight-grad output).");
 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
 auto W = [](torch::Tensor& t) {
 TORCH_CHECK(t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
 "sg2_fused_step: meta-net weights + scalar arrays must be contiguous fp32.");
 return t.data_ptr<float>();
 };
 float* st = state.data_ptr<float>();
 cudaError_t err = cudaSuccess;
 if (model == "transformer_decoder") {
 const int64_t total = kDecoderTotalElems;
 const int GH = 4;   // SG2Dims gru_hidden default
 const int64_t min_state = (int64_t)(5 + GH) * total + 1;   // [m|v|mu|loss|sharpness|slow|gru_state]
 TORCH_CHECK(params.numel() == total,
 "sg2_fused_step: decoder params has ", params.numel(), " != ", total);
 TORCH_CHECK(grad.numel() == total,
 "sg2_fused_step: decoder grad has ", grad.numel(), " != ", total);
 TORCH_CHECK(state.numel() >= min_state,
 "sg2_fused_step: decoder state needs >= ", min_state,
 " floats ([m|v|mu|loss|sharpness|slow|gru_state(total*", GH, ")]) got ",
 state.numel(), ".");
 TORCH_CHECK(input.scalar_type() == torch::kInt32 && input.is_contiguous(),
 "sg2_fused_step: decoder input must be contiguous int32 (tokens[B*kSeq]++targets[B]).");
 const int64_t in_n = input.numel();
 TORCH_CHECK(in_n % (kDecoderSeq + 1) == 0 && in_n > 0,
 "sg2_fused_step: decoder input.numel() must be B*(kSeq+1).");
 const int B = (int)(in_n / (kDecoderSeq + 1));
 DecoderScratch& dsc = decoder_scratch_for(params);
 fused::PersistentContext ctx{
 dsc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(dsc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(dsc.g_generation.data_ptr<int>()),
 30, 0u};
 float* loss_slot = st + 3 * total;
 err = fused::sm90::mega_decoder_sg2_tc(
 ctx, params.data_ptr<float>(),
 input.data_ptr<int>(), input.data_ptr<int>() + (int64_t)B * kDecoderSeq, B,
 st, grad.data_ptr<float>(), loss_slot,
 W(input_proj_W), W(input_proj_b),
 W(csa_q_W), W(csa_k_W), W(csa_v_W), W(csa_out_W),
 W(csa_compress_w), W(csa_idx_DQ), W(csa_idx_K),
 W(hca_q_W), W(hca_k_W), W(hca_v_W), W(hca_out_W),
 W(gru_Wz), W(gru_bz), W(gru_Wr), W(gru_br), W(gru_Wh), W(gru_bh),
 W(peer_query_Ws), W(prod_keys_A), W(prod_keys_B),
 W(expert_W1), W(expert_b1), W(expert_W2), W(expert_b2),
 W(sc_alpha), W(sc_gru_decay), W(sc_lamb_eff), W(sc_beta1), W(sc_bc1), W(sc_bc2),
 (float)rescale, (float)beta2, (float)lr, (float)wd, (float)eps,
 (float)rho, (float)sam_on, (int)step, stream, /*ncta_cap=*/0);
 } else if (model == "vit") {
 const int64_t total = kVitTotalElems;
 const int GH = 4;
 const int64_t min_state = (int64_t)(5 + GH) * total + 1;
 TORCH_CHECK(params.numel() == total,
 "sg2_fused_step: vit params has ", params.numel(), " != ", total);
 TORCH_CHECK(grad.numel() == total,
 "sg2_fused_step: vit grad has ", grad.numel(), " != ", total);
 TORCH_CHECK(state.numel() >= min_state,
 "sg2_fused_step: vit state needs >= ", min_state, " got ", state.numel());
 TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.is_contiguous(),
 "sg2_fused_step: vit input must be contiguous float32 (patches++target-bits).");
 const int64_t in_n = input.numel();
 TORCH_CHECK(in_n % (kVitPatchElems + 1) == 0 && in_n > 0,
 "sg2_fused_step: vit input.numel() must be B*(patch_elems+1).");
 const int B = (int)(in_n / (kVitPatchElems + 1));
 ViTScratch& vsc = vit_scratch_for(params);
 fused::PersistentContext ctx{
 vsc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(vsc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(vsc.g_generation.data_ptr<int>()),
 32,                    // kVitNumTensors (mirror; the .cu static-asserts it)
 0u};
 float* loss_slot = st + 3 * total;
 // targets are bit-reinterpreted in the trailing float slots (same as fused_step).
 const float* patches = input.data_ptr<float>();
 const int* targets = reinterpret_cast<const int*>(input.data_ptr<float>() + (int64_t)B * kVitPatchElems);
 err = fused::sm90::mega_vit_sg2_tc(
 ctx, params.data_ptr<float>(), patches, targets, B,
 st, grad.data_ptr<float>(), loss_slot,
 W(input_proj_W), W(input_proj_b),
 W(csa_q_W), W(csa_k_W), W(csa_v_W), W(csa_out_W),
 W(csa_compress_w), W(csa_idx_DQ), W(csa_idx_K),
 W(hca_q_W), W(hca_k_W), W(hca_v_W), W(hca_out_W),
 W(gru_Wz), W(gru_bz), W(gru_Wr), W(gru_br), W(gru_Wh), W(gru_bh),
 W(peer_query_Ws), W(prod_keys_A), W(prod_keys_B),
 W(expert_W1), W(expert_b1), W(expert_W2), W(expert_b2),
 W(sc_alpha), W(sc_gru_decay), W(sc_lamb_eff), W(sc_beta1), W(sc_bc1), W(sc_bc2),
 (float)rescale, (float)beta2, (float)lr, (float)wd, (float)eps,
 (float)rho, (float)sam_on, (int)step, stream, /*ncta_cap=*/0);
 } else if (model == "mamba3") {
 // The mamba twin of the decoder branch: int32 tokens, the SAME SG2 meta-net bundle
 // + per-tensor scalar arrays + state layout. The mamba megakernel's P3-SG2 phase
 // runs sg2_meta_stages. ⚠ A/A/A: the SAM double-forward + segmented sort re-exercise
 // the shared mamba forward (the .sg2_spec.md-flagged risk); the gate re-verifies
 // bit-determinism, and the cell is landed DORMANT (gated out of _L3_WGMMA_CELLS) if it
 // re-trips the race. Routed here regardless so the kernel is exercisable + gateable.
 const int64_t total = kMambaTotalElems;
 const int GH = 4;   // SG2Dims gru_hidden default
 const int64_t min_state = (int64_t)(5 + GH) * total + 1;   // [m|v|mu|loss|sharpness|slow|gru_state]
 TORCH_CHECK(params.numel() == total,
 "sg2_fused_step: mamba params has ", params.numel(), " != ", total);
 TORCH_CHECK(grad.numel() == total,
 "sg2_fused_step: mamba grad has ", grad.numel(), " != ", total);
 TORCH_CHECK(state.numel() >= min_state,
 "sg2_fused_step: mamba state needs >= ", min_state,
 " floats ([m|v|mu|loss|sharpness|slow|gru_state(total*", GH, ")]) got ",
 state.numel(), ".");
 TORCH_CHECK(input.scalar_type() == torch::kInt32 && input.is_contiguous(),
 "sg2_fused_step: mamba input must be contiguous int32 (tokens[B*kSeq]++targets[B]).");
 const int64_t in_n = input.numel();
 TORCH_CHECK(in_n % (kMambaSeq + 1) == 0 && in_n > 0,
 "sg2_fused_step: mamba input.numel() must be B*(kSeq+1).");
 const int B = (int)(in_n / (kMambaSeq + 1));
 MambaScratch& msc = mamba_scratch_for(params);
 fused::PersistentContext ctx{
 msc.g_next.data_ptr<int>(),
 reinterpret_cast<unsigned*>(msc.g_arrived.data_ptr<int>()),
 reinterpret_cast<unsigned*>(msc.g_generation.data_ptr<int>()),
 28,                    // kMambaNumTensors (mirror; the .cu static-asserts it)
 0u};
 float* loss_slot = st + 3 * total;
 err = fused::sm90::mega_mamba_sg2_tc(
 ctx, params.data_ptr<float>(),
 input.data_ptr<int>(), input.data_ptr<int>() + (int64_t)B * kMambaSeq, B,
 st, grad.data_ptr<float>(), loss_slot,
 W(input_proj_W), W(input_proj_b),
 W(csa_q_W), W(csa_k_W), W(csa_v_W), W(csa_out_W),
 W(csa_compress_w), W(csa_idx_DQ), W(csa_idx_K),
 W(hca_q_W), W(hca_k_W), W(hca_v_W), W(hca_out_W),
 W(gru_Wz), W(gru_bz), W(gru_Wr), W(gru_br), W(gru_Wh), W(gru_bh),
 W(peer_query_Ws), W(prod_keys_A), W(prod_keys_B),
 W(expert_W1), W(expert_b1), W(expert_W2), W(expert_b2),
 W(sc_alpha), W(sc_gru_decay), W(sc_lamb_eff), W(sc_beta1), W(sc_bc1), W(sc_bc2),
 (float)rescale, (float)beta2, (float)lr, (float)wd, (float)eps,
 (float)rho, (float)sam_on, (int)step, stream, /*ncta_cap=*/0);
 } else {
 throw std::runtime_error("sg2_fused_step: SuperGrok2 L3-TC is wired for "
 "transformer_decoder + vit + mamba3; got " + model);
 }
 if (err != cudaSuccess)
 throw std::runtime_error(std::string("sg2_fused_step launch failed: ") +
 cudaGetErrorString(err));
#else
 (void)model; (void)params; (void)input; (void)grad; (void)state;
 throw std::runtime_error("sg2_fused_step: built without CUDA.");
#endif
}

} // namespace sg
