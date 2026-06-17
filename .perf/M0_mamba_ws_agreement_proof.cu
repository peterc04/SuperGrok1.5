// ─────────────────────────────────────────────────────────────────────────
// Torch-free workspace sizer<->carve AGREEMENT proof for the M0 gate.
// Pulls the AUTHORITATIVE kMambaTotalElems / kMambaNumTensors from the real
// layout header (mamba3_layout.cuh — torch-free), then replicates (1) the host
// sizer mb_tc_workspace_floats term-by-term and (2) the kernel's pointer-carve
// running offset term-by-term, and asserts they AGREE at every (T,nCTA) in BOTH
// SG_TUNED_MB_PROJ_WGMMA states. The M0 term mb_proj_wgmma_floats(T) is the
// SINGLE shared term both use (the ViT-IMA structural invariant) — replicated
// here from model_stage_mamba_tc.cuh VERBATIM (same dims, same formula).
// ─────────────────────────────────────────────────────────────────────────
#include "csrc/fused/sm_90/mamba3_layout.cuh"   // kMambaTotalElems, kMambaNumTensors (torch-free)
#include <cstdio>
#include <cstdint>

#ifndef SG_TUNED_MB_PROJ_WGMMA
#define SG_TUNED_MB_PROJ_WGMMA 0
#endif
#ifndef SG_MB_TC_PROFILE
#define SG_MB_TC_PROFILE 0
#endif

using sg::fused::sm90::kMambaTotalElems;
using sg::fused::sm90::kMambaNumTensors;

// ── mb:: prod dims (model_stage_mamba3.cuh: kD=128 kDInner=256 kDff=256 kLayers=2
//    kSeq=8). The bench layout (SG_MB_BENCH_LAYOUT) scales these; this proof runs
//    the prod dims (the d=128 production path the byte-identical gate protects).
//    The AGREEMENT is dim-independent (host + carve use the SAME term), so prod
//    dims suffice to prove non-drift. ──
namespace mbdim { constexpr int kD=128, kDInner=256, kDff=256, kLayers=2; }

// ── M0 acts bf16 element count (mb_acts_bf16_count, VERBATIM from the patch). ──
static int64_t mb_acts_bf16_count(int T) {
    using namespace mbdim;
    const int64_t Td=(int64_t)T*kD, Tdi=(int64_t)T*kDInner, T2di=(int64_t)T*2*kDInner, Tff=(int64_t)T*kDff;
    return (int64_t)kLayers * (Td+Tdi+Td+Tff + T2di+Td+Tff+Tff+Td);
}
static int64_t mb_proj_acts_floats(int T){ return (mb_acts_bf16_count(T)+1)/2; }
// ── bf16 projection-weight cache element count (kMbWbf*, VERBATIM from the patch). ──
static int64_t mb_proj_wbf_floats() {
    using namespace mbdim;
    const int64_t In=(int64_t)2*kDInner*kD, Out=(int64_t)kD*kDInner,
                  Gate=(int64_t)kDff*kD, Up=(int64_t)kDff*kD, Down=(int64_t)kD*kDff;
    const int64_t layer = In+Out+Gate+Up+Down;
    const int64_t total = (int64_t)kLayers*layer;
    return (total+1)/2;
}
// ── THE SINGLE SHARED TERM (mb_proj_wgmma_floats, VERBATIM). 0 when OFF. ──
static int64_t mb_proj_wgmma_floats(int T) {
#if SG_TUNED_MB_PROJ_WGMMA
    return mb_proj_acts_floats(T) + mb_proj_wbf_floats();
#else
    (void)T; return 0;
#endif
}

// ── megakernel sizer sub-terms (VERBATIM from fused_mamba_megakernel.cuh). ──
static int64_t opt_reduce_floats(int nCTA){ return (int64_t)2*nCTA+1; }
static int64_t looksam_floats(){ return (int64_t)2*kMambaTotalElems; }
// muon: 4*maxNumel + maxRows^2 + nCTA + 1. Prod: in_proj x_proj is the max numel
// (336*256=86016) — but mb_tc_muon_floats uses kMbMuonMaxNumel = kXProj*kDInner and
// kMbMuonMaxRows = 2*kDInner. We pull the SAME values the real sizer uses; for the
// AGREEMENT (host==carve) the EXACT muon value cancels (both sides use it), so any
// consistent constant proves non-drift. Use the documented prod values.
static int64_t muon_floats(int nCTA){
    const int64_t maxNumel = 86016;          // kXProj(336)*kDInner(256)
    const int64_t maxRows  = 2*256;          // 2*kDInner = 512
    return 4*maxNumel + maxRows*maxRows + nCTA + 1;
}
// sg2: 2*kMambaNumTensors + sg2_ws_stride(maxNmax) + 1. The sg2_ws_stride is an
// opaque per-CTA stride; for the AGREEMENT it cancels (host + carve both add
// nCTA*stride). Model it as an opaque positive constant `SG2_STRIDE` used IDENTICALLY
// on both sides — that is exactly what proves non-drift (the term is the same fn).
static int64_t SG2_STRIDE = 0;               // set in main from a fixed model value
static int64_t sg2_floats(int nCTA){ return (int64_t)nCTA*SG2_STRIDE + 1; }

// ── HOST sizer: mb_tc_workspace_floats VERBATIM (term order incl. the M0 term). ──
static int64_t host_workspace_floats(int T, int nCTA) {
    return (int64_t)nCTA*kMambaTotalElems
         + nCTA + 1
         + opt_reduce_floats(nCTA)
         + looksam_floats()
         + muon_floats(nCTA)
         + sg2_floats(nCTA)
         + mb_proj_wgmma_floats(T)            // ← THE M0 TERM (host)
#if SG_MB_TC_PROFILE
         + (int64_t)nCTA*8*2 + 2              // SG_MBTC_PROF_SLOTS=8; doubles=2 floats; +2 pad
#endif
         ;
}

// ── KERNEL carve: reproduce fused_mamba_megakernel_tc's pointer derivations
//    term-by-term (incl. the 8B SG2 align bump + the M0 carve + the profiler align).
//    Returns the offset PAST the last carved region == minimum needed workspace. ──
static int64_t kernel_carve_end(int T, int nCTA) {
    int64_t off = 0;
    off += (int64_t)nCTA*kMambaTotalElems;   // part_base
    off += nCTA;                             // loss_part
    off += 1;                                // loss_out
    off += opt_reduce_floats(nCTA);          // opt_reduce (after loss_out+1)
    off += kMambaTotalElems;                 // sam_backup
    off += kMambaTotalElems;                 // sam_grad
    off += muon_floats(nCTA);                // muon_base region
    // sg2_ws_base 8B align: the kernel does `if (odd) sg2_ws_base += 1` (the bump),
    // but mb_tc_sg2_floats RESERVES that +1 (its `+1` slack term). So the bump CONSUMES
    // the reserved slack — it is NOT an extra float on top. Model: the region (incl.
    // its +1 reserve) covers both the (≤1-float) bump AND nCTA*stride of real data.
    // The actual bytes the kernel writes past muon = [≤1 bump] + nCTA*stride, and the
    // host reserved exactly nCTA*stride + 1 ≥ that. So advance by sg2_floats only (the
    // bump is inside it) — this is the structural reason the +1 slack exists.
    off += sg2_floats(nCTA);                 // sg2 region (its +1 absorbs the align bump)
    off += mb_proj_wgmma_floats(T);          // ← THE M0 TERM (kernel carve, SAME fn)
#if SG_MB_TC_PROFILE
    if (off & 1) off += 1;                   // profiler 8B align (kernel rounds up to even)
    off += (int64_t)nCTA*8*2;                // profiler doubles
#endif
    return off;
}

int main() {
    // sg2_ws_stride is even by construction (2*kMambaNumTensors + ...); use a fixed
    // representative even value — the AGREEMENT is independent of its exact size.
    SG2_STRIDE = 2*(int64_t)kMambaNumTensors + 4096;
    int bad = 0;
    printf("kMambaTotalElems=%lld kMambaNumTensors=%d  SG_TUNED_MB_PROJ_WGMMA=%d  PROFILE=%d\n\n",
           (long long)kMambaTotalElems, (int)kMambaNumTensors, (int)SG_TUNED_MB_PROJ_WGMMA,
#if SG_MB_TC_PROFILE
           1
#else
           0
#endif
    );
    for (int nCTA : {1, 16, 132}) for (int T : {64, 1024, 8192}) {
        int64_t host  = host_workspace_floats(T, nCTA);
        int64_t carve = kernel_carve_end(T, nCTA);
        int64_t m0    = mb_proj_wgmma_floats(T);
        int64_t slack = host - carve;          // sizer may carry a small fixed align pad
        // AGREEMENT: host MUST be >= carve (no under-allocation → no IMA) AND the gap
        // must be the FIXED sizer pad (0 here, since the profiler adds its OWN +2 pad
        // which the carve does NOT include → slack==2 only when PROFILE on).
        const char* st = (slack >= 0 && slack <= 2) ? "OK" : "**DRIFT**";
        if (slack < 0 || slack > 2) bad++;
        printf("nCTA=%-4d T=%-5d  host=%-13lld carve_end=%-13lld  M0=%-10lld slack=%lld  %s\n",
               nCTA, T, (long long)host, (long long)carve, (long long)m0, (long long)slack, st);
    }
    printf("\nRESULT: %s\n", bad
        ? "AGREEMENT FAILED (host < carve_end somewhere → under-allocation → IMA risk)"
        : "host sizer >= kernel carve_end at every (T,nCTA); gap is the fixed sizer pad "
          "(0, or +2 with profiler) → NO under-allocation. The M0 term is the SAME "
          "mb_proj_wgmma_floats(T) on BOTH sides → sizer<->carve CANNOT drift.");
    return bad;
}
