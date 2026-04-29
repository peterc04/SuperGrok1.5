// =====================================================================
//  arch_tier.h — minimal compile-time arch tier shim.
//
//  Recovered from the deleted csrc/common/dispatch.h@682eab4^. The
//  per-arch namespace refactor made runtime tier dispatch obsolete
//  (each TU knows its arch statically), but a few legacy sites in
//  csrc/kernels/{cuda,hip}/<arch>/distributed_{,scan_}pipeline_<arch>.*
//  still ask `get_arch_tier()` to pick between cp.async / TMA / generic
//  smem-load variants. Those files now include this header and consume
//  `kArchTier` (a constexpr) instead of the deleted runtime function.
//
//  The enum mirrors the pre-refactor one so the existing if/else chains
//  in those files still compile. Adding a new tier means extending the
//  enum and the per-arch macro switch below.
// =====================================================================

#pragma once

namespace sg {

enum class ArchTier {
    GENERIC,    // pre-Ampere NVIDIA, CDNA1-3 AMD (legacy bucket)
    AMPERE,     // sm_80, sm_86, sm_89 (Ada uses Ampere-tier dispatch)
    HOPPER,     // sm_90 — H100 / H200
    BLACKWELL,  // sm_100, sm_103, sm_120 — datacenter + ultra + consumer
    CDNA3,      // gfx942 — MI300X / MI300A
    CDNA4,      // gfx950 — MI350X / MI355X
};

#if   defined(SG_ARCH_SM80) || defined(SG_ARCH_SM89)
constexpr ArchTier kArchTier = ArchTier::AMPERE;
#elif defined(SG_ARCH_SM90)
constexpr ArchTier kArchTier = ArchTier::HOPPER;
#elif defined(SG_ARCH_SM100) || defined(SG_ARCH_SM103) || defined(SG_ARCH_SM120)
constexpr ArchTier kArchTier = ArchTier::BLACKWELL;
#elif defined(SG_ARCH_GFX942)
constexpr ArchTier kArchTier = ArchTier::CDNA3;
#elif defined(SG_ARCH_GFX950)
constexpr ArchTier kArchTier = ArchTier::CDNA4;
#else
constexpr ArchTier kArchTier = ArchTier::GENERIC;
#endif

inline constexpr ArchTier get_arch_tier() { return kArchTier; }

} // namespace sg

// The legacy code paths use unqualified `ArchTier::X` and `get_arch_tier()`.
// Re-expose them at translation-unit scope so existing if/else chains
// don't need rewriting.
using ArchTier = ::sg::ArchTier;
using ::sg::get_arch_tier;
