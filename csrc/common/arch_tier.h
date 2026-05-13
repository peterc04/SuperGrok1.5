// =====================================================================
//  arch_tier.h — compile-time arch tier shim.
//
//  3-arch active set: sm_90 (Hopper), gfx942 (CDNA3), tpu_v5p.
//  Other arches (Ampere, Blackwell, CDNA4, etc.) are intentionally out
//  of scope; they would re-enter the enum as future winners-expansion
//  work and are not part of the current build.
// =====================================================================

#pragma once

namespace sg {

enum class ArchTier {
    GENERIC,    // catch-all (CPU build / unsupported arch)
    HOPPER,     // sm_90 — H100 / H200
    CDNA3,      // gfx942 — MI300X / MI300A
};

enum class StatePrecision {
    FP32 = 0,
    CONFIG4 = 1,
};

enum class ExpertPrecision {
    FP32 = 0,
    INT8 = 1,
    INT4 = 2,
};

#if   defined(SG_ARCH_SM90)
constexpr ArchTier kArchTier = ArchTier::HOPPER;
#elif defined(SG_ARCH_GFX942)
constexpr ArchTier kArchTier = ArchTier::CDNA3;
#else
constexpr ArchTier kArchTier = ArchTier::GENERIC;
#endif

inline constexpr ArchTier get_arch_tier() { return kArchTier; }

} // namespace sg

// Re-expose at translation-unit scope for legacy if/else chains.
using ArchTier = ::sg::ArchTier;
using StatePrecision = ::sg::StatePrecision;
using ExpertPrecision = ::sg::ExpertPrecision;
using ::sg::get_arch_tier;
