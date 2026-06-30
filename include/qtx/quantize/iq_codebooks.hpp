// ============================================================================
// I-Quant fixed codebooks. Verbatim transcription of ggml's
// `kmask_iq2xs`, `ksigns_iq2xs`, `iq{2xxs,2xs,2s,3xxs,3s,1s}_grid` tables
// from third_party/ggml_src/src/ggml-common.h. Carried over under the
// ggml MIT License (Copyright 2023-2026 The ggml authors).
//
// These tables are the on-wire codebook for the codebook-based
// I-Quant family (IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S). The
// codebook is fixed and identical across all ggml runtimes; the values
// here must match byte-for-byte for decode interop.
// ============================================================================
#pragma once

#include <cstdint>

namespace qtx::quantize::iq_codebooks {

constexpr float kIQ1S_Delta = 0.125f;
constexpr float kIQ1M_Delta = 0.125f;

extern const std::uint8_t  kmask_iq2xs[8];
extern const std::uint8_t  ksigns_iq2xs[128];
extern const std::uint64_t iq2xxs_grid[256];
extern const std::uint64_t iq2xs_grid[512];
extern const std::uint64_t iq2s_grid[1024];
extern const std::uint32_t iq3xxs_grid[256];
extern const std::uint32_t iq3s_grid[512];
extern const std::uint64_t iq1s_grid[2048];

}  // namespace qtx::quantize::iq_codebooks
