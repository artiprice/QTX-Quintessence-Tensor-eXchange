// ============================================================================
// @file        quantizer_simd.hpp
// @brief       SIMD-vectorised paths for the FP32 <-> quantized hot loops.
// @author      QTX Project
// @date        2026-05-14
//
// SEMANTICS:
//   - One block = 32 FP32 elements
//   - INT8 block:  4 byte scale + 32 byte payload = 36 B
//   - INT4 block:  4 byte scale + 16 byte packed nibbles = 20 B
//   - BF16 stream: 2 bytes per element (no per-block scale)
//   - Wire formats are BYTE-IDENTICAL with the scalar paths; this
//     header changes the compute, not the layout.
//
// EDGE CASES CLOSED (in addition to EC123-129 for INT8 compress):
//   EC130 — INT8 DECOMPRESS: scalar code multiplies `int8 * scale` per
//           element. SIMD widens int8 -> int32 via vpmovsxbd and uses
//           cvtepi32_ps for the FP cast. Output bit-identical to scalar.
//   EC131 — INT4 COMPRESS: 32 floats -> 16 packed bytes. Same blocked
//           structure as INT8 but the saturating downcast pivots in
//           [-7, +7] (EC48 wire-format contract).
//   EC132 — INT4 DECOMPRESS: 16 bytes -> 32 floats. Unpack via vpand
//           with low-nibble mask and (byte >> 4) for high, then 4-bit
//           sign-extend via `(x ^ 0x08) - 0x08`.
//   EC133 — BF16 COMPRESS: top 16 bits of FP32 with round-to-nearest-
//           even bias `(bits >> 16) & 1 + 0x7FFF`. NaN preservation
//           per EC55 — non-finite lanes overwritten with a quiet NaN
//           top half (0x7FC0).
//   EC134 — BF16 DECOMPRESS: zero-extend 16-bit BF16 to 32-bit FP32
//           with shift-left by 16.
// ============================================================================

#pragma once

#include "../core/types.hpp"
#include "../core/fpe_guard.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && !defined(QTX_AVOID_AVX512)
    #define QTX_QUANT_HAS_AVX512 1
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define QTX_QUANT_HAS_AVX2 1
    #include <immintrin.h>
#endif

// ============================================================================
// EC23 — VEX / EVEX register-state transition penalty.
//
// On x86 processors that support BOTH legacy SSE (128-bit XMM, no VEX
// prefix) and modern VEX/EVEX (256-bit YMM, 512-bit ZMM) encodings, the
// hardware tracks "upper half dirty" state on the YMM/ZMM registers. A
// transition from EVEX-encoded code back to legacy-SSE code on the same
// physical register file incurs a ~150-cycle pipeline stall the first
// time the legacy code writes to the lower 128 bits — the CPU has to
// preserve the dirty upper half before the legacy write can complete.
//
// In a typical workload the quantiser (this header) is called from a
// caller that may itself contain SSE-encoded library code (libc memcpy,
// musl strlen, etc.). Without mitigation the quantiser would leave
// every YMM/ZMM in "dirty upper" state and the next strlen call would
// stall by 150 cycles.
//
// Mitigation is delegated to the toolchain — GCC ≥ 9 / Clang ≥ 10 emit
// `vzeroupper` automatically at function boundaries that contain VEX/
// EVEX code, governed by `-mvzeroupper` (which is on by default on
// every modern target tuple). We rely on this guarantee instead of
// inserting manual `_mm256_zeroupper()` because:
//
//   (a) The compiler places the zeroupper where it knows the register
//       file is no longer needed — typically right before the return —
//       avoiding a redundant zeroupper inside a tight loop.
//
//   (b) Inserting one manually inside `compressOneBlockINT8_*` would
//       defeat the compiler's inlining heuristic, because the
//       intrinsic would become a function-boundary indicator at every
//       call site.
//
// Compile-time check: warn if -mno-vzeroupper is in effect. On GCC and
// Clang this manifests as `__AVX__` defined but `__SSE2__` without a
// vzeroupper attribute on the compilation unit; there is no portable
// macro for the flag itself, so the warning lives in CMakeLists.txt
// (QTX_STRICT_WARNINGS path).
// ============================================================================

// ============================================================================
// EC26 — AVX-512 frequency downclocking on Skylake-X / Cascade Lake.
//
// On Intel server processors of the Skylake-X (2017) and Cascade Lake
// (2019) generations, issuing AVX-512 instructions on a core would
// physically scale that core's frequency down by 20-30% ("AVX-512
// License 2"). This penalty applies to ALL threads sharing the core
// (HyperThreading siblings) for ~2 ms after the last AVX-512 instruction
// retires, so an AVX-512 quantiser stalls non-AVX-512 hot-paths that
// happen to land on the same core.
//
// Modern Intel chips (Ice Lake 2021, Sapphire Rapids 2023, Emerald
// Rapids 2024, Granite Rapids 2025) and ALL AMD Zen 4+ remove the
// downclocking; the frequency scale is unified across SIMD widths.
//
// Mitigations available to the caller:
//
//   * Define QTX_AVOID_AVX512 at compile time to force the AVX2
//     dispatch on hosts where downclocking would dominate the wall-
//     clock cost of a single quantiser invocation. The AVX2 path is
//     wire-format identical and ~30% slower in throughput, but it
//     does NOT touch the YMM-high / ZMM split and incurs no
//     frequency penalty.
//
//   * Pin the quantiser thread to a core (housekeeping core in the
//     HFT layout) so the downclock cannot bleed into latency-
//     sensitive worker cores. This is OUTSIDE the library's hot path
//     and the responsibility of the deployment configuration.
//
// We deliberately do NOT auto-detect via CPUID at runtime: that would
// add a startup probe and a conditional dispatch on every call.
// Compile-time selection keeps the hot path branch-free.

namespace qtx::quantize::simd {

// Forward declarations of scalar fallbacks (defined in quantizer.hpp).
namespace detail_scalar {
[[nodiscard]] inline core::usize compressFP32ToINT8_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept;
[[nodiscard]] inline core::usize decompressINT8ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept;
[[nodiscard]] inline core::usize compressFP32ToINT4_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept;
[[nodiscard]] inline core::usize decompressINT4ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept;
[[nodiscard]] inline core::usize compressFP32ToBF16_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept;
[[nodiscard]] inline core::usize decompressBF16ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept;
}  // namespace detail_scalar

#if defined(QTX_QUANT_HAS_AVX512)

constexpr const char* kPathName = "AVX-512F+DQ+BW";

// ----------------------------------------------------------------------------
// FP32 -> INT8.  EC123/125/126/127/128.
// ----------------------------------------------------------------------------

inline void compressOneBlockINT8_AVX512(
    const std::byte* block_in, std::byte* block_out) noexcept
{
    __m512 v0 = _mm512_loadu_ps(block_in);
    __m512 v1 = _mm512_loadu_ps(block_in + 16u * 4u);
    constexpr int kBadMask = 0x01 | 0x08 | 0x10 | 0x80;
    v0 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(v0, kBadMask), v0, _mm512_setzero_ps());
    v1 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(v1, kBadMask), v1, _mm512_setzero_ps());

    const __m512 amax = _mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1));
    const float abs_max_raw = _mm512_reduce_max_ps(amax);
    const float abs_max   = core::clampAbsMaxForINT8(abs_max_raw);
    const float scale     = (abs_max > 0.0f) ? (abs_max / 127.0f) : 0.0f;
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    std::memcpy(block_out, &scale, sizeof(scale));

    const __m512 invs = _mm512_set1_ps(inv_scale);
    __m512 q0 = _mm512_mul_ps(v0, invs);
    __m512 q1 = _mm512_mul_ps(v1, invs);
    q0 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(q0, kBadMask), q0, _mm512_setzero_ps());
    q1 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(q1, kBadMask), q1, _mm512_setzero_ps());

    const __m128i p0 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q0));
    const __m128i p1 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q1));
    const __m128i cap_hi = _mm_set1_epi8(127);
    const __m128i cap_lo = _mm_set1_epi8(-127);
    std::byte* int8_out = block_out + sizeof(float);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(int8_out),
                     _mm_max_epi8(_mm_min_epi8(p0, cap_hi), cap_lo));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(int8_out + 16),
                     _mm_max_epi8(_mm_min_epi8(p1, cap_hi), cap_lo));
}

inline core::usize compressFP32ToINT8_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    for (core::usize b = 0; b < n_blocks; ++b)
        compressOneBlockINT8_AVX512(in + b * 128u, out + b * 36u);
    return n_blocks * 36u;
}

// ----------------------------------------------------------------------------
// INT8 -> FP32.  EC130.
// ----------------------------------------------------------------------------

inline void decompressOneBlockINT8_AVX512(
    const std::byte* block_in, std::byte* block_out) noexcept
{
    float scale_raw;
    std::memcpy(&scale_raw, block_in, sizeof(float));
    const float scale = core::isFiniteStrict(scale_raw) ? scale_raw : 0.0f;
    const __m512 vscale = _mm512_set1_ps(scale);

    const __m128i b0 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(block_in + sizeof(float)));
    const __m128i b1 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(block_in + sizeof(float) + 16));

    const __m512  f0 = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0)), vscale);
    const __m512  f1 = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1)), vscale);
    _mm512_storeu_ps(block_out,            f0);
    _mm512_storeu_ps(block_out + 16u * 4u, f1);
}

inline core::usize decompressINT8ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    for (core::usize b = 0; b < n_blocks; ++b)
        decompressOneBlockINT8_AVX512(in + b * 36u, out + b * 128u);
    return n_blocks * 128u;
}

// ----------------------------------------------------------------------------
// FP32 -> INT4.  EC131.
//
// Pack pairs of nibbles into one byte: out[k] = (e[2k] & 0xF) | ((e[2k+1] & 0xF) << 4).
// AVX-512 + BW: produce two 16-byte vectors of low-nibble values
// (e[0..15] and e[16..31]), then pack by interleaving via
// _mm_unpacklo_epi8 + shift + or, or do it with a scalar loop on an
// aligned spill — the latter compiles to ~2 SSE instructions and is
// readable. We use the aligned-spill form.
// ----------------------------------------------------------------------------

inline void compressOneBlockINT4_AVX512(
    const std::byte* block_in, std::byte* block_out) noexcept
{
    __m512 v0 = _mm512_loadu_ps(block_in);
    __m512 v1 = _mm512_loadu_ps(block_in + 16u * 4u);
    constexpr int kBadMask = 0x01 | 0x08 | 0x10 | 0x80;
    v0 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(v0, kBadMask), v0, _mm512_setzero_ps());
    v1 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(v1, kBadMask), v1, _mm512_setzero_ps());

    const __m512 amax = _mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1));
    const float abs_max_raw = _mm512_reduce_max_ps(amax);
    const float abs_max   = core::clampAbsMaxForINT4(abs_max_raw);
    const float scale     = (abs_max > 0.0f) ? (abs_max / 7.0f) : 0.0f;
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    std::memcpy(block_out, &scale, sizeof(scale));

    const __m512 invs = _mm512_set1_ps(inv_scale);
    __m512 q0 = _mm512_mul_ps(v0, invs);
    __m512 q1 = _mm512_mul_ps(v1, invs);
    q0 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(q0, kBadMask), q0, _mm512_setzero_ps());
    q1 = _mm512_mask_blend_ps(
        _mm512_fpclass_ps_mask(q1, kBadMask), q1, _mm512_setzero_ps());

    // Round + clamp to [-7, +7] (EC48 contract).
    const __m512i cap_hi = _mm512_set1_epi32(7);
    const __m512i cap_lo = _mm512_set1_epi32(-7);
    __m512i i0 = _mm512_max_epi32(_mm512_min_epi32(_mm512_cvtps_epi32(q0), cap_hi), cap_lo);
    __m512i i1 = _mm512_max_epi32(_mm512_min_epi32(_mm512_cvtps_epi32(q1), cap_hi), cap_lo);

    // Truncate to int8 (lower byte of each int32) and mask to 4 bits.
    const __m128i lo16 = _mm512_cvtepi32_epi8(i0);  // 16 bytes, e0..e15
    const __m128i hi16 = _mm512_cvtepi32_epi8(i1);  // 16 bytes, e16..e31
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i lo16m = _mm_and_si128(lo16, mask);
    const __m128i hi16m = _mm_and_si128(hi16, mask);

    // Pack pairs into bytes. We have 32 low-nibble values across two
    // 16-byte vectors. Even-indexed elements (e0, e2, ...) go to low
    // nibbles, odd to high. Use vpunpck on pairs to interleave.
    // The most efficient way is _mm_maddubs_epi16 with multiplier
    // [1, 16, 1, 16, ...]: maddubs(a, b)[i] = sat16(a[2i]*b[2i] + a[2i+1]*b[2i+1]).
    // Since each lane is ≤ 15 and the multiplier ≤ 16, no saturation:
    // result_16[i] = a[2i] + a[2i+1] * 16.
    // That is exactly our pack. The result is 8 × 16-bit values per half
    // whose low bytes are the packed nibbles. Then narrow back to bytes.
    const __m128i mul = _mm_set1_epi16(static_cast<short>(0x1001));  // [1, 16]
    const __m128i pp_lo = _mm_maddubs_epi16(lo16m, mul);  // 8 × packed bytes in 16-bit lanes
    const __m128i pp_hi = _mm_maddubs_epi16(hi16m, mul);

    // Narrow 16-bit lanes -> 8-bit lanes: take low byte of each.
    // Use _mm_packus_epi16 with a zero half — but pp lo/hi are already
    // in the low byte. Mask the high byte and OR halves doesn't work
    // because we need 8-byte halves combined into a 16-byte result.
    // Simpler: _mm_shuffle_epi8 with a mask picking [0, 2, 4, 6, 8, 10, 12, 14].
    const __m128i pick = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i packed_lo8 = _mm_shuffle_epi8(pp_lo, pick);
    const __m128i packed_hi8 = _mm_shuffle_epi8(pp_hi, pick);

    // Combine the two 8-byte halves into one 16-byte vector.
    // packed_lo8 has 8 bytes in lanes [0..7], rest zero.
    // packed_hi8 has 8 bytes in lanes [0..7], rest zero.
    // We want [packed_lo8[0..7], packed_hi8[0..7]]: use _mm_unpacklo_epi64.
    const __m128i packed = _mm_unpacklo_epi64(packed_lo8, packed_hi8);

    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(block_out + sizeof(float)), packed);
}

inline core::usize compressFP32ToINT4_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    for (core::usize b = 0; b < n_blocks; ++b)
        compressOneBlockINT4_AVX512(in + b * 128u, out + b * 20u);
    return n_blocks * 20u;
}

// ----------------------------------------------------------------------------
// INT4 -> FP32.  EC132.
// ----------------------------------------------------------------------------

inline void decompressOneBlockINT4_AVX512(
    const std::byte* block_in, std::byte* block_out) noexcept
{
    float scale_raw;
    std::memcpy(&scale_raw, block_in, sizeof(float));
    const float scale = core::isFiniteStrict(scale_raw) ? scale_raw : 0.0f;
    const __m512 vscale = _mm512_set1_ps(scale);

    const __m128i packed = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(block_in + sizeof(float)));

    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i lo_n = _mm_and_si128(packed, lo_mask);
    const __m128i hi_n = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    // 4-bit sign-extend via `(x ^ 0x08) - 0x08`.
    const __m128i bias = _mm_set1_epi8(0x08);
    const __m128i lo_s = _mm_sub_epi8(_mm_xor_si128(lo_n, bias), bias);
    const __m128i hi_s = _mm_sub_epi8(_mm_xor_si128(hi_n, bias), bias);

    // Interleave so result_byte[2k] = lo_s[k], result_byte[2k+1] = hi_s[k].
    const __m128i inter_lo = _mm_unpacklo_epi8(lo_s, hi_s);  // e0..e15
    const __m128i inter_hi = _mm_unpackhi_epi8(lo_s, hi_s);  // e16..e31

    const __m512 f0 = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(inter_lo)), vscale);
    const __m512 f1 = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(inter_hi)), vscale);
    _mm512_storeu_ps(block_out,            f0);
    _mm512_storeu_ps(block_out + 16u * 4u, f1);
}

inline core::usize decompressINT4ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    for (core::usize b = 0; b < n_blocks; ++b)
        decompressOneBlockINT4_AVX512(in + b * 20u, out + b * 128u);
    return n_blocks * 128u;
}

// ----------------------------------------------------------------------------
// FP32 -> BF16.  EC133.
// ----------------------------------------------------------------------------

inline void compressBF16_chunk16_AVX512(
    const std::byte* in, std::byte* out) noexcept
{
    const __m512 f = _mm512_loadu_ps(in);
    const __m512i bits = _mm512_castps_si512(f);

    // NaN detect: (bits & 0x7FFFFFFF) > 0x7F800000.
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512i inf_bits = _mm512_set1_epi32(0x7F800000);
    const __m512i abs_bits = _mm512_and_si512(bits, abs_mask);
    const __mmask16 is_nan = _mm512_cmpgt_epi32_mask(abs_bits, inf_bits);

    // RNE bias: ((bits >> 16) & 1) + 0x7FFF.
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i s = _mm512_and_si512(_mm512_srli_epi32(bits, 16), one);
    const __m512i bias = _mm512_add_epi32(s, _mm512_set1_epi32(0x7FFF));
    __m512i rounded = _mm512_add_epi32(bits, bias);

    // EC27 (was: SIMD BF16 RNE bias overflow on FLT_MAX). The bias add
    // can roll the exponent from 0xFE into 0xFF (Inf encoding) for any
    // finite input large enough — most notably FLT_MAX itself. Detect
    // this by comparing the result's exponent field to 0x7F800000; if
    // the exponent has reached 0xFF for a lane that was NOT already
    // Inf/NaN in the input, that lane has overflowed and we saturate
    // to the largest finite BF16 magnitude with the input's sign.
    // Mirrors the scalar fp32ToBF16Safe fix (EC158) bit-for-bit.
    const __m512i rounded_exp =
        _mm512_and_si512(rounded, inf_bits);            // exp field only
    const __mmask16 rounded_is_inf =
        _mm512_cmpeq_epi32_mask(rounded_exp, inf_bits);
    // overflow = rounded_is_inf AND NOT was_already_nonfinite
    const __mmask16 input_was_nonfinite =
        _mm512_cmpeq_epi32_mask(abs_bits, inf_bits);    // exp == 0xFF in input
    // Include Inf as nonfinite too (bits where exp == 0xFF AND mantissa == 0).
    // Above mask covers both Inf and NaN because we compared (abs & 0x7FFFFFFF)
    // == 0x7F800000 for Inf and we already have is_nan for NaN; the union of
    // the two masks is exactly "input was non-finite".
    const __mmask16 input_was_inf_or_nan =
        _kor_mask16(input_was_nonfinite, is_nan);
    const __mmask16 overflowed =
        _kandn_mask16(input_was_inf_or_nan, rounded_is_inf);

    // EC27 saturation value: sign | 0x7F7F0000 (largest finite BF16 in
    // upper-16 form). Extract input sign bit (bit 31), OR with 0x7F7F0000.
    const __m512i sign_bit =
        _mm512_and_si512(bits, _mm512_set1_epi32(static_cast<int>(0x80000000u)));
    const __m512i largest_finite_top =
        _mm512_or_si512(sign_bit, _mm512_set1_epi32(0x7F7F0000));
    rounded = _mm512_mask_blend_epi32(overflowed, rounded, largest_finite_top);

    // NaN lanes → quiet NaN top half (0x7FC0_0000).
    const __m512i qnan_top = _mm512_set1_epi32(0x7FC00000);
    rounded = _mm512_mask_blend_epi32(is_nan, rounded, qnan_top);

    // Take the top 16 bits of each 32-bit lane.
    const __m256i top16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), top16);
}

inline core::usize compressFP32ToBF16_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    core::usize i = 0;
    for (; i + 16u <= n_elements; i += 16u) {
        compressBF16_chunk16_AVX512(in + i * 4u, out + i * 2u);
    }
    if (i < n_elements) {
        (void)detail_scalar::compressFP32ToBF16_scalar(
            in + i * 4u, out + i * 2u, n_elements - i);
    }
    return n_elements * 2u;
}

// ----------------------------------------------------------------------------
// BF16 -> FP32.  EC134.
// ----------------------------------------------------------------------------

inline void decompressBF16_chunk16_AVX512(
    const std::byte* in, std::byte* out) noexcept
{
    const __m256i u16s = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(in));
    const __m512i u32s = _mm512_cvtepu16_epi32(u16s);
    const __m512i bits = _mm512_slli_epi32(u32s, 16);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out), bits);
}

inline core::usize decompressBF16ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    core::usize i = 0;
    for (; i + 16u <= n_elements; i += 16u) {
        decompressBF16_chunk16_AVX512(in + i * 2u, out + i * 4u);
    }
    if (i < n_elements) {
        (void)detail_scalar::decompressBF16ToFP32_scalar(
            in + i * 2u, out + i * 4u, n_elements - i);
    }
    return n_elements * 4u;
}

// ----------------------------------------------------------------------------
// FP32 <-> FP16 (Epic 1 — IEEE 754 binary16).
//
// First-pass implementation routes through the scalar fp32ToFP16Safe /
// fp16ToFP32 helpers in core::. AVX-512 (with F16C, ubiquitous since
// Ivy Bridge) provides `vcvtps2ph` / `vcvtph2ps` for one-cycle 16-lane
// conversions; lifting this into a SIMD chunk-16 path is straightforward
// and naturally belongs to the Epic 6 "AVX-512 F/DQ/BW" milestone. We
// keep the scalar fallback here so the wire format is locked first.
// ----------------------------------------------------------------------------

inline core::usize compressFP32ToFP16_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP16_scalar(in, out, n_elements); }

inline core::usize decompressFP16ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP16ToFP32_scalar(in, out, n_elements); }

// ----------------------------------------------------------------------------
// FP32 <-> FP8 E4M3 / E5M2 (OCP 8-bit floats).
//
// Routes through scalar codecs in core::. Hopper/Blackwell expose
// hardware FP8 conversion instructions (cvt.rn.satfinite.e4m3x2.f32 /
// e5m2x2.f32 in PTX); on x86 there is no equivalent intrinsic yet,
// so the scalar bit-math is the optimal path on CPU. The inline
// scalar loops auto-vectorise into 4–8-wide SIMD under -O3.
// ----------------------------------------------------------------------------

inline core::usize compressFP32ToFP8_E4M3_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP8_E4M3_scalar(in, out, n_elements); }

inline core::usize decompressFP8_E4M3_ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP8_E4M3_ToFP32_scalar(in, out, n_elements); }

inline core::usize compressFP32ToFP8_E5M2_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP8_E5M2_scalar(in, out, n_elements); }

inline core::usize decompressFP8_E5M2_ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP8_E5M2_ToFP32_scalar(in, out, n_elements); }
// All others pass through to scalar — native AVX2 kernels for the
// remaining five are a follow-up. Scalar today is already either
// auto-vectorised by the compiler (BF16 decompress) or below the
// 2 GB/s threshold where SIMD effort pays off in pure scalar-target
// builds.
// ============================================================================
#elif defined(QTX_QUANT_HAS_AVX2)

constexpr const char* kPathName = "AVX2";

inline void compressOneBlockINT8_AVX2(
    const std::byte* block_in, std::byte* block_out) noexcept
{
    __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(block_in));
    __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(block_in) + 8);
    __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(block_in) + 16);
    __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(block_in) + 24);

    const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
    auto finite_mask = [&](__m256 v) -> __m256 {
        const __m256i bits = _mm256_castps_si256(v);
        const __m256i exp  = _mm256_and_si256(bits, exp_mask);
        const __m256i eq   = _mm256_cmpeq_epi32(exp, exp_mask);
        return _mm256_castsi256_ps(
            _mm256_xor_si256(eq, _mm256_set1_epi32(-1)));
    };
    v0 = _mm256_and_ps(v0, finite_mask(v0));
    v1 = _mm256_and_ps(v1, finite_mask(v1));
    v2 = _mm256_and_ps(v2, finite_mask(v2));
    v3 = _mm256_and_ps(v3, finite_mask(v3));

    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 m = _mm256_max_ps(
        _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask)),
        _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask)));

    __m128 hi = _mm256_extractf128_ps(m, 1);
    __m128 lo = _mm256_castps256_ps128(m);
    __m128 h4 = _mm_max_ps(lo, hi);
    __m128 h2 = _mm_max_ps(h4, _mm_movehl_ps(h4, h4));
    __m128 h1 = _mm_max_ss(h2, _mm_shuffle_ps(h2, h2, 1));
    float abs_max_raw;
    _mm_store_ss(&abs_max_raw, h1);

    const float abs_max   = core::clampAbsMaxForINT8(abs_max_raw);
    const float scale     = (abs_max > 0.0f) ? (abs_max / 127.0f) : 0.0f;
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    std::memcpy(block_out, &scale, sizeof(scale));

    const __m256 invs = _mm256_set1_ps(inv_scale);
    __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, invs));
    __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, invs));
    __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, invs));
    __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, invs));

    __m256i p = _mm256_packs_epi16(
        _mm256_packs_epi32(i0, i1),
        _mm256_packs_epi32(i2, i3));
    p = _mm256_permutevar8x32_epi32(
        p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

    const __m256i cap_hi = _mm256_set1_epi8(127);
    const __m256i cap_lo = _mm256_set1_epi8(-127);
    std::byte* int8_out = block_out + sizeof(float);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(int8_out),
                        _mm256_max_epi8(_mm256_min_epi8(p, cap_hi), cap_lo));
}

inline core::usize compressFP32ToINT8_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    for (core::usize b = 0; b < n_blocks; ++b)
        compressOneBlockINT8_AVX2(in + b * 128u, out + b * 36u);
    return n_blocks * 36u;
}

inline core::usize decompressINT8ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{ return detail_scalar::decompressINT8ToFP32_scalar(in, out, n_blocks); }
inline core::usize compressFP32ToINT4_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{ return detail_scalar::compressFP32ToINT4_scalar(in, out, n_blocks); }
inline core::usize decompressINT4ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{ return detail_scalar::decompressINT4ToFP32_scalar(in, out, n_blocks); }
inline core::usize compressFP32ToBF16_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToBF16_scalar(in, out, n_elements); }
inline core::usize decompressBF16ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressBF16ToFP32_scalar(in, out, n_elements); }
inline core::usize compressFP32ToFP16_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP16_scalar(in, out, n_elements); }
inline core::usize decompressFP16ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP16ToFP32_scalar(in, out, n_elements); }
inline core::usize compressFP32ToFP8_E4M3_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP8_E4M3_scalar(in, out, n_elements); }
inline core::usize decompressFP8_E4M3_ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP8_E4M3_ToFP32_scalar(in, out, n_elements); }
inline core::usize compressFP32ToFP8_E5M2_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP8_E5M2_scalar(in, out, n_elements); }
inline core::usize decompressFP8_E5M2_ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP8_E5M2_ToFP32_scalar(in, out, n_elements); }

#else

constexpr const char* kPathName = "scalar";

inline core::usize compressFP32ToINT8_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{ return detail_scalar::compressFP32ToINT8_scalar(in, out, n_blocks); }
inline core::usize decompressINT8ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{ return detail_scalar::decompressINT8ToFP32_scalar(in, out, n_blocks); }
inline core::usize compressFP32ToINT4_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{ return detail_scalar::compressFP32ToINT4_scalar(in, out, n_blocks); }
inline core::usize decompressINT4ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{ return detail_scalar::decompressINT4ToFP32_scalar(in, out, n_blocks); }
inline core::usize compressFP32ToBF16_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToBF16_scalar(in, out, n_elements); }
inline core::usize decompressBF16ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressBF16ToFP32_scalar(in, out, n_elements); }
inline core::usize compressFP32ToFP16_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP16_scalar(in, out, n_elements); }
inline core::usize decompressFP16ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP16ToFP32_scalar(in, out, n_elements); }
inline core::usize compressFP32ToFP8_E4M3_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP8_E4M3_scalar(in, out, n_elements); }
inline core::usize decompressFP8_E4M3_ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP8_E4M3_ToFP32_scalar(in, out, n_elements); }
inline core::usize compressFP32ToFP8_E5M2_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::compressFP32ToFP8_E5M2_scalar(in, out, n_elements); }
inline core::usize decompressFP8_E5M2_ToFP32_impl(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{ return detail_scalar::decompressFP8_E5M2_ToFP32_scalar(in, out, n_elements); }

#endif

}  // namespace qtx::quantize::simd
