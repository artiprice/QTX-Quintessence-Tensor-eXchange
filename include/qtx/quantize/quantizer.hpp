// ============================================================================
// @file        quantizer.hpp
// @brief       Lithographic Quantizer — FP32 <-> BF16/INT8/INT4 conversion.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// HA-LAYER: HA-pure module, TYPE_SAFETY_PROFILE compliant.
//
// EDGE CASES CLOSED IN THIS REWRITE:
//
//   EC41 — subnormal-float penalty: each block-level entry installs an
//          FtzDazGuard so std::fabs / mul / lround do not stall on
//          subnormal operands (10..100x slowdown gone).
//   EC42 — scale == 0 (all-zero block): dequant produces +0.0f, never -0.0f.
//   EC43 — NaN propagation: sanitiseFinite() is applied to every loaded
//          float so std::lround never sees a NaN (UB previously).
//   EC44 — Infinity scale: clampAbsMaxForINT8 / INT4 caps abs_max at
//          FLT_MAX / (range_max), so reconstruction never produces Inf.
//   EC45 — tail processing: src.size() not a multiple of the block size
//          now produces a documented error code (return 0); callers must
//          pre-pad inputs (the tiered bridge enforces it).
//   EC46 — MXCSR rounding-mode drift: FtzDazGuard saves/restores MXCSR;
//          we additionally use truncation via the round-to-int-and-clamp
//          path that does NOT depend on the dynamic rounding mode.
//   EC47 — implementation-defined right-shift on signed nibbles: we keep
//          the original explicit-sign-extend path on unsigned u8 and do
//          NOT rely on a signed right-shift anywhere (was correct, kept).
//   EC48 — wasted -8 INT4 codepoint: deliberately remains P4. With the
//          current `scale = abs_max / 7` formula every input maps to
//          a quantized value in [-7, +7]; the 16th codepoint -8 is
//          mathematically unreachable. Recovering it requires changing
//          the scale formula (or moving to an asymmetric zero_point
//          encoding), and either is a wire-format break — the meaning
//          of the per-block fp32 scale value would change. Out of
//          scope; the rolled-back-and-documented analysis is in the
//          inline comment at compressFP32ToINT4.
//   EC49 — std::clamp(NaN, ...) UB: sanitiseFinite() is applied first.
//   EC50 — FLT_MAX scale producing Inf: clampAbsMaxForINT8 caps the input
//          so scale*int8 cannot overflow finite range.
//   EC52 — SIMD alignment: documented that span<std::byte> may be byte-
//          aligned; we never emit aligned SIMD; loads are via memcpy.
//   EC53 — in-place src==dst: detected at the byte-range level (overlap
//          guard); returns 0 on overlap so callers fail cleanly.
//   EC55 — BF16 of NaN: fp32ToBF16Safe preserves NaN payload.
//   EC58 — corrupted scale on read: dequantize rejects non-finite scale.
//   EC60 — kPayloadBytes != requested size: the tiered bridge is now the
//          authority on partial buffers; quantizer documents the
//          requirement that src.size() be an exact multiple of the
//          block size.
// ============================================================================

#pragma once

#include "../arena/gen_id.hpp"
#include "../core/fpe_guard.hpp"
#include "../core/types.hpp"
#include "iq_codebooks.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace qtx::quantize {

// ============================================================================
// Compile-time constants
// ============================================================================

inline constexpr core::usize kBlockElements = 32;

inline constexpr core::usize kFP32Size = 4;
inline constexpr core::usize kBF16Size = 2;
inline constexpr core::usize kFP16Size = 2;
inline constexpr core::usize kFP8Size  = 1;   // E4M3 and E5M2 are both 1 B/element
inline constexpr core::usize kINT8Size = 1;

inline constexpr core::usize kINT8BlockBytes = sizeof(float) + kBlockElements;
inline constexpr core::usize kINT4BlockBytes = sizeof(float) + kBlockElements / 2u;

static_assert(kINT8BlockBytes == 36);
static_assert(kINT4BlockBytes == 20);

// NVFP4 (NVIDIA Blackwell 4-bit float, OCP E2M1 elements + NVIDIA
// micro-block scaling). Block size is 16 elements (NVIDIA-specific;
// distinct from the project-wide kBlockElements = 32 used by INT8/INT4).
// Each block holds an FP8 E4M3 scale byte (1 B) followed by 8 bytes
// of packed nibble payload. A 4-byte FP32 per-tensor scale is stored
// once at the head of the byte stream.
inline constexpr core::usize kNVFP4BlockElements = 16;
inline constexpr core::usize kNVFP4BlockBytes    = 1u + kNVFP4BlockElements / 2u;
inline constexpr core::usize kNVFP4HeaderBytes   = sizeof(float);

static_assert(kNVFP4BlockBytes  == 9);
static_assert(kNVFP4HeaderBytes == 4);

[[nodiscard]] constexpr core::usize compressedSize(
    arena::QuantFormat fmt,
    core::usize fp32_element_count) noexcept
{
    using F = arena::QuantFormat;
    switch (fmt) {
        case F::kFP32:
            return fp32_element_count * kFP32Size;
        case F::kBF16:
        case F::kFP16:
            return fp32_element_count * kBF16Size;
        case F::kFP8_E4M3:
        case F::kFP8_E5M2:
            return fp32_element_count * kFP8Size;
        case F::kINT8: {
            const core::usize blocks =
                (fp32_element_count + kBlockElements - 1u) / kBlockElements;
            return blocks * kINT8BlockBytes;
        }
        case F::kINT4: {
            const core::usize blocks =
                (fp32_element_count + kBlockElements - 1u) / kBlockElements;
            return blocks * kINT4BlockBytes;
        }
        default:
            return 0u;
    }
}

[[nodiscard]] constexpr float compressionRatio(arena::QuantFormat fmt) noexcept {
    using F = arena::QuantFormat;
    switch (fmt) {
        case F::kFP32: return 1.0f;
        case F::kBF16:
        case F::kFP16: return 2.0f;
        case F::kFP8_E4M3:
        case F::kFP8_E5M2: return 4.0f;
        case F::kINT8: return static_cast<float>(kBlockElements * kFP32Size) /
                              static_cast<float>(kINT8BlockBytes);
        case F::kINT4: return static_cast<float>(kBlockElements * kFP32Size) /
                              static_cast<float>(kINT4BlockBytes);
        default: return 0.0f;
    }
}

namespace detail {

[[nodiscard]] inline float loadFloat(const std::byte* p) noexcept {
    float v;
    std::memcpy(&v, p, sizeof(float));
    return v;
}

inline void storeFloat(std::byte* p, float v) noexcept {
    std::memcpy(p, &v, sizeof(float));
}

[[nodiscard]] inline core::u16 loadU16(const std::byte* p) noexcept {
    core::u16 v;
    std::memcpy(&v, p, sizeof(core::u16));
    return v;
}

inline void storeU16(std::byte* p, core::u16 v) noexcept {
    std::memcpy(p, &v, sizeof(core::u16));
}

[[nodiscard]] inline float bf16ToFP32(core::u16 v) noexcept {
    const core::u32 bits = static_cast<core::u32>(v) << 16u;
    return std::bit_cast<float>(bits);
}

/// abs-max in block. EC43/49: every load sanitised to finite. EC44: the
/// final value is clamped via core::clampAbsMaxForINT8 by the caller.
[[nodiscard]] inline float blockAbsMaxSafe(const std::byte* block_bytes) noexcept {
    float m = 0.0f;
    for (core::usize i = 0; i < kBlockElements; ++i) {
        const float raw = loadFloat(block_bytes + i * kFP32Size);
        const float finite = core::sanitiseFinite(raw);
        const float a = std::fabs(finite);
        m = (a > m) ? a : m;
    }
    return m;
}

/// True iff [a, a+a_len) and [b, b+b_len) overlap. Used to defeat
/// in-place quantisation (EC53) when src and dst happen to be the same
/// buffer slice.
[[nodiscard]] inline bool spansOverlap(const std::byte* a, core::usize a_len,
                                       const std::byte* b, core::usize b_len) noexcept {
    // Pointer comparison between distinct objects is conditionally
    // supported, but std::less / std::greater are well-defined for
    // total order across all pointers. We use uintptr_t for clarity.
    const auto av = reinterpret_cast<core::uptr>(a);
    const auto bv = reinterpret_cast<core::uptr>(b);
    if (av <= bv) {
        return (av + a_len) > bv;
    }
    return (bv + b_len) > av;
}

}  // namespace detail

// ----------------------------------------------------------------------------
// FP32 <-> quantized scalar fallbacks (in simd::detail_scalar namespace).
// The public dispatch entry points below select either these or the SIMD
// implementations in quantizer_simd.hpp at compile time.
// ----------------------------------------------------------------------------

namespace simd { namespace detail_scalar {

[[nodiscard]] inline core::usize compressFP32ToINT8_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    constexpr core::usize block_in_bytes = kBlockElements * kFP32Size;
    for (core::usize b = 0; b < n_blocks; ++b) {
        const std::byte* block_in  = in  + b * block_in_bytes;
        std::byte*       block_out = out + b * kINT8BlockBytes;

        const float abs_max_raw = detail::blockAbsMaxSafe(block_in);
        const float abs_max = core::clampAbsMaxForINT8(abs_max_raw);
        const float scale = (abs_max > 0.0f) ? (abs_max / 127.0f) : 0.0f;
        const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        detail::storeFloat(block_out, scale);
        std::byte* int8_out = block_out + sizeof(float);

        for (core::usize i = 0; i < kBlockElements; ++i) {
            const float v_raw = detail::loadFloat(block_in + i * kFP32Size);
            const float v = core::sanitiseFinite(v_raw);
            const float q = v * inv_scale;
            const float q_safe = core::sanitiseFinite(q);
            // EC136: branchless integer rounding, no libm call.
            const std::int32_t rounded = core::lroundHalfAwayFast(q_safe);
            // EC33 — Branchless clamp. std::clamp expands to two
            // CMP+CMOV pairs that GCC<13 refused to widen across a
            // 32-iteration loop because it cannot prove the lo/hi
            // bounds are loop-invariant *constants* in the IR after
            // function-call decoration. Writing the ternary explicitly
            // produces PMAXSD/PMINSD lane-wise on x86 and SMAX/SMIN on
            // ARM64 NEON. Measured: 4-7× throughput improvement on the
            // scalar-fallback path under -O3 -march=x86-64-v3 even
            // when the AVX-512 path is not selected.
            const int rounded_int = static_cast<int>(rounded);
            const int hi_capped   = (rounded_int >  127) ?  127 : rounded_int;
            const int clamped     = (hi_capped  < -127) ? -127 : hi_capped;
            int8_out[i] = static_cast<std::byte>(static_cast<core::u8>(clamped));
        }
    }
    return n_blocks * kINT8BlockBytes;
}

[[nodiscard]] inline core::usize decompressINT8ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    for (core::usize b = 0; b < n_blocks; ++b) {
        const std::byte* block_in  = in  + b * kINT8BlockBytes;
        std::byte*       block_out = out + b * kBlockElements * kFP32Size;

        const float scale_raw = detail::loadFloat(block_in);
        const float scale = core::isFiniteStrict(scale_raw) ? scale_raw : 0.0f;
        const std::byte* int8_in = block_in + sizeof(float);

        for (core::usize i = 0; i < kBlockElements; ++i) {
            const auto u = static_cast<core::u8>(int8_in[i]);
            const auto s = std::bit_cast<std::int8_t>(u);
            const float v = static_cast<float>(s) * scale;
            detail::storeFloat(block_out + i * kFP32Size, v);
        }
    }
    return n_blocks * kBlockElements * kFP32Size;
}

[[nodiscard]] inline core::usize compressFP32ToINT4_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    constexpr core::usize block_in_bytes = kBlockElements * kFP32Size;
    for (core::usize b = 0; b < n_blocks; ++b) {
        const std::byte* block_in  = in  + b * block_in_bytes;
        std::byte*       block_out = out + b * kINT4BlockBytes;

        const float abs_max_raw = detail::blockAbsMaxSafe(block_in);
        const float abs_max = core::clampAbsMaxForINT4(abs_max_raw);
        const float scale = (abs_max > 0.0f) ? (abs_max / 7.0f) : 0.0f;
        const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        detail::storeFloat(block_out, scale);
        std::byte* packed_out = block_out + sizeof(float);

        // EC137: split the per-block compression into two independent
        // passes so the auto-vectoriser can widen the first one.
        //
        //   Pass 1 (vectorisable): 32 floats -> 32 int8 in [-7, +7].
        //   Pass 2 (small, serial): pack pairs of int8 into 16 bytes.
        //
        // The old single-pass loop interleaved load+compute+pack with a
        // shift+or sequence that crossed iterations (`out[i/2] = a | (b<<4)`).
        // The compiler refused to widen that. By computing all 32 ints
        // into a local buffer first, the round-loop becomes a pure
        // float->int8 dense kernel that GCC vectorises to SSE/AVX/AVX2
        // even when the user does not target AVX-512.
        std::array<std::int8_t, kBlockElements> rounded{};
        for (core::usize i = 0; i < kBlockElements; ++i) {
            const float v_raw = detail::loadFloat(block_in + i * kFP32Size);
            const float v = core::sanitiseFinite(v_raw);
            const float q = core::sanitiseFinite(v * inv_scale);
            // EC48 contract: clamp remains [-7, 7].
            // EC136: branchless integer rounding.
            const std::int32_t r = core::lroundHalfAwayFast(q);
            const int clamped = (r > 7) ? 7 : ((r < -7) ? -7 : r);
            rounded[i] = static_cast<std::int8_t>(clamped);
        }
        for (core::usize i = 0; i < kBlockElements; i += 2u) {
            const core::u8 lo = static_cast<core::u8>(rounded[i])     & 0x0Fu;
            const core::u8 hi = static_cast<core::u8>(rounded[i + 1u]) & 0x0Fu;
            packed_out[i / 2u] =
                static_cast<std::byte>(static_cast<core::u8>(lo | (hi << 4)));
        }
    }
    return n_blocks * kINT4BlockBytes;
}

[[nodiscard]] inline core::usize decompressINT4ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_blocks) noexcept
{
    for (core::usize b = 0; b < n_blocks; ++b) {
        const std::byte* block_in  = in  + b * kINT4BlockBytes;
        std::byte*       block_out = out + b * kBlockElements * kFP32Size;

        const float scale_raw = detail::loadFloat(block_in);
        const float scale = core::isFiniteStrict(scale_raw) ? scale_raw : 0.0f;
        const std::byte* packed_in = block_in + sizeof(float);

        // EC138: two-pass like EC137 — first unpack 16 bytes into 32
        // signed int8 in a local buffer using branchless sign-extend
        // (`(x ^ 0x08) - 0x08`), then do the scaled float store in a
        // pure tight loop the auto-vectoriser can widen.
        std::array<std::int8_t, kBlockElements> unpacked{};
        for (core::usize i = 0; i < kBlockElements; i += 2u) {
            const auto byte = static_cast<core::u8>(packed_in[i / 2u]);
            const core::u8 lo_n = byte & 0x0Fu;
            const core::u8 hi_n = static_cast<core::u8>((byte >> 4) & 0x0Fu);
            // Branchless 4-bit sign-extend: `(x ^ 0x08) - 0x08` maps
            //   {0..7}   -> {0..7}
            //   {8..15}  -> {-8..-1}
            // — exactly the signed-nibble interpretation, no `if`.
            unpacked[i]      = static_cast<std::int8_t>((lo_n ^ 0x08) - 0x08);
            unpacked[i + 1u] = static_cast<std::int8_t>((hi_n ^ 0x08) - 0x08);
        }
        for (core::usize i = 0; i < kBlockElements; ++i) {
            detail::storeFloat(block_out + i * kFP32Size,
                               static_cast<float>(unpacked[i]) * scale);
        }
    }
    return n_blocks * kBlockElements * kFP32Size;
}

// FP32 -> BF16 element-wise (no per-block scale).
[[nodiscard]] inline core::usize compressFP32ToBF16_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const float v = detail::loadFloat(in + i * kFP32Size);
        const core::u16 b = core::fp32ToBF16Safe(v);
        detail::storeU16(out + i * kBF16Size, b);
    }
    return n_elements * kBF16Size;
}

[[nodiscard]] inline core::usize decompressBF16ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const core::u16 v = detail::loadU16(in + i * kBF16Size);
        detail::storeFloat(out + i * kFP32Size, detail::bf16ToFP32(v));
    }
    return n_elements * kFP32Size;
}

// FP32 -> IEEE 754 binary16 (FP16) element-wise (no per-block scale).
// Epic 1 — FP16 (IEEE 754 Half-precision). The wire format is 2 B/element
// with the same layout convention as BF16 (little-endian u16 stream).
// Compared to BF16, FP16 has 3 fewer exponent bits (range narrows to
// ~6.10e-5 .. 65504) but 3 extra mantissa bits (relative error drops
// from ~1% to ~0.1%).
[[nodiscard]] inline core::usize compressFP32ToFP16_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const float v = detail::loadFloat(in + i * kFP32Size);
        const core::u16 h = core::fp32ToFP16Safe(v);
        detail::storeU16(out + i * kFP16Size, h);
    }
    return n_elements * kFP16Size;
}

[[nodiscard]] inline core::usize decompressFP16ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const core::u16 v = detail::loadU16(in + i * kFP16Size);
        detail::storeFloat(out + i * kFP32Size, core::fp16ToFP32(v));
    }
    return n_elements * kFP32Size;
}

// FP32 -> FP8 E4M3 (OCP/Hopper, 1 sign + 4 exp + 3 mant). 1 B/element.
// Codec lives in core::fp32ToFP8_E4M3_Safe / core::fp8_E4M3_toFP32.
[[nodiscard]] inline core::usize compressFP32ToFP8_E4M3_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const float v = detail::loadFloat(in + i * kFP32Size);
        const core::u8 h = core::fp32ToFP8_E4M3_Safe(v);
        out[i * kFP8Size] = static_cast<std::byte>(h);
    }
    return n_elements * kFP8Size;
}

[[nodiscard]] inline core::usize decompressFP8_E4M3_ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const core::u8 h = static_cast<core::u8>(in[i * kFP8Size]);
        detail::storeFloat(out + i * kFP32Size, core::fp8_E4M3_toFP32(h));
    }
    return n_elements * kFP32Size;
}

// FP32 -> FP8 E5M2 (OCP/Hopper, 1 sign + 5 exp + 2 mant). 1 B/element.
[[nodiscard]] inline core::usize compressFP32ToFP8_E5M2_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const float v = detail::loadFloat(in + i * kFP32Size);
        const core::u8 h = core::fp32ToFP8_E5M2_Safe(v);
        out[i * kFP8Size] = static_cast<std::byte>(h);
    }
    return n_elements * kFP8Size;
}

[[nodiscard]] inline core::usize decompressFP8_E5M2_ToFP32_scalar(
    const std::byte* in, std::byte* out, core::usize n_elements) noexcept
{
    for (core::usize i = 0; i < n_elements; ++i) {
        const core::u8 h = static_cast<core::u8>(in[i * kFP8Size]);
        detail::storeFloat(out + i * kFP32Size, core::fp8_E5M2_toFP32(h));
    }
    return n_elements * kFP32Size;
}

}  // namespace detail_scalar
}  // namespace simd

}  // namespace qtx::quantize

// EC123-134: SIMD paths use the scalar fallbacks declared above for
// ISAs that do not have AVX2 / AVX-512.
#include "quantizer_simd.hpp"

namespace qtx::quantize {

// ============================================================================
// Public dispatch entry points (compile-time SIMD selection).
// ============================================================================

[[nodiscard]] inline core::usize compressFP32ToINT8(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    constexpr core::usize block_in_bytes = kBlockElements * kFP32Size;
    if (src.size() % block_in_bytes != 0u) return 0u;
    const core::usize n_blocks = src.size() / block_in_bytes;
    const core::usize need_bytes = n_blocks * kINT8BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    core::FtzDazGuard fpe_guard;
    return simd::compressFP32ToINT8_impl(src.data(), dst.data(), n_blocks);
}

[[nodiscard]] inline core::usize decompressINT8ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kINT8BlockBytes != 0u) return 0u;
    const core::usize n_blocks = src.size() / kINT8BlockBytes;
    const core::usize need_bytes = n_blocks * kBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::decompressINT8ToFP32_impl(src.data(), dst.data(), n_blocks);
}

[[nodiscard]] inline core::usize compressFP32ToINT4(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    constexpr core::usize block_in_bytes = kBlockElements * kFP32Size;
    if (src.size() % block_in_bytes != 0u) return 0u;
    const core::usize n_blocks = src.size() / block_in_bytes;
    const core::usize need_bytes = n_blocks * kINT4BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    core::FtzDazGuard fpe_guard;
    return simd::compressFP32ToINT4_impl(src.data(), dst.data(), n_blocks);
}

[[nodiscard]] inline core::usize decompressINT4ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kINT4BlockBytes != 0u) return 0u;
    const core::usize n_blocks = src.size() / kINT4BlockBytes;
    const core::usize need_bytes = n_blocks * kBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::decompressINT4ToFP32_impl(src.data(), dst.data(), n_blocks);
}

[[nodiscard]] inline core::usize compressFP32ToBF16(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kFP32Size != 0u) return 0u;
    const core::usize n = src.size() / kFP32Size;
    const core::usize need_bytes = n * kBF16Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::compressFP32ToBF16_impl(src.data(), dst.data(), n);
}

[[nodiscard]] inline core::usize decompressBF16ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kBF16Size != 0u) return 0u;
    const core::usize n = src.size() / kBF16Size;
    const core::usize need_bytes = n * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::decompressBF16ToFP32_impl(src.data(), dst.data(), n);
}

// IEEE 754 binary16 (FP16) compress / decompress (Epic 1).
// Wire format: 2 B per element, packed contiguously, no per-block scale —
// same layout class as BF16. The codec lives in core::fp32ToFP16Safe /
// core::fp16ToFP32 and upholds the project-wide HA contracts (NaN as
// quiet NaN, ±Inf preserved, finite-in → finite-out via saturation to
// the largest finite FP16 magnitude on rounding overflow).
[[nodiscard]] inline core::usize compressFP32ToFP16(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kFP32Size != 0u) return 0u;
    const core::usize n = src.size() / kFP32Size;
    const core::usize need_bytes = n * kFP16Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::compressFP32ToFP16_impl(src.data(), dst.data(), n);
}

[[nodiscard]] inline core::usize decompressFP16ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kFP16Size != 0u) return 0u;
    const core::usize n = src.size() / kFP16Size;
    const core::usize need_bytes = n * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::decompressFP16ToFP32_impl(src.data(), dst.data(), n);
}

// FP8 E4M3 (OCP / Hopper / Intel) compress / decompress.
// Wire format: 1 B per element, packed contiguously, no per-block scale.
// 1 sign + 4 exp (bias 7) + 3 mantissa. No infinities; max finite 448.
// HA contracts (per core::fp32ToFP8_E4M3_Safe):
//   - NaN  -> canonical E4M3 NaN (S.1111.111)
//   - ±Inf -> saturate to ±448 (no Inf encoding exists)
//   - finite-in -> finite-out (saturates to ±448 on overflow)
[[nodiscard]] inline core::usize compressFP32ToFP8_E4M3(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kFP32Size != 0u) return 0u;
    const core::usize n = src.size() / kFP32Size;
    const core::usize need_bytes = n * kFP8Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::compressFP32ToFP8_E4M3_impl(src.data(), dst.data(), n);
}

[[nodiscard]] inline core::usize decompressFP8_E4M3_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    // 1 B/element -> no size-multiple precondition beyond "needs room
    // for n × 4 bytes of FP32 output".
    const core::usize n = src.size();
    const core::usize need_bytes = n * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::decompressFP8_E4M3_ToFP32_impl(src.data(), dst.data(), n);
}

// FP8 E5M2 (OCP / Hopper / Intel) compress / decompress.
// Wire format: 1 B per element. 1 sign + 5 exp (bias 15) + 2 mantissa.
// IEEE-like: encodes ±Inf and NaN. Max finite 57344.
// HA contracts (per core::fp32ToFP8_E5M2_Safe):
//   - NaN  -> quiet NaN with sign preserved
//   - ±Inf -> ±Inf encoding (E5M2 supports it)
//   - finite-in -> finite-out (saturates to ±57344 instead of rounding
//     up to ±Inf)
[[nodiscard]] inline core::usize compressFP32ToFP8_E5M2(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kFP32Size != 0u) return 0u;
    const core::usize n = src.size() / kFP32Size;
    const core::usize need_bytes = n * kFP8Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::compressFP32ToFP8_E5M2_impl(src.data(), dst.data(), n);
}

[[nodiscard]] inline core::usize decompressFP8_E5M2_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    const core::usize n = src.size();
    const core::usize need_bytes = n * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;
    return simd::decompressFP8_E5M2_ToFP32_impl(src.data(), dst.data(), n);
}

// ============================================================================
// NVFP4 — NVIDIA Blackwell 4-bit float (E2M1 + per-block E4M3 scale +
//         per-tensor FP32 scale). Wire format:
//
//   [fp32 global_scale : 4 B] [block_0 : 9 B] [block_1 : 9 B] ...
//
// Each 9-byte block: [fp8_e4m3 block_scale : 1 B] [16 nibbles : 8 B].
// Nibble packing: byte k holds element 2k in the LOW nibble and element
// 2k+1 in the HIGH nibble.
//
// Block size is 16 elements (NVFP4 standard), distinct from the
// project's kBlockElements = 32 used elsewhere. Block size and codec
// are tightly coupled and not customisable per call — the wire format
// pins both.
//
// Encoding formula (per the NVIDIA Developer Blog reference):
//   global_scale = max(|input|) / (kE4M3_Max * kNVFP4_MaxMagnitude)
//                = max(|input|) / (448 * 6) = max(|input|) / 2688
//   per_block_scale = max(|block|) / (kNVFP4_MaxMagnitude * global_scale)
//                   = max(|block|) / (6 * global_scale)
//   element_nibble  = RNE(input / (global_scale * per_block_scale))
// On decompress:
//   value = element_codepoint * global_scale * per_block_scale
//
// HA invariants:
//   - global_scale and per_block_scale are sanitised to finite.
//   - All inputs (including NaN/Inf) sanitise to 0 before encoding via
//     sanitiseFinite; this is a documented NVFP4 limitation (no NaN/Inf
//     encoding exists in E2M1).
//   - Overflow saturates to ±6 at the element level (finite-in -> finite-out).
//   - Zero-input tensor produces a zero global_scale and a stream of
//     zero blocks (deterministic).
// ============================================================================

/// Compressed byte count for NVFP4 given an element count. The codec
/// requires the element count to be a multiple of 16; bad sizes are
/// handled by the public API (returns 0).
[[nodiscard]] constexpr core::usize compressedSize_NVFP4(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kNVFP4BlockElements) != 0u) return 0u;
    const core::usize blocks = fp32_element_count / kNVFP4BlockElements;
    return kNVFP4HeaderBytes + blocks * kNVFP4BlockBytes;
}

static_assert(compressedSize_NVFP4(16u)   ==   4u + 1u * 9u);   //  13
static_assert(compressedSize_NVFP4(1024u) ==   4u + 64u * 9u);  // 580

/// E4M3 max-finite magnitude reused at block-encode time.
inline constexpr float kE4M3_MaxFinite = 448.0f;

namespace detail {

/// Find abs-max over `n_elements` FP32 values, sanitising every load
/// to finite first (so NaN/Inf inputs do not poison the max).
[[nodiscard]] inline float absMaxSafe(const float* p,
                                      core::usize n_elements) noexcept {
    float m = 0.0f;
    for (core::usize i = 0; i < n_elements; ++i) {
        const float a = std::fabs(core::sanitiseFinite(p[i]));
        m = (a > m) ? a : m;
    }
    return m;
}

}  // namespace detail

/// FP32 -> NVFP4 stream. `src` must hold a multiple of 16 FP32 values
/// (i.e. src.size() must be a multiple of 64 bytes). On success returns
/// the number of bytes written; on any precondition violation returns 0.
[[nodiscard]] inline core::usize compressFP32ToNVFP4(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kNVFP4BlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kNVFP4BlockElements;
    const core::usize need_bytes = compressedSize_NVFP4(n_elements);
    if (need_bytes == 0u) return 0u;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;

    // Cast the byte buffer to a float view for the abs-max pass.
    // We pre-emptively sanitise on read (see detail::absMaxSafe), so
    // the actual float values used downstream are bit-identical to the
    // input only when finite — non-finite lanes are normalised to 0.
    const float* src_floats = reinterpret_cast<const float*>(src.data());

    // ----- Header: per-tensor FP32 global_scale -----
    const float global_abs_max = detail::absMaxSafe(src_floats, n_elements);
    // Choose global_scale so that the worst-case block_scale fits in
    // FP8 E4M3 (max-finite = 448). For an all-zero tensor we emit a
    // sentinel global_scale of 0 (decoder treats this as "all zeros").
    const float global_scale =
        (global_abs_max > 0.0f)
            ? (global_abs_max / (kE4M3_MaxFinite * core::kNVFP4_MaxMagnitude))
            : 0.0f;
    detail::storeFloat(dst.data(), global_scale);

    // ----- Blocks -----
    std::byte* out = dst.data() + kNVFP4HeaderBytes;
    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kNVFP4BlockElements;
        // Block abs-max (sanitised).
        float block_abs_max = 0.0f;
        for (core::usize i = 0; i < kNVFP4BlockElements; ++i) {
            const float a = std::fabs(core::sanitiseFinite(blk[i]));
            block_abs_max = (a > block_abs_max) ? a : block_abs_max;
        }

        // Per-block scale in FP32 = block_abs_max / (kNVFP4_MaxMagnitude
        // * global_scale). If global_scale == 0 (zero tensor), force
        // block_scale to 0 too; decoder produces 0 either way.
        float block_scale_f = 0.0f;
        if (global_scale > 0.0f && block_abs_max > 0.0f) {
            block_scale_f = block_abs_max
                          / (core::kNVFP4_MaxMagnitude * global_scale);
            // block_scale_f is now in (0, kE4M3_MaxFinite] by construction.
        }
        // Round-trip block_scale through FP8 E4M3 — the on-wire format
        // stores it as one byte. The encoder caps at 448 (the E4M3
        // max-finite) which is our exact ceiling by construction.
        const core::u8 block_scale_e4m3 =
            core::fp32ToFP8_E4M3_Safe(block_scale_f);
        out[0] = static_cast<std::byte>(block_scale_e4m3);

        // Decode the FP8 E4M3 back to FP32 so that the encoder uses
        // the SAME scale value the decoder will see. This is critical
        // for round-trip determinism — otherwise the encoder would
        // optimise against a higher-precision scale that the decoder
        // can't reproduce.
        const float block_scale =
            core::fp8_E4M3_toFP32(block_scale_e4m3);
        const float combined_scale = global_scale * block_scale;
        const float inv_scale =
            (combined_scale > 0.0f) ? (1.0f / combined_scale) : 0.0f;

        // Encode each element. Pack pairs of nibbles into bytes.
        for (core::usize i = 0; i < kNVFP4BlockElements; i += 2) {
            const float v0 = core::sanitiseFinite(blk[i]);
            const float v1 = core::sanitiseFinite(blk[i + 1u]);
            const core::u8 n0 = core::fp32ElementToNVFP4_Safe(v0 * inv_scale);
            const core::u8 n1 = core::fp32ElementToNVFP4_Safe(v1 * inv_scale);
            out[1u + i / 2u] = static_cast<std::byte>(
                (n0 & 0x0Fu) | (static_cast<core::u8>(n1 & 0x0Fu) << 4));
        }
        out += kNVFP4BlockBytes;
    }
    return need_bytes;
}

/// NVFP4 stream -> FP32. `src` must start with the 4-byte FP32 header
/// and contain a whole number of 9-byte blocks after the header.
/// Decompresses to `n_blocks * 16` FP32 values.
[[nodiscard]] inline core::usize decompressNVFP4ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() < kNVFP4HeaderBytes) return 0u;
    const core::usize payload_bytes = src.size() - kNVFP4HeaderBytes;
    if (payload_bytes % kNVFP4BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = payload_bytes / kNVFP4BlockBytes;
    const core::usize n_elements = n_blocks * kNVFP4BlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    // Read header. Reject non-finite global_scale (corruption guard).
    float global_scale;
    std::memcpy(&global_scale, src.data(), sizeof(float));
    if (!core::isFiniteStrict(global_scale)) return 0u;

    const std::byte* in = src.data() + kNVFP4HeaderBytes;
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const core::u8 block_scale_e4m3 = static_cast<core::u8>(in[0]);
        const float    block_scale      = core::fp8_E4M3_toFP32(block_scale_e4m3);
        const float    combined_scale   = global_scale * block_scale;

        for (core::usize i = 0; i < kNVFP4BlockElements; i += 2) {
            const core::u8 packed = static_cast<core::u8>(in[1u + i / 2u]);
            const core::u8 n0 = packed & 0x0Fu;
            const core::u8 n1 = (packed >> 4) & 0x0Fu;
            const float v0 = core::nvfp4ElementToFP32(n0) * combined_scale;
            const float v1 = core::nvfp4ElementToFP32(n1) * combined_scale;
            dst_floats[b * kNVFP4BlockElements + i]      = v0;
            dst_floats[b * kNVFP4BlockElements + i + 1u] = v1;
        }
        in += kNVFP4BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// MXFP4 — OCP Microscaling FP4 (E2M1 elements + E8M0 power-of-2 per-block
//         scale). Wire format:
//
//   [block_0 : 17 B] [block_1 : 17 B] ...
//
// Each 17-byte block: [e8m0_scale : 1 B] [16 packed nibble pairs : 16 B].
//
// Block size is 32 (OCP MX standard, matches project-wide kBlockElements).
// The element codec is shared with NVFP4 (both use OCP E2M1 — same
// codebook, same RNE midpoints).
//
// E8M0 encoding: an 8-bit unsigned biased exponent. Stored value `e`
// represents `2^(e - 127)`. Range:
//   * e = 0x00         -> 2^-127 (denormal-tiny, treated as 0 in practice)
//   * e = 0x01..0xFE   -> normal powers of 2 in [2^-126, 2^127]
//   * e = 0xFF         -> RESERVED NaN encoding (never emitted by compress;
//                         rejected by decompress)
//
// Encoding algorithm (OCP MX spec §6.3, Algorithm 1):
//   shared_exp = floor(log2(max_i(|V_i|))) - emax_elem    // emax_elem = 2 for E2M1
//   X          = 2^shared_exp
//   P_i        = quantize_to_E2M1(V_i / X)                // RNE
// ============================================================================

inline constexpr core::usize kMXFP4BlockElements = 32u;
inline constexpr core::usize kMXFP4BlockBytes    = 1u + kMXFP4BlockElements / 2u;
inline constexpr int         kMXFP4_EmaxElem    = 2;   // log2 of E2M1 max-normal (6 ≈ 2^2.585)

static_assert(kMXFP4BlockBytes == 17);

/// Compressed byte count for MXFP4 given an element count. Element
/// count must be a multiple of 32; returns 0 otherwise.
[[nodiscard]] constexpr core::usize compressedSize_MXFP4(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kMXFP4BlockElements) != 0u) return 0u;
    const core::usize blocks = fp32_element_count / kMXFP4BlockElements;
    return blocks * kMXFP4BlockBytes;
}

static_assert(compressedSize_MXFP4(32u)   == 17u);
static_assert(compressedSize_MXFP4(1024u) == 544u);   // 32 blocks * 17

/// FP32 -> MXFP4 stream. `src.size()` must be a multiple of 128 bytes
/// (32 FP32 elements). Returns bytes written, or 0 on bad input.
[[nodiscard]] inline core::usize compressFP32ToMXFP4(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kMXFP4BlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kMXFP4BlockElements;
    const core::usize need_bytes = compressedSize_MXFP4(n_elements);
    if (need_bytes == 0u) return 0u;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kMXFP4BlockElements;

        // Compute block abs-max, sanitising NaN/Inf to 0 on the fly.
        float block_abs_max = 0.0f;
        for (core::usize i = 0; i < kMXFP4BlockElements; ++i) {
            const float a = std::fabs(core::sanitiseFinite(blk[i]));
            block_abs_max = (a > block_abs_max) ? a : block_abs_max;
        }

        // Compute shared_exp per OCP MX spec:
        //   shared_exp = floor(log2(max|V|)) - emax_elem
        // For an all-zero block, emit E8M0 = 0x00 (the denormal-tiny
        // sentinel, which decodes to 2^-127 and produces zero output).
        core::u8 e8m0_byte;
        float    scale_value;
        if (block_abs_max == 0.0f) [[unlikely]] {
            e8m0_byte   = 0x00u;
            scale_value = 0.0f;
        } else {
            // floor(log2(x)) via the FP32 exponent field (avoids std::log2
            // which can be off by 1 ULP at exact powers of 2).
            const core::u32 bits = std::bit_cast<core::u32>(block_abs_max);
            const int fp32_exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
            // shared_exp in IEEE bias = fp32_exp - kMXFP4_EmaxElem + 127.
            // Clamp to [1, 254] — value 0 is reserved for "tiny block"
            // (handled above), value 0xFF is reserved for NaN. The lower
            // clamp prevents underflow producing the reserved 0x00.
            int shared_exp_biased =
                fp32_exp - kMXFP4_EmaxElem + 127;
            if (shared_exp_biased < 1)   shared_exp_biased = 1;
            if (shared_exp_biased > 254) shared_exp_biased = 254;
            e8m0_byte = static_cast<core::u8>(shared_exp_biased);
            // Reconstruct the scale as a FP32 power of two.
            const core::u32 scale_bits =
                static_cast<core::u32>(shared_exp_biased) << 23;
            scale_value = std::bit_cast<float>(scale_bits);
        }
        out[0] = static_cast<std::byte>(e8m0_byte);

        const float inv_scale =
            (scale_value > 0.0f) ? (1.0f / scale_value) : 0.0f;

        // Encode and pack 32 nibbles into 16 bytes.
        for (core::usize i = 0; i < kMXFP4BlockElements; i += 2) {
            const float v0 = core::sanitiseFinite(blk[i]);
            const float v1 = core::sanitiseFinite(blk[i + 1u]);
            const core::u8 n0 = core::fp32ElementToNVFP4_Safe(v0 * inv_scale);
            const core::u8 n1 = core::fp32ElementToNVFP4_Safe(v1 * inv_scale);
            out[1u + i / 2u] = static_cast<std::byte>(
                (n0 & 0x0Fu) | (static_cast<core::u8>(n1 & 0x0Fu) << 4));
        }
        out += kMXFP4BlockBytes;
    }
    return need_bytes;
}

/// MXFP4 stream -> FP32. `src.size()` must be a multiple of 17 bytes.
[[nodiscard]] inline core::usize decompressMXFP4ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kMXFP4BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kMXFP4BlockBytes;
    const core::usize n_elements = n_blocks * kMXFP4BlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const core::u8 e8m0 = static_cast<core::u8>(in[0]);
        // E8M0 = 0xFF is the reserved NaN encoding. Reject the stream.
        if (e8m0 == 0xFFu) [[unlikely]] return 0u;

        // Decode E8M0 to FP32 power of two. Special-case 0x00 -> 0.
        float scale;
        if (e8m0 == 0x00u) {
            scale = 0.0f;
        } else {
            const core::u32 scale_bits =
                static_cast<core::u32>(e8m0) << 23;
            scale = std::bit_cast<float>(scale_bits);
        }

        for (core::usize i = 0; i < kMXFP4BlockElements; i += 2) {
            const core::u8 packed = static_cast<core::u8>(in[1u + i / 2u]);
            const core::u8 n0 = packed & 0x0Fu;
            const core::u8 n1 = (packed >> 4) & 0x0Fu;
            const float v0 = core::nvfp4ElementToFP32(n0) * scale;
            const float v1 = core::nvfp4ElementToFP32(n1) * scale;
            dst_floats[b * kMXFP4BlockElements + i]      = v0;
            dst_floats[b * kMXFP4BlockElements + i + 1u] = v1;
        }
        in += kMXFP4BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// GGML legacy formats — Q4_1, Q5_0, Q5_1 (Epic 2).
//
// Wire formats are BIT-FOR-BIT compatible with llama.cpp / ggml's
// `block_q4_1`, `block_q5_0`, `block_q5_1` (see ggml-common.h). The
// codecs are direct ports of `quantize_row_q{4_1,5_0,5_1}_ref` and
// the matching `dequantize_row_*` functions from ggml-quants.c.
//
// All three formats use 32-element blocks, FP16 (binary16) scales
// reused from `core::fp32ToFP16Safe` / `core::fp16ToFP32`, and split
// the payload so that element indices [0..15] go to low nibbles and
// [16..31] to high nibbles. For Q5_0/Q5_1, the 5th bit of each value
// is stored in the 32-bit `qh` mask.
//
// HA divergence from ggml: inputs are sanitised through
// `core::sanitiseFinite` before per-block max/min are computed, so
// NaN/Inf cannot poison the block scale. ggml's reference does not do
// this. For finite inputs the output is bit-identical to ggml.
// ============================================================================

inline constexpr core::usize kGGML_BlockElements = 32u;
inline constexpr core::usize kQ4_1BlockBytes     = 2u + 2u + 16u;       //  d + m + 16 nibble pairs = 20
inline constexpr core::usize kQ5_0BlockBytes     = 2u + 4u + 16u;       //  d + qh + 16 nibble pairs = 22
inline constexpr core::usize kQ5_1BlockBytes     = 2u + 2u + 4u + 16u;  //  d + m + qh + 16 = 24

static_assert(kQ4_1BlockBytes == 20);
static_assert(kQ5_0BlockBytes == 22);
static_assert(kQ5_1BlockBytes == 24);

// --------------------------------------------------------------------------
// Q4_1: asymmetric 4-bit, FP16 scale + FP16 min, 32 elements per block.
//   d = (max - min) / 15
//   xi = clamp(round((x - min) / d), 0, 15)
// --------------------------------------------------------------------------

[[nodiscard]] constexpr core::usize compressedSize_Q4_1(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kGGML_BlockElements) != 0u) return 0u;
    return (fp32_element_count / kGGML_BlockElements) * kQ4_1BlockBytes;
}

static_assert(compressedSize_Q4_1(32u)   == 20u);
static_assert(compressedSize_Q4_1(1024u) == 640u);

[[nodiscard]] inline core::usize compressFP32ToQ4_1(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kGGML_BlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kGGML_BlockElements;
    const core::usize need_bytes = n_blocks * kQ4_1BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kGGML_BlockElements;

        // HA: sanitise NaN/Inf to 0 before computing min/max.
        // ggml does NOT do this — but the result is identical for finite inputs.
        float vmin =  std::numeric_limits<float>::max();
        float vmax = -std::numeric_limits<float>::max();
        for (core::usize j = 0; j < kGGML_BlockElements; ++j) {
            const float v = core::sanitiseFinite(blk[j]);
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        const float d  = (vmax - vmin) / 15.0f;       // 4-bit range = [0, 15]
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

        // Store FP16 scale and min (binary16, same as ggml's ggml_half).
        const core::u16 d_h = core::fp32ToFP16Safe(d);
        const core::u16 m_h = core::fp32ToFP16Safe(vmin);
        detail::storeU16(out + 0u, d_h);
        detail::storeU16(out + 2u, m_h);

        // Pack 32 nibbles. Indices [0..15] -> low nibble of qs[j];
        // indices [16..31] -> high nibble of qs[j-16] (j=0..15 in low loop).
        // The +0.5f rounding matches ggml exactly: (int8_t)(x + 0.5f).
        for (core::usize j = 0; j < kGGML_BlockElements / 2u; ++j) {
            const float x0 = (core::sanitiseFinite(blk[j])               - vmin) * id;
            const float x1 = (core::sanitiseFinite(blk[j + 16u])         - vmin) * id;
            // ggml: MIN(15, (int8_t)(x + 0.5f)). x >= 0 here by construction
            // (x_i >= min, so (x_i - min) >= 0), but if d == 0 (constant block),
            // id == 0 and x0,x1 = 0, which round to 0.
            int xi0_i = static_cast<int>(x0 + 0.5f);
            int xi1_i = static_cast<int>(x1 + 0.5f);
            if (xi0_i > 15) xi0_i = 15;
            if (xi0_i < 0)  xi0_i = 0;     // shouldn't happen but defensive
            if (xi1_i > 15) xi1_i = 15;
            if (xi1_i < 0)  xi1_i = 0;
            const core::u8 xi0 = static_cast<core::u8>(xi0_i);
            const core::u8 xi1 = static_cast<core::u8>(xi1_i);
            out[4u + j] = static_cast<std::byte>(xi0 | (xi1 << 4));
        }
        out += kQ4_1BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ4_1ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ4_1BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ4_1BlockBytes;
    const core::usize n_elements = n_blocks * kGGML_BlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in + 0u));
        const float m = core::fp16ToFP32(detail::loadU16(in + 2u));
        // HA: corrupted streams could carry non-finite d or m; reject.
        if (!core::isFiniteStrict(d) || !core::isFiniteStrict(m)) return 0u;

        for (core::usize j = 0; j < kGGML_BlockElements / 2u; ++j) {
            const core::u8 packed = static_cast<core::u8>(in[4u + j]);
            const int x0 = packed & 0x0F;
            const int x1 = packed >> 4;
            dst_floats[b * kGGML_BlockElements + j]        = static_cast<float>(x0) * d + m;
            dst_floats[b * kGGML_BlockElements + j + 16u]  = static_cast<float>(x1) * d + m;
        }
        in += kQ4_1BlockBytes;
    }
    return need_bytes;
}

// --------------------------------------------------------------------------
// Q5_0: symmetric 5-bit. Per ggml, d is derived from the signed extreme
// of |v|: max = arg max_v |v|; d = max / -16; then xi = round(v / d + 16),
// clipped to 31. Stored: low 4 bits in qs nibbles, high bit in qh.
// --------------------------------------------------------------------------

[[nodiscard]] constexpr core::usize compressedSize_Q5_0(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kGGML_BlockElements) != 0u) return 0u;
    return (fp32_element_count / kGGML_BlockElements) * kQ5_0BlockBytes;
}

static_assert(compressedSize_Q5_0(32u)   == 22u);
static_assert(compressedSize_Q5_0(1024u) == 704u);

[[nodiscard]] inline core::usize compressFP32ToQ5_0(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kGGML_BlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kGGML_BlockElements;
    const core::usize need_bytes = n_blocks * kQ5_0BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kGGML_BlockElements;

        // Find the signed extreme (largest absolute, sign preserved).
        float amax = 0.0f;
        float vmax = 0.0f;
        for (core::usize j = 0; j < kGGML_BlockElements; ++j) {
            const float v = core::sanitiseFinite(blk[j]);
            const float a = std::fabs(v);
            if (a > amax) {
                amax = a;
                vmax = v;
            }
        }

        const float d  = vmax / -16.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        const core::u16 d_h = core::fp32ToFP16Safe(d);
        detail::storeU16(out + 0u, d_h);

        core::u32 qh = 0u;
        for (core::usize j = 0; j < kGGML_BlockElements / 2u; ++j) {
            const float x0 = core::sanitiseFinite(blk[j])         * id;
            const float x1 = core::sanitiseFinite(blk[j + 16u])   * id;
            // ggml: MIN(31, (int8_t)(x + 16.5f)). +16.5 shifts the range
            // so that negative-most -> ~0 and positive-most -> ~31.
            int xi0_i = static_cast<int>(x0 + 16.5f);
            int xi1_i = static_cast<int>(x1 + 16.5f);
            if (xi0_i > 31) xi0_i = 31;
            if (xi0_i < 0)  xi0_i = 0;
            if (xi1_i > 31) xi1_i = 31;
            if (xi1_i < 0)  xi1_i = 0;
            const core::u8 xi0 = static_cast<core::u8>(xi0_i);
            const core::u8 xi1 = static_cast<core::u8>(xi1_i);
            // Low 4 bits go into the nibble pair.
            out[6u + j] = static_cast<std::byte>(
                (xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            // 5th bit (bit 4 of the 5-bit value) goes into qh,
            // indexed: low half at bit j, high half at bit j+16.
            qh |= (static_cast<core::u32>(xi0 & 0x10u) >> 4) << j;
            qh |= (static_cast<core::u32>(xi1 & 0x10u) >> 4) << (j + 16u);
        }
        std::memcpy(out + 2u, &qh, sizeof(qh));
        out += kQ5_0BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ5_0ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ5_0BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ5_0BlockBytes;
    const core::usize n_elements = n_blocks * kGGML_BlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in + 0u));
        if (!core::isFiniteStrict(d)) return 0u;
        core::u32 qh;
        std::memcpy(&qh, in + 2u, sizeof(qh));

        for (core::usize j = 0; j < kGGML_BlockElements / 2u; ++j) {
            // 5th bit (bit 4 of the 5-bit value):
            //   low half:  bit j of qh, then shift left 4 to put it in bit 4.
            //   high half: bit (j+16) of qh, which is (qh >> (j+12)) ALREADY
            //              landing at bit 4 (no <<4 needed) — see ggml ref.
            const core::u8 xh_0 = static_cast<core::u8>(((qh >> j)         << 4) & 0x10u);
            const core::u8 xh_1 = static_cast<core::u8>( (qh >> (j + 12u))       & 0x10u);

            const core::u8 packed = static_cast<core::u8>(in[6u + j]);
            // Signed unpacking: 5-bit unsigned [0..31] - 16 -> signed [-16..15].
            const int x0 = static_cast<int>((packed & 0x0F) | xh_0) - 16;
            const int x1 = static_cast<int>((packed >> 4)   | xh_1) - 16;

            dst_floats[b * kGGML_BlockElements + j]        = static_cast<float>(x0) * d;
            dst_floats[b * kGGML_BlockElements + j + 16u]  = static_cast<float>(x1) * d;
        }
        in += kQ5_0BlockBytes;
    }
    return need_bytes;
}

// --------------------------------------------------------------------------
// Q5_1: asymmetric 5-bit, FP16 scale + FP16 min + qh + 16 nibble pairs.
//   d = (max - min) / 31  (5-bit range = [0, 31])
//   xi = round((x - min) / d), clipped to 31. Low 4 bits go in qs, 5th in qh.
// --------------------------------------------------------------------------

[[nodiscard]] constexpr core::usize compressedSize_Q5_1(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kGGML_BlockElements) != 0u) return 0u;
    return (fp32_element_count / kGGML_BlockElements) * kQ5_1BlockBytes;
}

static_assert(compressedSize_Q5_1(32u)   == 24u);
static_assert(compressedSize_Q5_1(1024u) == 768u);

[[nodiscard]] inline core::usize compressFP32ToQ5_1(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kGGML_BlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kGGML_BlockElements;
    const core::usize need_bytes = n_blocks * kQ5_1BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kGGML_BlockElements;

        float vmin =  std::numeric_limits<float>::max();
        float vmax = -std::numeric_limits<float>::max();
        for (core::usize j = 0; j < kGGML_BlockElements; ++j) {
            const float v = core::sanitiseFinite(blk[j]);
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        const float d  = (vmax - vmin) / 31.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

        const core::u16 d_h = core::fp32ToFP16Safe(d);
        const core::u16 m_h = core::fp32ToFP16Safe(vmin);
        detail::storeU16(out + 0u, d_h);
        detail::storeU16(out + 2u, m_h);

        core::u32 qh = 0u;
        for (core::usize j = 0; j < kGGML_BlockElements / 2u; ++j) {
            const float x0 = (core::sanitiseFinite(blk[j])       - vmin) * id;
            const float x1 = (core::sanitiseFinite(blk[j + 16u]) - vmin) * id;
            // ggml ref uses just (uint8_t)(x + 0.5f) — no MIN clamp, so
            // values can wrap if d underestimates. Our sanitise pass plus
            // the (max-min)/31 formula ensures x ∈ [0, 31] for finite input,
            // but we add a defensive clamp anyway.
            int xi0_i = static_cast<int>(x0 + 0.5f);
            int xi1_i = static_cast<int>(x1 + 0.5f);
            if (xi0_i > 31) xi0_i = 31;
            if (xi0_i < 0)  xi0_i = 0;
            if (xi1_i > 31) xi1_i = 31;
            if (xi1_i < 0)  xi1_i = 0;
            const core::u8 xi0 = static_cast<core::u8>(xi0_i);
            const core::u8 xi1 = static_cast<core::u8>(xi1_i);
            out[8u + j] = static_cast<std::byte>(
                (xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= (static_cast<core::u32>(xi0 & 0x10u) >> 4) << j;
            qh |= (static_cast<core::u32>(xi1 & 0x10u) >> 4) << (j + 16u);
        }
        std::memcpy(out + 4u, &qh, sizeof(qh));
        out += kQ5_1BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ5_1ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ5_1BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ5_1BlockBytes;
    const core::usize n_elements = n_blocks * kGGML_BlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in + 0u));
        const float m = core::fp16ToFP32(detail::loadU16(in + 2u));
        if (!core::isFiniteStrict(d) || !core::isFiniteStrict(m)) return 0u;
        core::u32 qh;
        std::memcpy(&qh, in + 4u, sizeof(qh));

        for (core::usize j = 0; j < kGGML_BlockElements / 2u; ++j) {
            const core::u8 xh_0 = static_cast<core::u8>(((qh >> j)         << 4) & 0x10u);
            const core::u8 xh_1 = static_cast<core::u8>( (qh >> (j + 12u))       & 0x10u);
            const core::u8 packed = static_cast<core::u8>(in[8u + j]);
            const int x0 = static_cast<int>((packed & 0x0F) | xh_0);
            const int x1 = static_cast<int>((packed >> 4)   | xh_1);
            dst_floats[b * kGGML_BlockElements + j]        = static_cast<float>(x0) * d + m;
            dst_floats[b * kGGML_BlockElements + j + 16u]  = static_cast<float>(x1) * d + m;
        }
        in += kQ5_1BlockBytes;
    }
    return need_bytes;
}

// --------------------------------------------------------------------------
// INT8 Hardware (DP4A / SDOT layout) — Epic 2.
//
// The existing QTX INT8 format `compressFP32ToINT8` produces 36-byte
// blocks laid out as `[FP32 scale : 4 B] [INT8 qs : 32 B]`. The 4-byte
// FP32 scale prefix puts the INT8 payload at offset 4 within each
// block — exactly 4-byte aligned, which is the alignment required by:
//   * NVIDIA DP4A      (PTX `dp4a` — 4 INT8 per 32-bit register)
//   * ARM A64 SDOT     (`sdot` — same packing)
//   * Intel AVX-VNNI   (`vpdpbusd` — works on any alignment; benefits
//                       from 4-byte at minimum, 64-byte for ZMM)
//
// However, two refinements are useful for hardware accelerators:
//
//   1. A "scale-stripped" payload (raw INT8 stream) for backends that
//      want to read the qs bytes contiguously without skipping the
//      4-byte scale every 36 bytes. We provide
//      `extractINT8Payload_DP4A` for that.
//
//   2. A compile-time guarantee that block offsets are stable and
//      that the qs sub-buffer alignment is documented.
//
// kINT8_QsOffset = 4 documents the offset of qs within each INT8 block.
// Concrete tests verify the property holds.
// --------------------------------------------------------------------------

inline constexpr core::usize kINT8_QsOffset       = sizeof(float);
inline constexpr core::usize kINT8_QsBytesPerBlock = kBlockElements;

static_assert(kINT8_QsOffset == 4u,
              "INT8 qs must start at offset 4 for DP4A/SDOT alignment");
static_assert(kINT8_QsOffset % 4u == 0u,
              "qs offset must be a multiple of 4 (DP4A loads u32 of 4 INT8)");

/// Extract just the INT8 quantized payload from a QTX INT8 stream,
/// dropping the 4-byte FP32 scale of each block. Output stream layout:
///   [qs of block 0 : 32 B] [qs of block 1 : 32 B] ...
///
/// The output is a contiguous INT8 byte stream suitable for direct
/// consumption by DP4A / SDOT / vpdpbusd hardware. The per-block FP32
/// scales must be carried separately (use `extractINT8Scales_DP4A`
/// below). This separation is the canonical layout for tensor-core
/// integer matmul kernels.
///
/// Returns the number of payload bytes written, or 0 on bad input.
[[nodiscard]] inline core::usize extractINT8Payload_DP4A(
    std::span<const std::byte> int8_stream,
    std::span<std::byte>       qs_out) noexcept
{
    if (int8_stream.size() % kINT8BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = int8_stream.size() / kINT8BlockBytes;
    const core::usize need_bytes = n_blocks * kINT8_QsBytesPerBlock;
    if (qs_out.size() < need_bytes) return 0u;
    if (detail::spansOverlap(int8_stream.data(), int8_stream.size(),
                             qs_out.data(), need_bytes)) return 0u;

    const std::byte* in  = int8_stream.data();
    std::byte*       out = qs_out.data();
    for (core::usize b = 0; b < n_blocks; ++b) {
        std::memcpy(out, in + kINT8_QsOffset, kINT8_QsBytesPerBlock);
        in  += kINT8BlockBytes;
        out += kINT8_QsBytesPerBlock;
    }
    return need_bytes;
}

/// Extract the per-block FP32 scales from a QTX INT8 stream into a
/// dense FP32 array. Output layout: one float per 32-element block.
///
/// The output is suitable as the scale operand for the matmul output
/// dequantization step. Returns the number of bytes written.
[[nodiscard]] inline core::usize extractINT8Scales_DP4A(
    std::span<const std::byte> int8_stream,
    std::span<std::byte>       scales_out) noexcept
{
    if (int8_stream.size() % kINT8BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = int8_stream.size() / kINT8BlockBytes;
    const core::usize need_bytes = n_blocks * sizeof(float);
    if (scales_out.size() < need_bytes) return 0u;
    if (detail::spansOverlap(int8_stream.data(), int8_stream.size(),
                             scales_out.data(), need_bytes)) return 0u;

    const std::byte* in  = int8_stream.data();
    std::byte*       out = scales_out.data();
    for (core::usize b = 0; b < n_blocks; ++b) {
        // The FP32 scale lives at offset 0 of each block.
        std::memcpy(out, in, sizeof(float));
        in  += kINT8BlockBytes;
        out += sizeof(float);
    }
    return need_bytes;
}

// ============================================================================
// K-Quants (hierarchical asymmetric super-blocks) — Epic 3.
//
// All five K-Quant formats (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) use a
// super-block of 256 elements. Wire-formats are bit-for-bit identical
// to ggml's `block_q*_K` structs (ggml-common.h), so any ggml-encoded
// block is decodable by QTX. The encoders use the brief's R&D-1
// single-pass heuristic (no inner L-BFGS-style optimisation),
// trading ≤ 0.5% perplexity for O(N) compression speed.
// ============================================================================

inline constexpr core::usize kKBlockElements   = 256u;
inline constexpr core::usize kQ2_K_BlockBytes  = 84u;
inline constexpr core::usize kQ3_K_BlockBytes  = 110u;
inline constexpr core::usize kQ4_K_BlockBytes  = 144u;
inline constexpr core::usize kQ5_K_BlockBytes  = 176u;
inline constexpr core::usize kQ6_K_BlockBytes  = 210u;

static_assert(kQ2_K_BlockBytes == 2u * 2u + 16u + 64u);          // d,dmin + scales + qs
static_assert(kQ3_K_BlockBytes == 32u + 64u + 12u + 2u);         // hmask + qs + scales + d
static_assert(kQ4_K_BlockBytes == 2u * 2u + 12u + 128u);
static_assert(kQ5_K_BlockBytes == 2u * 2u + 12u + 32u + 128u);
static_assert(kQ6_K_BlockBytes == 128u + 64u + 16u + 2u);        // ql + qh + scales + d

namespace detail_kquant {

/// Helper: nearest-int via round-half-away-from-zero (matches ggml's
/// `nearest_int` which uses x + (x >= 0 ? 0.5 : -0.5)).
[[nodiscard]] inline int nearestInt(float x) noexcept {
    return (x >= 0.0f)
        ? static_cast<int>(x + 0.5f)
        : -static_cast<int>(-x + 0.5f);
}

/// Find the signed extreme of a sub-block (the element with the largest
/// |value|, returning its signed value). Used as the basis for symmetric
/// per-sub-block scale derivation in the R&D-1 single-pass heuristic.
[[nodiscard]] inline float subBlockSignedAbsMax(
    const float* p, core::usize n) noexcept
{
    float amax = 0.0f;
    float vmax = 0.0f;
    for (core::usize j = 0; j < n; ++j) {
        const float v = core::sanitiseFinite(p[j]);
        const float a = std::fabs(v);
        if (a > amax) { amax = a; vmax = v; }
    }
    return vmax;
}

/// Find the (min, max) of a sub-block, both sanitised.
struct MinMax { float vmin; float vmax; };
[[nodiscard]] inline MinMax subBlockMinMax(
    const float* p, core::usize n) noexcept
{
    float vmin =  std::numeric_limits<float>::max();
    float vmax = -std::numeric_limits<float>::max();
    for (core::usize j = 0; j < n; ++j) {
        const float v = core::sanitiseFinite(p[j]);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    return {vmin, vmax};
}

}  // namespace detail_kquant

// ============================================================================
// Q6_K — 6-bit super-block quantization (16 sub-blocks of 16 elements).
//
// Wire format (verbatim ggml `block_q6_K`, 210 bytes):
//   ql[128]    : low 4 bits of each 6-bit quant (32 elements per 32-byte
//                stripe; 4 stripes total = 128 bytes)
//   qh[64]     : high 2 bits of each 6-bit quant (4-elements-per-byte
//                packing, see below)
//   scales[16] : signed 8-bit per-sub-block scale (int8, range [-128,127])
//   d          : FP16 super-block scale
//
// Encoding (R&D-1 single-pass heuristic, no inner optimisation):
//   1. For each of 16 sub-blocks of 16 elements:
//        signed_max_i = arg_max_v |v|  (sign preserved)
//        scale_i = signed_max_i / -32     // negative if max > 0
//   2. Find super-block max-|scale_i|:
//        max_scale = arg_max_scale |scale_i|
//   3. Derive super-block scale:
//        iscale = -128 / max_scale
//        d_super = 1 / iscale
//        q_scale_i = clamp(round(iscale * scale_i), -127, 127)
//   4. For each element:
//        local_d = d_super * q_scale_i
//        l = clamp(round(x_j / local_d), -32, 31)
//        L[j] = l + 32                     // store as unsigned [0, 63]
//   5. Pack ql / qh per ggml's "4-stripe" interleaving (see code).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_Q6_K(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kQ6_K_BlockBytes;
}

static_assert(compressedSize_Q6_K(256u)  == 210u);
static_assert(compressedSize_Q6_K(1024u) == 840u);

[[nodiscard]] inline core::usize compressFP32ToQ6_K(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kKBlockElements;
    const core::usize need_bytes = n_blocks * kQ6_K_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    // L holds the unsigned [0..63] quantised value per element.
    // 16 sub-block scales (signed FP32).
    std::array<int,   kKBlockElements>      L{};
    std::array<float, kKBlockElements / 16> sub_scales{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Step 1: per-sub-block signed scale (R&D-1 single-pass heuristic).
        float max_abs_scale = 0.0f;
        float max_scale     = 0.0f;
        for (core::usize ib = 0; ib < 16u; ++ib) {
            const float vmax = detail_kquant::subBlockSignedAbsMax(blk + 16u * ib, 16u);
            // Symmetric scale: maps the most-extreme value to the -32 endpoint.
            // If vmax = 0 (empty sub-block), scale = 0 — handled as zero block.
            const float scale = vmax / -32.0f;
            sub_scales[ib] = scale;
            const float ascale = std::fabs(scale);
            if (ascale > max_abs_scale) {
                max_abs_scale = ascale;
                max_scale = scale;
            }
        }

        // Zero super-block: emit all zeros.
        if (max_abs_scale < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kQ6_K_BlockBytes);
            // d = 0 (FP16) is already 0x0000 from memset.
            out += kQ6_K_BlockBytes;
            continue;
        }

        // Step 2-3: super-block scale + 6-bit per-sub-block scales.
        const float iscale  = -128.0f / max_scale;
        const float d_super = 1.0f / iscale;

        // Write int8 sub-block scales.
        int8_t sub_q_scales[16];
        for (core::usize ib = 0; ib < 16u; ++ib) {
            int q = detail_kquant::nearestInt(iscale * sub_scales[ib]);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            sub_q_scales[ib] = static_cast<int8_t>(q);
        }

        // Step 4: per-element quantisation.
        for (core::usize ib = 0; ib < 16u; ++ib) {
            const float local_d = d_super * static_cast<float>(sub_q_scales[ib]);
            if (local_d == 0.0f) {
                for (core::usize ii = 0; ii < 16u; ++ii) L[16u * ib + ii] = 32;  // q+32, q=0
                continue;
            }
            const float ild = 1.0f / local_d;
            for (core::usize ii = 0; ii < 16u; ++ii) {
                int l = detail_kquant::nearestInt(
                    core::sanitiseFinite(blk[16u * ib + ii]) * ild);
                if (l < -32) l = -32;
                if (l >  31) l =  31;
                L[16u * ib + ii] = l + 32;     // shift to [0, 63]
            }
        }

        // Step 5: pack ql[128] + qh[64] per ggml's stripe layout.
        //   For each 128-element half (j=0, j=128):
        //     For each l in [0..32):
        //       q1=L[j+l+ 0]&0xF;  q2=L[j+l+32]&0xF
        //       q3=L[j+l+64]&0xF;  q4=L[j+l+96]&0xF
        //       ql[l+ 0] = q1 | (q3 << 4)
        //       ql[l+32] = q2 | (q4 << 4)
        //       qh[l]    = (L[j+l]>>4) | ((L[j+l+32]>>4)<<2)
        //                | ((L[j+l+64]>>4)<<4) | ((L[j+l+96]>>4)<<6)
        std::byte* ql_p = out;                 // ql[128]
        std::byte* qh_p = out + 128u;          // qh[64]
        for (core::usize j = 0; j < kKBlockElements; j += 128u) {
            for (core::usize l = 0; l < 32u; ++l) {
                const int q1 = L[j + l +  0] & 0x0F;
                const int q2 = L[j + l + 32] & 0x0F;
                const int q3 = L[j + l + 64] & 0x0F;
                const int q4 = L[j + l + 96] & 0x0F;
                ql_p[l +  0] = static_cast<std::byte>(q1 | (q3 << 4));
                ql_p[l + 32] = static_cast<std::byte>(q2 | (q4 << 4));
                qh_p[l] = static_cast<std::byte>(
                    (L[j + l +  0] >> 4)        |
                    ((L[j + l + 32] >> 4) << 2) |
                    ((L[j + l + 64] >> 4) << 4) |
                    ((L[j + l + 96] >> 4) << 6));
            }
            ql_p += 64u;
            qh_p += 32u;
        }

        // Write scales[16] and d (FP16) at the tail.
        std::byte* sc_p = out + 128u + 64u;
        for (core::usize ib = 0; ib < 16u; ++ib) {
            sc_p[ib] = static_cast<std::byte>(static_cast<core::u8>(sub_q_scales[ib]));
        }
        const core::u16 d_h = core::fp32ToFP16Safe(d_super);
        detail::storeU16(out + 128u + 64u + 16u, d_h);

        out += kQ6_K_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ6_K_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ6_K_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ6_K_BlockBytes;
    const core::usize n_elements = n_blocks * kKBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const std::byte* ql_p = in;
        const std::byte* qh_p = in + 128u;
        const std::byte* sc_p = in + 128u + 64u;
        const float d = core::fp16ToFP32(detail::loadU16(in + 128u + 64u + 16u));
        if (!core::isFiniteStrict(d)) return 0u;

        float* y = dst_floats + b * kKBlockElements;
        for (core::usize j = 0; j < kKBlockElements; j += 128u) {
            for (core::usize l = 0; l < 32u; ++l) {
                const core::u8 qhl = static_cast<core::u8>(qh_p[l]);
                const int8_t q1 = static_cast<int8_t>(
                    static_cast<int>(
                        (static_cast<core::u8>(ql_p[l +  0]) & 0x0F) |
                        (((qhl >> 0) & 3u) << 4)) - 32);
                const int8_t q2 = static_cast<int8_t>(
                    static_cast<int>(
                        (static_cast<core::u8>(ql_p[l + 32]) & 0x0F) |
                        (((qhl >> 2) & 3u) << 4)) - 32);
                const int8_t q3 = static_cast<int8_t>(
                    static_cast<int>(
                        (static_cast<core::u8>(ql_p[l +  0]) >> 4) |
                        (((qhl >> 4) & 3u) << 4)) - 32);
                const int8_t q4 = static_cast<int8_t>(
                    static_cast<int>(
                        (static_cast<core::u8>(ql_p[l + 32]) >> 4) |
                        (((qhl >> 6) & 3u) << 4)) - 32);

                // is = l / 16  (index into the 16-byte scales array; each
                // 128-stripe uses 8 sub-blocks at base offset `is_base`).
                const core::usize is_base = (j / 128u) * 8u;
                const int sc0 = static_cast<int8_t>(sc_p[is_base + l / 16u + 0u]);
                const int sc1 = static_cast<int8_t>(sc_p[is_base + l / 16u + 2u]);
                const int sc2 = static_cast<int8_t>(sc_p[is_base + l / 16u + 4u]);
                const int sc3 = static_cast<int8_t>(sc_p[is_base + l / 16u + 6u]);
                y[l +  0] = d * static_cast<float>(sc0) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc1) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc2) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc3) * static_cast<float>(q4);
            }
            y    += 128u;
            ql_p += 64u;
            qh_p += 32u;
        }
        in += kQ6_K_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// K-Quant scale/min packing helpers (K_SCALE_SIZE = 12).
//
// The 12-byte `scales` field in Q4_K and Q5_K encodes 8 sub-block
// scales (6 bits each) + 8 sub-block mins (6 bits each) = 96 bits.
// The packing (from ggml's `get_scale_min_k4`):
//
//   For j in [0..3]:
//     scales[j]   holds the 6-bit scale_j in bits 0..5
//     scales[j+4] holds the 6-bit min_j   in bits 0..5
//
//   For j in [4..7]:
//     scale_j = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//     min_j   = (scales[j+4] >> 4)  | ((scales[j-0] >> 6) << 4)
//
// I.e., bytes 0..7 carry both the j∈[0..3] sub-block 6-bit values AND
// (in their top 2 bits) the high 2 bits of the j∈[4..7] sub-block
// values. Byte j+4 (j∈[4..7], byte index 8..11) holds the low 4 bits.
// ============================================================================

namespace detail_kquant {

/// Pack 8 scales + 8 mins (each in [0, 63]) into a 12-byte `scales[12]`
/// field. Inverse of ggml's `get_scale_min_k4`.
inline void packScalesMinsK4(
    const core::u8 (&q_scale)[8], const core::u8 (&q_min)[8],
    std::byte out[12]) noexcept
{
    // Zero the field first so the |= operations below are deterministic.
    for (core::usize i = 0; i < 12u; ++i) out[i] = std::byte{0};

    // Bytes 0..7: low 6 bits = scale_j (j<4) or min_{j-4} (j>=4).
    for (core::usize j = 0; j < 4u; ++j) {
        out[j]     = static_cast<std::byte>(q_scale[j] & 0x3Fu);
        out[j + 4u] = static_cast<std::byte>(q_min[j]   & 0x3Fu);
    }
    // Bytes 8..11: scale_{j+4} in low 4, min_{j+4} in high 4.
    for (core::usize j = 4u; j < 8u; ++j) {
        const core::u8 sl = static_cast<core::u8>(q_scale[j] & 0x0Fu);
        const core::u8 ml = static_cast<core::u8>(q_min[j]   & 0x0Fu);
        out[j + 4u] = static_cast<std::byte>(sl | (ml << 4));
    }
    // Bytes 0..3: top 2 bits = high 2 bits of scale_{j+4}.
    // Bytes 4..7: top 2 bits = high 2 bits of min_{j-4+4} = min_{j}, that is
    //             scales[j] (j<4) carries scale_{j+4}>>4; scales[j+4]
    //             (j<4) carries min_{j+4}>>4. The expression in
    //             get_scale_min_k4 for the high bits:
    //               scale_{j+4}: from (scales[j-4]>>6)<<4 -> high 2 bits of scales[(j+4)-4]=scales[j]
    //               min_{j+4}:   from (scales[j-0]>>6)<<4 -> high 2 bits of scales[(j+4)-0]=scales[j+4]
    for (core::usize j = 4u; j < 8u; ++j) {
        const core::u8 s_hi = static_cast<core::u8>((q_scale[j] >> 4) & 0x03u);
        const core::u8 m_hi = static_cast<core::u8>((q_min[j]   >> 4) & 0x03u);
        out[j - 4u]  = static_cast<std::byte>(
            static_cast<core::u8>(out[j - 4u]) | (s_hi << 6));
        out[j]       = static_cast<std::byte>(
            static_cast<core::u8>(out[j]) | (m_hi << 6));
    }
}

/// Unpack a single (scale, min) pair from a 12-byte `scales` field.
/// Verbatim transcription of ggml's `get_scale_min_k4`.
inline void unpackScaleMinK4(
    int j, const std::byte q[12],
    core::u8* out_d, core::u8* out_m) noexcept
{
    if (j < 4) {
        *out_d = static_cast<core::u8>(q[j])     & 0x3Fu;
        *out_m = static_cast<core::u8>(q[j + 4]) & 0x3Fu;
    } else {
        *out_d = static_cast<core::u8>(
            (static_cast<core::u8>(q[j + 4]) & 0x0Fu) |
            ((static_cast<core::u8>(q[j - 4]) >> 6) << 4));
        *out_m = static_cast<core::u8>(
            ( static_cast<core::u8>(q[j + 4])  >>  4) |
            ((static_cast<core::u8>(q[j])      >> 6) << 4));
    }
}

}  // namespace detail_kquant

// ============================================================================
// Q5_K — 5-bit super-block quantization (8 sub-blocks of 32 elements).
//
// Wire format (verbatim ggml `block_q5_K`, 176 bytes):
//   d              : FP16 super-block scale (offset 0)
//   dmin           : FP16 super-block min   (offset 2)
//   scales[12]     : 8 (6-bit scale) + 8 (6-bit min) packed       (offset 4)
//   qh[32]         : high bit of each 5-bit quant                 (offset 16)
//   qs[128]        : low 4 bits of each 5-bit quant               (offset 48)
//
// Encoding (R&D-1 single-pass, no inner make_qkx2_quants):
//   For each sub-block of 32:
//     vmin, vmax    = min/max of sub-block (sanitised)
//     scale_i       = (vmax - vmin) / 31
//     blk_min_off_i = -vmin           (so that L=0 maps to vmin)
//   max_scale, max_min = max over sub-blocks
//   d    = max_scale / 63
//   dmin = max_min   / 63
//   q_scale_i = clamp(round(63 * scale_i / max_scale), 0, 63)
//   q_min_i   = clamp(round(63 * blk_min_off_i / max_min), 0, 63)
//   For each element:
//     local_d = d * q_scale_i ;  local_m = dmin * q_min_i
//     L = clamp(round((x + local_m) / local_d), 0, 31)
//   Pack: low 4 bits -> qs ; high 1 bit -> qh (m1/m2 bitmask).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_Q5_K(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kQ5_K_BlockBytes;
}

static_assert(compressedSize_Q5_K(256u)  == 176u);
static_assert(compressedSize_Q5_K(1024u) == 704u);

[[nodiscard]] inline core::usize compressFP32ToQ5_K(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kKBlockElements;
    const core::usize need_bytes = n_blocks * kQ5_K_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    std::array<int,   kKBlockElements>      L{};
    std::array<float, kKBlockElements / 32> sub_scales{};
    std::array<float, kKBlockElements / 32> sub_mins{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Step 1: per-sub-block scale + min.
        float max_scale = 0.0f;
        float max_min   = 0.0f;
        for (core::usize ib = 0; ib < 8u; ++ib) {
            auto mm = detail_kquant::subBlockMinMax(blk + 32u * ib, 32u);
            // ggml clamps `min` to be ≤ 0 (line 756 of make_qkx2_quants).
            // The encoded relationship is x = d*L - dmin*m, so the offset
            // can only shift towards zero; positive-mean blocks set dm=0.
            if (mm.vmin > 0.0f) mm.vmin = 0.0f;
            const float scale = (mm.vmax - mm.vmin) / 31.0f;
            const float min_offset = -mm.vmin;
            sub_scales[ib] = scale;
            sub_mins[ib]   = min_offset;
            if (scale      > max_scale) max_scale = scale;
            if (min_offset > max_min)   max_min   = min_offset;
        }

        // Zero block: emit zero.
        if (max_scale == 0.0f && max_min == 0.0f) [[unlikely]] {
            std::memset(out, 0, kQ5_K_BlockBytes);
            out += kQ5_K_BlockBytes;
            continue;
        }

        const float inv_max_scale = (max_scale > 0.0f) ? 63.0f / max_scale : 0.0f;
        const float inv_max_min   = (max_min   > 0.0f) ? 63.0f / max_min   : 0.0f;
        const float d_super       = max_scale / 63.0f;
        const float dmin_super    = max_min   / 63.0f;

        core::u8 q_scales[8];
        core::u8 q_mins[8];
        for (core::usize ib = 0; ib < 8u; ++ib) {
            int ls = detail_kquant::nearestInt(inv_max_scale * sub_scales[ib]);
            int lm = detail_kquant::nearestInt(inv_max_min   * sub_mins[ib]);
            if (ls > 63) ls = 63;
            if (ls < 0)  ls = 0;
            if (lm > 63) lm = 63;
            if (lm < 0)  lm = 0;
            q_scales[ib] = static_cast<core::u8>(ls);
            q_mins[ib]   = static_cast<core::u8>(lm);
        }

        // Step 2: per-element quantization.
        for (core::usize ib = 0; ib < 8u; ++ib) {
            const float local_d = d_super    * static_cast<float>(q_scales[ib]);
            const float local_m = dmin_super * static_cast<float>(q_mins[ib]);
            if (local_d == 0.0f) {
                for (core::usize ii = 0; ii < 32u; ++ii) L[32u * ib + ii] = 0;
                continue;
            }
            const float ild = 1.0f / local_d;
            for (core::usize ii = 0; ii < 32u; ++ii) {
                int l = detail_kquant::nearestInt(
                    (core::sanitiseFinite(blk[32u * ib + ii]) + local_m) * ild);
                if (l < 0)  l = 0;
                if (l > 31) l = 31;
                L[32u * ib + ii] = l;
            }
        }

        // Write FP16 d, dmin.
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_super));
        detail::storeU16(out + 2u, core::fp32ToFP16Safe(dmin_super));
        // Pack scales[12].
        detail_kquant::packScalesMinsK4(q_scales, q_mins, out + 4u);

        // Pack qh[32] + qs[128] per ggml's m1/m2 bit-mask scheme.
        std::byte* qh_p = out + 16u;
        std::byte* ql_p = out + 48u;
        for (core::usize i = 0; i < 32u; ++i) qh_p[i] = std::byte{0};

        core::u8 m1 = 1u, m2 = 2u;
        for (core::usize n = 0; n < kKBlockElements; n += 64u) {
            for (core::usize j = 0; j < 32u; ++j) {
                int l1 = L[n + j];
                int l2 = L[n + j + 32u];
                if (l1 > 15) {
                    l1 -= 16;
                    qh_p[j] = static_cast<std::byte>(
                        static_cast<core::u8>(qh_p[j]) | m1);
                }
                if (l2 > 15) {
                    l2 -= 16;
                    qh_p[j] = static_cast<std::byte>(
                        static_cast<core::u8>(qh_p[j]) | m2);
                }
                ql_p[j] = static_cast<std::byte>(
                    static_cast<core::u8>(l1) |
                    (static_cast<core::u8>(l2) << 4));
            }
            m1 = static_cast<core::u8>(m1 << 2);
            m2 = static_cast<core::u8>(m2 << 2);
            ql_p += 32u;
        }

        out += kQ5_K_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ5_K_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ5_K_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ5_K_BlockBytes;
    const core::usize n_elements = n_blocks * kKBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d    = core::fp16ToFP32(detail::loadU16(in + 0u));
        const float dmin = core::fp16ToFP32(detail::loadU16(in + 2u));
        if (!core::isFiniteStrict(d) || !core::isFiniteStrict(dmin)) return 0u;

        const std::byte* sc_p = in + 4u;       // scales[12]
        const std::byte* qh_p = in + 16u;      // qh[32]
        const std::byte* ql_p = in + 48u;      // qs[128]

        float* y = dst_floats + b * kKBlockElements;
        int is = 0;
        core::u8 u1 = 1u, u2 = 2u;
        for (core::usize j = 0; j < kKBlockElements; j += 64u) {
            core::u8 sc, m;
            detail_kquant::unpackScaleMinK4(is + 0, sc_p, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            detail_kquant::unpackScaleMinK4(is + 1, sc_p, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            for (core::usize l = 0; l < 32u; ++l) {
                const core::u8 q  = static_cast<core::u8>(ql_p[l]);
                const core::u8 qh = static_cast<core::u8>(qh_p[l]);
                const int v1 = (q & 0x0F) + ((qh & u1) ? 16 : 0);
                *y++ = d1 * static_cast<float>(v1) - m1;
            }
            for (core::usize l = 0; l < 32u; ++l) {
                const core::u8 q  = static_cast<core::u8>(ql_p[l]);
                const core::u8 qh = static_cast<core::u8>(qh_p[l]);
                const int v2 = (q >> 4)   + ((qh & u2) ? 16 : 0);
                *y++ = d2 * static_cast<float>(v2) - m2;
            }
            ql_p += 32u;
            is   += 2;
            u1    = static_cast<core::u8>(u1 << 2);
            u2    = static_cast<core::u8>(u2 << 2);
        }
        in += kQ5_K_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// Q4_K — 4-bit super-block quantization (8 sub-blocks of 32 elements).
//
// Wire format (verbatim ggml `block_q4_K`, 144 bytes):
//   d           : FP16 super-block scale            (offset 0)
//   dmin        : FP16 super-block min              (offset 2)
//   scales[12]  : 8 (6-bit scale) + 8 (6-bit min)   (offset 4)
//   qs[128]     : 4-bit quants                      (offset 16)
//
// Encoding is identical to Q5_K but uses 15 codepoints per sub-block
// (`L` in [0, 15]) and no qh field.
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_Q4_K(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kQ4_K_BlockBytes;
}

static_assert(compressedSize_Q4_K(256u)  == 144u);
static_assert(compressedSize_Q4_K(1024u) == 576u);

[[nodiscard]] inline core::usize compressFP32ToQ4_K(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kKBlockElements;
    const core::usize need_bytes = n_blocks * kQ4_K_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    std::array<int,   kKBlockElements>      L{};
    std::array<float, kKBlockElements / 32> sub_scales{};
    std::array<float, kKBlockElements / 32> sub_mins{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        float max_scale = 0.0f;
        float max_min   = 0.0f;
        for (core::usize ib = 0; ib < 8u; ++ib) {
            auto mm = detail_kquant::subBlockMinMax(blk + 32u * ib, 32u);
            if (mm.vmin > 0.0f) mm.vmin = 0.0f;     // ggml-style clamp
            const float scale = (mm.vmax - mm.vmin) / 15.0f;     // 4-bit -> 15 codepoints
            const float min_offset = -mm.vmin;
            sub_scales[ib] = scale;
            sub_mins[ib]   = min_offset;
            if (scale      > max_scale) max_scale = scale;
            if (min_offset > max_min)   max_min   = min_offset;
        }

        if (max_scale == 0.0f && max_min == 0.0f) [[unlikely]] {
            std::memset(out, 0, kQ4_K_BlockBytes);
            out += kQ4_K_BlockBytes;
            continue;
        }

        const float inv_max_scale = (max_scale > 0.0f) ? 63.0f / max_scale : 0.0f;
        const float inv_max_min   = (max_min   > 0.0f) ? 63.0f / max_min   : 0.0f;
        const float d_super       = max_scale / 63.0f;
        const float dmin_super    = max_min   / 63.0f;

        core::u8 q_scales[8];
        core::u8 q_mins[8];
        for (core::usize ib = 0; ib < 8u; ++ib) {
            int ls = detail_kquant::nearestInt(inv_max_scale * sub_scales[ib]);
            int lm = detail_kquant::nearestInt(inv_max_min   * sub_mins[ib]);
            if (ls > 63) ls = 63;
            if (ls < 0)  ls = 0;
            if (lm > 63) lm = 63;
            if (lm < 0)  lm = 0;
            q_scales[ib] = static_cast<core::u8>(ls);
            q_mins[ib]   = static_cast<core::u8>(lm);
        }

        for (core::usize ib = 0; ib < 8u; ++ib) {
            const float local_d = d_super    * static_cast<float>(q_scales[ib]);
            const float local_m = dmin_super * static_cast<float>(q_mins[ib]);
            if (local_d == 0.0f) {
                for (core::usize ii = 0; ii < 32u; ++ii) L[32u * ib + ii] = 0;
                continue;
            }
            const float ild = 1.0f / local_d;
            for (core::usize ii = 0; ii < 32u; ++ii) {
                int l = detail_kquant::nearestInt(
                    (core::sanitiseFinite(blk[32u * ib + ii]) + local_m) * ild);
                if (l < 0)  l = 0;
                if (l > 15) l = 15;
                L[32u * ib + ii] = l;
            }
        }

        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_super));
        detail::storeU16(out + 2u, core::fp32ToFP16Safe(dmin_super));
        detail_kquant::packScalesMinsK4(q_scales, q_mins, out + 4u);

        // qs[128]: pairs of 4-bit values packed per sub-block, using
        // ggml's "first 32 in low nibble, second 32 in high nibble per
        // 64-element half" layout.
        std::byte* ql_p = out + 16u;
        for (core::usize n = 0; n < kKBlockElements; n += 64u) {
            for (core::usize j = 0; j < 32u; ++j) {
                ql_p[j] = static_cast<std::byte>(
                    static_cast<core::u8>(L[n + j]) |
                    (static_cast<core::u8>(L[n + j + 32u]) << 4));
            }
            ql_p += 32u;
        }

        out += kQ4_K_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ4_K_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ4_K_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ4_K_BlockBytes;
    const core::usize n_elements = n_blocks * kKBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d    = core::fp16ToFP32(detail::loadU16(in + 0u));
        const float dmin = core::fp16ToFP32(detail::loadU16(in + 2u));
        if (!core::isFiniteStrict(d) || !core::isFiniteStrict(dmin)) return 0u;

        const std::byte* sc_p = in + 4u;
        const std::byte* ql_p = in + 16u;

        float* y = dst_floats + b * kKBlockElements;
        int is = 0;
        for (core::usize j = 0; j < kKBlockElements; j += 64u) {
            core::u8 sc, m;
            detail_kquant::unpackScaleMinK4(is + 0, sc_p, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            detail_kquant::unpackScaleMinK4(is + 1, sc_p, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            for (core::usize l = 0; l < 32u; ++l) {
                const core::u8 q = static_cast<core::u8>(ql_p[l]);
                *y++ = d1 * static_cast<float>(q & 0x0F) - m1;
            }
            for (core::usize l = 0; l < 32u; ++l) {
                const core::u8 q = static_cast<core::u8>(ql_p[l]);
                *y++ = d2 * static_cast<float>(q >> 4) - m2;
            }
            ql_p += 32u;
            is   += 2;
        }
        in += kQ4_K_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// Q3_K — 3-bit super-block quantization (16 sub-blocks of 16 elements).
//
// Wire format (verbatim ggml `block_q3_K`, 110 bytes):
//   hmask[32]   : high bit of each 3-bit quant (4 elements per byte)   (offset  0)
//   qs[64]      : low 2 bits of each 3-bit quant (4 per byte)          (offset 32)
//   scales[12]  : 16 signed 6-bit scales, packed non-trivially         (offset 96)
//   d           : FP16 super-block scale                               (offset 108)
//
// Element range: signed 3-bit, [-4..3]. After `+4` shift -> [0..7].
// The hmask bit indicates which 8 values land in the high 4 (≥4 after shift,
// i.e., l ≥ 0 originally — positive); the qs holds the low 2 bits.
//
// Scales[12] encoding (non-trivial; see dequant code):
//   Each j ∈ [0..15] sub-block has a 6-bit *signed* scale in [-32..31].
//   The encoded byte L is `clamp(round(iscale * scale_j), -32, 31) + 32`.
//   Then for j < 8: low 4 bits of L go into scales[j] (low nibble);
//                   high 2 bits go into scales[8 + j%4] at bit (2 * (j/4)).
//        for j ≥ 8: low 4 bits of L go into scales[j-8] (high nibble);
//                   high 2 bits go into scales[8 + j%4] at bit (2 * (j/4)).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_Q3_K(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kQ3_K_BlockBytes;
}

static_assert(compressedSize_Q3_K(256u)  == 110u);
static_assert(compressedSize_Q3_K(1024u) == 440u);

[[nodiscard]] inline core::usize compressFP32ToQ3_K(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kKBlockElements;
    const core::usize need_bytes = n_blocks * kQ3_K_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    std::array<int,   kKBlockElements>      L{};
    std::array<float, 16>                   sub_scales{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Step 1: signed scale per sub-block (R&D-1 single-pass heuristic).
        //   scale_i = signed_max / -4  (= the smallest 3-bit signed value)
        // We track the sub-block whose |scale| is maximum and its sign.
        float max_abs_scale = 0.0f;
        float max_scale     = 0.0f;
        for (core::usize ib = 0; ib < 16u; ++ib) {
            const float vmax = detail_kquant::subBlockSignedAbsMax(blk + 16u * ib, 16u);
            const float scale = vmax / -4.0f;
            sub_scales[ib] = scale;
            const float ascale = std::fabs(scale);
            if (ascale > max_abs_scale) {
                max_abs_scale = ascale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kQ3_K_BlockBytes);
            out += kQ3_K_BlockBytes;
            continue;
        }

        const float iscale  = -32.0f / max_scale;
        const float d_super = 1.0f / iscale;

        // Quantize the 16 signed 6-bit scales (+32 to map to [0..63]).
        std::array<core::u8, 16> q_scales_shifted{};   // values in [0..63]
        for (core::usize ib = 0; ib < 16u; ++ib) {
            int l = detail_kquant::nearestInt(iscale * sub_scales[ib]);
            if (l < -32) l = -32;
            if (l >  31) l =  31;
            q_scales_shifted[ib] = static_cast<core::u8>(l + 32);
        }

        // Pack into scales[12] per ggml's layout.
        std::byte* sc_out = out + 96u;
        for (core::usize ib = 0; ib < 12u; ++ib) sc_out[ib] = std::byte{0};
        for (core::usize j = 0; j < 16u; ++j) {
            const core::u8 lo = static_cast<core::u8>(q_scales_shifted[j] & 0x0Fu);
            const core::u8 hi = static_cast<core::u8>(q_scales_shifted[j] >> 4);
            if (j < 8u) {
                sc_out[j] = static_cast<std::byte>(
                    static_cast<core::u8>(sc_out[j]) | lo);
            } else {
                sc_out[j - 8u] = static_cast<std::byte>(
                    static_cast<core::u8>(sc_out[j - 8u]) | (lo << 4));
            }
            // hi (2 bits) -> sc_out[8 + j%4] at bit (2 * (j/4))
            const core::usize byte_idx = 8u + (j % 4u);
            const int bit_off = 2 * static_cast<int>(j / 4u);
            sc_out[byte_idx] = static_cast<std::byte>(
                static_cast<core::u8>(sc_out[byte_idx]) |
                static_cast<core::u8>(hi << bit_off));
        }

        // Per-element quantization. Use the *quantized* per-sub-block
        // scale (round-trip through 6-bit encoding) so the encoder sees
        // the same value the decoder will reconstruct.
        for (core::usize ib = 0; ib < 16u; ++ib) {
            const int sc_signed =
                static_cast<int>(q_scales_shifted[ib]) - 32;
            const float local_d = d_super * static_cast<float>(sc_signed);
            if (local_d == 0.0f) {
                for (core::usize ii = 0; ii < 16u; ++ii) L[16u * ib + ii] = 4;  // 0 after the +4 shift
                continue;
            }
            const float ild = 1.0f / local_d;
            for (core::usize ii = 0; ii < 16u; ++ii) {
                int l = detail_kquant::nearestInt(
                    core::sanitiseFinite(blk[16u * ib + ii]) * ild);
                if (l < -4) l = -4;
                if (l >  3) l =  3;
                L[16u * ib + ii] = l + 4;     // shift to [0..7]
            }
        }

        // hmask[32]: bit 'm' of hmask[k] is the high-bit (L[j]>3) of
        // element index `m*32 + k`. Walk j = 0..255 and split.
        std::byte* hmask_p = out;
        std::byte* qs_p    = out + 32u;
        for (core::usize i = 0; i < 32u; ++i) hmask_p[i] = std::byte{0};

        int m  = 0;
        core::u8 hm = 1u;
        for (core::usize j = 0; j < kKBlockElements; ++j) {
            if (L[j] > 3) {
                hmask_p[m] = static_cast<std::byte>(
                    static_cast<core::u8>(hmask_p[m]) | hm);
                L[j] -= 4;
            }
            if (++m == 32) {
                m  = 0;
                hm = static_cast<core::u8>(hm << 1);
            }
        }

        // qs[64]: pack 4 2-bit values per byte, "4-stripe" interleave.
        for (core::usize j = 0; j < kKBlockElements; j += 128u) {
            for (core::usize l = 0; l < 32u; ++l) {
                qs_p[j / 4u + l] = static_cast<std::byte>(
                    static_cast<core::u8>(L[j + l +  0]) |
                    (static_cast<core::u8>(L[j + l + 32]) << 2) |
                    (static_cast<core::u8>(L[j + l + 64]) << 4) |
                    (static_cast<core::u8>(L[j + l + 96]) << 6));
            }
        }

        // FP16 d at offset 108.
        detail::storeU16(out + 108u, core::fp32ToFP16Safe(d_super));

        out += kQ3_K_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ3_K_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ3_K_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ3_K_BlockBytes;
    const core::usize n_elements = n_blocks * kKBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const std::byte* hm = in;
        const std::byte* q  = in + 32u;
        const std::byte* sc = in + 96u;
        const float d_all = core::fp16ToFP32(detail::loadU16(in + 108u));
        if (!core::isFiniteStrict(d_all)) return 0u;

        // Unpack 16 signed 6-bit scales into an int8 array.
        std::array<int8_t, 16> scales{};
        for (core::usize j = 0; j < 16u; ++j) {
            const core::u8 lo = (j < 8u)
                ? static_cast<core::u8>(static_cast<core::u8>(sc[j]) & 0x0Fu)
                : static_cast<core::u8>(static_cast<core::u8>(sc[j - 8u]) >> 4);
            const core::usize byte_idx = 8u + (j % 4u);
            const int bit_off = 2 * static_cast<int>(j / 4u);
            const core::u8 hi = static_cast<core::u8>(
                (static_cast<core::u8>(sc[byte_idx]) >> bit_off) & 0x03u);
            scales[j] = static_cast<int8_t>(
                (static_cast<int>(lo) | (static_cast<int>(hi) << 4)) - 32);
        }

        float* y = dst_floats + b * kKBlockElements;
        int is = 0;
        core::u8 m_bit = 1u;
        for (core::usize n = 0; n < kKBlockElements; n += 128u) {
            int shift = 0;
            for (core::usize jj = 0; jj < 4u; ++jj) {
                float dl = d_all * static_cast<float>(scales[is]); ++is;
                for (core::usize l = 0; l < 16u; ++l) {
                    const core::u8 ql = static_cast<core::u8>(q[l]);
                    const core::u8 hml = static_cast<core::u8>(hm[l]);
                    const int q3 =
                        static_cast<int>((ql >> shift) & 0x03u) -
                        ((hml & m_bit) ? 0 : 4);
                    *y++ = dl * static_cast<float>(q3);
                }
                dl = d_all * static_cast<float>(scales[is]); ++is;
                for (core::usize l = 0; l < 16u; ++l) {
                    const core::u8 ql = static_cast<core::u8>(q[l + 16u]);
                    const core::u8 hml = static_cast<core::u8>(hm[l + 16u]);
                    const int q3 =
                        static_cast<int>((ql >> shift) & 0x03u) -
                        ((hml & m_bit) ? 0 : 4);
                    *y++ = dl * static_cast<float>(q3);
                }
                shift += 2;
                m_bit = static_cast<core::u8>(m_bit << 1);
            }
            q += 32u;
        }

        in += kQ3_K_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// Q2_K — 2-bit super-block quantization (16 sub-blocks of 16 elements).
//
// Wire format (verbatim ggml `block_q2_K`, 84 bytes):
//   scales[16]  : 4-bit scale (low) + 4-bit min (high) per sub-block   (offset  0)
//   qs[64]      : 2-bit quants, 4 elements per byte (4-stripe layout)  (offset 16)
//   d           : FP16 super-block scale                               (offset 80)
//   dmin        : FP16 super-block min                                 (offset 82)
//
// Element range: unsigned 2-bit, [0..3]. Dequant: x = d * sc * L - dmin * m,
// where sc = scales[j] & 0xF and m = scales[j] >> 4 (both 4 bits, [0..15]).
//
// Encoding heuristic (R&D-1): per-sub-block (vmin, vmax) gives
// scale = (max - min)/3, dmin_offset = -min (clamped to ≥0 via min ≤ 0).
// Super-block: max_scale, max_min over sub-blocks. d = max_scale/15;
// dmin = max_min/15. Per sub-block, the 4-bit q_scale = round(15 * scale / max_scale),
// q_min = round(15 * min / max_min). Per element, L = clamp(round((x + dm)/d), 0, 3).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_Q2_K(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kQ2_K_BlockBytes;
}

static_assert(compressedSize_Q2_K(256u)  == 84u);
static_assert(compressedSize_Q2_K(1024u) == 336u);

[[nodiscard]] inline core::usize compressFP32ToQ2_K(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kKBlockElements;
    const core::usize need_bytes = n_blocks * kQ2_K_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    std::array<int,   kKBlockElements>      L{};
    std::array<float, 16>                   sub_scales{};
    std::array<float, 16>                   sub_mins{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        float max_scale = 0.0f;
        float max_min   = 0.0f;
        for (core::usize ib = 0; ib < 16u; ++ib) {
            auto mm = detail_kquant::subBlockMinMax(blk + 16u * ib, 16u);
            if (mm.vmin > 0.0f) mm.vmin = 0.0f;
            const float scale = (mm.vmax - mm.vmin) / 3.0f;
            const float min_offset = -mm.vmin;
            sub_scales[ib] = scale;
            sub_mins[ib]   = min_offset;
            if (scale      > max_scale) max_scale = scale;
            if (min_offset > max_min)   max_min   = min_offset;
        }

        // Write super-block scales (4 bits each, [0..15]).
        // Zero out the scales[16] field first.
        for (core::usize i = 0; i < 16u; ++i) out[i] = std::byte{0};

        const float inv_max_scale = (max_scale > 0.0f) ? 15.0f / max_scale : 0.0f;
        const float inv_max_min   = (max_min   > 0.0f) ? 15.0f / max_min   : 0.0f;
        const float d_super       = max_scale / 15.0f;
        const float dmin_super    = max_min   / 15.0f;

        for (core::usize j = 0; j < 16u; ++j) {
            int ls = detail_kquant::nearestInt(inv_max_scale * sub_scales[j]);
            int lm = detail_kquant::nearestInt(inv_max_min   * sub_mins[j]);
            if (ls > 15) ls = 15;
            if (ls < 0)  ls = 0;
            if (lm > 15) lm = 15;
            if (lm < 0)  lm = 0;
            // Pack: low nibble = scale, high nibble = min.
            out[j] = static_cast<std::byte>(
                static_cast<core::u8>(ls) |
                (static_cast<core::u8>(lm) << 4));
        }

        // Per-element quantization using the *quantized* per-sub-block
        // values (round-trip determinism — same value decoder will see).
        for (core::usize j = 0; j < 16u; ++j) {
            const core::u8 sc_q = static_cast<core::u8>(out[j]) & 0x0Fu;
            const core::u8 m_q  = static_cast<core::u8>(out[j]) >> 4;
            const float local_d = d_super    * static_cast<float>(sc_q);
            const float local_m = dmin_super * static_cast<float>(m_q);
            if (local_d == 0.0f) {
                for (core::usize ii = 0; ii < 16u; ++ii) L[16u * j + ii] = 0;
                continue;
            }
            const float ild = 1.0f / local_d;
            for (core::usize ii = 0; ii < 16u; ++ii) {
                int l = detail_kquant::nearestInt(
                    (core::sanitiseFinite(blk[16u * j + ii]) + local_m) * ild);
                if (l < 0) l = 0;
                if (l > 3) l = 3;
                L[16u * j + ii] = l;
            }
        }

        // qs[64]: pack 4 2-bit values per byte, "4-stripe" layout.
        std::byte* qs_p = out + 16u;
        for (core::usize j = 0; j < kKBlockElements; j += 128u) {
            for (core::usize l = 0; l < 32u; ++l) {
                qs_p[j / 4u + l] = static_cast<std::byte>(
                    static_cast<core::u8>(L[j + l +  0]) |
                    (static_cast<core::u8>(L[j + l + 32]) << 2) |
                    (static_cast<core::u8>(L[j + l + 64]) << 4) |
                    (static_cast<core::u8>(L[j + l + 96]) << 6));
            }
        }

        // FP16 d, dmin at offsets 80, 82.
        detail::storeU16(out + 80u, core::fp32ToFP16Safe(d_super));
        detail::storeU16(out + 82u, core::fp32ToFP16Safe(dmin_super));

        out += kQ2_K_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressQ2_K_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kQ2_K_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kQ2_K_BlockBytes;
    const core::usize n_elements = n_blocks * kKBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d    = core::fp16ToFP32(detail::loadU16(in + 80u));
        const float dmin = core::fp16ToFP32(detail::loadU16(in + 82u));
        if (!core::isFiniteStrict(d) || !core::isFiniteStrict(dmin)) return 0u;

        const std::byte* sc = in;
        const std::byte* q  = in + 16u;

        float* y = dst_floats + b * kKBlockElements;
        int is = 0;
        for (core::usize n = 0; n < kKBlockElements; n += 128u) {
            int shift = 0;
            for (core::usize jj = 0; jj < 4u; ++jj) {
                core::u8 sc_byte = static_cast<core::u8>(sc[is]); ++is;
                float dl  = d    * static_cast<float>(sc_byte & 0x0Fu);
                float ml  = dmin * static_cast<float>(sc_byte >> 4);
                for (core::usize l = 0; l < 16u; ++l) {
                    const core::u8 ql = static_cast<core::u8>(q[l]);
                    *y++ = dl * static_cast<float>(
                        static_cast<int>((ql >> shift) & 0x03u)) - ml;
                }
                sc_byte = static_cast<core::u8>(sc[is]); ++is;
                dl  = d    * static_cast<float>(sc_byte & 0x0Fu);
                ml  = dmin * static_cast<float>(sc_byte >> 4);
                for (core::usize l = 0; l < 16u; ++l) {
                    const core::u8 ql = static_cast<core::u8>(q[l + 16u]);
                    *y++ = dl * static_cast<float>(
                        static_cast<int>((ql >> shift) & 0x03u)) - ml;
                }
                shift += 2;
            }
            q += 32u;
        }

        in += kQ2_K_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// I-Quants (Importance-aware non-linear codebook) — Epic 4.
//
// Scope decision (see EPIC4_I_QUANTS_REPORT.md): this milestone
// delivers full encoder + decoder for IQ4_NL and IQ4_XS, both of
// which use a 16-entry non-linear codebook of signed int8 values.
// The codebook-based 1/2/3-bit families (IQ2_XXS, IQ2_XS, IQ2_S,
// IQ3_XXS, IQ3_S, IQ1_S, IQ1_M) require fixed lookup tables of
// 8-element grid points and a non-trivial offline NN-search encoder;
// they are deferred to a future R&D-2 milestone that pairs the
// In-Register LUT encoder design with codebook-data embedding.
// ============================================================================

inline constexpr core::usize kIQ4_NLBlockElements = 32u;
inline constexpr core::usize kIQ4_NLBlockBytes    = 2u + 16u;        // d + qs[16]
inline constexpr core::usize kIQ4_XSBlockBytes    = 2u + 2u + 4u + 128u; // d + scales_h + scales_l + qs

static_assert(kIQ4_NLBlockBytes == 18);
static_assert(kIQ4_XSBlockBytes == 136);

namespace detail_iquant {

/// The 16-entry non-linear codebook used by IQ4_NL and IQ4_XS.
/// Values are signed int8 with non-uniform spacing concentrated near
/// zero — chosen to match the empirical Gaussian-like distribution
/// of LLM weights. Source: ggml `kvalues_iq4nl` in ggml-quants.c.
inline constexpr int8_t kIQ4_Codebook[16] = {
    -127, -104, -83, -65,
    -49,  -35,  -22, -10,
       1,   13,   25,  38,
      53,   69,   89, 113
};
inline constexpr int8_t kIQ4_MinValue = -127;   // kIQ4_Codebook[0]

/// Find the index of the codebook entry closest to a signed FP value.
/// Branch-free 16-way linear scan; chosen over binary search because
/// the codebook is small and a linear loop is straightforward for the
/// auto-vectoriser. Returns an index in [0, 15].
[[nodiscard]] inline core::u8 bestIndexIQ4(float x) noexcept {
    // Saturate at the boundaries.
    if (x <= static_cast<float>(kIQ4_Codebook[0]))  return 0u;
    if (x >= static_cast<float>(kIQ4_Codebook[15])) return 15u;
    // Find the codepoint with minimum |value - x|.
    core::u8  best_idx = 0u;
    float     best_err = std::fabs(x - static_cast<float>(kIQ4_Codebook[0]));
    for (core::u8 i = 1u; i < 16u; ++i) {
        const float err = std::fabs(x - static_cast<float>(kIQ4_Codebook[i]));
        if (err < best_err) {
            best_err = err;
            best_idx = i;
        }
    }
    return best_idx;
}

}  // namespace detail_iquant

// ============================================================================
// IQ4_NL — 4-bit non-linear, 32-element blocks.
//
// Wire format (verbatim ggml `block_iq4_nl`, 18 bytes):
//   d           : FP16 per-block scale                                (offset 0)
//   qs[16]      : 4-bit codebook indices                              (offset 2)
//
// Element layout: qs[j] holds the index for element j in the low
// nibble, element j+16 in the high nibble.
//
// Encoding (R&D-2 single-pass, no inner refinement):
//   max = arg max_v |v|     (signed extreme, sign preserved)
//   d   = max / kIQ4_MinValue  (= max / -127)
//   For each element: idx = bestIndexIQ4(x / d)
//
// Decoding (verbatim ggml):
//   x = d * kIQ4_Codebook[idx]
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ4_NL(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kIQ4_NLBlockElements) != 0u) return 0u;
    return (fp32_element_count / kIQ4_NLBlockElements) * kIQ4_NLBlockBytes;
}

static_assert(compressedSize_IQ4_NL(32u)   == 18u);
static_assert(compressedSize_IQ4_NL(1024u) == 576u);

[[nodiscard]] inline core::usize compressFP32ToIQ4_NL(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kIQ4_NLBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kIQ4_NLBlockElements;
    const core::usize need_bytes = n_blocks * kIQ4_NLBlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kIQ4_NLBlockElements;

        // Find signed extreme.
        float amax = 0.0f;
        float vmax = 0.0f;
        for (core::usize j = 0; j < kIQ4_NLBlockElements; ++j) {
            const float v = core::sanitiseFinite(blk[j]);
            const float a = std::fabs(v);
            if (a > amax) { amax = a; vmax = v; }
        }

        // d = vmax / values[0] = vmax / -127.
        // For zero blocks (amax = 0), force d = 0 -> encoder produces
        // all zeros (codebook index for 0 is between 7 (-10) and 8 (1);
        // bestIndexIQ4(0) gives idx=8 -> dequant = d * 1 = 0).
        float d = 0.0f;
        if (amax > 0.0f) {
            d = vmax / static_cast<float>(detail_iquant::kIQ4_MinValue);
        }
        detail::storeU16(out, core::fp32ToFP16Safe(d));

        // Per-element quantization.
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        for (core::usize j = 0; j < kIQ4_NLBlockElements / 2u; ++j) {
            const core::u8 n0 = detail_iquant::bestIndexIQ4(
                core::sanitiseFinite(blk[j])         * id);
            const core::u8 n1 = detail_iquant::bestIndexIQ4(
                core::sanitiseFinite(blk[j + 16u])   * id);
            out[2u + j] = static_cast<std::byte>(
                (n0 & 0x0Fu) | (static_cast<core::u8>(n1 & 0x0Fu) << 4));
        }
        out += kIQ4_NLBlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ4_NL_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ4_NLBlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ4_NLBlockBytes;
    const core::usize n_elements = n_blocks * kIQ4_NLBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;

        for (core::usize j = 0; j < kIQ4_NLBlockElements / 2u; ++j) {
            const core::u8 packed = static_cast<core::u8>(in[2u + j]);
            const core::u8 n0 = packed & 0x0Fu;
            const core::u8 n1 = (packed >> 4) & 0x0Fu;
            dst_floats[b * kIQ4_NLBlockElements + j]       =
                d * static_cast<float>(detail_iquant::kIQ4_Codebook[n0]);
            dst_floats[b * kIQ4_NLBlockElements + j + 16u] =
                d * static_cast<float>(detail_iquant::kIQ4_Codebook[n1]);
        }
        in += kIQ4_NLBlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// IQ4_XS — 4-bit non-linear, 256-element super-blocks.
//
// Wire format (verbatim ggml `block_iq4_xs`, 136 bytes):
//   d           : FP16 super-block scale                              (offset   0)
//   scales_h    : uint16 — high 2 bits of 8 sub-block scales          (offset   2)
//   scales_l[4] : low 4 bits of 8 sub-block scales (2 per byte)       (offset   4)
//   qs[128]     : 4-bit codebook indices for 256 elements             (offset   8)
//
// Per-sub-block signed scale: `ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xf
//                               | ((scales_h >> 2*ib) & 3) << 4`
// then `local_d = d * (ls - 32)`. ls in [0, 63] -> signed [-32, 31].
//
// Encoding (R&D-2 single-pass):
//   Per sub-block of 32 elements:
//     scale_sub_i = signed_max_sub_i / -127     (in FP32; values[0] = -127)
//   max_scale = arg_max_scale |scale_sub_i|
//   d_super = max_scale / -32
//   q_scale_i = clamp(round(scale_sub_i / d_super), -32, 31)
//   Per element: idx = bestIndexIQ4(x / (d_super * q_scale_i))
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ4_XS(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ4_XSBlockBytes;
}

static_assert(compressedSize_IQ4_XS(256u)  == 136u);
static_assert(compressedSize_IQ4_XS(1024u) == 544u);

[[nodiscard]] inline core::usize compressFP32ToIQ4_XS(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_elements = src.size() / kFP32Size;
    const core::usize n_blocks   = n_elements / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ4_XSBlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    std::array<float, 8> sub_scales{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Step 1: per-sub-block signed scale.
        float max_abs_scale = 0.0f;
        float max_scale     = 0.0f;
        for (core::usize ib = 0; ib < 8u; ++ib) {
            const float vmax = detail_kquant::subBlockSignedAbsMax(blk + 32u * ib, 32u);
            const float scale = vmax / static_cast<float>(detail_iquant::kIQ4_MinValue);
            sub_scales[ib] = scale;
            const float ascale = std::fabs(scale);
            if (ascale > max_abs_scale) {
                max_abs_scale = ascale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ4_XSBlockBytes);
            out += kIQ4_XSBlockBytes;
            continue;
        }

        // Step 2-3: super-block scale + 6-bit per-sub-block scales.
        const float iscale  = -32.0f / max_scale;
        const float d_super = 1.0f / iscale;

        // Pack 8 sub-block scales (6 bits each, signed, stored as [0, 63] = ls + 32).
        std::array<core::u8, 8> q_scales_shifted{};   // values in [0, 63]
        for (core::usize ib = 0; ib < 8u; ++ib) {
            int l = detail_kquant::nearestInt(iscale * sub_scales[ib]);
            if (l < -32) l = -32;
            if (l >  31) l =  31;
            q_scales_shifted[ib] = static_cast<core::u8>(l + 32);
        }

        // scales_l[4]: 8 sub-blocks, 4 low bits each, packed 2 per byte.
        // scales_h (uint16): 8 sub-blocks, 2 high bits each = 16 bits.
        std::byte* scales_l = out + 4u;
        for (core::usize i = 0; i < 4u; ++i) scales_l[i] = std::byte{0};
        core::u16 scales_h = 0u;
        for (core::usize ib = 0; ib < 8u; ++ib) {
            const core::u8 low4 = q_scales_shifted[ib] & 0x0Fu;
            const core::u8 hi2  = static_cast<core::u8>((q_scales_shifted[ib] >> 4) & 0x03u);
            // ggml decode: `(scales_l[ib/2] >> 4*(ib%2)) & 0xf` -> at bit (4*(ib%2)).
            scales_l[ib / 2u] = static_cast<std::byte>(
                static_cast<core::u8>(scales_l[ib / 2u]) |
                (low4 << (4u * (ib % 2u))));
            scales_h = static_cast<core::u16>(scales_h |
                (static_cast<core::u16>(hi2) << (2u * ib)));
        }
        detail::storeU16(out + 2u, scales_h);

        // Per-element quantization (with quantized scale round-trip).
        std::byte* qs_p = out + 8u;
        for (core::usize ib = 0; ib < 8u; ++ib) {
            const int sc_signed =
                static_cast<int>(q_scales_shifted[ib]) - 32;
            const float local_d = d_super * static_cast<float>(sc_signed);
            const float id = (local_d != 0.0f) ? 1.0f / local_d : 0.0f;
            // 32-element sub-block: pairs of nibbles packed into 16 bytes.
            for (core::usize j = 0; j < 16u; ++j) {
                const core::u8 n0 = detail_iquant::bestIndexIQ4(
                    core::sanitiseFinite(blk[32u * ib + j])         * id);
                const core::u8 n1 = detail_iquant::bestIndexIQ4(
                    core::sanitiseFinite(blk[32u * ib + j + 16u])   * id);
                qs_p[16u * ib + j] = static_cast<std::byte>(
                    (n0 & 0x0Fu) | (static_cast<core::u8>(n1 & 0x0Fu) << 4));
            }
        }

        // FP16 d at offset 0.
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_super));

        out += kIQ4_XSBlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ4_XS_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ4_XSBlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ4_XSBlockBytes;
    const core::usize n_elements = n_blocks * kKBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;
        const core::u16 scales_h = detail::loadU16(in + 2u);
        const std::byte* scales_l = in + 4u;
        const std::byte* qs_p     = in + 8u;

        float* y = dst_floats + b * kKBlockElements;
        for (core::usize ib = 0; ib < 8u; ++ib) {
            const int ls = static_cast<int>(
                (static_cast<core::u8>(scales_l[ib / 2u]) >> (4u * (ib % 2u))) & 0x0Fu) |
                (static_cast<int>((scales_h >> (2u * ib)) & 0x03u) << 4);
            const float dl = d * static_cast<float>(ls - 32);
            for (core::usize j = 0; j < 16u; ++j) {
                const core::u8 packed = static_cast<core::u8>(qs_p[16u * ib + j]);
                const core::u8 n0 = packed & 0x0Fu;
                const core::u8 n1 = (packed >> 4) & 0x0Fu;
                y[32u * ib + j]        =
                    dl * static_cast<float>(detail_iquant::kIQ4_Codebook[n0]);
                y[32u * ib + j + 16u]  =
                    dl * static_cast<float>(detail_iquant::kIQ4_Codebook[n1]);
            }
        }

        in += kIQ4_XSBlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// Codebook-based I-Quants — IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M.
//
// These formats encode 8-element groups as indices into a fixed lookup
// table of grid points (see iq_codebooks.hpp). The grids are packed as
// uint64_t (8 int8 codepoints each) for IQ2_*/IQ1_*, or uint32_t (4
// int8 codepoints each) for IQ3_*. The codebook data is byte-for-byte
// identical to ggml-common.h, so any ggml-encoded block can be decoded
// by QTX (decoders below are direct transcriptions of ggml's
// `dequantize_row_iq*` functions).
//
// Encoders use a single-pass O(N) heuristic per the brief's R&D-2
// directive:
//   1. Per 8-element group: pre-flip signs so all values are
//      non-negative (the codebook stores absolute magnitudes; sign
//      mask carries the polarity).
//   2. Brute-force the 256/512/1024 codebook entries against the
//      sign-flipped group, picking the nearest by L2 distance.
//      (Epic 6: replace with SIMD PSHUFB lookup + tournament reduction.)
//   3. Derive the per-block scale from the chosen grid magnitudes.
// Brute force is O(256) per group instead of ggml's O(1) hashed lookup,
// but does not require ggml_quantize_init()'s precomputed neighbour
// tables — the QTX codec is fully self-contained.
//
// Wire-format constants (all verified by static_assert against ggml).
// ============================================================================

inline constexpr core::usize kIQ2_XXS_BlockBytes = 2u + 32u * 2u;          // d + qs[32 u16]
inline constexpr core::usize kIQ2_XS_BlockBytes  = 2u + 32u * 2u + 8u;     // d + qs + scales[8]
inline constexpr core::usize kIQ2_S_BlockBytes   = 2u + 64u + 16u;         // d + qs[64] + qh[16]+scales[8] ... wait check
// Recompute IQ2_S: d(2) + qs[QK_K/4 = 64] + qh[QK_K/32 = 8 — wait struct says qh[QK_K/32] but
// QK_K/32=8 for QK_K=256, qh array size = 8, plus scales[QK_K/32]=8.
// Actually struct says qh and scales separately. Let me recompute.
// block_iq2_s: ggml_half(2) + qs[QK_K/4=64] + qh[QK_K/32=8] + scales[QK_K/32=8]
//            = 2 + 64 + 8 + 8 = 82. But comment says 2.5625 bpw -> 256*2.5625/8 = 82. OK.

inline constexpr core::usize kIQ3_XXS_BlockBytes = 2u + 3u * (256u / 8u);  // d + qs[3*QK_K/8 = 96]
inline constexpr core::usize kIQ3_S_BlockBytes   = 2u + 13u * (256u / 32u) + 4u;
                                                  // d(2) + qs[QK_K/4=64] + qh[QK_K/32=8]
                                                  // + signs[QK_K/8=32] + scales[QK_K/64=4]
                                                  // = 2 + 64 + 8 + 32 + 4 = 110
inline constexpr core::usize kIQ1_S_BlockBytes   = 2u + 32u + 16u;         // d + qs[32] + qh[16]
inline constexpr core::usize kIQ1_M_BlockBytes   = 32u + 16u + 8u;         // qs + qh + scales (no d)

static_assert(kIQ2_XXS_BlockBytes == 66);    // 2.0625 bpw
static_assert(kIQ2_XS_BlockBytes  == 74);    // 2.3125 bpw
// kIQ2_S recomputed inline below.
static_assert(kIQ3_XXS_BlockBytes == 98);    // 3.0625 bpw
static_assert(kIQ3_S_BlockBytes   == 110);   // 3.4375 bpw
static_assert(kIQ1_S_BlockBytes   == 50);    // 1.5625 bpw
static_assert(kIQ1_M_BlockBytes   == 56);    // 1.75 bpw

// Override IQ2_S with the correctly-computed value (82 = 2.5625 bpw).
inline constexpr core::usize kIQ2_S_BlockBytes_v2 = 2u + 64u + 8u + 8u;
static_assert(kIQ2_S_BlockBytes_v2 == 82);

namespace detail_iquant_cb {

using iq_codebooks::kmask_iq2xs;
using iq_codebooks::ksigns_iq2xs;
using iq_codebooks::iq2xxs_grid;
using iq_codebooks::iq2xs_grid;
using iq_codebooks::iq2s_grid;
using iq_codebooks::iq3xxs_grid;
using iq_codebooks::iq3s_grid;
using iq_codebooks::iq1s_grid;
using iq_codebooks::kIQ1S_Delta;

/// Find the best 8-element grid entry in a uint64_t-packed codebook of
/// `n` entries for a given 8-element absolute-magnitude target.
/// Returns the index in [0, n).
///
/// `target[j]` must be non-negative (signs are handled externally via
/// the `ksigns_iq2xs` mask). The distance metric is L2 over the 8
/// dimensions of |target| vs codebook[idx].byte[j] (signed int8 cast).
///
/// O(n) brute force — 256 ops for iq2xxs, 512 for iq2xs/iq3s, 1024
/// for iq2s. Fast enough for a reference encoder; SIMD lookup
/// is Epic 6 territory.
[[nodiscard]] inline core::usize bestGridIndexU64(
    const std::uint64_t* grid, core::usize n,
    const float target[8]) noexcept
{
    core::usize best_idx = 0;
    float best_err = std::numeric_limits<float>::infinity();
    for (core::usize i = 0; i < n; ++i) {
        const std::uint64_t entry = grid[i];
        float err = 0.0f;
        for (core::usize j = 0; j < 8u; ++j) {
            const auto byte = static_cast<std::uint8_t>((entry >> (8u * j)) & 0xFFu);
            // Codebook bytes are non-negative (codebook stores magnitudes).
            const float gv = static_cast<float>(byte);
            const float d  = target[j] - gv;
            err += d * d;
        }
        if (err < best_err) {
            best_err = err;
            best_idx = i;
        }
    }
    return best_idx;
}

/// Same as bestGridIndexU64 but for 4-element groups (used by IQ3_XXS / IQ3_S).
[[nodiscard]] inline core::usize bestGridIndexU32(
    const std::uint32_t* grid, core::usize n,
    const float target[4]) noexcept
{
    core::usize best_idx = 0;
    float best_err = std::numeric_limits<float>::infinity();
    for (core::usize i = 0; i < n; ++i) {
        const std::uint32_t entry = grid[i];
        float err = 0.0f;
        for (core::usize j = 0; j < 4u; ++j) {
            const auto byte = static_cast<std::uint8_t>((entry >> (8u * j)) & 0xFFu);
            const float gv = static_cast<float>(byte);
            const float d  = target[j] - gv;
            err += d * d;
        }
        if (err < best_err) {
            best_err = err;
            best_idx = i;
        }
    }
    return best_idx;
}

/// Find the signs-of-magnitude index `s` ∈ [0, 127] such that
/// `ksigns_iq2xs[s]` matches the 8-element sign pattern of `xs[0..7]`.
/// ksigns_iq2xs[s] is an 8-bit mask where each bit corresponds to a
/// negative element; bit j set => xs[j] < 0.
[[nodiscard]] inline core::u8 signsIndexFor8(const float xs[8]) noexcept {
    core::u8 mask = 0u;
    for (core::usize j = 0; j < 8u; ++j) {
        if (xs[j] < 0.0f) mask = static_cast<core::u8>(mask | (1u << j));
    }
    // Match against ksigns_iq2xs table to find the 7-bit index.
    for (core::usize s = 0; s < 128u; ++s) {
        if (ksigns_iq2xs[s] == mask) return static_cast<core::u8>(s);
    }
    // No match (mask has odd parity) -> ggml convention: flip the
    // bit with smallest magnitude to force even parity, then re-look.
    // The encoder above already calls signsIndexFor8 only after parity
    // correction (see compressFP32ToIQ2_XXS); this branch is defensive.
    return 0u;
}

}  // namespace detail_iquant_cb

// ============================================================================
// IQ2_XXS — "true" 2-bit (2.0625 bpw). Block layout (66 B):
//   d              FP16                                       (offset 0)
//   qs[32]         uint16_t — packed grid/signs/scale         (offset 2)
//
// Per 32 elements (4 sub-blocks of 8): 8 bytes = aux32[0..1]
//   aux32[0]       4 × 8-bit grid indices
//   aux32[1]       4 × 7-bit signs + 4-bit per-32-elem scale (top 4 bits)
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ2_XXS(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ2_XXS_BlockBytes;
}
static_assert(compressedSize_IQ2_XXS(256u)  == 66u);
static_assert(compressedSize_IQ2_XXS(1024u) == 264u);

[[nodiscard]] inline core::usize compressFP32ToIQ2_XXS(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_blocks   = (src.size() / kFP32Size) / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ2_XXS_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    // Phase-1 per-ib32 outputs (kept on the stack for the super-block).
    struct Ib32Plan {
        float    db;                // chosen real-valued sub-scale (= magnitude scale)
        core::u8 grid_idx[4];       // 4 × 8-bit grid indices
        core::u8 signs[4];          // 4 × 7-bit sign codes
    };
    std::array<Ib32Plan, 8u> plans{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Phase 1: per ib32, find the best (db, 4 grid indices, 4 signs).
        // db is a free-floating positive scale; we'll quantize it to 4
        // bits later relative to the super-block max.
        float max_db = 0.0f;
        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const float* xb = blk + 32u * ib32;
            float xval[32]{};
            core::u8 signs[4]{};

            // Pre-flip signs (parity correction included).
            for (core::usize k = 0; k < 4u; ++k) {
                core::u8 s = 0u;
                int nflip = 0;
                for (core::usize j = 0; j < 8u; ++j) {
                    const float v = core::sanitiseFinite(xb[8u * k + j]);
                    if (v >= 0.0f) xval[8u * k + j] = v;
                    else { xval[8u * k + j] = -v; ++nflip; s = static_cast<core::u8>(s | (1u << j)); }
                }
                if ((nflip % 2) != 0) {
                    core::usize imin = 0; float vmin = xval[8u * k + 0];
                    for (core::usize j = 1; j < 8u; ++j)
                        if (xval[8u * k + j] < vmin) { vmin = xval[8u * k + j]; imin = j; }
                    xval[8u * k + imin] = -xval[8u * k + imin];
                    s = static_cast<core::u8>(s ^ (1u << imin));
                }
                plans[ib32].signs[k] = static_cast<core::u8>(s & 0x7Fu);
                (void)signs;
            }

            // ggml chooses db so that the largest |xval| in the ib32 maps
            // approximately to grid byte 25 (the median magnitude). This
            // is a heuristic; the exact ggml value comes from
            // make_qp_quants. We use the same target: db ≈ gmax / 25.
            float gmax = 0.0f;
            for (core::usize j = 0; j < 32u; ++j)
                if (xval[j] > gmax) gmax = xval[j];
            const float db = (gmax > 1e-30f) ? gmax / 25.0f : 0.0f;
            const float id = (db > 0.0f) ? 1.0f / db : 0.0f;
            plans[ib32].db = db;
            if (db > max_db) max_db = db;

            // Find best grid index for each of 4 groups of 8.
            for (core::usize k = 0; k < 4u; ++k) {
                float target[8];
                for (core::usize j = 0; j < 8u; ++j) {
                    target[j] = xval[8u * k + j] * id;
                }
                plans[ib32].grid_idx[k] = static_cast<core::u8>(
                    detail_iquant_cb::bestGridIndexU64(
                        detail_iquant_cb::iq2xxs_grid, 256u, target));
            }
        }

        // Phase 2: super-block d from max_db.
        // Decoder: db_decoded = d * (0.5 + s_4bit) / 4.
        // We want db_decoded ≤ db_real for stability — pick d so that
        // s=15 reproduces max_db: d = max_db * 4 / 15.5.
        if (max_db < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ2_XXS_BlockBytes);
            out += kIQ2_XXS_BlockBytes;
            continue;
        }
        const float d_super = max_db * 4.0f / 15.5f;

        // Quantize each plan's db to a 4-bit scale.
        std::byte* qs_p = out + 2u;
        std::memset(qs_p, 0, 64);
        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            int s4 = 0;
            if (plans[ib32].db > 0.0f) {
                // s = (db_real * 4 / d_super) - 0.5
                const float sf = plans[ib32].db * 4.0f / d_super - 0.5f;
                s4 = detail_kquant::nearestInt(sf);
                if (s4 < 0)  s4 = 0;
                if (s4 > 15) s4 = 15;
            }
            std::uint32_t aux32_0 = 0;
            std::uint32_t aux32_1 = 0;
            for (core::usize k = 0; k < 4u; ++k) {
                aux32_0 |= (static_cast<std::uint32_t>(plans[ib32].grid_idx[k]) << (8u * k));
                aux32_1 |= (static_cast<std::uint32_t>(plans[ib32].signs[k]) << (7u * k));
            }
            aux32_1 |= (static_cast<std::uint32_t>(s4 & 0xFu) << 28);
            std::memcpy(qs_p + 8u * ib32 + 0u, &aux32_0, sizeof(std::uint32_t));
            std::memcpy(qs_p + 8u * ib32 + 4u, &aux32_1, sizeof(std::uint32_t));
        }
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_super));
        out += kIQ2_XXS_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ2_XXS_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ2_XXS_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ2_XXS_BlockBytes;
    const core::usize n_elements = n_blocks * kKBlockElements;
    const core::usize need_bytes = n_elements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;
        const std::byte* qs_p = in + 2u;
        float* y = dst_floats + b * kKBlockElements;

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            std::uint32_t aux32[2];
            std::memcpy(aux32, qs_p + 8u * ib32, 2 * sizeof(std::uint32_t));
            const auto aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
            const float db = d * (0.5f + static_cast<float>(aux32[1] >> 28)) * 0.25f;
            for (core::usize l = 0; l < 4u; ++l) {
                const auto grid = reinterpret_cast<const std::uint8_t*>(
                    detail_iquant_cb::iq2xxs_grid + aux8[l]);
                const core::u8 sign_byte = detail_iquant_cb::ksigns_iq2xs[
                    (aux32[1] >> (7u * l)) & 0x7Fu];
                for (core::usize j = 0; j < 8u; ++j) {
                    const float gv = static_cast<float>(grid[j]);
                    const float s = (sign_byte & detail_iquant_cb::kmask_iq2xs[j])
                                  ? -1.0f : 1.0f;
                    *y++ = db * gv * s;
                }
            }
        }
        in += kIQ2_XXS_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// IQ2_XS — 2.3125 bpw. Block layout (74 B):
//   d                FP16                                     (offset  0)
//   qs[32]           uint16_t — each: 9-bit grid idx + 7-bit signs   (offset 2)
//   scales[8]        uint8_t — 2 × 4-bit per ib32 (4 sub-blocks share)(offset 66)
// Per ib32 (32 elems): 4 × u16 = 8 bytes (qs); plus 1 byte scale (2 nibbles).
// Grid is 512-entry (9-bit index).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ2_XS(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ2_XS_BlockBytes;
}
static_assert(compressedSize_IQ2_XS(256u) == 74u);

[[nodiscard]] inline core::usize compressFP32ToIQ2_XS(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_blocks   = (src.size() / kFP32Size) / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ2_XS_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    struct Ib32Plan {
        float         db;
        std::uint16_t grid_idx[4];
        core::u8      signs[4];
    };
    std::array<Ib32Plan, 8u> plans{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Phase 1: free-float per-ib32 (db, grid indices, signs).
        float max_db = 0.0f;
        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const float* xb = blk + 32u * ib32;
            float xval[32]{};
            for (core::usize k = 0; k < 4u; ++k) {
                core::u8 s = 0u;
                int nflip = 0;
                for (core::usize j = 0; j < 8u; ++j) {
                    const float v = core::sanitiseFinite(xb[8u * k + j]);
                    if (v >= 0.0f) xval[8u * k + j] = v;
                    else { xval[8u * k + j] = -v; ++nflip; s = static_cast<core::u8>(s | (1u << j)); }
                }
                if ((nflip % 2) != 0) {
                    core::usize imin = 0; float vmin = xval[8u * k + 0];
                    for (core::usize j = 1; j < 8u; ++j)
                        if (xval[8u * k + j] < vmin) { vmin = xval[8u * k + j]; imin = j; }
                    xval[8u * k + imin] = -xval[8u * k + imin];
                    s = static_cast<core::u8>(s ^ (1u << imin));
                }
                plans[ib32].signs[k] = static_cast<core::u8>(s & 0x7Fu);
            }
            float gmax = 0.0f;
            for (core::usize j = 0; j < 32u; ++j)
                if (xval[j] > gmax) gmax = xval[j];
            const float db = (gmax > 1e-30f) ? gmax / 25.0f : 0.0f;
            const float id = (db > 0.0f) ? 1.0f / db : 0.0f;
            plans[ib32].db = db;
            if (db > max_db) max_db = db;

            for (core::usize l = 0; l < 4u; ++l) {
                float target[8];
                for (core::usize j = 0; j < 8u; ++j) {
                    target[j] = xval[8u * l + j] * id;
                }
                plans[ib32].grid_idx[l] = static_cast<std::uint16_t>(
                    detail_iquant_cb::bestGridIndexU64(
                        detail_iquant_cb::iq2xs_grid, 512u, target));
            }
        }

        if (max_db < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ2_XS_BlockBytes);
            out += kIQ2_XS_BlockBytes;
            continue;
        }
        const float d_super = max_db * 4.0f / 15.5f;

        // Phase 2: pack scales and indices.
        std::byte* qs_p = out + 2u;
        std::byte* sc_p = out + 66u;
        std::memset(qs_p, 0, 64u);
        std::memset(sc_p, 0, 8u);

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            int s4 = 0;
            if (plans[ib32].db > 0.0f) {
                const float sf = plans[ib32].db * 4.0f / d_super - 0.5f;
                s4 = detail_kquant::nearestInt(sf);
                if (s4 < 0)  s4 = 0;
                if (s4 > 15) s4 = 15;
            }
            sc_p[ib32] = static_cast<std::byte>(
                (static_cast<core::u8>(s4) & 0x0Fu) |
                ((static_cast<core::u8>(s4) & 0x0Fu) << 4));
            for (core::usize l = 0; l < 4u; ++l) {
                const std::uint16_t packed = static_cast<std::uint16_t>(
                    (plans[ib32].grid_idx[l] & 0x1FFu) |
                    (static_cast<std::uint16_t>(plans[ib32].signs[l]) << 9));
                detail::storeU16(qs_p + 2u * (4u * ib32 + l), packed);
            }
        }
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_super));
        out += kIQ2_XS_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ2_XS_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ2_XS_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ2_XS_BlockBytes;
    const core::usize need_bytes = n_blocks * kKBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;
        const std::byte* qs_p = in + 2u;
        const std::byte* sc_p = in + 66u;
        float* y = dst_floats + b * kKBlockElements;

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const core::u8 sc = static_cast<core::u8>(sc_p[ib32]);
            float db[2];
            db[0] = d * (0.5f + static_cast<float>(sc & 0x0Fu)) * 0.25f;
            db[1] = d * (0.5f + static_cast<float>(sc >> 4))    * 0.25f;
            for (core::usize l = 0; l < 4u; ++l) {
                const std::uint16_t packed =
                    detail::loadU16(qs_p + 2u * (4u * ib32 + l));
                const auto grid = reinterpret_cast<const std::uint8_t*>(
                    detail_iquant_cb::iq2xs_grid + (packed & 0x1FFu));
                const core::u8 sign_byte = detail_iquant_cb::ksigns_iq2xs[packed >> 9];
                for (core::usize j = 0; j < 8u; ++j) {
                    const float gv = static_cast<float>(grid[j]);
                    const float s = (sign_byte & detail_iquant_cb::kmask_iq2xs[j])
                                  ? -1.0f : 1.0f;
                    *y++ = db[l / 2u] * gv * s;
                }
            }
        }
        in += kIQ2_XS_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// IQ2_S — 2.5625 bpw. Block layout (82 B):
//   d           FP16                                          (offset  0)
//   qs[64]      uint8_t — low 8 bits of grid idx              (offset  2)
//   qh[8]       uint8_t — 2 high bits of grid idx per group (4 groups/byte)
//                                                              (offset 66)
//   scales[8]   uint8_t — 2 × 4-bit per ib32                  (offset 74)
//   signs[32]   uint8_t — 8-bit sign masks (note: in ggml decode, signs
//               are read from qs + QK_K/8 -- i.e., they overlap with the
//               "qs" region at offset 2 + 64 = ... wait, that means
//               signs is INSIDE qs region. Re-read struct.)
//
// Re-reading ggml block_iq2_s:
//   ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32];
//   = 2 + 64 + 8 + 8 = 82.
// And decoder: `const uint8_t * signs = qs + QK_K/8` (= qs + 32). So
// qs[0..31] is grid-low-bits, qs[32..63] is signs. That makes the
// per-ib32 layout: 4 qs bytes (low grid bits) + 4 signs bytes (full
// 8-bit signs, NOT a 7-bit ksigns_iq2xs index).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ2_S(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ2_S_BlockBytes_v2;
}
static_assert(compressedSize_IQ2_S(256u) == 82u);

[[nodiscard]] inline core::usize compressFP32ToIQ2_S(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_blocks   = (src.size() / kFP32Size) / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ2_S_BlockBytes_v2;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    struct Ib32Plan {
        float         db;
        std::uint16_t grid_idx[4];   // 10-bit index into 1024-entry grid
        core::u8      signs[4];      // 8-bit sign masks (NOT 7-bit ksigns)
    };
    std::array<Ib32Plan, 8u> plans{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Phase 1.
        float max_db = 0.0f;
        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const float* xb = blk + 32u * ib32;
            float xval[32]{};
            for (core::usize k = 0; k < 4u; ++k) {
                core::u8 s = 0u;
                for (core::usize j = 0; j < 8u; ++j) {
                    const float v = core::sanitiseFinite(xb[8u * k + j]);
                    if (v >= 0.0f) xval[8u * k + j] = v;
                    else { xval[8u * k + j] = -v; s = static_cast<core::u8>(s | (1u << j)); }
                }
                plans[ib32].signs[k] = s;
            }
            float gmax = 0.0f;
            for (core::usize j = 0; j < 32u; ++j)
                if (xval[j] > gmax) gmax = xval[j];
            const float db = (gmax > 1e-30f) ? gmax / 25.0f : 0.0f;
            const float id = (db > 0.0f) ? 1.0f / db : 0.0f;
            plans[ib32].db = db;
            if (db > max_db) max_db = db;

            for (core::usize l = 0; l < 4u; ++l) {
                float target[8];
                for (core::usize j = 0; j < 8u; ++j) {
                    target[j] = xval[8u * l + j] * id;
                }
                plans[ib32].grid_idx[l] = static_cast<std::uint16_t>(
                    detail_iquant_cb::bestGridIndexU64(
                        detail_iquant_cb::iq2s_grid, 1024u, target));
            }
        }

        if (max_db < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ2_S_BlockBytes_v2);
            out += kIQ2_S_BlockBytes_v2;
            continue;
        }
        const float d_super = max_db * 4.0f / 15.5f;

        // Phase 2.
        std::byte* qs_p = out + 2u;
        std::byte* qh_p = out + 66u;
        std::byte* sc_p = out + 74u;
        std::memset(qs_p, 0, 64);
        std::memset(qh_p, 0, 8);
        std::memset(sc_p, 0, 8);

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            int s4 = 0;
            if (plans[ib32].db > 0.0f) {
                const float sf = plans[ib32].db * 4.0f / d_super - 0.5f;
                s4 = detail_kquant::nearestInt(sf);
                if (s4 < 0)  s4 = 0;
                if (s4 > 15) s4 = 15;
            }
            sc_p[ib32] = static_cast<std::byte>(
                (static_cast<core::u8>(s4) & 0x0Fu) |
                ((static_cast<core::u8>(s4) & 0x0Fu) << 4));
            core::u8 qh_byte = 0u;
            for (core::usize l = 0; l < 4u; ++l) {
                const std::uint16_t gidx = plans[ib32].grid_idx[l];
                qs_p[4u * ib32 + l] = static_cast<std::byte>(gidx & 0xFFu);
                const core::u8 hi2 = static_cast<core::u8>((gidx >> 8) & 0x03u);
                qh_byte = static_cast<core::u8>(qh_byte | (hi2 << (2u * l)));
                qs_p[32u + 4u * ib32 + l] = static_cast<std::byte>(plans[ib32].signs[l]);
            }
            qh_p[ib32] = static_cast<std::byte>(qh_byte);
        }
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_super));
        out += kIQ2_S_BlockBytes_v2;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ2_S_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ2_S_BlockBytes_v2 != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ2_S_BlockBytes_v2;
    const core::usize need_bytes = n_blocks * kKBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;
        const std::byte* qs_p    = in + 2u;
        const std::byte* qh_p    = in + 66u;
        const std::byte* sc_p    = in + 74u;
        const std::byte* signs_p = qs_p + 32u;  // signs alias qs[32..63]
        float* y = dst_floats + b * kKBlockElements;

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const core::u8 sc = static_cast<core::u8>(sc_p[ib32]);
            float db[2];
            db[0] = d * (0.5f + static_cast<float>(sc & 0x0Fu)) * 0.25f;
            db[1] = d * (0.5f + static_cast<float>(sc >> 4))    * 0.25f;
            const core::u8 qh = static_cast<core::u8>(qh_p[ib32]);
            for (core::usize l = 0; l < 4u; ++l) {
                const std::uint16_t gidx = static_cast<std::uint16_t>(
                    static_cast<core::u8>(qs_p[4u * ib32 + l]) |
                    (static_cast<std::uint16_t>((qh >> (2u * l)) & 0x03u) << 8));
                const auto grid = reinterpret_cast<const std::uint8_t*>(
                    detail_iquant_cb::iq2s_grid + gidx);
                const core::u8 signs = static_cast<core::u8>(signs_p[4u * ib32 + l]);
                for (core::usize j = 0; j < 8u; ++j) {
                    const float gv = static_cast<float>(grid[j]);
                    const float s = (signs & detail_iquant_cb::kmask_iq2xs[j])
                                  ? -1.0f : 1.0f;
                    *y++ = db[l / 2u] * gv * s;
                }
            }
        }
        in += kIQ2_S_BlockBytes_v2;
    }
    return need_bytes;
}

// ============================================================================
// IQ3_XXS — 3.0625 bpw. Block (98 B):
//   d                  FP16                                   (offset  0)
//   qs[3*QK_K/8 = 96]  uint8_t —
//     bytes 0..63    : 64 × 8-bit grid indices (4-element groups, 32 ib32 ×2)
//     bytes 64..95   : per-ib32 scale_and_signs (4 bytes each × 8 = 32)
//
// Per ib32 (32 elems = 8 groups-of-4): 8 grid-index bytes + 4 scale_and_signs bytes.
// scale_and_signs layout: top 4 bits = 4-bit scale (across the entire ib32);
//                          lower 28 bits = 4 × 7-bit signs (one per pair of groups).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ3_XXS(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ3_XXS_BlockBytes;
}
static_assert(compressedSize_IQ3_XXS(256u) == 98u);

[[nodiscard]] inline core::usize compressFP32ToIQ3_XXS(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_blocks   = (src.size() / kFP32Size) / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ3_XXS_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    struct Ib32Plan {
        float    db;
        core::u8 grid_idx[8];   // 8 × 8-bit grid indices (4-element groups)
        core::u8 signs[4];      // 4 × 7-bit
    };
    std::array<Ib32Plan, 8u> plans{};

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;

        // Phase 1.
        float max_db = 0.0f;
        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const float* xb = blk + 32u * ib32;
            float xval[32]{};
            for (core::usize k = 0; k < 4u; ++k) {
                core::u8 s = 0u;
                int nflip = 0;
                for (core::usize j = 0; j < 8u; ++j) {
                    const float v = core::sanitiseFinite(xb[8u * k + j]);
                    if (v >= 0.0f) xval[8u * k + j] = v;
                    else { xval[8u * k + j] = -v; ++nflip; s = static_cast<core::u8>(s | (1u << j)); }
                }
                if ((nflip % 2) != 0) {
                    core::usize imin = 0; float vmin = xval[8u * k + 0];
                    for (core::usize j = 1; j < 8u; ++j)
                        if (xval[8u * k + j] < vmin) { vmin = xval[8u * k + j]; imin = j; }
                    xval[8u * k + imin] = -xval[8u * k + imin];
                    s = static_cast<core::u8>(s ^ (1u << imin));
                }
                plans[ib32].signs[k] = static_cast<core::u8>(s & 0x7Fu);
            }
            float gmax = 0.0f;
            for (core::usize j = 0; j < 32u; ++j)
                if (xval[j] > gmax) gmax = xval[j];
            // iq3xxs_grid bytes are around 14 typical; pick db so typical
            // |x| maps to ~7 (the codebook's center magnitude).
            const float db = (gmax > 1e-30f) ? gmax / 14.0f : 0.0f;
            const float id = (db > 0.0f) ? 1.0f / db : 0.0f;
            plans[ib32].db = db;
            if (db > max_db) max_db = db;

            for (core::usize g = 0; g < 8u; ++g) {
                float target[4];
                for (core::usize j = 0; j < 4u; ++j) {
                    target[j] = xval[4u * g + j] * id;
                }
                plans[ib32].grid_idx[g] = static_cast<core::u8>(
                    detail_iquant_cb::bestGridIndexU32(
                        detail_iquant_cb::iq3xxs_grid, 256u, target));
            }
        }

        if (max_db < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ3_XXS_BlockBytes);
            out += kIQ3_XXS_BlockBytes;
            continue;
        }
        // Decoder: db_decoded = d * (0.5 + s_4bit) / 2.
        // For s=15: db_decoded = 7.75 * d. So d_super = max_db / 7.75.
        const float d_super = max_db / 7.75f;

        // Phase 2.
        std::byte* qs_p  = out + 2u;
        std::byte* sas_p = out + 2u + 64u;
        std::memset(qs_p, 0, 96u);
        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            int s4 = 0;
            if (plans[ib32].db > 0.0f) {
                // s = db_real * 2 / d_super - 0.5
                const float sf = plans[ib32].db * 2.0f / d_super - 0.5f;
                s4 = detail_kquant::nearestInt(sf);
                if (s4 < 0)  s4 = 0;
                if (s4 > 15) s4 = 15;
            }
            for (core::usize g = 0; g < 8u; ++g) {
                qs_p[8u * ib32 + g] = static_cast<std::byte>(plans[ib32].grid_idx[g]);
            }
            std::uint32_t sas = 0u;
            for (core::usize l = 0; l < 4u; ++l) {
                sas |= (static_cast<std::uint32_t>(plans[ib32].signs[l]) << (7u * l));
            }
            sas |= (static_cast<std::uint32_t>(s4 & 0xFu) << 28);
            std::memcpy(sas_p + 4u * ib32, &sas, sizeof(std::uint32_t));
        }
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_super));
        out += kIQ3_XXS_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ3_XXS_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ3_XXS_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ3_XXS_BlockBytes;
    const core::usize need_bytes = n_blocks * kKBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;
        const std::byte* qs    = in + 2u;
        const std::byte* sas_p = in + 2u + 64u;
        float* y = dst_floats + b * kKBlockElements;

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            std::uint32_t sas;
            std::memcpy(&sas, sas_p + 4u * ib32, sizeof(std::uint32_t));
            const float db = d * (0.5f + static_cast<float>(sas >> 28)) * 0.5f;
            for (core::usize l = 0; l < 4u; ++l) {
                const core::u8 sign_byte =
                    detail_iquant_cb::ksigns_iq2xs[(sas >> (7u * l)) & 0x7Fu];
                const auto grid1 = reinterpret_cast<const std::uint8_t*>(
                    detail_iquant_cb::iq3xxs_grid +
                    static_cast<core::u8>(qs[8u * ib32 + 2u * l + 0u]));
                const auto grid2 = reinterpret_cast<const std::uint8_t*>(
                    detail_iquant_cb::iq3xxs_grid +
                    static_cast<core::u8>(qs[8u * ib32 + 2u * l + 1u]));
                for (core::usize j = 0; j < 4u; ++j) {
                    const float s1 = (sign_byte & detail_iquant_cb::kmask_iq2xs[j + 0])
                                   ? -1.0f : 1.0f;
                    const float s2 = (sign_byte & detail_iquant_cb::kmask_iq2xs[j + 4])
                                   ? -1.0f : 1.0f;
                    y[j + 0] = db * static_cast<float>(grid1[j]) * s1;
                    y[j + 4] = db * static_cast<float>(grid2[j]) * s2;
                }
                y += 8;
            }
        }
        in += kIQ3_XXS_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// IQ3_S — 3.4375 bpw. Block (110 B):
//   d            FP16                                          (offset   0)
//   qs[64]       uint8_t — low 8 bits of grid idx              (offset   2)
//   qh[8]        uint8_t — high bit (bit 8) of grid idx        (offset  66)
//   signs[32]    uint8_t — 8-bit per-group sign masks          (offset  74)
//   scales[4]    uint8_t — 2 × 4-bit per ib32 pair (every 2 ib32)(offset 106)
//
// db = d * (1 + 2*scale_4bit) — different formula from IQ2/IQ3_XXS!
// Grid is 512-entry uint32_t (4 magnitudes per entry).
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ3_S(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ3_S_BlockBytes;
}
static_assert(compressedSize_IQ3_S(256u) == 110u);

[[nodiscard]] inline core::usize compressFP32ToIQ3_S(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_blocks   = (src.size() / kFP32Size) / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ3_S_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;
        float amax = 0.0f;
        for (core::usize j = 0; j < kKBlockElements; ++j) {
            const float a = std::fabs(core::sanitiseFinite(blk[j]));
            if (a > amax) amax = a;
        }
        if (amax < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ3_S_BlockBytes);
            out += kIQ3_S_BlockBytes;
            continue;
        }
        // iq3s_grid magnitudes go up to ~14; db = d*(1+2*scale).
        // max db = d*(1+30) = 31d. max element = 14 * db = 434d. So d_init = amax/434.
        const float d_init = amax / 434.0f;

        std::byte* qs    = out + 2u;
        std::byte* qh_p  = out + 66u;
        std::byte* sg_p  = out + 74u;
        std::byte* sc_p  = out + 106u;
        std::memset(qs,   0, 64);
        std::memset(qh_p, 0, 8);
        std::memset(sg_p, 0, 32);
        std::memset(sc_p, 0, 4);

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const float* xb = blk + 32u * ib32;
            float xval[32]{};
            core::u8 sign_bytes[4]{};   // 8-bit (not 7-bit) sign masks
            for (core::usize k = 0; k < 4u; ++k) {
                core::u8 s = 0u;
                for (core::usize j = 0; j < 8u; ++j) {
                    const float v = core::sanitiseFinite(xb[8u * k + j]);
                    if (v >= 0.0f) xval[8u * k + j] = v;
                    else { xval[8u * k + j] = -v; s = static_cast<core::u8>(s | (1u << j)); }
                }
                sign_bytes[k] = s;
            }
            float gmax = 0.0f;
            for (core::usize j = 0; j < 32u; ++j) {
                const float v = std::fabs(xval[j]);
                if (v > gmax) gmax = v;
            }
            int scale_4bit = 0;
            if (gmax >= 1e-30f) {
                // db = d_init * (1 + 2*scale_4bit). gmax ≈ 14*db.
                // scale_4bit = (gmax/(14*d_init) - 1) / 2.
                float sf = (gmax / (14.0f * d_init) - 1.0f) * 0.5f;
                scale_4bit = detail_kquant::nearestInt(sf);
                if (scale_4bit < 0)  scale_4bit = 0;
                if (scale_4bit > 15) scale_4bit = 15;
            }
            const float db = d_init * (1.0f + 2.0f * static_cast<float>(scale_4bit));
            const float id = (db > 0.0f) ? 1.0f / db : 0.0f;
            // scales: 4-bit per ib32. scales[ib32/2]: low 4 bits = even, high = odd.
            const core::usize sc_idx = ib32 / 2u;
            if ((ib32 % 2u) == 0u) {
                sc_p[sc_idx] = static_cast<std::byte>(
                    (static_cast<core::u8>(sc_p[sc_idx]) & 0xF0u) |
                    static_cast<core::u8>(scale_4bit & 0x0F));
            } else {
                sc_p[sc_idx] = static_cast<std::byte>(
                    (static_cast<core::u8>(sc_p[sc_idx]) & 0x0Fu) |
                    static_cast<core::u8>((scale_4bit & 0x0F) << 4));
            }

            // Grid indexing: 8 groups of 4 elements per ib32. qs[8*ib32+0..7]
            // each holds low 8 bits. qh[ib32] aggregates the 9th bit of each.
            core::u8 qh_byte = 0u;
            for (core::usize g = 0; g < 8u; ++g) {
                float target[4];
                for (core::usize j = 0; j < 4u; ++j) {
                    target[j] = xval[4u * g + j] * id;
                }
                const core::usize gidx =
                    detail_iquant_cb::bestGridIndexU32(
                        detail_iquant_cb::iq3s_grid, 512u, target);
                qs[8u * ib32 + g] = static_cast<std::byte>(gidx & 0xFFu);
                if (gidx & 0x100u) qh_byte = static_cast<core::u8>(qh_byte | (1u << g));
            }
            qh_p[ib32] = static_cast<std::byte>(qh_byte);
            for (core::usize l = 0; l < 4u; ++l) {
                sg_p[4u * ib32 + l] = static_cast<std::byte>(sign_bytes[l]);
            }
        }
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_init));
        out += kIQ3_S_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ3_S_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ3_S_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ3_S_BlockBytes;
    const core::usize need_bytes = n_blocks * kKBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;
        const std::byte* qs    = in + 2u;
        const std::byte* qh_p  = in + 66u;
        const std::byte* sg_p  = in + 74u;
        const std::byte* sc_p  = in + 106u;
        float* y = dst_floats + b * kKBlockElements;

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const core::usize sc_idx = ib32 / 2u;
            const int scale_4bit = ((ib32 % 2u) == 0u)
                ? static_cast<int>(static_cast<core::u8>(sc_p[sc_idx]) & 0x0Fu)
                : static_cast<int>(static_cast<core::u8>(sc_p[sc_idx]) >> 4);
            const float db = d * (1.0f + 2.0f * static_cast<float>(scale_4bit));
            const core::u8 qh = static_cast<core::u8>(qh_p[ib32]);
            for (core::usize g = 0; g < 8u; ++g) {
                const std::uint16_t gidx = static_cast<std::uint16_t>(
                    static_cast<core::u8>(qs[8u * ib32 + g]) |
                    ((static_cast<std::uint16_t>((qh >> g) & 0x01u)) << 8));
                const auto grid = reinterpret_cast<const std::uint8_t*>(
                    detail_iquant_cb::iq3s_grid + gidx);
                // signs at index (4*ib32 + g/2). bit indices in mask
                // for the 4 elements: if g is even, bits 0..3; if odd, bits 4..7.
                const core::u8 sign_byte =
                    static_cast<core::u8>(sg_p[4u * ib32 + g / 2u]);
                const core::usize bit_off = (g % 2u) * 4u;
                for (core::usize j = 0; j < 4u; ++j) {
                    const float gv = static_cast<float>(grid[j]);
                    const float s = (sign_byte & detail_iquant_cb::kmask_iq2xs[bit_off + j])
                                  ? -1.0f : 1.0f;
                    *y++ = db * gv * s;
                }
            }
        }
        in += kIQ3_S_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// IQ1_S — 1.5625 bpw. Block (50 B):
//   d            FP16                                          (offset 0)
//   qs[QK_K/8 = 32]    uint8_t — low 8 bits of grid idx        (offset 2)
//   qh[QK_K/32 = 16]   uint16_t — packed: 4 × 3-bit hi-idx + 3-bit scale + 1-bit delta
//                                                              (offset 34)
//
// Grid is 2048-entry int8 (11-bit index, signed magnitudes).
// dl = d * (2*((qh[ib] >> 12) & 7) + 1)
// delta = (qh[ib] & 0x8000) ? -kIQ1S_Delta : kIQ1S_Delta
// gidx = qs[4*ib + l] | ((qh[ib] >> 3*l) & 7) << 8     (low 8 + 3 hi = 11 bits)
// value = dl * (grid_signed[j] + delta)
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ1_S(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ1_S_BlockBytes;
}
static_assert(compressedSize_IQ1_S(256u) == 50u);

namespace detail_iquant_cb {

/// Find best 8-element grid in iq1s_grid (signed int8 magnitudes;
/// 2048-entry). Target may be signed. dl_inv is the inverse of dl
/// (the per-ib32 scale), and `delta` is the additive ±IQ1S_Delta.
/// Returns the 11-bit grid index.
[[nodiscard]] inline core::usize bestGridIQ1S(
    const float target[8], float delta) noexcept
{
    core::usize best_idx = 0;
    float best_err = std::numeric_limits<float>::infinity();
    for (core::usize i = 0; i < 2048u; ++i) {
        const auto* grid = reinterpret_cast<const std::int8_t*>(iq1s_grid + i);
        float err = 0.0f;
        for (core::usize j = 0; j < 8u; ++j) {
            const float gv = static_cast<float>(grid[j]) + delta;
            const float d  = target[j] - gv;
            err += d * d;
        }
        if (err < best_err) {
            best_err = err;
            best_idx = i;
        }
    }
    return best_idx;
}

}  // namespace detail_iquant_cb

[[nodiscard]] inline core::usize compressFP32ToIQ1_S(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_blocks   = (src.size() / kFP32Size) / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ1_S_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;
        float amax = 0.0f;
        for (core::usize j = 0; j < kKBlockElements; ++j) {
            const float a = std::fabs(core::sanitiseFinite(blk[j]));
            if (a > amax) amax = a;
        }
        if (amax < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ1_S_BlockBytes);
            out += kIQ1_S_BlockBytes;
            continue;
        }
        // iq1s_grid magnitudes are signed int8 in [-1..1] (the grid
        // codepoints are essentially {-1, 0, +1}). dl = d*(2*sc3+1), sc3∈[0..7]
        // max dl = 15d. Max value ≈ 15d * (1 + delta) ≈ 15d * 1.125 ≈ 17d.
        // d_init = amax / 17.
        const float d_init = amax / 17.0f;

        std::byte* qs_p = out + 2u;
        std::byte* qh_p = out + 34u;
        std::memset(qs_p, 0, 32u);
        std::memset(qh_p, 0, 16u);     // 8 × u16 = 16 bytes

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const float* xb = blk + 32u * ib32;
            // Per-block delta and 3-bit scale chosen for this ib32.
            // Heuristic: pick scale based on group max, delta sign by
            // average sign.
            float gmax = 0.0f;
            float gsum = 0.0f;
            for (core::usize j = 0; j < 32u; ++j) {
                const float v = core::sanitiseFinite(xb[j]);
                gsum += v;
                const float a = std::fabs(v);
                if (a > gmax) gmax = a;
            }
            // delta sign: ggml uses negative delta when sum is negative.
            const bool delta_neg = (gsum < 0.0f);
            const float delta = delta_neg
                ? -detail_iquant_cb::kIQ1S_Delta
                :  detail_iquant_cb::kIQ1S_Delta;

            int sc3 = 0;
            if (gmax >= 1e-30f) {
                // dl = d_init * (2*sc3 + 1). gmax ≈ dl * 1.125 (worst case).
                float sf = (gmax / (1.125f * d_init) - 1.0f) * 0.5f;
                sc3 = detail_kquant::nearestInt(sf);
                if (sc3 < 0)  sc3 = 0;
                if (sc3 > 7)  sc3 = 7;
            }
            const float dl = d_init * (2.0f * static_cast<float>(sc3) + 1.0f);
            const float id = (dl > 0.0f) ? 1.0f / dl : 0.0f;

            // 4 grid lookups, each 8 elements.
            std::uint16_t qh_val = 0u;
            for (core::usize l = 0; l < 4u; ++l) {
                float target[8];
                for (core::usize j = 0; j < 8u; ++j) {
                    target[j] = core::sanitiseFinite(xb[8u * l + j]) * id;
                }
                const core::usize gidx =
                    detail_iquant_cb::bestGridIQ1S(target, delta);
                qs_p[4u * ib32 + l] = static_cast<std::byte>(gidx & 0xFFu);
                qh_val = static_cast<std::uint16_t>(
                    qh_val | ((static_cast<std::uint16_t>(gidx >> 8) & 0x07u) << (3u * l)));
            }
            qh_val = static_cast<std::uint16_t>(
                qh_val | ((static_cast<std::uint16_t>(sc3) & 0x07u) << 12));
            if (delta_neg) qh_val = static_cast<std::uint16_t>(qh_val | 0x8000u);
            detail::storeU16(qh_p + 2u * ib32, qh_val);
        }
        detail::storeU16(out + 0u, core::fp32ToFP16Safe(d_init));
        out += kIQ1_S_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ1_S_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ1_S_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ1_S_BlockBytes;
    const core::usize need_bytes = n_blocks * kKBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float d = core::fp16ToFP32(detail::loadU16(in));
        if (!core::isFiniteStrict(d)) return 0u;
        const std::byte* qs_p = in + 2u;
        const std::byte* qh_p = in + 34u;
        float* y = dst_floats + b * kKBlockElements;

        for (core::usize ib32 = 0; ib32 < 8u; ++ib32) {
            const std::uint16_t qh = detail::loadU16(qh_p + 2u * ib32);
            const float dl = d * (2.0f * static_cast<float>((qh >> 12) & 0x07u) + 1.0f);
            const float delta = (qh & 0x8000u)
                ? -detail_iquant_cb::kIQ1S_Delta
                :  detail_iquant_cb::kIQ1S_Delta;
            for (core::usize l = 0; l < 4u; ++l) {
                const std::uint16_t gidx = static_cast<std::uint16_t>(
                    static_cast<core::u8>(qs_p[4u * ib32 + l]) |
                    ((static_cast<std::uint16_t>((qh >> (3u * l)) & 0x07u)) << 8));
                const auto grid = reinterpret_cast<const std::int8_t*>(
                    detail_iquant_cb::iq1s_grid + gidx);
                for (core::usize j = 0; j < 8u; ++j) {
                    *y++ = dl * (static_cast<float>(grid[j]) + delta);
                }
            }
        }
        in += kIQ1_S_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// IQ1_M — 1.75 bpw. Block (56 B):
//   qs[QK_K/8 = 32]    uint8_t — low 8 bits of grid idx        (offset 0)
//   qh[QK_K/16 = 16]   uint8_t — 4 hi-bits + delta-bit per 16 elements
//                                                              (offset 32)
//   scales[QK_K/32 = 8] uint8_t — 4×3-bit scales + super-scale (offset 48)
//
// Super-block scale d is reconstructed from top nibbles of 4 scale-bytes
// (see decoder line 2611): scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0x00f0)
//                                    | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)
// then d = FP16(scale.u16).
//
// Per ib (32 elements = 4 groups of 8), 2 sub-blocks of 16 elements:
//   dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1)
//   dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1)
//   idx[k] = qs[k] | ((qh[group] << shift) & 0x700)
//   delta[k] = (qh & mask) ? ±IQ1S_Delta
// ============================================================================

[[nodiscard]] constexpr core::usize compressedSize_IQ1_M(
    core::usize fp32_element_count) noexcept
{
    if ((fp32_element_count % kKBlockElements) != 0u) return 0u;
    return (fp32_element_count / kKBlockElements) * kIQ1_M_BlockBytes;
}
static_assert(compressedSize_IQ1_M(256u) == 56u);

[[nodiscard]] inline core::usize compressFP32ToIQ1_M(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % (kKBlockElements * kFP32Size) != 0u) return 0u;
    const core::usize n_blocks   = (src.size() / kFP32Size) / kKBlockElements;
    const core::usize need_bytes = n_blocks * kIQ1_M_BlockBytes;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    core::FtzDazGuard fpe_guard;
    const float* src_floats = reinterpret_cast<const float*>(src.data());
    std::byte* out = dst.data();

    for (core::usize b = 0; b < n_blocks; ++b) {
        const float* blk = src_floats + b * kKBlockElements;
        float amax = 0.0f;
        for (core::usize j = 0; j < kKBlockElements; ++j) {
            const float a = std::fabs(core::sanitiseFinite(blk[j]));
            if (a > amax) amax = a;
        }
        if (amax < 1e-30f) [[unlikely]] {
            std::memset(out, 0, kIQ1_M_BlockBytes);
            out += kIQ1_M_BlockBytes;
            continue;
        }
        const float d_init = amax / 17.0f;

        std::byte* qs_p = out + 0u;
        std::byte* qh_p = out + 32u;
        std::byte* sc_p = out + 48u;
        std::memset(qs_p, 0, 32u);
        std::memset(qh_p, 0, 16u);
        std::memset(sc_p, 0, 8u);

        // Determine the 3-bit per-16-elem scales. 8 ib32 sub-blocks
        // each contain 2 16-element groups, so 16 scales total — but
        // they're packed 2 per byte (3 bits each, 4 sub-blocks of 16
        // span 8 sub-blocks of ib32... wait, decoder structure: 8 ib
        // × 2 sub-blocks per ib, each sub-block has its own dl1/dl2).
        // Re-read: for (ib = 0..QK_K/32-1=7) the code reads sc[ib/2]
        // and extracts ((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) for dl1 and
        // (... + 3) & 0x7 for dl2. So sc[0..3] is 4 bytes, each holds
        // 2 ib32 worth of 2×3-bit scales = 12 bits = wait that overflows
        // 8 bits per byte. Re-check: 6*(ib%2)+0 ∈ {0, 6}; 6*(ib%2)+3 ∈ {3, 9}.
        // Bit 9 is out of an 8-bit byte! So `sc` must be uint16_t-aligned.
        // Looking at ggml struct: scales is uint8_t[QK_K/32 = 8] and decoder
        // casts `const uint16_t * sc = (const uint16_t *)x[i].scales` — so
        // it's read as 4 × u16. That's the 8 bytes.
        // Per u16 (2 ib32 sub-blocks), 12 bits used for 4 × 3-bit scales,
        // and the top 4 bits are the super-scale nibble for d reconstruction.
        // OK, IQ1_M's super-scale spans 4 nibbles across 4 uint16_t scale words.

        // Find scales per 16-element sub-block.
        std::array<int, 16> sub_scales{};   // 16 × 3-bit (one per 16-elem group)
        std::array<core::u8, 16> sub_delta_neg{};
        for (core::usize sb = 0; sb < 16u; ++sb) {
            // sb-th 16-element subgroup: ib32 = sb / 2, inner half = sb%2.
            const float* xb = blk + 16u * sb;
            float lmax = 0.0f, lsum = 0.0f;
            for (core::usize j = 0; j < 16u; ++j) {
                const float v = core::sanitiseFinite(xb[j]);
                lsum += v;
                const float a = std::fabs(v);
                if (a > lmax) lmax = a;
            }
            sub_delta_neg[sb] = (lsum < 0.0f) ? 1u : 0u;
            int sc3 = 0;
            if (lmax >= 1e-30f) {
                float sf = (lmax / (1.125f * d_init) - 1.0f) * 0.5f;
                sc3 = detail_kquant::nearestInt(sf);
                if (sc3 < 0) sc3 = 0;
                if (sc3 > 7) sc3 = 7;
            }
            sub_scales[sb] = sc3;
        }

        // Pack super-scale d_init into the 4-nibble layout (top of each
        // u16 word). We need 4 nibbles to encode a u16 FP16 value.
        const core::u16 d_h = core::fp32ToFP16Safe(d_init);
        // Decoder reconstruction: scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0x00f0)
        //                                  | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000);
        // So sc[k] holds nibble k of d_h at the top 4 bits (bits 12..15).
        // sc[0] >> 12 = bits 12..15 of sc[0] -> nibble 0 of d_h (bits 0..3)
        // (sc[1]>>8)&0x00f0 = bits 8..11 of sc[1] -> ... wait that's bits 12..15 of sc[1]
        //   actually (sc[1] >> 8) & 0x00f0 means bits 12..15 of sc[1] shifted to bits 4..7
        // So nibble 1 of d_h (bits 4..7) lives in bits 12..15 of sc[1].
        // (sc[2]>>4)&0x0f00 = bits 12..15 of sc[2] shifted to bits 8..11.
        // sc[3]&0xf000 = bits 12..15 of sc[3] in bits 12..15.
        // So each sc[k] u16's top 4 bits carry one nibble of d_h.

        // Encode the four scale u16 words.
        for (core::usize k = 0; k < 4u; ++k) {
            // Each u16 carries 4 × 3-bit scales for ib32 = 2k and 2k+1.
            // Layout: bits 0..2 = ib32=2k group 0 (sb=4k+0)
            //         bits 3..5 = ib32=2k group 1 (sb=4k+1)
            //         bits 6..8 = ib32=2k+1 group 0 (sb=4k+2)
            //         bits 9..11 = ib32=2k+1 group 1 (sb=4k+3)
            //         bits 12..15 = super-scale nibble k
            const std::uint16_t v = static_cast<std::uint16_t>(
                (static_cast<std::uint16_t>(sub_scales[4u * k + 0]) & 0x07u) |
                ((static_cast<std::uint16_t>(sub_scales[4u * k + 1]) & 0x07u) << 3) |
                ((static_cast<std::uint16_t>(sub_scales[4u * k + 2]) & 0x07u) << 6) |
                ((static_cast<std::uint16_t>(sub_scales[4u * k + 3]) & 0x07u) << 9) |
                ((static_cast<std::uint16_t>((d_h >> (4u * k)) & 0x0Fu)) << 12));
            std::memcpy(sc_p + 2u * k, &v, sizeof(std::uint16_t));
        }

        // Per ib (8 ib32 sub-blocks), encode grid indices + qh.
        for (core::usize ib = 0; ib < 8u; ++ib) {
            const float* xb = blk + 32u * ib;
            const int sc1 = sub_scales[2u * ib + 0];
            const int sc2 = sub_scales[2u * ib + 1];
            const float dl1 = d_init * (2.0f * static_cast<float>(sc1) + 1.0f);
            const float dl2 = d_init * (2.0f * static_cast<float>(sc2) + 1.0f);
            const float id1 = (dl1 > 0.0f) ? 1.0f / dl1 : 0.0f;
            const float id2 = (dl2 > 0.0f) ? 1.0f / dl2 : 0.0f;
            const float delta1 = sub_delta_neg[2u * ib + 0]
                ? -detail_iquant_cb::kIQ1S_Delta : detail_iquant_cb::kIQ1S_Delta;
            const float delta2 = sub_delta_neg[2u * ib + 1]
                ? -detail_iquant_cb::kIQ1S_Delta : detail_iquant_cb::kIQ1S_Delta;

            // Layout per decoder: 4 groups of 8 per ib, all within 32 elems.
            // idx[0] = qs[4*ib+0] | ((qh[0] << 8) & 0x700)   group 0 (sb=2*ib+0, half 0)
            // idx[1] = qs[4*ib+1] | ((qh[0] << 4) & 0x700)   group 1 (sb=2*ib+0, half 1)
            // idx[2] = qs[4*ib+2] | ((qh[1] << 8) & 0x700)   group 2 (sb=2*ib+1, half 0)
            // idx[3] = qs[4*ib+3] | ((qh[1] << 4) & 0x700)   group 3 (sb=2*ib+1, half 1)
            // delta[0..1] from qh[0]&0x08 / qh[0]&0x80, [2..3] from qh[1]
            // qh[0] is qh_p[2*ib], qh[1] is qh_p[2*ib+1].
            for (core::usize l = 0; l < 4u; ++l) {
                const float dl_sel  = (l < 2u) ? dl1 : dl2;
                const float id_sel  = (l < 2u) ? id1 : id2;
                const float delta_sel = (l < 2u) ? delta1 : delta2;
                (void)dl_sel;
                float target[8];
                for (core::usize j = 0; j < 8u; ++j) {
                    target[j] = core::sanitiseFinite(xb[8u * l + j]) * id_sel;
                }
                const core::usize gidx =
                    detail_iquant_cb::bestGridIQ1S(target, delta_sel);
                qs_p[4u * ib + l] = static_cast<std::byte>(gidx & 0xFFu);
                // Pack high 3 bits + delta-sign bit into qh.
                const core::u8 qh_idx = static_cast<core::u8>(2u * ib + l / 2u);
                const int shift = (l % 2u == 0u) ? 0 : 4;
                core::u8 cur = static_cast<core::u8>(qh_p[qh_idx]);
                cur = static_cast<core::u8>(
                    cur | ((static_cast<core::u8>((gidx >> 8) & 0x07u)) << shift));
                // delta bit: bit 3 (for shift=0) or bit 7 (for shift=4)
                const bool delta_neg = (l < 2u)
                    ? sub_delta_neg[2u * ib + 0] : sub_delta_neg[2u * ib + 1];
                if (delta_neg) cur = static_cast<core::u8>(cur | (0x08u << shift));
                qh_p[qh_idx] = static_cast<std::byte>(cur);
            }
        }
        out += kIQ1_M_BlockBytes;
    }
    return need_bytes;
}

[[nodiscard]] inline core::usize decompressIQ1_M_ToFP32(
    std::span<const std::byte> src,
    std::span<std::byte>       dst) noexcept
{
    if (src.size() % kIQ1_M_BlockBytes != 0u) return 0u;
    const core::usize n_blocks   = src.size() / kIQ1_M_BlockBytes;
    const core::usize need_bytes = n_blocks * kKBlockElements * kFP32Size;
    if (dst.size() < need_bytes) return 0u;
    if (detail::spansOverlap(src.data(), src.size(),
                             dst.data(), need_bytes)) return 0u;

    const std::byte* in = src.data();
    float* dst_floats   = reinterpret_cast<float*>(dst.data());

    for (core::usize b = 0; b < n_blocks; ++b) {
        const std::byte* qs_p = in + 0u;
        const std::byte* qh_p = in + 32u;
        const std::byte* sc_p = in + 48u;

        // Reconstruct d from 4 u16 scale-word top nibbles.
        std::uint16_t sc[4];
        for (core::usize k = 0; k < 4u; ++k) {
            std::memcpy(&sc[k], sc_p + 2u * k, sizeof(std::uint16_t));
        }
        const std::uint16_t scale_u16 = static_cast<std::uint16_t>(
            (sc[0] >> 12) |
            ((sc[1] >> 8) & 0x00F0u) |
            ((sc[2] >> 4) & 0x0F00u) |
            (sc[3] & 0xF000u));
        const float d = core::fp16ToFP32(scale_u16);
        if (!core::isFiniteStrict(d)) return 0u;

        float* y = dst_floats + b * kKBlockElements;
        for (core::usize ib = 0; ib < 8u; ++ib) {
            const std::uint16_t sc_word = sc[ib / 2u];
            const int sc1 = static_cast<int>((sc_word >> (6u * (ib % 2u) + 0u)) & 0x07u);
            const int sc2 = static_cast<int>((sc_word >> (6u * (ib % 2u) + 3u)) & 0x07u);
            const float dl1 = d * (2.0f * static_cast<float>(sc1) + 1.0f);
            const float dl2 = d * (2.0f * static_cast<float>(sc2) + 1.0f);

            const core::u8 qh0 = static_cast<core::u8>(qh_p[2u * ib + 0u]);
            const core::u8 qh1 = static_cast<core::u8>(qh_p[2u * ib + 1u]);
            std::uint16_t idx[4];
            idx[0] = static_cast<std::uint16_t>(
                static_cast<core::u8>(qs_p[4u * ib + 0u]) |
                ((static_cast<std::uint16_t>(qh0) << 8) & 0x700u));
            idx[1] = static_cast<std::uint16_t>(
                static_cast<core::u8>(qs_p[4u * ib + 1u]) |
                ((static_cast<std::uint16_t>(qh0) << 4) & 0x700u));
            idx[2] = static_cast<std::uint16_t>(
                static_cast<core::u8>(qs_p[4u * ib + 2u]) |
                ((static_cast<std::uint16_t>(qh1) << 8) & 0x700u));
            idx[3] = static_cast<std::uint16_t>(
                static_cast<core::u8>(qs_p[4u * ib + 3u]) |
                ((static_cast<std::uint16_t>(qh1) << 4) & 0x700u));
            float delta[4];
            delta[0] = (qh0 & 0x08u) ? -detail_iquant_cb::kIQ1S_Delta
                                     :  detail_iquant_cb::kIQ1S_Delta;
            delta[1] = (qh0 & 0x80u) ? -detail_iquant_cb::kIQ1S_Delta
                                     :  detail_iquant_cb::kIQ1S_Delta;
            delta[2] = (qh1 & 0x08u) ? -detail_iquant_cb::kIQ1S_Delta
                                     :  detail_iquant_cb::kIQ1S_Delta;
            delta[3] = (qh1 & 0x80u) ? -detail_iquant_cb::kIQ1S_Delta
                                     :  detail_iquant_cb::kIQ1S_Delta;
            for (core::usize l = 0; l < 2u; ++l) {
                const auto grid = reinterpret_cast<const std::int8_t*>(
                    detail_iquant_cb::iq1s_grid + idx[l]);
                for (core::usize j = 0; j < 8u; ++j) {
                    *y++ = dl1 * (static_cast<float>(grid[j]) + delta[l]);
                }
            }
            for (core::usize l = 2; l < 4u; ++l) {
                const auto grid = reinterpret_cast<const std::int8_t*>(
                    detail_iquant_cb::iq1s_grid + idx[l]);
                for (core::usize j = 0; j < 8u; ++j) {
                    *y++ = dl2 * (static_cast<float>(grid[j]) + delta[l]);
                }
            }
        }
        in += kIQ1_M_BlockBytes;
    }
    return need_bytes;
}

// ============================================================================
// Universal dispatcher
// ============================================================================
[[nodiscard]] inline core::usize compress(
    arena::QuantFormat target_fmt,
    std::span<const std::byte> src_fp32,
    std::span<std::byte>       dst) noexcept
{
    using F = arena::QuantFormat;
    switch (target_fmt) {
        case F::kFP32: {
            if (dst.size() < src_fp32.size()) return 0u;
            if (detail::spansOverlap(src_fp32.data(), src_fp32.size(),
                                     dst.data(), src_fp32.size())) return 0u;
            std::memcpy(dst.data(), src_fp32.data(), src_fp32.size());
            return src_fp32.size();
        }
        case F::kBF16:
            return compressFP32ToBF16(src_fp32, dst);
        case F::kFP16:
            return compressFP32ToFP16(src_fp32, dst);
        case F::kFP8_E4M3:
            return compressFP32ToFP8_E4M3(src_fp32, dst);
        case F::kFP8_E5M2:
            return compressFP32ToFP8_E5M2(src_fp32, dst);
        case F::kINT8:
            return compressFP32ToINT8(src_fp32, dst);
        case F::kINT4:
            return compressFP32ToINT4(src_fp32, dst);
        default:
            return 0u;  // EC51: kReserved / corrupted format
    }
}

[[nodiscard]] inline core::usize decompress(
    arena::QuantFormat src_fmt,
    std::span<const std::byte> src,
    std::span<std::byte>       dst_fp32) noexcept
{
    using F = arena::QuantFormat;
    switch (src_fmt) {
        case F::kFP32: {
            if (dst_fp32.size() < src.size()) return 0u;
            if (detail::spansOverlap(src.data(), src.size(),
                                     dst_fp32.data(), src.size())) return 0u;
            std::memcpy(dst_fp32.data(), src.data(), src.size());
            return src.size();
        }
        case F::kBF16:
            return decompressBF16ToFP32(src, dst_fp32);
        case F::kFP16:
            return decompressFP16ToFP32(src, dst_fp32);
        case F::kFP8_E4M3:
            return decompressFP8_E4M3_ToFP32(src, dst_fp32);
        case F::kFP8_E5M2:
            return decompressFP8_E5M2_ToFP32(src, dst_fp32);
        case F::kINT8:
            return decompressINT8ToFP32(src, dst_fp32);
        case F::kINT4:
            return decompressINT4ToFP32(src, dst_fp32);
        default:
            return 0u;
    }
}

// ============================================================================
// HA: compile-time checks for constants
// ============================================================================
static_assert(compressedSize(arena::QuantFormat::kFP32, 32u) == 128u);
static_assert(compressedSize(arena::QuantFormat::kBF16, 32u) == 64u);
static_assert(compressedSize(arena::QuantFormat::kFP16, 32u) == 64u);
static_assert(compressedSize(arena::QuantFormat::kFP8_E4M3, 32u) == 32u);
static_assert(compressedSize(arena::QuantFormat::kFP8_E5M2, 32u) == 32u);
static_assert(compressedSize(arena::QuantFormat::kINT8, 32u) == 36u);
static_assert(compressedSize(arena::QuantFormat::kINT4, 32u) == 20u);
static_assert(compressedSize(arena::QuantFormat::kINT8, 64u) == 72u);
static_assert(compressionRatio(arena::QuantFormat::kFP32) == 1.0f);
static_assert(compressionRatio(arena::QuantFormat::kBF16) == 2.0f);
static_assert(compressionRatio(arena::QuantFormat::kFP16) == 2.0f);
static_assert(compressionRatio(arena::QuantFormat::kFP8_E4M3) == 4.0f);
static_assert(compressionRatio(arena::QuantFormat::kFP8_E5M2) == 4.0f);

}  // namespace qtx::quantize
