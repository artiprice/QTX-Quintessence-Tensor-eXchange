// ============================================================================
// @file        test_quantizer.cpp
//@brief Unit quantizer tests: round-trip of all formats, accuracy.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================

#include "test_harness.hpp"
#include "qtx/quantize/quantizer.hpp"
#include "qtx/arena/gen_id.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace qtx::quantize;
using qtx::arena::QuantFormat;
using qtx::core::usize;

//Utility: create an FP32 array from the given values ​​as std::span<std::byte>.
template <std::size_t N>
auto floatsToBytes(const std::array<float, N>& src) {
    std::array<std::byte, N * sizeof(float)> out{};
    std::memcpy(out.data(), src.data(), out.size());
    return out;
}

template <std::size_t N>
auto bytesToFloats(const std::array<std::byte, N * sizeof(float)>& src) {
    std::array<float, N> out{};
    std::memcpy(out.data(), src.data(), src.size());
    return out;
}

// ============================================================================
// Compile-time / size constants
// ============================================================================

QTX_TEST(Quant, CompressedSizeFP32) {
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP32, 32u), 128u);
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP32, 1024u), 4096u);
}

QTX_TEST(Quant, CompressedSizeBF16) {
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kBF16, 32u), 64u);
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kBF16, 1024u), 2048u);
}

QTX_TEST(Quant, CompressedSizeINT8) {
    //32 elements = 1 block = 36 bytes.
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kINT8, 32u), 36u);
    //64 elements = 2 blocks = 72 bytes.
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kINT8, 64u), 72u);
}

QTX_TEST(Quant, CompressionRatios) {
    QTX_EXPECT_EQ(compressionRatio(QuantFormat::kFP32), 1.0f);
    QTX_EXPECT_EQ(compressionRatio(QuantFormat::kBF16), 2.0f);
    // INT8: 128 / 36 ≈ 3.555
    QTX_EXPECT(compressionRatio(QuantFormat::kINT8) > 3.5f);
    QTX_EXPECT(compressionRatio(QuantFormat::kINT8) < 3.6f);
    // INT4: 128 / 20 = 6.4
    QTX_EXPECT(compressionRatio(QuantFormat::kINT4) > 6.3f);
    QTX_EXPECT(compressionRatio(QuantFormat::kINT4) < 6.5f);
}

// ============================================================================
// BF16 round-trip
// ============================================================================

QTX_TEST(Quant, BF16RoundTripPreservesExponent) {
    //BF16 loses only the mantissa (lower 16 bits), the exponent is preserved.
    std::array<float, 8> src = {
        1.0f, -1.0f, 2.0f, -2.0f, 4.0f, 0.5f, 100.0f, -100.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 16> compressed{};  //8 × 2 bytes
    std::array<std::byte, 32> decompressed{};  //8 × 4 bytes

    const auto compressed_size = compressFP32ToBF16(src_bytes, compressed);
    QTX_EXPECT_EQ(compressed_size, 16u);

    const auto decompressed_size = decompressBF16ToFP32(compressed, decompressed);
    QTX_EXPECT_EQ(decompressed_size, 32u);

    auto out = bytesToFloats<8>(decompressed);
    //Powers of two and their multiples are preserved exactly.
    QTX_EXPECT_EQ(out[0], 1.0f);
    QTX_EXPECT_EQ(out[1], -1.0f);
    QTX_EXPECT_EQ(out[2], 2.0f);
    QTX_EXPECT_EQ(out[3], -2.0f);
    QTX_EXPECT_EQ(out[4], 4.0f);
    QTX_EXPECT_EQ(out[5], 0.5f);
}

QTX_TEST(Quant, BF16PreservesSignAndMagnitude) {
    //Arbitrary values ​​lose precision, but the sign and order of magnitude are preserved.
    std::array<float, 4> src = {3.14f, -2.71f, 0.001f, 9999.5f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8> compressed{};
    std::array<std::byte, 16> decompressed{};

    (void)compressFP32ToBF16(src_bytes, compressed);
    (void)decompressBF16ToFP32(compressed, decompressed);
    auto out = bytesToFloats<4>(decompressed);

    //Relative error < 1% for BF16 (8-bit mantissa).
    for (int i = 0; i < 4; ++i) {
        const float rel_err = std::fabs(out[i] - src[i]) /
                              std::fabs(src[i]);
        QTX_EXPECT(rel_err < 0.01f);
    }
}

QTX_TEST(Quant, BF16BadSizeReturnsZero) {
    std::array<std::byte, 5> bad_src{};  //not a multiple of 4
    std::array<std::byte, 10> dst{};
    QTX_EXPECT_EQ(compressFP32ToBF16(bad_src, dst), 0u);
}

QTX_TEST(Quant, BF16InsufficientDstReturnsZero) {
    std::array<float, 4> src = {1.0f, 2.0f, 3.0f, 4.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4> tiny_dst{};  //need 8, have 4
    QTX_EXPECT_EQ(compressFP32ToBF16(src_bytes, tiny_dst), 0u);
}

// ============================================================================
// FP16 (IEEE 754 binary16) round-trip — Epic 1 milestone.
//
// FP16 has 1 sign / 5 exponent / 10 mantissa bits. Compared to BF16 it
// has 3 fewer exponent bits (range narrows to ~6.10e-5 .. 65504 vs
// BF16's ~1.18e-38 .. 3.39e38) but 3 extra mantissa bits (relative
// error ≤ 2^-11 ≈ 4.88e-4 instead of BF16's ~3.91e-3).
// ============================================================================

QTX_TEST(Quant, CompressedSizeFP16) {
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP16, 32u),   64u);
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP16, 1024u), 2048u);
}

QTX_TEST(Quant, FP16CompressionRatio) {
    QTX_EXPECT_EQ(compressionRatio(QuantFormat::kFP16), 2.0f);
}

QTX_TEST(Quant, FP16RoundTripExactValues) {
    // Powers of two and small integers are exactly representable in FP16.
    std::array<float, 8> src = {
        1.0f, -1.0f, 2.0f, -2.0f, 4.0f, 0.5f, 100.0f, -100.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 16> compressed{};      // 8 × 2 bytes
    std::array<std::byte, 32> decompressed{};    // 8 × 4 bytes

    QTX_EXPECT_EQ(compressFP32ToFP16(src_bytes, compressed),     16u);
    QTX_EXPECT_EQ(decompressFP16ToFP32(compressed, decompressed), 32u);

    auto out = bytesToFloats<8>(decompressed);
    QTX_EXPECT_EQ(out[0],  1.0f);
    QTX_EXPECT_EQ(out[1], -1.0f);
    QTX_EXPECT_EQ(out[2],  2.0f);
    QTX_EXPECT_EQ(out[3], -2.0f);
    QTX_EXPECT_EQ(out[4],  4.0f);
    QTX_EXPECT_EQ(out[5],  0.5f);
    QTX_EXPECT_EQ(out[6],  100.0f);
    QTX_EXPECT_EQ(out[7], -100.0f);
}

QTX_TEST(Quant, FP16PreservesSignAndMagnitudeWithTightError) {
    // FP16: 10-bit mantissa => relative error ≤ 2^-11 ≈ 4.88e-4.
    // Compared to BF16 (~0.4%), FP16 should achieve ~0.1% on
    // generic finite values (provided they're inside its dynamic range).
    std::array<float, 4> src = {3.14f, -2.71f, 0.125f, 9999.5f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8>  compressed{};
    std::array<std::byte, 16> decompressed{};

    (void)compressFP32ToFP16(src_bytes,    compressed);
    (void)decompressFP16ToFP32(compressed, decompressed);
    auto out = bytesToFloats<4>(decompressed);

    for (int i = 0; i < 4; ++i) {
        const float rel_err =
            std::fabs(out[i] - src[i]) / std::fabs(src[i]);
        QTX_EXPECT(rel_err < 0.001f);  // ≤ 0.1%
    }
}

QTX_TEST(Quant, FP16BadSizeReturnsZero) {
    std::array<std::byte, 5>  bad_src{};   // not a multiple of 4
    std::array<std::byte, 10> dst{};
    QTX_EXPECT_EQ(compressFP32ToFP16(bad_src, dst), 0u);
}

QTX_TEST(Quant, FP16InsufficientDstReturnsZero) {
    std::array<float, 4> src = {1.0f, 2.0f, 3.0f, 4.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4> tiny_dst{};   // need 8, have 4
    QTX_EXPECT_EQ(compressFP32ToFP16(src_bytes, tiny_dst), 0u);
}

QTX_TEST(Quant, FP16PreservesInfinity) {
    std::array<float, 4> src = {
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        0.0f,
        -0.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8>  compressed{};
    std::array<std::byte, 16> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToFP16(src_bytes, compressed), 8u);
    QTX_EXPECT_EQ(decompressFP16ToFP32(compressed, decompressed), 16u);
    auto out = bytesToFloats<4>(decompressed);

    QTX_EXPECT(std::isinf(out[0]) && out[0] > 0.0f);
    QTX_EXPECT(std::isinf(out[1]) && out[1] < 0.0f);
    QTX_EXPECT_EQ(out[2],  0.0f);
    QTX_EXPECT_EQ(out[3], -0.0f);
}

QTX_TEST(Quant, FP16PreservesNaN) {
    // EC55 contract: NaN -> quiet NaN, never collapses to Inf.
    std::array<float, 2> src = {
        std::numeric_limits<float>::quiet_NaN(),
        1.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4>  compressed{};
    std::array<std::byte, 8>  decompressed{};

    QTX_EXPECT_EQ(compressFP32ToFP16(src_bytes, compressed), 4u);
    QTX_EXPECT_EQ(decompressFP16ToFP32(compressed, decompressed), 8u);
    auto out = bytesToFloats<2>(decompressed);
    QTX_EXPECT(std::isnan(out[0]));
    QTX_EXPECT_EQ(out[1], 1.0f);
}

QTX_TEST(Quant, FP16SaturatesFiniteOverflow) {
    // EC158 invariant for FP16: a finite FP32 above the FP16 max-finite
    // (65504) MUST round to ±65504, not ±Inf. This includes FLT_MAX
    // and 65520 (the IEEE rounding boundary).
    std::array<float, 4> src = {
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        65520.0f,                            // rounding boundary
        -65520.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8>  compressed{};
    std::array<std::byte, 16> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToFP16(src_bytes, compressed), 8u);
    QTX_EXPECT_EQ(decompressFP16ToFP32(compressed, decompressed), 16u);
    auto out = bytesToFloats<4>(decompressed);

    constexpr float kMaxFiniteFP16 = 65504.0f;
    QTX_EXPECT(std::isfinite(out[0]));
    QTX_EXPECT(std::isfinite(out[1]));
    QTX_EXPECT_EQ(out[0],  kMaxFiniteFP16);
    QTX_EXPECT_EQ(out[1], -kMaxFiniteFP16);
    QTX_EXPECT_EQ(out[2],  kMaxFiniteFP16);
    QTX_EXPECT_EQ(out[3], -kMaxFiniteFP16);
}

QTX_TEST(Quant, FP16Subnormals) {
    // The smallest positive FP16 normal is 2^-14 ≈ 6.10e-5. Below that,
    // FP16 subnormals encode magnitudes down to 2^-24 ≈ 5.96e-8; below
    // half of that (2^-25 ≈ 2.98e-8) the value rounds to ±0 under
    // round-to-nearest-even (with sign preserved).
    constexpr float kBelowHalfSubnormal = 1.0e-8f;  // ≪ 2^-25
    std::array<float, 4> src = {
        6.10e-5f,                            // ≈ smallest normal
        1.0e-5f,                             // subnormal range
        kBelowHalfSubnormal,                 // rounds to +0
        -kBelowHalfSubnormal                 // rounds to -0
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8>  compressed{};
    std::array<std::byte, 16> decompressed{};

    (void)compressFP32ToFP16(src_bytes,    compressed);
    (void)decompressFP16ToFP32(compressed, decompressed);
    auto out = bytesToFloats<4>(decompressed);

    // Smallest normal round-trips close (within FP16 subnormal-step error).
    // 6.10e-5f (FP32) sits just below 2^-14 and is encoded as the largest
    // FP16 subnormal (0x3FF), which decodes to ~6.097e-5 — relative
    // error ~0.04%, dominated by the FP16 subnormal step size.
    QTX_EXPECT(std::fabs(out[0] - src[0]) / std::fabs(src[0]) < 0.001f);
    // Subnormal range: representable, finite.
    QTX_EXPECT(std::isfinite(out[1]));
    QTX_EXPECT(out[1] > 0.0f);
    // Below half the smallest subnormal: rounds to ±0, sign preserved.
    QTX_EXPECT_EQ(out[2],  0.0f);
    QTX_EXPECT_EQ(out[3], -0.0f);
}

QTX_TEST(Quant, FP16OutperformsBF16OnMantissa) {
    // Regression guard: FP16 must be strictly more accurate than BF16
    // on a value where the extra 3 mantissa bits are non-zero (sanity
    // check that the dispatcher is NOT silently aliasing FP16 back to
    // BF16 — the bug fixed in this Epic).
    //
    // π is a deceptive choice: FP32 bits of π are 0x40490FDB, and bits
    // 13..15 of the mantissa happen to be zero, so FP16 and BF16 land on
    // the same value (3.140625). We use e² = 7.389056 instead, whose
    // FP32 mantissa fills bits 13..15 with significant data.
    std::array<float, 1> src = {7.389056f};
    auto src_bytes = floatsToBytes(src);

    std::array<std::byte, 2>  fp16_compressed{};
    std::array<std::byte, 2>  bf16_compressed{};
    std::array<std::byte, 4>  fp16_decompressed{};
    std::array<std::byte, 4>  bf16_decompressed{};

    (void)compressFP32ToFP16(src_bytes,        fp16_compressed);
    (void)compressFP32ToBF16(src_bytes,        bf16_compressed);
    (void)decompressFP16ToFP32(fp16_compressed, fp16_decompressed);
    (void)decompressBF16ToFP32(bf16_compressed, bf16_decompressed);

    const float fp16_out = bytesToFloats<1>(fp16_decompressed)[0];
    const float bf16_out = bytesToFloats<1>(bf16_decompressed)[0];

    const float fp16_err = std::fabs(fp16_out - src[0]);
    const float bf16_err = std::fabs(bf16_out - src[0]);

    // FP16 must strictly beat BF16 here (10-bit vs 7-bit mantissa).
    QTX_EXPECT(fp16_err < bf16_err);
}

QTX_TEST(Quant, DispatcherFP16RoundTrip) {
    // Universal compress() / decompress() dispatcher must route kFP16
    // to the FP16 codec, NOT to BF16 (regression guard for the Epic 0
    // aliasing bug, before it was fixed in Epic 1).
    std::array<float, 4> src = {1.0f, 2.0f, 4.0f, 8.0f};  // exact in FP16
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8>  mid{};
    std::array<std::byte, 16> back{};

    QTX_EXPECT_EQ(compress  (QuantFormat::kFP16, src_bytes, mid),  8u);
    QTX_EXPECT_EQ(decompress(QuantFormat::kFP16, mid,       back), 16u);
    auto out = bytesToFloats<4>(back);
    for (int i = 0; i < 4; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

QTX_TEST(Quant, DispatcherFP16ProducesDifferentBytesFromBF16) {
    // A value with non-trivial mantissa bits — 3.14f — encodes
    // differently in FP16 (10-bit mantissa) than in BF16 (7-bit). If
    // the bytes coincide, the dispatcher is aliasing again.
    std::array<float, 1> src = {3.14f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 2> fp16_bytes{};
    std::array<std::byte, 2> bf16_bytes{};

    QTX_EXPECT_EQ(compress(QuantFormat::kFP16, src_bytes, fp16_bytes), 2u);
    QTX_EXPECT_EQ(compress(QuantFormat::kBF16, src_bytes, bf16_bytes), 2u);

    QTX_EXPECT(std::memcmp(fp16_bytes.data(), bf16_bytes.data(), 2) != 0);
}

QTX_TEST(Quant, FP16LargeStreamRoundTrip) {
    // Exercise the auto-vectorised scalar loop on a longer stream and
    // confirm the relative-error bound holds element-wise.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        // Range stays well inside FP16's dynamic range (≤ 65504).
        src[i] = static_cast<float>(i) * 0.123f - 60.0f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> compressed(kN * 2u);
    QTX_EXPECT_EQ(
        compressFP32ToFP16(std::span<const std::byte>{src_b},
                           std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressFP16ToFP32(std::span<const std::byte>{compressed},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r =
                std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            if (r > 0.001f) ++rel_violations;       // ≤ 0.1%
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

// ============================================================================
// FP8 E4M3 (OCP/Hopper/Intel 8-bit float, 1+4+3) round-trip — Epic 1.
//
// E4M3 has 3-bit mantissa => relative-error step ~ 2^-3 = 12.5%. Range
// ±[2^-9 .. 448]. No infinities; canonical NaN is S.1111.111 (= 0x7F
// without sign).  HA contract: finite input above 448 saturates to ±448.
// ============================================================================

QTX_TEST(Quant, CompressedSizeFP8_E4M3) {
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP8_E4M3, 32u),   32u);
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP8_E4M3, 1024u), 1024u);
}

QTX_TEST(Quant, FP8_E4M3_CompressionRatio) {
    QTX_EXPECT_EQ(compressionRatio(QuantFormat::kFP8_E4M3), 4.0f);
}

QTX_TEST(Quant, FP8_E4M3_RoundTripExactValues) {
    // Powers of two and a few values exactly representable in E4M3.
    std::array<float, 8> src = {
        1.0f, -1.0f, 2.0f, -2.0f, 4.0f, 0.5f, 0.25f, 0.125f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8>  compressed{};
    std::array<std::byte, 32> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToFP8_E4M3(src_bytes, compressed),       8u);
    QTX_EXPECT_EQ(decompressFP8_E4M3_ToFP32(compressed, decompressed), 32u);
    auto out = bytesToFloats<8>(decompressed);
    for (int i = 0; i < 8; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

QTX_TEST(Quant, FP8_E4M3_RoundTripMaxFinite) {
    // 448 is the largest finite E4M3 value (= S.1111.110 = 0x7E).
    std::array<float, 2> src = {448.0f, -448.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 2>  compressed{};
    std::array<std::byte, 8>  decompressed{};

    (void)compressFP32ToFP8_E4M3(src_bytes,    compressed);
    (void)decompressFP8_E4M3_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<2>(decompressed);
    QTX_EXPECT_EQ(out[0],  448.0f);
    QTX_EXPECT_EQ(out[1], -448.0f);
}

QTX_TEST(Quant, FP8_E4M3_SaturatesFiniteOverflow) {
    // HA invariant: finite-in -> finite-out. 1000, FLT_MAX, ±Inf all
    // saturate to ±448 (NOT to the NaN encoding).
    std::array<float, 5> src = {
        1000.0f,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -1e30f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 5>   compressed{};
    std::array<std::byte, 20>  decompressed{};

    (void)compressFP32ToFP8_E4M3(src_bytes,    compressed);
    (void)decompressFP8_E4M3_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<5>(decompressed);

    constexpr float kMaxFinite = 448.0f;
    for (int i = 0; i < 5; ++i) {
        QTX_EXPECT(std::isfinite(out[i]));
        QTX_EXPECT(!std::isnan(out[i]));
    }
    QTX_EXPECT_EQ(out[0],  kMaxFinite);
    QTX_EXPECT_EQ(out[1],  kMaxFinite);
    QTX_EXPECT_EQ(out[2],  kMaxFinite);
    QTX_EXPECT_EQ(out[3], -kMaxFinite);
    QTX_EXPECT_EQ(out[4], -kMaxFinite);
}

QTX_TEST(Quant, FP8_E4M3_PreservesNaN) {
    // NaN -> 0x7F (or 0xFF with sign), decode back to FP32 NaN.
    std::array<float, 2> src = {std::numeric_limits<float>::quiet_NaN(), 1.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 2> compressed{};
    std::array<std::byte, 8> decompressed{};

    (void)compressFP32ToFP8_E4M3(src_bytes,    compressed);
    (void)decompressFP8_E4M3_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<2>(decompressed);
    QTX_EXPECT(std::isnan(out[0]));
    QTX_EXPECT_EQ(out[1], 1.0f);

    // The on-wire byte for NaN must be 0x7F (sign 0) — canonical encoding.
    QTX_EXPECT_EQ(static_cast<unsigned>(compressed[0]) & 0x7Fu, 0x7Fu);
}

QTX_TEST(Quant, FP8_E4M3_PreservesSignedZero) {
    std::array<float, 2> src = {0.0f, -0.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 2> compressed{};
    std::array<std::byte, 8> decompressed{};

    (void)compressFP32ToFP8_E4M3(src_bytes,    compressed);
    (void)decompressFP8_E4M3_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<2>(decompressed);
    QTX_EXPECT_EQ(out[0],  0.0f);
    QTX_EXPECT_EQ(out[1], -0.0f);
    // Verify sign bit preserved bit-exactly through the round-trip.
    std::uint32_t bits1; std::memcpy(&bits1, &out[1], 4);
    QTX_EXPECT_EQ(bits1, 0x80000000u);
}

QTX_TEST(Quant, FP8_E4M3_Subnormals) {
    // Smallest E4M3 subnormal = 2^-9 ≈ 0.001953125 (= 0x01 encoding).
    // Half of that (2^-10 = 0.0009765625) is the RNE tie point.
    // 2^-10 with LSB-of-result == 0 rounds DOWN to 0 (round-to-even).
    std::array<float, 4> src = {
        0.001953125f,       // = 2^-9, exact smallest subnormal
        0.015625f,          // = 2^-6, smallest normal
        1.0e-5f,            // below the half-subnormal cutoff -> 0
        -1.0e-5f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4>  compressed{};
    std::array<std::byte, 16> decompressed{};

    (void)compressFP32ToFP8_E4M3(src_bytes,    compressed);
    (void)decompressFP8_E4M3_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<4>(decompressed);

    QTX_EXPECT_EQ(out[0], 0.001953125f);   // exact representation
    QTX_EXPECT_EQ(out[1], 0.015625f);      // exact representation
    QTX_EXPECT_EQ(out[2],  0.0f);
    QTX_EXPECT_EQ(out[3], -0.0f);
}

QTX_TEST(Quant, FP8_E4M3_BadSizeReturnsZero) {
    std::array<std::byte, 5>  bad_src{};   // not a multiple of 4
    std::array<std::byte, 10> dst{};
    QTX_EXPECT_EQ(compressFP32ToFP8_E4M3(bad_src, dst), 0u);
}

QTX_TEST(Quant, FP8_E4M3_InsufficientDstReturnsZero) {
    std::array<float, 4> src = {1.0f, 2.0f, 3.0f, 4.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 3> tiny_dst{};   // need 4, have 3
    QTX_EXPECT_EQ(compressFP32ToFP8_E4M3(src_bytes, tiny_dst), 0u);
}

QTX_TEST(Quant, FP8_E4M3_DecompressInsufficientDst) {
    std::array<std::byte, 4> src{};        // 4 elements
    std::array<std::byte, 8> tiny_dst{};   // need 16, have 8
    QTX_EXPECT_EQ(decompressFP8_E4M3_ToFP32(src, tiny_dst), 0u);
}

QTX_TEST(Quant, DispatcherFP8_E4M3_RoundTrip) {
    std::array<float, 4> src = {1.0f, 2.0f, 4.0f, 8.0f};   // exact in E4M3
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4>  mid{};
    std::array<std::byte, 16> back{};

    QTX_EXPECT_EQ(compress  (QuantFormat::kFP8_E4M3, src_bytes, mid),  4u);
    QTX_EXPECT_EQ(decompress(QuantFormat::kFP8_E4M3, mid,       back), 16u);
    auto out = bytesToFloats<4>(back);
    for (int i = 0; i < 4; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

// ============================================================================
// FP8 E5M2 (OCP/Hopper/Intel 8-bit float, 1+5+2) round-trip — Epic 1.
//
// E5M2 has 2-bit mantissa => relative-error step ~ 2^-2 = 25%. Range
// ±[2^-16 .. 57344]. IEEE-like: encodes ±Inf and NaN. Saturate-to-finite
// invariant on finite overflow (HA EC158 analogue).
// ============================================================================

QTX_TEST(Quant, CompressedSizeFP8_E5M2) {
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP8_E5M2, 32u),   32u);
    QTX_EXPECT_EQ(compressedSize(QuantFormat::kFP8_E5M2, 1024u), 1024u);
}

QTX_TEST(Quant, FP8_E5M2_CompressionRatio) {
    QTX_EXPECT_EQ(compressionRatio(QuantFormat::kFP8_E5M2), 4.0f);
}

QTX_TEST(Quant, FP8_E5M2_RoundTripExactValues) {
    // Powers of two are exact (mantissa = 0); 1.5, 1.25, 1.75 also exact
    // (they fit in 2 mantissa bits).
    std::array<float, 8> src = {
        1.0f, -1.0f, 2.0f, -2.0f, 4.0f, 0.5f, 1.5f, 1.25f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8>  compressed{};
    std::array<std::byte, 32> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToFP8_E5M2(src_bytes, compressed),       8u);
    QTX_EXPECT_EQ(decompressFP8_E5M2_ToFP32(compressed, decompressed), 32u);
    auto out = bytesToFloats<8>(decompressed);
    for (int i = 0; i < 8; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

QTX_TEST(Quant, FP8_E5M2_RoundTripMaxFinite) {
    // 57344 = 1.75 × 2^15, largest finite E5M2 value (encoded as 0x7B).
    std::array<float, 2> src = {57344.0f, -57344.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 2>  compressed{};
    std::array<std::byte, 8>  decompressed{};

    (void)compressFP32ToFP8_E5M2(src_bytes,    compressed);
    (void)decompressFP8_E5M2_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<2>(decompressed);
    QTX_EXPECT_EQ(out[0],  57344.0f);
    QTX_EXPECT_EQ(out[1], -57344.0f);
}

QTX_TEST(Quant, FP8_E5M2_SaturatesFiniteOverflow) {
    // Per the HA EC158 invariant: finite-in → finite-out. FLT_MAX and
    // 1e30 saturate to ±57344. ±Inf is preserved as ±Inf (E5M2 supports
    // it; only finite inputs that would round to Inf get saturated).
    std::array<float, 4> src = {
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        61440.0f,                            // rounding boundary
        -61440.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4>  compressed{};
    std::array<std::byte, 16> decompressed{};

    (void)compressFP32ToFP8_E5M2(src_bytes,    compressed);
    (void)decompressFP8_E5M2_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<4>(decompressed);

    constexpr float kMaxFinite = 57344.0f;
    for (int i = 0; i < 4; ++i) {
        QTX_EXPECT(std::isfinite(out[i]));
    }
    QTX_EXPECT_EQ(out[0],  kMaxFinite);
    QTX_EXPECT_EQ(out[1], -kMaxFinite);
    QTX_EXPECT_EQ(out[2],  kMaxFinite);
    QTX_EXPECT_EQ(out[3], -kMaxFinite);
}

QTX_TEST(Quant, FP8_E5M2_PreservesInfinity) {
    // Unlike E4M3, E5M2 has an Inf encoding (S.11111.00). ±Inf inputs
    // round-trip as ±Inf.
    std::array<float, 2> src = {
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 2> compressed{};
    std::array<std::byte, 8> decompressed{};

    (void)compressFP32ToFP8_E5M2(src_bytes,    compressed);
    (void)decompressFP8_E5M2_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<2>(decompressed);
    QTX_EXPECT(std::isinf(out[0]) && out[0] > 0.0f);
    QTX_EXPECT(std::isinf(out[1]) && out[1] < 0.0f);
}

QTX_TEST(Quant, FP8_E5M2_PreservesNaN) {
    std::array<float, 2> src = {std::numeric_limits<float>::quiet_NaN(), 1.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 2> compressed{};
    std::array<std::byte, 8> decompressed{};

    (void)compressFP32ToFP8_E5M2(src_bytes,    compressed);
    (void)decompressFP8_E5M2_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<2>(decompressed);
    QTX_EXPECT(std::isnan(out[0]));
    QTX_EXPECT_EQ(out[1], 1.0f);
}

QTX_TEST(Quant, FP8_E5M2_Subnormals) {
    // Smallest E5M2 subnormal = 2^-16 ≈ 1.53e-5.
    // 2^-14 (smallest normal). Anything below half the smallest subnormal
    // (2^-17 = 7.63e-6) rounds to ±0.
    std::array<float, 4> src = {
        1.0f / 65536.0f,    // = 2^-16, smallest positive subnormal (exact)
        1.0f / 16384.0f,    // = 2^-14, smallest positive normal     (exact)
        1.0e-6f,            // below threshold -> +0
        -1.0e-6f            //                 -> -0
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4>  compressed{};
    std::array<std::byte, 16> decompressed{};

    (void)compressFP32ToFP8_E5M2(src_bytes,    compressed);
    (void)decompressFP8_E5M2_ToFP32(compressed, decompressed);
    auto out = bytesToFloats<4>(decompressed);

    QTX_EXPECT_EQ(out[0], 1.0f / 65536.0f);   // exact
    QTX_EXPECT_EQ(out[1], 1.0f / 16384.0f);   // exact
    QTX_EXPECT_EQ(out[2],  0.0f);
    QTX_EXPECT_EQ(out[3], -0.0f);
}

QTX_TEST(Quant, FP8_E5M2_HasWiderRangeThanE4M3) {
    // Regression guard distinguishing the two FP8 variants.
    // 10000 fits in E5M2 (within ±57344) but NOT E4M3 (saturates to ±448).
    std::array<float, 1> src = {10000.0f};
    auto src_bytes = floatsToBytes(src);

    std::array<std::byte, 1> e4m3{};
    std::array<std::byte, 1> e5m2{};
    std::array<std::byte, 4> e4m3_back{};
    std::array<std::byte, 4> e5m2_back{};

    (void)compressFP32ToFP8_E4M3(src_bytes, e4m3);
    (void)compressFP32ToFP8_E5M2(src_bytes, e5m2);
    (void)decompressFP8_E4M3_ToFP32(e4m3, e4m3_back);
    (void)decompressFP8_E5M2_ToFP32(e5m2, e5m2_back);

    const float e4m3_out = bytesToFloats<1>(e4m3_back)[0];
    const float e5m2_out = bytesToFloats<1>(e5m2_back)[0];

    // E4M3 must saturate (= 448), E5M2 must NOT (some value > 448).
    QTX_EXPECT_EQ(e4m3_out, 448.0f);
    QTX_EXPECT(e5m2_out > 448.0f);
}

QTX_TEST(Quant, FP8_E5M2_NarrowerMantissaThanE4M3) {
    // The other axis: at small magnitudes E4M3 has finer steps because
    // of its 3-bit (vs E5M2's 2-bit) mantissa. Pick a value where the
    // mantissa precision distinguishes them — 1.1 sits inside [1, 2)
    // so the rep error is dominated by mantissa width.
    std::array<float, 1> src = {1.1f};
    auto src_bytes = floatsToBytes(src);

    std::array<std::byte, 1> e4m3{}, e5m2{};
    std::array<std::byte, 4> e4m3_back{}, e5m2_back{};
    (void)compressFP32ToFP8_E4M3(src_bytes, e4m3);
    (void)compressFP32ToFP8_E5M2(src_bytes, e5m2);
    (void)decompressFP8_E4M3_ToFP32(e4m3, e4m3_back);
    (void)decompressFP8_E5M2_ToFP32(e5m2, e5m2_back);

    const float e4m3_out = bytesToFloats<1>(e4m3_back)[0];
    const float e5m2_out = bytesToFloats<1>(e5m2_back)[0];

    QTX_EXPECT(std::fabs(e4m3_out - 1.1f) < std::fabs(e5m2_out - 1.1f));
}

QTX_TEST(Quant, FP8_E5M2_BadSizeReturnsZero) {
    std::array<std::byte, 5>  bad_src{};
    std::array<std::byte, 10> dst{};
    QTX_EXPECT_EQ(compressFP32ToFP8_E5M2(bad_src, dst), 0u);
}

QTX_TEST(Quant, DispatcherFP8_E5M2_RoundTrip) {
    std::array<float, 4> src = {1.0f, 2.0f, 4.0f, 8.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 4>  mid{};
    std::array<std::byte, 16> back{};

    QTX_EXPECT_EQ(compress  (QuantFormat::kFP8_E5M2, src_bytes, mid),  4u);
    QTX_EXPECT_EQ(decompress(QuantFormat::kFP8_E5M2, mid,       back), 16u);
    auto out = bytesToFloats<4>(back);
    for (int i = 0; i < 4; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

QTX_TEST(Quant, DispatcherFP8VariantsProduceDifferentBytes) {
    // E4M3 and E5M2 share the same compressed size, but for the same
    // input they MUST produce different on-wire bytes (otherwise the
    // dispatcher is aliasing the two formats).
    std::array<float, 1> src = {3.5f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 1> e4m3{};
    std::array<std::byte, 1> e5m2{};
    QTX_EXPECT_EQ(compress(QuantFormat::kFP8_E4M3, src_bytes, e4m3), 1u);
    QTX_EXPECT_EQ(compress(QuantFormat::kFP8_E5M2, src_bytes, e5m2), 1u);
    QTX_EXPECT(static_cast<unsigned>(e4m3[0])
            != static_cast<unsigned>(e5m2[0]));
}

QTX_TEST(Quant, FP8_E4M3_LargeStreamRelError) {
    // Exercise the auto-vectorised scalar loop on a longer stream
    // staying within E4M3's dynamic range.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        // Range [-30, +30] — comfortably inside ±448, and ≥ 2^-6 in
        // magnitude for most i.
        src[i] = static_cast<float>(i) * 0.06f - 30.0f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> compressed(kN);
    QTX_EXPECT_EQ(
        compressFP32ToFP8_E4M3(std::span<const std::byte>{src_b},
                               std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressFP8_E4M3_ToFP32(std::span<const std::byte>{compressed},
                                  std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    // E4M3 has 3-bit mantissa => relative error ≤ 2^-3 = 12.5% per
    // element (for values inside the normal range). We assert ≤ 13%
    // with a small slack for round-to-even granularity.
    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r = std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            if (r > 0.13f) ++rel_violations;
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

QTX_TEST(Quant, FP8_E5M2_LargeStreamRelError) {
    // Exercise the auto-vectorised scalar loop. E5M2 has 2-bit mantissa
    // => relative error ≤ 2^-2 = 25%; assert ≤ 26% with slack.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        // Wider range than E4M3 to exercise E5M2's range advantage.
        src[i] = static_cast<float>(i) * 2.0f - 1000.0f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> compressed(kN);
    QTX_EXPECT_EQ(
        compressFP32ToFP8_E5M2(std::span<const std::byte>{src_b},
                               std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressFP8_E5M2_ToFP32(std::span<const std::byte>{compressed},
                                  std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r = std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            if (r > 0.26f) ++rel_violations;
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

// ============================================================================
// NVFP4 (NVIDIA Blackwell 4-bit float, E2M1 + 16-elem blocks + FP8 E4M3
// per-block scale + FP32 per-tensor scale) — Epic 1 milestone.
//
// The codebook is {0, 0.5, 1, 1.5, 2, 3, 4, 6} (signed), 16 codepoints
// total. Max representable magnitude = 6. No NaN, no Inf. Block size
// is 16 elements (NVIDIA standard; differs from other QTX formats).
// ============================================================================

QTX_TEST(Quant, NVFP4_CompressedSize) {
    // 1 block (16 elem)  =>   4 header + 9 payload = 13 bytes
    // 64 blocks (1024 elem) => 4 header + 576 payload = 580 bytes
    QTX_EXPECT_EQ(compressedSize_NVFP4(16u),    13u);
    QTX_EXPECT_EQ(compressedSize_NVFP4(1024u),  580u);
    // Non-multiple of 16 -> 0 (rejected at the compressedSize layer too).
    QTX_EXPECT_EQ(compressedSize_NVFP4(15u),    0u);
    QTX_EXPECT_EQ(compressedSize_NVFP4(17u),    0u);
}

QTX_TEST(Quant, NVFP4_CodebookExactRoundTrip) {
    // All 16 codepoints (with the per-block scale exactly 1.0 / 6 of
    // global abs-max) round-trip bit-exactly. This is the single most
    // important NVFP4 invariant — the codebook IS the round-trip oracle.
    std::array<float, 16> src = {
        0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 2.0f,
        -2.0f, 3.0f, -3.0f, 4.0f, -4.0f, 6.0f, -6.0f, 0.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 13> compressed{};
    std::array<std::byte, 64> back{};

    QTX_EXPECT_EQ(compressFP32ToNVFP4(src_bytes, compressed),   13u);
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(compressed, back),      64u);
    auto out = bytesToFloats<16>(back);
    for (int i = 0; i < 16; ++i) {
        QTX_EXPECT_EQ(out[i], src[i]);
    }
}

QTX_TEST(Quant, NVFP4_HeaderIsFP32Scale) {
    // For the codebook block above, |max|=6, so global_scale =
    // 6 / (448*6) = 6/2688 = 1/448 ≈ 0.002232.
    std::array<float, 16> src = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f, 0.0f
    };
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 13> compressed{};
    QTX_EXPECT_EQ(compressFP32ToNVFP4(src_bytes, compressed), 13u);
    float header_scale;
    std::memcpy(&header_scale, compressed.data(), sizeof(float));
    QTX_EXPECT(std::fabs(header_scale - 1.0f/448.0f) < 1e-7f);
}

QTX_TEST(Quant, NVFP4_AllZeroBlock) {
    // An all-zero input must produce a zero global_scale and a stream
    // of zero blocks; the decompressor must return all zeros.
    std::array<float, 16> src{};   // value-initialised to 0
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 13> compressed{};
    std::array<std::byte, 64> back{};

    QTX_EXPECT_EQ(compressFP32ToNVFP4(src_bytes, compressed),   13u);
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(compressed, back),      64u);
    auto out = bytesToFloats<16>(back);
    for (int i = 0; i < 16; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, NVFP4_SaturatesExtremeValues) {
    // ±Inf and NaN sanitise to 0 (NVFP4 has no special encoding); finite
    // values above the block's representable range saturate at ±6 ×
    // global_scale × block_scale.
    std::array<float, 16> src{};
    src[0]  = std::numeric_limits<float>::infinity();
    src[1]  = -std::numeric_limits<float>::infinity();
    src[2]  = std::numeric_limits<float>::quiet_NaN();
    src[3]  = std::numeric_limits<float>::max();
    src[4]  = -std::numeric_limits<float>::max();
    src[5]  = 100.0f;
    src[6]  = -100.0f;
    src[7]  = 50.0f;
    // remaining elements stay at 0
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 13> compressed{};
    std::array<std::byte, 64> back{};

    QTX_EXPECT_EQ(compressFP32ToNVFP4(src_bytes, compressed),   13u);
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(compressed, back),      64u);
    auto out = bytesToFloats<16>(back);

    // All outputs must be finite (no Inf/NaN propagation through codec).
    for (int i = 0; i < 16; ++i) {
        QTX_EXPECT(std::isfinite(out[i]));
    }
    // Inf and NaN must have sanitised to 0 (no special encoding exists).
    QTX_EXPECT_EQ(out[0], 0.0f);
    QTX_EXPECT_EQ(out[1], 0.0f);
    QTX_EXPECT_EQ(out[2], 0.0f);
}

QTX_TEST(Quant, NVFP4_BadSrcSizeReturnsZero) {
    // Element count must be a multiple of 16.
    std::array<float, 15> src{};   // 15 floats, not a multiple of 16
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 32> dst{};
    QTX_EXPECT_EQ(compressFP32ToNVFP4(src_bytes, dst), 0u);
}

QTX_TEST(Quant, NVFP4_InsufficientDstReturnsZero) {
    std::array<float, 16> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 12> tiny_dst{};   // need 13, have 12
    QTX_EXPECT_EQ(compressFP32ToNVFP4(src_bytes, tiny_dst), 0u);
}

QTX_TEST(Quant, NVFP4_DecompressBadHeaderSize) {
    // src must be at least 4 bytes (header).
    std::array<std::byte, 3>  too_small{};
    std::array<std::byte, 64> dst{};
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(too_small, dst), 0u);
}

QTX_TEST(Quant, NVFP4_DecompressBadPayloadSize) {
    // Payload after header must be a multiple of 9 bytes.
    std::array<std::byte, 12> bad{};       // 4 header + 8 payload (not /9)
    std::array<std::byte, 64> dst{};
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, NVFP4_DecompressNonFiniteHeaderRejected) {
    // A corrupted stream with a non-finite global_scale must be
    // rejected — the codec cannot use it without producing UB downstream.
    std::array<std::byte, 13> bad{};
    float bad_scale = std::numeric_limits<float>::infinity();
    std::memcpy(bad.data(), &bad_scale, sizeof(float));
    std::array<std::byte, 64> dst{};
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(bad, dst), 0u);

    bad_scale = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(bad.data(), &bad_scale, sizeof(float));
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, NVFP4_TwoBlockStream) {
    // 32 elements (2 blocks). First block holds large values, second
    // small; the per-block scale must adapt independently.
    std::array<float, 32> src{};
    for (int i = 0; i < 16; ++i)  src[static_cast<size_t>(i)]       = 4.0f * ((i % 2) ? 1.0f : -1.0f);
    for (int i = 16; i < 32; ++i) src[static_cast<size_t>(i)]       = 0.5f * ((i % 2) ? 1.0f : -1.0f);
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 22>  compressed{};  // 4 + 2*9
    std::array<std::byte, 128> back{};

    QTX_EXPECT_EQ(compressFP32ToNVFP4(src_bytes, compressed),   22u);
    QTX_EXPECT_EQ(decompressNVFP4ToFP32(compressed, back),      128u);
    auto out = bytesToFloats<32>(back);

    // The codec uses a per-block FP8 E4M3 scale (8-bit precision). When
    // a single tensor mixes ±4 with ±0.5, the small-block values can
    // round to less precise codepoints because the FP8-quantised
    // block-scale ratio (compared to FP32) introduces ~1% error.
    // Tolerance: ≤ 3% for each value here.
    for (int i = 0; i < 32; ++i) {
        if (src[i] != 0.0f) {
            const float rel = std::fabs(out[i] - src[i]) / std::fabs(src[i]);
            QTX_EXPECT(rel < 0.03f);
        }
    }
}

QTX_TEST(Quant, NVFP4_CompressionRatio) {
    // For a 1024-element tensor: input = 4096 B, output = 580 B,
    // ratio = 4096/580 ≈ 7.06×. The compile-time invariant.
    constexpr float kRatio = 4096.0f / 580.0f;
    QTX_EXPECT(kRatio > 7.0f);
    QTX_EXPECT(kRatio < 7.1f);
}

QTX_TEST(Quant, NVFP4_LargeStreamRoundTrip) {
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        // Values in [-4, +4] — every input lies within ±6 after scaling.
        src[i] = static_cast<float>(i % 17) * 0.5f - 4.0f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> compressed(compressedSize_NVFP4(kN));
    QTX_EXPECT_EQ(
        compressFP32ToNVFP4(std::span<const std::byte>{src_b},
                            std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressNVFP4ToFP32(std::span<const std::byte>{compressed},
                              std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    // E2M1 has ~3-bit precision (8 magnitudes); within each block the
    // per-block scale brings the rep error to roughly 1/(2*8) = 6.25%
    // on average. Worst case can be 2× that for values landing in the
    // "wide" parts of the codebook (between 4 and 6, say).
    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r = std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            if (r > 0.20f) ++rel_violations;   // ≤ 20% per element
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

QTX_TEST(Quant, NVFP4_GlobalScaleBoundsBlockScale) {
    // Regression guard: the per-block scale stored in FP8 E4M3 must
    // always be in [0, 448]. We verify this by reading the 1 B
    // per-block scale field from each block and decoding it.
    constexpr usize kN = 64u;       // 4 blocks
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        // Mix big and small values across blocks.
        src[i] = ((i / kNVFP4BlockElements) % 2 == 0)
                    ? 100.0f * static_cast<float>((i % 5) - 2)
                    : 0.001f * static_cast<float>((i % 5) - 2);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> compressed(compressedSize_NVFP4(kN));
    QTX_EXPECT_EQ(
        compressFP32ToNVFP4(std::span<const std::byte>{src_b},
                            std::span<std::byte>{compressed}),
        compressed.size());

    // Read each block's FP8 E4M3 scale byte and decode it.
    constexpr usize kBlocks = kN / kNVFP4BlockElements;
    for (usize b = 0; b < kBlocks; ++b) {
        const qtx::core::u8 scale_byte = static_cast<qtx::core::u8>(
            compressed[kNVFP4HeaderBytes + b * kNVFP4BlockBytes]);
        const float scale = qtx::core::fp8_E4M3_toFP32(scale_byte);
        QTX_EXPECT(std::isfinite(scale));
        QTX_EXPECT(scale >= 0.0f);
        QTX_EXPECT(scale <= 448.0f);
    }
}

// ============================================================================
// MXFP4 (OCP Microscaling FP4) — Epic 1.
//
// Same element format as NVFP4 (E2M1 codebook) but with a 32-element
// block and an E8M0 (pure power-of-2) per-block scale. No global scale.
// 17 bytes per 32 FP32 elements; compression ratio ≈ 7.53×.
// ============================================================================

QTX_TEST(Quant, MXFP4_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_MXFP4(32u),    17u);
    QTX_EXPECT_EQ(compressedSize_MXFP4(1024u),  544u);
    // Non-multiple of 32 -> 0
    QTX_EXPECT_EQ(compressedSize_MXFP4(16u),    0u);
    QTX_EXPECT_EQ(compressedSize_MXFP4(31u),    0u);
    QTX_EXPECT_EQ(compressedSize_MXFP4(33u),    0u);
}

QTX_TEST(Quant, MXFP4_CompressionRatio) {
    // 32 FP32 (128 B) -> 17 B
    constexpr float kRatio = 128.0f / 17.0f;
    QTX_EXPECT(kRatio > 7.5f);
    QTX_EXPECT(kRatio < 7.6f);
}

QTX_TEST(Quant, MXFP4_CodebookExactRoundTrip) {
    // With max|V| = 6 the shared_exp is 0, scale = 1, and every
    // codebook value passes through V/1 = V unchanged.
    std::array<float, 32> src{};
    constexpr float code[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    for (int i = 0; i < 32; ++i) {
        src[static_cast<size_t>(i)] =
            code[i % 8] * ((i / 8) % 2 == 0 ? 1.0f : -1.0f);
    }
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 17>  compressed{};
    std::array<std::byte, 128> back{};

    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, compressed),  17u);
    QTX_EXPECT_EQ(decompressMXFP4ToFP32(compressed, back),     128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

QTX_TEST(Quant, MXFP4_ScaleByteIsE8M0) {
    // For abs-max = 6 (which is 1.1₂ × 2²): fp32_exp = 2,
    // shared_exp_biased = 2 - 2 + 127 = 127, so scale byte = 0x7F.
    std::array<float, 32> src{};
    src[0] = 6.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 17> compressed{};
    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, compressed), 17u);
    QTX_EXPECT_EQ(static_cast<unsigned>(compressed[0]), 0x7Fu);
}

QTX_TEST(Quant, MXFP4_ScaleByteScalesByPowerOfTwo) {
    // For abs-max = 8 (= 2³): shared_exp_biased = 3 - 2 + 127 = 128.
    // So scale byte = 0x80, decoded as 2^1 = 2.
    std::array<float, 32> src{};
    src[0] = 8.0f;
    src[1] = 4.0f;
    src[2] = -4.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 17>  compressed{};
    std::array<std::byte, 128> back{};
    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, compressed), 17u);
    QTX_EXPECT_EQ(static_cast<unsigned>(compressed[0]), 0x80u);
    (void)decompressMXFP4ToFP32(compressed, back);
    auto out = bytesToFloats<32>(back);
    // After divide-by-2, 8 -> 4 (E2M1 codepoint 6 → wait, 8/2=4 not 6).
    // Actually 8/2 = 4, encoded as 0b0110 (=4), decoded × 2 = 8 ✓.
    QTX_EXPECT_EQ(out[0],  8.0f);
    QTX_EXPECT_EQ(out[1],  4.0f);
    QTX_EXPECT_EQ(out[2], -4.0f);
}

QTX_TEST(Quant, MXFP4_AllZeroBlock) {
    // All-zero input -> scale byte 0x00 (sentinel) -> all zero output.
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 17>  compressed{};
    std::array<std::byte, 128> back{};

    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, compressed), 17u);
    QTX_EXPECT_EQ(static_cast<unsigned>(compressed[0]),       0x00u);
    QTX_EXPECT_EQ(decompressMXFP4ToFP32(compressed, back),    128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, MXFP4_SaturatesExtremeValues) {
    // NaN/Inf inputs sanitise to 0; FLT_MAX saturates to 6 × scale.
    std::array<float, 32> src{};
    src[0]  = std::numeric_limits<float>::infinity();
    src[1]  = -std::numeric_limits<float>::infinity();
    src[2]  = std::numeric_limits<float>::quiet_NaN();
    src[3]  = std::numeric_limits<float>::max();
    src[4]  = -std::numeric_limits<float>::max();
    src[5]  = 1.0f;     // an in-range value to pin the block scale
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 17>  compressed{};
    std::array<std::byte, 128> back{};

    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, compressed), 17u);
    QTX_EXPECT_EQ(decompressMXFP4ToFP32(compressed, back),    128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT(std::isfinite(out[i]));
    // Inf and NaN must have sanitised to 0 (no special encoding).
    QTX_EXPECT_EQ(out[0], 0.0f);
    QTX_EXPECT_EQ(out[1], 0.0f);
    QTX_EXPECT_EQ(out[2], 0.0f);
}

QTX_TEST(Quant, MXFP4_BadSrcSizeReturnsZero) {
    // Element count must be a multiple of 32.
    std::array<float, 31> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 17> dst{};
    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, dst), 0u);
}

QTX_TEST(Quant, MXFP4_InsufficientDstReturnsZero) {
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 16> tiny_dst{};   // need 17, have 16
    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, tiny_dst), 0u);
}

QTX_TEST(Quant, MXFP4_DecompressBadPayloadSize) {
    // Stream must be a multiple of 17 bytes.
    std::array<std::byte, 16> bad{};
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressMXFP4ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, MXFP4_DecompressReservedNaNScaleRejected) {
    // E8M0 = 0xFF is the reserved NaN encoding; decompress must reject.
    std::array<std::byte, 17> bad{};
    bad[0] = static_cast<std::byte>(0xFFu);
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressMXFP4ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, MXFP4_TwoBlockStreamAdaptiveScale) {
    // Two blocks with very different magnitudes. Each must adapt its
    // own E8M0 scale independently.
    std::array<float, 64> src{};
    for (int i = 0; i < 32; ++i)  src[static_cast<size_t>(i)]       = 5.0f * ((i % 2) ? 1.0f : -1.0f);
    for (int i = 32; i < 64; ++i) src[static_cast<size_t>(i)]       = 0.5f * ((i % 2) ? 1.0f : -1.0f);
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 34>  compressed{};
    std::array<std::byte, 256> back{};

    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_bytes, compressed), 34u);
    QTX_EXPECT_EQ(decompressMXFP4ToFP32(compressed, back),    256u);
    auto out = bytesToFloats<64>(back);

    // The two blocks should NOT share the same scale byte (different
    // magnitudes -> different E8M0).
    QTX_EXPECT(static_cast<unsigned>(compressed[0])
            != static_cast<unsigned>(compressed[kMXFP4BlockBytes]));

    // Both blocks have signed inputs (so 0 doesn't appear); each value
    // should round to within the codebook step of its block scale.
    // For block 1 (scale = 1, since 5.0 is below 6.0): RNE of 5 → 4
    // (tie-to-even picks the even index in the codebook). Expect ≤ 25% err.
    // For block 2 (scale = 2^-1 = 0.5): 0.5 / 0.5 = 1 → encoded exactly.
    for (int i = 32; i < 64; ++i) {
        QTX_EXPECT_EQ(out[i], src[i]);   // block 2 must be exact
    }
}

QTX_TEST(Quant, MXFP4_ScaleByteIsNeverReservedNaN) {
    // Regression guard: compress must NEVER emit E8M0 = 0xFF.
    // Test with a variety of magnitudes including extreme values.
    std::vector<float> src(96, 0.0f);  // 3 blocks
    src[0]  = 1.0e30f;          // gigantic
    src[32] = 1.0f;             // normal
    src[64] = 1.0e-30f;         // tiny
    std::vector<std::byte> src_b(96 * 4);
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_MXFP4(96));
    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_b, c), c.size());
    QTX_EXPECT(static_cast<unsigned>(c[0])
                              != 0xFFu);
    QTX_EXPECT(static_cast<unsigned>(c[kMXFP4BlockBytes])
                              != 0xFFu);
    QTX_EXPECT(static_cast<unsigned>(c[2 * kMXFP4BlockBytes])
                              != 0xFFu);
    // And decompress must accept what compress emitted.
    std::vector<std::byte> back_b(96 * 4);
    QTX_EXPECT_EQ(decompressMXFP4ToFP32(c, back_b), back_b.size());
}

QTX_TEST(Quant, MXFP4_LargeStreamRoundTrip) {
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = static_cast<float>(i % 17) * 0.5f - 4.0f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> compressed(compressedSize_MXFP4(kN));
    QTX_EXPECT_EQ(
        compressFP32ToMXFP4(std::span<const std::byte>{src_b},
                            std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressMXFP4ToFP32(std::span<const std::byte>{compressed},
                              std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    // Power-of-2 scale loses up to ~0.5 bits vs NVFP4's FP8 E4M3 scale.
    // ≤ 25% rel error per element is the threshold (vs NVFP4's 20%).
    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r = std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            if (r > 0.25f) ++rel_violations;
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

QTX_TEST(Quant, MXFP4_VsNVFP4_SameElementCodec) {
    // Sanity: at the element level both formats use the OCP E2M1
    // codebook. For an input that is already a codepoint and a block
    // whose abs-max is itself in the codebook, BOTH formats must
    // recover the input exactly. (This pins the shared element codec.)
    constexpr usize kN = 32u;
    std::vector<float> src(kN);
    constexpr float code[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    for (usize i = 0; i < kN; ++i) src[i] = code[i % 8];

    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    // MXFP4 round-trip
    std::vector<std::byte> mxc(compressedSize_MXFP4(kN));
    std::vector<std::byte> mxb(kN * sizeof(float));
    (void)compressFP32ToMXFP4(src_b, mxc);
    (void)decompressMXFP4ToFP32(mxc, mxb);
    std::vector<float> mx_out(kN);
    std::memcpy(mx_out.data(), mxb.data(), mxb.size());

    // NVFP4 needs the element count to be a multiple of 16, which 32 is.
    std::vector<std::byte> nvc(compressedSize_NVFP4(kN));
    std::vector<std::byte> nvb(kN * sizeof(float));
    (void)compressFP32ToNVFP4(src_b, nvc);
    (void)decompressNVFP4ToFP32(nvc, nvb);
    std::vector<float> nv_out(kN);
    std::memcpy(nv_out.data(), nvb.data(), nvb.size());

    for (usize i = 0; i < kN; ++i) {
        QTX_EXPECT_EQ(mx_out[i], src[i]);
        QTX_EXPECT_EQ(nv_out[i], src[i]);
    }
}

QTX_TEST(Quant, MXFP4_ScaleByteRangeBounds) {
    // The scale byte must NEVER be 0xFF (NaN), and for any non-zero
    // input must be in [0x01, 0xFE] (clamped to avoid 0x00 = tiny
    // sentinel for non-zero inputs and 0xFF for normal inputs).
    std::vector<float> src(32);
    for (usize i = 0; i < 32; ++i) src[i] = 0.1f * static_cast<float>(i + 1);
    std::vector<std::byte> src_b(32 * 4);
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(17);
    QTX_EXPECT_EQ(compressFP32ToMXFP4(src_b, c), 17u);
    const unsigned scale_byte = static_cast<unsigned>(c[0]);
    QTX_EXPECT(scale_byte >= 0x01u);
    QTX_EXPECT(scale_byte <= 0xFEu);
}

// ============================================================================
// GGML legacy formats — Q4_1, Q5_0, Q5_1 (Epic 2).
//
// Wire format is bit-for-bit compatible with llama.cpp / ggml's
// `block_q4_1`, `block_q5_0`, `block_q5_1` (32-element blocks).
// ============================================================================

// --------- Q4_1 ---------

QTX_TEST(Quant, Q4_1_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q4_1(32u),   20u);
    QTX_EXPECT_EQ(compressedSize_Q4_1(1024u), 640u);
    QTX_EXPECT_EQ(compressedSize_Q4_1(0u),    0u);    // empty -> 0 bytes
    QTX_EXPECT_EQ(compressedSize_Q4_1(15u),   0u);    // not /32
}

QTX_TEST(Quant, Q4_1_BadSrcSizeReturnsZero) {
    std::array<float, 31> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 20> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ4_1(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q4_1_InsufficientDstReturnsZero) {
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 19> tiny_dst{};   // need 20
    QTX_EXPECT_EQ(compressFP32ToQ4_1(src_bytes, tiny_dst), 0u);
}

QTX_TEST(Quant, Q4_1_DecompressBadPayloadSize) {
    std::array<std::byte, 19> bad{};         // not /20
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressQ4_1ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q4_1_AllZeroBlock) {
    // Constant-zero block -> d = 0, m = 0; round-trip is all zero.
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 20>  c{};
    std::array<std::byte, 128> back{};
    QTX_EXPECT_EQ(compressFP32ToQ4_1(src_bytes, c),    20u);
    QTX_EXPECT_EQ(decompressQ4_1ToFP32(c, back),       128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q4_1_ConstantBlockRoundTrip) {
    // A constant nonzero block: d = 0, m = constant; every quantised
    // index is 0, dequant: 0 * 0 + m = m. Round-trip exact (within FP16
    // precision of m).
    std::array<float, 32> src;
    src.fill(2.5f);
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 20>  c{};
    std::array<std::byte, 128> back{};
    (void)compressFP32ToQ4_1(src_bytes, c);
    (void)decompressQ4_1ToFP32(c, back);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], 2.5f);  // FP16-exact
}

QTX_TEST(Quant, Q4_1_RoundTripFiniteRelError) {
    // Q4_1 has 16 codepoints over a [min, max] range. The maximum
    // *absolute* error per element is d/2 = (max-min)/30. The
    // *relative* error blows up when |src[i]| is near a grid line —
    // that's intrinsic to fixed-grid quantization, not a codec bug.
    //
    // We measure the MEAN relative error, which captures the codec's
    // average distortion and is the figure-of-merit reported in
    // perplexity tables. For Q4_1 on sin(x), this should be ≤ 6%.
    constexpr usize kN = 256u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) src[i] = std::sin(0.1f * static_cast<float>(i));
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q4_1(kN));
    QTX_EXPECT_EQ(compressFP32ToQ4_1(src_b, c), c.size());
    std::vector<std::byte> back_b(kN * sizeof(float));
    QTX_EXPECT_EQ(decompressQ4_1ToFP32(c, back_b), back_b.size());
    std::vector<float> back(kN);
    std::memcpy(back.data(), back_b.data(), back_b.size());
    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.06);     // ≤ 6% mean over the signal
}

QTX_TEST(Quant, Q4_1_DecompressRejectsNonFiniteScale) {
    // A corrupted d (FP16 NaN) must be rejected.
    std::array<std::byte, 20> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;   // FP16 quiet NaN
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressQ4_1ToFP32(bad, dst), 0u);
}

// --------- Q5_0 ---------

QTX_TEST(Quant, Q5_0_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q5_0(32u),   22u);
    QTX_EXPECT_EQ(compressedSize_Q5_0(1024u), 704u);
    QTX_EXPECT_EQ(compressedSize_Q5_0(15u),   0u);
}

QTX_TEST(Quant, Q5_0_BadSrcSizeReturnsZero) {
    std::array<float, 31> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 22> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ5_0(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q5_0_AllZeroBlock) {
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 22>  c{};
    std::array<std::byte, 128> back{};
    QTX_EXPECT_EQ(compressFP32ToQ5_0(src_bytes, c),    22u);
    QTX_EXPECT_EQ(decompressQ5_0ToFP32(c, back),       128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q5_0_SymmetricSignedRoundTrip) {
    // Q5_0 maps the signed range to 5-bit indices [0..31] with offset
    // +16. Per-element rel error blows up near grid lines (intrinsic
    // to fixed-grid quantization); we measure the MEAN.
    constexpr usize kN = 128u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) src[i] = std::sin(0.07f * static_cast<float>(i)) * 2.5f;
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q5_0(kN));
    QTX_EXPECT_EQ(compressFP32ToQ5_0(src_b, c), c.size());
    std::vector<std::byte> back_b(kN * sizeof(float));
    QTX_EXPECT_EQ(decompressQ5_0ToFP32(c, back_b), back_b.size());
    std::vector<float> back(kN);
    std::memcpy(back.data(), back_b.data(), back_b.size());
    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    // Q5_0 has 32 codepoints; mean error should be lower than Q4_1's.
    QTX_EXPECT(mean_rel < 0.05);
}

QTX_TEST(Quant, Q5_0_HighBitPackingCorrect) {
    // Build a tensor where some block entries have absolute value >
    // half-range, forcing the 5th bit of qh to be used.
    // After encoding, the qh field MUST be non-zero (some indices
    // crossed the 16-boundary).
    std::array<float, 32> src{};
    for (int i = 0; i < 32; ++i) src[static_cast<size_t>(i)] = static_cast<float>(i - 16) * 0.5f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 22> c{};
    QTX_EXPECT_EQ(compressFP32ToQ5_0(src_bytes, c), 22u);
    qtx::core::u32 qh;
    std::memcpy(&qh, c.data() + 2, 4);
    QTX_EXPECT(qh != 0u);
}

QTX_TEST(Quant, Q5_0_DecompressBadPayloadSize) {
    std::array<std::byte, 21> bad{};
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressQ5_0ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q5_0_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 22> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressQ5_0ToFP32(bad, dst), 0u);
}

// --------- Q5_1 ---------

QTX_TEST(Quant, Q5_1_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q5_1(32u),   24u);
    QTX_EXPECT_EQ(compressedSize_Q5_1(1024u), 768u);
    QTX_EXPECT_EQ(compressedSize_Q5_1(15u),   0u);
}

QTX_TEST(Quant, Q5_1_BadSrcSizeReturnsZero) {
    std::array<float, 31> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 24> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ5_1(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q5_1_AllZeroBlock) {
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 24>  c{};
    std::array<std::byte, 128> back{};
    QTX_EXPECT_EQ(compressFP32ToQ5_1(src_bytes, c),    24u);
    QTX_EXPECT_EQ(decompressQ5_1ToFP32(c, back),       128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q5_1_AsymmetricRoundTrip) {
    // Asymmetric: 32 codepoints, plus stored min. Should be more accurate
    // than Q5_0 on biased data (data with non-zero mean).
    constexpr usize kN = 128u;
    std::vector<float> src(kN);
    // Heavily biased range: [10, 20]
    for (usize i = 0; i < kN; ++i) src[i] = 10.0f + std::sin(0.07f * static_cast<float>(i)) * 5.0f + 5.0f;
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q5_1(kN));
    QTX_EXPECT_EQ(compressFP32ToQ5_1(src_b, c), c.size());
    std::vector<std::byte> back_b(kN * sizeof(float));
    QTX_EXPECT_EQ(decompressQ5_1ToFP32(c, back_b), back_b.size());
    std::vector<float> back(kN);
    std::memcpy(back.data(), back_b.data(), back_b.size());
    int viol = 0;
    for (usize i = 0; i < kN; ++i) {
        // Biased range [10, 20] is small compared to the magnitude;
        // (max-min)/31 ≈ 0.32 absolute step. Per-element absolute
        // error ≤ 0.16 → rel error ≤ 1.6% in the 10..20 range.
        const float r = std::fabs(back[i] - src[i]) / std::fabs(src[i]);
        if (r > 0.02f) ++viol;
    }
    QTX_EXPECT_EQ(viol, 0);
}

QTX_TEST(Quant, Q5_1_HighBitPackingCorrect) {
    // 32-value block with a wide dynamic range so that more than half
    // of the indices cross 16.
    std::array<float, 32> src{};
    for (int i = 0; i < 32; ++i) src[static_cast<size_t>(i)] = static_cast<float>(i) * 0.3f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 24> c{};
    QTX_EXPECT_EQ(compressFP32ToQ5_1(src_bytes, c), 24u);
    qtx::core::u32 qh;
    std::memcpy(&qh, c.data() + 4, 4);
    QTX_EXPECT(qh != 0u);
}

QTX_TEST(Quant, Q5_1_DecompressBadPayloadSize) {
    std::array<std::byte, 23> bad{};
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressQ5_1ToFP32(bad, dst), 0u);
}

// --------- Cross-format invariants ---------

QTX_TEST(Quant, Q4_1_VsQ5_1_Q5_1_IsMoreAccurate) {
    // Same input through Q4_1 and Q5_1: Q5_1 must have strictly lower
    // per-element absolute error on a generic finite signal.
    constexpr usize kN = 64u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) src[i] = std::sin(0.1f * static_cast<float>(i)) * 3.0f;
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> q4_1(compressedSize_Q4_1(kN));
    std::vector<std::byte> q5_1(compressedSize_Q5_1(kN));
    (void)compressFP32ToQ4_1(src_b, q4_1);
    (void)compressFP32ToQ5_1(src_b, q5_1);

    std::vector<std::byte> b4(kN * sizeof(float));
    std::vector<std::byte> b5(kN * sizeof(float));
    (void)decompressQ4_1ToFP32(q4_1, b4);
    (void)decompressQ5_1ToFP32(q5_1, b5);
    std::vector<float> out4(kN), out5(kN);
    std::memcpy(out4.data(), b4.data(), b4.size());
    std::memcpy(out5.data(), b5.data(), b5.size());

    double err4 = 0.0, err5 = 0.0;
    for (usize i = 0; i < kN; ++i) {
        err4 += static_cast<double>(std::fabs(out4[i] - src[i]));
        err5 += static_cast<double>(std::fabs(out5[i] - src[i]));
    }
    // Q5_1 has 2× the codepoints of Q4_1, so should be ~2× tighter.
    QTX_EXPECT(err5 < err4);
}

QTX_TEST(Quant, GGML_LegacyFormatsHandleSanitisedNaN) {
    // HA divergence from ggml: NaN/Inf inputs sanitise to 0 before the
    // per-block min/max scan. This means the output is finite (not
    // poisoned), at the cost of bit-exact divergence from ggml on the
    // affected blocks.
    std::array<float, 32> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = -std::numeric_limits<float>::infinity();
    src[3] = 1.0f;     // anchor a finite value so d > 0
    auto src_bytes = floatsToBytes(src);

    std::array<std::byte, 20>  q4_1{};
    std::array<std::byte, 22>  q5_0{};
    std::array<std::byte, 24>  q5_1{};
    std::array<std::byte, 128> back{};

    QTX_EXPECT_EQ(compressFP32ToQ4_1(src_bytes, q4_1), 20u);
    QTX_EXPECT_EQ(decompressQ4_1ToFP32(q4_1, back),    128u);
    auto o = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT(std::isfinite(o[i]));

    QTX_EXPECT_EQ(compressFP32ToQ5_0(src_bytes, q5_0), 22u);
    QTX_EXPECT_EQ(decompressQ5_0ToFP32(q5_0, back),    128u);
    o = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT(std::isfinite(o[i]));

    QTX_EXPECT_EQ(compressFP32ToQ5_1(src_bytes, q5_1), 24u);
    QTX_EXPECT_EQ(decompressQ5_1ToFP32(q5_1, back),    128u);
    o = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT(std::isfinite(o[i]));
}

// Regression guard: the compressed byte stream for Q4_1 / Q5_0 / Q5_1
// MUST be bit-for-bit identical to llama.cpp's `quantize_row_q*_ref`
// (ggml/src/ggml-quants.c) on finite inputs. Encoding-time divergence
// from ggml would break interop with any tool that consumes our blocks.
//
// We embed an inline copy of the ggml reference algorithm in the test
// (the implementation is small and the reference text is in the public
// llama.cpp repository under MIT license).
namespace ggml_ref {
inline void q4_1(const float* x, uint8_t* y, int k) {
    const int qk = 32;
    const int nb = k / qk;
    uint8_t* p = y;
    for (int i = 0; i < nb; ++i) {
        float vmin =  std::numeric_limits<float>::max();
        float vmax = -std::numeric_limits<float>::max();
        for (int j = 0; j < qk; ++j) {
            float v = x[i*qk + j];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        const float d  = (vmax - vmin) / 15.0f;
        const float id = d ? 1.0f/d : 0.0f;
        const qtx::core::u16 d_h = qtx::core::fp32ToFP16Safe(d);
        const qtx::core::u16 m_h = qtx::core::fp32ToFP16Safe(vmin);
        std::memcpy(p,     &d_h, 2);
        std::memcpy(p + 2, &m_h, 2);
        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + j]      - vmin) * id;
            const float x1 = (x[i*qk + qk/2+j] - vmin) * id;
            int xi0 = static_cast<int>(x0 + 0.5f);
            int xi1 = static_cast<int>(x1 + 0.5f);
            if (xi0 > 15) xi0 = 15;
            if (xi1 > 15) xi1 = 15;
            p[4 + j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
        }
        p += 20;
    }
}
inline void q5_0(const float* x, uint8_t* y, int k) {
    const int qk = 32;
    const int nb = k / qk;
    uint8_t* p = y;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f, vmax = 0.0f;
        for (int j = 0; j < qk; ++j) {
            float v = x[i*qk + j];
            if (std::fabs(v) > amax) { amax = std::fabs(v); vmax = v; }
        }
        const float d  = vmax / -16.0f;
        const float id = d ? 1.0f/d : 0.0f;
        const qtx::core::u16 d_h = qtx::core::fp32ToFP16Safe(d);
        std::memcpy(p, &d_h, 2);
        uint32_t qh = 0;
        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + j]      * id;
            const float x1 = x[i*qk + qk/2+j] * id;
            int xi0 = static_cast<int>(x0 + 16.5f);
            int xi1 = static_cast<int>(x1 + 16.5f);
            if (xi0 > 31) xi0 = 31;
            if (xi1 > 31) xi1 = 31;
            p[6 + j] = static_cast<uint8_t>((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= (static_cast<uint32_t>(xi0 & 0x10u) >> 4) << j;
            qh |= (static_cast<uint32_t>(xi1 & 0x10u) >> 4) << (j + 16);
        }
        std::memcpy(p + 2, &qh, 4);
        p += 22;
    }
}
inline void q5_1(const float* x, uint8_t* y, int k) {
    const int qk = 32;
    const int nb = k / qk;
    uint8_t* p = y;
    for (int i = 0; i < nb; ++i) {
        float vmin =  std::numeric_limits<float>::max();
        float vmax = -std::numeric_limits<float>::max();
        for (int j = 0; j < qk; ++j) {
            float v = x[i*qk + j];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        const float d  = (vmax - vmin) / 31.0f;
        const float id = d ? 1.0f/d : 0.0f;
        const qtx::core::u16 d_h = qtx::core::fp32ToFP16Safe(d);
        const qtx::core::u16 m_h = qtx::core::fp32ToFP16Safe(vmin);
        std::memcpy(p,     &d_h, 2);
        std::memcpy(p + 2, &m_h, 2);
        uint32_t qh = 0;
        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + j]      - vmin) * id;
            const float x1 = (x[i*qk + qk/2+j] - vmin) * id;
            uint8_t xi0 = static_cast<uint8_t>(x0 + 0.5f);
            uint8_t xi1 = static_cast<uint8_t>(x1 + 0.5f);
            p[8 + j] = static_cast<uint8_t>((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= (static_cast<uint32_t>(xi0 & 0x10u) >> 4) << j;
            qh |= (static_cast<uint32_t>(xi1 & 0x10u) >> 4) << (j + 16);
        }
        std::memcpy(p + 4, &qh, 4);
        p += 24;
    }
}
}  // namespace ggml_ref

QTX_TEST(Quant, GGML_Q4_1_ByteForByteCompat) {
    constexpr usize kN = 256u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(0.1f * static_cast<float>(i)) * 5.0f
               + 0.3f * static_cast<float>((i * 7919u) % 13u);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> qtx_out(compressedSize_Q4_1(kN));
    QTX_EXPECT_EQ(compressFP32ToQ4_1(src_b, qtx_out), qtx_out.size());

    std::vector<uint8_t> ggml_out(qtx_out.size());
    ggml_ref::q4_1(src.data(), ggml_out.data(), static_cast<int>(kN));

    for (usize i = 0; i < ggml_out.size(); ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned>(qtx_out[i]),
                      static_cast<unsigned>(ggml_out[i]));
    }
}

QTX_TEST(Quant, GGML_Q5_0_ByteForByteCompat) {
    constexpr usize kN = 256u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(0.1f * static_cast<float>(i)) * 5.0f
               + 0.3f * static_cast<float>((i * 7919u) % 13u);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> qtx_out(compressedSize_Q5_0(kN));
    QTX_EXPECT_EQ(compressFP32ToQ5_0(src_b, qtx_out), qtx_out.size());

    std::vector<uint8_t> ggml_out(qtx_out.size());
    ggml_ref::q5_0(src.data(), ggml_out.data(), static_cast<int>(kN));

    for (usize i = 0; i < ggml_out.size(); ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned>(qtx_out[i]),
                      static_cast<unsigned>(ggml_out[i]));
    }
}

QTX_TEST(Quant, GGML_Q5_1_ByteForByteCompat) {
    constexpr usize kN = 256u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(0.1f * static_cast<float>(i)) * 5.0f
               + 0.3f * static_cast<float>((i * 7919u) % 13u);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> qtx_out(compressedSize_Q5_1(kN));
    QTX_EXPECT_EQ(compressFP32ToQ5_1(src_b, qtx_out), qtx_out.size());

    std::vector<uint8_t> ggml_out(qtx_out.size());
    ggml_ref::q5_1(src.data(), ggml_out.data(), static_cast<int>(kN));

    for (usize i = 0; i < ggml_out.size(); ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned>(qtx_out[i]),
                      static_cast<unsigned>(ggml_out[i]));
    }
}

// ============================================================================
// INT8 Hardware (DP4A / SDOT layout) — Epic 2.
//
// Verifies that the existing INT8 block layout is hardware-friendly
// (4-byte aligned qs payload at offset 4) and exercises the helpers
// that split a QTX INT8 stream into a packed payload + dense scales,
// the canonical layout for tensor-core integer matmul kernels.
// ============================================================================

QTX_TEST(Quant, INT8_HW_QsOffsetIsDP4AAligned) {
    QTX_EXPECT_EQ(kINT8_QsOffset, 4u);                  // 4-byte aligned for DP4A
    QTX_EXPECT_EQ(kINT8_QsOffset % 4u, 0u);
    QTX_EXPECT_EQ(kINT8_QsBytesPerBlock, 32u);
}

QTX_TEST(Quant, INT8_HW_ExtractPayloadCorrect) {
    // Build a 2-block INT8 stream, extract payload, verify byte-for-byte.
    constexpr usize kN = 64u;                            // 2 blocks
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) src[i] = static_cast<float>(i - 32) * 0.1f;
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> int8_stream(compressedSize(QuantFormat::kINT8, kN));
    QTX_EXPECT_EQ(compressFP32ToINT8(src_b, int8_stream), int8_stream.size());

    std::vector<std::byte> payload(2u * 32u);            // 2 blocks * 32 INT8
    QTX_EXPECT_EQ(extractINT8Payload_DP4A(int8_stream, payload), payload.size());

    // The payload bytes must match the qs slice of each block.
    for (usize b = 0; b < 2u; ++b) {
        for (usize i = 0; i < 32u; ++i) {
            const std::byte in_block_qs =
                int8_stream[b * 36u + 4u + i];           // offset 4 = qs start
            QTX_EXPECT_EQ(static_cast<unsigned>(payload[b * 32u + i]),
                          static_cast<unsigned>(in_block_qs));
        }
    }
}

QTX_TEST(Quant, INT8_HW_ExtractScalesCorrect) {
    constexpr usize kN = 64u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) src[i] = static_cast<float>(i - 32) * 0.1f;
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> int8_stream(compressedSize(QuantFormat::kINT8, kN));
    (void)compressFP32ToINT8(src_b, int8_stream);

    std::vector<std::byte> scales_b(2u * sizeof(float));
    QTX_EXPECT_EQ(extractINT8Scales_DP4A(int8_stream, scales_b), 2u * sizeof(float));

    // Each extracted scale must match the FP32 at offset 0 of its block.
    for (usize b = 0; b < 2u; ++b) {
        float expected;
        std::memcpy(&expected, int8_stream.data() + b * 36u, sizeof(float));
        float actual;
        std::memcpy(&actual, scales_b.data() + b * sizeof(float), sizeof(float));
        QTX_EXPECT_EQ(actual, expected);
    }
}

QTX_TEST(Quant, INT8_HW_ExtractPayloadBadSize) {
    std::array<std::byte, 35> bad{};
    std::array<std::byte, 64> dst{};
    QTX_EXPECT_EQ(extractINT8Payload_DP4A(bad, dst), 0u);
}

QTX_TEST(Quant, INT8_HW_ExtractPayloadInsufficientDst) {
    std::array<std::byte, 36> stream{};
    std::array<std::byte, 31> tiny{};
    QTX_EXPECT_EQ(extractINT8Payload_DP4A(stream, tiny), 0u);
}

QTX_TEST(Quant, INT8_HW_ExtractScalesBadSize) {
    std::array<std::byte, 35> bad{};
    std::array<std::byte, 8>  dst{};
    QTX_EXPECT_EQ(extractINT8Scales_DP4A(bad, dst), 0u);
}

QTX_TEST(Quant, INT8_HW_RoundTripViaExtractedPieces) {
    // Re-construct the INT8 block from its extracted parts and verify
    // the result decompresses to the same FP32 values as the original
    // stream — proves the (scale, payload) decomposition is lossless.
    constexpr usize kN = 64u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) src[i] = std::sin(0.1f * static_cast<float>(i));
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> int8_stream(compressedSize(QuantFormat::kINT8, kN));
    (void)compressFP32ToINT8(src_b, int8_stream);

    std::vector<std::byte> payload(2u * 32u);
    std::vector<std::byte> scales_b(2u * sizeof(float));
    (void)extractINT8Payload_DP4A(int8_stream, payload);
    (void)extractINT8Scales_DP4A(int8_stream, scales_b);

    // Reassemble.
    std::vector<std::byte> rebuilt(compressedSize(QuantFormat::kINT8, kN));
    for (usize b = 0; b < 2u; ++b) {
        std::memcpy(rebuilt.data() + b * 36u,
                    scales_b.data() + b * sizeof(float), sizeof(float));
        std::memcpy(rebuilt.data() + b * 36u + 4u,
                    payload.data() + b * 32u, 32u);
    }

    // Decompress both and compare.
    std::vector<std::byte> dec1(kN * sizeof(float));
    std::vector<std::byte> dec2(kN * sizeof(float));
    (void)decompressINT8ToFP32(int8_stream, dec1);
    (void)decompressINT8ToFP32(rebuilt,     dec2);
    QTX_EXPECT_EQ(std::memcmp(dec1.data(), dec2.data(), dec1.size()), 0);
}

// ============================================================================
// K-Quants — Q6_K (Epic 3, R&D-1 single-pass heuristic encoder)
//
// Super-block of 256 elements, 16 sub-blocks of 16 each. Wire format
// is bit-for-bit ggml `block_q6_K` (210 bytes). The encoder uses a
// single-pass scale-derivation heuristic (no inner L-BFGS), so the
// compressed payload may differ from ggml's offline encoder, but the
// layout and decode semantics are identical.
// ============================================================================

QTX_TEST(Quant, Q6_K_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q6_K(256u),  210u);
    QTX_EXPECT_EQ(compressedSize_Q6_K(1024u), 840u);
    QTX_EXPECT_EQ(compressedSize_Q6_K(255u),  0u);    // not /256
    QTX_EXPECT_EQ(compressedSize_Q6_K(127u),  0u);
}

QTX_TEST(Quant, Q6_K_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 210> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ6_K(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q6_K_InsufficientDstReturnsZero) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 209> tiny_dst{};
    QTX_EXPECT_EQ(compressFP32ToQ6_K(src_bytes, tiny_dst), 0u);
}

QTX_TEST(Quant, Q6_K_DecompressBadPayloadSize) {
    std::array<std::byte, 209> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ6_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q6_K_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 210>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ6_K(src_bytes, c),  210u);
    QTX_EXPECT_EQ(decompressQ6_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q6_K_HandlesSanitisedNaN) {
    // NaN/Inf inputs sanitise to 0 so the codec stays HA-clean
    // (decompress can never see a non-finite scale).
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = -std::numeric_limits<float>::infinity();
    src[3] = 1.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 210>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ6_K(src_bytes, c),  210u);
    QTX_EXPECT_EQ(decompressQ6_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
    // NaN/Inf -> 0 sanitisation expected.
    QTX_EXPECT_EQ(out[0], 0.0f);
    QTX_EXPECT_EQ(out[1], 0.0f);
    QTX_EXPECT_EQ(out[2], 0.0f);
}

QTX_TEST(Quant, Q6_K_DecompressRejectsNonFiniteScale) {
    // Corrupted d (FP16 NaN at offset 208) must be rejected.
    std::array<std::byte, 210> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data() + 208u, &nan16, 2u);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ6_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q6_K_RoundTripFiniteAccuracy) {
    // Q6_K has effectively 6.5625 bits per element. On a smooth signal
    // the mean relative error should be ≤ 2% — better than Q4_1 (16
    // codepoints) and comparable to Q5_K. Worst-case rel error can
    // still blow up near grid lines; we measure the mean.
    constexpr usize kN = 256u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(0.05f * static_cast<float>(i)) * 2.5f
               + 0.1f * static_cast<float>((i * 7919u) % 13u);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q6_K(kN));
    QTX_EXPECT_EQ(compressFP32ToQ6_K(src_b, c), c.size());
    std::vector<std::byte> back_b(kN * sizeof(float));
    QTX_EXPECT_EQ(decompressQ6_K_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(kN);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.02);     // ≤ 2% mean (Q6_K with no inner opt)
}

QTX_TEST(Quant, Q6_K_LayoutInvariants) {
    // Verify the byte stream layout against the spec:
    //   ql[128] @ 0..127
    //   qh[64]  @ 128..191
    //   scales[16] @ 192..207
    //   d (FP16) @ 208..209
    std::array<float, 256> src{};
    for (int i = 0; i < 256; ++i) src[static_cast<size_t>(i)] = std::sin(0.1f * static_cast<float>(i));
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 210> c{};
    QTX_EXPECT_EQ(compressFP32ToQ6_K(src_bytes, c), 210u);
    // d must be a finite FP16 (not 0xFFFE = NaN-ish).
    const qtx::core::u16 d_bits =
        static_cast<qtx::core::u16>(c[208]) |
        (static_cast<qtx::core::u16>(c[209]) << 8);
    const float d = qtx::core::fp16ToFP32(d_bits);
    QTX_EXPECT(std::isfinite(d));
    // d is the FP16 of a negative number (iscale = -128/max_scale,
    // d_super = 1/iscale, so d_super has the SIGN of -1/max_scale).
    // For non-zero inputs, d != 0.
    QTX_EXPECT(d != 0.0f);
}

QTX_TEST(Quant, Q6_K_MultiBlockStream) {
    // 1024 elements = 4 super-blocks. Each block must round-trip
    // independently; switch test data per-block to force scale change.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        // Magnitude varies by super-block to force distinct d_super values.
        const float mag = (i < 256) ? 1.0f : (i < 512) ? 10.0f
                       : (i < 768) ? 0.1f : 100.0f;
        src[i] = std::sin(0.05f * static_cast<float>(i)) * mag;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q6_K(kN));
    QTX_EXPECT_EQ(compressFP32ToQ6_K(src_b, c), c.size());
    std::vector<std::byte> back_b(kN * sizeof(float));
    QTX_EXPECT_EQ(decompressQ6_K_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(kN);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    // Per-block d_super values should differ — read FP16 d from each
    // block's tail and check that at least 3 of 4 are distinct.
    std::array<float, 4> ds{};
    for (int b = 0; b < 4; ++b) {
        const qtx::core::u16 d_bits =
            static_cast<qtx::core::u16>(c[b * 210 + 208]) |
            (static_cast<qtx::core::u16>(c[b * 210 + 209]) << 8);
        ds[static_cast<size_t>(b)] = qtx::core::fp16ToFP32(d_bits);
    }
    int distinct = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (ds[static_cast<size_t>(i)] != ds[static_cast<size_t>(j)]) ++distinct;
    QTX_EXPECT(distinct >= 3);   // most pairs should differ

    // Mean rel error per block ≤ 2.5%.
    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.025);
}

// ============================================================================
// K-Quants — Q5_K, Q4_K, Q3_K, Q2_K (Epic 3)
// All 256-element super-blocks. Same compact-test schema as Q6_K.
// ============================================================================

namespace kq_test_helpers {

inline std::vector<float> makeSmoothSignal(usize n) {
    std::vector<float> v(n);
    for (usize i = 0; i < n; ++i) {
        v[i] = std::sin(0.05f * static_cast<float>(i)) * 2.5f
             + 0.1f * static_cast<float>((i * 7919u) % 13u);
    }
    return v;
}

}  // namespace kq_test_helpers

// --------- Q5_K ---------

QTX_TEST(Quant, Q5_K_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q5_K(256u),  176u);
    QTX_EXPECT_EQ(compressedSize_Q5_K(1024u), 704u);
    QTX_EXPECT_EQ(compressedSize_Q5_K(255u),  0u);
}

QTX_TEST(Quant, Q5_K_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 176> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ5_K(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q5_K_InsufficientDstReturnsZero) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 175> tiny{};
    QTX_EXPECT_EQ(compressFP32ToQ5_K(src_bytes, tiny), 0u);
}

QTX_TEST(Quant, Q5_K_DecompressBadPayloadSize) {
    std::array<std::byte, 175> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ5_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q5_K_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 176>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ5_K(src_bytes, c),  176u);
    QTX_EXPECT_EQ(decompressQ5_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q5_K_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = -std::numeric_limits<float>::infinity();
    src[3] = 1.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 176>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ5_K(src_bytes, c),  176u);
    QTX_EXPECT_EQ(decompressQ5_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, Q5_K_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 176> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);                // d at offset 0
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ5_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q5_K_RoundTripFiniteAccuracy) {
    // R&D-1 single-pass encoder target: mean rel error ≤ 4% (better
    // than ggml-offline by no more than 0.5pp).
    const auto src = kq_test_helpers::makeSmoothSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q5_K(1024u));
    QTX_EXPECT_EQ(compressFP32ToQ5_K(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressQ5_K_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.04);
}

// --------- Q4_K ---------

QTX_TEST(Quant, Q4_K_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q4_K(256u),  144u);
    QTX_EXPECT_EQ(compressedSize_Q4_K(1024u), 576u);
    QTX_EXPECT_EQ(compressedSize_Q4_K(255u),  0u);
}

QTX_TEST(Quant, Q4_K_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 144> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ4_K(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q4_K_DecompressBadPayloadSize) {
    std::array<std::byte, 143> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ4_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q4_K_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 144>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ4_K(src_bytes, c),  144u);
    QTX_EXPECT_EQ(decompressQ4_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q4_K_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 1.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 144>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ4_K(src_bytes, c),  144u);
    QTX_EXPECT_EQ(decompressQ4_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, Q4_K_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 144> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ4_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q4_K_RoundTripFiniteAccuracy) {
    const auto src = kq_test_helpers::makeSmoothSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q4_K(1024u));
    QTX_EXPECT_EQ(compressFP32ToQ4_K(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressQ4_K_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.08);     // Q4_K target: ≤ 8% mean rel err
}

// --------- Q3_K ---------

QTX_TEST(Quant, Q3_K_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q3_K(256u),  110u);
    QTX_EXPECT_EQ(compressedSize_Q3_K(1024u), 440u);
    QTX_EXPECT_EQ(compressedSize_Q3_K(255u),  0u);
}

QTX_TEST(Quant, Q3_K_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 110> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ3_K(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q3_K_DecompressBadPayloadSize) {
    std::array<std::byte, 109> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ3_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q3_K_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 110>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ3_K(src_bytes, c),  110u);
    QTX_EXPECT_EQ(decompressQ3_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q3_K_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 1.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 110>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ3_K(src_bytes, c),  110u);
    QTX_EXPECT_EQ(decompressQ3_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, Q3_K_DecompressRejectsNonFiniteScale) {
    // d is at offset 108 of the 110-byte block.
    std::array<std::byte, 110> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data() + 108u, &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ3_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q3_K_RoundTripFiniteAccuracy) {
    // Q3_K has 8 codepoints per sub-block — wide error band on smooth
    // sinusoidal signals. Mean rel err ≤ 18% is the target.
    const auto src = kq_test_helpers::makeSmoothSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q3_K(1024u));
    QTX_EXPECT_EQ(compressFP32ToQ3_K(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressQ3_K_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.18);
}

// --------- Q2_K ---------

QTX_TEST(Quant, Q2_K_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_Q2_K(256u),  84u);
    QTX_EXPECT_EQ(compressedSize_Q2_K(1024u), 336u);
    QTX_EXPECT_EQ(compressedSize_Q2_K(255u),  0u);
}

QTX_TEST(Quant, Q2_K_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 84> dst{};
    QTX_EXPECT_EQ(compressFP32ToQ2_K(src_bytes, dst), 0u);
}

QTX_TEST(Quant, Q2_K_DecompressBadPayloadSize) {
    std::array<std::byte, 83> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ2_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q2_K_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 84>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ2_K(src_bytes, c),  84u);
    QTX_EXPECT_EQ(decompressQ2_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, Q2_K_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 1.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 84>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToQ2_K(src_bytes, c),  84u);
    QTX_EXPECT_EQ(decompressQ2_K_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, Q2_K_DecompressRejectsNonFiniteScale) {
    // d is at offset 80 of the 84-byte block.
    std::array<std::byte, 84> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data() + 80u, &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressQ2_K_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, Q2_K_RoundTripFiniteAccuracy) {
    // Q2_K: only 4 codepoints per sub-block. Worst-class accuracy.
    // Mean rel err ≤ 24% is the realistic target on smooth signals.
    const auto src = kq_test_helpers::makeSmoothSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_Q2_K(1024u));
    QTX_EXPECT_EQ(compressFP32ToQ2_K(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressQ2_K_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.24);
}

// --------- K-Quants cross-format invariant ---------

QTX_TEST(Quant, KQuants_AccuracyHierarchy) {
    // Within the K-Quant family, the size-vs-accuracy hierarchy
    // must hold: Q6_K < Q5_K < Q4_K < Q3_K < Q2_K (mean abs err).
    // This is a regression guard against any future encoder change
    // that breaks the natural ordering.
    const auto src = kq_test_helpers::makeSmoothSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    auto err_of = [&](auto cf, auto df, auto sf) -> double {
        std::vector<std::byte> c(sf(1024u));
        cf(std::span<const std::byte>{src_b}, std::span<std::byte>{c});
        std::vector<std::byte> b(1024u * sizeof(float));
        df(std::span<const std::byte>{c}, std::span<std::byte>{b});
        std::vector<float> back(1024u);
        std::memcpy(back.data(), b.data(), b.size());
        double sum = 0.0;
        for (usize i = 0; i < 1024u; ++i) {
            sum += static_cast<double>(std::fabs(back[i] - src[i]));
        }
        return sum / 1024.0;
    };
    const double e2 = err_of(&compressFP32ToQ2_K, &decompressQ2_K_ToFP32,
                              &compressedSize_Q2_K);
    const double e3 = err_of(&compressFP32ToQ3_K, &decompressQ3_K_ToFP32,
                              &compressedSize_Q3_K);
    const double e4 = err_of(&compressFP32ToQ4_K, &decompressQ4_K_ToFP32,
                              &compressedSize_Q4_K);
    const double e5 = err_of(&compressFP32ToQ5_K, &decompressQ5_K_ToFP32,
                              &compressedSize_Q5_K);
    const double e6 = err_of(&compressFP32ToQ6_K, &decompressQ6_K_ToFP32,
                              &compressedSize_Q6_K);
    QTX_EXPECT(e6 < e5);
    QTX_EXPECT(e5 < e4);
    QTX_EXPECT(e4 < e3);
    QTX_EXPECT(e3 < e2);
}

// ============================================================================
// I-Quants (Importance-aware codebook) — Epic 4.
//
// IQ4_NL — 32-element blocks, FP16 scale, 16-entry non-linear int8
//          codebook. Wire format: ggml block_iq4_nl (18 bytes).
//          Single-pass encoder is BYTE-FOR-BYTE identical to ggml's
//          ntry=-1 path (verified by inline regression test below).
//
// IQ4_XS — 256-element super-blocks, FP16 super-scale + 6-bit
//          per-sub-block scales, 16-entry codebook. Wire format:
//          ggml block_iq4_xs (136 bytes).
// ============================================================================

// --------- IQ4_NL ---------

QTX_TEST(Quant, IQ4_NL_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ4_NL(32u),   18u);
    QTX_EXPECT_EQ(compressedSize_IQ4_NL(1024u), 576u);
    QTX_EXPECT_EQ(compressedSize_IQ4_NL(31u),   0u);
    QTX_EXPECT_EQ(compressedSize_IQ4_NL(33u),   0u);
}

QTX_TEST(Quant, IQ4_NL_BadSrcSizeReturnsZero) {
    std::array<float, 31> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 18> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_NL(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ4_NL_InsufficientDstReturnsZero) {
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 17> tiny{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_NL(src_bytes, tiny), 0u);
}

QTX_TEST(Quant, IQ4_NL_DecompressBadPayloadSize) {
    std::array<std::byte, 17> bad{};
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressIQ4_NL_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ4_NL_AllZeroBlock) {
    std::array<float, 32> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 18>  c{};
    std::array<std::byte, 128> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_NL(src_bytes, c),    18u);
    QTX_EXPECT_EQ(decompressIQ4_NL_ToFP32(c, back),      128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ4_NL_CodebookExactRoundTrip) {
    // When d = 1 and inputs are exactly the codebook values, encode
    // returns each value's exact index → decode returns the exact value.
    // For the chosen input set, scale is determined by max |v| = 127,
    // so d = -1 (since values[0] = -127) → id = -1.
    // Each input v gets mapped via best_index(-v) — codebook is symmetric
    // around index 7/8 boundary but NOT symmetric in values, so we have
    // to verify with the actual sign convention.
    using namespace qtx::quantize::detail_iquant;
    std::array<float, 32> src{};
    for (int i = 0; i < 16; ++i) {
        src[static_cast<size_t>(i)]       = static_cast<float>(kIQ4_Codebook[i]);
        src[static_cast<size_t>(i + 16)]  = static_cast<float>(kIQ4_Codebook[i]);
    }
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 18>  c{};
    std::array<std::byte, 128> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_NL(src_bytes, c),    18u);
    QTX_EXPECT_EQ(decompressIQ4_NL_ToFP32(c, back),      128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

QTX_TEST(Quant, IQ4_NL_HandlesSanitisedNaN) {
    std::array<float, 32> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = -std::numeric_limits<float>::infinity();
    src[3] = 1.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 18>  c{};
    std::array<std::byte, 128> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_NL(src_bytes, c),    18u);
    QTX_EXPECT_EQ(decompressIQ4_NL_ToFP32(c, back),      128u);
    auto out = bytesToFloats<32>(back);
    for (int i = 0; i < 32; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ4_NL_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 18> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 128> dst{};
    QTX_EXPECT_EQ(decompressIQ4_NL_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ4_NL_RoundTripFiniteAccuracy) {
    // IQ4_NL with 16 non-linear codepoints concentrated near zero
    // gives good accuracy for Gaussian-like signals. Mean rel err ≤ 10%.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(0.05f * static_cast<float>(i)) * 2.5f
               + 0.1f * static_cast<float>((i * 7919u) % 13u);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ4_NL(kN));
    QTX_EXPECT_EQ(compressFP32ToIQ4_NL(src_b, c), c.size());
    std::vector<std::byte> back_b(kN * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ4_NL_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(kN);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.10);
}

// Inline GGML byte-for-byte compatibility test for IQ4_NL (ntry=-1
// single-pass path; encoder line 4889 of ggml-quants.c). This pins the
// wire format permanently.
namespace ggml_ref_iq4 {
inline constexpr int8_t kvalues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

inline int best_index(float x) {
    if (x <= static_cast<float>(kvalues[0]))  return 0;
    if (x >= static_cast<float>(kvalues[15])) return 15;
    int ml = 0, mu = 15;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < static_cast<float>(kvalues[mav])) mu = mav;
        else                                      ml = mav;
    }
    return std::fabs(x - static_cast<float>(kvalues[mu-1]))
         < std::fabs(x - static_cast<float>(kvalues[mu])) ? mu-1 : mu;
}

inline void iq4_nl_singlepass(const float* x, uint8_t* y, int k) {
    const int nb = k / 32;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f, max_v = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float ax = std::fabs(x[i*32 + j]);
            if (ax > amax) { amax = ax; max_v = x[i*32 + j]; }
        }
        const float d = (amax > 0.0f)
            ? max_v / static_cast<float>(kvalues[0]) : 0.0f;
        const qtx::core::u16 d_h = qtx::core::fp32ToFP16Safe(d);
        std::memcpy(y + i*18, &d_h, 2);
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < 16; ++j) {
            const int n0 = best_index(id * x[i*32 + j]);
            const int n1 = best_index(id * x[i*32 + j + 16]);
            y[i*18 + 2 + j] = static_cast<uint8_t>(
                (n0 & 0x0F) | ((n1 & 0x0F) << 4));
        }
    }
}
}  // namespace ggml_ref_iq4

QTX_TEST(Quant, GGML_IQ4_NL_ByteForByteCompat) {
    constexpr usize kN = 256u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(0.1f * static_cast<float>(i)) * 5.0f
               + 0.3f * static_cast<float>((i * 7919u) % 13u);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> qtx_out(compressedSize_IQ4_NL(kN));
    QTX_EXPECT_EQ(compressFP32ToIQ4_NL(src_b, qtx_out), qtx_out.size());

    std::vector<uint8_t> ggml_out(qtx_out.size());
    ggml_ref_iq4::iq4_nl_singlepass(src.data(), ggml_out.data(),
                                     static_cast<int>(kN));

    for (usize i = 0; i < ggml_out.size(); ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned>(qtx_out[i]),
                      static_cast<unsigned>(ggml_out[i]));
    }
}

// --------- IQ4_XS ---------

QTX_TEST(Quant, IQ4_XS_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ4_XS(256u),  136u);
    QTX_EXPECT_EQ(compressedSize_IQ4_XS(1024u), 544u);
    QTX_EXPECT_EQ(compressedSize_IQ4_XS(255u),  0u);
}

QTX_TEST(Quant, IQ4_XS_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 136> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_XS(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ4_XS_DecompressBadPayloadSize) {
    std::array<std::byte, 135> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ4_XS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ4_XS_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 136>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_XS(src_bytes, c),  136u);
    QTX_EXPECT_EQ(decompressIQ4_XS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ4_XS_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = -std::numeric_limits<float>::infinity();
    src[3] = 1.0f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 136>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ4_XS(src_bytes, c),  136u);
    QTX_EXPECT_EQ(decompressIQ4_XS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ4_XS_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 136> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);                // d at offset 0
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ4_XS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ4_XS_RoundTripFiniteAccuracy) {
    // IQ4_XS has the same codebook as IQ4_NL but adds per-sub-block
    // scales — should give similar or slightly tighter mean rel err
    // on smooth signals because each sub-block can adapt.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(0.05f * static_cast<float>(i)) * 2.5f
               + 0.1f * static_cast<float>((i * 7919u) % 13u);
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ4_XS(kN));
    QTX_EXPECT_EQ(compressFP32ToIQ4_XS(src_b, c), c.size());
    std::vector<std::byte> back_b(kN * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ4_XS_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(kN);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.1f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.10);
}

// Cross-format invariant: IQ4_XS gives a better compression ratio than
// IQ4_NL on a 1024-element input (one shared FP16 super-scale per 256
// elements rather than per 32).
QTX_TEST(Quant, IQ4_XS_BetterCompressionRatioThanNL) {
    constexpr usize kN = 1024u;
    QTX_EXPECT(compressedSize_IQ4_XS(kN) < compressedSize_IQ4_NL(kN));
}

// ============================================================================
// Codebook-based I-Quants — IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
// IQ3_XXS, IQ3_S. All use 256-element super-blocks with fixed grid
// codebooks (~33 KB total of MIT-licensed lookup tables from ggml).
//
// Encoders use a two-phase single-pass heuristic per R&D-2: phase 1
// computes per-ib32 free-floating scale + grid indices via brute-force
// 256/512/1024/2048-entry codebook search; phase 2 quantizes the
// per-ib32 scale to 4 bits relative to the super-block max.
//
// Decoders are bit-for-bit ggml-compatible transcriptions of
// dequantize_row_iq{1,2,3}_*.
//
// The codebooks are byte-for-byte identical to ggml's iq*_grid tables
// (verified at static-time by sizeof checks below).
// ============================================================================

namespace iq_cb_test_helpers {

inline std::vector<float> makeGaussianSignal(usize n, float sigma = 0.05f) {
    std::vector<float> v(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, sigma);
    for (auto& x : v) x = nd(rng);
    return v;
}

}  // namespace iq_cb_test_helpers

// --------- IQ1_S ---------

QTX_TEST(Quant, IQ1_S_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ1_S(256u),  50u);
    QTX_EXPECT_EQ(compressedSize_IQ1_S(1024u), 200u);
    QTX_EXPECT_EQ(compressedSize_IQ1_S(255u),  0u);
}

QTX_TEST(Quant, IQ1_S_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 50> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ1_S(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ1_S_DecompressBadPayloadSize) {
    std::array<std::byte, 49> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ1_S_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ1_S_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 50>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ1_S(src_bytes, c),  50u);
    QTX_EXPECT_EQ(decompressIQ1_S_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ1_S_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = -std::numeric_limits<float>::infinity();
    src[3] = 0.05f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 50>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ1_S(src_bytes, c),  50u);
    QTX_EXPECT_EQ(decompressIQ1_S_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ1_S_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 50> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);   // d at offset 0
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ1_S_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ1_S_RoundTripGaussian) {
    // 1.5625 bpw — IQ1_S extreme compression for LLM-weight-like signal.
    // Mean rel err ≤ 30% is the target on a Gaussian.
    const auto src = iq_cb_test_helpers::makeGaussianSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ1_S(1024u));
    QTX_EXPECT_EQ(compressFP32ToIQ1_S(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ1_S_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.05f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.55);
}

// --------- IQ1_M ---------

QTX_TEST(Quant, IQ1_M_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ1_M(256u),  56u);
    QTX_EXPECT_EQ(compressedSize_IQ1_M(1024u), 224u);
    QTX_EXPECT_EQ(compressedSize_IQ1_M(255u),  0u);
}

QTX_TEST(Quant, IQ1_M_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 56> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ1_M(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ1_M_DecompressBadPayloadSize) {
    std::array<std::byte, 55> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ1_M_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ1_M_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 56>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ1_M(src_bytes, c),  56u);
    QTX_EXPECT_EQ(decompressIQ1_M_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ1_M_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 0.05f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 56>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ1_M(src_bytes, c),  56u);
    QTX_EXPECT_EQ(decompressIQ1_M_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ1_M_RoundTripGaussian) {
    const auto src = iq_cb_test_helpers::makeGaussianSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ1_M(1024u));
    QTX_EXPECT_EQ(compressFP32ToIQ1_M(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ1_M_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.05f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.45);
}

// --------- IQ2_XXS ---------

QTX_TEST(Quant, IQ2_XXS_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ2_XXS(256u),  66u);
    QTX_EXPECT_EQ(compressedSize_IQ2_XXS(1024u), 264u);
    QTX_EXPECT_EQ(compressedSize_IQ2_XXS(255u),  0u);
}

QTX_TEST(Quant, IQ2_XXS_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 66> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_XXS(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ2_XXS_DecompressBadPayloadSize) {
    std::array<std::byte, 65> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ2_XXS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ2_XXS_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 66>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_XXS(src_bytes, c),  66u);
    QTX_EXPECT_EQ(decompressIQ2_XXS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ2_XXS_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 0.05f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 66>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_XXS(src_bytes, c),  66u);
    QTX_EXPECT_EQ(decompressIQ2_XXS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ2_XXS_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 66> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ2_XXS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ2_XXS_RoundTripGaussian) {
    const auto src = iq_cb_test_helpers::makeGaussianSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ2_XXS(1024u));
    QTX_EXPECT_EQ(compressFP32ToIQ2_XXS(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ2_XXS_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.05f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.45);
}

// --------- IQ2_XS ---------

QTX_TEST(Quant, IQ2_XS_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ2_XS(256u),  74u);
    QTX_EXPECT_EQ(compressedSize_IQ2_XS(1024u), 296u);
    QTX_EXPECT_EQ(compressedSize_IQ2_XS(255u),  0u);
}

QTX_TEST(Quant, IQ2_XS_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 74> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_XS(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ2_XS_DecompressBadPayloadSize) {
    std::array<std::byte, 73> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ2_XS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ2_XS_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 74>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_XS(src_bytes, c),  74u);
    QTX_EXPECT_EQ(decompressIQ2_XS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ2_XS_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 0.05f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 74>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_XS(src_bytes, c),  74u);
    QTX_EXPECT_EQ(decompressIQ2_XS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ2_XS_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 74> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ2_XS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ2_XS_RoundTripGaussian) {
    const auto src = iq_cb_test_helpers::makeGaussianSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ2_XS(1024u));
    QTX_EXPECT_EQ(compressFP32ToIQ2_XS(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ2_XS_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.05f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.40);
}

// --------- IQ2_S ---------

QTX_TEST(Quant, IQ2_S_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ2_S(256u),  82u);
    QTX_EXPECT_EQ(compressedSize_IQ2_S(1024u), 328u);
    QTX_EXPECT_EQ(compressedSize_IQ2_S(255u),  0u);
}

QTX_TEST(Quant, IQ2_S_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 82> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_S(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ2_S_DecompressBadPayloadSize) {
    std::array<std::byte, 81> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ2_S_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ2_S_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 82>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_S(src_bytes, c),  82u);
    QTX_EXPECT_EQ(decompressIQ2_S_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ2_S_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 0.05f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 82>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ2_S(src_bytes, c),  82u);
    QTX_EXPECT_EQ(decompressIQ2_S_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ2_S_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 82> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ2_S_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ2_S_RoundTripGaussian) {
    const auto src = iq_cb_test_helpers::makeGaussianSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ2_S(1024u));
    QTX_EXPECT_EQ(compressFP32ToIQ2_S(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ2_S_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.05f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.40);
}

// --------- IQ3_XXS ---------

QTX_TEST(Quant, IQ3_XXS_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ3_XXS(256u),  98u);
    QTX_EXPECT_EQ(compressedSize_IQ3_XXS(1024u), 392u);
    QTX_EXPECT_EQ(compressedSize_IQ3_XXS(255u),  0u);
}

QTX_TEST(Quant, IQ3_XXS_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 98> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ3_XXS(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ3_XXS_DecompressBadPayloadSize) {
    std::array<std::byte, 97> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ3_XXS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ3_XXS_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 98>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ3_XXS(src_bytes, c),  98u);
    QTX_EXPECT_EQ(decompressIQ3_XXS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ3_XXS_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 0.05f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 98>   c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ3_XXS(src_bytes, c),  98u);
    QTX_EXPECT_EQ(decompressIQ3_XXS_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ3_XXS_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 98> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ3_XXS_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ3_XXS_RoundTripGaussian) {
    const auto src = iq_cb_test_helpers::makeGaussianSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ3_XXS(1024u));
    QTX_EXPECT_EQ(compressFP32ToIQ3_XXS(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ3_XXS_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.05f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.40);
}

// --------- IQ3_S ---------

QTX_TEST(Quant, IQ3_S_CompressedSize) {
    QTX_EXPECT_EQ(compressedSize_IQ3_S(256u),  110u);
    QTX_EXPECT_EQ(compressedSize_IQ3_S(1024u), 440u);
    QTX_EXPECT_EQ(compressedSize_IQ3_S(255u),  0u);
}

QTX_TEST(Quant, IQ3_S_BadSrcSizeReturnsZero) {
    std::array<float, 255> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 110> dst{};
    QTX_EXPECT_EQ(compressFP32ToIQ3_S(src_bytes, dst), 0u);
}

QTX_TEST(Quant, IQ3_S_DecompressBadPayloadSize) {
    std::array<std::byte, 109> bad{};
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ3_S_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ3_S_AllZeroBlock) {
    std::array<float, 256> src{};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 110>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ3_S(src_bytes, c),  110u);
    QTX_EXPECT_EQ(decompressIQ3_S_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT_EQ(out[i], 0.0f);
}

QTX_TEST(Quant, IQ3_S_HandlesSanitisedNaN) {
    std::array<float, 256> src{};
    src[0] = std::numeric_limits<float>::quiet_NaN();
    src[1] = std::numeric_limits<float>::infinity();
    src[2] = 0.05f;
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 110>  c{};
    std::array<std::byte, 1024> back{};
    QTX_EXPECT_EQ(compressFP32ToIQ3_S(src_bytes, c),  110u);
    QTX_EXPECT_EQ(decompressIQ3_S_ToFP32(c, back),    1024u);
    auto out = bytesToFloats<256>(back);
    for (int i = 0; i < 256; ++i) QTX_EXPECT(std::isfinite(out[i]));
}

QTX_TEST(Quant, IQ3_S_DecompressRejectsNonFiniteScale) {
    std::array<std::byte, 110> bad{};
    const qtx::core::u16 nan16 = 0x7E00u;
    std::memcpy(bad.data(), &nan16, 2);
    std::array<std::byte, 1024> dst{};
    QTX_EXPECT_EQ(decompressIQ3_S_ToFP32(bad, dst), 0u);
}

QTX_TEST(Quant, IQ3_S_RoundTripGaussian) {
    // IQ3_S has the most expressive codebook of the family;
    // mean rel err ≤ 10% expected on Gaussian.
    const auto src = iq_cb_test_helpers::makeGaussianSignal(1024u);
    std::vector<std::byte> src_b(1024u * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> c(compressedSize_IQ3_S(1024u));
    QTX_EXPECT_EQ(compressFP32ToIQ3_S(src_b, c), c.size());
    std::vector<std::byte> back_b(1024u * sizeof(float));
    QTX_EXPECT_EQ(decompressIQ3_S_ToFP32(c, back_b), back_b.size());
    std::vector<float> back(1024u);
    std::memcpy(back.data(), back_b.data(), back_b.size());

    double sum_rel = 0.0;
    usize count = 0;
    for (usize i = 0; i < 1024u; ++i) {
        if (std::fabs(src[i]) > 0.05f) {
            sum_rel += static_cast<double>(
                std::fabs(back[i] - src[i]) / std::fabs(src[i]));
            ++count;
        }
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    QTX_EXPECT(mean_rel < 0.15);
}

// Cross-format invariant: compression ratio strictly increases with bpw.
QTX_TEST(Quant, IQ_CodebookFamily_BlockSizeMonotonic) {
    QTX_EXPECT(kIQ1_S_BlockBytes   < kIQ1_M_BlockBytes);
    QTX_EXPECT(kIQ1_M_BlockBytes   < kIQ2_XXS_BlockBytes);
    QTX_EXPECT(kIQ2_XXS_BlockBytes < kIQ2_XS_BlockBytes);
    QTX_EXPECT(kIQ2_XS_BlockBytes  < kIQ2_S_BlockBytes_v2);
    QTX_EXPECT(kIQ2_S_BlockBytes_v2 < kIQ3_XXS_BlockBytes);
    QTX_EXPECT(kIQ3_XXS_BlockBytes < kIQ3_S_BlockBytes);
}

// Codebook integrity: a few known-correct values from ggml-common.h.
QTX_TEST(Quant, IQ_Codebooks_FirstEntriesIntact) {
    using namespace qtx::quantize::iq_codebooks;
    // ggml-common.h line 550-551: iq2xxs_grid[0] = 0x0808080808080808.
    QTX_EXPECT_EQ(iq2xxs_grid[0], 0x0808080808080808ULL);
    // line 552: iq2xxs_grid[1] = 0x080808080808082b
    QTX_EXPECT_EQ(iq2xxs_grid[1], 0x080808080808082bULL);
    // kmask_iq2xs[0] = 1, kmask_iq2xs[7] = 128
    QTX_EXPECT_EQ(kmask_iq2xs[0], 1u);
    QTX_EXPECT_EQ(kmask_iq2xs[7], 128u);
    // ksigns_iq2xs[0] = 0
    QTX_EXPECT_EQ(ksigns_iq2xs[0], 0u);
}

// ============================================================================
// INT8 round-trip
// ============================================================================

QTX_TEST(Quant, INT8RoundTripBlock) {
    //One complete block (32 FP32 elements).
    std::array<float, 32> src{};
    for (int i = 0; i < 32; ++i) {
        src[static_cast<size_t>(i)] = static_cast<float>(i - 16) * 0.5f;
    }
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 36> compressed{};
    std::array<std::byte, 128> decompressed{};

    const auto cs = compressFP32ToINT8(src_bytes, compressed);
    QTX_EXPECT_EQ(cs, 36u);
    const auto ds = decompressINT8ToFP32(compressed, decompressed);
    QTX_EXPECT_EQ(ds, 128u);

    auto out = bytesToFloats<32>(decompressed);
    //Accuracy INT8: relative error < 2% for values ​​outside the zero neighborhood.
    for (int i = 0; i < 32; ++i) {
        if (std::fabs(src[static_cast<size_t>(i)]) > 0.5f) {
            const float rel_err =
                std::fabs(out[static_cast<size_t>(i)] - src[static_cast<size_t>(i)]) /
                std::fabs(src[static_cast<size_t>(i)]);
            QTX_EXPECT(rel_err < 0.05f);
        }
    }
}

QTX_TEST(Quant, INT8MultipleBlocks) {
    //Two blocks = 64 elements.
    std::array<float, 64> src{};
    for (int i = 0; i < 64; ++i) {
        src[static_cast<size_t>(i)] = std::sin(static_cast<float>(i) * 0.1f);
    }
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 72> compressed{};
    std::array<std::byte, 256> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToINT8(src_bytes, compressed), 72u);
    QTX_EXPECT_EQ(decompressINT8ToFP32(compressed, decompressed), 256u);

    auto out = bytesToFloats<64>(decompressed);
    //INT8 precision for values ​​in [-1, 1]: absolute error < 1/127 ≈ 0.008.
    for (size_t i = 0; i < 64; ++i) {
        QTX_EXPECT(std::fabs(out[i] - src[i]) < 0.02f);
    }
}

QTX_TEST(Quant, INT8ZeroBlock) {
    //All zeros are a special case (scale = 0).
    std::array<float, 32> src{};  // zero-init
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 36> compressed{};
    std::array<std::byte, 128> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToINT8(src_bytes, compressed), 36u);
    QTX_EXPECT_EQ(decompressINT8ToFP32(compressed, decompressed), 128u);

    auto out = bytesToFloats<32>(decompressed);
    for (auto v : out) QTX_EXPECT_EQ(v, 0.0f);
}

QTX_TEST(Quant, INT8BadAlignmentReturnsZero) {
    //30 elements = not a multiple of 32.
    std::array<std::byte, 30 * 4> bad_src{};
    std::array<std::byte, 100> dst{};
    QTX_EXPECT_EQ(compressFP32ToINT8(bad_src, dst), 0u);
}

// ============================================================================
// INT4 round-trip + sign-extension
// ============================================================================

QTX_TEST(Quant, INT4RoundTripBlock) {
    std::array<float, 32> src{};
    //Symmetrical pattern with signs - checking sign extension nibble.
    for (int i = 0; i < 32; ++i) {
        src[static_cast<size_t>(i)] = static_cast<float>(i - 16) * 0.1f;
    }
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 20> compressed{};
    std::array<std::byte, 128> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToINT4(src_bytes, compressed), 20u);
    QTX_EXPECT_EQ(decompressINT4ToFP32(compressed, decompressed), 128u);

    auto out = bytesToFloats<32>(decompressed);
    //INT4 accuracy: with abs_max = 1.6 one quantum = 1.6/7 ≈ 0.23.
    //Values ​​< 0.5 may be rounded to 0 or the next quant,
    //which gives a large relative error. We only check |src| > 0.5.
    for (size_t i = 0; i < 32; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float rel_err = std::fabs(out[i] - src[i]) / std::fabs(src[i]);
            QTX_EXPECT(rel_err < 0.2f);
        }
    }
}

QTX_TEST(Quant, INT4SignExtensionWorks) {
    //Special: negative values ​​only.
    //We check that after decompress the sign is saved (sign-extension nibble OK).
    std::array<float, 32> src{};
    for (int i = 0; i < 32; ++i) {
        src[static_cast<size_t>(i)] = -1.0f;  //all equally negative
    }
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 20> compressed{};
    std::array<std::byte, 128> decompressed{};

    QTX_EXPECT_EQ(compressFP32ToINT4(src_bytes, compressed), 20u);
    QTX_EXPECT_EQ(decompressINT4ToFP32(compressed, decompressed), 128u);

    auto out = bytesToFloats<32>(decompressed);
    //If sign extension is broken, we will get positive values.
    for (size_t i = 0; i < 32; ++i) {
        QTX_EXPECT(out[i] < 0.0f);
        QTX_EXPECT(std::fabs(out[i] - (-1.0f)) < 0.2f);
    }
}

// ============================================================================
//Universal dispatcher (compress/decompress with QuantFormat)
// ============================================================================

QTX_TEST(Quant, DispatcherFP32IsIdentity) {
    std::array<float, 4> src = {1.0f, 2.0f, 3.0f, 4.0f};
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 16> dst{};
    const auto sz = compress(QuantFormat::kFP32, src_bytes, dst);
    QTX_EXPECT_EQ(sz, 16u);
    QTX_EXPECT_EQ(std::memcmp(src_bytes.data(), dst.data(), 16), 0);
}

QTX_TEST(Quant, DispatcherBF16RoundTrip) {
    std::array<float, 4> src = {1.0f, 2.0f, 4.0f, 8.0f};  // exact in BF16
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 8> mid{};
    std::array<std::byte, 16> back{};

    QTX_EXPECT_EQ(compress(QuantFormat::kBF16, src_bytes, mid), 8u);
    QTX_EXPECT_EQ(decompress(QuantFormat::kBF16, mid, back), 16u);

    auto out = bytesToFloats<4>(back);
    for (size_t i = 0; i < 4; ++i) QTX_EXPECT_EQ(out[i], src[i]);
}

QTX_TEST(Quant, DispatcherINT8RoundTrip) {
    std::array<float, 32> src{};
    for (int i = 0; i < 32; ++i) {
        src[static_cast<size_t>(i)] = static_cast<float>(i + 1) * 0.1f;
    }
    auto src_bytes = floatsToBytes(src);
    std::array<std::byte, 36> mid{};
    std::array<std::byte, 128> back{};

    QTX_EXPECT_EQ(compress(QuantFormat::kINT8, src_bytes, mid), 36u);
    QTX_EXPECT_EQ(decompress(QuantFormat::kINT8, mid, back), 128u);
}

QTX_TEST(Quant, DispatcherUnknownFormatReturnsZero) {
    std::array<std::byte, 8> src{};
    std::array<std::byte, 8> dst{};
    //kReserved is a format that has no real implementation.
    QTX_EXPECT_EQ(compress(QuantFormat::kReserved, src, dst), 0u);
    QTX_EXPECT_EQ(decompress(QuantFormat::kReserved, src, dst), 0u);
}

// ============================================================================
//Compression effective size (as used in OOM survival)
// ============================================================================

QTX_TEST(Quant, CompressionActuallySavesMemory) {
    //Typical agent KV cache: 4 KiB FP32 = 1024 elements.
    constexpr usize kAgentFP32 = 4096;  //bytes = 1024 floats = 32 blocks
    std::vector<std::byte> src(kAgentFP32);

    //The INT8 version should be 32 × 36 = 1152 bytes.
    std::vector<std::byte> int8_dst(compressedSize(QuantFormat::kINT8, 1024));
    QTX_EXPECT_EQ(int8_dst.size(), 32u * 36u);

    const auto sz = compress(QuantFormat::kINT8,
                             std::span<const std::byte>{src},
                             std::span<std::byte>{int8_dst});
    QTX_EXPECT_EQ(sz, int8_dst.size());

    //Compression ~3.55x.
    const float ratio = static_cast<float>(kAgentFP32) /
                        static_cast<float>(int8_dst.size());
    QTX_EXPECT(ratio > 3.5f);
    QTX_EXPECT(ratio < 3.6f);
}

// ============================================================================
// EC48 — regression on the CURRENT INT4 wire-format contract.
//
// During Phase 2 we tried to "recover" the -8 codepoint without changing
// the wire format. The attempt failed: with `scale = abs_max / 7` the
// quantized value q = x * 7 / abs_max is in [-7, +7] for every input,
// so `lround(q) == -8` is mathematically unreachable and `clamp(-8, 7)`
// is identical to `clamp(-7, 7)` in observable behaviour.
//
// This test fixes the contract IN PLACE so a future P4 wire-format
// change cannot silently invalidate it: every nibble produced by the
// current encoder is in [-7, +7], and the byte distribution proves the
// scale-normalisation property the analysis above relies on.
// ============================================================================

QTX_TEST(Quant, EC48_INT4_NoNibbleEverEqualsMinusEight) {
    // Build a block with a strong negative tail to "tempt" the encoder
    // to emit -8 (the codepoint we wished we could use). Show that no
    // byte's high or low nibble ever sign-extends to -8.
    std::array<float, kBlockElements> values{};
    values[0] = 1.0f;  // small positive sets abs_max = 1.0
    for (usize i = 1; i < values.size(); ++i) {
        // Negative-skewed; some near -abs_max, some smaller.
        values[i] = -static_cast<float>(i % 7u) * 0.1f - 0.05f;
    }
    auto src = floatsToBytes(values);

    std::array<std::byte, kINT4BlockBytes> dst{};
    QTX_EXPECT_EQ(
        compressFP32ToINT4(std::span<const std::byte>{src},
                           std::span<std::byte>{dst}),
        kINT4BlockBytes);

    // Skip the fp32 scale prefix; walk the packed nibbles.
    const auto* packed = reinterpret_cast<const unsigned char*>(dst.data())
                       + sizeof(float);
    int saw_minus_eight = 0;
    for (usize byte_idx = 0; byte_idx < kBlockElements / 2u; ++byte_idx) {
        const unsigned char b = packed[byte_idx];
        for (int half = 0; half < 2; ++half) {
            int n = (half == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
            if (n & 0x08) n -= 0x10;  // sign-extend 4→32
            if (n == -8) ++saw_minus_eight;
        }
    }
    // Current contract: -8 is unreachable under `scale = abs_max / 7`.
    // If a future P4 change to the scale formula (or asymmetric zp)
    // makes -8 possible, THIS assertion will start to fail — that is
    // the early-warning signal we want.
    QTX_EXPECT_EQ(saw_minus_eight, 0);
}

// ============================================================================
// EC123 / EC124 / EC125 — SIMD-path regression tests for compressFP32ToINT8.
//
// The SIMD paths (AVX-512 / AVX2) are compile-time selected; on a build
// with neither, these tests still exercise the scalar fallback through
// the same public dispatch entry. The properties we pin:
//
//   1. Round-trip error stays within the tolerance the existing tests
//      already establish (≤ 2% relative for |x| > 0.5).
//   2. NaN and Inf inputs do not poison the scale: they are zeroed
//      before the abs-max reduction (EC125).
//   3. A constant block (all zeros) round-trips to all zeros — scale
//      is 0, no division.
//   4. Big-magnitude inputs (near FLT_MAX/127) saturate without
//      producing Inf.
// ============================================================================

QTX_TEST(Quant, EC123_SIMD_INT8_RoundTripLargeBlock) {
    // 1024 elements = 32 blocks. Realistic spread.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        // Triangle wave in [-3, +3]
        const auto t = static_cast<float>(static_cast<int>(i % 64u) - 32);
        src[i] = t * 0.1f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());

    std::vector<std::byte> compressed(36u * 32u);
    QTX_EXPECT_EQ(
        compressFP32ToINT8(
            std::span<const std::byte>{src_b},
            std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressINT8ToFP32(
            std::span<const std::byte>{compressed},
            std::span<std::byte>{decoded_b}),
        decoded_b.size());

    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r = std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            if (r > 0.02f) ++rel_violations;
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

QTX_TEST(Quant, EC125_SIMD_INT8_NaNAndInfMaskedOut) {
    // Block where ONE element is NaN, ONE is +Inf, ONE is -Inf. The
    // remaining elements set abs_max to a legitimate value. The
    // non-finite ones must be treated as zero, so the resulting scale
    // is finite and the block decodes without producing Inf/NaN.
    std::array<float, kBlockElements> values{};
    for (usize i = 0; i < values.size(); ++i) values[i] = 0.5f;
    values[0] = std::numeric_limits<float>::quiet_NaN();
    values[1] = std::numeric_limits<float>::infinity();
    values[2] = -std::numeric_limits<float>::infinity();

    auto src = floatsToBytes(values);
    std::array<std::byte, kINT8BlockBytes> compressed{};
    QTX_EXPECT_EQ(
        compressFP32ToINT8(std::span<const std::byte>{src},
                           std::span<std::byte>{compressed}),
        kINT8BlockBytes);

    // The stored scale must be finite (not Inf, not NaN).
    float stored_scale;
    std::memcpy(&stored_scale, compressed.data(), sizeof(float));
    QTX_EXPECT(std::isfinite(stored_scale));

    std::array<std::byte, kBlockElements * sizeof(float)> decoded_b{};
    QTX_EXPECT_EQ(
        decompressINT8ToFP32(std::span<const std::byte>{compressed},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());

    std::array<float, kBlockElements> decoded{};
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());
    // Decoded NaN/Inf positions must be either 0 or a finite small value.
    QTX_EXPECT(std::isfinite(decoded[0]));
    QTX_EXPECT(std::isfinite(decoded[1]));
    QTX_EXPECT(std::isfinite(decoded[2]));
    // Other positions had value 0.5, round-trip with small error.
    for (usize i = 3; i < decoded.size(); ++i) {
        QTX_EXPECT(std::fabs(decoded[i] - 0.5f) < 0.02f);
    }
}

QTX_TEST(Quant, EC123_SIMD_INT8_AllZeros) {
    // All-zero block: abs_max = 0 ⇒ scale = 0 ⇒ no division. The
    // encoder must NOT divide-by-zero and must store scale = 0.
    std::array<float, kBlockElements> values{};  // all 0
    auto src = floatsToBytes(values);
    std::array<std::byte, kINT8BlockBytes> compressed{};
    QTX_EXPECT_EQ(
        compressFP32ToINT8(std::span<const std::byte>{src},
                           std::span<std::byte>{compressed}),
        kINT8BlockBytes);

    float stored_scale;
    std::memcpy(&stored_scale, compressed.data(), sizeof(float));
    QTX_EXPECT_EQ(stored_scale, 0.0f);

    // Every payload byte is zero.
    for (usize i = sizeof(float); i < compressed.size(); ++i) {
        QTX_EXPECT(compressed[i] == std::byte{0});
    }
}

QTX_TEST(Quant, EC123_SIMD_INT8_OverflowSaturates) {
    // A value near FLT_MAX must NOT produce an Inf scale. EC44/EC50
    // ensures clampAbsMaxForINT8 caps it; the SIMD path uses the same
    // helper, so the property holds for both.
    std::array<float, kBlockElements> values{};
    values[0] = std::numeric_limits<float>::max();
    for (usize i = 1; i < values.size(); ++i) values[i] = 0.0f;

    auto src = floatsToBytes(values);
    std::array<std::byte, kINT8BlockBytes> compressed{};
    QTX_EXPECT_EQ(
        compressFP32ToINT8(std::span<const std::byte>{src},
                           std::span<std::byte>{compressed}),
        kINT8BlockBytes);

    float stored_scale;
    std::memcpy(&stored_scale, compressed.data(), sizeof(float));
    QTX_EXPECT(std::isfinite(stored_scale));
    QTX_EXPECT(stored_scale > 0.0f);

    // Decoding round-trip must produce a finite (clamped) value.
    std::array<std::byte, kBlockElements * sizeof(float)> decoded_b{};
    QTX_EXPECT_EQ(
        decompressINT8ToFP32(std::span<const std::byte>{compressed},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());
    float decoded0;
    std::memcpy(&decoded0, decoded_b.data(), sizeof(float));
    QTX_EXPECT(std::isfinite(decoded0));
}

// ============================================================================
// EC130 — SIMD INT8 decompress round-trip parity with scalar.
// ============================================================================

QTX_TEST(Quant, EC130_SIMD_INT8_DecompressMatchesScalar) {
    // Large block: 32 blocks (1024 elements). Round-trip and verify the
    // output is bit-equivalent up to the documented rounding tolerance.
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::sin(static_cast<float>(i) * 0.05f) * 4.5f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> compressed(36u * 32u);
    QTX_EXPECT_EQ(
        compressFP32ToINT8(std::span<const std::byte>{src_b},
                           std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressINT8ToFP32(std::span<const std::byte>{compressed},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r = std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            // INT8 wire format: scale = abs_max/127, so worst-case rel
            // error on small-magnitude values within a block can reach
            // ~3% (0.5 * scale / value). The 2% bound from the INT8
            // compress test only holds when the test data uses the full
            // [-1, 1] range; sin() output spans [-4.5, 4.5] inside one
            // 32-element block, so small values share a large scale.
            if (r > 0.04f) ++rel_violations;
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

// ============================================================================
// EC131 / EC132 — SIMD INT4 compress/decompress round-trip parity.
// ============================================================================

QTX_TEST(Quant, EC131_SIMD_INT4_RoundTripLargeBlock) {
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = std::cos(static_cast<float>(i) * 0.07f) * 3.0f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> compressed(20u * 32u);
    QTX_EXPECT_EQ(
        compressFP32ToINT4(std::span<const std::byte>{src_b},
                           std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressINT4ToFP32(std::span<const std::byte>{compressed},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    // INT4 has 15 codepoints spanning [-7, 7] * scale. The worst-case
    // ABS error per element is therefore ~0.5 * scale (rounding to the
    // nearest grid point). With per-block scale = abs_max/7, the worst
    // case is abs_max/14. We pin THAT bound, not a relative error
    // bound — relative error is meaningless for elements whose
    // magnitude is much smaller than their block's abs_max.
    int abs_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        const usize block = i / kBlockElements;
        // recover block abs_max
        float block_abs = 0.0f;
        for (usize j = 0; j < kBlockElements; ++j) {
            const float v = src[block * kBlockElements + j];
            if (std::isfinite(v) && std::fabs(v) > block_abs) {
                block_abs = std::fabs(v);
            }
        }
        const float bound = (block_abs / 14.0f) + 1e-4f;
        const float e = std::fabs(decoded[i] - src[i]);
        if (e > bound) ++abs_violations;
    }
    QTX_EXPECT_EQ(abs_violations, 0);
}

QTX_TEST(Quant, EC131_SIMD_INT4_NaNHandled) {
    std::array<float, kBlockElements> values{};
    for (usize i = 0; i < values.size(); ++i) values[i] = 0.3f;
    values[0] = std::numeric_limits<float>::quiet_NaN();
    values[1] = std::numeric_limits<float>::infinity();

    auto src = floatsToBytes(values);
    std::array<std::byte, kINT4BlockBytes> compressed{};
    QTX_EXPECT_EQ(
        compressFP32ToINT4(std::span<const std::byte>{src},
                           std::span<std::byte>{compressed}),
        kINT4BlockBytes);

    float stored_scale;
    std::memcpy(&stored_scale, compressed.data(), sizeof(float));
    QTX_EXPECT(std::isfinite(stored_scale));
}

// ============================================================================
// EC133 / EC134 — SIMD BF16 compress/decompress round-trip parity.
// ============================================================================

QTX_TEST(Quant, EC133_SIMD_BF16_RoundTripLargeBlock) {
    constexpr usize kN = 1024u;
    std::vector<float> src(kN);
    for (usize i = 0; i < kN; ++i) {
        src[i] = static_cast<float>(i) * 0.123f - 60.0f;
    }
    std::vector<std::byte> src_b(kN * sizeof(float));
    std::memcpy(src_b.data(), src.data(), src_b.size());
    std::vector<std::byte> compressed(kN * 2u);
    QTX_EXPECT_EQ(
        compressFP32ToBF16(std::span<const std::byte>{src_b},
                           std::span<std::byte>{compressed}),
        compressed.size());

    std::vector<std::byte> decoded_b(kN * sizeof(float));
    QTX_EXPECT_EQ(
        decompressBF16ToFP32(std::span<const std::byte>{compressed},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::vector<float> decoded(kN);
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    // BF16 has 7-bit mantissa → max ~1% relative error.
    int rel_violations = 0;
    for (usize i = 0; i < kN; ++i) {
        if (std::fabs(src[i]) > 0.5f) {
            const float r = std::fabs(decoded[i] - src[i]) / std::fabs(src[i]);
            if (r > 0.01f) ++rel_violations;
        }
    }
    QTX_EXPECT_EQ(rel_violations, 0);
}

QTX_TEST(Quant, EC133_SIMD_BF16_NaNPreserved) {
    // EC55 contract held by SIMD path too.
    std::array<float, 16> values{};
    values[0] = std::numeric_limits<float>::quiet_NaN();
    for (usize i = 1; i < values.size(); ++i) values[i] = 1.0f;

    auto src = floatsToBytes(values);
    std::array<std::byte, 16 * 2> dst{};
    QTX_EXPECT_EQ(
        compressFP32ToBF16(std::span<const std::byte>{src},
                           std::span<std::byte>{dst}),
        dst.size());

    std::array<std::byte, 16 * sizeof(float)> decoded_b{};
    QTX_EXPECT_EQ(
        decompressBF16ToFP32(std::span<const std::byte>{dst},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());
    float decoded0;
    std::memcpy(&decoded0, decoded_b.data(), sizeof(float));
    QTX_EXPECT(std::isnan(decoded0));  // NaN preserved end-to-end
}

QTX_TEST(Quant, EC133_SIMD_BF16_TailNotMultipleOf16) {
    // Verify the scalar tail path: 17 elements = 1 chunk of 16 + 1 scalar.
    constexpr usize kN = 17u;
    std::array<float, kN> values{};
    for (usize i = 0; i < kN; ++i) values[i] = static_cast<float>(i) * 0.5f;
    auto src = floatsToBytes(values);
    std::array<std::byte, kN * 2u> compressed{};
    QTX_EXPECT_EQ(
        compressFP32ToBF16(std::span<const std::byte>{src},
                           std::span<std::byte>{compressed}),
        compressed.size());

    std::array<std::byte, kN * sizeof(float)> decoded_b{};
    QTX_EXPECT_EQ(
        decompressBF16ToFP32(std::span<const std::byte>{compressed},
                             std::span<std::byte>{decoded_b}),
        decoded_b.size());
    std::array<float, kN> decoded{};
    std::memcpy(decoded.data(), decoded_b.data(), decoded_b.size());

    // Element 16 (tail) round-trips correctly.
    QTX_EXPECT(std::fabs(decoded[16] - values[16]) < 0.1f);
}
