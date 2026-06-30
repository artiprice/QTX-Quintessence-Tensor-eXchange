// ============================================================================
// @file        test_cold_codec.cpp
// @brief       Tests for the pluggable cold-tier codec adapter.
// @author      QTX Project
// ============================================================================
//
// Proves the TieredArenaBridge cold format is no longer hard-wired to
// INT8: the same bridge can be compiled with INT4 / FP16 cold storage, a
// FOREIGN codec (function-pointer adapter) can be plugged in, and a codec
// can be selected at runtime from a string flag.
// ============================================================================

#include "test_harness.hpp"
#include "qtx/tiered/tiered_bridge.hpp"
#include "qtx/tiered/cold_codec.hpp"
#include "qtx/arena/fractal_arena.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <span>

using namespace qtx;
using qtx::core::usize;

// Same arena geometry as the main tiered tests: 64 hot × 4 KiB,
// 512 cold × 1152 bytes. INT8 fills the cold slot exactly (1152);
// INT4 (640) and FP16 (2048 -> does NOT fit) let us exercise the
// fit / no-fit boundary.
using HotA  = arena::FractalArena<64, 4096>;
using ColdA = arena::FractalArena<512, 1152>;

// ----------------------------------------------------------------------------
// Compile-time policy selection
// ----------------------------------------------------------------------------

using TieredInt8 = tiered::TieredArenaBridge<HotA, ColdA, 4096u,
                                             tiered::codec::Int8ColdCodec>;
using TieredInt4 = tiered::TieredArenaBridge<HotA, ColdA, 4096u,
                                             tiered::codec::Int4ColdCodec>;

QTX_TEST(ColdCodec, DefaultIsInt8) {
    // The default template parameter must keep the historical INT8 cold
    // format, so existing code that wrote `TieredArenaBridge<HotA,ColdA>`
    // is unchanged.
    using Default = tiered::TieredArenaBridge<HotA, ColdA>;
    QTX_EXPECT_EQ(Default::coldCodecName(), std::string_view{"int8"});
    QTX_EXPECT_EQ(Default::kCompressedBytes, 32u * 36u);   // INT8: 1152
}

QTX_TEST(ColdCodec, Int4CodecHasSmallerFootprint) {
    QTX_EXPECT_EQ(TieredInt4::coldCodecName(), std::string_view{"int4"});
    // INT4 = 32 blocks × 20 bytes = 640, < INT8's 1152.
    QTX_EXPECT_EQ(TieredInt4::kCompressedBytes, 32u * 20u);
    QTX_EXPECT(TieredInt4::coldCompressedBytes() < TieredInt8::coldCompressedBytes());
}

QTX_TEST(ColdCodec, Int4RoundTripThroughEviction) {
    // Full dynamic-quantization cycle with INT4 cold storage: write FP32,
    // force eviction (compress to INT4), promote back (decompress), verify.
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    TieredInt4 tb(hot.get(), cold.get());

    auto h0 = tb.createTenant();
    auto view = tb.acquireFP32(h0);
    QTX_EXPECT(!view.empty());

    constexpr usize n = TieredInt4::kPayloadElements;
    for (usize i = 0; i < n; ++i) {
        const float v = std::sin(static_cast<float>(i) * 0.01f);
        std::memcpy(view.data() + i * sizeof(float), &v, sizeof(float));
    }
    QTX_EXPECT(tb.release(h0));

    // Fill 65 more tenants to force at least one eviction.
    for (int i = 0; i < 65; ++i) {
        QTX_EXPECT(tb.createTenant().is_valid);
    }
    QTX_EXPECT(tb.stats().evictions >= 1u);

    auto back = tb.acquireFP32(h0);
    QTX_EXPECT(!back.empty());

    // INT4 is a true 4-bit codec: a handful of samples near zero-
    // crossings and signal peaks exceed a tight band. We require the
    // signal to SURVIVE the FP32->INT4->FP32 round trip: at least 95%
    // of significant samples within 20% relative error, and no sample
    // worse than 35% (catches a broken codec / wrong format).
    int total = 0, within = 0;
    float worst = 0.0f;
    for (usize i = 0; i < n; ++i) {
        float restored;
        std::memcpy(&restored, back.data() + i * sizeof(float), sizeof(float));
        const float original = std::sin(static_cast<float>(i) * 0.01f);
        if (std::fabs(original) > 0.05f) {
            ++total;
            const float rel = std::fabs(restored - original) / std::fabs(original);
            if (rel <= 0.20f) ++within;
            if (rel > worst) worst = rel;
        }
    }
    QTX_EXPECT(total > 0);
    QTX_EXPECT(static_cast<double>(within) / static_cast<double>(total) >= 0.95);
    QTX_EXPECT(worst < 0.35f);
    QTX_EXPECT(tb.release(h0));
}

// ----------------------------------------------------------------------------
// Foreign codec via function-pointer adapter
// ----------------------------------------------------------------------------

namespace {

// A trivial "foreign" codec: lossless FP32 passthrough (no quantization).
// Stands in for any third-party quantizer the bridge knows nothing about.
core::usize foreignBound(core::usize elements) noexcept {
    return elements * sizeof(float);   // lossless: same size
}
core::usize foreignCompress(std::span<const std::byte> src,
                            std::span<std::byte> dst) noexcept {
    if (dst.size() < src.size()) return 0u;
    std::memcpy(dst.data(), src.data(), src.size());
    return src.size();
}
core::usize foreignDecompress(std::span<const std::byte> src,
                              std::span<std::byte> dst) noexcept {
    if (dst.size() < src.size()) return 0u;
    std::memcpy(dst.data(), src.data(), src.size());
    return src.size();
}

}  // namespace

QTX_TEST(ColdCodec, ForeignAdapterRoundTrip) {
    tiered::codec::ForeignCodecAdapter adapter{
        &foreignBound, &foreignCompress, &foreignDecompress, "foreign-lossless"};

    // Through the runtime adapter, the bound for 8 elements is 32 bytes.
    QTX_EXPECT_EQ(adapter.name(), std::string_view{"foreign-lossless"});
    QTX_EXPECT_EQ(adapter.compressedBound(8u), 32u);

    // Round-trip a buffer directly through the adapter vtable.
    std::array<float, 8> in{1.0f, -2.5f, 3.25f, 0.0f, 100.0f, -0.001f, 42.0f, -7.0f};
    std::array<std::byte, 32> mid{};
    std::array<float, 8> out{};

    auto src = std::as_bytes(std::span<const float>{in});
    const auto w = adapter.compress(src, mid);
    QTX_EXPECT_EQ(w, 32u);
    auto out_bytes = std::as_writable_bytes(std::span<float>{out});
    const auto r = adapter.decompress(mid, out_bytes);
    QTX_EXPECT_EQ(r, 32u);
    for (int i = 0; i < 8; ++i) QTX_EXPECT_EQ(in[static_cast<usize>(i)],
                                              out[static_cast<usize>(i)]);
}

QTX_TEST(ColdCodec, ForeignAdapterNullPointersFailClean) {
    // A misconfigured foreign adapter (missing function pointers) must
    // fail clean (return 0), never crash.
    tiered::codec::ForeignCodecAdapter empty{};
    std::array<std::byte, 16> buf{};
    QTX_EXPECT_EQ(empty.compressedBound(8u), 0u);
    QTX_EXPECT_EQ(empty.compress(buf, buf), 0u);
    QTX_EXPECT_EQ(empty.decompress(buf, buf), 0u);
}

// ----------------------------------------------------------------------------
// Runtime selection by name / format
// ----------------------------------------------------------------------------

QTX_TEST(ColdCodec, RegistryResolvesKnownNames) {
    using namespace tiered::codec;
    QTX_EXPECT(adapterByName("int8")  != nullptr);
    QTX_EXPECT(adapterByName("int4")  != nullptr);
    QTX_EXPECT(adapterByName("fp16")  != nullptr);
    QTX_EXPECT(adapterByName("bf16")  != nullptr);
    QTX_EXPECT(adapterByName("fp8")   != nullptr);
    QTX_EXPECT(adapterByName("nonsense") == nullptr);

    QTX_EXPECT_EQ(adapterByName("int4")->name(), std::string_view{"int4"});
    QTX_EXPECT_EQ(adapterByName("int4")->compressedBound(1024u), 32u * 20u);
    QTX_EXPECT_EQ(adapterByName("int8")->compressedBound(1024u), 32u * 36u);
}

QTX_TEST(ColdCodec, RegistryByFormatEnum) {
    using namespace tiered::codec;
    QTX_EXPECT(adapterByFormat(arena::QuantFormat::kINT8) != nullptr);
    QTX_EXPECT(adapterByFormat(arena::QuantFormat::kINT4) != nullptr);
    // kFP32 is not a cold-compression target -> not registered.
    QTX_EXPECT(adapterByFormat(arena::QuantFormat::kFP32) == nullptr);
}

QTX_TEST(ColdCodec, RuntimePolicySwitchesCodec) {
    // RuntimeColdCodec forwards to whatever adapter is installed. Default
    // is INT8; install INT4 and confirm the bound changes.
    using RT = tiered::codec::RuntimeColdCodec<>;
    // Default before any setAdapter call.
    QTX_EXPECT_EQ(RT::compressedBound(1024u), 32u * 36u);   // int8

    RT::setAdapter(tiered::codec::adapterByName("int4"));
    QTX_EXPECT_EQ(RT::name(), std::string_view{"int4"});
    QTX_EXPECT_EQ(RT::compressedBound(1024u), 32u * 20u);   // int4

    // Reset to default for hygiene (other tests may use RT<>).
    RT::setAdapter(nullptr);
    QTX_EXPECT_EQ(RT::name(), std::string_view{"int8"});
    QTX_EXPECT_EQ(RT::compressedBound(1024u), 32u * 36u);
}

QTX_TEST(ColdCodec, PolicyConceptSatisfiedByBuiltins) {
    // Compile-time guard: every built-in alias models ColdCodecPolicy.
    static_assert(tiered::codec::ColdCodecPolicy<tiered::codec::Int8ColdCodec>);
    static_assert(tiered::codec::ColdCodecPolicy<tiered::codec::Int4ColdCodec>);
    static_assert(tiered::codec::ColdCodecPolicy<tiered::codec::Fp16ColdCodec>);
    static_assert(tiered::codec::ColdCodecPolicy<tiered::codec::Bf16ColdCodec>);
    static_assert(tiered::codec::ColdCodecPolicy<tiered::codec::RuntimeColdCodec<>>);
    QTX_EXPECT(true);
}
