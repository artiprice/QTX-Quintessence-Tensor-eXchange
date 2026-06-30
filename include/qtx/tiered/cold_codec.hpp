// ============================================================================
// @file        cold_codec.hpp
// @brief       Pluggable cold-tier codec adapter for TieredBridge.
// @author      QTX Project
// @license     GNU AGPL v3.0
// ============================================================================
//
// PURPOSE
//   The TieredBridge migrates "sleeper" tensors from the hot (FP32) tier
//   to the cold tier, compressing them on the way down and decompressing
//   on the way back up. Historically the cold format was hard-wired to
//   INT8. This header decouples that choice behind an ADAPTER so that:
//
//     * the cold format is selectable (any of QTX's 25 built-in formats),
//     * a runtime flag / string can pick the format (utility front-ends),
//     * THIRD-PARTY quantizers can be plugged in without touching the
//       bridge — implement the small ICodecAdapter vtable (or satisfy the
//       compile-time ColdCodecPolicy concept) and pass it in.
//
// TWO LAYERS
//   1. Compile-time policy (`ColdCodecPolicy` concept). Zero-overhead:
//      the bridge is a template on the policy, the calls inline to direct
//      compress/decompress. Use when the cold format is fixed at build
//      time. Built-ins: Int8ColdCodec, Int4ColdCodec, Fp16ColdCodec,
//      Bf16ColdCodec, and the generic FormatColdCodec<Fmt>.
//
//   2. Runtime adapter (`ICodecAdapter` vtable). Pick the codec from a
//      flag/string at runtime, or inject a foreign quantizer. The bridge
//      can be templated on `RuntimeColdCodec`, which forwards to whatever
//      ICodecAdapter it was handed at construction.
//
// HA-SAFETY
//   * Every adapter call is noexcept and returns a byte count (0 == clean
//     failure). No exceptions cross the boundary, including from foreign
//     codecs (PolicyCodecAdapter wraps the policy call in noexcept).
//   * compressedBound(elements) is the worst-case cold-slot size; the
//     bridge static_asserts / runtime-checks that the cold slot can hold
//     it before ever calling compress, so compress never overflows dst.
//   * A foreign codec that violates the contract (writes more than it
//     reported, or fails to decode its own output) can only corrupt the
//     tenant whose data it owns — the bridge treats a wrong byte count as
//     a clean failure and leaves the tenant in its prior state.
// ============================================================================
#pragma once

#include "qtx/core/types.hpp"
#include "qtx/arena/gen_id.hpp"
#include "qtx/quantize/quantizer.hpp"

#include <array>
#include <span>
#include <string_view>

namespace qtx::tiered::codec {

// ----------------------------------------------------------------------------
// Compile-time policy concept.
//
// A ColdCodecPolicy is any type exposing three static members:
//
//   static constexpr core::usize compressedBound(core::usize elements) noexcept;
//       Worst-case compressed byte count for `elements` FP32 values.
//       MUST be >= the value compress() actually writes for any input of
//       that element count. Returns 0 if `elements` is unsupported (e.g.
//       not a multiple of the format's block size).
//
//   static core::usize compress(span<const byte> src_fp32,
//                               span<byte> dst) noexcept;
//       Compress `src_fp32` (FP32 bytes) into `dst`. Returns bytes
//       written, or 0 on failure. MUST NOT write more than
//       compressedBound(src_fp32.size()/4) bytes.
//
//   static core::usize decompress(span<const byte> src,
//                                 span<byte> dst_fp32) noexcept;
//       Decompress `src` into `dst_fp32`. Returns FP32 bytes written, or
//       0 on failure.
//
//   static constexpr std::string_view name() noexcept;   // human label
// ----------------------------------------------------------------------------
template <typename P>
concept ColdCodecPolicy = requires(
    std::span<const std::byte> cs, std::span<std::byte> s, core::usize n)
{
    { P::compressedBound(n) } noexcept -> std::same_as<core::usize>;
    { P::compress(cs, s) }    noexcept -> std::same_as<core::usize>;
    { P::decompress(cs, s) }  noexcept -> std::same_as<core::usize>;
    { P::name() }             noexcept -> std::convertible_to<std::string_view>;
};

// ----------------------------------------------------------------------------
// Generic built-in policy: any QuantFormat handled by the universal
// quantize::compress / quantize::decompress dispatcher.
//
// `Fmt` is the cold storage format. compressedBound forwards to the
// quantizer's constexpr compressedSize for that format.
// ----------------------------------------------------------------------------
template <arena::QuantFormat Fmt>
struct FormatColdCodec {
    [[nodiscard]] static constexpr core::usize compressedBound(
        core::usize elements) noexcept
    {
        return quantize::compressedSize(Fmt, elements);
    }

    [[nodiscard]] static core::usize compress(
        std::span<const std::byte> src_fp32,
        std::span<std::byte>       dst) noexcept
    {
        return quantize::compress(Fmt, src_fp32, dst);
    }

    [[nodiscard]] static core::usize decompress(
        std::span<const std::byte> src,
        std::span<std::byte>       dst_fp32) noexcept
    {
        return quantize::decompress(Fmt, src, dst_fp32);
    }

    [[nodiscard]] static constexpr std::string_view name() noexcept {
        return formatName(Fmt);
    }

    [[nodiscard]] static constexpr std::string_view formatName(
        arena::QuantFormat f) noexcept
    {
        using F = arena::QuantFormat;
        switch (f) {
            case F::kFP32:     return "fp32";
            case F::kBF16:     return "bf16";
            case F::kFP16:     return "fp16";
            case F::kFP8_E4M3: return "fp8_e4m3";
            case F::kINT8:     return "int8";
            case F::kINT4:     return "int4";
            case F::kFP8_E5M2: return "fp8_e5m2";
            default:           return "unknown";
        }
    }
};

// Convenience aliases for the formats the bridge can use as cold storage
// (the ones the universal dispatcher supports — slot-scarce K/I-Quants are
// reachable via the runtime adapter, see ForeignCodecAdapter usage below).
using Int8ColdCodec = FormatColdCodec<arena::QuantFormat::kINT8>;
using Int4ColdCodec = FormatColdCodec<arena::QuantFormat::kINT4>;
using Fp16ColdCodec = FormatColdCodec<arena::QuantFormat::kFP16>;
using Bf16ColdCodec = FormatColdCodec<arena::QuantFormat::kBF16>;

static_assert(ColdCodecPolicy<Int8ColdCodec>);
static_assert(ColdCodecPolicy<Int4ColdCodec>);
static_assert(ColdCodecPolicy<Fp16ColdCodec>);
static_assert(ColdCodecPolicy<Bf16ColdCodec>);

// ----------------------------------------------------------------------------
// Runtime adapter (vtable). Implement this to plug a codec chosen at
// runtime — including a THIRD-PARTY quantizer that QTX knows nothing
// about. The bridge, when templated on RuntimeColdCodec, forwards every
// cold-tier operation through a pointer to one of these.
// ----------------------------------------------------------------------------
class ICodecAdapter {
public:
    virtual ~ICodecAdapter() = default;

    /// Worst-case compressed size for `elements` FP32 values, or 0 if the
    /// element count is unsupported. The bridge uses this to size / verify
    /// the cold slot; it must never be exceeded by compress().
    [[nodiscard]] virtual core::usize compressedBound(
        core::usize elements) const noexcept = 0;

    /// Compress FP32 bytes into dst. Returns bytes written, 0 on failure.
    [[nodiscard]] virtual core::usize compress(
        std::span<const std::byte> src_fp32,
        std::span<std::byte>       dst) const noexcept = 0;

    /// Decompress into FP32 bytes. Returns FP32 bytes written, 0 on failure.
    [[nodiscard]] virtual core::usize decompress(
        std::span<const std::byte> src,
        std::span<std::byte>       dst_fp32) const noexcept = 0;

    /// Human-readable codec name (for telemetry / CLI echo).
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

// ----------------------------------------------------------------------------
// Bridge any compile-time ColdCodecPolicy into the runtime ICodecAdapter
// vtable. Lets every built-in policy be used through the runtime path with
// zero extra code.
// ----------------------------------------------------------------------------
template <ColdCodecPolicy Policy>
class PolicyCodecAdapter final : public ICodecAdapter {
public:
    [[nodiscard]] core::usize compressedBound(
        core::usize elements) const noexcept override
    {
        return Policy::compressedBound(elements);
    }
    [[nodiscard]] core::usize compress(
        std::span<const std::byte> src_fp32,
        std::span<std::byte>       dst) const noexcept override
    {
        return Policy::compress(src_fp32, dst);
    }
    [[nodiscard]] core::usize decompress(
        std::span<const std::byte> src,
        std::span<std::byte>       dst_fp32) const noexcept override
    {
        return Policy::decompress(src, dst_fp32);
    }
    [[nodiscard]] std::string_view name() const noexcept override {
        return Policy::name();
    }
};

// ----------------------------------------------------------------------------
// Function-pointer adapter. The lightest way to wrap a FOREIGN codec: the
// caller supplies three free functions (matching the C-ABI-friendly
// signatures) and an optional name. No inheritance needed on their side.
//
//   codec::ForeignCodecAdapter my_codec{
//       &third_party_bound, &third_party_compress,
//       &third_party_decompress, "awq-int4"};
//   bridge.setColdCodec(&my_codec);
//
// All three function pointers MUST be non-null and noexcept-safe. The
// adapter performs the null-check defensively and fails clean (returns 0)
// if any pointer is missing.
// ----------------------------------------------------------------------------
class ForeignCodecAdapter final : public ICodecAdapter {
public:
    using BoundFn = core::usize (*)(core::usize elements) noexcept;
    using CompressFn = core::usize (*)(
        std::span<const std::byte> src_fp32,
        std::span<std::byte>       dst) noexcept;
    using DecompressFn = core::usize (*)(
        std::span<const std::byte> src,
        std::span<std::byte>       dst_fp32) noexcept;

    BoundFn          bound_fn      = nullptr;
    CompressFn       compress_fn   = nullptr;
    DecompressFn     decompress_fn = nullptr;
    std::string_view label         = "foreign";

    ForeignCodecAdapter() noexcept = default;
    ForeignCodecAdapter(BoundFn b, CompressFn c, DecompressFn d,
                        std::string_view l = "foreign") noexcept
        : bound_fn(b), compress_fn(c), decompress_fn(d), label(l) {}

    [[nodiscard]] core::usize compressedBound(
        core::usize elements) const noexcept override
    {
        return bound_fn ? bound_fn(elements) : 0u;
    }
    [[nodiscard]] core::usize compress(
        std::span<const std::byte> src_fp32,
        std::span<std::byte>       dst) const noexcept override
    {
        return compress_fn ? compress_fn(src_fp32, dst) : 0u;
    }
    [[nodiscard]] core::usize decompress(
        std::span<const std::byte> src,
        std::span<std::byte>       dst_fp32) const noexcept override
    {
        return decompress_fn ? decompress_fn(src, dst_fp32) : 0u;
    }
    [[nodiscard]] std::string_view name() const noexcept override {
        return label;
    }
};

// ----------------------------------------------------------------------------
// Runtime cold-codec policy. Satisfies ColdCodecPolicy by forwarding to a
// pointer-held ICodecAdapter. This is the policy you template the bridge on
// when the cold format is chosen at runtime (e.g. from a CLI flag).
//
// The adapter pointer is thread-local-free: it is set once at bridge
// construction (or via setColdCodec before any tenant exists) and read in
// the hot path. Swapping it while tenants hold cold data is a caller
// contract violation (the old format's bytes would be decoded by the new
// codec) — guard documented, not enforced, to keep the hot path branchless.
//
// NOTE: because ColdCodecPolicy requires STATIC members, RuntimeColdCodec
// holds the active adapter in a function-local static pointer. For a
// process that needs multiple bridges with different runtime codecs
// simultaneously, use distinct tag types (RuntimeColdCodec<Tag>).
// ----------------------------------------------------------------------------
template <typename Tag = struct DefaultRuntimeTag>
struct RuntimeColdCodec {
    [[nodiscard]] static const ICodecAdapter*& slot() noexcept {
        static const ICodecAdapter* p = defaultAdapter();
        return p;
    }

    /// The fallback codec used until setAdapter() is called: INT8, the
    /// historical default, so existing deployments behave identically.
    [[nodiscard]] static const ICodecAdapter* defaultAdapter() noexcept {
        static const PolicyCodecAdapter<Int8ColdCodec> kInt8;
        return &kInt8;
    }

    /// Install the active runtime codec. Pass nullptr to reset to INT8.
    static void setAdapter(const ICodecAdapter* a) noexcept {
        slot() = a ? a : defaultAdapter();
    }

    [[nodiscard]] static const ICodecAdapter* adapter() noexcept {
        return slot();
    }

    [[nodiscard]] static core::usize compressedBound(
        core::usize elements) noexcept
    {
        return slot()->compressedBound(elements);
    }
    [[nodiscard]] static core::usize compress(
        std::span<const std::byte> src_fp32,
        std::span<std::byte>       dst) noexcept
    {
        return slot()->compress(src_fp32, dst);
    }
    [[nodiscard]] static core::usize decompress(
        std::span<const std::byte> src,
        std::span<std::byte>       dst_fp32) noexcept
    {
        return slot()->decompress(src, dst_fp32);
    }
    [[nodiscard]] static std::string_view name() noexcept {
        return slot()->name();
    }
};

static_assert(ColdCodecPolicy<RuntimeColdCodec<>>);

// ----------------------------------------------------------------------------
// String / enum registry: map a utility flag to a built-in codec adapter.
//
// Returns a pointer to a static adapter instance (stable for the process
// lifetime) for the given format name, or nullptr if unknown. Intended for
// CLI wiring:  --cold-format int4   ->  codec::adapterByName("int4").
//
// Only the universal-dispatcher formats are registered here; foreign /
// K-/I-Quant codecs are wired by the caller via ForeignCodecAdapter.
// ----------------------------------------------------------------------------
[[nodiscard]] inline const ICodecAdapter* adapterByName(
    std::string_view name) noexcept
{
    static const PolicyCodecAdapter<FormatColdCodec<arena::QuantFormat::kBF16>>     kBf16;
    static const PolicyCodecAdapter<FormatColdCodec<arena::QuantFormat::kFP16>>     kFp16;
    static const PolicyCodecAdapter<FormatColdCodec<arena::QuantFormat::kFP8_E4M3>> kFp8E4m3;
    static const PolicyCodecAdapter<FormatColdCodec<arena::QuantFormat::kFP8_E5M2>> kFp8E5m2;
    static const PolicyCodecAdapter<FormatColdCodec<arena::QuantFormat::kINT8>>     kInt8;
    static const PolicyCodecAdapter<FormatColdCodec<arena::QuantFormat::kINT4>>     kInt4;

    if (name == "bf16")     return &kBf16;
    if (name == "fp16")     return &kFp16;
    if (name == "fp8_e4m3" || name == "fp8") return &kFp8E4m3;
    if (name == "fp8_e5m2") return &kFp8E5m2;
    if (name == "int8")     return &kInt8;
    if (name == "int4")     return &kInt4;
    return nullptr;
}

/// Same mapping keyed by the QuantFormat enum (for programmatic wiring).
[[nodiscard]] inline const ICodecAdapter* adapterByFormat(
    arena::QuantFormat fmt) noexcept
{
    using F = arena::QuantFormat;
    switch (fmt) {
        case F::kBF16:     return adapterByName("bf16");
        case F::kFP16:     return adapterByName("fp16");
        case F::kFP8_E4M3: return adapterByName("fp8_e4m3");
        case F::kFP8_E5M2: return adapterByName("fp8_e5m2");
        case F::kINT8:     return adapterByName("int8");
        case F::kINT4:     return adapterByName("int4");
        default:           return nullptr;
    }
}

}  // namespace qtx::tiered::codec
