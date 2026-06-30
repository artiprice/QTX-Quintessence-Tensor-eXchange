// ============================================================================
// @file        gen_id.hpp
//@brief Generational Index - 64-bit value-type descriptor of the entity.
// @author      QTX Project
// @date        2026-05-12
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
//HA-LAYER: HA-pure module. No signs, no allocations,
//no exceptions. All operations are value-based, constexpr-friendly.
//
//SEMANTICS:
//GenID identifies a slot in FractalArena. Contains:
//- id (24 bits) : slot index, 16.7M unique values
//- version (16 bits): generation, ABA protection after reuse
//- alignment_class (7 bits) : padding offset 0..127 (Apple Silicon 128B-aware)
//- quant_format (3 bits): quantization format (8 values)
//- state (2 bits): slot lifecycle
//- sub_slot (5 bits) : Phase 2 Block C sub-slabbing index, [0..31]
//- metadata (7 bits) : opaque per-bridge tag, [0..127]
//Sum: 24 + 16 + 7 + 3 + 2 + 5 + 7 = 64 bits.
//
//HA-CONTRACT:
//- Layout is encapsulated: NOT union, NOT bit-field struct (UB on different ABIs).
//- Encoding/decoding via explicit bit operations (verifiable).
//- All operations are noexcept, constexpr, trivially-copyable.
//- Backward compatibility (EC110): sub_slot=0 AND metadata=0 means
//  "this GenID names the entire slot" — every legacy GenID has those
//  bits at 0 by construction, so old callers see no behaviour change.
//
// EDGE CASES CLOSED IN THIS REVISION:
//   EC110 — split the 12 previously-reserved bits into 5+7 (sub_slot +
//           metadata) for Phase 2 Block C sub-slabbing. Out-of-range
//           values in `withSubSlot()` / `withMetadata()` return the
//           original handle unchanged (fail-safe, no bit poisoning).
//
// ============================================================================

#pragma once

#include "../core/types.hpp"

#include <compare>
#include <cstdint>
#include <type_traits>

namespace qtx::arena {

// ============================================================================
// @enum QuantFormat
//@brief Sleeper agent quantization formats. 3 bits, 8 values.
//
//Expands via reserved GenID bits without breaking change. Each value
//corresponds to one physical representation of the KV cache tensors.
// ============================================================================
enum class QuantFormat : core::u8 {
    kFP32      = 0u,  ///< Full precision, hot path (L1)
    kBF16      = 1u,  ///< bfloat16, balance of accuracy and density
    kFP16      = 2u,  ///< IEEE half precision
    kFP8_E4M3  = 3u,  ///< OCP/Hopper/Intel 8-bit float, 4-exp 3-mant (acts/weights)
    kINT8      = 4u,  ///< 8-bit signed quantization
    kINT4      = 5u,  ///< 4-bit, aggressive compression
    kFP8_E5M2  = 6u,  ///< OCP/Hopper/Intel 8-bit float, 5-exp 2-mant (grads, wider range)
    kReserved  = 7u,  ///< Reserve for future formats
};

// ============================================================================
// @enum SlotState
//@brief The state of the slot in the lifecycle. 2 bits, 4 values.
// ============================================================================
enum class SlotState : core::u8 {
    kFree      = 0u,  ///< Slot free (default, zero-init safe)
    kActive    = 1u,  ///< Agent active, data in L1/L2
    kDormant   = 2u,  ///< Dormant, not compressed (evicted to L3/RAM)
    kQuantized = 3u,  ///< Sleeping + quantized (lossy compression applied)
};

// ============================================================================
// @class GenID
//@brief 64-bit value-type descriptor. Replaces raw pointers in qtx.
//
//HA-invariants:
//   - sizeof(GenID) == 8
//   - alignof(GenID) == 8
//   - std::is_trivially_copyable_v<GenID>
//   - std::is_standard_layout_v<GenID>
//- Zero-initialized GenID validly represents "null" (state == kFree).
//
// Layout (LSB → MSB):
//   [0..23]   id              (slot index, 16.7M values)
//   [24..39]  version         (ABA generation)
//   [40..46]  alignment_class (padding 0..127)
//   [47..49]  quant_format    (FP32/BF16/INT8/INT4/...)
//   [50..51]  state           (Free/Active/Dormant/Quantized)
//   [52..56]  sub_slot        (EC110: 0..31 sub-slabbing index)
//   [57..63]  metadata        (EC110: 0..127 opaque per-bridge tag)
// ============================================================================
class GenID {
public:
    //=== Bit widths (single source of truth) ===
    static constexpr core::u32 kIdBits        = 24u;
    static constexpr core::u32 kVersionBits   = 16u;
    static constexpr core::u32 kAlignmentBits =  7u;
    // EC55 — alignment_class is log2(alignment_bytes). With 7 bits we
    // can encode alignments up to 2^127 bytes in theory, but in
    // practice the meaningful upper bound is the platform's largest
    // architectural alignment: 128B cache line (log2 = 7). Anything
    // above 128B (e.g. Huge Pages: 2 MiB = log2 21, 1 GiB = log2 30)
    // is a virtual-memory mapping concern handled OUTSIDE the GenID
    // layout (the arena's `alignas` directive on `die_`). The bit
    // budget covers up to 128 distinct log2 values which is more than
    // enough for the [1, 128] byte range the field actually represents.
    //
    // FUTURE WORK (P4): if and when sub-slot Huge-Page alignments
    // become a thing (e.g. an arena backed by hugepage-mmap with
    // multiple 2MB regions per arena), kAlignmentBits would need to
    // grow to ≥ 21. That would force a corresponding shrink in
    // another GenID field — the natural candidate is kVersionBits
    // (16 → 11, still 2048 generations per slot, more than enough
    // given the slot-lifecycle pattern). Until then, the constraint
    // is encoded as a static_assert that we never accept an alignment
    // exceeding what alignment_class can represent.
    static constexpr core::u32 kQuantBits     =  3u;
    static constexpr core::u32 kStateBits     =  2u;
    // EC110 (Phase 2 Block C task 4.1): the 12 previously-reserved bits
    // are split into a 5-bit sub-slot index (32 sub-cells per slot) and
    // a 7-bit metadata field (128 user-defined tags, e.g. prompt type).
    // Backward compatibility: the default value 0 in BOTH fields means
    // "no sub-slabbing, this GenID names the entire slot" — every GenID
    // built before this change has those bits at 0 by construction.
    static constexpr core::u32 kSubSlotBits   =  5u;
    static constexpr core::u32 kMetadataBits  =  7u;
    static constexpr core::u32 kReservedBits  = kSubSlotBits + kMetadataBits;

    static_assert(kIdBits + kVersionBits + kAlignmentBits +
                  kQuantBits + kStateBits + kReservedBits == 64u,
                  "GenID bit layout must sum to exactly 64");
    static_assert(kReservedBits == 12u,
                  "Reserved-bit budget MUST stay at 12 for ABI stability");

    //=== Bit offsets ===
    static constexpr core::u32 kIdShift        = 0u;
    static constexpr core::u32 kVersionShift   = kIdShift + kIdBits;
    static constexpr core::u32 kAlignmentShift = kVersionShift + kVersionBits;
    static constexpr core::u32 kQuantShift     = kAlignmentShift + kAlignmentBits;
    static constexpr core::u32 kStateShift     = kQuantShift + kQuantBits;
    static constexpr core::u32 kSubSlotShift   = kStateShift + kStateBits;
    static constexpr core::u32 kMetadataShift  = kSubSlotShift + kSubSlotBits;
    // Legacy alias retained for any code still computing offsets from
    // the start of the reserved region (e.g. test harnesses).
    static constexpr core::u32 kReservedShift  = kSubSlotShift;

    //===Maximum field values ​​===
    static constexpr core::u32 kMaxId         = (1u << kIdBits) - 1u;
    static constexpr core::u32 kMaxVersion    = (1u << kVersionBits) - 1u;
    static constexpr core::u32 kMaxAlignment  = (1u << kAlignmentBits) - 1u;
    static constexpr core::u32 kMaxSubSlot    = (1u << kSubSlotBits) - 1u;   //  31
    static constexpr core::u32 kMaxMetadata   = (1u << kMetadataBits) - 1u;  // 127

    //===Field masks (for extraction) ===
    static constexpr core::u64 kIdMask        = (core::u64{1} << kIdBits) - 1u;
    static constexpr core::u64 kVersionMask   = (core::u64{1} << kVersionBits) - 1u;
    static constexpr core::u64 kAlignmentMask = (core::u64{1} << kAlignmentBits) - 1u;
    static constexpr core::u64 kQuantMask     = (core::u64{1} << kQuantBits) - 1u;
    static constexpr core::u64 kStateMask     = (core::u64{1} << kStateBits) - 1u;
    static constexpr core::u64 kSubSlotMask   = (core::u64{1} << kSubSlotBits) - 1u;
    static constexpr core::u64 kMetadataMask  = (core::u64{1} << kMetadataBits) - 1u;

    //=== Constructors ===

    /// Default constructor: creates null-GenID (state == kFree, all zeros).
    constexpr GenID() noexcept = default;

    /// Constructor from raw bits (for deserialization from C-ABI boundaries).
    [[nodiscard]] static constexpr GenID fromRaw(core::u64 raw) noexcept {
        GenID g;
        g.bits_ = raw;
        return g;
    }

    /// Builder: creates a fully configured GenID.
    ///HA: If any argument exceeds its range, returns null-GenID
    /// (fail-fast in caller via isNull()/state()).
    [[nodiscard]] static constexpr GenID make(
        core::u32  id,
        core::u32  version,
        core::u32  alignment_class,
        QuantFormat quant,
        SlotState  state) noexcept
    {
        if (id > kMaxId ||
            version > kMaxVersion ||
            alignment_class > kMaxAlignment) {
            return GenID{};  // null, fail-fast
        }
        GenID g;
        g.bits_  = (static_cast<core::u64>(id)              & kIdMask)        << kIdShift;
        g.bits_ |= (static_cast<core::u64>(version)         & kVersionMask)   << kVersionShift;
        g.bits_ |= (static_cast<core::u64>(alignment_class) & kAlignmentMask) << kAlignmentShift;
        g.bits_ |= (static_cast<core::u64>(quant)           & kQuantMask)     << kQuantShift;
        g.bits_ |= (static_cast<core::u64>(state)           & kStateMask)     << kStateShift;
        return g;
    }

    //=== Accessors (constexpr, noexcept, [[nodiscard]]) ===

    [[nodiscard]] constexpr core::u32 id() const noexcept {
        return static_cast<core::u32>((bits_ >> kIdShift) & kIdMask);
    }

    [[nodiscard]] constexpr core::u32 version() const noexcept {
        return static_cast<core::u32>((bits_ >> kVersionShift) & kVersionMask);
    }

    [[nodiscard]] constexpr core::u32 alignmentClass() const noexcept {
        return static_cast<core::u32>((bits_ >> kAlignmentShift) & kAlignmentMask);
    }

    [[nodiscard]] constexpr QuantFormat quantFormat() const noexcept {
        return static_cast<QuantFormat>((bits_ >> kQuantShift) & kQuantMask);
    }

    [[nodiscard]] constexpr SlotState state() const noexcept {
        return static_cast<SlotState>((bits_ >> kStateShift) & kStateMask);
    }

    /// EC110: sub-slot index within a sub-slabbed slot, [0..31]. A value
    /// of 0 with `metadata() == 0` is indistinguishable from a plain
    /// (non-sub-slabbed) GenID — this is intentional, so old code paths
    /// see `subSlotIndex() == 0` for every legacy handle and treat it
    /// as "the whole slot".
    [[nodiscard]] constexpr core::u32 subSlotIndex() const noexcept {
        return static_cast<core::u32>((bits_ >> kSubSlotShift) & kSubSlotMask);
    }

    /// EC110: opaque metadata tag set by the bridge, [0..127]. Not
    /// interpreted by FractalArena; the tiered bridge uses this to mark
    /// sub-slabbed payload kinds (e.g. "system prompt" vs "dormant
    /// agent KV"). Backward-compat default is 0.
    [[nodiscard]] constexpr core::u32 metadata() const noexcept {
        return static_cast<core::u32>((bits_ >> kMetadataShift) & kMetadataMask);
    }

    [[nodiscard]] constexpr core::u64 raw() const noexcept {
        return bits_;
    }

    //=== Immutable modifiers (return new GenID) ===

    [[nodiscard]] constexpr GenID withState(SlotState s) const noexcept {
        GenID g = *this;
        g.bits_ &= ~(kStateMask << kStateShift);
        g.bits_ |=  (static_cast<core::u64>(s) & kStateMask) << kStateShift;
        return g;
    }

    [[nodiscard]] constexpr GenID withQuantFormat(QuantFormat q) const noexcept {
        GenID g = *this;
        g.bits_ &= ~(kQuantMask << kQuantShift);
        g.bits_ |=  (static_cast<core::u64>(q) & kQuantMask) << kQuantShift;
        return g;
    }

    [[nodiscard]] constexpr GenID withAlignmentClass(core::u32 a) const noexcept {
        if (a > kMaxAlignment) return *this;  //fail-safe, don't touch
        GenID g = *this;
        g.bits_ &= ~(kAlignmentMask << kAlignmentShift);
        g.bits_ |=  (static_cast<core::u64>(a) & kAlignmentMask) << kAlignmentShift;
        return g;
    }

    /// EC110: set the sub-slot index. Out-of-range values are rejected
    /// (fail-safe: return the original handle unchanged), so a bug in
    /// the bridge that asks for sub-slot 99 cannot poison adjacent bits
    /// (e.g. silently flip the high bit of state).
    ///
    /// EC105 — Caller contract. Returning the original handle on
    /// out-of-range is a fail-CLOSED behaviour (no bit poisoning),
    /// but it is ALSO fail-SILENT: a caller who blindly trusts the
    /// returned GenID will write to sub-slot 0 of the ORIGINAL slot
    /// (because that is where the unchanged handle still points),
    /// not detect the bug. To close this hole, every caller of
    /// withSubSlot MUST validate the index against
    /// FractalArena::subSlotCount(sub_slot_size) BEFORE the call
    /// (TieredBridge does this in EC8). The contract is "no caller
    /// passes out-of-range; library catches the violation as a
    /// no-op so a buggy caller cannot corrupt other tenants".
    /// The static_assert at the bottom of this file proves the
    /// no-op behaviour on the rare round-trip path.
    [[nodiscard]] constexpr GenID withSubSlot(core::u32 idx) const noexcept {
        if (idx > kMaxSubSlot) return *this;
        GenID g = *this;
        g.bits_ &= ~(kSubSlotMask << kSubSlotShift);
        g.bits_ |=  (static_cast<core::u64>(idx) & kSubSlotMask) << kSubSlotShift;
        return g;
    }

    [[nodiscard]] constexpr GenID withMetadata(core::u32 m) const noexcept {
        if (m > kMaxMetadata) return *this;
        GenID g = *this;
        g.bits_ &= ~(kMetadataMask << kMetadataShift);
        g.bits_ |=  (static_cast<core::u64>(m) & kMetadataMask) << kMetadataShift;
        return g;
    }

    /// Generation increment. Wrap-around is protected by a mask (after 65535 → 0).
    /// HA: this is semantically valid - wrap means "new life for the slot".
    [[nodiscard]] constexpr GenID bumpVersion() const noexcept {
        const core::u32 v = (version() + 1u) & kMaxVersion;
        GenID g = *this;
        g.bits_ &= ~(kVersionMask << kVersionShift);
        g.bits_ |=  (static_cast<core::u64>(v) & kVersionMask) << kVersionShift;
        return g;
    }

    //=== Predicates ===

    /// Null-GenID: all bits are zero, state == kFree.
    /// Used as sentinel for "invalid/non-existent".
    [[nodiscard]] constexpr bool isNull() const noexcept {
        return bits_ == 0u;
    }

    [[nodiscard]] constexpr bool isActive() const noexcept {
        return state() == SlotState::kActive;
    }

    [[nodiscard]] constexpr bool isDormant() const noexcept {
        const auto s = state();
        return s == SlotState::kDormant || s == SlotState::kQuantized;
    }

    //=== Comparison (defaulted, completely ordered) ===
    [[nodiscard]] constexpr bool operator==(const GenID&) const noexcept = default;
    [[nodiscard]] constexpr auto operator<=>(const GenID&) const noexcept = default;

private:
    core::u64 bits_ = 0u;
};

//=== HA: fundamental type invariants (compile-time) ===
static_assert(sizeof(GenID) == 8,
              "GenID MUST be exactly 8 bytes (cache-line packing depends on it)");
static_assert(alignof(GenID) == 8,
              "GenID MUST be 8-byte aligned (atomic operations rely on it)");
static_assert(std::is_trivially_copyable_v<GenID>,
              "GenID MUST be trivially copyable (FFI boundary requirement)");
static_assert(std::is_standard_layout_v<GenID>,
              "GenID MUST be standard layout (memcpy-safe)");
static_assert(std::is_trivially_destructible_v<GenID>,
              "GenID MUST be trivially destructible (no RAII overhead)");
static_assert(core::ArenaStorable<GenID>,
              "GenID MUST satisfy ArenaStorable concept");

//=== HA: value invariants (compile-time) ===
static_assert(GenID{}.isNull(),
              "Default-constructed GenID must be null");
static_assert(GenID{}.state() == SlotState::kFree,
              "Default-constructed GenID must have state=kFree");
static_assert(GenID::kMaxId == 0xFF'FF'FFu);
static_assert(GenID::kMaxVersion == 0xFF'FFu);
static_assert(GenID::kMaxAlignment == 127u,
              "alignment_class must cover 128B cache line offset");

//=== HA: round-trip encoding (compile-time) ===
namespace detail {
inline constexpr auto kRoundTripTest = GenID::make(
    /*id=*/12345u,
    /*version=*/42u,
    /*alignment_class=*/64u,
    QuantFormat::kBF16,
    SlotState::kActive);
}  // namespace detail

static_assert(detail::kRoundTripTest.id() == 12345u);
static_assert(detail::kRoundTripTest.version() == 42u);
static_assert(detail::kRoundTripTest.alignmentClass() == 64u);
static_assert(detail::kRoundTripTest.quantFormat() == QuantFormat::kBF16);
static_assert(detail::kRoundTripTest.state() == SlotState::kActive);
static_assert(detail::kRoundTripTest.isActive());
static_assert(!detail::kRoundTripTest.isNull());

//=== HA: modifiers do not damage adjacent fields (compile-time) ===
namespace detail {
inline constexpr auto kModifierTest =
    detail::kRoundTripTest.withState(SlotState::kQuantized);
}

static_assert(detail::kModifierTest.id() == 12345u);
static_assert(detail::kModifierTest.version() == 42u);
static_assert(detail::kModifierTest.alignmentClass() == 64u);
static_assert(detail::kModifierTest.state() == SlotState::kQuantized);
static_assert(detail::kModifierTest.isDormant());

//=== HA: bumpVersion wraps around correctly ===
static_assert(
    GenID::make(0u, GenID::kMaxVersion, 0u, QuantFormat::kFP32, SlotState::kActive)
        .bumpVersion().version() == 0u,
    "bumpVersion must wrap from kMaxVersion to 0");

//=== HA: edge cases of make() - out-of-range → null ===
static_assert(
    GenID::make(GenID::kMaxId + 1u, 0u, 0u, QuantFormat::kFP32, SlotState::kActive)
        .isNull(),
    "make() with out-of-range id must return null GenID");

//=== EC110: sub-slot and metadata invariants (compile-time) ===

// Width budget unchanged: 5 + 7 = 12 = kReservedBits.
static_assert(GenID::kSubSlotBits + GenID::kMetadataBits == GenID::kReservedBits,
              "sub_slot + metadata must fully cover the 12 reserved bits");
static_assert(GenID::kMaxSubSlot  == 31u);
static_assert(GenID::kMaxMetadata == 127u);

// Default (legacy) GenID still reports sub_slot == 0 and metadata == 0.
// This is the backward-compatibility contract: any code written before
// the split observes the same value it would have read from a 12-bit
// reserved zero.
namespace detail {
inline constexpr auto kLegacyDefault = GenID::make(
    /*id=*/0u, /*version=*/1u, /*alignment_class=*/0u,
    QuantFormat::kFP32, SlotState::kActive);
}  // namespace detail
static_assert(detail::kLegacyDefault.subSlotIndex() == 0u);
static_assert(detail::kLegacyDefault.metadata()     == 0u);

// withSubSlot / withMetadata round-trip preserves all other fields.
namespace detail {
inline constexpr auto kSubSlotTest =
    kRoundTripTest.withSubSlot(17u).withMetadata(99u);
}  // namespace detail
static_assert(detail::kSubSlotTest.id()             == 12345u);
static_assert(detail::kSubSlotTest.version()        == 42u);
static_assert(detail::kSubSlotTest.alignmentClass() == 64u);
static_assert(detail::kSubSlotTest.quantFormat()    == QuantFormat::kBF16);
static_assert(detail::kSubSlotTest.state()          == SlotState::kActive);
static_assert(detail::kSubSlotTest.subSlotIndex()   == 17u);
static_assert(detail::kSubSlotTest.metadata()       == 99u);

// Out-of-range sub-slot / metadata: handle returned unchanged.
static_assert(
    detail::kRoundTripTest.withSubSlot(99u).subSlotIndex() == 0u,
    "withSubSlot(out-of-range) must fail-safe to original");
static_assert(
    detail::kRoundTripTest.withMetadata(200u).metadata() == 0u,
    "withMetadata(out-of-range) must fail-safe to original");

// EC110 cross-field non-interference: setting sub_slot does not touch
// state, and setting state does not touch sub_slot. This is the bug we
// most fear from the bit-shuffling rewrite: a setter that flips an
// adjacent field would silently corrupt every handle.
static_assert(
    detail::kSubSlotTest.withState(SlotState::kQuantized).subSlotIndex() == 17u);
static_assert(
    detail::kSubSlotTest.withState(SlotState::kQuantized).metadata() == 99u);
static_assert(
    detail::kSubSlotTest.withSubSlot(0u).state() == SlotState::kActive);
static_assert(
    detail::kSubSlotTest.withMetadata(0u).state() == SlotState::kActive);

}  // namespace qtx::arena
