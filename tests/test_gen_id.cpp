// ============================================================================
// @file        test_gen_id.cpp
//@brief Unit tests for GenID: layout, round-trip, edge cases.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================

#include "test_harness.hpp"
#include "qtx/arena/gen_id.hpp"

#include <type_traits>

using namespace qtx::arena;
using qtx::core::u32;
using qtx::core::u64;

// ============================================================================
// Type traits
// ============================================================================

QTX_TEST(GenID, IsTriviallyCopyable) {
    QTX_EXPECT(std::is_trivially_copyable_v<GenID>);
}

QTX_TEST(GenID, IsStandardLayout) {
    QTX_EXPECT(std::is_standard_layout_v<GenID>);
}

QTX_TEST(GenID, ExactSize) {
    QTX_EXPECT_EQ(sizeof(GenID), 8u);
}

QTX_TEST(GenID, ExactAlignment) {
    QTX_EXPECT_EQ(alignof(GenID), 8u);
}

// ============================================================================
// Null/default semantics
// ============================================================================

QTX_TEST(GenID, DefaultIsNull) {
    GenID g;
    QTX_EXPECT(g.isNull());
    QTX_EXPECT_EQ(g.raw(), 0u);
}

QTX_TEST(GenID, DefaultStateIsFree) {
    GenID g;
    QTX_EXPECT(g.state() == SlotState::kFree);
}

QTX_TEST(GenID, DefaultNotActive) {
    GenID g;
    QTX_EXPECT(!g.isActive());
    QTX_EXPECT(!g.isDormant());
}

// ============================================================================
//Round-trip encoding of all fields
// ============================================================================

QTX_TEST(GenID, RoundTripBasic) {
    auto g = GenID::make(/*id=*/0xABCDEFu, /*ver=*/0x1234u,
                         /*align=*/64u, QuantFormat::kBF16,
                         SlotState::kActive);
    QTX_EXPECT_EQ(g.id(), 0xABCDEFu);
    QTX_EXPECT_EQ(g.version(), 0x1234u);
    QTX_EXPECT_EQ(g.alignmentClass(), 64u);
    QTX_EXPECT(g.quantFormat() == QuantFormat::kBF16);
    QTX_EXPECT(g.state() == SlotState::kActive);
}

QTX_TEST(GenID, RoundTripMaxValues) {
    auto g = GenID::make(GenID::kMaxId, GenID::kMaxVersion,
                         GenID::kMaxAlignment,
                         QuantFormat::kReserved, SlotState::kQuantized);
    QTX_EXPECT_EQ(g.id(), GenID::kMaxId);
    QTX_EXPECT_EQ(g.version(), GenID::kMaxVersion);
    QTX_EXPECT_EQ(g.alignmentClass(), GenID::kMaxAlignment);
    QTX_EXPECT(g.quantFormat() == QuantFormat::kReserved);
    QTX_EXPECT(g.state() == SlotState::kQuantized);
}

QTX_TEST(GenID, RoundTripZeroValues) {
    auto g = GenID::make(0u, 0u, 0u, QuantFormat::kFP32, SlotState::kActive);
    QTX_EXPECT_EQ(g.id(), 0u);
    QTX_EXPECT_EQ(g.version(), 0u);
    QTX_EXPECT_EQ(g.alignmentClass(), 0u);
    QTX_EXPECT(g.quantFormat() == QuantFormat::kFP32);
    QTX_EXPECT(g.state() == SlotState::kActive);
    QTX_EXPECT(!g.isNull());  //state=kActive => not null
}

// ============================================================================
// Boundary validation: out-of-range → null
// ============================================================================

QTX_TEST(GenID, OutOfRangeIdReturnsNull) {
    auto g = GenID::make(GenID::kMaxId + 1u, 0u, 0u,
                         QuantFormat::kFP32, SlotState::kActive);
    QTX_EXPECT(g.isNull());
}

QTX_TEST(GenID, OutOfRangeVersionReturnsNull) {
    auto g = GenID::make(0u, GenID::kMaxVersion + 1u, 0u,
                         QuantFormat::kFP32, SlotState::kActive);
    QTX_EXPECT(g.isNull());
}

QTX_TEST(GenID, OutOfRangeAlignmentReturnsNull) {
    auto g = GenID::make(0u, 0u, GenID::kMaxAlignment + 1u,
                         QuantFormat::kFP32, SlotState::kActive);
    QTX_EXPECT(g.isNull());
}

// ============================================================================
//Immutable modifiers (with-of-the-form withX)
// ============================================================================

QTX_TEST(GenID, WithStateDoesNotCorruptOtherFields) {
    auto g0 = GenID::make(42u, 7u, 32u, QuantFormat::kFP16, SlotState::kActive);
    auto g1 = g0.withState(SlotState::kDormant);

    QTX_EXPECT_EQ(g1.id(), 42u);
    QTX_EXPECT_EQ(g1.version(), 7u);
    QTX_EXPECT_EQ(g1.alignmentClass(), 32u);
    QTX_EXPECT(g1.quantFormat() == QuantFormat::kFP16);
    QTX_EXPECT(g1.state() == SlotState::kDormant);
    QTX_EXPECT(g1.isDormant());
}

QTX_TEST(GenID, WithQuantFormatDoesNotCorruptOtherFields) {
    auto g0 = GenID::make(99u, 13u, 16u, QuantFormat::kFP32, SlotState::kActive);
    auto g1 = g0.withQuantFormat(QuantFormat::kINT4);

    QTX_EXPECT_EQ(g1.id(), 99u);
    QTX_EXPECT_EQ(g1.version(), 13u);
    QTX_EXPECT_EQ(g1.alignmentClass(), 16u);
    QTX_EXPECT(g1.quantFormat() == QuantFormat::kINT4);
    QTX_EXPECT(g1.state() == SlotState::kActive);
}

QTX_TEST(GenID, BumpVersionIncrementsByOne) {
    auto g0 = GenID::make(5u, 100u, 0u, QuantFormat::kFP32, SlotState::kActive);
    auto g1 = g0.bumpVersion();
    QTX_EXPECT_EQ(g1.version(), 101u);
    QTX_EXPECT_EQ(g1.id(), 5u);  //other fields are not touched
}

QTX_TEST(GenID, BumpVersionWrapsAtMax) {
    auto g0 = GenID::make(0u, GenID::kMaxVersion, 0u,
                          QuantFormat::kFP32, SlotState::kActive);
    auto g1 = g0.bumpVersion();
    QTX_EXPECT_EQ(g1.version(), 0u);  // wrap-around
    QTX_EXPECT(g1.state() == SlotState::kActive);  //state is intact
}

// ============================================================================
// Equality / ordering
// ============================================================================

QTX_TEST(GenID, EqualityBasic) {
    auto a = GenID::make(1u, 2u, 3u, QuantFormat::kBF16, SlotState::kActive);
    auto b = GenID::make(1u, 2u, 3u, QuantFormat::kBF16, SlotState::kActive);
    QTX_EXPECT(a == b);
}

QTX_TEST(GenID, EqualityDifferentVersion) {
    auto a = GenID::make(1u, 2u, 3u, QuantFormat::kBF16, SlotState::kActive);
    auto b = GenID::make(1u, 3u, 3u, QuantFormat::kBF16, SlotState::kActive);
    QTX_EXPECT(a != b);
}

QTX_TEST(GenID, FromRawRoundTrip) {
    auto original = GenID::make(0x123u, 0xCAFEu, 100u,
                                QuantFormat::kFP8_E4M3, SlotState::kDormant);
    auto copied = GenID::fromRaw(original.raw());
    QTX_EXPECT(original == copied);
}

// ============================================================================
//Bit-field isolation: verify no overlap
// ============================================================================

QTX_TEST(GenID, BitFieldIsolation) {
    //Set only id, leave everything else as zero.
    auto only_id = GenID::make(GenID::kMaxId, 0u, 0u,
                               QuantFormat::kFP32, SlotState::kActive);
    //version and alignment must be zero, regardless of max id.
    QTX_EXPECT_EQ(only_id.version(), 0u);
    QTX_EXPECT_EQ(only_id.alignmentClass(), 0u);

    //Set only version, id should be 0.
    auto only_ver = GenID::make(0u, GenID::kMaxVersion, 0u,
                                QuantFormat::kFP32, SlotState::kActive);
    QTX_EXPECT_EQ(only_ver.id(), 0u);
    QTX_EXPECT_EQ(only_ver.alignmentClass(), 0u);

    //Likewise for alignment.
    auto only_align = GenID::make(0u, 0u, GenID::kMaxAlignment,
                                  QuantFormat::kFP32, SlotState::kActive);
    QTX_EXPECT_EQ(only_align.id(), 0u);
    QTX_EXPECT_EQ(only_align.version(), 0u);
}

// ============================================================================
// EC110 — sub_slot + metadata bit splits (Phase 2 Block C task 4.1)
// ============================================================================

QTX_TEST(GenID, EC110_LegacyDefaultStillReadsAsZero) {
    // A GenID built by the pre-Block-C make() must read 0 for both new
    // accessors. This is the backward-compat contract.
    auto g = GenID::make(/*id=*/42u, /*version=*/3u, /*alignment=*/0u,
                         QuantFormat::kFP32, SlotState::kActive);
    QTX_EXPECT_EQ(g.subSlotIndex(), 0u);
    QTX_EXPECT_EQ(g.metadata(),     0u);
    // And it must still pass isActive() / isNull() etc. unchanged.
    QTX_EXPECT(g.isActive());
    QTX_EXPECT(!g.isNull());
}

QTX_TEST(GenID, EC110_SubSlotRoundTrip) {
    auto g = GenID::make(7u, 5u, 32u, QuantFormat::kINT8, SlotState::kActive);
    for (u32 i = 0; i <= GenID::kMaxSubSlot; ++i) {
        auto h = g.withSubSlot(i);
        QTX_EXPECT_EQ(h.subSlotIndex(), i);
        // Other fields must not move.
        QTX_EXPECT_EQ(h.id(),             7u);
        QTX_EXPECT_EQ(h.version(),        5u);
        QTX_EXPECT_EQ(h.alignmentClass(), 32u);
        QTX_EXPECT(h.quantFormat() == QuantFormat::kINT8);
        QTX_EXPECT(h.state()       == SlotState::kActive);
        QTX_EXPECT_EQ(h.metadata(), 0u);
    }
}

QTX_TEST(GenID, EC110_MetadataRoundTrip) {
    auto g = GenID::make(11u, 9u, 16u, QuantFormat::kBF16, SlotState::kDormant);
    for (u32 m : {0u, 1u, 63u, 64u, 127u}) {
        auto h = g.withMetadata(m);
        QTX_EXPECT_EQ(h.metadata(), m);
        // Other fields must not move.
        QTX_EXPECT_EQ(h.id(),             11u);
        QTX_EXPECT_EQ(h.version(),        9u);
        QTX_EXPECT_EQ(h.alignmentClass(), 16u);
        QTX_EXPECT(h.quantFormat() == QuantFormat::kBF16);
        QTX_EXPECT(h.state()       == SlotState::kDormant);
        QTX_EXPECT_EQ(h.subSlotIndex(), 0u);
    }
}

QTX_TEST(GenID, EC110_OutOfRangeFailSafe) {
    // The cardinal sin we're guarding against: a buggy bridge calls
    // withSubSlot(99). The bit slot only holds [0..31], so writing 99
    // would either truncate to 99 & 31 == 3 (data corruption) or spill
    // into the metadata field (cross-field corruption). Our setter
    // returns the original handle unchanged in this case.
    auto g = GenID::make(1u, 1u, 0u, QuantFormat::kFP32, SlotState::kActive)
                .withSubSlot(7u).withMetadata(42u);

    auto bad_sub = g.withSubSlot(99u);   // out of range, must no-op
    QTX_EXPECT_EQ(bad_sub.subSlotIndex(), 7u);     // unchanged
    QTX_EXPECT_EQ(bad_sub.metadata(),     42u);    // untouched

    auto bad_meta = g.withMetadata(200u);  // out of range
    QTX_EXPECT_EQ(bad_meta.subSlotIndex(), 7u);    // untouched
    QTX_EXPECT_EQ(bad_meta.metadata(),     42u);   // unchanged
}

QTX_TEST(GenID, EC110_NoFieldOverlap) {
    // Stress: set every accessor to its max value, then verify each
    // accessor still reads back its max. If any two fields overlapped
    // even by one bit, this would fail with a truncation.
    auto g = GenID::make(GenID::kMaxId,
                         GenID::kMaxVersion,
                         GenID::kMaxAlignment,
                         QuantFormat::kFP8_E5M2,  // 6, fits 3 bits
                         SlotState::kQuantized);
    auto h = g.withSubSlot(GenID::kMaxSubSlot)
              .withMetadata(GenID::kMaxMetadata);
    QTX_EXPECT_EQ(h.id(),             GenID::kMaxId);
    QTX_EXPECT_EQ(h.version(),        GenID::kMaxVersion);
    QTX_EXPECT_EQ(h.alignmentClass(), GenID::kMaxAlignment);
    QTX_EXPECT(h.quantFormat() == QuantFormat::kFP8_E5M2);
    QTX_EXPECT(h.state()       == SlotState::kQuantized);
    QTX_EXPECT_EQ(h.subSlotIndex(),   GenID::kMaxSubSlot);
    QTX_EXPECT_EQ(h.metadata(),       GenID::kMaxMetadata);
}
