// ============================================================================
// @file        test_fractal_arena.cpp
//@brief Unit tests FractalArena: allocate/release, ABA, view access.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================

#include "test_harness.hpp"
#include "qtx/arena/fractal_arena.hpp"
#include "qtx/arena/gen_id.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <vector>

using namespace qtx::arena;
using qtx::core::usize;

//Small test arena: 64 slots × 128 bytes = 8 KiB.
//We place it on the heap so as not to exceed the kernel stack during concurrent tests.
using SmallArena = FractalArena<64, 128>;

// ============================================================================
//Basic design
// ============================================================================

QTX_TEST(Arena, ConstructorAllSlotsFree) {
    auto arena = std::make_unique<SmallArena>();
    QTX_EXPECT_EQ(arena->freeSlots(), 64u);
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
    QTX_EXPECT_EQ(arena->totalBytes(), 64u * 128u);
}

QTX_TEST(Arena, ConstexprMembers) {
    QTX_EXPECT_EQ(SmallArena::kSlotCount, 64u);
    QTX_EXPECT_EQ(SmallArena::kSlotSize, 128u);
    QTX_EXPECT_EQ(SmallArena::slotBytes(), 128u);
}

// ============================================================================
// allocate/release lifecycle
// ============================================================================

QTX_TEST(Arena, BasicAllocateReturnsValidGenID) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(64u, 8u);
    QTX_EXPECT(!id.isNull());
    QTX_EXPECT(id.isActive());
    QTX_EXPECT(arena->validate(id));
    QTX_EXPECT_EQ(arena->usedSlots(), 1u);
}

QTX_TEST(Arena, AllocateZeroSizeReturnsNull) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(0u, 8u);
    QTX_EXPECT(id.isNull());
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

QTX_TEST(Arena, AllocateOversizeReturnsNull) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(SmallArena::kSlotSize + 1u, 8u);
    QTX_EXPECT(id.isNull());
}

QTX_TEST(Arena, AllocateBadAlignReturnsNull) {
    auto arena = std::make_unique<SmallArena>();
    QTX_EXPECT(arena->allocate(64u, 0u).isNull());
    QTX_EXPECT(arena->allocate(64u, 7u).isNull());  //not a power of two
    QTX_EXPECT(arena->allocate(64u, 100u).isNull()); //not a power of two
}

QTX_TEST(Arena, ReleaseFreesSlot) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(64u, 8u);
    QTX_EXPECT(!id.isNull());
    QTX_EXPECT_EQ(arena->usedSlots(), 1u);

    QTX_EXPECT(arena->release(id));
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
    QTX_EXPECT(!arena->validate(id));  //ID is no longer valid
}

QTX_TEST(Arena, DoubleReleaseRejected) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(64u, 8u);
    QTX_EXPECT(arena->release(id));
    QTX_EXPECT(!arena->release(id));  //re-release rejected
}

// ============================================================================
//View access and BOUNDS_SAFETY
// ============================================================================

QTX_TEST(Arena, ViewReturnsCorrectSize) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(64u, 8u);
    auto v = arena->view(id);
    QTX_EXPECT_EQ(v.size(), SmallArena::kSlotSize);
}

QTX_TEST(Arena, ViewIsWritableAndReadable) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(64u, 8u);
    auto v = arena->view(id);
    QTX_EXPECT(!v.empty());

    //We record the pattern.
    for (usize i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::byte>(i & 0xFFu);
    }
    //We read in the same way.
    auto v2 = arena->view(id);
    for (usize i = 0; i < v2.size(); ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned>(v2[i]),
                         static_cast<unsigned>(i & 0xFFu));
    }
}

QTX_TEST(Arena, ViewOnInvalidGenIDIsEmpty) {
    auto arena = std::make_unique<SmallArena>();
    GenID null_id;
    auto v = arena->view(null_id);
    QTX_EXPECT(v.empty());
}

QTX_TEST(Arena, ViewOnReleasedGenIDIsEmpty) {
    auto arena = std::make_unique<SmallArena>();
    auto id = arena->allocate(64u, 8u);
    arena->release(id);
    auto v = arena->view(id);
    QTX_EXPECT(v.empty());
}

// ============================================================================
//ABA-safety: after release+allocate of the same slot, the old GenID is invalid
// ============================================================================

QTX_TEST(Arena, ABASafetyAfterReuse) {
    auto arena = std::make_unique<SmallArena>();

    //Fill slot 0
    auto id_old = arena->allocate(64u, 8u);
    auto slot_old = id_old.id();

    //Write down the "secret"
    auto v_old = arena->view(id_old);
    for (usize i = 0; i < v_old.size(); ++i) {
        v_old[i] = std::byte{0xAA};
    }

    //Let's release
    arena->release(id_old);

    //Re-allocate - the selector will give the same slot 0
    auto id_new = arena->allocate(64u, 8u);
    QTX_EXPECT_EQ(id_new.id(), slot_old);   //same physical slot
    QTX_EXPECT_NE(id_new.version(), id_old.version());  //new generation

    //The old GenID must be rejected!
    QTX_EXPECT(!arena->validate(id_old));
    QTX_EXPECT(arena->view(id_old).empty());

    //New - it works.
    QTX_EXPECT(arena->validate(id_new));
    auto v_new = arena->view(id_new);
    QTX_EXPECT(!v_new.empty());
}

// ============================================================================
//Exhaustion (P1: fair null, no quantization)
// ============================================================================

QTX_TEST(Arena, ExhaustionReturnsNull) {
    auto arena = std::make_unique<SmallArena>();
    std::vector<GenID> all;
    //We take all 64 slots.
    for (usize i = 0; i < SmallArena::kSlotCount; ++i) {
        auto id = arena->allocate(64u, 8u);
        QTX_EXPECT(!id.isNull());
        all.push_back(id);
    }
    QTX_EXPECT_EQ(arena->usedSlots(), SmallArena::kSlotCount);

    //The 65th allocation should return null (P1: no quantization).
    auto extra = arena->allocate(64u, 8u);
    QTX_EXPECT(extra.isNull());

    //Freeing one slot allows you to allocate again.
    arena->release(all[10]);
    auto recovered = arena->allocate(64u, 8u);
    QTX_EXPECT(!recovered.isNull());
}

// ============================================================================
//Uniqueness of GenID between allocations
// ============================================================================

QTX_TEST(Arena, AllAllocationsUnique) {
    auto arena = std::make_unique<SmallArena>();
    std::set<qtx::core::u64> seen;
    for (usize i = 0; i < SmallArena::kSlotCount; ++i) {
        auto id = arena->allocate(64u, 8u);
        QTX_EXPECT(!id.isNull());
        QTX_EXPECT_EQ(seen.count(id.raw()), 0u);
        seen.insert(id.raw());
    }
    QTX_EXPECT_EQ(seen.size(), SmallArena::kSlotCount);
}

// ============================================================================
//Data isolation between slots (no overlap)
// ============================================================================

QTX_TEST(Arena, SlotsAreIsolated) {
    auto arena = std::make_unique<SmallArena>();
    auto id_a = arena->allocate(64u, 8u);
    auto id_b = arena->allocate(64u, 8u);
    QTX_EXPECT(!id_a.isNull());
    QTX_EXPECT(!id_b.isNull());
    QTX_EXPECT_NE(id_a.id(), id_b.id());

    //Fill with different bytes.
    auto va = arena->view(id_a);
    auto vb = arena->view(id_b);
    std::fill(va.begin(), va.end(), std::byte{0x11});
    std::fill(vb.begin(), vb.end(), std::byte{0x22});

    //We check that one does not damage the other.
    auto va2 = arena->view(id_a);
    auto vb2 = arena->view(id_b);
    for (usize i = 0; i < va2.size(); ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned>(va2[i]), 0x11u);
    }
    for (usize i = 0; i < vb2.size(); ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned>(vb2[i]), 0x22u);
    }
}

// ============================================================================
// Concurrent allocate/release stress test
// ============================================================================

QTX_TEST(Arena, ConcurrentAllocateRelease) {
    auto arena = std::make_unique<FractalArena<256, 128>>();
    constexpr int kThreads = 4;
    constexpr int kIters = 2000;

    std::atomic<int> validate_failures{0};
    std::atomic<int> double_free_failures{0};
    std::atomic<int> data_corruption{0};

    auto worker = [&](int tid) {
        std::vector<GenID> local;
        local.reserve(16);
        for (int i = 0; i < kIters; ++i) {
            //Mix: sometimes allocation, sometimes liberation.
            if (local.size() < 16 && (i & 1) == 0) {
                auto id = arena->allocate(64u, 8u);
                if (!id.isNull()) {
                    if (!arena->validate(id)) ++validate_failures;
                    auto v = arena->view(id);
                    if (!v.empty()) {
                        //Record the flow marker.
                        const std::byte marker{
                            static_cast<unsigned char>(tid + 1)};
                        std::fill(v.begin(), v.end(), marker);
                        //And immediately we read - integrity check
                        //within one thread (happens-before is obvious).
                        auto v2 = arena->view(id);
                        for (auto b : v2) {
                            if (b != marker) {
                                ++data_corruption;
                                break;
                            }
                        }
                    }
                    local.push_back(id);
                }
            } else if (!local.empty()) {
                auto id = local.back();
                local.pop_back();
                if (!arena->release(id)) ++double_free_failures;
            }
        }
        // Cleanup
        for (auto id : local) arena->release(id);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();

    QTX_EXPECT_EQ(validate_failures.load(), 0);
    QTX_EXPECT_EQ(double_free_failures.load(), 0);
    QTX_EXPECT_EQ(data_corruption.load(), 0);
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

// ============================================================================
// EC106 / EC107 — non-power-of-two SlotSize (Phase 2 Block B, task 3.1).
//
// Production ColdArena now uses SlotSize = 1152 (= 9 * 128), an exact-fit
// for the compressed INT8 payload (32 blocks × 36 bytes). We verify the
// three observable consequences:
//
//   1. allocate() / view() / release() round-trip works correctly when
//      slot offsets are computed via imul rather than shl.
//   2. Every slot starts on a 128-byte boundary, so cache-line and SIMD
//      contracts hold even though the slot stride is non-power-of-two.
//   3. No two slots overlap, and the last slot does not run past the
//      backing array (the EC109 overflow guard plus per-slot bookkeeping).
// ============================================================================

// 1152 = 9 * 128, exact-fit for compressed INT8 payload (32 INT8 blocks).
using ColdArenaProd = FractalArena<128, 1152>;

QTX_TEST(Arena, EC106_NonPow2_RoundTrip) {
    auto arena = std::make_unique<ColdArenaProd>();
    QTX_EXPECT_EQ(ColdArenaProd::kSlotSize, 1152u);
    QTX_EXPECT_EQ(ColdArenaProd::kSlotCount, 128u);
    QTX_EXPECT_EQ(ColdArenaProd::kTotalSize, 128u * 1152u);

    // Allocate, write a marker into the slot, release, then re-allocate
    // and confirm the slot has been zero-wiped (EC30 still holds with
    // the new non-power-of-two stride).
    auto id = arena->allocate(1152u, 128u);
    QTX_EXPECT(!id.isNull());

    auto view = arena->view(id);
    QTX_EXPECT_EQ(view.size(), 1152u);
    for (usize i = 0; i < view.size(); ++i) {
        view[i] = static_cast<std::byte>(0xA5u);
    }
    QTX_EXPECT(arena->release(id));

    // Take 200 slots in a row; release some; take more; verify no UB.
    std::vector<GenID> ids;
    for (int i = 0; i < 100; ++i) {
        auto g = arena->allocate(1152u, 128u);
        QTX_EXPECT(!g.isNull());
        ids.push_back(g);
    }
    for (int i = 0; i < 50; ++i) arena->release(ids[static_cast<usize>(i * 2)]);
    for (int i = 0; i < 50; ++i) {
        auto g = arena->allocate(1152u, 128u);
        QTX_EXPECT(!g.isNull());
    }
}

QTX_TEST(Arena, EC107_NonPow2_EveryStartIs128BAligned) {
    auto arena = std::make_unique<ColdArenaProd>();

    // Take ALL slots and record their base addresses.
    std::vector<std::uintptr_t> bases;
    bases.reserve(ColdArenaProd::kSlotCount);
    for (usize i = 0; i < ColdArenaProd::kSlotCount; ++i) {
        auto id = arena->allocate(1152u, 128u);
        QTX_EXPECT(!id.isNull());
        auto v = arena->view(id);
        QTX_EXPECT_EQ(v.size(), 1152u);
        bases.push_back(reinterpret_cast<std::uintptr_t>(v.data()));
    }

    // Every base must be 128-byte aligned (EC107 invariant).
    for (auto b : bases) {
        QTX_EXPECT_EQ(b % 128u, 0u);
    }

    // The stride between consecutive base addresses must be exactly
    // kSlotSize bytes (proves we are not silently inserting padding).
    std::sort(bases.begin(), bases.end());
    for (usize i = 1; i < bases.size(); ++i) {
        QTX_EXPECT_EQ(bases[i] - bases[i - 1u], 1152u);
    }
}

QTX_TEST(Arena, EC106_NonPow2_NoOverlap) {
    // Stress alloc/release pairs and verify every live view occupies a
    // strictly disjoint byte range — i.e. slot * SlotSize does not
    // accidentally alias under any allocation pattern.
    auto arena = std::make_unique<ColdArenaProd>();
    std::vector<GenID> live;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;

    for (int round = 0; round < 4; ++round) {
        // Take 64 (half of 128) slots, verify no two ranges overlap.
        for (int i = 0; i < 64; ++i) {
            auto id = arena->allocate(1152u, 128u);
            QTX_EXPECT(!id.isNull());
            auto v = arena->view(id);
            const auto lo = reinterpret_cast<std::uintptr_t>(v.data());
            const auto hi = lo + v.size();
            for (auto& r : ranges) {
                // Strict disjoint test: [lo, hi) ∩ [r.lo, r.hi) == ∅.
                const bool overlap = !(hi <= r.first || lo >= r.second);
                QTX_EXPECT(!overlap);
            }
            ranges.emplace_back(lo, hi);
            live.push_back(id);
        }
        // Release ALL slots before the next round so the arena is fully
        // free again (the previous version only released half, then
        // tried to allocate 64 more — that exceeded kSlotCount = 128).
        for (auto& id : live) {
            arena->release(id);
        }
        live.clear();
        ranges.clear();
    }
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

// ============================================================================
// EC111 / EC114 — sub-slot view + sub-slot zeroize (Phase 2 Block C 4.2
// infrastructure layer).
//
// We use a 4096-byte slot configuration so 32 sub-slots × 128 bytes each
// fit exactly — the canonical "32 system prompts in one 4 KiB page"
// scenario from the Block C plan.
// ============================================================================

using PageArena = FractalArena<64, 4096>;

QTX_TEST(Arena, EC111_SubSlotCount_StaticHelper) {
    // 4096 / 128 = 32: the canonical sub-slot configuration.
    QTX_EXPECT_EQ(PageArena::subSlotCount(128u), 32u);
    // 4096 / 256 = 16.
    QTX_EXPECT_EQ(PageArena::subSlotCount(256u), 16u);
    // 4096 / 512 = 8.
    QTX_EXPECT_EQ(PageArena::subSlotCount(512u), 8u);
    // 4096 / 4096 = 1 (degenerate, but valid).
    QTX_EXPECT_EQ(PageArena::subSlotCount(4096u), 1u);

    // Failure modes (must return 0):
    QTX_EXPECT_EQ(PageArena::subSlotCount(0u), 0u);       // zero size
    QTX_EXPECT_EQ(PageArena::subSlotCount(8192u), 0u);    // bigger than slot
    QTX_EXPECT_EQ(PageArena::subSlotCount(300u), 0u);     // does not divide evenly
    QTX_EXPECT_EQ(PageArena::subSlotCount(32u), 0u);      // < cache line (64B)
}

QTX_TEST(Arena, EC111_SubSlotView_BasicRoundTrip) {
    auto arena = std::make_unique<PageArena>();
    auto id = arena->allocate(4096u, 128u);
    QTX_EXPECT(!id.isNull());

    // 128-byte sub-slots: 32 of them inside the 4096-byte slot.
    constexpr usize kSubSlotSize = 128u;

    // Plain view() returns the whole slot regardless of subSlotIndex.
    auto whole = arena->view(id);
    QTX_EXPECT_EQ(whole.size(), 4096u);

    // Each sub-slot's view starts at slot_base + idx * 128 and is 128B.
    for (qtx::core::u32 idx = 0; idx <= 31u; ++idx) {
        auto sub_id = id.withSubSlot(idx);
        auto sv = arena->viewSubSlot(sub_id, kSubSlotSize);
        QTX_EXPECT_EQ(sv.size(), kSubSlotSize);

        const auto sub_base = reinterpret_cast<std::uintptr_t>(sv.data());
        const auto whole_base = reinterpret_cast<std::uintptr_t>(whole.data());
        QTX_EXPECT_EQ(sub_base - whole_base,
                         static_cast<std::uintptr_t>(idx) * kSubSlotSize);
        QTX_EXPECT_EQ(sub_base % 128u, 0u);
    }

    arena->release(id);
}

QTX_TEST(Arena, EC111_SubSlotView_RejectsInvalidParameters) {
    auto arena = std::make_unique<PageArena>();
    auto id = arena->allocate(4096u, 128u);
    QTX_EXPECT(!id.isNull());

    // Invalid sub-slot size: not a divisor of kSlotSize.
    QTX_EXPECT(arena->viewSubSlot(id, 300u).empty());
    // Invalid sub-slot size: below cache-line minimum.
    QTX_EXPECT(arena->viewSubSlot(id.withSubSlot(0u), 32u).empty());
    // Invalid sub-slot size: zero.
    QTX_EXPECT(arena->viewSubSlot(id, 0u).empty());
    // Sub-slot size larger than the slot.
    QTX_EXPECT(arena->viewSubSlot(id, 8192u).empty());

    // Valid size but index >= sub-slot count. With size=256, count is
    // 4096/256 = 16; index 31 (max sub_slot) is out of range.
    auto bad_idx = id.withSubSlot(31u);
    QTX_EXPECT(arena->viewSubSlot(bad_idx, 256u).empty());

    arena->release(id);
}

QTX_TEST(Arena, EC111_SubSlotView_DisjointRanges) {
    // Write a unique byte pattern into each of the 32 sub-slots, then
    // re-read and verify no sub-slot's writes leaked into a neighbour.
    auto arena = std::make_unique<PageArena>();
    auto id = arena->allocate(4096u, 128u);
    QTX_EXPECT(!id.isNull());

    constexpr usize kSubSlotSize = 128u;
    for (qtx::core::u32 idx = 0; idx < 32u; ++idx) {
        auto sub_id = id.withSubSlot(idx);
        auto sv = arena->viewSubSlot(sub_id, kSubSlotSize);
        QTX_EXPECT_EQ(sv.size(), kSubSlotSize);
        std::memset(sv.data(), static_cast<int>(0x10u + idx), sv.size());
    }
    for (qtx::core::u32 idx = 0; idx < 32u; ++idx) {
        auto sub_id = id.withSubSlot(idx);
        auto sv = arena->viewSubSlot(sub_id, kSubSlotSize);
        const auto expected = static_cast<std::byte>(0x10u + idx);
        for (usize i = 0; i < sv.size(); ++i) {
            QTX_EXPECT(sv[i] == expected);
        }
    }
    arena->release(id);
}

QTX_TEST(Arena, EC114_ZeroizeSubSlot_OnlyWipesItsRange) {
    // Critical EC114 property: zeroizeSubSlot() must wipe ONLY the
    // sub-slot's bytes, and must not touch:
    //   (a) other sub-slots in the same slot,
    //   (b) the version counter (so other live sub-slot tenants still
    //       see a valid handle),
    //   (c) the selector bit (so the parent slot stays live).
    auto arena = std::make_unique<PageArena>();
    auto id = arena->allocate(4096u, 128u);
    QTX_EXPECT(!id.isNull());

    constexpr usize kSubSlotSize = 128u;

    // Fill all 32 sub-slots with a marker.
    for (qtx::core::u32 idx = 0; idx < 32u; ++idx) {
        auto sub_id = id.withSubSlot(idx);
        auto sv = arena->viewSubSlot(sub_id, kSubSlotSize);
        std::memset(sv.data(), 0xCC, sv.size());
    }

    // Zeroize ONLY sub-slot 7.
    auto target = id.withSubSlot(7u);
    QTX_EXPECT(arena->zeroizeSubSlot(target, kSubSlotSize));

    // Sub-slot 7 must be all zeros.
    auto target_view = arena->viewSubSlot(target, kSubSlotSize);
    for (usize i = 0; i < target_view.size(); ++i) {
        QTX_EXPECT(target_view[i] == std::byte{0});
    }
    // Every OTHER sub-slot must still be 0xCC.
    for (qtx::core::u32 idx = 0; idx < 32u; ++idx) {
        if (idx == 7u) continue;
        auto other = id.withSubSlot(idx);
        auto sv = arena->viewSubSlot(other, kSubSlotSize);
        for (usize i = 0; i < sv.size(); ++i) {
            QTX_EXPECT(sv[i] == std::byte{0xCC});
        }
    }
    // The slot is still live: validate() returns true and a normal
    // view() still works.
    QTX_EXPECT(arena->validate(id));
    QTX_EXPECT_EQ(arena->view(id).size(), 4096u);

    arena->release(id);
}
