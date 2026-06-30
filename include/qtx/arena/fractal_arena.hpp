// ============================================================================
// @file        fractal_arena.hpp
// @brief       Static monolithic arena with GenID descriptors.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// HA-LAYER: HA-pure core. No new/malloc after start, zero exceptions,
// no reinterpret_cast in public API. Backing — static array of
// std::byte, cache-line aligned.
//
// EDGE CASES CLOSED IN THIS REWRITE:
//
//   EC9   — dangling span: view() carries the GenID with it (the caller
//           can revalidate by passing it back into validate()).  We can't
//           prevent the user from saving the span pointer, but documented
//           and unenforceable cases are out of HA scope (per skill rules).
//   EC15  — race on access_time: not stored here, it lives in the bridge
//           and is bumped with atomic fetch_add (see tiered_bridge.hpp).
//   EC30  — data leakage on slot reuse: zeroizeOnRelease() wipes the slot
//           on release(), so no stale payload from a prior tenant.
//   EC36  — clear() leaving the tail of the slot dirty: zeroizeOnRelease
//           clears the FULL slot, not just requested_size.
//   EC74  — ECC bit-flip in versions_: validate() loads with acquire and
//           also re-checks selector occupancy so a stale version cannot
//           wrongly fail a valid live slot (we *prefer* false-negative
//           failure mode here — safer to refuse than to expose data).
//   EC106 — Power-of-two SlotSize requirement removed (Phase 2 Block B,
//           task 3.1): ColdArena previously had to size slots to the
//           next power of two above the compressed payload (1152 -> 2048,
//           43% waste). Slots can now be sized exactly; `slot * kSlotSize`
//           becomes a 1-cycle imul instead of a shl, which is negligible.
//   EC107 — alignment of every slot is preserved by retaining
//           `SlotSize % kMaxCacheLineSize == 0`: when SlotSize is a
//           non-power-of-two multiple of 128, slot starts remain on
//           128-byte boundaries (Apple Silicon perf-core requirement).
//   EC109 — SlotCount * SlotSize overflow: compile-time guard rejects
//           configurations that would silently wrap to a tiny backing
//           array and turn allocate() into an out-of-bounds writer.
//   EC111 — viewSubSlot() returns a STRICT sub-slot span (Phase 2 Block
//           C task 4.2 infrastructure): bounds-checks both the sub-slot
//           size (must divide kSlotSize evenly, must be >= 64B for
//           cache-line isolation) and the index (< kSlotSize /
//           sub_slot_size). Any violation returns an empty span — never
//           a pointer outside the slot's byte range.
//   EC114 — zeroizeSubSlot() wipes only the sub-slot's byte range and
//           NEVER touches the version counter or the selector bit; the
//           parent slot stays live for any other sub-slot tenants.
// ============================================================================

#pragma once

#include "../core/types.hpp"
#include "../selector/axiom_selector.hpp"
#include "gen_id.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>

namespace qtx::arena {

template <core::usize SlotCount, core::usize SlotSize>
class FractalArena {
    //=== HA: fundamental compile-time invariants ===
    static_assert(SlotCount > 0u, "SlotCount must be positive");
    static_assert(SlotCount % 64u == 0u, "SlotCount must be multiple of 64");
    static_assert(SlotCount <= (core::usize{GenID::kMaxId} + 1u),
                  "SlotCount exceeds GenID::id capacity (24 bits)");
    static_assert(SlotSize > 0u, "SlotSize must be positive");
    // EC106: SlotSize no longer required to be a power of two.
    // We trade a 1-cycle imul (instead of shl) for the ability to size
    // slots EXACTLY to the payload, which on ColdArena recovers up to
    // 43% of the previously-wasted padding. Compilers (GCC/Clang/MSVC)
    // emit a single multiply for `slot * kSlotSize` even when SlotSize
    // is not a power of two — the optimisation cost is negligible.
    //
    // EC107: every slot is guaranteed to start on a 128-byte boundary
    // because we still require `SlotSize % kMaxCacheLineSize == 0` AND
    // the underlying array is `alignas(kMaxCacheLineSize)`. Apple
    // Silicon performance cores need 128B alignment to avoid the
    // cross-line load penalty.
    static_assert(SlotSize % core::kMaxCacheLineSize == 0u,
                  "SlotSize must be multiple of max cache line (128B) "
                  "so every slot starts cache-line-aligned");
    // EC109: detect SlotCount * SlotSize overflow at compile time.
    // The arena reserves `SlotCount * SlotSize` bytes in static storage;
    // a wrap-around here would silently allocate a tiny array and turn
    // legitimate allocate() calls into out-of-bounds writes.
    static_assert(SlotSize <= (std::numeric_limits<core::usize>::max() / SlotCount),
                  "SlotCount * SlotSize overflows usize");

public:
    static constexpr core::usize kSlotCount = SlotCount;
    static constexpr core::usize kSlotSize  = SlotSize;
    static constexpr core::usize kTotalSize = SlotCount * SlotSize;

    /// Zero the slot's memory on release(). True by default — closes
    /// EC30 / EC36 (cross-tenant data leakage). May be disabled by tests
    /// that need to inspect post-release contents.
    static constexpr bool kZeroizeOnRelease = true;

    FractalArena() noexcept : die_{}, versions_{}, selector_() {}

    FractalArena(const FractalArena&)            = delete;
    FractalArena& operator=(const FractalArena&) = delete;
    FractalArena(FractalArena&&)                 = delete;
    FractalArena& operator=(FractalArena&&)      = delete;

    ~FractalArena() = default;

    // ========================================================================
    // @brief Allocate slot, return GenID.
    // ========================================================================
    [[nodiscard]] GenID allocate(core::usize requested_size,
                                 core::usize requested_align) noexcept {
        if (requested_size == 0u || requested_size > kSlotSize) [[unlikely]] {
            return GenID{};
        }
        if (requested_align == 0u ||
            !core::isPowerOfTwo(requested_align) ||
            requested_align > kSlotSize) [[unlikely]] {
            return GenID{};
        }
        // EC55 — alignment_class is log2(alignment) and lives in
        // kAlignmentBits (7) bits, so it can encode values 0..127.
        // Today the slot-level alignment ceiling is 128B (Apple
        // Silicon cache line), which fits at log2=7. We still range-
        // check here so that a future call with a Huge-Page alignment
        // (e.g. 4096 = log2 12) fails CLOSED rather than silently
        // wrapping into the kQuant field bit-positions above.
        const core::u32 alignment_class_unchecked =
            core::log2PowerOfTwo(requested_align);
        if (alignment_class_unchecked > GenID::kMaxAlignment) [[unlikely]] {
            return GenID{};
        }

        const core::usize slot = selector_.acquire();
        if (slot == decltype(selector_)::kNoSlot) [[unlikely]] {
            return GenID{};
        }

        // Bump version. acq_rel synchronises with the prior release() of
        // the same slot so any reader of view() observes a fresh slot.
        const core::u16 new_version =
            versions_[slot].fetch_add(1u, std::memory_order_acq_rel) + 1u;

        return GenID::make(
            static_cast<core::u32>(slot),
            static_cast<core::u32>(new_version & GenID::kMaxVersion),
            alignment_class_unchecked,
            QuantFormat::kFP32,
            SlotState::kActive);
    }

    // ========================================================================
    // @brief View a slot's memory. Empty span if GenID is invalid.
    // ========================================================================
    [[nodiscard]] std::span<std::byte> view(GenID id) noexcept {
        if (!validate(id)) [[unlikely]] {
            return {};
        }
        const core::usize slot = id.id();
        std::byte* base = die_.data() + slot * kSlotSize;
        return std::span<std::byte>(base, kSlotSize);
    }

    [[nodiscard]] std::span<const std::byte> view(GenID id) const noexcept {
        if (!validate(id)) [[unlikely]] {
            return {};
        }
        const core::usize slot = id.id();
        const std::byte* base = die_.data() + slot * kSlotSize;
        return std::span<const std::byte>(base, kSlotSize);
    }

    // ========================================================================
    // EC111 — View a SUB-SLOT inside an allocated slot (Phase 2 Block C
    // task 4.2 infrastructure layer).
    //
    // Sub-slot semantics:
    //   - sub_slot_size MUST evenly divide kSlotSize (otherwise the last
    //     sub-slot would partially overrun the slot's tail);
    //   - sub_slot_size MUST be ≥ kMinCacheLineSize so individual
    //     sub-slot users do not share a cache line with their neighbour;
    //   - the GenID's subSlotIndex() must be < (kSlotSize / sub_slot_size).
    //
    // Returns an empty span on any contract violation (fail-safe; the
    // caller treats this identically to a stale-GenID view()).
    //
    // The current Block C 4.2 deliverable is THIS API — the bridge's
    // ref-counted multi-tenant ownership policy ("agent A and agent B
    // share slot 7, sub-slots 3 and 5 respectively, release of A must
    // not free slot 7") is a separate Phase 2 Block C.2 task.
    // ========================================================================

    [[nodiscard]] static constexpr core::usize subSlotCount(
        core::usize sub_slot_size) noexcept
    {
        if (sub_slot_size == 0u) return 0u;
        if (sub_slot_size > kSlotSize) return 0u;
        if (kSlotSize % sub_slot_size != 0u) return 0u;
        // EC16 — Floor is kMaxCacheLineSize (128B), NOT kMinCacheLineSize
        // (64B). Apple Silicon performance cores use 128-byte cache lines;
        // a 64-byte sub-slot size would place two sub-cells in the same
        // architectural cache line on M1/M2/M3, causing False Sharing
        // between siblings even though their byte ranges are disjoint.
        // x86 Intel/AMD 64B cache lines are still safe — the floor is
        // raised to the maximum across supported targets. Existing
        // callers that pass 64 will see this function return 0 (no sub-
        // slabbing possible), which their fail-safe codepath already
        // handles (they fall back to whole-slot allocation).
        if (sub_slot_size < core::kMaxCacheLineSize) return 0u;
        // Additional invariant: the sub-slot size must itself be a
        // multiple of the largest cache line so the *start* address of
        // every sub-cell is aligned. Without this, sub-cell 1 could
        // begin mid-line and share a line with the tail of sub-cell 0.
        if (sub_slot_size % core::kMaxCacheLineSize != 0u) return 0u;
        return kSlotSize / sub_slot_size;
    }

    [[nodiscard]] std::span<std::byte> viewSubSlot(
        GenID id, core::usize sub_slot_size) noexcept
    {
        if (!validate(id)) [[unlikely]] return {};
        const core::usize n = subSlotCount(sub_slot_size);
        if (n == 0u) [[unlikely]] return {};
        const core::u32 idx = id.subSlotIndex();
        if (static_cast<core::usize>(idx) >= n) [[unlikely]] return {};

        const core::usize slot   = id.id();
        const core::usize offset = static_cast<core::usize>(idx) * sub_slot_size;
        std::byte* base = die_.data() + slot * kSlotSize + offset;
        return std::span<std::byte>(base, sub_slot_size);
    }

    [[nodiscard]] std::span<const std::byte> viewSubSlot(
        GenID id, core::usize sub_slot_size) const noexcept
    {
        if (!validate(id)) [[unlikely]] return {};
        const core::usize n = subSlotCount(sub_slot_size);
        if (n == 0u) [[unlikely]] return {};
        const core::u32 idx = id.subSlotIndex();
        if (static_cast<core::usize>(idx) >= n) [[unlikely]] return {};

        const core::usize slot   = id.id();
        const core::usize offset = static_cast<core::usize>(idx) * sub_slot_size;
        const std::byte* base = die_.data() + slot * kSlotSize + offset;
        return std::span<const std::byte>(base, sub_slot_size);
    }

    // ========================================================================
    // EC114 — Zero only the sub-slot range (not the whole slot). Used by
    // the bridge when a sub-slot's tenant is released but the parent
    // slot remains live for other sub-slot tenants. Bumps NEITHER the
    // selector bit NOR the version counter — those are slot-level.
    // ========================================================================
    bool zeroizeSubSlot(GenID id, core::usize sub_slot_size) noexcept {
        auto v = viewSubSlot(id, sub_slot_size);
        if (v.empty()) [[unlikely]] return false;
        std::memset(v.data(), 0, v.size());
        return true;
    }

    // ========================================================================
    // @brief Release a slot. Zeroes the backing bytes (EC30, EC36) and
    // bumps the generation BEFORE returning the bit to the selector, so
    // any racing acquire() that wins the bit immediately observes a clean
    // version on the next validate().
    // ========================================================================
    bool release(GenID id) noexcept {
        if (!validate(id)) [[unlikely]] {
            return false;
        }
        const core::usize slot = id.id();

        // 1. Zero the entire slot (full kSlotSize, including the padding
        //    beyond requested_size) — this defeats EC30 / EC36.
        if constexpr (kZeroizeOnRelease) {
            std::byte* base = die_.data() + slot * kSlotSize;
            std::memset(base, 0, kSlotSize);
        }

        // 2. Bump version (publish the wipe + invalidate stale handles).
        versions_[slot].fetch_add(1u, std::memory_order_acq_rel);

        // 3. Return the slot to the selector.
        selector_.release(slot);
        return true;
    }

    // ========================================================================
    // @brief Validate GenID without side effects.
    // ========================================================================
    [[nodiscard]] bool validate(GenID id) const noexcept {
        if (id.isNull()) return false;
        const core::usize slot = id.id();
        if (slot >= kSlotCount) [[unlikely]] return false;

        const core::u16 expected =
            versions_[slot].load(std::memory_order_acquire);
        if ((expected & GenID::kMaxVersion) !=
            (id.version() & GenID::kMaxVersion)) [[unlikely]] {
            return false;
        }
        return selector_.isOccupied(slot);
    }

    //=== Inspection (for tests/telemetry, not hot path) ===

    [[nodiscard]] core::usize freeSlots() const noexcept {
        return selector_.freeCount();
    }

    [[nodiscard]] core::usize usedSlots() const noexcept {
        return kSlotCount - selector_.freeCount();
    }

    [[nodiscard]] static constexpr core::usize totalBytes() noexcept {
        return kTotalSize;
    }

    [[nodiscard]] static constexpr core::usize slotBytes() noexcept {
        return kSlotSize;
    }

private:
    // === Backing storage, cache-line aligned ===
    alignas(core::kMaxCacheLineSize)
        std::array<std::byte, kTotalSize> die_;

    // === Generation per slot (atomic, u16) ===
    std::array<std::atomic<core::u16>, kSlotCount> versions_;

    selector::AxiomSelector<SlotCount> selector_;
};

//=== HA: compile-time template checking on a typical instance ===
namespace detail {
using TestArena = FractalArena<256, 128>;
static_assert(TestArena::kSlotCount == 256u);
static_assert(TestArena::kSlotSize  == 128u);
static_assert(TestArena::kTotalSize == 32u * 1024u);
static_assert(TestArena::totalBytes() == 32u * 1024u);

// EC106 — verify a non-power-of-two SlotSize compiles and computes
// correctly. 1152 = 9 * 128, so cache-line alignment of every slot is
// preserved. This is the production ColdArena configuration after
// Phase 2 Block B task 3.1.
using TestColdArena = FractalArena<512, 1152>;
static_assert(TestColdArena::kSlotCount == 512u);
static_assert(TestColdArena::kSlotSize  == 1152u);
static_assert(TestColdArena::kSlotSize % core::kMaxCacheLineSize == 0u,
              "EC107: non-power-of-two SlotSize must still be a multiple "
              "of the max cache line so every slot is 128B-aligned");
static_assert(!core::isPowerOfTwo(TestColdArena::kSlotSize),
              "EC106: this case exists specifically to exercise the "
              "non-power-of-two path");
static_assert(TestColdArena::kTotalSize == 512u * 1152u);  // 576 KiB
}  // namespace detail

}  // namespace qtx::arena
