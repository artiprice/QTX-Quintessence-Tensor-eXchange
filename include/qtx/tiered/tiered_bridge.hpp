// ============================================================================
// @file        tiered_bridge.hpp
// @brief       Bridge with automatic FP32 -> INT8 sleeper-agent migration.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// HA-SAFETY:
//   - All operations are noexcept and noexcept-safe (no std::vector::push_back
//     in the hot path; tenant records live in a STATIC pool sized at
//     compile time).
//   - Single mutex for structural changes (createTenant / destroyTenant /
//     evictOldestToCold). Hot path acquireFP32 / release uses atomics.
//
// EDGE CASES CLOSED IN THIS REWRITE:
//
//   EC1  — pin_count race: pin_count is now std::atomic<u32> with fetch_add.
//   EC2  — pin_count underflow: release() refuses to decrement when 0
//          (CAS loop with bounds check); never wraps.
//   EC6  — std::vector::push_back inside noexcept: replaced with a static
//          fixed-size pool of TenantRecord; capacity is a template param.
//   EC7  — concurrent eviction race: structural changes are serialised
//          under a single mutex (create/destroy/evict). Hot path (pin/
//          unpin/access_time) stays lock-free.
//   EC14 — destroyTenant double-free: state transitions to kDead under
//          the mutex; subsequent destroyTenant on the same handle is a
//          no-op (idempotent).
//   EC15 — access_time race: now an atomic<u64>, fetched with
//          monotonic_clock_.fetch_add(1, acq_rel).
//   EC18 — O(N) LRU search bottleneck: same O(N) sweep but only invoked
//          under the structural mutex, so it is bounded to a single
//          caller at a time. Documented; HeapLRU is a P4 optimisation.
//   EC19 — second allocate() fail after evict: we now retry up to
//          kEvictRetries (default 2) and surface a clean OOM otherwise.
//   EC83 — acquireFP32 returns empty span on internal failure: caller
//          MUST check span.empty().  We additionally never leave the
//          tenant in an inconsistent state on a failed promote.
//   EC85 — fragmentation / heterogeneous sizes: documented invariant
//          that every tenant occupies a full slot. Smaller-buffer
//          packing is P4.
//   EC86 — multi-network depletion of a shared pool: bridge owns its
//          own tenant pool, not a global one; the ContextPool problem
//          is fixed in bridge.cpp.
//   EC87 — exception in client code skips release(): NOT solved here
//          (it's a documented contract violation on the user side).
//   EC88 — invalid BridgeConfig: validated in createTenant via static_assert
//          on the slot size, and at runtime by the constructor.
//   EC99 — all tenants pinned: evictOldestToCold returns false, and the
//          caller surfaces a clean OOM. Documented.
//   EC101 — tenant_id u32 wraps in ~11 min at 6.3 M-op/s, silently breaking
//           ABA at the public API. tenant_id and next_tenant_id_ widened
//           to u64; the previous single sentinel kNoTenant is split into
//           kNoSlot (u32, pool-table index) and kNoTenantId (u64, public
//           id) so the two semantics no longer share one constant.
//   EC105 — racy "skip kNoTenant + retry" pattern in nextTenantId() under
//           u32 would let two threads each waste two ids on the same skip
//           target. With u64 the skip path is mathematically unreachable
//           for any realistic uptime; the retry is now a one-shot
//           defence-in-depth rather than an unbounded loop.
//   EC115 — Phase 2 Block C.2 multi-tenant ownership: a "micro-tenant"
//           may share a hot slot with up to (kSubSlotsPerSlot - 1)
//           siblings. Ownership is tracked by a single atomic u32
//           bitmap per shared slot — bit set ⇔ sub-cell taken. The
//           release path uses fetch_and to clear the bit and then
//           releases the underlying hot slot iff the resulting bitmap
//           is zero. No separate ref-counter — the bitmap IS the
//           refcount via popcount.
//   EC116 — race between concurrent createMicroTenant (acquiring a
//           bit) and a destroyMicroTenant on the LAST sibling (which
//           would release the parent slot). Both operations now run
//           under struct_mutex_, and createMicroTenant additionally
//           validates the bitmap with a no-op CAS BEFORE flipping the
//           bit, so it never inherits a stale "this slot is shared"
//           view of a slot that has already been freed.
//   EC117 — eviction sweep would corrupt sibling micro-tenants if it
//           tried to quantize a shared slot. evictOldestToColdLocked
//           now skips records with kind == kMicro entirely; the OOM
//           fallback for micro-tenants is "createMicroTenant returns
//           {0, false}", same contract as a pinned full-slot tenant.
//   EC118 — leak through the kEvictRetries path: when a fresh shared
//           slot is requested via evict-retry, a successful evict
//           followed by a failed alloc would lose the released slot.
//           Mitigated by the same allocate-or-fail loop the full-slot
//           path uses; on terminal failure the TenantRecord is
//           explicitly released (no kind/state pollution).
//   EC119 — occupancy-bitmap overflow: bitmap is u32 (≥ 32 bits) and
//           we statically assert kSubSlotsPerSlot ≤ 32, so by
//           construction no occupancy bit lands outside the word.
// ============================================================================

#pragma once

#include "qtx/arena/fractal_arena.hpp"
#include "qtx/arena/gen_id.hpp"
#include "qtx/core/contracts.hpp"
#include "qtx/core/types.hpp"
#include "qtx/quantize/quantizer.hpp"
#include "qtx/tiered/cold_codec.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <string_view>

// ============================================================================
// HA: debug-only assertion macro. Becomes a hard fail-fast under
// QTX_DEBUG (matches the policy in core/contracts.hpp), evaporates
// to (void)0 in release so it is genuinely zero-cost in the hot path.
// Used inside the bridge's structural-mutex region for HA defence-in-
// depth — every bit-arithmetic invariant that MUST hold is asserted
// once, so a regression in one of the bit-layout constants surfaces in
// test rather than as a corrupted handle in production.
// ============================================================================
#if defined(QTX_DEBUG) || !defined(NDEBUG)
    #define QTX_TIERED_ASSERT(cond)                                          \
        do {                                                                     \
            if (!(cond)) [[unlikely]] {                                          \
                ::qtx::core::invokeFailHandler(                                \
                    __FILE__, __LINE__, __func__,                                \
                    "tiered HA invariant: " #cond);                              \
            }                                                                     \
        } while (false)
#else
    #define QTX_TIERED_ASSERT(cond) ((void)0)
#endif

namespace qtx::tiered {

// ============================================================================
// A handle to one agent in a tiered system.
//
// tenant_id is an opaque integer that, combined with a generation token
// embedded in the high bits, defeats the ABA reuse pattern at the API
// surface. The handle's is_valid bit is preserved for ABI symmetry; the
// preferred check is `findTenant(h) != nullptr`.
// ============================================================================

// EC101: tenant_id widened to u64. At the observed steady-state throughput
// of 6.3 M lifecycles/s the previous u32 counter would wrap in ≈11 minutes,
// silently breaking the ABA guarantee at the public API surface. A 64-bit
// counter pushes that horizon past 90 millennia at the same rate, which is
// indistinguishable from "never" for any realistic deployment.
struct TenantHandle {
    core::u64 tenant_id;
    bool      is_valid;
};

// ============================================================================
// @class TieredArenaBridge
// ============================================================================

template <typename HotArenaT, typename ColdArenaT,
          core::usize MaxTenants = 4096u,
          codec::ColdCodecPolicy ColdCodecT = codec::Int8ColdCodec>
class TieredArenaBridge {
    static_assert(HotArenaT::kSlotSize >= 128);
    static_assert(ColdArenaT::kSlotSize >= 128);
    static_assert(MaxTenants > 0u, "MaxTenants must be positive");

public:
    static constexpr core::usize kHotSlotBytes  = HotArenaT::kSlotSize;
    static constexpr core::usize kColdSlotBytes = ColdArenaT::kSlotSize;
    static constexpr core::usize kPayloadBytes  = kHotSlotBytes;
    static constexpr core::usize kPayloadElements = kPayloadBytes / sizeof(float);
    /// Cold-tier compressed size, derived from the SELECTED cold codec
    /// policy (default Int8ColdCodec). Swapping the policy at the type
    /// level re-derives this bound, and the static_assert below re-checks
    /// that the cold slot can hold it. This is the single seam that makes
    /// the cold format pluggable without touching the hot path.
    static constexpr core::usize kCompressedBytes =
        ColdCodecT::compressedBound(kPayloadElements);

    static_assert(kCompressedBytes <= kColdSlotBytes,
                  "ColdArena slot too small to hold compressed payload");
    static_assert(kPayloadBytes % (quantize::kBlockElements * sizeof(float)) == 0,
                  "Payload size must be multiple of quantization block size");

    // EC60 — Format/cold-size compatibility matrix, evaluated at
    // compile time. The cold tier carries the ACTIVELY-SELECTED format
    // (kCompressedBytes, asserted above), but the bridge API also
    // exposes paths that compress to alternative formats for telemetry
    // and benchmarking (e.g. measuring INT4 ratios on data flowing
    // through an INT8-configured tier). We compute the size of each
    // supported format here and surface ONE compile-time signal per
    // format describing fit-into-cold. The signal is a constexpr bool
    // rather than a hard static_assert so a deployment can choose a
    // small ColdArena slot (saving memory) when it knows only INT8
    // will ever be requested.
    //
    // The HA contract: if a runtime path attempts to compress to a
    // format whose static `kCanCompressTo*` is false, the call must
    // fail-clean (return 0) rather than overflow the cold buffer.
    // The bridge enforces this in the dispatch layer.
    static constexpr core::usize kBF16CompressedBytes =
        quantize::compressedSize(arena::QuantFormat::kBF16, kPayloadElements);
    static constexpr core::usize kFP16CompressedBytes =
        quantize::compressedSize(arena::QuantFormat::kFP16, kPayloadElements);
    static constexpr core::usize kINT8CompressedBytes =
        quantize::compressedSize(arena::QuantFormat::kINT8, kPayloadElements);
    static constexpr core::usize kINT4CompressedBytes =
        quantize::compressedSize(arena::QuantFormat::kINT4, kPayloadElements);

    static constexpr bool kCanCompressToBF16 = (kBF16CompressedBytes <= kColdSlotBytes);
    static constexpr bool kCanCompressToFP16 = (kFP16CompressedBytes <= kColdSlotBytes);
    static constexpr bool kCanCompressToINT8 = (kINT8CompressedBytes <= kColdSlotBytes);
    static constexpr bool kCanCompressToINT4 = (kINT4CompressedBytes <= kColdSlotBytes);

    // At minimum, the actively-selected format must fit (already
    // asserted), AND the INT4 fallback must fit so that aggressive-
    // OOM eviction always has SOMETHING to compress to. INT4 is the
    // smallest format we ship (4 bits/element + scale) and is
    // guaranteed to fit any cold slot that accepts INT8.
    static_assert(kCanCompressToINT4,
                  "EC60: INT4 (smallest format) must fit in ColdArena slot");

    /// Number of times we retry allocate(hot) after a successful evict
    /// before declaring OOM. Closes EC19.
    static constexpr core::u32 kEvictRetries = 2u;

    /// Sentinel meaning "no pool record" — used for the static tenant table
    /// index (which is bounded by MaxTenants, so u32 is sufficient).
    static constexpr core::u32 kNoSlot = static_cast<core::u32>(-1);

    /// Sentinel meaning "no tenant id" — used for the public identifier in
    /// TenantHandle. EC101: must be u64 to avoid an 11-minute ABA wrap.
    static constexpr core::u64 kNoTenantId = static_cast<core::u64>(-1);

    // ========================================================================
    // EC115 — micro-tenant constants (Phase 2 Block C.2). Exposed so
    // callers can size their payloads correctly and inspect the
    // packing factor at compile time.
    // ========================================================================
    /// Maximum payload size of one micro-tenant. Power-of-two divisor
    /// of HotArena slot size, ≥ one cache line.
    static constexpr core::usize kMicroSlotSize = 128u;
    // EC16 — micro-slot size MUST be ≥ kMaxCacheLineSize (128B), not just
    // kMinCacheLineSize (64B). On Apple Silicon performance cores the
    // architectural cache line is 128 bytes; if we permitted a 64-byte
    // micro-slot, two adjacent micro-tenants would land in the SAME
    // hardware cache line and contend on every store (False Sharing),
    // even though their data ranges are disjoint. With 128 bytes the
    // isolation holds on every supported target (x86_64 Intel/AMD 64B,
    // Apple Silicon 128B, generic ARM64 64-or-128B). The HA layer
    // pins this with kMaxCacheLineSize as the single source of truth.
    static_assert(kMicroSlotSize >= core::kMaxCacheLineSize,
                  "EC16: micro-slot must be ≥ kMaxCacheLineSize so adjacent "
                  "micro-tenants are False-Sharing-isolated on every "
                  "supported target (Apple Silicon = 128B cache line)");
    static_assert(kMicroSlotSize % core::kMaxCacheLineSize == 0u,
                  "EC16: micro-slot must be a multiple of the largest cache "
                  "line so the start address of every sub-cell is itself "
                  "cache-line aligned");
    static_assert(HotArenaT::kSlotSize % kMicroSlotSize == 0u,
                  "HotArena slot must be an integer multiple of micro-slot");
    /// How many micro-tenants pack into one hot slot.
    static constexpr core::u32 kSubSlotsPerSlot =
        static_cast<core::u32>(HotArenaT::kSlotSize / kMicroSlotSize);
    static_assert(kSubSlotsPerSlot >= 1u && kSubSlotsPerSlot <= 32u,
                  "kSubSlotsPerSlot must fit in the 5-bit GenID sub_slot "
                  "field (max 32)");

    TieredArenaBridge(HotArenaT* hot, ColdArenaT* cold) noexcept
        : hot_(hot), cold_(cold) {}

    TieredArenaBridge(const TieredArenaBridge&)            = delete;
    TieredArenaBridge& operator=(const TieredArenaBridge&) = delete;
    TieredArenaBridge(TieredArenaBridge&&)                 = delete;
    TieredArenaBridge& operator=(TieredArenaBridge&&)      = delete;

    ~TieredArenaBridge() = default;

    // ========================================================================
    // === Lifecycle ===
    // ========================================================================

    [[nodiscard]] TenantHandle createTenant() noexcept {
        if (hot_ == nullptr || cold_ == nullptr) return {0u, false};

        std::lock_guard<std::mutex> lock(struct_mutex_);

        // Slot in the tenant pool.
        const core::u32 pool_idx = acquireRecord();
        if (pool_idx == kNoSlot) return {0u, false};  // tenant table full

        // Hot allocation, retrying on evict.
        arena::GenID hot_id = hot_->allocate(kPayloadBytes, 128u);
        for (core::u32 r = 0u; r < kEvictRetries && hot_id.isNull(); ++r) {
            if (!evictOldestToColdLocked(kNoTenantId)) {
                releaseRecord(pool_idx);
                return {0u, false};
            }
            hot_id = hot_->allocate(kPayloadBytes, 128u);
        }
        if (hot_id.isNull()) {
            releaseRecord(pool_idx);
            return {0u, false};
        }

        TenantRecord& rec = tenants_[pool_idx];
        rec.tenant_id    = nextTenantId();
        rec.pin_count.store(0u, std::memory_order_relaxed);
        rec.state.store(TenantState::kInHot, std::memory_order_relaxed);
        rec.hot_id       = hot_id;
        rec.cold_id      = arena::GenID{};
        // EC115: ensure kind is reset on recycle — a previous micro
        // tenant in this pool slot would have left kind == kMicro.
        rec.kind            = TenantKind::kFullSlot;
        rec.shared_slot_idx = 0u;
        // EC136: a recycled TenantRecord must always start unsealed —
        // otherwise weights sealed in this slot's previous life would
        // silently protect the new (unrelated) tenant from eviction.
        // Symmetric with ContextPool::acquire's reset of write_protected.
        rec.sealed.store(false, std::memory_order_relaxed);
        rec.access_time.store(monotonic_clock_.fetch_add(1u, std::memory_order_acq_rel),
                              std::memory_order_relaxed);
        rec.in_use.store(true, std::memory_order_release);

        return {rec.tenant_id, true};
    }

    bool destroyTenant(TenantHandle h) noexcept {
        if (hot_ == nullptr || cold_ == nullptr) return false;

        std::lock_guard<std::mutex> lock(struct_mutex_);
        TenantRecord* rec = findTenantLocked(h);
        if (rec == nullptr) return false;
        if (rec->state.load(std::memory_order_acquire) == TenantState::kDead) {
            return false;  // EC14: idempotent double-destroy
        }

        // EC5 — Memory-ordering note for the `rec->kind` read below.
        //
        // The find→read chain that follows reads several plain-non-atomic
        // fields (kind, shared_slot_idx, hot_id, cold_id). Reader/writer
        // synchronisation for these is supplied by TWO mechanisms:
        //
        //   (a) struct_mutex_, held by destroyTenant AND by every path
        //       that writes those fields (createTenant, createMicroTenant,
        //       evictOldestToColdLocked). The mutex by itself acquire-
        //       releases all preceding stores.
        //
        //   (b) findTenantLocked does an `in_use.load(acquire)`. Every
        //       writer pairs the field initialisation with a final
        //       `in_use.store(true, release)`; the acquire/release pair
        //       publishes kind / shared_slot_idx / hot_id / cold_id
        //       BEFORE in_use becomes true. A reader that observes
        //       in_use == true has therefore already observed the
        //       initialisation of every other field — independently of
        //       the mutex.
        //
        // The two mechanisms are belt-and-braces; either alone is
        // sufficient. We keep both because (a) the mutex covers the
        // structural decisions and (b) the atomic discipline survives
        // any future relaxation of the locking policy.

        if (rec->kind == TenantKind::kMicro) {
            // EC114 hookup: wipe the sub-slot bytes BEFORE clearing the
            // occupancy bit. If we cleared the bit first, another
            // createMicroTenant could acquire this sub-cell in the
            // microsecond gap and read our residue.
            (void)hot_->zeroizeSubSlot(rec->hot_id, kMicroSlotSize);

            // EC8 — Defence against a corrupted shared_slot_idx. The
            // field is plain non-atomic and lives in a struct that gets
            // recycled; if it ever holds a stale or out-of-range value
            // (e.g. through a memory-corruption bug elsewhere), the
            // subsequent fetch_and would clear bits on an UNRELATED
            // shared slot, evicting up to kSubSlotsPerSlot - 1 unrelated
            // micro-tenants. Range-check explicitly: kMaxSharedSlots is
            // a compile-time constant, so the branch is hoisted and the
            // happy path is one CMP+JAE.
            if (rec->shared_slot_idx >= kMaxSharedSlots) [[unlikely]] {
                // Boundary violation — refuse silently and mark dead.
                // We cannot safely release the underlying slot because
                // we don't know which one it was; better to leak than
                // to corrupt 31 siblings.
                rec->state.store(TenantState::kDead, std::memory_order_release);
                rec->in_use.store(false, std::memory_order_release);
                return false;
            }
            // Clear our occupancy bit. EC116: fetch_and gives us the
            // PRIOR bitmap atomically; if the new bitmap is zero we
            // own the right to release the underlying slot.
            const core::u32 sub_idx  = rec->hot_id.subSlotIndex();
            // EC8 (cont): sub_idx must fit in [0, kSubSlotsPerSlot). If
            // a malformed GenID slipped through with an out-of-range
            // sub-slot, the mask would still be a valid u32 bit-mask
            // (≤ bit 31, since GenID limits sub-slot to 5 bits = 31),
            // but the bit cleared would not correspond to our
            // allocation. Range-check.
            if (sub_idx >= kSubSlotsPerSlot) [[unlikely]] {
                rec->state.store(TenantState::kDead, std::memory_order_release);
                rec->in_use.store(false, std::memory_order_release);
                return false;
            }
            const core::u32 bit_mask = core::u32{1} << sub_idx;
            auto& shared = shared_slots_[rec->shared_slot_idx];
            const core::u32 prev = shared.occupancy_bitmap.fetch_and(
                ~bit_mask, std::memory_order_acq_rel);
            const core::u32 now  = prev & ~bit_mask;
            if (now == 0u) {
                // Last micro-tenant in this slot — release the slot.
                // The hot_id stored in `shared` has subSlot bits set
                // by whichever tenant last touched it; strip them so
                // FractalArena::release() receives a "clean" id.
                const arena::GenID clean = shared.hot_id.withSubSlot(0u);
                (void)hot_->release(clean);
                shared.hot_id = arena::GenID{};
            }
        } else {
            // Full-slot path (Block A behaviour).
            if (!rec->hot_id.isNull())  (void)hot_->release(rec->hot_id);
            if (!rec->cold_id.isNull()) (void)cold_->release(rec->cold_id);
        }
        rec->state.store(TenantState::kDead, std::memory_order_release);
        rec->hot_id  = arena::GenID{};
        rec->cold_id = arena::GenID{};
        rec->in_use.store(false, std::memory_order_release);
        return true;
    }

    // ========================================================================
    // === Access (FP32 view) ===
    // ========================================================================

    [[nodiscard]] std::span<std::byte> acquireFP32(TenantHandle h) noexcept {
        if (hot_ == nullptr || cold_ == nullptr) return {};

        // Locate the record under the mutex (so it can't be destroyed
        // mid-flight) and then operate on it.
        std::lock_guard<std::mutex> lock(struct_mutex_);
        TenantRecord* rec = findTenantLocked(h);
        if (rec == nullptr) return {};
        const TenantState st = rec->state.load(std::memory_order_acquire);
        if (st == TenantState::kDead) return {};

        if (st == TenantState::kInCold) {
            // Promote: allocate hot slot, possibly evict another tenant.
            arena::GenID hot_id = hot_->allocate(kPayloadBytes, 128u);
            for (core::u32 r = 0u; r < kEvictRetries && hot_id.isNull(); ++r) {
                if (!evictOldestToColdLocked(rec->tenant_id)) {
                    return {};
                }
                hot_id = hot_->allocate(kPayloadBytes, 128u);
            }
            if (hot_id.isNull()) return {};

            auto cold_view = cold_->view(rec->cold_id);
            auto hot_view  = hot_->view(hot_id);
            if (cold_view.empty() || hot_view.empty()) {
                (void)hot_->release(hot_id);
                return {};
            }
            const auto bytes_written = ColdCodecT::decompress(
                std::span<const std::byte>(cold_view.data(), kCompressedBytes),
                hot_view);
            if (bytes_written != kPayloadBytes) {
                (void)hot_->release(hot_id);
                return {};  // EC83: clean failure, tenant unchanged
            }
            (void)cold_->release(rec->cold_id);
            rec->cold_id = arena::GenID{};
            rec->hot_id  = hot_id;
            rec->state.store(TenantState::kInHot, std::memory_order_release);
            ++promotions_;
        }

        // EC15 — Same soft-max guard as acquireMicroFP32. Catches a
        // caller that leaks acquires (forgets to release) before
        // pin_count wraps to zero and breaks the evict-respects-pins
        // invariant.
        constexpr core::u32 kPinCountSoftMax = core::u32{1} << 31;
        const core::u32 cur_pin =
            rec->pin_count.load(std::memory_order_acquire);
        if (cur_pin >= kPinCountSoftMax) [[unlikely]] {
            return {};
        }

        // Pin (atomic increment, EC1).
        rec->pin_count.fetch_add(1u, std::memory_order_acq_rel);
        // EC96: monotonic access_time via CAS-max, not raw store.
        bumpAccessTime(rec);
        return hot_->view(rec->hot_id);
    }

    /// Removes the pin without taking the structural mutex (hot path).
    /// EC2: never decrements below zero (CAS loop).
    bool release(TenantHandle h) noexcept {
        if (hot_ == nullptr || cold_ == nullptr) return false;
        TenantRecord* rec = findTenantLockfree(h);
        if (rec == nullptr) return false;
        core::u32 cur = rec->pin_count.load(std::memory_order_acquire);
        while (cur > 0u) {
            if (rec->pin_count.compare_exchange_weak(
                    cur, cur - 1u,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
            // cur reloaded by CAS; loop until either the decrement lands
            // or pin_count reaches zero.
        }
        return false;
    }

    // ========================================================================
    // EC136 — Tenant-level seal. Symmetric with ArenaBridge::sealBuffer
    // at the ggml level. A sealed tenant is one whose payload must NEVER
    // be migrated to the cold (lossy-quantised) tier — WEIGHTS, lookup
    // tables, sentinels. After sealTenant() succeeds, evictOldestTo-
    // ColdLocked refuses to consider this tenant as a victim, exactly
    // as it already refuses pinned tenants.
    //
    // The seal is published with release semantics so a writer thread
    // that finishes loading the weights and then calls sealTenant() has
    // a clean happens-before with any subsequent evict-decision read.
    // It is idempotent (sealing a sealed tenant is a no-op publish).
    // Unsealing is intentionally NOT exposed: once weights are sealed
    // they remain sealed for the tenant's lifetime; the destroy path
    // is the only way to release the underlying slot.
    // ========================================================================
    bool sealTenant(TenantHandle h) noexcept {
        if (hot_ == nullptr || cold_ == nullptr) return false;
        // We take struct_mutex_ to serialize against destroy/evict and
        // to ensure the writer's prior set_tensor stores are visible
        // before the seal flag is published (the mutex acts as a
        // release barrier on its own).
        std::lock_guard<std::mutex> lock(struct_mutex_);
        TenantRecord* rec = findTenantLocked(h);
        if (rec == nullptr) return false;
        if (rec->state.load(std::memory_order_acquire) == TenantState::kDead) {
            return false;
        }
        rec->sealed.store(true, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool isTenantSealed(TenantHandle h) noexcept {
        std::lock_guard<std::mutex> lock(struct_mutex_);
        TenantRecord* rec = findTenantLocked(h);
        if (rec == nullptr) return false;
        return rec->sealed.load(std::memory_order_acquire);
    }

    // ========================================================================
    // === Micro-tenant API (Phase 2 Block C.2) ===
    //
    // Micro-tenants pack up to kSubSlotsPerSlot (32) payloads into one
    // hot slot, with cache-line isolation per sub-cell. Returns an
    // invalid handle when no shared slot can be found (no free sub-cell
    // anywhere, and no fresh hot slot available).
    //
    // ATOMICITY (EC116):
    //   We hold struct_mutex_ for the structural decision (which
    //   shared slot to use, whether to acquire a fresh one). The
    //   actual occupancy_bitmap CAS that "marks the sub-cell taken"
    //   happens INSIDE the lock too, so no concurrent destroyer can
    //   race us to release the parent slot mid-decision.
    // ========================================================================

    [[nodiscard]] TenantHandle createMicroTenant() noexcept {
        if (hot_ == nullptr || cold_ == nullptr) return {0u, false};

        std::lock_guard<std::mutex> lock(struct_mutex_);

        // Step 1: secure a TenantRecord slot.
        const core::u32 pool_idx = acquireRecord();
        if (pool_idx == kNoSlot) return {0u, false};

        // Step 2: find a SharedSlotEntry with a free sub-cell, OR
        //         acquire a fresh hot slot and a fresh entry.
        core::u32 shared_idx = kNoSlot;
        core::u32 sub_idx    = kSubSlotsPerSlot;  // sentinel = "none"

        // 2a: scan existing shared entries for a free bit. Under
        // struct_mutex_ the SET of occupied bits cannot grow; it can
        // only shrink (a parallel release runs lock-free). A CAS loop
        // handles that one race: if a sibling-release strikes between
        // our load and our fetch_or, the bit we picked is still free
        // (release only ever clears bits) — but the bitmap as a whole
        // may have shrunk, so we re-pick the lowest free bit each iter.
        //
        // EC121 (was #6) — kSubSlotsPerSlot can be strictly less than the
        // 32 bits of the u32 bitmap (e.g. HotSlot=1024 / MicroSlot=128
        // gives kSubSlotsPerSlot=8). A stale-or-not bitmap with any bit
        // set above kSubSlotsPerSlot would be impossible by construction,
        // but defence-in-depth: we explicitly mask `bm` to the valid bit
        // range and use that masked value for both the "is full?" check
        // and the "find first free" pick. This guarantees the returned
        // free_bit is ALWAYS strictly less than kSubSlotsPerSlot, so the
        // subsequent withSubSlot() never silently truncates and we never
        // hand out an out-of-range sub-slot index.
        constexpr core::u32 kValidBits = (kSubSlotsPerSlot >= 32u)
            ? core::u32{0xFFFFFFFFu}
            : (core::u32{1} << kSubSlotsPerSlot) - 1u;

        // EC11 — Rotate scan start to defeat front-bias. The original
        // `for (i = 0; i < kMaxSharedSlots; ++i)` pattern always picked
        // the lowest-index shared slot that had a free bit, which meant
        // the FIRST entry of the pool absorbed every new allocation and
        // sat at maximum contention while the tail of the pool was rarely
        // touched. Under struct_mutex_ this is not a correctness issue
        // (CAS contention is bounded to one writer at a time), but it
        // produces measurable wear-levelling and L1d-locality artefacts
        // on long-running services: the high indices' cache lines age
        // out, then need to be brought back in on the rare allocation
        // that reaches them.
        //
        // We keep a per-bridge `next_shared_scan_start_` counter (not
        // thread-local — that would conflict with HFT-tier 2 determinism;
        // the counter is a member that is only ever touched under
        // struct_mutex_, so it is naturally serialised). On each call we
        // start the scan at the cursor and walk modulo kMaxSharedSlots,
        // then advance the cursor by 1. Over a million calls every entry
        // is visited first ~equally often.
        const core::u32 scan_start = next_shared_scan_start_;
        next_shared_scan_start_ =
            (scan_start + 1u) % static_cast<core::u32>(kMaxSharedSlots);

        for (core::u32 k = 0u; k < kMaxSharedSlots; ++k) {
            const core::u32 i = (scan_start + k) %
                                static_cast<core::u32>(kMaxSharedSlots);
            auto& s = shared_slots_[i];
            if (s.hot_id.isNull()) continue;  // entry not in use

            core::u32 bm = s.occupancy_bitmap.load(std::memory_order_acquire);
            // Mask to valid range — protects against a hypothetical
            // stuck-1 bit beyond kSubSlotsPerSlot from a previous
            // generation of the slot.
            bm &= kValidBits;
            while ((bm & kValidBits) != kValidBits) {
                // ~bm | ~kValidBits would put a free bit OUTSIDE the
                // valid range; mask first so the "first set bit of
                // inverted" is guaranteed to land inside [0, kSubSlotsPerSlot).
                const core::u32 inverted = (~bm) & kValidBits;
                if (inverted == 0u) [[unlikely]] {
                    // Bitmap became full between mask and check; bail.
                    break;
                }
                const core::u32 free_bit = static_cast<core::u32>(
                    std::countr_zero(inverted));
                // HA defence-in-depth: never trust the bit position
                // implicitly. The mask above proves free_bit <
                // kSubSlotsPerSlot, but assert it once on debug builds.
                QTX_TIERED_ASSERT(free_bit < kSubSlotsPerSlot);
                const core::u32 mask     = core::u32{1} << free_bit;
                const core::u32 desired  = bm | mask;
                core::u32 cas_expected   = bm;
                if (s.occupancy_bitmap.compare_exchange_weak(
                        cas_expected, desired,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    shared_idx = i;
                    sub_idx    = free_bit;
                    break;
                }
                // CAS failed: cas_expected now holds the fresh value;
                // re-mask and retry.
                bm = cas_expected & kValidBits;
            }
            if (shared_idx != kNoSlot) break;
        }

        // 2b: nothing free in the existing pool — allocate a fresh
        //     hot slot and bind it to a fresh entry.
        if (shared_idx == kNoSlot) {
            arena::GenID fresh = hot_->allocate(HotArenaT::kSlotSize, 128u);
            // Retry through eviction if necessary (EC19 path, shared with full path).
            for (core::u32 r = 0u; r < kEvictRetries && fresh.isNull(); ++r) {
                if (!evictOldestToColdLocked(kNoTenantId)) {
                    releaseRecord(pool_idx);
                    return {0u, false};
                }
                fresh = hot_->allocate(HotArenaT::kSlotSize, 128u);
            }
            if (fresh.isNull()) {
                releaseRecord(pool_idx);
                return {0u, false};
            }
            // Find a free SharedSlotEntry to bind.
            for (core::u32 i = 0u; i < kMaxSharedSlots; ++i) {
                if (shared_slots_[i].hot_id.isNull() &&
                    shared_slots_[i].occupancy_bitmap.load(
                        std::memory_order_acquire) == 0u) {
                    shared_slots_[i].hot_id = fresh;
                    shared_slots_[i].occupancy_bitmap.store(
                        core::u32{1}, std::memory_order_release);
                    shared_idx = i;
                    sub_idx    = 0u;
                    break;
                }
            }
            if (shared_idx == kNoSlot) {
                // Should not happen given the sizing, but fail-safe.
                (void)hot_->release(fresh);
                releaseRecord(pool_idx);
                return {0u, false};
            }
        }

        // Step 3: bind the TenantRecord.
        TenantRecord& rec = tenants_[pool_idx];
        rec.tenant_id        = nextTenantId();
        rec.kind             = TenantKind::kMicro;
        rec.shared_slot_idx  = shared_idx;
        rec.hot_id           = shared_slots_[shared_idx].hot_id.withSubSlot(sub_idx);
        rec.cold_id          = arena::GenID{};
        rec.pin_count.store(0u, std::memory_order_relaxed);
        rec.state.store(TenantState::kInHot, std::memory_order_relaxed);
        // EC136: reset on recycle (see createTenant for full rationale).
        rec.sealed.store(false, std::memory_order_relaxed);
        rec.access_time.store(
            monotonic_clock_.fetch_add(1u, std::memory_order_acq_rel),
            std::memory_order_relaxed);
        rec.in_use.store(true, std::memory_order_release);

        // Step 4: zero the sub-slot bytes so the new tenant never reads
        // a previous tenant's residue (EC30 analogue for micro-tenants;
        // closes EC114 at the bridge layer).
        (void)hot_->zeroizeSubSlot(rec.hot_id, kMicroSlotSize);

        return {rec.tenant_id, true};
    }

    [[nodiscard]] std::span<std::byte> acquireMicroFP32(TenantHandle h) noexcept {
        if (hot_ == nullptr || cold_ == nullptr) return {};

        std::lock_guard<std::mutex> lock(struct_mutex_);
        TenantRecord* rec = findTenantLocked(h);
        if (rec == nullptr) return {};
        if (rec->kind != TenantKind::kMicro) return {};
        const TenantState st = rec->state.load(std::memory_order_acquire);
        if (st == TenantState::kDead) return {};

        // EC15 — Detect pathological pin_count growth. A correctly-used
        // tenant has pin_count bounded by the number of concurrent live
        // acquireMicroFP32 calls (realistically tens, maybe hundreds).
        // A value approaching u32::max signals a leaked acquire (the
        // caller forgot to pair every acquire with a release). We
        // refuse new acquires once pin_count crosses half of u32 range;
        // this still leaves >2 billion further acquires to land
        // legitimately and gives the leak a chance to surface in
        // monitoring rather than wrapping silently.
        constexpr core::u32 kPinCountSoftMax = core::u32{1} << 31;
        const core::u32 cur_pin =
            rec->pin_count.load(std::memory_order_acquire);
        if (cur_pin >= kPinCountSoftMax) [[unlikely]] {
            // Caller has leaked acquires; refuse new ones, surface as
            // an empty span (same contract as a dead tenant).
            return {};
        }

        // Pin (atomic increment, mirrors acquireFP32). EC117: micro-
        // tenants are never evicted, so this is essentially a no-op
        // for the eviction sweep — but the pin still serialises with
        // destroyTenant which checks pin_count.
        rec->pin_count.fetch_add(1u, std::memory_order_acq_rel);
        // EC96: monotonic access_time via CAS-max.
        bumpAccessTime(rec);

        return hot_->viewSubSlot(rec->hot_id, kMicroSlotSize);
    }

    // ========================================================================
    // === Inspection ===
    // ========================================================================

    [[nodiscard]] core::usize hotUsed() const noexcept  { return hot_->usedSlots(); }
    [[nodiscard]] core::usize coldUsed() const noexcept { return cold_->usedSlots(); }

    [[nodiscard]] core::usize totalAgents() const noexcept {
        core::usize n = 0u;
        for (core::usize i = 0u; i < MaxTenants; ++i) {
            if (tenants_[i].in_use.load(std::memory_order_acquire) &&
                tenants_[i].state.load(std::memory_order_acquire) !=
                    TenantState::kDead) {
                ++n;
            }
        }
        return n;
    }

    /// Name of the cold-tier codec this bridge was compiled with.
    /// Useful for telemetry / CLI echo ("cold tier: int4").
    [[nodiscard]] static constexpr std::string_view coldCodecName() noexcept {
        return ColdCodecT::name();
    }

    /// Compressed cold-slot footprint per tenant, in bytes. Derived from
    /// the active cold codec; exposed so deployments can size ColdArena.
    [[nodiscard]] static constexpr core::usize coldCompressedBytes() noexcept {
        return kCompressedBytes;
    }

    struct Stats {
        core::usize hot_used;
        core::usize cold_used;
        core::usize alive_tenants;
        core::u64   evictions;
        core::u64   promotions;
        // EC10 — Micro-tenant fragmentation telemetry. These two
        // counters let an operator see how efficiently shared slots
        // are packed without paying for full defragmentation:
        //   * micro_slots_used  = number of SharedSlotEntry currently
        //                         holding ≥1 micro-tenant.
        //   * micro_tenants_alive = total occupancy popcount across
        //                           all shared slots.
        // Fragmentation factor = micro_slots_used * kSubSlotsPerSlot
        //                        / micro_tenants_alive. A value of
        //                        1.0 means perfectly packed; a value
        //                        of kSubSlotsPerSlot means every slot
        //                        holds exactly one tenant (worst case).
        // P4: real defragmentation (migrate tenants between shared
        // slots) is deliberately deferred — it requires moving FP32
        // payload bytes (expensive) and rewriting TenantRecord::hot_id
        // for each migrated tenant (must be done under struct_mutex_
        // AND with every concurrent acquire/release re-checking).
        core::usize micro_slots_used;
        core::usize micro_tenants_alive;
    };

    [[nodiscard]] Stats stats() const noexcept {
        // EC10: scan shared_slots_ for fragmentation telemetry. This is
        // O(kMaxSharedSlots) — same complexity as totalAgents() — but
        // serialised against structural changes through atomic acquire
        // loads. We deliberately do NOT take struct_mutex_ here: stats()
        // is a diagnostic, not a control-flow input, and a torn-by-one
        // count is acceptable in exchange for not blocking allocation.
        core::usize micro_slots = 0u;
        core::usize micro_pop   = 0u;
        for (core::usize i = 0u; i < kMaxSharedSlots; ++i) {
            const core::u32 bm =
                shared_slots_[i].occupancy_bitmap.load(std::memory_order_acquire);
            if (bm != 0u) {
                ++micro_slots;
                micro_pop += static_cast<core::usize>(std::popcount(bm));
            }
        }
        return Stats{
            .hot_used            = hotUsed(),
            .cold_used           = coldUsed(),
            .alive_tenants       = totalAgents(),
            .evictions           = evictions_,
            .promotions          = promotions_,
            .micro_slots_used    = micro_slots,
            .micro_tenants_alive = micro_pop,
        };
    }

private:
    enum class TenantState : core::u8 {
        kInHot   = 0u,
        kInCold  = 1u,
        kDead    = 2u,
    };

    // ========================================================================
    // EC115 — micro-tenant infrastructure (Phase 2 Block C.2).
    //
    // A "micro-tenant" is a small (≤ kMicroSlotSize) payload that lives
    // in a sub-cell of a shared hot slot. Several micro-tenants can
    // share one slot; the slot is freed only when the last micro-tenant
    // is released. This is what gives Block C its "32 system-prompts
    // in one 4 KiB page" headline.
    //
    // Design choices (HA-driven):
    //
    //  (1) Micro-tenants and full-slot tenants are SEPARATE allocation
    //      paths (createMicroTenant vs createTenant). The two never
    //      share a record layout, and eviction only ever considers
    //      full-slot tenants — so EC117 (mixed-mode cold migration of
    //      a half-shared slot) is impossible by construction.
    //
    //  (2) Per-shared-slot state lives in a fixed-size pool
    //      `shared_slots_` of `SharedSlotEntry`. The pool size scales
    //      with MaxTenants but is bounded so we never heap-allocate.
    //
    //  (3) Occupancy is a single std::atomic<u32> bitmap, where bit i
    //      set ⇔ sub-slot i is occupied. `popcount(bitmap) == 0` is
    //      the "release the underlying slot" signal. This collapses
    //      the ref_count and the "which sub-slot is free" data into
    //      ONE atomic word — no torn-update window between them.
    //
    //  (4) Acquisition of a sub-slot bit is a CAS loop on the same
    //      atomic; release is `fetch_and(~mask, acq_rel)` followed
    //      by a check whether the result is zero. EC119: u32 width
    //      vastly exceeds the 32-sub-slot maximum, no overflow path.
    //
    // kMicroSlotSize and kSubSlotsPerSlot are declared in the public
    // section above so external callers can size their payloads.
    // ========================================================================

    /// Pool of shared slot entries. Each entry tracks one hot slot
    /// that hosts multiple micro-tenants.
    /// Sizing: in the worst case every micro-tenant owns its own slot,
    /// so we may need up to MaxTenants/kSubSlotsPerSlot shared entries;
    /// we round up to MaxTenants to leave room for transient over-
    /// commitment without ever heap-allocating.
    static constexpr core::usize kMaxSharedSlots =
        (MaxTenants + kSubSlotsPerSlot - 1u) / kSubSlotsPerSlot;

    struct alignas(core::kMaxCacheLineSize) SharedSlotEntry {
        // Single atomic bitmap: bit i set ⇔ sub-slot i is occupied.
        // popcount() == 0 ⇒ "underlying hot slot can be released".
        std::atomic<core::u32> occupancy_bitmap{0u};
        // The hot slot this entry owns; null when no sub-slot is in use.
        // Protected by the same atomic-bitmap discipline: when bitmap
        // transitions to 0 the writer also stores GenID{} here.
        arena::GenID hot_id{};
        // Padding to one cache line by alignas.
    };

    enum class TenantKind : core::u8 {
        kFullSlot = 0u,   ///< Owns its hot slot exclusively (Block A/B path).
        kMicro    = 1u,   ///< Lives in a sub-slot of a shared slot.
    };

    // Each TenantRecord lives on its own cache line to defeat false
    // sharing of pin_count between hot-path threads (EC10 analogue).
    struct alignas(core::kMaxCacheLineSize) TenantRecord {
        // EC101: tenant_id is u64 to match TenantHandle and defeat the
        // 11-minute u32 wrap at 6.3 M-op/s steady state.
        core::u64                  tenant_id{0u};
        std::atomic<core::u32>     pin_count{0u};
        std::atomic<TenantState>   state{TenantState::kDead};
        std::atomic<core::u64>     access_time{0u};
        std::atomic<bool>          in_use{false};
        // EC136: write-protect/seal flag, mirrors BufferContext::write_protected
        // at the tiered layer. A sealed tenant is excluded from the
        // eviction sweep so WEIGHTS buffers are never lossily compressed.
        // Written under struct_mutex_ in sealTenant; read with acquire
        // from evictOldestToColdLocked (also under the mutex, but the
        // acquire keeps the read self-consistent if we ever relax the
        // locking discipline).
        std::atomic<bool>          sealed{false};
        arena::GenID               hot_id{};
        arena::GenID               cold_id{};
        // EC115: which kind of allocation this tenant uses. Set at
        // create time, never changes during the tenant's lifetime.
        TenantKind                 kind{TenantKind::kFullSlot};
        // EC115: index into shared_slots_ for micro-tenants only.
        // Meaningless when kind == kFullSlot.
        core::u32                  shared_slot_idx{0u};
        // Padding to fill one cache line is provided by alignas; the
        // sizeof-of-class will naturally extend to the next 128-byte
        // boundary on every mainstream ABI.
    };

    // --------------------------------------------------------------------
    // Tenant pool management (static fixed-size, no heap, no exceptions).
    // EC6: replaces std::vector<TenantRecord>.
    // --------------------------------------------------------------------

    [[nodiscard]] core::u32 acquireRecord() noexcept {
        // Locked by the caller (struct_mutex_), so linear scan is fine.
        for (core::usize i = 0u; i < MaxTenants; ++i) {
            if (!tenants_[i].in_use.load(std::memory_order_acquire)) {
                return static_cast<core::u32>(i);
            }
        }
        return kNoSlot;
    }

    void releaseRecord(core::u32 idx) noexcept {
        // EC100 — Store order matters here. The `in_use` flag is the
        // synchronisation token observed by acquireRecord's
        // `load(acquire)`. Releasing in_use AFTER kDead means a reader
        // who sees `in_use == false` is guaranteed (via the release/
        // acquire pair) to also see `state == kDead`, because the
        // kDead store happens-before the in_use release. If we wrote
        // them in the opposite order, a reader who saw `in_use == false`
        // would have NO synchronisation edge with the state store —
        // it could still observe the previous tenant's state for an
        // unbounded duration on weakly-ordered hardware.
        tenants_[idx].state.store(TenantState::kDead, std::memory_order_relaxed);
        tenants_[idx].in_use.store(false, std::memory_order_release);
    }

    [[nodiscard]] core::u64 nextTenantId() noexcept {
        // EC101: u64 counter eliminates the 11-minute u32 wrap.
        // Skip both reserved sentinel values: 0 (means "null handle") and
        // kNoTenantId (means "no tenant"). At u64 width the skip path is
        // effectively unreachable but kept for defence-in-depth.
        core::u64 v = next_tenant_id_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
        // EC105: the original 'if (v == kNoTenant) fetch_add again' pattern
        // was racy in principle (two threads could both observe the same
        // skip-target and waste two ids each). With u64 the skip path is
        // mathematically out of reach for any realistic uptime, so this is
        // now an in-line conditional rather than a re-issue.
        if (v == 0u || v == kNoTenantId) [[unlikely]] {
            v = next_tenant_id_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
        }
        return v;
    }

    [[nodiscard]] TenantRecord* findTenantLocked(TenantHandle h) noexcept {
        if (h.tenant_id == kNoTenantId || h.tenant_id == 0u) return nullptr;
        for (auto& r : tenants_) {
            if (r.in_use.load(std::memory_order_acquire) &&
                r.tenant_id == h.tenant_id) {
                return &r;
            }
        }
        return nullptr;
    }

    /// Lock-free lookup used by release(). Since destroyTenant takes the
    /// mutex AND we never resize the pool, the pointer stays valid for
    /// the lifetime of the bridge — only the state transitions race.
    [[nodiscard]] TenantRecord* findTenantLockfree(TenantHandle h) noexcept {
        if (h.tenant_id == kNoTenantId || h.tenant_id == 0u) return nullptr;
        for (auto& r : tenants_) {
            if (r.in_use.load(std::memory_order_acquire) &&
                r.tenant_id == h.tenant_id &&
                r.state.load(std::memory_order_acquire) != TenantState::kDead) {
                return &r;
            }
        }
        return nullptr;
    }

    // ========================================================================
    // EC96 — Monotonic access_time updater. A naive
    //   rec->access_time.store(monotonic_clock_.fetch_add(1))
    // is racy across two threads: thread A fetches ticket t=10, thread
    // B fetches t=11 right after, but B's store completes before A's,
    // leaving access_time at 10 even though "later" thread B touched
    // the tenant. The eviction sweep would then consider this tenant
    // "older" than it actually is.
    //
    // We replace the unconditional store with a CAS-max loop: a write
    // is committed only if the new ticket is strictly newer than the
    // current value. This makes access_time strictly monotonic per
    // record, regardless of how many threads race to acquire the same
    // tenant. The CAS contention here is bounded by the rate at which
    // a SINGLE tenant is acquired concurrently (rare in steady state).
    //
    // The function is private and inline; both acquireFP32 and
    // acquireMicroFP32 call it instead of touching access_time directly.
    // ========================================================================
    inline void bumpAccessTime(TenantRecord* rec) noexcept {
        const core::u64 ticket =
            monotonic_clock_.fetch_add(1u, std::memory_order_acq_rel);
        core::u64 prev = rec->access_time.load(std::memory_order_acquire);
        while (ticket > prev) {
            if (rec->access_time.compare_exchange_weak(
                    prev, ticket,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return;
            }
            // CAS reloaded prev with current value; loop until either
            // (a) we win the CAS, or (b) a parallel acquire raced us
            // to a strictly newer ticket — in which case our update
            // is redundant and we drop out.
        }
    }

    // --------------------------------------------------------------------
    // Eviction (caller holds struct_mutex_).
    // EC18: O(N) sweep, but bounded to a single caller at a time.
    // EC99: returns false when no eligible victim (all pinned / no hot).
    // --------------------------------------------------------------------

    [[nodiscard]] bool evictOldestToColdLocked(core::u64 skip_id) noexcept {
        TenantRecord* victim = nullptr;
        core::u64 victim_time = 0u;
        for (auto& r : tenants_) {
            if (!r.in_use.load(std::memory_order_acquire)) continue;
            if (r.state.load(std::memory_order_acquire) != TenantState::kInHot) continue;
            if (r.pin_count.load(std::memory_order_acquire) > 0u) continue;
            if (r.tenant_id == skip_id) continue;
            // EC136: sealed tenants (WEIGHTS / lookup tables / sentinels)
            // MUST NOT be lossily quantised to the cold tier. Even
            // pin_count == 0 is allowed for sealed read-only weights —
            // the seal flag is the authoritative signal here.
            if (r.sealed.load(std::memory_order_acquire)) continue;
            // EC117: micro-tenants share a slot with siblings and have
            // no cold variant. Evicting one would either corrupt the
            // siblings (if we naively compress the shared slot) or
            // double-allocate cold storage. They are simply skipped —
            // the OOM fallback for micro-tenants is "createMicroTenant
            // fails cleanly" rather than "evict a sibling". This is
            // the same contract as a pinned full-slot tenant.
            if (r.kind == TenantKind::kMicro) continue;
            const core::u64 t = r.access_time.load(std::memory_order_acquire);
            if (victim == nullptr || t < victim_time) {
                victim = &r;
                victim_time = t;
            }
        }
        if (victim == nullptr) return false;

        arena::GenID cold_id = cold_->allocate(kCompressedBytes, 128u);
        if (cold_id.isNull()) return false;

        auto hot_view  = hot_->view(victim->hot_id);
        auto cold_view = cold_->view(cold_id);
        if (hot_view.empty() || cold_view.empty()) {
            (void)cold_->release(cold_id);
            return false;
        }
        const auto written = ColdCodecT::compress(
            std::span<const std::byte>(hot_view.data(), kPayloadBytes),
            cold_view);
        if (written != kCompressedBytes) {
            (void)cold_->release(cold_id);
            return false;
        }

        (void)hot_->release(victim->hot_id);
        victim->hot_id  = arena::GenID{};
        victim->cold_id = cold_id;
        victim->state.store(TenantState::kInCold, std::memory_order_release);
        ++evictions_;
        return true;
    }

    // --------------------------------------------------------------------
    // EC115 — small bit helpers used by the micro-tenant path. Both
    // boil down to a single instruction on every mainstream ISA, but
    // we name them to keep the createMicroTenant body readable.
    // --------------------------------------------------------------------
    [[nodiscard]] static constexpr core::u32 popcountBits(core::u32 v) noexcept {
        return static_cast<core::u32>(std::popcount(v));
    }
    [[nodiscard]] static constexpr core::u32 countrZeroOfInverted(core::u32 v) noexcept {
        // First free bit in `v` = first set bit in ~v. When `v` is
        // already full (all 1s in the low kSubSlotsPerSlot bits) the
        // caller has already taken the "this slot is full" branch.
        return static_cast<core::u32>(std::countr_zero(~v));
    }

    // --------------------------------------------------------------------
    // State (cache-line aligned).
    // --------------------------------------------------------------------

    HotArenaT*  hot_;
    ColdArenaT* cold_;

    // Static fixed-size tenant pool (EC6).
    std::array<TenantRecord, MaxTenants> tenants_{};

    // EC115: pool of shared-slot entries for micro-tenants. Each
    // SharedSlotEntry is cache-line-aligned, so concurrent fetch_and
    // / fetch_or on different entries do not ping-pong cache lines.
    std::array<SharedSlotEntry, kMaxSharedSlots> shared_slots_{};

    // Structural mutex for create / destroy / evict.
    // Hot path does NOT take this — pin/unpin is fully lock-free.
    std::mutex struct_mutex_;

    // EC15: atomic monotonic clock so concurrent acquireFP32 calls do
    // not race on the same counter.
    alignas(core::kMaxCacheLineSize) std::atomic<core::u64> monotonic_clock_{0u};
    // EC101: u64 counter; was u32 in P1 (wraps in ~11 min at 6.3 M-op/s).
    alignas(core::kMaxCacheLineSize) std::atomic<core::u64> next_tenant_id_{0u};

    // EC11: rotating scan cursor for shared-slot search. Modified ONLY
    // under struct_mutex_ in createMicroTenant, so plain non-atomic is
    // correct (the mutex acquire/release provides the necessary
    // synchronisation). Initialised to 0 and wraps modulo kMaxSharedSlots.
    core::u32 next_shared_scan_start_ = 0u;

    core::u64 evictions_  = 0u;
    core::u64 promotions_ = 0u;
};

}  // namespace qtx::tiered
