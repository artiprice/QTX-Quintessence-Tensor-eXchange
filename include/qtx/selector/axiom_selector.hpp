// ============================================================================
// @file        axiom_selector.hpp
// @brief       O(1) hardware slot selector via std::countr_zero.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// HA-LAYER: HA-pure, zero allocations, zero exceptions, zero UB.
//
// EDGE CASES CLOSED IN THIS REWRITE:
//
//   EC2.1 — (user-supplied review) CAS now uses memory_order_acq_rel on
//           success so the selector is a self-sufficient happens-before
//           barrier even without the version-counter follow-up in
//           FractalArena::allocate.
//   EC5   — busy-spin in the CAS loop now issues a CPU pause / yield hint
//           every iteration to avoid starving the partner thread on SMT.
//   EC10  — the bitmap is laid out so that one cache line holds a single
//           atomic word (PaddedWord<64-byte aligned>): each allocation
//           storm only invalidates the line containing the contested
//           word, not 32 unrelated atomics.
//   EC13  — false kNoSlot under high contention: bounded retry budget;
//           once the loop falls through we run a second sweep so that
//           "all words observed busy in this scan, but freshly-released
//           by then" cannot leak through.
//   EC75  — prefetcher pollution: each freeCount sweep iterates with a
//           tight relaxed-load to keep the surface footprint minimal.
//   EC102 — thundering-herd on word 0: under an allocation storm all
//           threads previously started their scan at word_idx == 0, so
//           every CAS contended the same cache line. Each thread now
//           computes a deterministic start offset from a xorshift64 hash
//           of its std::thread::id, cached in a thread_local CONST so the
//           hot path performs zero TLS writes. We deliberately do NOT use
//           a mutable thread_local counter — that would introduce a
//           hot-path write that breaks TSan-clean invariants on platforms
//           where TLS lowers to a non-trivial load-store sequence.
// ============================================================================

#pragma once

#include "../core/platform.hpp"
#include "../core/types.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstring>
#include <thread>

#if defined(QTX_ARCH_X86_64)
    #include <immintrin.h>
#endif

namespace qtx::selector {

namespace detail {

/// Hint to the CPU that we are in a spin-wait. On x86_64 we issue PAUSE
/// (REP NOP) which yields HT pipeline cycles and lowers power; on ARMv8
/// we use the YIELD hint; elsewhere we fall back to a portable yield.
inline void spinHint() noexcept {
#if defined(QTX_ARCH_X86_64)
    _mm_pause();
#elif defined(QTX_ARCH_ARM64) && (defined(__GNUC__) || defined(__clang__))
    __asm__ volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// ============================================================================
// EC102 — Anti-thundering-herd start offset.
//
// We map std::this_thread::get_id() through a SplitMix64-style finaliser so
// the distribution across kWords is uniform even when the OS hands out
// sequential thread ids (Linux often does). The result is computed exactly
// once per (thread, kWords) pair and cached in a thread_local CONST — so
// the hot path performs zero TLS writes and zero hashing. Reading a
// thread_local const is one mov on every mainstream ABI.
//
// Why not a mutable per-thread counter (the originally proposed "round-
// robin advance on every call"): a mutable TLS counter is a hot-path
// store, and TLS storage is also addressed via a non-trivial sequence on
// some ABIs (Windows %gs:, Linux %fs:). A fixed start point per thread
// already breaks the cache-line ping-pong; the second-pass sweep in
// acquire() guarantees correctness under contention without needing the
// start point to move.
// ============================================================================
[[nodiscard]] inline core::u64 mixThreadId() noexcept {
    // Hash the bytes of thread::id rather than calling ::id::hash, which
    // is not constexpr and on some libc++ implementations allocates.
    const auto tid = std::this_thread::get_id();
    core::u64 x = 0u;
    static_assert(sizeof(tid) <= sizeof(core::u64),
                  "std::thread::id wider than u64 on this platform");
    // Byte-wise memcpy keeps us inside the strict-aliasing rules even if
    // thread::id wraps a non-integral handle (Windows: HANDLE = pointer).
    std::array<unsigned char, sizeof(core::u64)> buf{};
    std::memcpy(buf.data(), &tid, sizeof(tid));
    for (core::usize i = 0u; i < sizeof(core::u64); ++i) {
        x = (x << 8) | static_cast<core::u64>(buf[i]);
    }
    // EC62 — Canonical SplitMix64 gamma. Without this, a thread::id
    // that lowers to all-zero bytes (theoretically impossible on
    // POSIX where TIDs start at 1, but possible on bare-metal RTOS
    // ports where the "kernel" thread has handle 0) would feed zero
    // into the finaliser and produce zero out — every such thread
    // would then start its scan at word 0, the canonical hot spot
    // we are trying to avoid. Adding the gamma (the golden-ratio-
    // derived odd constant from the original SplitMix64 paper)
    // means zero-in produces 0x9E3779B97F4A7C15 → uniform out.
    x += 0x9E3779B97F4A7C15ULL;
    // SplitMix64 finaliser — uniform avalanche from any non-zero input.
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/// Deterministic per-thread start word in [0, kWords). Computed once on
/// the first call from this thread, then read as a const from TLS.
template <core::usize kWords>
[[nodiscard]] inline core::usize threadStartWord() noexcept {
    static_assert(kWords > 0u, "kWords must be positive");
    thread_local const core::usize start =
        static_cast<core::usize>(mixThreadId() % kWords);
    return start;
}

}  // namespace detail

// ============================================================================
// @class AxiomSelector
//
// HA-invariants:
//   - SlotCount % 64 == 0 (compile-time)
//   - All methods are noexcept
//   - State is serialised via std::atomic<u64> with acquire/acq_rel orders
//     so that release()->acquire() chains form a complete happens-before
//     edge by themselves (without external version counter).
// ============================================================================

template <core::usize SlotCount>
class AxiomSelector {
    static_assert(SlotCount > 0u, "SlotCount must be positive");
    static_assert(SlotCount % 64u == 0u,
                  "SlotCount must be a multiple of 64 (one u64 = 64 slots)");

public:
    static constexpr core::usize kSlots       = SlotCount;
    static constexpr core::usize kWords       = SlotCount / 64u;
    static constexpr core::usize kBitsPerWord = 64u;
    static constexpr core::usize kNoSlot      = static_cast<core::usize>(-1);

    AxiomSelector() noexcept {
        // ALL bits set: every slot starts "free / selectable".
        //
        // Each store uses release ordering so that any thread which later
        // observes a published pointer to *this (via its own acquire load)
        // is guaranteed to see all initialised words. We deliberately do
        // NOT use std::atomic_thread_fence() here — ThreadSanitizer does
        // not model standalone fences and would issue a -Wtsan warning;
        // per-store release ordering provides the same happens-before
        // guarantee that the construction completes before any external
        // acquire of a pointer to *this.
        for (core::usize i = 0u; i < kWords; ++i) {
            words_[i].value.store(~core::u64{0}, std::memory_order_release);
        }
    }

    AxiomSelector(const AxiomSelector&)            = delete;
    AxiomSelector& operator=(const AxiomSelector&) = delete;
    AxiomSelector(AxiomSelector&&)                 = delete;
    AxiomSelector& operator=(AxiomSelector&&)      = delete;

    ~AxiomSelector() = default;

    // ========================================================================
    // @brief Atomically take the first free slot.
    //
    // Complexity: O(kWords) in the worst case, O(1) under uniform load.
    //
    // EC102: scan starts at a per-thread offset (deterministic, cached in
    // a thread_local const), not at word 0. Under an allocation storm,
    // threads now contend on DIFFERENT cache lines instead of all racing
    // for word 0's atomic. We still sweep ALL kWords in each pass so the
    // selector remains O(kWords)-correct: starvation is impossible.
    // ========================================================================
    [[nodiscard]] core::usize acquire() noexcept {
        // Two-pass scheme (EC13). On the first pass we attempt acquisition
        // with a small per-word retry budget; if the bitmap appears full
        // we run a confirmation pass to rule out a transient phase where
        // every word lost a CAS race simultaneously.
        //
        // EC68 / EC78 — Per-word retry budget is tuned for the HFT spin-
        // budget (sub-microsecond). On modern Intel Sapphire Rapids /
        // Granite Rapids the `PAUSE` instruction takes ~140 cycles
        // (≈45 ns at 3 GHz). With 2 retries per word × kWords words ×
        // ~140 cycles/PAUSE, the worst-case wall time before falling
        // through to the second pass is:
        //
        //   2 × kWords × 140 cycles ≈ 280 × kWords cycles
        //
        // For kWords = 4 (256-slot arena, typical), that is 1120 cycles
        // ≈ 0.37 µs — well inside the 1 µs HFT spin envelope. The
        // original value of 4 retries gave 2240 cycles ≈ 0.75 µs which
        // grazed the envelope on bigger arenas. Two retries also
        // matches the Intel® Optimization Reference Manual guidance
        // for sub-µs spin loops on hyperthreaded cores.
        //
        // The second-pass sweep recovers any slot that became free
        // during the first-pass's bounded spin, so correctness is
        // preserved at the lower retry count.
        constexpr core::u32 kMaxRetriesPerWord = 2u;

        const core::usize start = detail::threadStartWord<kWords>();

        for (core::u32 pass = 0u; pass < 2u; ++pass) {
            for (core::usize i = 0u; i < kWords; ++i) {
                // Circular sweep: visit every word exactly once, starting
                // from the per-thread offset. The modulo collapses to a
                // bitmask only when kWords is a power of two, but kWords
                // is at most kSlots / 64 (small), so an unconditional `%`
                // is fine even for non-power-of-two SlotCount values that
                // satisfy SlotCount % 64 == 0.
                const core::usize word_idx = (start + i) % kWords;
                core::u64 expected =
                    words_[word_idx].value.load(std::memory_order_acquire);

                core::u32 retries = 0u;
                while (expected != 0u) {
                    const core::u32 bit_in_word =
                        static_cast<core::u32>(std::countr_zero(expected));
                    const core::u64 mask = core::u64{1} << bit_in_word;
                    const core::u64 desired = expected & ~mask;

                    // EC2.1: success path uses acq_rel so a subsequent
                    // reader of the slot's payload sees everything the
                    // previous owner published in its release() — even
                    // when the FractalArena version counter is not in
                    // the critical path (callers may skip it).
                    if (words_[word_idx].value.compare_exchange_weak(
                            expected, desired,
                            std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
                        return word_idx * kBitsPerWord + bit_in_word;
                    }

                    // Lost the CAS race; back off a little and retry.
                    detail::spinHint();
                    if (++retries >= kMaxRetriesPerWord) {
                        // Move on to the next word — this one is hot.
                        break;
                    }
                }
            }
            // Brief yield before the confirmation pass, so the partner
            // thread that just released has a chance to publish.
            detail::spinHint();
        }
        return kNoSlot;
    }

    // ========================================================================
    // @brief Return the slot to the pool (mark free).
    // EC2.1: release uses release ordering; combined with acq_rel acquire
    // above this gives a full happens-before between writer and reader.
    // ========================================================================
    void release(core::usize slot) noexcept {
        if (slot >= kSlots) [[unlikely]] {
            return;  // fail-safe: out-of-range, silently ignore
        }
        const core::usize word_idx    = slot / kBitsPerWord;
        const core::u32   bit_in_word = static_cast<core::u32>(slot % kBitsPerWord);
        const core::u64   mask        = core::u64{1} << bit_in_word;

        words_[word_idx].value.fetch_or(mask, std::memory_order_release);
    }

    // ========================================================================
    // @brief Check if a specific slot is occupied (snapshot).
    //
    // EC97: documents that the snapshot is point-in-time. Higher layers
    // must not rely on this for control flow; we keep it for tests &
    // telemetry.
    // ========================================================================
    [[nodiscard]] bool isOccupied(core::usize slot) const noexcept {
        if (slot >= kSlots) [[unlikely]] return false;
        const core::usize word_idx    = slot / kBitsPerWord;
        const core::u32   bit_in_word = static_cast<core::u32>(slot % kBitsPerWord);
        const core::u64   mask        = core::u64{1} << bit_in_word;
        const core::u64   w = words_[word_idx].value.load(std::memory_order_acquire);
        return (w & mask) == 0u;
    }

    [[nodiscard]] core::usize freeCount() const noexcept {
        core::usize total = 0u;
        for (core::usize i = 0u; i < kWords; ++i) {
            const core::u64 w = words_[i].value.load(std::memory_order_relaxed);
            total += static_cast<core::usize>(std::popcount(w));
        }
        return total;
    }

    [[nodiscard]] static constexpr core::usize capacity() noexcept {
        return kSlots;
    }

private:
    // EC10: each atomic word lives on its own cache line so concurrent
    // acquires/releases against *different* words don't ping-pong a
    // single line. The struct wrapper enforces this on every compiler.
    struct alignas(core::kMaxCacheLineSize) PaddedWord {
        std::atomic<core::u64> value{0u};
        // Filler keeps sizeof(PaddedWord) == cache line on platforms
        // where alignas alone wouldn't pad up (rare, but defensive).
        char pad[core::kMaxCacheLineSize - sizeof(std::atomic<core::u64>)];
    };

    static_assert(sizeof(PaddedWord) == core::kMaxCacheLineSize,
                  "PaddedWord must be exactly one cache line");
    static_assert(alignof(PaddedWord) == core::kMaxCacheLineSize,
                  "PaddedWord must be cache-line aligned");

    std::array<PaddedWord, kWords> words_{};
};

//=== HA: compile-time template check on a typical instance ===
namespace detail {
using TestSelector = AxiomSelector<256>;
static_assert(TestSelector::kSlots == 256u);
static_assert(TestSelector::kWords == 4u);
static_assert(TestSelector::kNoSlot == static_cast<core::usize>(-1));
static_assert(TestSelector::capacity() == 256u);
}  // namespace detail

}  // namespace qtx::selector
