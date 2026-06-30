// ============================================================================
// @file        test_axiom_selector.cpp
//@brief Unit tests AxiomSelector: acquire/release, exhaustion, MT-safety.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================

#include "test_harness.hpp"
#include "qtx/selector/axiom_selector.hpp"

#include <array>
#include <atomic>
#include <set>
#include <thread>
#include <vector>

using namespace qtx::selector;
using qtx::core::usize;

// ============================================================================
//Basic life: capacity, init, simple acquire
// ============================================================================

QTX_TEST(Selector, CapacityMatchesTemplate) {
    AxiomSelector<64> s64;
    AxiomSelector<128> s128;
    AxiomSelector<256> s256;
    QTX_EXPECT_EQ(s64.capacity(), 64u);
    QTX_EXPECT_EQ(s128.capacity(), 128u);
    QTX_EXPECT_EQ(s256.capacity(), 256u);
}

QTX_TEST(Selector, InitialStateAllFree) {
    AxiomSelector<128> s;
    QTX_EXPECT_EQ(s.freeCount(), 128u);
    for (usize i = 0; i < 128; ++i) {
        QTX_EXPECT(!s.isOccupied(i));
    }
}

QTX_TEST(Selector, SingleAcquireReturnsZero) {
    AxiomSelector<64> s;
    //The first bit is the least significant bit, tzcnt will return 0.
    auto slot = s.acquire();
    QTX_EXPECT_EQ(slot, 0u);
    QTX_EXPECT(s.isOccupied(0u));
    QTX_EXPECT_EQ(s.freeCount(), 63u);
}

QTX_TEST(Selector, SequentialAcquireFillsLowToHigh) {
    AxiomSelector<64> s;
    for (usize expected = 0; expected < 64; ++expected) {
        auto slot = s.acquire();
        QTX_EXPECT_EQ(slot, expected);
    }
    QTX_EXPECT_EQ(s.freeCount(), 0u);
}

QTX_TEST(Selector, ExhaustionReturnsNoSlot) {
    AxiomSelector<64> s;
    for (usize i = 0; i < 64; ++i) {
        (void)s.acquire();
    }
    auto slot = s.acquire();
    QTX_EXPECT_EQ(slot, AxiomSelector<64>::kNoSlot);
}

// ============================================================================
//Multi-word: checking the transition between uint64_t words
// ============================================================================

QTX_TEST(Selector, MultiWordTraversal) {
    AxiomSelector<256> s;  //4 words
    //Take all 256 - you must fill in all 4 words in order.
    std::set<usize> seen;
    for (usize i = 0; i < 256; ++i) {
        auto slot = s.acquire();
        QTX_EXPECT(slot != AxiomSelector<256>::kNoSlot);
        QTX_EXPECT_EQ(seen.count(slot), 0u);  //no duplicates
        seen.insert(slot);
    }
    QTX_EXPECT_EQ(seen.size(), 256u);
    QTX_EXPECT_EQ(s.freeCount(), 0u);
}

// ============================================================================
// Release / reuse
// ============================================================================

QTX_TEST(Selector, ReleaseRestoresSlot) {
    AxiomSelector<64> s;
    auto slot = s.acquire();
    QTX_EXPECT(s.isOccupied(slot));
    s.release(slot);
    QTX_EXPECT(!s.isOccupied(slot));
    QTX_EXPECT_EQ(s.freeCount(), 64u);
}

QTX_TEST(Selector, ReleasedSlotReused) {
    AxiomSelector<64> s;
    auto a = s.acquire();  // 0
    auto b = s.acquire();  // 1
    auto c = s.acquire();  // 2
    s.release(b);           //release 1

    auto d = s.acquire();
    //The selector takes the least significant bit set => 1.
    QTX_EXPECT_EQ(d, 1u);
    (void)a; (void)c;
}

QTX_TEST(Selector, ReleaseInvalidIndexIsNoOp) {
    AxiomSelector<64> s;
    s.release(999u);  // out of range
    QTX_EXPECT_EQ(s.freeCount(), 64u);  //nothing has changed
}

QTX_TEST(Selector, DoubleReleaseIsIdempotent) {
    AxiomSelector<64> s;
    auto slot = s.acquire();
    s.release(slot);
    s.release(slot);  //again
    QTX_EXPECT(!s.isOccupied(slot));
    QTX_EXPECT_EQ(s.freeCount(), 64u);
}

// ============================================================================
//Concurrent stress: 4 threads × 10000 acquire/release
// ============================================================================

QTX_TEST(Selector, ConcurrentAcquireRelease) {
    constexpr usize kSlots = 256;
    constexpr int kThreads = 4;
    constexpr int kIters = 5000;

    AxiomSelector<kSlots> s;
    std::atomic<int> conflicts{0};
    std::atomic<int> oom_events{0};

    auto worker = [&]() {
        for (int i = 0; i < kIters; ++i) {
            auto slot = s.acquire();
            if (slot == AxiomSelector<kSlots>::kNoSlot) {
                ++oom_events;
                continue;
            }
            //Minimal work with the slot
            if (!s.isOccupied(slot)) {
                ++conflicts;  //SERIOUSLY: the bit must be occupied
            }
            s.release(slot);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();

    QTX_EXPECT_EQ(conflicts.load(), 0);  //zero races
    QTX_EXPECT_EQ(s.freeCount(), kSlots);  //everything is freed
    //oom_events may be > 0, this is normal under high load.
}

// ============================================================================
//Conservation: number of free = capacity - number acquired
// ============================================================================

QTX_TEST(Selector, ConservationLaw) {
    AxiomSelector<128> s;
    std::vector<usize> acquired;
    for (int i = 0; i < 100; ++i) {
        auto slot = s.acquire();
        QTX_EXPECT(slot != AxiomSelector<128>::kNoSlot);
        acquired.push_back(slot);
    }
    QTX_EXPECT_EQ(s.freeCount(), 128u - 100u);

    for (auto slot : acquired) {
        s.release(slot);
    }
    QTX_EXPECT_EQ(s.freeCount(), 128u);
}

// ============================================================================
// EC102 — stochastic start offset: a single thread must still visit every
// word; the per-thread offset is a constant cached in TLS, so two calls
// from the same thread observe the same start. We can't introspect that
// constant directly without exposing internals, but we can verify the two
// observable consequences:
//   (a) acquire() still empties the entire bitmap (no word is unreachable);
//   (b) different threads end up acquiring slots from different cache
//       lines under contention.
// ============================================================================

QTX_TEST(Selector, EC102_SingleThreadStillExhausts) {
    // 1024 slots = 16 words. If the offset were ever computed as kWords
    // (out-of-range) we would miss some words. The implementation uses
    // `% kWords`, but this test pins the property at the API surface.
    constexpr usize kSlots = 1024;
    AxiomSelector<kSlots> s;

    std::vector<usize> taken;
    taken.reserve(kSlots);
    for (usize i = 0; i < kSlots; ++i) {
        auto slot = s.acquire();
        QTX_EXPECT(slot != AxiomSelector<kSlots>::kNoSlot);
        taken.push_back(slot);
    }
    QTX_EXPECT_EQ(s.freeCount(), 0u);

    // Verify every slot index in [0, kSlots) was returned exactly once.
    std::set<usize> unique_slots(taken.begin(), taken.end());
    QTX_EXPECT_EQ(unique_slots.size(), kSlots);
    QTX_EXPECT_EQ(*unique_slots.begin(), 0u);
    QTX_EXPECT_EQ(*unique_slots.rbegin(), kSlots - 1u);
}

QTX_TEST(Selector, EC102_MultiThreadCacheLineSpread) {
    // Under contention with the OLD scheme (start at word 0), all threads
    // would race for slots in word 0 first. With EC102 they fan out to
    // different start words. We measure this by counting how many DIFFERENT
    // words received at least one allocation in the first round of takes.
    //
    // We deliberately use only kThreads slots so each thread takes exactly
    // one slot and the distribution is observable.
    constexpr usize kSlots   = 1024;     // 16 words
    constexpr int   kThreads = 8;
    AxiomSelector<kSlots> s;

    std::array<std::atomic<usize>, kThreads> winners{};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&s, &winners, t]() {
            const usize slot = s.acquire();
            winners[t].store(slot, std::memory_order_release);
        });
    }
    for (auto& th : threads) th.join();

    std::set<usize> hit_words;
    for (int t = 0; t < kThreads; ++t) {
        const usize slot = winners[t].load(std::memory_order_acquire);
        QTX_EXPECT(slot != AxiomSelector<kSlots>::kNoSlot);
        hit_words.insert(slot / 64u);
    }

    // Property: we can't deterministically assert "exactly N different
    // words" (the hash and scheduling vary), but with 8 threads and 16
    // possible start words we expect at least 2 distinct words in the
    // overwhelming majority of runs. The OLD scheme would deterministically
    // give 1 (everyone races for word 0). We assert the weak floor.
    QTX_EXPECT(hit_words.size() >= 2u);
}

// ============================================================================
// EC102 — high-contention starvation guard: with kWords-circular sweep, no
// thread is ever locked out under a sustained alloc/release storm. We
// run the existing pattern from MultiThreadedNoConflict at twice the
// load and confirm zero kNoSlot returns.
// ============================================================================

QTX_TEST(Selector, EC102_NoStarvationUnderStorm) {
    constexpr usize kSlots = 256;
    constexpr int kThreads = 4;
    constexpr int kIterPerThread = 5000;
    AxiomSelector<kSlots> s;
    std::atomic<int> oom{0};

    auto worker = [&]() {
        for (int i = 0; i < kIterPerThread; ++i) {
            const usize slot = s.acquire();
            if (slot == AxiomSelector<kSlots>::kNoSlot) {
                oom.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            s.release(slot);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    // With kSlots == 256 and only kThreads == 4 concurrent acquirers,
    // OOM is impossible by counting argument: at any instant at most
    // kThreads slots are taken simultaneously.
    QTX_EXPECT_EQ(oom.load(), 0);
    QTX_EXPECT_EQ(s.freeCount(), kSlots);
}
