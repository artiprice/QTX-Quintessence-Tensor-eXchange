// ============================================================================
// @file        test_spsc_ring_buffer.cpp
//@brief Unit tests SPSC ring buffer: lifecycle, edge cases,
//              concurrent producer/consumer.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================

#include "test_harness.hpp"
#include "qtx/ipc/spsc_ring_buffer.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using namespace qtx::ipc;
using qtx::core::u32;
using qtx::core::u64;
using qtx::core::usize;

//A simple POD for a queue.
struct Message {
    u64 id;
    u64 payload;

    constexpr bool operator==(const Message&) const noexcept = default;
};
static_assert(qtx::core::ArenaStorable<Message>);

// ============================================================================
//Basic design and invariants
// ============================================================================

QTX_TEST(SPSC, ConstructorEmpty) {
    SPSCRingBuffer<Message, 64> q;
    QTX_EXPECT(q.empty());
    QTX_EXPECT(!q.full());
    QTX_EXPECT_EQ(q.size(), 0u);
}

QTX_TEST(SPSC, CapacityReflectsTemplate) {
    SPSCRingBuffer<Message, 64> q64;
    SPSCRingBuffer<Message, 256> q256;
    QTX_EXPECT_EQ(q64.capacity(), 64u);
    QTX_EXPECT_EQ(q256.capacity(), 256u);
}

QTX_TEST(SPSC, MaskIsCapacityMinusOne) {
    using Q = SPSCRingBuffer<Message, 128>;
    QTX_EXPECT_EQ(Q::kMask, 127u);
}

// ============================================================================
// Single-threaded push/pop
// ============================================================================

QTX_TEST(SPSC, PushThenPopRoundTrip) {
    SPSCRingBuffer<Message, 64> q;
    Message in{42u, 0xCAFEu};
    QTX_EXPECT(q.try_push(in));
    QTX_EXPECT_EQ(q.size(), 1u);

    auto out = q.try_pop();
    QTX_EXPECT(out.has_value());
    QTX_EXPECT(out.value() == in);
    QTX_EXPECT(q.empty());
}

QTX_TEST(SPSC, PopOnEmptyReturnsNullopt) {
    SPSCRingBuffer<Message, 64> q;
    auto out = q.try_pop();
    QTX_EXPECT(!out.has_value());
}

QTX_TEST(SPSC, PopOutParamOnEmptyReturnsFalse) {
    SPSCRingBuffer<Message, 64> q;
    Message tmp{};
    QTX_EXPECT(!q.try_pop(tmp));
}

QTX_TEST(SPSC, FillToCapacityThenPushReturnsFalse) {
    SPSCRingBuffer<Message, 16> q;
    for (u64 i = 0; i < 16u; ++i) {
        QTX_EXPECT(q.try_push({i, i * 2u}));
    }
    QTX_EXPECT_EQ(q.size(), 16u);
    QTX_EXPECT(q.full());
    //The 17th attempt should return false.
    QTX_EXPECT(!q.try_push({999u, 0u}));
}

QTX_TEST(SPSC, FIFOOrderPreserved) {
    SPSCRingBuffer<Message, 64> q;
    for (u64 i = 0; i < 50u; ++i) {
        QTX_EXPECT(q.try_push({i, i * 3u}));
    }
    for (u64 i = 0; i < 50u; ++i) {
        auto out = q.try_pop();
        QTX_EXPECT(out.has_value());
        QTX_EXPECT_EQ(out->id, i);
        QTX_EXPECT_EQ(out->payload, i * 3u);
    }
    QTX_EXPECT(q.empty());
}

// ============================================================================
//Wrap-around: indexes go over the buffer boundary
// ============================================================================

QTX_TEST(SPSC, IndicesWrapCorrectly) {
    SPSCRingBuffer<Message, 8> q;  //small buffer for quick wrap

    //We do 100 push+pop cycles across border 8.
    for (u64 round = 0; round < 100u; ++round) {
        for (u64 j = 0; j < 5u; ++j) {
            QTX_EXPECT(q.try_push({round * 10u + j, j}));
        }
        for (u64 j = 0; j < 5u; ++j) {
            auto out = q.try_pop();
            QTX_EXPECT(out.has_value());
            QTX_EXPECT_EQ(out->id, round * 10u + j);
        }
    }
    QTX_EXPECT(q.empty());
}

QTX_TEST(SPSC, AlternatingPushPopAtFullCapacity) {
    SPSCRingBuffer<Message, 4> q;

    QTX_EXPECT(q.try_push({1u, 0u}));
    QTX_EXPECT(q.try_push({2u, 0u}));
    QTX_EXPECT(q.try_push({3u, 0u}));
    QTX_EXPECT(q.try_push({4u, 0u}));
    QTX_EXPECT(!q.try_push({5u, 0u}));  // full

    auto a = q.try_pop();
    QTX_EXPECT(a.has_value() && a->id == 1u);

    QTX_EXPECT(q.try_push({5u, 0u}));  //now it fits again
    QTX_EXPECT(!q.try_push({6u, 0u})); //full again

    //We suck everything out in the correct FIFO order.
    for (u64 expected : {2u, 3u, 4u, 5u}) {
        auto p = q.try_pop();
        QTX_EXPECT(p.has_value());
        QTX_EXPECT_EQ(p->id, expected);
    }
    QTX_EXPECT(q.empty());
}

// ============================================================================
// Two-output-param API parity
// ============================================================================

QTX_TEST(SPSC, OutParamApiMatchesOptionalApi) {
    SPSCRingBuffer<Message, 16> q;
    QTX_EXPECT(q.try_push({77u, 0xBEEFu}));

    Message m{};
    QTX_EXPECT(q.try_pop(m));
    QTX_EXPECT_EQ(m.id, 77u);
    QTX_EXPECT_EQ(m.payload, 0xBEEFu);
    QTX_EXPECT(q.empty());
}

// ============================================================================
//CONCURRENT: classic SPSC test on two streams
// ============================================================================

QTX_TEST(SPSC, ConcurrentProducerConsumerOrdering) {
    constexpr u64 kCount = 100'000u;
    SPSCRingBuffer<Message, 1024> q;

    std::atomic<bool> ok{true};

    std::thread producer([&]() {
        for (u64 i = 0; i < kCount; ++i) {
            //payload is a checksum (id * known_constant).
            Message m{i, i * 0xDEADBEEFu};
            //Spin-push: we wait for a place without going to the kernel.
            while (!q.try_push(m)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (u64 i = 0; i < kCount; ++i) {
            Message m;
            while (!q.try_pop(m)) {
                std::this_thread::yield();
            }
            // FIFO + integrity checks
            if (m.id != i) {
                ok.store(false, std::memory_order_relaxed);
                break;
            }
            if (m.payload != i * 0xDEADBEEFu) {
                ok.store(false, std::memory_order_relaxed);
                break;
            }
        }
    });

    producer.join();
    consumer.join();

    QTX_EXPECT(ok.load());
    QTX_EXPECT(q.empty());
}

// ============================================================================
//CONCURRENT with a small buffer - stress on wrap-around
// ============================================================================

QTX_TEST(SPSC, ConcurrentTinyBufferStress) {
    constexpr u64 kCount = 50'000u;
    SPSCRingBuffer<Message, 16> q;  //very small - a lot of backpressure

    std::atomic<bool> ok{true};
    std::atomic<u64> produced{0};
    std::atomic<u64> consumed{0};

    std::thread producer([&]() {
        for (u64 i = 0; i < kCount; ++i) {
            while (!q.try_push({i, i ^ 0xA5A5A5A5u})) {
                std::this_thread::yield();
            }
            produced.fetch_add(1u, std::memory_order_relaxed);
        }
    });

    std::thread consumer([&]() {
        for (u64 i = 0; i < kCount; ++i) {
            Message m;
            while (!q.try_pop(m)) {
                std::this_thread::yield();
            }
            if (m.id != i || m.payload != (i ^ 0xA5A5A5A5u)) {
                ok.store(false, std::memory_order_relaxed);
                break;
            }
            consumed.fetch_add(1u, std::memory_order_relaxed);
        }
    });

    producer.join();
    consumer.join();

    QTX_EXPECT(ok.load());
    QTX_EXPECT_EQ(produced.load(), kCount);
    QTX_EXPECT_EQ(consumed.load(), kCount);
    QTX_EXPECT(q.empty());
}

// ============================================================================
//CONCURRENT: control happens-before via side-channel
//(the consumer must see the producer's entry BEFORE the corresponding push)
// ============================================================================

QTX_TEST(SPSC, HappensBeforeAcrossPushPop) {
    //This test is critical under TSAN: if memory_orders are incorrect,
    //TSAN reports data race to shared_data.
    constexpr u64 kCount = 10'000u;
    SPSCRingBuffer<u64, 256> q;

    //Producer writes to shared_data BEFORE push, consumer reads AFTER pop.
    //If memory orderings are correct, there should be no races.
    std::vector<u64> shared_data(kCount, 0u);

    std::atomic<bool> ok{true};

    std::thread producer([&]() {
        for (u64 i = 0; i < kCount; ++i) {
            shared_data[i] = i * 7u + 3u;  //plain write, protected by release below
            while (!q.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (u64 i = 0; i < kCount; ++i) {
            u64 idx;
            while (!q.try_pop(idx)) {
                std::this_thread::yield();
            }
            //After acquire-load in try_pop we should see the entry
            //producer in shared_data[idx].
            if (shared_data[idx] != idx * 7u + 3u) {
                ok.store(false, std::memory_order_relaxed);
                break;
            }
        }
    });

    producer.join();
    consumer.join();
    QTX_EXPECT(ok.load());
}
