// ============================================================================
// @file        test_integration.cpp
//@brief Integration test: SPSC + FractalArena - GenID transfer
//between producer and consumer threads via a lock-free queue.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================
//
//SCENARIO:
//Producer stream (simulates an AI encoder agent):
//1. Allocates a slot in FractalArena.
//2. Fills it with data (simulating a KV cache entry).
//3. Sends GenID via SPSCRingBuffer to the consumer.
//
//Consumer flow (simulates an AI testing agent):
//1. Obtains GenID from SPSC.
//2. Through arena.view(genid) reads the data.
//3. Validates the checksum.
//4. Frees up the slot.
//
//This is the basic pattern of “a swarm of agents communicates via GenID without copying”
//data" is the main use case of the entire system.
//
// ============================================================================

#include "test_harness.hpp"
#include "qtx/arena/fractal_arena.hpp"
#include "qtx/arena/gen_id.hpp"
#include "qtx/ipc/spsc_ring_buffer.hpp"

#include <atomic>
#include <cstring>
#include <memory>
#include <thread>

using namespace qtx;
using qtx::core::u64;
using qtx::core::usize;

//Payload: the first 8 bytes of each slot are the checksum,
//the rest is pseudo-random data to verify integrity.
static constexpr u64 kChecksumSeed = 0x9E3779B97F4A7C15ull;  // FNV-style

static void fillSlot(std::span<std::byte> view, u64 message_id) noexcept {
    //We write message_id in the first 8 bytes as a marker.
    const u64 marker = message_id;
    std::memcpy(view.data(), &marker, sizeof(marker));

    //We fill in the rest with “pseudodata” with a known pattern.
    for (usize i = sizeof(marker); i < view.size(); ++i) {
        const u64 mixed = (message_id ^ kChecksumSeed) + i;
        view[i] = static_cast<std::byte>(mixed & 0xFFu);
    }
}

[[nodiscard]] static bool verifySlot(std::span<const std::byte> view,
                                     u64 expected_message_id) noexcept {
    if (view.size() < sizeof(u64)) return false;

    u64 read_marker = 0;
    std::memcpy(&read_marker, view.data(), sizeof(read_marker));
    if (read_marker != expected_message_id) return false;

    for (usize i = sizeof(u64); i < view.size(); ++i) {
        const u64 expected = (expected_message_id ^ kChecksumSeed) + i;
        if (static_cast<unsigned>(view[i]) !=
            static_cast<unsigned>(expected & 0xFFu)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
//Main integration scenario
// ============================================================================

QTX_TEST(Integration, ProducerConsumerThroughGenID) {
    //256 slots × 128 bytes = 32 KiB - comfortably lies in L1/L2.
    using Arena = arena::FractalArena<256, 128>;
    //64-slot GenID queue between threads.
    using Queue = ipc::SPSCRingBuffer<arena::GenID, 64>;

    auto arena_ptr = std::make_unique<Arena>();
    auto queue_ptr = std::make_unique<Queue>();

    constexpr u64 kMessages = 50'000u;

    std::atomic<u64> verify_failures{0};
    std::atomic<u64> alloc_failures{0};

    // === Producer ===
    std::thread producer([&]() {
        for (u64 i = 0; i < kMessages; ++i) {
            //Spin until successful allocation (consumer releases slots).
            arena::GenID id;
            for (;;) {
                id = arena_ptr->allocate(64u, 8u);
                if (!id.isNull()) break;
                std::this_thread::yield();
            }

            //We fill the slot with data with marker i.
            auto view = arena_ptr->view(id);
            if (view.empty()) {
                alloc_failures.fetch_add(1u, std::memory_order_relaxed);
                continue;
            }
            fillSlot(view, i);

            //Spin-push in SPSC.
            while (!queue_ptr->try_push(id)) {
                std::this_thread::yield();
            }
        }
    });

    // === Consumer ===
    std::thread consumer([&]() {
        for (u64 i = 0; i < kMessages; ++i) {
            arena::GenID id;
            while (!queue_ptr->try_pop(id)) {
                std::this_thread::yield();
            }

            //Should see the producer's entry (release/acquire via
            //SPSC creates happens-before).
            auto view = arena_ptr->view(id);
            if (view.empty()) {
                verify_failures.fetch_add(1u, std::memory_order_relaxed);
                continue;
            }
            if (!verifySlot(view, i)) {
                verify_failures.fetch_add(1u, std::memory_order_relaxed);
            }
            //We free the slot - the producer will be able to reuse it.
            (void)arena_ptr->release(id);
        }
    });

    producer.join();
    consumer.join();

    QTX_EXPECT_EQ(alloc_failures.load(), 0u);
    QTX_EXPECT_EQ(verify_failures.load(), 0u);
    //All slots must be freed.
    QTX_EXPECT_EQ(arena_ptr->usedSlots(), 0u);
    QTX_EXPECT(queue_ptr->empty());
}

// ============================================================================
//Backpressure scenario: small queue + small arena,
//We check that the system operates correctly under conditions of complete saturation.
// ============================================================================

QTX_TEST(Integration, BackpressureWithSmallResources) {
    using Arena = arena::FractalArena<64, 128>;     //only 64 slots
    using Queue = ipc::SPSCRingBuffer<arena::GenID, 8>;  //and a tight line

    auto arena_ptr = std::make_unique<Arena>();
    auto queue_ptr = std::make_unique<Queue>();

    constexpr u64 kMessages = 20'000u;

    std::atomic<u64> verify_failures{0};

    std::thread producer([&]() {
        for (u64 i = 0; i < kMessages; ++i) {
            arena::GenID id;
            for (;;) {
                id = arena_ptr->allocate(64u, 8u);
                if (!id.isNull()) break;
                std::this_thread::yield();
            }
            auto view = arena_ptr->view(id);
            fillSlot(view, i);
            while (!queue_ptr->try_push(id)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (u64 i = 0; i < kMessages; ++i) {
            arena::GenID id;
            while (!queue_ptr->try_pop(id)) {
                std::this_thread::yield();
            }
            auto view = arena_ptr->view(id);
            if (!verifySlot(view, i)) {
                verify_failures.fetch_add(1u, std::memory_order_relaxed);
            }
            (void)arena_ptr->release(id);
        }
    });

    producer.join();
    consumer.join();

    QTX_EXPECT_EQ(verify_failures.load(), 0u);
    QTX_EXPECT_EQ(arena_ptr->usedSlots(), 0u);
}

// ============================================================================
//ABA critical scene: same physical slot is reused
//many times. We check that generational indices correctly distinguish
//"generations" of one cell.
// ============================================================================

QTX_TEST(Integration, ABAUnderHeavyReuse) {
    using Arena = arena::FractalArena<64, 128>;
    using Queue = ipc::SPSCRingBuffer<arena::GenID, 16>;

    auto arena_ptr = std::make_unique<Arena>();
    auto queue_ptr = std::make_unique<Queue>();

    //10,000 messages across 64 slots → each slot is reused ~150 times.
    //This will traverse the typical "wrap" of a 16-bit version field several times
    //on a small slot pool.
    constexpr u64 kMessages = 10'000u;

    std::atomic<u64> id_mismatch{0};

    std::thread producer([&]() {
        for (u64 i = 0; i < kMessages; ++i) {
            arena::GenID id;
            for (;;) {
                id = arena_ptr->allocate(64u, 8u);
                if (!id.isNull()) break;
                std::this_thread::yield();
            }
            //Producer fills the slot.
            auto view = arena_ptr->view(id);
            fillSlot(view, i);
            while (!queue_ptr->try_push(id)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (u64 i = 0; i < kMessages; ++i) {
            arena::GenID id;
            while (!queue_ptr->try_pop(id)) {
                std::this_thread::yield();
            }
            //validate MUST pass - this GenID was just sent by the producer.
            if (!arena_ptr->validate(id)) {
                id_mismatch.fetch_add(1u, std::memory_order_relaxed);
                continue;
            }
            auto view = arena_ptr->view(id);
            //There could be an ABA problem here if GenID
            //did not distinguish between slot generations.
            u64 read_marker = 0;
            std::memcpy(&read_marker, view.data(), sizeof(read_marker));
            if (read_marker != i) {
                id_mismatch.fetch_add(1u, std::memory_order_relaxed);
            }
            (void)arena_ptr->release(id);
        }
    });

    producer.join();
    consumer.join();

    QTX_EXPECT_EQ(id_mismatch.load(), 0u);
    QTX_EXPECT_EQ(arena_ptr->usedSlots(), 0u);
}
