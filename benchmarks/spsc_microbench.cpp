// ============================================================================
// @file        spsc_microbench.cpp
// @brief       2-thread SPSC throughput.
// ============================================================================

#include "qtx/ipc/spsc_ring_buffer.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace qtx::ipc;
using qtx::core::u64;

struct Payload {
    u64 a;
    u64 b;
};

int main() {
    constexpr int kMessages = 2'000'000;
    constexpr qtx::core::usize kCap = 1024;
    SPSCRingBuffer<Payload, kCap> q;

    std::atomic<bool> start{false};

    std::thread producer([&]() {
        while (!start.load(std::memory_order_acquire)) { /* spin */ }
        for (int i = 0; i < kMessages; ++i) {
            q.wait_push(Payload{static_cast<u64>(i), static_cast<u64>(i * 2)});
        }
    });

    std::thread consumer([&]() {
        while (!start.load(std::memory_order_acquire)) { /* spin */ }
        Payload p;
        u64 checksum = 0;
        for (int i = 0; i < kMessages; ++i) {
            q.wait_pop(p);
            checksum ^= p.a + p.b;
        }
        // Sink — prevent optimisation.
        std::printf("  Checksum:        %lu (sink)\n",
                    static_cast<unsigned long>(checksum));
    });

    auto t0 = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    producer.join();
    consumer.join();
    auto t1 = std::chrono::steady_clock::now();

    const double ns_total =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_msg = ns_total / kMessages;
    const double mmsg_per_s = 1000.0 / ns_per_msg;

    std::printf("SPSC 2-thread throughput\n");
    std::printf("  Messages:        %d\n", kMessages);
    std::printf("  Payload:         %zu bytes\n", sizeof(Payload));
    std::printf("  Capacity:        %zu\n", static_cast<size_t>(kCap));
    std::printf("  Total time:      %.2f ms\n", ns_total / 1e6);
    std::printf("  Per message:     %.0f ns\n", ns_per_msg);
    std::printf("  Throughput:      %.2f M-msg/s\n", mmsg_per_s);
    return 0;
}
