// ============================================================================
// @file        perf_microbench.cpp
// @brief       Throughput / latency microbenchmarks for HFT-critical paths.
// ============================================================================

#include "qtx/arena/fractal_arena.hpp"
#include "qtx/selector/axiom_selector.hpp"
#include "qtx/ipc/spsc_ring_buffer.hpp"
#include "qtx/quantize/quantizer.hpp"
#include "qtx/tiered/tiered_bridge.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
using ns  = std::chrono::nanoseconds;

template <typename F>
double bench_ns(F&& f, std::size_t iters) {
    // warmup
    for (std::size_t i = 0; i < iters / 10 + 1; ++i) {
        f();
        asm volatile("" ::: "memory");
    }
    auto t0 = clk::now();
    for (std::size_t i = 0; i < iters; ++i) {
        f();
        asm volatile("" ::: "memory");
    }
    auto t1 = clk::now();
    return std::chrono::duration_cast<ns>(t1 - t0).count() / double(iters);
}

template <typename F>
double bench_throughput_mops(F&& f, std::size_t iters) {
    return 1000.0 / bench_ns(std::forward<F>(f), iters);
}

void hr() { std::puts("----------------------------------------------"); }

// ----------------------------------------------------------------------------
// 1) AxiomSelector — acquire/release round-trip
// ----------------------------------------------------------------------------
void bench_selector() {
    std::puts("\n[Selector] acquire+release round-trip");
    hr();
    using namespace qtx::selector;
    AxiomSelector<256> sel;
    auto fn = [&]() {
        auto s = sel.acquire();
        if (s != AxiomSelector<256>::kNoSlot) sel.release(s);
    };
    const auto ns_per = bench_ns(fn, 5'000'000);
    std::printf("  single-threaded: %.1f ns/op  (%.2f Mop/s)\n",
                ns_per, 1000.0 / ns_per);
}

// ----------------------------------------------------------------------------
// 2) FractalArena — allocate+release round-trip
// ----------------------------------------------------------------------------
void bench_arena() {
    std::puts("\n[Arena] allocate(128B) + release round-trip");
    hr();
    using Arena = qtx::arena::FractalArena<256, 128>;
    auto arena = std::make_unique<Arena>();
    auto fn = [&]() {
        auto id = arena->allocate(64, 64);
        if (id.isActive()) (void)arena->release(id);
    };
    const auto ns_per = bench_ns(fn, 2'000'000);
    std::printf("  single-threaded: %.1f ns/op  (%.2f Mop/s)\n",
                ns_per, 1000.0 / ns_per);
}

// ----------------------------------------------------------------------------
// 3) SPSC ring buffer — single-producer single-consumer
// ----------------------------------------------------------------------------
struct Payload {
    std::uint64_t a, b;
};

void bench_spsc() {
    std::puts("\n[SPSC] push+pop round-trip (same thread)");
    hr();
    using Q = qtx::ipc::SPSCRingBuffer<Payload, 1024>;
    auto q = std::make_unique<Q>();
    Payload p{42, 7};
    auto fn = [&]() {
        (void)q->try_push(p);
        Payload out{};
        (void)q->try_pop(out);
    };
    const auto ns_per = bench_ns(fn, 5'000'000);
    std::printf("  same-thread push+pop: %.1f ns/op  (%.2f Mop/s)\n",
                ns_per, 1000.0 / ns_per);

    std::puts("\n[SPSC] real 2-thread throughput, capacity=4096");
    hr();
    using Q2 = qtx::ipc::SPSCRingBuffer<Payload, 4096>;
    auto q2 = std::make_unique<Q2>();
    std::atomic<bool> stop{false};
    const std::size_t target = 2'000'000;
    std::atomic<std::size_t> received{0};

    auto producer = std::thread([&]() {
        Payload pp{0, 0};
        for (std::size_t i = 0; i < target; ++i) {
            pp.a = i;
            while (!q2->try_push(pp)) { /* spin */ }
        }
    });
    auto consumer = std::thread([&]() {
        Payload out;
        std::size_t got = 0;
        while (got < target) {
            if (q2->try_pop(out)) ++got;
        }
        received.store(got);
    });

    auto t0 = clk::now();
    producer.join();
    consumer.join();
    auto t1 = clk::now();
    (void)stop.store(true);

    const double total_ns =
        std::chrono::duration_cast<ns>(t1 - t0).count();
    const double mops = double(target) * 1000.0 / total_ns;
    const double ns_per_msg = total_ns / double(target);
    std::printf("  %zu messages in %.2f ms\n",
                target, total_ns / 1e6);
    std::printf("  throughput: %.2f Mmsg/s\n", mops);
    std::printf("  latency:    %.1f ns/msg\n", ns_per_msg);
}

// Noinline wrappers around quantize entry points to defeat CSE across
// benchmark iterations on identical inputs (the compiler is free to
// observe that the result is deterministic and only call once).
__attribute__((noinline))
std::size_t bench_decompress_int8(std::span<const std::byte> src,
                                  std::span<std::byte> dst) {
    auto r = qtx::quantize::decompress(
        qtx::arena::QuantFormat::kINT8, src, dst);
    asm volatile("" : : "r"(dst.data()) : "memory");
    return r;
}

__attribute__((noinline))
std::size_t bench_compress_int8(std::span<const std::byte> src,
                                std::span<std::byte> dst) {
    auto r = qtx::quantize::compress(
        qtx::arena::QuantFormat::kINT8, src, dst);
    asm volatile("" : : "r"(dst.data()) : "memory");
    return r;
}

__attribute__((noinline))
std::size_t bench_compress_bf16(std::span<const std::byte> src,
                                std::span<std::byte> dst) {
    auto r = qtx::quantize::compress(
        qtx::arena::QuantFormat::kBF16, src, dst);
    asm volatile("" : : "r"(dst.data()) : "memory");
    return r;
}

// ----------------------------------------------------------------------------
// 4) Quantizer — INT8 compress / decompress
// ----------------------------------------------------------------------------
void bench_quantize() {
    using namespace qtx::quantize;
    using namespace qtx::arena;
    std::puts("\n[Quantize] FP32 -> INT8 (1024 elements = 32 blocks)");
    hr();
    constexpr std::size_t N = 1024;
    alignas(64) std::array<float, N> src{};
    alignas(64) std::array<std::byte, N * 4> src_bytes{};
    alignas(64) std::array<std::byte, N * 2> dst{};
    alignas(64) std::array<std::byte, N * 4> back{};

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : src) v = dist(rng);
    std::memcpy(src_bytes.data(), src.data(), N * 4);

    // Compress first and capture the actual size, since decompress validates
    // that src.size() is an exact multiple of kINT8BlockBytes (36).
    const std::size_t compressed_size =
        bench_compress_int8(std::span<const std::byte>(src_bytes),
                            std::span<std::byte>(dst));

    volatile std::uint64_t global_sink = 0;
    auto compress_fn = [&]() {
        global_sink ^= bench_compress_int8(
            std::span<const std::byte>(src_bytes),
            std::span<std::byte>(dst));
    };
    auto decompress_fn = [&]() {
        const auto w = bench_decompress_int8(
            std::span<const std::byte>(dst.data(), compressed_size),
            std::span<std::byte>(back));
        global_sink ^= w + std::uint64_t(static_cast<unsigned char>(back[0]));
    };

    const auto comp_ns = bench_ns(compress_fn, 200'000);
    const auto deco_ns = bench_ns(decompress_fn, 200'000);
    const double bytes_per_op = N * 4;  // 4096 bytes FP32 input
    const double comp_gbs = bytes_per_op / comp_ns;  // ns→GB/s scaling
    const double deco_gbs = bytes_per_op / deco_ns;
    std::printf("  compress:   %.0f ns/op  (%.2f GB/s in)\n", comp_ns, comp_gbs);
    std::printf("  decompress: %.0f ns/op  (%.2f GB/s out)\n", deco_ns, deco_gbs);

    std::puts("\n[Quantize] FP32 -> BF16 (1024 elements)");
    hr();
    alignas(64) std::array<std::byte, N * 2> bf_dst{};
    auto bf_fn = [&]() {
        global_sink ^= bench_compress_bf16(
            std::span<const std::byte>(src_bytes),
            std::span<std::byte>(bf_dst));
    };
    const auto bf_ns = bench_ns(bf_fn, 200'000);
    std::printf("  compress:   %.0f ns/op  (%.2f GB/s)\n",
                bf_ns, bytes_per_op / bf_ns);
}

// ----------------------------------------------------------------------------
// 5) AxiomSelector — multithreaded contention
// ----------------------------------------------------------------------------
void bench_selector_mt() {
    std::puts("\n[Selector] multithreaded contention (4 threads x 250K ops)");
    hr();
    using namespace qtx::selector;
    AxiomSelector<256> sel;
    constexpr int kThreads = 4;
    constexpr std::size_t kPerThread = 250'000;

    std::atomic<bool> go{false};
    std::vector<std::thread> ts;
    auto t0 = clk::now();
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&]() {
            while (!go.load(std::memory_order_acquire)) { /* spin */ }
            for (std::size_t i = 0; i < kPerThread; ++i) {
                auto s = sel.acquire();
                if (s != AxiomSelector<256>::kNoSlot) sel.release(s);
            }
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& t : ts) t.join();
    auto t1 = clk::now();
    const double total_ns =
        std::chrono::duration_cast<ns>(t1 - t0).count();
    const double total_ops = double(kThreads) * double(kPerThread);
    std::printf("  %.0f ops in %.2f ms\n", total_ops, total_ns / 1e6);
    std::printf("  aggregate: %.2f Mop/s, per-op: %.0f ns avg\n",
                total_ops * 1000.0 / total_ns,
                total_ns / total_ops);
}

// ----------------------------------------------------------------------------
// 6) Tiered bridge — create / acquire / release loop
// ----------------------------------------------------------------------------
void bench_tiered() {
    std::puts("\n[Tiered] createTenant + acquireFP32 + release + destroyTenant");
    hr();
    using Hot = qtx::arena::FractalArena<64, 4096>;
    using Cold = qtx::arena::FractalArena<256, 2048>;
    using Bridge = qtx::tiered::TieredArenaBridge<Hot, Cold>;
    auto hot = std::make_unique<Hot>();
    auto cold = std::make_unique<Cold>();
    auto bridge = std::make_unique<Bridge>(hot.get(), cold.get());

    auto fn = [&]() {
        auto h = bridge->createTenant();
        if (!h.is_valid) return;
        (void)bridge->acquireFP32(h);
        (void)bridge->release(h);
        (void)bridge->destroyTenant(h);
    };
    const auto ns_per = bench_ns(fn, 200'000);
    std::printf("  full lifecycle: %.0f ns/op  (%.0f K-op/s)\n",
                ns_per, 1'000'000.0 / ns_per);
}

int main() {
    std::puts("╔════════════════════════════════════════════════════════════╗");
    std::puts("║              QTX  — PERFORMANCE MICROBENCH              ║");
    std::puts("║  Compiler: g++ 13.3   Build: -O2 -DNDEBUG (no sanitisers)  ║");
    std::puts("╚════════════════════════════════════════════════════════════╝");

    bench_selector();
    bench_arena();
    bench_spsc();
    bench_quantize();
    bench_selector_mt();
    bench_tiered();

    std::puts("\n=== done ===\n");
    return 0;
}
