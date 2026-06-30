// ============================================================================
// @file        micro_tenant_demo.cpp
// @brief       Phase 2 Block C.2 demo: 32 micro-tenants in ONE 4 KiB slot.
//
// This benchmark complements oom_survival_test.cpp by demonstrating the
// OTHER axis of compression that Block C.2 introduces: packing many
// small ("micro") tenants — typical of dormant system-prompts, few-shot
// headers, or per-agent metadata — into a single shared hot slot.
//
// Headline number: 128 micro-tenants live in 4 hot slots (4096 / 128 ×
// 4 = 16 KiB) versus 128 hot slots in the naive approach (128 × 4096 =
// 512 KiB). 32× packing factor.
// ============================================================================

#include "qtx/arena/fractal_arena.hpp"
#include "qtx/tiered/tiered_bridge.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

namespace arena  = qtx::arena;
namespace tiered = qtx::tiered;

using HotArena  = arena::FractalArena<64,  4096>;
using ColdArena = arena::FractalArena<256, 1152>;
using Bridge    = tiered::TieredArenaBridge<HotArena, ColdArena>;

static constexpr std::size_t kTargetMicroTenants = 128u;

int main() {
    auto hot  = std::make_unique<HotArena>();
    auto cold = std::make_unique<ColdArena>();
    Bridge bridge(hot.get(), cold.get());

    std::printf(
        "╔════════════════════════════════════════════════════════════╗\n"
        "║  QTX — Micro-Tenant Packing Demo (Block C.2)            ║\n"
        "╠════════════════════════════════════════════════════════════╣\n"
        "║ HotArena slot size:    %4zu B                               ║\n"
        "║ Micro-tenant size:     %4zu B  (× %u per slot)              ║\n"
        "║ Target micro-tenants:  %4zu                                 ║\n"
        "╚════════════════════════════════════════════════════════════╝\n\n",
        HotArena::kSlotSize, Bridge::kMicroSlotSize,
        Bridge::kSubSlotsPerSlot, kTargetMicroTenants);

    // ---- Create N micro-tenants and write a unique pattern into each.
    auto t0 = std::chrono::steady_clock::now();
    std::vector<tiered::TenantHandle> handles;
    handles.reserve(kTargetMicroTenants);

    for (std::size_t i = 0; i < kTargetMicroTenants; ++i) {
        auto h = bridge.createMicroTenant();
        if (!h.is_valid) {
            std::printf("  ✗ createMicroTenant failed at i=%zu\n", i);
            return 1;
        }
        auto v = bridge.acquireMicroFP32(h);
        if (v.size() != Bridge::kMicroSlotSize) {
            std::printf("  ✗ unexpected view size at i=%zu\n", i);
            return 1;
        }
        // Tag the sub-slot with its index so we can verify later.
        std::memset(v.data(), static_cast<int>(i & 0xFFu), v.size());
        bridge.release(h);
        handles.push_back(h);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double create_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto stats = bridge.stats();
    const std::size_t hot_slots_used   = bridge.hotUsed();
    const std::size_t physical_bytes   = hot_slots_used * HotArena::kSlotSize;
    const std::size_t naive_bytes      = kTargetMicroTenants * HotArena::kSlotSize;
    const double      packing_factor   =
        static_cast<double>(naive_bytes) / static_cast<double>(physical_bytes);

    std::printf("──── Phase 1: Create %zu micro-tenants ────\n",
                kTargetMicroTenants);
    std::printf("  Alive:               %zu\n", stats.alive_tenants);
    std::printf("  Hot slots used:      %zu\n", hot_slots_used);
    std::printf("  Physical bytes:      %zu B  (%zu KiB)\n",
                physical_bytes, physical_bytes / 1024u);
    std::printf("  Naive equivalent:    %zu B  (%zu KiB)\n",
                naive_bytes, naive_bytes / 1024u);
    std::printf("  Packing factor:      \033[1;32m%.2fx\033[0m\n",
                packing_factor);
    std::printf("  Wall time:           %.2f ms\n\n", create_ms);

    // ---- Verify: every sub-slot still holds its tag.
    auto t2 = std::chrono::steady_clock::now();
    std::size_t leaks = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        auto v = bridge.acquireMicroFP32(handles[i]);
        if (v.size() != Bridge::kMicroSlotSize) { ++leaks; continue; }
        const auto expected = static_cast<std::byte>(i & 0xFFu);
        for (std::size_t j = 0; j < v.size(); ++j) {
            if (v[j] != expected) { ++leaks; break; }
        }
        bridge.release(handles[i]);
    }
    auto t3 = std::chrono::steady_clock::now();
    const double verify_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("──── Phase 2: Verify data integrity ────\n");
    std::printf("  Sub-slot leaks:      %zu / %zu\n", leaks, handles.size());
    std::printf("  Wall time:           %.2f ms\n\n", verify_ms);

    // ---- Cleanup.
    auto t4 = std::chrono::steady_clock::now();
    for (auto h : handles) bridge.destroyTenant(h);
    auto t5 = std::chrono::steady_clock::now();
    const double destroy_ms =
        std::chrono::duration<double, std::milli>(t5 - t4).count();

    std::printf("──── Phase 3: Destroy all ────\n");
    std::printf("  Hot slots after:     %zu  (expected 0)\n", bridge.hotUsed());
    std::printf("  Wall time:           %.2f ms\n\n", destroy_ms);

    if (leaks == 0 && bridge.hotUsed() == 0u) {
        std::printf(
            "╔════════════════════════════════════════════════════════════╗\n"
            "║  DEMO COMPLETE — %zu micro-tenants packed at %.2fx density    ║\n"
            "╚════════════════════════════════════════════════════════════╝\n",
            kTargetMicroTenants, packing_factor);
        return 0;
    }
    std::printf("  ✗ leaks=%zu, residual hot slots=%zu\n",
                leaks, bridge.hotUsed());
    return 1;
}
