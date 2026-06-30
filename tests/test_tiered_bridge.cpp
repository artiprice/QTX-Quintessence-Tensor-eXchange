// ============================================================================
// @file        test_tiered_bridge.cpp
//@brief TieredArenaBridge tests: auto-eviction, promotion, pinning.
// @author      QTX Project
// @date        2026-05-13
// ============================================================================

#include "test_harness.hpp"
#include "qtx/tiered/tiered_bridge.hpp"
#include "qtx/arena/fractal_arena.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using namespace qtx;
using qtx::core::usize;
using qtx::core::u64;

//Test arenas (AxiomSelector requires SlotCount to be a multiple of 64):
//- 64 hot slots × 4 KiB = 256 KiB FP32
//- 512 cold slots × 1152 bytes = 576 KiB INT8 (exact-fit per EC106).
//  Previously we used 2048 byte slots (1 MiB total) because of the
//  power-of-two constraint; now we are exact-fit. The change exercises
//  the non-power-of-two path of FractalArena.
using HotA  = arena::FractalArena<64, 4096>;
using ColdA = arena::FractalArena<512, 1152>;
using Tiered = tiered::TieredArenaBridge<HotA, ColdA>;

// ============================================================================
//Basic invariants
// ============================================================================

QTX_TEST(Tiered, ConstructEmpty) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());
    QTX_EXPECT_EQ(tb.hotUsed(), 0u);
    QTX_EXPECT_EQ(tb.coldUsed(), 0u);
    QTX_EXPECT_EQ(tb.totalAgents(), 0u);
}

QTX_TEST(Tiered, ConstantSizes) {
    QTX_EXPECT_EQ(Tiered::kPayloadBytes, 4096u);
    QTX_EXPECT_EQ(Tiered::kPayloadElements, 1024u);
    //32 blocks × 36 bytes = 1152 bytes of compressed payload.
    QTX_EXPECT_EQ(Tiered::kCompressedBytes, 32u * 36u);
}

// ============================================================================
// Lifecycle
// ============================================================================

QTX_TEST(Tiered, CreateSingleTenant) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    auto h = tb.createTenant();
    QTX_EXPECT(h.is_valid);
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);
    QTX_EXPECT_EQ(tb.totalAgents(), 1u);
}

QTX_TEST(Tiered, DestroyReleasesHotSlot) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    auto h = tb.createTenant();
    QTX_EXPECT(tb.destroyTenant(h));
    QTX_EXPECT_EQ(tb.hotUsed(), 0u);
}

QTX_TEST(Tiered, FillHotToCapacity) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    for (int i = 0; i < 64; ++i) {
        auto h = tb.createTenant();
        QTX_EXPECT(h.is_valid);
    }
    QTX_EXPECT_EQ(tb.hotUsed(), 64u);
    QTX_EXPECT_EQ(tb.coldUsed(), 0u);
}

// ============================================================================
// Auto-eviction
// ============================================================================

QTX_TEST(Tiered, AutoEvictOnHotOOM) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    //Fill in hot to the end.
    for (int i = 0; i < 64; ++i) {
        QTX_EXPECT(tb.createTenant().is_valid);
    }
    QTX_EXPECT_EQ(tb.hotUsed(), 64u);
    QTX_EXPECT_EQ(tb.coldUsed(), 0u);

    //65th tenant - calls auto-eviction.
    auto h65 = tb.createTenant();
    QTX_EXPECT(h65.is_valid);
    QTX_EXPECT_EQ(tb.hotUsed(), 64u);
    QTX_EXPECT_EQ(tb.coldUsed(), 1u);
    QTX_EXPECT_EQ(tb.totalAgents(), 65u);
    QTX_EXPECT_EQ(tb.stats().evictions, 1u);
}

QTX_TEST(Tiered, MultipleAutoEvictions) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    //300 tenants - there will be 64 hot + 236 cold.
    for (int i = 0; i < 300; ++i) {
        QTX_EXPECT(tb.createTenant().is_valid);
    }
    auto s = tb.stats();
    QTX_EXPECT_EQ(s.hot_used, 64u);
    QTX_EXPECT_EQ(s.cold_used, 236u);
    QTX_EXPECT_EQ(s.alive_tenants, 300u);
    QTX_EXPECT_EQ(s.evictions, 236u);
}

// ============================================================================
//Data integrity via quant cycle
// ============================================================================

QTX_TEST(Tiered, EvictThenPromoteRestoresData) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    auto h0 = tb.createTenant();
    auto view = tb.acquireFP32(h0);
    QTX_EXPECT(!view.empty());

    constexpr usize n_elements = Tiered::kPayloadElements;
    for (usize i = 0; i < n_elements; ++i) {
        const float v = std::sin(static_cast<float>(i) * 0.01f);
        std::memcpy(view.data() + i * sizeof(float), &v, sizeof(float));
    }
    QTX_EXPECT(tb.release(h0));

    //We fill 65 more tenants (calls eviction).
    for (int i = 0; i < 65; ++i) {
        QTX_EXPECT(tb.createTenant().is_valid);
    }
    QTX_EXPECT(tb.stats().evictions >= 1u);

    //Promote h0 back.
    auto view_back = tb.acquireFP32(h0);
    QTX_EXPECT(!view_back.empty());

    int significant_errors = 0;
    for (usize i = 0; i < n_elements; ++i) {
        float restored;
        std::memcpy(&restored, view_back.data() + i * sizeof(float), sizeof(float));
        const float original = std::sin(static_cast<float>(i) * 0.01f);
        if (std::fabs(original) > 0.05f) {
            const float rel_err = std::fabs(restored - original) /
                                  std::fabs(original);
            if (rel_err > 0.05f) ++significant_errors;
        }
    }
    QTX_EXPECT_EQ(significant_errors, 0);
    QTX_EXPECT(tb.release(h0));
}

// ============================================================================
// Pin protection
// ============================================================================

QTX_TEST(Tiered, PinnedTenantNotEvicted) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    auto h_pinned = tb.createTenant();
    auto view = tb.acquireFP32(h_pinned);
    QTX_EXPECT(!view.empty());

    //We fill the remaining 63 hot slots.
    for (int i = 0; i < 63; ++i) {
        QTX_EXPECT(tb.createTenant().is_valid);
    }
    QTX_EXPECT_EQ(tb.hotUsed(), 64u);

    //Creating another one should evict one of 63, NOT pinned.
    auto h_extra = tb.createTenant();
    QTX_EXPECT(h_extra.is_valid);
    QTX_EXPECT_EQ(tb.stats().evictions, 1u);

    //pinned tenant remains available (re-acquiring gives view).
    auto view2 = tb.acquireFP32(h_pinned);
    QTX_EXPECT(!view2.empty());

    QTX_EXPECT(tb.release(h_pinned));
    QTX_EXPECT(tb.release(h_pinned));
}

// ============================================================================
// OOM
// ============================================================================

QTX_TEST(Tiered, FullOOMReturnsInvalid) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    //Maximum: hot=64 + cold=512 = 576 tenants.
    int valid_count = 0;
    int invalid_count = 0;
    for (int i = 0; i < 700; ++i) {
        auto h = tb.createTenant();
        if (h.is_valid) ++valid_count; else ++invalid_count;
    }
    QTX_EXPECT_EQ(valid_count, 576);
    QTX_EXPECT_EQ(invalid_count, 124);
}

// ============================================================================
// Swarm simulation
// ============================================================================

QTX_TEST(Tiered, SwarmSimulation) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    std::vector<tiered::TenantHandle> agents;
    for (int i = 0; i < 200; ++i) {
        auto h = tb.createTenant();
        QTX_EXPECT(h.is_valid);
        auto v = tb.acquireFP32(h);
        if (!v.empty()) {
            for (usize j = 0; j < v.size() / sizeof(float); ++j) {
                const float val = static_cast<float>(i + 1) * 0.01f;
                std::memcpy(v.data() + j * sizeof(float), &val, sizeof(float));
            }
            tb.release(h);
        }
        agents.push_back(h);
    }

    auto s = tb.stats();
    QTX_EXPECT_EQ(s.alive_tenants, 200u);
    QTX_EXPECT_EQ(s.hot_used, 64u);

    int verify_errors = 0;
    for (int i = 0; i < 200; ++i) {
        auto v = tb.acquireFP32(agents[i]);
        if (v.empty()) { ++verify_errors; continue; }
        float first;
        std::memcpy(&first, v.data(), sizeof(float));
        const float expected = static_cast<float>(i + 1) * 0.01f;
        if (std::fabs(first - expected) > 0.01f) ++verify_errors;
        tb.release(agents[i]);
    }
    QTX_EXPECT_EQ(verify_errors, 0);
}

// ============================================================================
// EC101 — tenant_id widened to u64. Three observable properties:
//
//   1. Compile-time: TenantHandle::tenant_id is exactly 8 bytes.
//   2. Sentinel separation: kNoSlot (u32, pool index) and kNoTenantId
//      (u64, public id) are distinct types with distinct widths.
//   3. The bridge never returns the u32-wrap value UINT32_MAX as a valid
//      tenant_id — a property the previous u32 implementation could not
//      offer because the entire id range was reachable by definition.
// ============================================================================

static_assert(sizeof(tiered::TenantHandle::tenant_id) == 8u,
              "EC101: tenant_id MUST be 8 bytes (u64) to defeat 11-min ABA wrap");
static_assert(std::is_same_v<decltype(Tiered::kNoSlot), const qtx::core::u32>,
              "EC101: kNoSlot must be u32 (pool-table index)");
static_assert(std::is_same_v<decltype(Tiered::kNoTenantId), const qtx::core::u64>,
              "EC101: kNoTenantId must be u64 (public id sentinel)");

QTX_TEST(Tiered, EC101_TenantIdsAreMonotonicAndWide) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    // Create a small batch and verify ids strictly increase. We do not
    // assert they are sequential — the skip-zero / skip-kNoTenantId path
    // is allowed to consume one slot of the counter — only that they
    // grow strictly monotonically.
    constexpr int kBatch = 32;
    std::vector<u64> ids;
    ids.reserve(kBatch);
    for (int i = 0; i < kBatch; ++i) {
        auto h = tb.createTenant();
        QTX_EXPECT(h.is_valid);
        QTX_EXPECT(h.tenant_id != 0u);                    // not the null id
        QTX_EXPECT(h.tenant_id != Tiered::kNoTenantId);   // not the sentinel
        ids.push_back(h.tenant_id);
    }
    for (int i = 1; i < kBatch; ++i) {
        QTX_EXPECT(ids[static_cast<usize>(i)] > ids[static_cast<usize>(i - 1)]);
    }
}

QTX_TEST(Tiered, EC101_BogusHandleRejected) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    // A handle whose tenant_id is the explicit sentinel must be refused
    // by every entry point. Previously kNoTenant was a u32; under u64
    // the sentinel UINT32_MAX is a perfectly legitimate live id, so a
    // forged handle carrying it must not collide with anything live.
    tiered::TenantHandle bogus_sentinel{Tiered::kNoTenantId, true};
    QTX_EXPECT(tb.acquireFP32(bogus_sentinel).empty());
    QTX_EXPECT(!tb.destroyTenant(bogus_sentinel));
    QTX_EXPECT(!tb.release(bogus_sentinel));

    tiered::TenantHandle bogus_zero{0u, true};
    QTX_EXPECT(tb.acquireFP32(bogus_zero).empty());
    QTX_EXPECT(!tb.destroyTenant(bogus_zero));
    QTX_EXPECT(!tb.release(bogus_zero));

    // The u32 wrap value, which under the OLD type would have been
    // kNoTenant, is now a legitimate (if very high) tenant_id slot, and
    // must NOT alias the sentinel.
    tiered::TenantHandle bogus_u32max{
        static_cast<u64>(static_cast<qtx::core::u32>(-1)), true};
    QTX_EXPECT(bogus_u32max.tenant_id != Tiered::kNoTenantId);
    QTX_EXPECT(tb.acquireFP32(bogus_u32max).empty());  // not live yet
}

// ============================================================================
// EC115 / EC116 / EC117 / EC118 — micro-tenant lifecycle (Phase 2 Block C.2)
// ============================================================================

QTX_TEST(Tiered, EC115_MicroTenant_PackingFactor) {
    // 4096-byte hot slot / 128-byte micro = 32 sub-slots per slot.
    // The constant is exposed for callers to inspect.
    QTX_EXPECT_EQ(Tiered::kMicroSlotSize, 128u);
    QTX_EXPECT_EQ(Tiered::kSubSlotsPerSlot, 32u);
}

QTX_TEST(Tiered, EC115_MicroTenant_CreateAndDestroySingle) {
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    auto h = tb.createMicroTenant();
    QTX_EXPECT(h.is_valid);
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);  // one slot consumed
    QTX_EXPECT_EQ(tb.totalAgents(), 1u);

    auto v = tb.acquireMicroFP32(h);
    QTX_EXPECT_EQ(v.size(), Tiered::kMicroSlotSize);  // 128 bytes

    QTX_EXPECT(tb.release(h));
    QTX_EXPECT(tb.destroyTenant(h));

    // After destroying the last micro-tenant, the shared slot is freed.
    QTX_EXPECT_EQ(tb.hotUsed(), 0u);
    QTX_EXPECT_EQ(tb.totalAgents(), 0u);
}

QTX_TEST(Tiered, EC115_MicroTenant_PackInto32) {
    // The headline scenario: 32 micro-tenants in ONE 4 KiB hot slot.
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    std::vector<tiered::TenantHandle> handles;
    handles.reserve(32u);
    for (int i = 0; i < 32; ++i) {
        auto h = tb.createMicroTenant();
        QTX_EXPECT(h.is_valid);
        handles.push_back(h);
    }
    // All 32 fit into a SINGLE hot slot — packing factor is 32×.
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);
    QTX_EXPECT_EQ(tb.totalAgents(), 32u);

    // The 33rd micro-tenant should claim a NEW hot slot.
    auto h33 = tb.createMicroTenant();
    QTX_EXPECT(h33.is_valid);
    QTX_EXPECT_EQ(tb.hotUsed(), 2u);
    handles.push_back(h33);

    // Cleanup.
    for (auto h : handles) QTX_EXPECT(tb.destroyTenant(h));
    QTX_EXPECT_EQ(tb.hotUsed(), 0u);
}

QTX_TEST(Tiered, EC115_MicroTenant_DataIsolation) {
    // Write a unique pattern into each of 32 sibling micro-tenants;
    // then verify every read-back is exactly its own pattern (no
    // sub-slot leak between siblings). This is the EC114 hookup
    // verified end-to-end through the bridge.
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    std::vector<tiered::TenantHandle> handles;
    for (int i = 0; i < 32; ++i) {
        auto h = tb.createMicroTenant();
        QTX_EXPECT(h.is_valid);
        auto v = tb.acquireMicroFP32(h);
        QTX_EXPECT_EQ(v.size(), 128u);
        std::memset(v.data(), static_cast<int>(0x40u + i), v.size());
        QTX_EXPECT(tb.release(h));
        handles.push_back(h);
    }

    int leaks = 0;
    for (int i = 0; i < 32; ++i) {
        auto v = tb.acquireMicroFP32(handles[static_cast<usize>(i)]);
        if (v.size() != 128u) { ++leaks; continue; }
        const auto expected = static_cast<std::byte>(0x40u + i);
        for (usize j = 0; j < v.size(); ++j) {
            if (v[j] != expected) { ++leaks; break; }
        }
        QTX_EXPECT(tb.release(handles[static_cast<usize>(i)]));
    }
    QTX_EXPECT_EQ(leaks, 0);

    for (auto h : handles) QTX_EXPECT(tb.destroyTenant(h));
}

QTX_TEST(Tiered, EC115_MicroTenant_PartialReleaseKeepsSlot) {
    // Destroy 31 of 32 siblings; the slot must remain live because
    // tenant 0 is still in it. Verify that tenant 0 still acquires
    // a valid view, and that hotUsed() is still 1.
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    std::vector<tiered::TenantHandle> handles;
    for (int i = 0; i < 32; ++i) handles.push_back(tb.createMicroTenant());
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);

    // Write a marker into tenant 0.
    {
        auto v = tb.acquireMicroFP32(handles[0]);
        std::memset(v.data(), 0xAB, v.size());
        QTX_EXPECT(tb.release(handles[0]));
    }

    // Kill tenants 1..31.
    for (int i = 1; i < 32; ++i) tb.destroyTenant(handles[static_cast<usize>(i)]);

    // The hot slot must still be live — only one occupancy bit remains.
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);
    QTX_EXPECT_EQ(tb.totalAgents(), 1u);

    // Tenant 0 still owns its sub-slot; data is intact.
    auto v = tb.acquireMicroFP32(handles[0]);
    QTX_EXPECT_EQ(v.size(), 128u);
    int bad = 0;
    for (usize i = 0; i < v.size(); ++i) {
        if (v[i] != std::byte{0xAB}) ++bad;
    }
    QTX_EXPECT_EQ(bad, 0);
    QTX_EXPECT(tb.release(handles[0]));

    tb.destroyTenant(handles[0]);
    QTX_EXPECT_EQ(tb.hotUsed(), 0u);
}

QTX_TEST(Tiered, EC115_MicroTenant_FillsHolesNotNewSlots) {
    // After destroying a tenant from a shared slot, a new
    // createMicroTenant must REUSE the freed sub-slot before
    // allocating a new hot slot. Otherwise we'd fragment.
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    std::vector<tiered::TenantHandle> handles;
    for (int i = 0; i < 32; ++i) handles.push_back(tb.createMicroTenant());
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);

    // Free tenant 7.
    tb.destroyTenant(handles[7]);
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);     // slot survives
    QTX_EXPECT_EQ(tb.totalAgents(), 31u);

    // New micro-tenant: must land in the FREED sub-slot, not a new hot slot.
    auto h_new = tb.createMicroTenant();
    QTX_EXPECT(h_new.is_valid);
    QTX_EXPECT_EQ(tb.hotUsed(), 1u);     // STILL one hot slot
    QTX_EXPECT_EQ(tb.totalAgents(), 32u);

    // Cleanup.
    handles[7] = h_new;
    for (auto h : handles) tb.destroyTenant(h);
}

QTX_TEST(Tiered, EC117_MicroTenant_DoesNotEvict) {
    // Fill all 64 hot slots with FULL tenants. Create 32 micro-
    // tenants — these go into shared slots ABOVE the 64 hot full
    // ones, but the full-tenant eviction path must NOT evict a
    // micro-tenant when promoting a cold full-tenant back.
    //
    // Simpler smoke: create one micro, fill hot to capacity, ensure
    // no eviction event names the micro slot.
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    auto micro = tb.createMicroTenant();
    QTX_EXPECT(micro.is_valid);
    const auto stats_before = tb.stats();

    // Fill the remaining 63 full slots and force evictions.
    std::vector<tiered::TenantHandle> full;
    for (int i = 0; i < 63; ++i) full.push_back(tb.createTenant());
    QTX_EXPECT_EQ(tb.hotUsed(), 64u);

    // 65th full tenant must evict one of the previous 63 full tenants
    // — not the micro tenant.
    auto h65 = tb.createTenant();
    QTX_EXPECT(h65.is_valid);
    const auto stats_after = tb.stats();
    QTX_EXPECT(stats_after.evictions > stats_before.evictions);

    // Verify the micro-tenant is still acquirable.
    auto v = tb.acquireMicroFP32(micro);
    QTX_EXPECT_EQ(v.size(), 128u);
    QTX_EXPECT(tb.release(micro));

    // Cleanup.
    tb.destroyTenant(micro);
    for (auto h : full) tb.destroyTenant(h);
    tb.destroyTenant(h65);
}

QTX_TEST(Tiered, EC115_MicroTenant_DestroyRejectsWrongHandleType) {
    // acquireMicroFP32 on a full-slot tenant must return empty.
    // This protects against the bridge accidentally serving the
    // wrong API to the caller (which would otherwise read the
    // first 128B of a 4 KiB payload as if it were the whole thing).
    auto hot  = std::make_unique<HotA>();
    auto cold = std::make_unique<ColdA>();
    Tiered tb(hot.get(), cold.get());

    auto full = tb.createTenant();
    QTX_EXPECT(full.is_valid);
    auto v = tb.acquireMicroFP32(full);
    QTX_EXPECT(v.empty());  // refusal — wrong kind

    tb.destroyTenant(full);
}
