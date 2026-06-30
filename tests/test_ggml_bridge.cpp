// ============================================================================
// @file        test_ggml_bridge.cpp
//@brief Unit tests unsafe ggml_bridge - checking operation via
//real ggml headers (without linking the full ggml runtime).
// @author      QTX Project
// @date        2026-05-12
// ============================================================================

#include "test_harness.hpp"
#include "qtx/unsafe/ggml_bridge/bridge.hpp"
#include "qtx/arena/fractal_arena.hpp"

extern "C" {
#include "ggml.h"
#include "ggml-backend.h"
}

#include <cstring>
#include <memory>
#include <set>
#include <vector>

using namespace qtx::unsafe::ggml_bridge;
using qtx::arena::FractalArena;
using qtx::core::usize;

using SmallArena = FractalArena<64, 4096>;
using LargeArena = FractalArena<256, 4096>;

// ============================================================================
//Basic bridge creation
// ============================================================================

QTX_TEST(Bridge, ConstructWithValidArena) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    QTX_EXPECT(bridge.bufferType() != nullptr);
    QTX_EXPECT_EQ(bridge.arena(), arena.get());
}

QTX_TEST(Bridge, ConstructWithNullArena) {
    ArenaBridge<SmallArena> bridge(nullptr, defaultConfig(2048));
    QTX_EXPECT(bridge.bufferType() == nullptr);
}

QTX_TEST(Bridge, BuftHasCorrectName) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();
    QTX_EXPECT(buft != nullptr);
    const char* name = ggml_backend_buft_name(buft);
    QTX_EXPECT(name != nullptr);
    QTX_EXPECT_EQ(std::strcmp(name, "Qtx-FractalArena"), 0);
}

QTX_TEST(Bridge, BuftReportsCorrectAlignment) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();
    QTX_EXPECT_EQ(ggml_backend_buft_get_alignment(buft), 128u);
}

QTX_TEST(Bridge, BuftReportsCorrectMaxSize) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();
    QTX_EXPECT_EQ(ggml_backend_buft_get_max_size(buft), 2048u);
}

QTX_TEST(Bridge, BuftIsHost) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();
    QTX_EXPECT(ggml_backend_buft_is_host(buft));
}

QTX_TEST(Bridge, MaxBufferSizeClampedToSlotSize) {
    auto arena = std::make_unique<SmallArena>();  //4096-byte slots
    //We request 100 MiB - the bridge must clamp to slotBytes() = 4096.
    ArenaBridge<SmallArena> bridge(arena.get(), BridgeConfig{
        .max_buffer_size  = 100u * 1024u * 1024u,
        .tensor_alignment = 128u,
    });
    auto* buft = bridge.bufferType();
    QTX_EXPECT_EQ(ggml_backend_buft_get_max_size(buft), 4096u);
}

// ============================================================================
//Allocate/free buffer via ggml public API
// ============================================================================

QTX_TEST(Bridge, AllocBufferReturnsValidBuffer) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();

    auto* buf = ggml_backend_buft_alloc_buffer(buft, 1024);
    QTX_EXPECT(buf != nullptr);
    QTX_EXPECT_EQ(ggml_backend_buffer_get_size(buf), 1024u);
    QTX_EXPECT(ggml_backend_buffer_get_base(buf) != nullptr);

    QTX_EXPECT_EQ(arena->usedSlots(), 1u);  //slot is occupied in the arena
    ggml_backend_buffer_free(buf);
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);  //released back
}

QTX_TEST(Bridge, AllocBufferZeroSizeReturnsDummy) {
    //ggml runtime (see ggml-backend.cpp:38-45) on size=0 returns
    //dummy buffer with iface={} (not NULL) without calling our alloc_buffer.
    //This is documented behavior, not a bridge violation.
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();
    auto* buf = ggml_backend_buft_alloc_buffer(buft, 0);
    //The main thing: arena is not used (our bridge was not called).
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
    //Cleanup dummy buffer (ggml runtime).
    if (buf) ggml_backend_buffer_free(buf);
}

QTX_TEST(Bridge, AllocBufferOversizeFails) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();
    auto* buf = ggml_backend_buft_alloc_buffer(buft, 10000);  // > 2048
    QTX_EXPECT(buf == nullptr);
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

QTX_TEST(Bridge, BaseAddressIsInsideArenaMemory) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();
    auto* buf = ggml_backend_buft_alloc_buffer(buft, 1024);

    void* base = ggml_backend_buffer_get_base(buf);
    QTX_EXPECT(base != nullptr);
    //We record the pattern via the ggml API and check that it is included in the arena.
    auto* bytes = static_cast<unsigned char*>(base);
    bytes[0]    = 0xAB;
    bytes[1023] = 0xCD;

    //We do not equalize addresses (the arena is private), but write to base
    //should be visible on the next view() of the same slot... but we don't
    //we know GenID. It is enough that the recording took place without AV.
    QTX_EXPECT_EQ(bytes[0], 0xABu);
    QTX_EXPECT_EQ(bytes[1023], 0xCDu);
    ggml_backend_buffer_free(buf);
}

// ============================================================================
//Multiple buffers - independence of slots
// ============================================================================

QTX_TEST(Bridge, MultipleBuffersIndependent) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();

    auto* buf_a = ggml_backend_buft_alloc_buffer(buft, 1024);
    auto* buf_b = ggml_backend_buft_alloc_buffer(buft, 1024);
    auto* buf_c = ggml_backend_buft_alloc_buffer(buft, 1024);

    QTX_EXPECT(buf_a && buf_b && buf_c);
    void* base_a = ggml_backend_buffer_get_base(buf_a);
    void* base_b = ggml_backend_buffer_get_base(buf_b);
    void* base_c = ggml_backend_buffer_get_base(buf_c);

    //The bases must be different (different slots in the arena).
    QTX_EXPECT(base_a != base_b);
    QTX_EXPECT(base_b != base_c);
    QTX_EXPECT(base_a != base_c);

    //We write unique patterns and check that they do not intersect.
    std::memset(base_a, 0x11, 1024);
    std::memset(base_b, 0x22, 1024);
    std::memset(base_c, 0x33, 1024);

    for (int i = 0; i < 1024; ++i) {
        QTX_EXPECT_EQ(static_cast<unsigned char*>(base_a)[i], 0x11u);
        QTX_EXPECT_EQ(static_cast<unsigned char*>(base_b)[i], 0x22u);
        QTX_EXPECT_EQ(static_cast<unsigned char*>(base_c)[i], 0x33u);
    }

    QTX_EXPECT_EQ(arena->usedSlots(), 3u);
    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_b);
    ggml_backend_buffer_free(buf_c);
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

// ============================================================================
// Arena exhaustion → bridge returns NULL (gracefully)
// ============================================================================

QTX_TEST(Bridge, ArenaExhaustionReturnsNull) {
    auto arena = std::make_unique<SmallArena>();  //64 slots
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();

    std::vector<ggml_backend_buffer_t> buffers;
    //We fill all 64 slots.
    for (int i = 0; i < 64; ++i) {
        auto* buf = ggml_backend_buft_alloc_buffer(buft, 1024);
        QTX_EXPECT(buf != nullptr);
        buffers.push_back(buf);
    }
    //The 65th attempt should return NULL (P1: no quantization).
    auto* extra = ggml_backend_buft_alloc_buffer(buft, 1024);
    QTX_EXPECT(extra == nullptr);

    //We release all buffers.
    for (auto* b : buffers) {
        ggml_backend_buffer_free(b);
    }
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);

    //After cleanup, allocation is possible again.
    auto* recovered = ggml_backend_buft_alloc_buffer(buft, 1024);
    QTX_EXPECT(recovered != nullptr);
    ggml_backend_buffer_free(recovered);
}

// ============================================================================
//Clear via ggml iface should clear the buffer
// ============================================================================

QTX_TEST(Bridge, BufferClearWorksThroughIface) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();

    auto* buf = ggml_backend_buft_alloc_buffer(buft, 1024);
    auto* base = static_cast<unsigned char*>(ggml_backend_buffer_get_base(buf));

    //Fill in 0xFF
    std::memset(base, 0xFF, 1024);
    for (int i = 0; i < 1024; ++i) QTX_EXPECT_EQ(base[i], 0xFFu);

    //Clear via ggml API → should reset.
    ggml_backend_buffer_clear(buf, 0x00);
    for (int i = 0; i < 1024; ++i) QTX_EXPECT_EQ(base[i], 0x00u);

    //Clear with a different meaning.
    ggml_backend_buffer_clear(buf, 0xA5);
    for (int i = 0; i < 1024; ++i) QTX_EXPECT_EQ(base[i], 0xA5u);

    ggml_backend_buffer_free(buf);
}

// ============================================================================
//Pool isolation: different bridges for one arena
//(This is a controversial scenario here - but if you do this, there shouldn't be
//  catastrophic failure.)
// ============================================================================

QTX_TEST(Bridge, MultipleBridgesOnDifferentArenas) {
    auto arena_a = std::make_unique<SmallArena>();
    auto arena_b = std::make_unique<LargeArena>();
    ArenaBridge<SmallArena> bridge_a(arena_a.get(), defaultConfig(2048));
    ArenaBridge<LargeArena> bridge_b(arena_b.get(), defaultConfig(2048));

    QTX_EXPECT(bridge_a.bufferType() != bridge_b.bufferType());
    QTX_EXPECT(bridge_a.bufferType() != nullptr);
    QTX_EXPECT(bridge_b.bufferType() != nullptr);

    auto* buf_a = ggml_backend_buft_alloc_buffer(bridge_a.bufferType(), 512);
    auto* buf_b = ggml_backend_buft_alloc_buffer(bridge_b.bufferType(), 512);
    QTX_EXPECT(buf_a && buf_b);

    QTX_EXPECT_EQ(arena_a->usedSlots(), 1u);
    QTX_EXPECT_EQ(arena_b->usedSlots(), 1u);

    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_b);

    QTX_EXPECT_EQ(arena_a->usedSlots(), 0u);
    QTX_EXPECT_EQ(arena_b->usedSlots(), 0u);
}

// ============================================================================
//Stress test: many allocations + releases via ggml API
// ============================================================================

QTX_TEST(Bridge, StressAllocReleaseLoop) {
    auto arena = std::make_unique<LargeArena>();  //256 slots × 4096
    ArenaBridge<LargeArena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();

    constexpr int kRounds = 1000;
    for (int round = 0; round < kRounds; ++round) {
        //We take 50 buffers, free them, repeat.
        std::vector<ggml_backend_buffer_t> buffers;
        buffers.reserve(50);
        for (int i = 0; i < 50; ++i) {
            auto* buf = ggml_backend_buft_alloc_buffer(buft, 1024);
            QTX_EXPECT(buf != nullptr);
            buffers.push_back(buf);
        }
        //We write different data in each.
        for (size_t i = 0; i < buffers.size(); ++i) {
            auto* base = static_cast<unsigned char*>(
                ggml_backend_buffer_get_base(buffers[i]));
            std::memset(base, static_cast<int>(i & 0xFF), 1024);
        }
        for (auto* b : buffers) ggml_backend_buffer_free(b);
    }
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

// ============================================================================
// EC95 — write-protect for WEIGHTS-class buffers.
//
// A sealed buffer must:
//   (1) accept reads & get_base normally;
//   (2) reject set_tensor / memset_tensor / clear (return without writing);
//   (3) still be freeable via ggml_backend_buffer_free;
//   (4) be idempotent — sealing twice is a successful no-op;
//   (5) reject seal from a foreign ArenaBridge (install_token mismatch).
// ============================================================================

QTX_TEST(Bridge, EC95_SealedBufferRejectsClear) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buf = bridge.allocBuffer(1024);
    QTX_EXPECT(buf != nullptr);
    auto* base = static_cast<unsigned char*>(ggml_backend_buffer_get_base(buf));

    // Fill with a known pattern BEFORE sealing.
    std::memset(base, 0xAA, 1024);

    // Seal and confirm.
    QTX_EXPECT(bridge.sealBuffer(buf));
    QTX_EXPECT(bridge.isBufferSealed(buf));

    // clear() should now be a no-op — pattern survives.
    // This exercises the same write_protected check that gates
    // set_tensor and memset_tensor (those callbacks are only
    // reachable from inside ggml, not from this test harness).
    ggml_backend_buffer_clear(buf, 0x00);
    int corrupted = 0;
    for (size_t i = 0; i < 1024; ++i) if (base[i] != 0xAA) ++corrupted;
    QTX_EXPECT_EQ(corrupted, 0);

    // Lifecycle path still works (free is NOT gated).
    ggml_backend_buffer_free(buf);
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

QTX_TEST(Bridge, EC95_UnsealedBufferAcceptsClear) {
    // Symmetric negative: a buffer that was NEVER sealed must
    // accept clear() normally — proves the gate is path-specific.
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buf = bridge.allocBuffer(1024);
    QTX_EXPECT(buf != nullptr);
    QTX_EXPECT(!bridge.isBufferSealed(buf));  // starts mutable

    auto* base = static_cast<unsigned char*>(ggml_backend_buffer_get_base(buf));
    std::memset(base, 0xAA, 1024);
    ggml_backend_buffer_clear(buf, 0x00);
    int cleared = 0;
    for (size_t i = 0; i < 1024; ++i) if (base[i] == 0x00) ++cleared;
    QTX_EXPECT_EQ(cleared, 1024);  // ALL bytes were cleared

    ggml_backend_buffer_free(buf);
}

QTX_TEST(Bridge, EC95_SealIsIdempotent) {
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));
    auto* buf = bridge.allocBuffer(512);
    QTX_EXPECT(buf != nullptr);

    QTX_EXPECT(bridge.sealBuffer(buf));
    QTX_EXPECT(bridge.isBufferSealed(buf));
    // Second seal must succeed (idempotent) and leave the buffer sealed.
    QTX_EXPECT(bridge.sealBuffer(buf));
    QTX_EXPECT(bridge.isBufferSealed(buf));

    ggml_backend_buffer_free(buf);
}

QTX_TEST(Bridge, EC95_NullAndForeignBuffersRejected) {
    auto arena_a = std::make_unique<SmallArena>();
    auto arena_b = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge_a(arena_a.get(), defaultConfig(2048));
    ArenaBridge<SmallArena> bridge_b(arena_b.get(), defaultConfig(2048));

    // null buffer
    QTX_EXPECT(!bridge_a.sealBuffer(nullptr));
    QTX_EXPECT(!bridge_a.isBufferSealed(nullptr));

    // Buffer that belongs to a DIFFERENT bridge instance must be
    // rejected — install_token mismatch.
    auto* buf_b = bridge_b.allocBuffer(256);
    QTX_EXPECT(buf_b != nullptr);
    QTX_EXPECT(!bridge_a.sealBuffer(buf_b));
    QTX_EXPECT(!bridge_a.isBufferSealed(buf_b));
    // bridge_b CAN seal its own buffer.
    QTX_EXPECT(bridge_b.sealBuffer(buf_b));
    QTX_EXPECT(bridge_b.isBufferSealed(buf_b));

    ggml_backend_buffer_free(buf_b);
}

QTX_TEST(Bridge, EC95_FreshBufferStartsMutable) {
    // Recycle-safety: after destroying a sealed buffer, the next
    // allocation from the SAME pool must start mutable. Without
    // EC95's explicit reset in ContextPool::acquire, the recycled
    // BufferContext would inherit the previous tenant's seal.
    auto arena = std::make_unique<SmallArena>();
    ArenaBridge<SmallArena> bridge(arena.get(), defaultConfig(2048));

    auto* buf1 = bridge.allocBuffer(512);
    QTX_EXPECT(buf1 != nullptr);
    QTX_EXPECT(bridge.sealBuffer(buf1));
    QTX_EXPECT(bridge.isBufferSealed(buf1));
    ggml_backend_buffer_free(buf1);

    // The next allocate may or may not reuse the same pool slot —
    // both paths must yield a mutable buffer.
    auto* buf2 = bridge.allocBuffer(512);
    QTX_EXPECT(buf2 != nullptr);
    QTX_EXPECT(!bridge.isBufferSealed(buf2));
    // Verify by writing through clear() and observing the change.
    auto* base = static_cast<unsigned char*>(ggml_backend_buffer_get_base(buf2));
    std::memset(base, 0xFF, 512);
    ggml_backend_buffer_clear(buf2, 0x00);
    int changed = 0;
    for (size_t i = 0; i < 512; ++i) if (base[i] == 0x00) ++changed;
    QTX_EXPECT_EQ(changed, 512);  // ALL bytes were cleared

    ggml_backend_buffer_free(buf2);
}
