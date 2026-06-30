// ============================================================================
// @file        test_boundary_safety.cpp
//@brief Tests safety boundary: TaggedHeader, checked_cast,
//fail-fast on ABI violations.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================
//
//We test only the positive path (correct use). Test
//fail-fast cannot be done directly - it calls std::abort(), which terminates
//test process. To verify fail-fast behavior, you need a separate
//death-test fork/exec mechanism that goes beyond the scope of unit tests.
//
//However, these tests check:
//1. Magic tags are set correctly by constructors.
//2. checked_cast returns the correct type if magic is valid.
//3. markFreed() invalidates the tag (for subsequent fail-fast detection).
//4. Bridge works correctly through checked_cast in full lifecycle.
//
//If checked_cast were broken, these tests would fail with SIGABRT.
//without reaching assertions.
// ============================================================================

#include "test_harness.hpp"
#include "qtx/unsafe/ggml_bridge/bridge.hpp"
#include "qtx/arena/fractal_arena.hpp"

extern "C" {
#include "ggml.h"
#include "ggml-backend.h"
}

//Direct access to safety.hpp - testing low-level primitives.
#include "safety.hpp"

#include <cstring>
#include <memory>
#include <vector>

using namespace qtx::unsafe::ggml_bridge;
using namespace qtx::unsafe::ggml_bridge::safety;
using qtx::arena::FractalArena;
using qtx::core::u64;

// ============================================================================
//TaggedHeader: basic layout and invariants
// ============================================================================

QTX_TEST(Boundary, TaggedHeaderSize) {
    // EC137: TaggedHeader grew from 16 to 24 bytes when install_token
    // widened from u32 to u64 to defeat the 2-billion wrap.
    QTX_EXPECT_EQ(sizeof(TaggedHeader), 24u);
    QTX_EXPECT_EQ(alignof(TaggedHeader), 8u);
}

QTX_TEST(Boundary, TaggedHeaderDefaultConstruction) {
    TaggedHeader h;
    QTX_EXPECT_EQ(h.magic, kUninitMagic);
    QTX_EXPECT_EQ(h.generation, 0u);
}

QTX_TEST(Boundary, TaggedHeaderExplicitMagic) {
    TaggedHeader h(kBufferContextMagic);
    QTX_EXPECT_EQ(h.magic, kBufferContextMagic);
    QTX_EXPECT_EQ(h.generation, 0u);
}

QTX_TEST(Boundary, MagicValuesAreDistinct) {
    //Each magic must be unique for reliable diagnosis.
    QTX_EXPECT_NE(kBufferContextMagic, kArenaAccessMagic);
    QTX_EXPECT_NE(kBufferContextMagic, kBuftStorageMagic);
    QTX_EXPECT_NE(kBufferContextMagic, kFreedMagic);
    QTX_EXPECT_NE(kArenaAccessMagic,   kBuftStorageMagic);
    QTX_EXPECT_NE(kArenaAccessMagic,   kFreedMagic);
    QTX_EXPECT_NE(kBuftStorageMagic,   kFreedMagic);
    QTX_EXPECT_NE(kUninitMagic,        kFreedMagic);
}

QTX_TEST(Boundary, MagicValuesContainAsciiSignature) {
    //The upper bytes should contain "QTX" for post-mortem grep.
    // 0x4C495448 = "LITH" (little-endian byte view: 4C 49 54 48 = L I T H)
    constexpr u64 lith_prefix = 0x4C495448ull;
    QTX_EXPECT_EQ(kBufferContextMagic >> 32u, lith_prefix);
    QTX_EXPECT_EQ(kArenaAccessMagic   >> 32u, lith_prefix);
    QTX_EXPECT_EQ(kBuftStorageMagic   >> 32u, lith_prefix);
}

QTX_TEST(Boundary, MarkFreedInvalidatesMagic) {
    TaggedHeader h(kBufferContextMagic);
    QTX_EXPECT_EQ(h.magic, kBufferContextMagic);

    markFreed(h);

    QTX_EXPECT_EQ(h.magic, kFreedMagic);
    QTX_EXPECT_EQ(h.generation, 1u);  // generation incremented
}

QTX_TEST(Boundary, MarkFreedBumpsGeneration) {
    TaggedHeader h(kBufferContextMagic);
    markFreed(h);
    markFreed(h);
    markFreed(h);
    //Generation accumulates across releases - can be used
    //to track life cycle history.
    QTX_EXPECT_EQ(h.generation, 3u);
}

// ============================================================================
//checked_cast: if correct, magic returns a valid pointer
// ============================================================================

namespace {
//Test structure with TaggedHeader as the first field.
struct TestContext {
    TaggedHeader header;
    int value;
    double padding;
};
static_assert(offsetof(TestContext, header) == 0);
static_assert(std::is_standard_layout_v<TestContext>);

constexpr u64 kTestContextMagic = 0x4C495448'CAFE0000ull;
}  // namespace

QTX_TEST(Boundary, CheckedCastValidMagic) {
    TestContext ctx;
    ctx.header = TaggedHeader(kTestContextMagic);
    ctx.value = 42;
    ctx.padding = 3.14;

    //We transmit via void* (simulating the C-ABI border).
    void* opaque = &ctx;
    auto* recovered = checked_cast<TestContext>(opaque, kTestContextMagic, "test");

    QTX_EXPECT(recovered == &ctx);
    QTX_EXPECT_EQ(recovered->value, 42);
    QTX_EXPECT_EQ(recovered->padding, 3.14);
}

QTX_TEST(Boundary, CheckedCastConstOverload) {
    TestContext ctx;
    ctx.header = TaggedHeader(kTestContextMagic);
    ctx.value = 99;

    const void* opaque = &ctx;
    const auto* recovered = checked_cast<TestContext>(
        opaque, kTestContextMagic, "test const");
    QTX_EXPECT(recovered == &ctx);
    QTX_EXPECT_EQ(recovered->value, 99);
}

QTX_TEST(Boundary, CheckedCastReturnsActualType) {
    //We make sure that the compiler really returned TestContext*,
    //and not just void*. This is automatically checked by the compiler
    //(if void* were returned, ctx->value would not compile).
    TestContext ctx;
    ctx.header = TaggedHeader(kTestContextMagic);
    ctx.value = 7;

    void* opaque = &ctx;
    auto* recovered = checked_cast<TestContext>(opaque, kTestContextMagic, "type test");
    //The compiler-checked return type is TestContext*.
    static_assert(std::is_same_v<decltype(recovered), TestContext*>);
    QTX_EXPECT_EQ(recovered->value, 7);
}

// ============================================================================
//Bridge through full lifecycle - all checked_casts will pass
// ============================================================================

QTX_TEST(Boundary, BridgeFullLifecycleThroughCheckedCast) {
    //This test duplicates Bridge/AllocBufferReturnsValidBuffer, but
    //in the context of safety: if checked_cast were broken, we would
    //got SIGABRT here.
    using Arena = FractalArena<64, 4096>;
    auto arena = std::make_unique<Arena>();
    ArenaBridge<Arena> bridge(arena.get(), defaultConfig(2048));

    auto* buft = bridge.bufferType();
    QTX_EXPECT(buft != nullptr);

    //Each of these calls internally does a checked_cast.
    QTX_EXPECT_EQ(ggml_backend_buft_get_alignment(buft), 128u);
    QTX_EXPECT_EQ(ggml_backend_buft_get_max_size(buft), 2048u);

    auto* buf = ggml_backend_buft_alloc_buffer(buft, 1024);
    QTX_EXPECT(buf != nullptr);

    //get_base also goes through checked_cast.
    auto* base = ggml_backend_buffer_get_base(buf);
    QTX_EXPECT(base != nullptr);

    //Recording via bridge → checked_cast is called inside memset_tensor.
    std::memset(base, 0xAB, 1024);

    //Clear is another way through checked_cast.
    ggml_backend_buffer_clear(buf, 0x00);
    auto* bytes = static_cast<unsigned char*>(base);
    for (int i = 0; i < 1024; ++i) QTX_EXPECT_EQ(bytes[i], 0x00u);

    //Free is the last checked_cast of the path.
    ggml_backend_buffer_free(buf);
    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}

QTX_TEST(Boundary, MultipleBridgesIndependentMagic) {
    //Two different bridges - each creates its own BuftStorage with
    //with its magic-initialized header. Let's check that
    //they don't conflict.
    using Arena = FractalArena<64, 4096>;
    auto arena_a = std::make_unique<Arena>();
    auto arena_b = std::make_unique<Arena>();

    ArenaBridge<Arena> bridge_a(arena_a.get(), defaultConfig(2048));
    ArenaBridge<Arena> bridge_b(arena_b.get(), defaultConfig(2048));

    auto* buft_a = bridge_a.bufferType();
    auto* buft_b = bridge_b.bufferType();
    QTX_EXPECT(buft_a != buft_b);

    //Each buft goes through its own checked_cast during the alignment query.
    QTX_EXPECT_EQ(ggml_backend_buft_get_alignment(buft_a), 128u);
    QTX_EXPECT_EQ(ggml_backend_buft_get_alignment(buft_b), 128u);

    auto* buf_a = ggml_backend_buft_alloc_buffer(buft_a, 512);
    auto* buf_b = ggml_backend_buft_alloc_buffer(buft_b, 512);
    QTX_EXPECT(buf_a && buf_b);

    QTX_EXPECT_EQ(arena_a->usedSlots(), 1u);
    QTX_EXPECT_EQ(arena_b->usedSlots(), 1u);

    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_b);
}

QTX_TEST(Boundary, ContextPoolReturnsValidMagicAfterReuse) {
    //ContextPool reuses slots. After release() the slot is marked
    //kFreedMagic. After re-acquire() it should be
    //kBufferContextMagic reinitialized.
    //
    //We check this indirectly through the bridge: create, release,
    //create again - both buffers should work.
    using Arena = FractalArena<64, 4096>;
    auto arena = std::make_unique<Arena>();
    ArenaBridge<Arena> bridge(arena.get(), defaultConfig(2048));
    auto* buft = bridge.bufferType();

    //We create and release several times - reusing the pool.
    for (int round = 0; round < 100; ++round) {
        auto* buf = ggml_backend_buft_alloc_buffer(buft, 256);
        QTX_EXPECT(buf != nullptr);
        //If magic is not reinitialized, get_base via
        //checked_cast would call abort.
        auto* base = ggml_backend_buffer_get_base(buf);
        QTX_EXPECT(base != nullptr);
        ggml_backend_buffer_free(buf);
    }

    QTX_EXPECT_EQ(arena->usedSlots(), 0u);
}
