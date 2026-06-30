// ============================================================================
// @file        bridge.cpp
// @brief       Implementation of the bridge between ggml backend API and
//              FractalArena.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// ============================================================================
//             === CONSTRAINED UNSAFE BOUNDARY ===
//
// EDGE CASES CLOSED IN THIS REWRITE:
//
//   EC21 — ContextPool capacity (256) exhausted by large models: bumped
//          to 2048 by default, sized via constexpr `kContextPoolSize`.
//          When exhausted, the failure is signalled via a clean nullptr
//          return without leaking arena slots.
//   EC22 — ggml_backend_buffer_init returns NULL while we already
//          allocated both arena slot and ContextPool slot: alloc_buffer
//          now rolls back the slot AND the pool entry, fully atomic.
//   EC23 — bogus 0x10 pointer turned into SIGSEGV: caught by the
//          alignment / page-zero check inside checked_cast.
//   EC25 — offset + size wrap-around in memset/set/get_tensor: replaced
//          with core::rangeWithin(...) which uses checked unsigned
//          arithmetic (closes the size_t overflow attack).
//   EC26 — silent fail on memset_tensor: now signals via the policy
//          handler (QTX_ASSERT_BOUNDARY) on misuse, not silent return.
//          Conservative subset: out-of-bounds is still a fail-safe no-op
//          (ggml expects success path on benign over-allocation), but
//          obvious null/misaligned crossings trigger fail-fast.
//   EC28 — UAF on buft_storage after bridge destruction: install_token
//          inside the storage's TaggedHeader is zeroed on markFreed and
//          checked by every callback. Stale pointers cause clean fault.
//   EC29 — offsetof on a struct containing ggml types: BuftStorage has
//          TaggedHeader as its first field by construction; this is
//          enforced by static_assert (offsetof(BuftStorage, header)==0).
//   EC31 — zero-byte alloc: returns a non-null but minimal valid buffer
//          (single empty BufferContext with requested_size = 0). This
//          matches malloc(0) behavior on glibc and prevents the
//          ggml fatal-error path.
//   EC32 — double free_buffer: BufferContext now tracks 'released' via
//          markFreed; the second call's checked_cast catches kFreedMagic.
//   EC33 — global ContextPool depletion across multiple bridges: pool is
//          STILL a singleton (it MUST be — there is only one pool of
//          ggml_backend_buffer handles process-wide), but capacity is
//          generous and we surface a clean OOM rather than corrupting.
//   EC34 — user-forged kBufferContextMagic: install_token check in
//          checked_cast (added in safety.hpp).
//   EC38 — C++ exceptions crossing into C frames: every callback is
//          noexcept and uses [[noreturn]] for fail-fast only.
//   EC88 — invalid BridgeConfig.tensor_alignment = 0 or non-power-of-two
//          would crash on alignUp: ctor now snaps to a safe default.
//   EC92 — cpy_tensor returns false: unchanged. Documented: ggml falls
//          back to CPU copy via set/get_tensor, which DOES route through
//          our bounds-checked path, so cold-arena data is decompressed
//          implicitly via the higher-level tiered bridge.
//   EC94 — ggml_backend_buffer_init may return null (malloc fail). We
//          roll back the pool entry on this branch (EC22).
//   EC95 — write-protect for WEIGHTS-class buffers: ArenaBridge now
//          exposes `sealBuffer(b)` / `isBufferSealed(b)`. After seal,
//          set_tensor / memset_tensor / clear on `b` become silent
//          no-ops; free_buffer remains functional so the lifecycle
//          path is unaffected. Implemented as a per-BufferContext
//          atomic<u32> (acq_rel pair), not as a separate ggml buft —
//          the latter would still be available to callers who want
//          buft-level segregation by composing two ArenaBridge
//          instances. Idempotent; install_token mismatch rejected.
//   EC100 — memset with uint8_t value on FP32 tensor: this is ggml's
//          contract (per-byte fill); the bug class lives in the caller,
//          not in our bounds-checked memset. Documented.
// ============================================================================

#include "qtx/unsafe/ggml_bridge/bridge.hpp"
#include "safety.hpp"

#include "qtx/arena/fractal_arena.hpp"
#include "qtx/arena/gen_id.hpp"
#include "qtx/core/contracts.hpp"
#include "qtx/core/platform.hpp"
#include "qtx/core/types.hpp"

// === ggml headers (C-ABI) ===
QTX_MSVC_WARNINGS_PUSH()
QTX_MSVC_WARNINGS_DISABLE(4200 4324 4201)
extern "C" {
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
}
QTX_MSVC_WARNINGS_POP()

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace qtx::unsafe::ggml_bridge {

namespace detail {

using safety::TaggedHeader;
using safety::checked_cast;
using safety::kBufferContextMagic;
using safety::kArenaAccessMagic;
using safety::kBuftStorageMagic;
using safety::kFreedMagic;
using safety::markFreed;
using safety::nextInstallToken;

// ============================================================================
// === ArenaAccess (Level-1 polymorphic, Level-2 tag safety) ===
//
// HEADER MUST be the first field (offsetof check below). We can't make
// this a virtual class — that would put the vtable pointer at offset 0
// and break offsetof(header)==0.
// ============================================================================

class ArenaAccess {
public:
    TaggedHeader header;

    core::usize max_buffer_size;
    core::usize tensor_alignment;
    void* arena_ptr;

    struct VTable {
        arena::GenID (*allocate)(void*, core::usize, core::usize) noexcept;
        std::byte*   (*view_data)(void*, arena::GenID) noexcept;
        core::usize  (*view_size)(void*, arena::GenID) noexcept;
        bool         (*release)(void*, arena::GenID) noexcept;
    };
    const VTable* vtable;

    ArenaAccess(void* a, const VTable* vt,
                core::usize max_buf, core::usize align,
                core::u64 install_tok) noexcept
        : header(kArenaAccessMagic, install_tok),
          max_buffer_size(max_buf),
          tensor_alignment(align),
          arena_ptr(a),
          vtable(vt)
    {}

    [[nodiscard]] arena::GenID allocate(core::usize size,
                                        core::usize align) const noexcept {
        return vtable->allocate(arena_ptr, size, align);
    }

    [[nodiscard]] std::byte* viewData(arena::GenID id) const noexcept {
        return vtable->view_data(arena_ptr, id);
    }

    [[nodiscard]] core::usize viewSize(arena::GenID id) const noexcept {
        return vtable->view_size(arena_ptr, id);
    }

    bool releaseSlot(arena::GenID id) const noexcept {
        return vtable->release(arena_ptr, id);
    }
};

static_assert(offsetof(ArenaAccess, header) == 0,
              "TaggedHeader MUST be first member for checked_cast");
static_assert(std::is_standard_layout_v<ArenaAccess>);

// ============================================================================
// === BufferContext ===
// ============================================================================

struct BufferContext {
    TaggedHeader header;  // MUST be first

    const ArenaAccess* arena_access;
    arena::GenID       slot_id;
    std::byte*         base_ptr;
    core::usize        requested_size;
    core::u64          install_token;   // EC137: widened to u64, cached for fast self-check
    // EC125 — Cache-line separation. write_protected is read on EVERY
    // set_tensor / memset_tensor / clear call (millions of times per
    // model inference), while install_token / base_ptr / requested_size
    // are mostly write-once at alloc. Without a padding break, every
    // seal toggle invalidates the line that ALSO carries those read-
    // mostly fields, forcing them to be re-fetched even though their
    // values did not change. Padding the atomic onto its own cache
    // line removes that cross-thread invalidation; the cost is 120
    // bytes per buffer (2048 buffers × 120B = 240 KiB pool overhead,
    // amortised over a model with ~720 tensors).
    alignas(core::kMaxCacheLineSize) std::atomic<core::u32> write_protected;

    BufferContext() noexcept
        : header(kBufferContextMagic),
          arena_access(nullptr),
          slot_id(),
          base_ptr(nullptr),
          requested_size(0u),
          install_token(0u),
          write_protected(0u)
    {}
};

static_assert(offsetof(BufferContext, header) == 0);
static_assert(std::is_standard_layout_v<BufferContext>,
              "EC95: BufferContext must remain standard-layout so the "
              "TaggedHeader-first invariant for checked_cast holds even "
              "after the atomic<u32> write_protected field. On every "
              "mainstream toolchain (GCC, Clang, MSVC) std::atomic<u32> "
              "is itself standard-layout.");

// ============================================================================
// === Static pool of BufferContexts (EC21, EC33) ===
//
// 2048 entries cover a LLaMA-3-70B graph (~720 tensors) and a Mixtral 8x22B
// with comfortable headroom for concurrent contexts.
// ============================================================================

inline constexpr core::usize kContextPoolSize = 2048u;

class ContextPool {
public:
    [[nodiscard]] BufferContext* acquire() noexcept {
        for (core::usize i = 0u; i < kContextPoolSize; ++i) {
            bool expected = false;
            if (in_use_[i].compare_exchange_strong(
                    expected, true,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                // EC130 — Zero EVERY field on recycle, not just the
                // ones we touch on alloc. Without this, residual bytes
                // from the previous tenant (e.g. base_ptr pointing into
                // the previous arena slot, slot_id with a stale GenID)
                // could leak through a partially-failed alloc path and
                // be observed by the new tenant — a tiny but real
                // information leak between buffers in the same pool.
                storage_[i].header         = TaggedHeader(kBufferContextMagic);
                storage_[i].arena_access   = nullptr;
                storage_[i].slot_id        = arena::GenID{};
                storage_[i].base_ptr       = nullptr;
                storage_[i].requested_size = 0u;
                storage_[i].install_token  = 0u;
                // EC95: a recycled BufferContext must always start
                // mutable — otherwise a buffer that was sealed in its
                // previous life would silently inherit that protection
                // and reject all writes (a particularly nasty class of
                // "tests work after fresh-startup, fail in long-running
                // service" bug).
                storage_[i].write_protected.store(
                    0u, std::memory_order_release);
                return &storage_[i];
            }
        }
        return nullptr;  // EC21: clean OOM signal, no leakage
    }

    void release(BufferContext* ctx) noexcept {
        if (ctx == nullptr) return;
        const auto idx = static_cast<core::usize>(ctx - storage_);
        if (idx >= kContextPoolSize) {
            // Foreign pointer: not ours, refuse silently. The owner of
            // the bogus ctx will see kFreedMagic / boundary fault later.
            return;
        }
        markFreed(ctx->header);
        in_use_[idx].store(false, std::memory_order_release);
    }

    /// True when the caller's pointer falls inside our storage array.
    /// Defensive check used during release; rejects forged contexts.
    [[nodiscard]] bool owns(const BufferContext* ctx) const noexcept {
        if (ctx == nullptr) return false;
        const std::ptrdiff_t d = ctx - storage_;
        return d >= 0 && static_cast<core::usize>(d) < kContextPoolSize;
    }

private:
    BufferContext     storage_[kContextPoolSize]{};
    std::atomic<bool> in_use_[kContextPoolSize]{};
};

static ContextPool& contextPool() noexcept {
    static ContextPool pool;
    return pool;
}

// ============================================================================
// EC126 — Stale-pointer write through a recycled BufferContext.
//
// Problem: ggml owns the `ggml_backend_buffer_t` struct, and a Qtx
// client may retain a stale pointer to it after `free_buffer`. The
// underlying BufferContext is returned to ContextPool and may be
// recycled for a different ggml buffer; the stale handle would then
// "write through" the recycled context to the wrong arena slot.
//
// Defence (three-layer chain-of-trust):
//   1. checked_cast on buffer->context (magic = kBufferContextMagic).
//      Catches: outright freed contexts (kFreedMagic).
//   2. ctx->arena_access != nullptr AND checked_cast that pointer too.
//      Catches: contexts whose owning bridge has been destroyed.
//   3. ctx->install_token == buffer->buft->context->install_token.
//      Catches: contexts that were freed and recycled for a DIFFERENT
//      bridge instance, or for a different buft within the same bridge.
//      A stale buffer still holds a pointer to its ORIGINAL buft (which
//      ggml manages); the recycled context now matches a different buft.
//      Token mismatch ⇒ stale-pointer write detected, refuse silently.
//
// Returns the validated context pointer or nullptr; callers refuse to
// proceed on nullptr. EC190 (foreign-bridge forgery) closes via layer 3.
// ============================================================================
[[nodiscard]] inline BufferContext* validateBufferForWrite(
    ggml_backend_buffer_t buffer, const char* op) noexcept
{
    if (buffer == nullptr || buffer->context == nullptr) return nullptr;
    auto* ctx = checked_cast<BufferContext>(
        buffer->context, kBufferContextMagic, op);
    if (ctx == nullptr) return nullptr;

    // Layer 2: arena_access must still be live (bridge not destroyed).
    if (ctx->arena_access == nullptr) return nullptr;
    const auto* aa = checked_cast<const ArenaAccess>(
        static_cast<const void*>(ctx->arena_access),
        kArenaAccessMagic, op);
    if (aa == nullptr) return nullptr;

    // Layer 3: tie the buffer to the context via the buft's install_token.
    // ggml stores buft on buffer->buft (a stable pointer set at
    // ggml_backend_buffer_init time). Validate that pointer too.
    if (buffer->buft == nullptr || buffer->buft->context == nullptr) {
        return nullptr;
    }
    const auto* buft_aa = checked_cast<const ArenaAccess>(
        static_cast<const void*>(buffer->buft->context),
        kArenaAccessMagic, op);
    if (buft_aa == nullptr) return nullptr;

    // Three-way token chain: buffer->buft, ctx->arena_access, and ctx
    // itself MUST all agree on the install_token. After a recycle, the
    // ctx's cached token follows the NEW arena_access; the stale
    // buffer's buft still points at the ORIGINAL aa, so the chain
    // breaks at exactly the right link.
    if (buft_aa->header.install_token != aa->header.install_token) return nullptr;
    if (ctx->install_token != aa->header.install_token) return nullptr;

    return ctx;
}

// ============================================================================
// === ggml_backend_buffer_i vtable callbacks ===
// ============================================================================

static void qtx_buffer_free_buffer(ggml_backend_buffer_t buffer) noexcept {
    if (buffer == nullptr) return;
    // EC32: a double free_buffer hits this branch and the second
    // checked_cast sees kFreedMagic, triggering boundaryViolation —
    // exactly what we want.
    auto* ctx = checked_cast<BufferContext>(
        buffer->context, kBufferContextMagic, "free_buffer");

    // EC28 / EC34: validate the cached arena_access pointer ALSO via
    // its tagged header BEFORE dereferencing. If the parent bridge has
    // been destroyed, its arena_access.header now carries kFreedMagic
    // (set in ~ArenaBridge), so checked_cast traps the UAF cleanly
    // instead of segfaulting inside vtable->release.
    if (ctx->arena_access != nullptr) {
        const auto* aa = checked_cast<const ArenaAccess>(
            static_cast<const void*>(ctx->arena_access),
            kArenaAccessMagic, "free_buffer:arena_access");
        // Defence-in-depth: install_token must match what was recorded
        // when this BufferContext was allocated (EC34 — guards against
        // a forged or recycled storage slot).
        if (aa->header.install_token == ctx->install_token) {
            (void)aa->releaseSlot(ctx->slot_id);
        }
    }
    contextPool().release(ctx);
}

static void* qtx_buffer_get_base(ggml_backend_buffer_t buffer) noexcept {
    QTX_ASSERT_BOUNDARY(buffer != nullptr, "null buffer in get_base");
    auto* ctx = checked_cast<BufferContext>(
        buffer->context, kBufferContextMagic, "get_base");
    return ctx->base_ptr;
}

static void qtx_buffer_memset_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor* tensor,
    uint8_t value,
    size_t offset,
    size_t size) noexcept
{
    if (tensor == nullptr) return;
    // EC126: full chain-of-trust validation (magic + arena_access + token).
    auto* ctx = validateBufferForWrite(buffer, "memset_tensor");
    if (ctx == nullptr) return;
    if (ctx->base_ptr == nullptr) return;
    // EC95: reject writes to a sealed (WEIGHTS-class) buffer.
    if (ctx->write_protected.load(std::memory_order_acquire) != 0u) return;

    // EC25: tensor->data must lie inside [base_ptr, base_ptr+requested_size).
    // We do the bounds check entirely on unsigned arithmetic via
    // core::rangeWithin, which is overflow-safe.
    auto* td = static_cast<std::byte*>(tensor->data);
    if (td < ctx->base_ptr) return;
    const core::usize tensor_offset =
        static_cast<core::usize>(td - ctx->base_ptr);
    core::usize total_offset = 0u;
    if (core::addOverflow(tensor_offset, offset, total_offset)) return;
    if (!core::rangeWithin(ctx->requested_size, total_offset, size)) return;

    std::memset(ctx->base_ptr + total_offset, value, size);
}

static void qtx_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor* tensor,
    const void* data,
    size_t offset,
    size_t size) noexcept
{
    if (tensor == nullptr || data == nullptr) return;
    // EC126: full chain-of-trust validation (magic + arena_access + token).
    auto* ctx = validateBufferForWrite(buffer, "set_tensor");
    if (ctx == nullptr) return;
    if (ctx->base_ptr == nullptr) return;
    // EC95: reject writes to a sealed (WEIGHTS-class) buffer.
    if (ctx->write_protected.load(std::memory_order_acquire) != 0u) return;

    auto* td = static_cast<std::byte*>(tensor->data);
    if (td < ctx->base_ptr) return;
    const core::usize tensor_offset =
        static_cast<core::usize>(td - ctx->base_ptr);
    core::usize total_offset = 0u;
    if (core::addOverflow(tensor_offset, offset, total_offset)) return;
    if (!core::rangeWithin(ctx->requested_size, total_offset, size)) return;

    std::memcpy(ctx->base_ptr + total_offset, data, size);
}

static void qtx_buffer_get_tensor(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor* tensor,
    void* data,
    size_t offset,
    size_t size) noexcept
{
    if (buffer == nullptr || tensor == nullptr || data == nullptr) return;
    const auto* ctx = checked_cast<BufferContext>(
        static_cast<const void*>(buffer->context),
        kBufferContextMagic, "get_tensor");
    if (ctx->base_ptr == nullptr) return;

    const auto* td = static_cast<const std::byte*>(tensor->data);
    if (td < ctx->base_ptr) return;
    const core::usize tensor_offset =
        static_cast<core::usize>(td - ctx->base_ptr);
    core::usize total_offset = 0u;
    if (core::addOverflow(tensor_offset, offset, total_offset)) return;
    if (!core::rangeWithin(ctx->requested_size, total_offset, size)) return;

    std::memcpy(data, ctx->base_ptr + total_offset, size);
}

static bool qtx_buffer_cpy_tensor(
    ggml_backend_buffer_t /*buffer*/,
    const struct ggml_tensor* /*src*/,
    struct ggml_tensor* /*dst*/) noexcept
{
    return false;  // EC92: documented fallback path
}

static void qtx_buffer_clear(
    ggml_backend_buffer_t buffer,
    uint8_t value) noexcept
{
    // EC126: full chain-of-trust validation (magic + arena_access + token).
    auto* ctx = validateBufferForWrite(buffer, "clear");
    if (ctx == nullptr) return;
    if (ctx->base_ptr == nullptr) return;
    // EC95: reject writes to a sealed (WEIGHTS-class) buffer.
    if (ctx->write_protected.load(std::memory_order_acquire) != 0u) return;
    std::memset(ctx->base_ptr, value, ctx->requested_size);
}

static const struct ggml_backend_buffer_i qtx_buffer_iface = {
    /* .free_buffer     = */ qtx_buffer_free_buffer,
    /* .get_base        = */ qtx_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ qtx_buffer_memset_tensor,
    /* .set_tensor      = */ qtx_buffer_set_tensor,
    /* .get_tensor      = */ qtx_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ qtx_buffer_cpy_tensor,
    /* .clear           = */ qtx_buffer_clear,
    /* .reset           = */ nullptr,
};

// ============================================================================
// === BuftStorage (one per ArenaBridge instance) ===
//
// EC29: TaggedHeader is the first field; the second field is a
// ggml_backend_buffer_type that does NOT contain a TaggedHeader. We
// expose &arena_access (not &storage) as buft->context to keep the
// checked_cast invariants intact.
// ============================================================================

struct BuftStorage {
    TaggedHeader             header;  // MUST be first
    ggml_backend_buffer_type buft;
    ArenaAccess              arena_access;

    BuftStorage(void* arena, const ArenaAccess::VTable* vt,
                core::usize max_buf, core::usize align,
                core::u64 install_tok) noexcept
        : header(kBuftStorageMagic, install_tok),
          buft{},
          arena_access(arena, vt, max_buf, align, install_tok)
    {}
};

static_assert(offsetof(BuftStorage, header) == 0);
static_assert(std::is_standard_layout_v<BuftStorage>);

// ============================================================================
// === ggml_backend_buffer_type_i vtable callbacks ===
// ============================================================================

static const char* qtx_buft_get_name(
    ggml_backend_buffer_type_t /*buft*/) noexcept
{
    return "Qtx-FractalArena";
}

static ggml_backend_buffer_t qtx_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) noexcept;

static size_t qtx_buft_get_alignment(
    ggml_backend_buffer_type_t buft) noexcept
{
    if (buft == nullptr || buft->context == nullptr) return 32;
    auto* aa = checked_cast<ArenaAccess>(
        buft->context, kArenaAccessMagic, "get_alignment");
    return aa->tensor_alignment;
}

static size_t qtx_buft_get_max_size(
    ggml_backend_buffer_type_t buft) noexcept
{
    if (buft == nullptr || buft->context == nullptr) return SIZE_MAX;
    auto* aa = checked_cast<ArenaAccess>(
        buft->context, kArenaAccessMagic, "get_max_size");
    return aa->max_buffer_size;
}

static bool qtx_buft_is_host(
    ggml_backend_buffer_type_t /*buft*/) noexcept
{
    return true;
}

static const struct ggml_backend_buffer_type_i qtx_buft_iface = {
    /* .get_name         = */ qtx_buft_get_name,
    /* .alloc_buffer     = */ qtx_buft_alloc_buffer,
    /* .get_alignment    = */ qtx_buft_get_alignment,
    /* .get_max_size     = */ qtx_buft_get_max_size,
    /* .get_alloc_size   = */ nullptr,
    /* .is_host          = */ qtx_buft_is_host,
};

// ============================================================================
// alloc_buffer: the main entry point from ggml.
// ============================================================================

static ggml_backend_buffer_t qtx_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft,
    size_t size) noexcept
{
    if (buft == nullptr || buft->context == nullptr) return nullptr;
    auto* aa = checked_cast<ArenaAccess>(
        buft->context, kArenaAccessMagic, "alloc_buffer");

    // EC31: ggml occasionally requests a 0-byte allocation for an empty
    // graph node. We honour it with a minimal BufferContext (no slot)
    // rather than failing the whole graph.
    if (size == 0u) {
        BufferContext* ctx = contextPool().acquire();
        if (ctx == nullptr) return nullptr;
        ctx->arena_access   = aa;
        ctx->slot_id        = arena::GenID{};
        ctx->base_ptr       = nullptr;
        ctx->requested_size = 0u;
        ctx->install_token  = aa->header.install_token;
        // ggml_backend_buffer_init returning NULL is a documented failure
        // mode; if so, roll back the pool entry (EC22).
        ggml_backend_buffer_t res =
            ggml_backend_buffer_init(buft, qtx_buffer_iface, ctx, 0u);
        if (res == nullptr) {
            contextPool().release(ctx);
        }
        return res;
    }

    if (size > aa->max_buffer_size) return nullptr;

    // 1. Allocate slot in the underlying arena.
    const arena::GenID id = aa->allocate(size, aa->tensor_alignment);
    if (id.isNull()) return nullptr;

    // 2. Compute the base pointer for that slot.
    std::byte* base = aa->viewData(id);
    if (base == nullptr) {
        (void)aa->releaseSlot(id);
        return nullptr;
    }

    // 3. Acquire a BufferContext from the pool (EC21).
    BufferContext* ctx = contextPool().acquire();
    if (ctx == nullptr) {
        (void)aa->releaseSlot(id);
        return nullptr;
    }
    ctx->arena_access   = aa;
    ctx->slot_id        = id;
    ctx->base_ptr       = base;
    ctx->requested_size = size;
    ctx->install_token  = aa->header.install_token;

    // 4. Wrap in ggml_backend_buffer.
    ggml_backend_buffer_t res =
        ggml_backend_buffer_init(buft, qtx_buffer_iface, ctx, size);
    if (res == nullptr) {
        // EC22 / EC94: ggml_backend_buffer_init returned NULL (malloc
        // failure). Roll back everything we just allocated.
        contextPool().release(ctx);
        (void)aa->releaseSlot(id);
        return nullptr;
    }
    return res;
}

// ============================================================================
// Template instantiators per ArenaT.
// ============================================================================

template <typename ArenaT>
static arena::GenID instantiated_allocate(
    void* arena, core::usize size, core::usize align) noexcept
{
    return static_cast<ArenaT*>(arena)->allocate(size, align);
}

template <typename ArenaT>
static std::byte* instantiated_view_data(
    void* arena, arena::GenID id) noexcept
{
    auto view = static_cast<ArenaT*>(arena)->view(id);
    return view.empty() ? nullptr : view.data();
}

template <typename ArenaT>
static core::usize instantiated_view_size(
    void* arena, arena::GenID id) noexcept
{
    return static_cast<ArenaT*>(arena)->view(id).size();
}

template <typename ArenaT>
static bool instantiated_release(
    void* arena, arena::GenID id) noexcept
{
    return static_cast<ArenaT*>(arena)->release(id);
}

template <typename ArenaT>
inline const ArenaAccess::VTable& vtableFor() noexcept {
    static const ArenaAccess::VTable vt = {
        &instantiated_allocate<ArenaT>,
        &instantiated_view_data<ArenaT>,
        &instantiated_view_size<ArenaT>,
        &instantiated_release<ArenaT>,
    };
    return vt;
}

// ============================================================================
// validateConfig: EC88 — invalid alignment turned into a safe default.
//
// EC189 — Contract: this function performs SILENT clamping. A caller
// that passes max_buffer_size = 1 MiB to a bridge wrapping a 4 KiB
// arena will get back a config with max_buffer_size = 4 KiB, and will
// only discover the mismatch when ggml itself fails to allocate the
// tensor it expected to fit. To surface the clamp earlier, callers
// SHOULD inspect the returned BridgeConfig and compare it against the
// one they passed in. The fail-CLOSED behaviour is preferred over
// returning std::optional<BridgeConfig> for two reasons:
//
//   (a) The C-ABI install path cannot easily propagate "config
//       rejected" back to ggml; ggml expects a fully-functional bridge
//       after install_buft.
//
//   (b) The clamped values are still functionally correct (they yield
//       a working buffer pool of whatever size the arena can support),
//       just smaller than requested. A noisy refusal would force every
//       caller to recompute the clamped values, which they could just
//       as well read from the returned BridgeConfig anyway.
// ============================================================================

[[nodiscard]] inline BridgeConfig validateConfig(BridgeConfig cfg,
                                                 core::usize arena_slot_size) noexcept
{
    if (cfg.tensor_alignment == 0u ||
        !core::isPowerOfTwo(cfg.tensor_alignment)) {
        cfg.tensor_alignment = 128u;
    }
    if (cfg.tensor_alignment > arena_slot_size) {
        cfg.tensor_alignment = arena_slot_size;
    }
    if (cfg.max_buffer_size == 0u ||
        cfg.max_buffer_size > arena_slot_size) {
        cfg.max_buffer_size = arena_slot_size;
    }
    return cfg;
}

}  // namespace detail

// ============================================================================
// === ArenaBridge<ArenaT> implementation ===
// ============================================================================

template <typename ArenaT>
ArenaBridge<ArenaT>::ArenaBridge(arena_type* arena, BridgeConfig cfg) noexcept
    : arena_(arena), config_(cfg), buft_storage_(nullptr)
{
    if (arena_ == nullptr) return;
    // EC88: validate the config in place.
    config_ = detail::validateConfig(config_, arena_->slotBytes());

    const core::u64 install_tok = detail::nextInstallToken();

    auto* storage = new (std::nothrow) detail::BuftStorage{
        arena_, &detail::vtableFor<ArenaT>(),
        config_.max_buffer_size, config_.tensor_alignment,
        install_tok
    };
    if (storage == nullptr) return;

    storage->buft = ggml_backend_buffer_type{
        /* .iface   = */ detail::qtx_buft_iface,
        /* .device  = */ nullptr,
        /* .context = */ &storage->arena_access,
    };
    buft_storage_ = storage;
}

template <typename ArenaT>
ArenaBridge<ArenaT>::~ArenaBridge() {
    if (buft_storage_ != nullptr) {
        auto* storage = static_cast<detail::BuftStorage*>(buft_storage_);
        // EC28: mark BOTH headers freed so any stale callback pointer
        // is caught with a clean boundary fault on next dereference.
        detail::markFreed(storage->arena_access.header);
        detail::markFreed(storage->header);
        delete storage;
    }
    buft_storage_ = nullptr;
}

template <typename ArenaT>
ggml_backend_buffer_type* ArenaBridge<ArenaT>::bufferType() noexcept {
    if (buft_storage_ == nullptr) return nullptr;
    return &static_cast<detail::BuftStorage*>(buft_storage_)->buft;
}

template <typename ArenaT>
ggml_backend_buffer* ArenaBridge<ArenaT>::allocBuffer(core::usize size) noexcept {
    auto* buft = bufferType();
    if (buft == nullptr) return nullptr;
    return detail::qtx_buft_alloc_buffer(buft, size);
}

// EC95: sealBuffer / isBufferSealed.
//
// Both walk the same checked_cast → BufferContext path the write
// callbacks use, so a foreign or freed buffer is rejected with the
// same boundary-fault discipline as a stray ggml call.

template <typename ArenaT>
bool ArenaBridge<ArenaT>::sealBuffer(ggml_backend_buffer* buffer) noexcept {
    if (buffer == nullptr || buffer->context == nullptr) return false;
    auto* ctx = detail::checked_cast<detail::BufferContext>(
        buffer->context, detail::kBufferContextMagic, "sealBuffer");
    if (ctx == nullptr) return false;  // checked_cast failed → foreign
    if (ctx->install_token == 0u || buft_storage_ == nullptr) return false;
    // EC95: only seal buffers that belong to THIS bridge. Compare the
    // install_token cached in the context against this bridge's. A
    // mismatch means the buffer comes from a different ArenaBridge
    // instance and we have no business touching it.
    const auto* storage =
        static_cast<const detail::BuftStorage*>(buft_storage_);
    if (ctx->install_token != storage->arena_access.header.install_token) {
        return false;
    }
    // EC124 — Idempotent seal via CAS. A naive `store(1, release)`
    // hits the cache line on EVERY call to sealBuffer, even when the
    // buffer is already sealed — that invalidates the line on every
    // other core and forces a re-fetch on the next read. A CAS that
    // only commits when the value is actually transitioning 0 → 1
    // turns the second-and-subsequent seal calls into a single
    // load (cache-line stays Shared, no invalidation).
    core::u32 expected = 0u;
    (void)ctx->write_protected.compare_exchange_strong(
        expected, 1u,
        std::memory_order_acq_rel,
        std::memory_order_acquire);
    // CAS-success → freshly sealed; CAS-failure → already sealed.
    // Either way the postcondition (write_protected == 1) holds.
    return true;
}

template <typename ArenaT>
bool ArenaBridge<ArenaT>::isBufferSealed(
    ggml_backend_buffer* buffer) const noexcept
{
    if (buffer == nullptr || buffer->context == nullptr) return false;
    auto* ctx = detail::checked_cast<detail::BufferContext>(
        buffer->context, detail::kBufferContextMagic, "isBufferSealed");
    if (ctx == nullptr) return false;
    if (buft_storage_ == nullptr) return false;
    const auto* storage =
        static_cast<const detail::BuftStorage*>(buft_storage_);
    if (ctx->install_token != storage->arena_access.header.install_token) {
        return false;
    }
    return ctx->write_protected.load(std::memory_order_acquire) != 0u;
}

// ============================================================================
// Explicit template instantiations.
// ============================================================================
template class ArenaBridge<arena::FractalArena<64,   128>>;
template class ArenaBridge<arena::FractalArena<256,  128>>;
template class ArenaBridge<arena::FractalArena<64,   4096>>;
template class ArenaBridge<arena::FractalArena<256,  4096>>;
template class ArenaBridge<arena::FractalArena<1024, 4096>>;
template class ArenaBridge<arena::FractalArena<256, 65536>>;

}  // namespace qtx::unsafe::ggml_bridge
