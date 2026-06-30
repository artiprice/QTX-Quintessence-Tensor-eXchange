// ============================================================================
// @file        bridge.hpp
//@brief Public API GGML-bridge. HA-pure, handle-based.
// @author      QTX Project
// @date        2026-05-12
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// ============================================================================
//                      === UNSAFE FFI BOUNDARY ===
//
//This namespace (qtx::unsafe::ggml_bridge) interprets
//ggml v0.11.1 ABI and provides a ggml-compatible backend buffer,
//served through FractalArena.
//
//This header (bridge.hpp) does NOT contain unsafe operations. He's just
//declares a handle-based API for the upper layers. All pointer-tricks
//and reinterpret_casts are localized in bridge.cpp.
//
//CI-rule: HA-core (include/qtx/{core,arena,selector,ipc}) does NOT have
//rights to include this header. Bridge is a separate layer that
//uses HA-core, but is not used by it.
//
//Behavioral contract:
//- bridge does not own the arena; the caller passes a pointer
//to the statically long-lived FractalArena.
//- bridge returns ggml-compatible handles; owner handles
//remains ggml runtime.
//- all errors are signaled via returning nullptr / null GenID.
// ============================================================================

#pragma once

#include "qtx/arena/fractal_arena.hpp"
#include "qtx/arena/gen_id.hpp"

//Public ggml-handle types (opaque pointers).
//Full definitions are pulled along with ggml-backend.h into bridge.cpp.
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;

namespace qtx::unsafe::ggml_bridge {

// ============================================================================
//Arena parameters for which the bridge is adjusted.
//
//HA: these constants must be the same as those used when
//instance of FractalArena in the caller, otherwise the bridge will not be able to
//return GenIDs valid for the caller's arena.
//
//P1-MVP Design: bridge supports ONE arena type per challenge
//initBuffer(). This gives a strong static guarantee (compile-time
//arena type), without losing flexibility (multiple backends can live
//nearby, each with its own arena).
// ============================================================================
struct BridgeConfig {
    /// The maximum size of a single buffer that ggml can request.
    /// Must be ≤ slotBytes() arena.
    std::size_t max_buffer_size;

    /// The alignment that bridge reports to ggml (via get_alignment).
    /// Must be a power of two, ≥ ggml::TENSOR_ALIGNMENT (32).
    std::size_t tensor_alignment;
};

// ============================================================================
//Default configuration for P2-MVP: 128-byte tensor alignment.
//
//Note: ggml uses 32 bytes as TENSOR_ALIGNMENT by default,
//we raise to 128 for consistency with qtx cache-line alignment.
// ============================================================================
[[nodiscard]] inline constexpr BridgeConfig defaultConfig(std::size_t max_buf) noexcept {
    return BridgeConfig{
        .max_buffer_size  = max_buf,
        .tensor_alignment = 128u,
    };
}

// ============================================================================
// @class ArenaBridge
//@tparam ArenaT Type FractalArena (qtx::arena::FractalArena<...>).
//@brief A template facade that binds the ggml backend buffer to
//specific FractalArena instance.
//
//Life cycle:
//1) ArenaBridge bridge(&my_arena, cfg);     // nothing ggml-specific
//2) auto* buft = bridge.bufferType();        // get buffer_type
//3) ... pass buft to ggml ...
//4) ggml will call buft->iface.alloc_buffer(buft, size);
//5) bridge will allocate a slot in my_arena and return ggml_backend_buffer_t.
//
//HA-invariants:
//- Does not own the arena (lifetime: calling code).
//- Does not use new/malloc after the constructor (buffers from arena).
//- bufferType() returns the same pointer for one bridge.
// ============================================================================
template <typename ArenaT>
class ArenaBridge {
public:
    using arena_type = ArenaT;

    /// Create a bridge for the specified arena.
    /// arena MUST remain a valid pointer for the entire life of the bridge.
    /// arena == nullptr → bridge will be incorrect, all operations will be
    /// return null/error.
    ArenaBridge(arena_type* arena, BridgeConfig cfg) noexcept;

    ArenaBridge(const ArenaBridge&)            = delete;
    ArenaBridge& operator=(const ArenaBridge&) = delete;
    ArenaBridge(ArenaBridge&&)                 = delete;
    ArenaBridge& operator=(ArenaBridge&&)      = delete;

    ~ArenaBridge();

    /// Get a ggml-compatible buffer_type to pass to the ggml runtime.
    /// Returns a pointer that is valid for the entire life of the bridge.
    [[nodiscard]] ggml_backend_buffer_type* bufferType() noexcept;

    /// Direct buffer initialization (for tests and custom integration).
    /// In a production scenario, ggml itself will call alloc_buffer via bufferType().
    /// Returns nullptr on error (OOM or invalid size).
    [[nodiscard]] ggml_backend_buffer* allocBuffer(std::size_t size) noexcept;

    /// EC95 — mark a buffer as write-protected (WEIGHTS-class semantics).
    ///
    /// After this call, every subsequent `set_tensor`, `memset_tensor`,
    /// and `clear` on the buffer becomes a silent no-op. `free_buffer`
    /// is NOT gated — the buffer can still be deallocated normally.
    /// This emulates a separate WEIGHTS-typed buft without changing the
    /// ggml_backend_buffer_type identity (callers who want the buffer
    /// in a distinct buft pool can still allocate it via a separate
    /// ArenaBridge instance — orthogonal to this seal).
    ///
    /// Idempotent: sealing an already-sealed buffer is a successful
    /// no-op. Sealing a null or foreign buffer returns false.
    bool sealBuffer(ggml_backend_buffer* buffer) noexcept;

    /// EC95 — query whether a buffer is currently sealed.
    /// Returns false for null, foreign, or already-freed buffers.
    [[nodiscard]] bool isBufferSealed(ggml_backend_buffer* buffer) const noexcept;

    /// Get the associated arena (for tests).
    [[nodiscard]] arena_type* arena() noexcept { return arena_; }

    [[nodiscard]] const BridgeConfig& config() const noexcept { return config_; }

private:
    arena_type*  arena_;
    BridgeConfig config_;
    //Storing buffer_type: real definition in bridge.cpp,
    //here is just a pointer so as not to drag ggml-headers into the public API.
    void* buft_storage_;
};

}  // namespace qtx::unsafe::ggml_bridge
