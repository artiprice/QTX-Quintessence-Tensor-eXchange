// ============================================================================
// @file        safety.hpp
// @brief       Centralised safety primitives for the UNSAFE FFI BOUNDARY.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// ============================================================================
//             === BOUNDARY SAFETY PROTECTIONS (Level 2) ===
//
// EDGE CASES CLOSED IN THIS REWRITE:
//
//   EC23 — checked_cast on bogus 0x10 pointer: we now also reject any
//          mis-aligned pointer (alignof(TaggedHeader) violation) before
//          dereferencing, turning a SIGSEGV into a clean boundary fault.
//   EC34 — user-forged kBufferContextMagic: each magic must agree with
//          the install_token recorded by the bridge on construction.
//   EC39 — neighbouring kFreedMagic overwrite by buggy agent: install
//          token is cleared on markFreed, defeating a magic-only forge.
//   EC40 — DoS via std::abort in multi-tenant hosts: BoundaryViolation
//          now routes through core::invokeFailHandler so the host may
//          unwind the request instead of terminating the process.
//
// Generation tracking, magic constants, and the public checked_cast<T>()
// signature are PRESERVED for ABI compatibility with all consumers.
// ============================================================================

#pragma once

#include "qtx/core/contracts.hpp"
#include "qtx/core/platform.hpp"
#include "qtx/core/types.hpp"

#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace qtx::unsafe::ggml_bridge::safety {

// ============================================================================
// Magic tags for different context types.
// Format: high half = 6-byte legacy ABI tag (bytes 0x4C 0x49 0x54 0x48 0x4F
//         0x53), preserved verbatim for binary compatibility across the
//         C-ABI ggml boundary; low half = type discriminant.
// NOTE: The hex byte sequence is part of the chain-of-trust protocol and
//       MUST NOT be changed. See Epic 0 (rebrand) HARD RULE.
// ============================================================================

inline constexpr core::u64 kBufferContextMagic = 0x4C495448'4F530001ull;  // legacy ABI tag + 0x01
inline constexpr core::u64 kArenaAccessMagic   = 0x4C495448'4F530002ull;  // legacy ABI tag + 0x02
inline constexpr core::u64 kBuftStorageMagic   = 0x4C495448'4F530003ull;  // legacy ABI tag + 0x03

inline constexpr core::u64 kFreedMagic         = 0xDEADBEEF'DEADBEEFull;
inline constexpr core::u64 kUninitMagic        = 0xCAFEBABE'CAFEBABEull;

// ============================================================================
// boundaryViolation: routes through the policy handler installed via
// core::installBoundaryFailHandler. Default policy is std::abort(); a
// multi-tenant host overrides this with longjmp/exception unwinding.
// ============================================================================

[[noreturn]] inline void boundaryViolation(
    const char* file,
    int line,
    const char* func,
    const char* what,
    core::u64 actual_magic = 0u,
    core::u64 expected_magic = 0u) noexcept
{
    std::fprintf(stderr,
        "\n=============================================================\n"
        "  QTX BOUNDARY VIOLATION (fail-fast termination)\n"
        "=============================================================\n"
        "  Location  : %s:%d (%s)\n"
        "  Violation : %s\n"
        "  Magic got : 0x%016" PRIx64 "\n"
        "  Magic want: 0x%016" PRIx64 "\n"
        "=============================================================\n",
        file, line, func, what,
        static_cast<std::uint64_t>(actual_magic),
        static_cast<std::uint64_t>(expected_magic));
    std::fflush(stderr);
    core::invokeFailHandler(file, line, func, what,
                            actual_magic, expected_magic);
}

#define QTX_ASSERT_BOUNDARY(cond, msg)                                  \
    do {                                                                   \
        if (!(cond)) [[unlikely]] {                                        \
            ::qtx::unsafe::ggml_bridge::safety::boundaryViolation(      \
                __FILE__, __LINE__, __func__, (msg));                      \
        }                                                                  \
    } while (0)

// ============================================================================
// === TaggedHeader (24 bytes) — install_token widened to u64 (EC137) ===
//
// Layout:
//   [0..7]   magic         : type discriminant
//   [8..11]  generation    : bumped on markFreed (use-after-free defence)
//   [12..15] _pad0         : explicit padding for u64 alignment of next field
//   [16..23] install_token : full 64-bit per-bridge install token; defeats
//                            the 2^31 wrap-around the original u32 had
//                            (high-churn embedding services that create and
//                            free ggml buffers every ms would wrap u32 in
//                            ~24 days; u64 wraps in ~580 million years).
//
// Size = 24 bytes, alignment = 8 bytes. ABI break vs P1, but TaggedHeader
// is an internal-only detail header (no out-of-tree consumers).
// ============================================================================

struct alignas(8) TaggedHeader {
    core::u64 magic;
    core::u32 generation;
    core::u32 _pad0;          // explicit padding for u64 alignment below
    core::u64 install_token;

    constexpr TaggedHeader() noexcept
        : magic(kUninitMagic), generation(0u), _pad0(0u), install_token(0u) {}

    constexpr explicit TaggedHeader(core::u64 m,
                                    core::u64 tok = 0u) noexcept
        : magic(m), generation(0u), _pad0(0u), install_token(tok) {}
};

static_assert(sizeof(TaggedHeader) == 24);
static_assert(alignof(TaggedHeader) == 8);
static_assert(std::is_trivially_destructible_v<TaggedHeader>);

// ============================================================================
// === Install token (per-bridge anti-forgery, EC34) ===
// ============================================================================

namespace detail {

inline std::atomic<core::u64>& installTokenCounter() noexcept {
    static std::atomic<core::u64> c{0xA110C8'00000001ull};
    return c;
}

}  // namespace detail

[[nodiscard]] inline core::u64 nextInstallToken() noexcept {
    const core::u64 v = detail::installTokenCounter().fetch_add(
        1u, std::memory_order_acq_rel);
    // EC137: keep the top-bit set so install_token=0 (the initial state
    // of a recycled context) is NEVER mistaken for a valid token.
    return v | 0x8000000000000000ull;
}

// ============================================================================
// === Alignment / range plausibility check (EC23) ===
// ============================================================================

[[nodiscard]] inline bool pointerLooksValid(const void* p) noexcept {
    if (p == nullptr) return false;
    const auto v = reinterpret_cast<core::uptr>(p);
    if ((v & (alignof(TaggedHeader) - 1u)) != 0u) return false;
    // Reject page-0 addresses (e.g. user passing 0x10) before we
    // dereference them. 4096 is the smallest page size we ever assume.
    return v >= 4096u;
}

// ============================================================================
// checked_cast<T>(void* p, magic, [install_token])
// ============================================================================

QTX_MSVC_WARNINGS_PUSH()
QTX_MSVC_WARNINGS_DISABLE(4200 4324)

template <typename T>
[[nodiscard]] inline T* checked_cast(void* p, core::u64 expected_magic,
                                     const char* context = "checked_cast",
                                     core::u64 expected_token = 0u) noexcept
{
    static_assert(std::is_standard_layout_v<T>,
                  "checked_cast requires standard-layout type");
    static_assert(offsetof(T, header) == 0,
                  "Type passed to checked_cast must have TaggedHeader as first member named 'header'");

    if (!pointerLooksValid(p)) [[unlikely]] {
        boundaryViolation(__FILE__, __LINE__, context,
            "null/misaligned/bogus pointer crossed boundary",
            reinterpret_cast<core::u64>(p), 0u);
    }

    auto* hdr = static_cast<TaggedHeader*>(p);
    if (hdr->magic != expected_magic) [[unlikely]] {
        boundaryViolation(__FILE__, __LINE__, context,
            "magic tag mismatch (corrupted context or wrong type)",
            hdr->magic, expected_magic);
    }
    if (expected_token != 0u && hdr->install_token != expected_token) [[unlikely]] {
        boundaryViolation(__FILE__, __LINE__, context,
            "install token mismatch (forged context or stale handle)",
            hdr->install_token,
            expected_token);
    }
    return static_cast<T*>(p);
}

template <typename T>
[[nodiscard]] inline const T* checked_cast(const void* p, core::u64 expected_magic,
                                           const char* context = "checked_cast",
                                           core::u64 expected_token = 0u) noexcept
{
    static_assert(std::is_standard_layout_v<T>);
    static_assert(offsetof(T, header) == 0,
                  "Type passed to checked_cast must have TaggedHeader as first member named 'header'");

    if (!pointerLooksValid(p)) [[unlikely]] {
        boundaryViolation(__FILE__, __LINE__, context,
            "null/misaligned/bogus pointer crossed boundary",
            reinterpret_cast<core::u64>(p), 0u);
    }

    const auto* hdr = static_cast<const TaggedHeader*>(p);
    if (hdr->magic != expected_magic) [[unlikely]] {
        boundaryViolation(__FILE__, __LINE__, context,
            "magic tag mismatch (corrupted context or wrong type)",
            hdr->magic, expected_magic);
    }
    if (expected_token != 0u && hdr->install_token != expected_token) [[unlikely]] {
        boundaryViolation(__FILE__, __LINE__, context,
            "install token mismatch (forged context or stale handle)",
            hdr->install_token,
            expected_token);
    }
    return static_cast<const T*>(p);
}

QTX_MSVC_WARNINGS_POP()

// ============================================================================
// markFreed: write kFreedMagic, bump generation, clear install_token.
// After return, any subsequent checked_cast against the same pointer
// is rejected via magic mismatch.  (EC39)
// ============================================================================

inline void markFreed(TaggedHeader& hdr) noexcept {
    hdr.generation += 1u;
    hdr.install_token = 0u;
    hdr.magic = kFreedMagic;
}

}  // namespace qtx::unsafe::ggml_bridge::safety
