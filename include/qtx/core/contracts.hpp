// ============================================================================
// @file        contracts.hpp
// @brief       HA contract primitives: checked arithmetic, fail-fast policy.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// HA-LAYER: This file is part of the formally-safe core.
//
// PURPOSE: Centralises three concerns that the EC-100 catalogue surfaced:
//
//   1. Checked unsigned arithmetic (addOverflow / mulOverflow / addBounded)
//      — closes EC25 (offset + size wrap-around in C-ABI), EC85
//        (alloc-size arithmetic in tiered), EC50 (FLT_MAX scale).
//
//   2. Compile-time policy for "fail-fast" termination at the FFI boundary
//      (QTX_FAIL_POLICY_ABORT vs QTX_FAIL_POLICY_TRAP vs
//      QTX_FAIL_POLICY_CALLBACK) — closes EC40 (DoS via std::abort in
//      multi-tenant servers), EC73 (__builtin_trap in production).
//
//   3. A single sink for boundary-violation diagnostics that respects the
//      chosen policy — the bridge.cpp file installs the callback that
//      hosting code (e.g. an HTTP API server) wants used instead of abort.
//
// All facilities are constexpr / inline / noexcept and add zero overhead
// in the happy path: predicated on the [[unlikely]] error branch only.
// ============================================================================

#pragma once

#include "platform.hpp"
#include "types.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace qtx::core {

// ============================================================================
// === Checked unsigned arithmetic (constexpr, noexcept) ===
//
// Used to detect integer overflow BEFORE forming a pointer at an FFI
// boundary (EC25). Return value semantics mimic C++26 std::add_sat /
// std::ckd_add: bool result tells the caller whether the operation was
// safe; the sum/product is stored in the out parameter even on overflow
// (it is wrapped and therefore safe to inspect, but unsafe to use as a
// length).
// ============================================================================

[[nodiscard]] constexpr bool addOverflow(usize a, usize b, usize& out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, &out);
#else
    out = a + b;
    return out < a;  // wrap-around detection for unsigned
#endif
}

[[nodiscard]] constexpr bool mulOverflow(usize a, usize b, usize& out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, &out);
#else
    if (a == 0u || b == 0u) { out = 0u; return false; }
    out = a * b;
    return (out / a) != b;
#endif
}

/// addBounded: returns false if either the sum overflows OR exceeds limit.
/// Closes EC25 by collapsing two checks into one branchless conjunction.
[[nodiscard]] constexpr bool addBounded(usize a, usize b,
                                        usize limit,
                                        usize& out) noexcept {
    const bool overflow = addOverflow(a, b, out);
    return !overflow && (out <= limit);
}

// ============================================================================
// === Bytewise pointer-range bounds check (HA: NO pointer arithmetic) ===
//
// rangeWithin(buf_base, buf_size, offset, len) -> true iff
//   [base + offset, base + offset + len)  ⊆  [base, base + buf_size).
//
// Implemented purely on usize arithmetic with overflow protection — no
// raw pointer subtraction (forbidden by HA TYPE_SAFETY_PROFILE outside
// the constrained-unsafe boundary file).
// ============================================================================

[[nodiscard]] constexpr bool rangeWithin(usize buffer_size,
                                         usize offset,
                                         usize length) noexcept {
    if (offset > buffer_size) return false;
    usize end = 0u;
    if (addOverflow(offset, length, end)) return false;
    return end <= buffer_size;
}

// ============================================================================
// === Fail policy: abort | trap | callback ===
//
// EC40: a hosted Qtx serving 100 sessions cannot call std::abort() on
// a single corrupted ggml context — that would be a denial-of-service. We
// expose a registration point so hosts may swap abort for an exception,
// a longjmp out of the request, or a per-thread shutdown.
//
// EC73: __builtin_trap in release builds causes core dumps in production.
// QTX_FAIL_POLICY_TRAP is gated behind the QTX_DEBUG macro, so it
// is *never* the default in optimised builds.
// ============================================================================

/// Signature of the host-supplied boundary-violation handler.
/// MUST NOT return. If it returns, Qtx falls back to std::abort().
using BoundaryFailHandler = void (*)(const char* file, int line,
                                     const char* func, const char* what,
                                     u64 actual_magic,
                                     u64 expected_magic) noexcept;

namespace detail {

inline std::atomic<BoundaryFailHandler>& failHandler() noexcept {
    // Meyer's singleton: thread-safe init in C++11+, no destructor
    // ordering issues (process-lifetime).
    static std::atomic<BoundaryFailHandler> h{nullptr};
    return h;
}

}  // namespace detail

/// Install a custom handler. Returns the previous value (nullptr if none).
/// Pass nullptr to revert to default (std::abort()).
///
/// EC185 — Contract: last-writer-wins. The function is atomic via
/// std::atomic::exchange, so the install operation itself is data-race
/// free; but if two plugins call this from different threads with
/// different handler functions, the SECOND call wins unconditionally
/// and the previous handler is returned to the caller for it to
/// restore later. Callers that need stacked handlers must implement
/// their own chain by saving the returned previous handler and
/// invoking it from their own handler — exactly the pattern used by
/// std::set_terminate. The library does NOT manage a registry; that
/// is a deliberate scope limit (a registry would require dynamic
/// allocation, which the HA layer forbids).
inline BoundaryFailHandler installBoundaryFailHandler(
    BoundaryFailHandler h) noexcept
{
    return detail::failHandler().exchange(h, std::memory_order_acq_rel);
}

/// Invoke the registered handler if any, then std::abort() as a hard floor.
/// Marked [[noreturn]] because the function never returns under any policy.
[[noreturn]] inline void invokeFailHandler(
    const char* file, int line, const char* func, const char* what,
    u64 actual_magic = 0u, u64 expected_magic = 0u) noexcept
{
    BoundaryFailHandler h = detail::failHandler().load(std::memory_order_acquire);
    if (h != nullptr) {
        h(file, line, func, what, actual_magic, expected_magic);
        // If the host handler returns (it must not), fall through to abort.
    }
    std::fflush(stderr);
    std::abort();
}

}  // namespace qtx::core

//=== Compile-time invariants ===
static_assert(sizeof(qtx::core::usize) >= sizeof(std::size_t));

namespace qtx::core::detail {
// Round-trip checks for checked arithmetic (compile-time).
constexpr bool addOverflowCheck() noexcept {
    usize out = 0u;
    if (addOverflow(usize{5u}, usize{7u}, out) || out != 12u) return false;
    out = 0u;
    if (!addOverflow(std::numeric_limits<usize>::max(), usize{1u}, out)) return false;
    return true;
}
static_assert(addOverflowCheck());

constexpr bool mulOverflowCheck() noexcept {
    usize out = 0u;
    if (mulOverflow(usize{6u}, usize{7u}, out) || out != 42u) return false;
    if (!mulOverflow(std::numeric_limits<usize>::max(), usize{2u}, out)) return false;
    return true;
}
static_assert(mulOverflowCheck());

constexpr bool rangeWithinCheck() noexcept {
    if (!rangeWithin(100u, 10u, 80u)) return false;       // ok
    if ( rangeWithin(100u, 10u, 91u)) return false;       // overrun
    if ( rangeWithin(100u, std::numeric_limits<usize>::max(), 1u))
        return false;                                     // offset overflow
    if ( rangeWithin(100u, 50u, std::numeric_limits<usize>::max()))
        return false;                                     // length overflow
    return true;
}
static_assert(rangeWithinCheck());
}  // namespace qtx::core::detail
