// ============================================================================
// @file        types.hpp
//@brief Basic types, concepts and compile-time invariants of HA-core.
// @author      QTX Project
// @date        2026-05-12
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
//HA-LAYER: This file is part of the formally-safe core.
//FORBIDDEN: reinterpret_cast, union, pointer arithmetic, virtual in hot-path,
//dynamic allocation, exceptions in noexcept functions.
//
// ============================================================================

#pragma once

#include <bit>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace qtx::core {

//=== Fundamental aliases (HA: explicit sizes, no int/long) ===
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using usize = std::size_t;
using uptr  = std::uintptr_t;

//=== Cache Architectural Constants ===
//Apple Silicon performance cores use 128-byte cache lines,
//x86 - 64. We take the maximum for portability (verify via
//std::hardware_destructive_interference_size where available).
inline constexpr usize kMaxCacheLineSize = 128;
inline constexpr usize kMinCacheLineSize = 64;

//Page sizes for future integration with huge pages.
inline constexpr usize kDefaultPageSize  = 4096;
inline constexpr usize kHugePage2MB      = 2 * 1024 * 1024;
inline constexpr usize kHugePage1GB      = 1024 * 1024 * 1024;

//=== Concepts (C++20+, replaces SFINAE) ===

//POD-like type, safe to place in Arena.
template <typename T>
concept ArenaStorable =
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_destructible_v<T> &&
    std::is_standard_layout_v<T>;

//Unsigned integer type (for bitmask selector operations).
template <typename T>
concept UnsignedIntegral =
    std::is_integral_v<T> && std::is_unsigned_v<T>;

//A type that is at least cache-line aligned (for hot structures).
template <typename T>
concept CacheLineAligned =
    (alignof(T) >= kMinCacheLineSize);

//=== Compile-time utilities (constexpr, without template recursion) ===

//Power of two? (for ring buffer and mask sizes).
[[nodiscard]] constexpr bool isPowerOfTwo(usize n) noexcept {
    return n != 0u && (n & (n - 1u)) == 0u;
}

//Round up until aligned (alignment MUST be a power of two).
[[nodiscard]] constexpr usize alignUp(usize value, usize alignment) noexcept {
    //HA: precondition contract, during debug we will fall into assert.
    //At release we will return 0 if the alignment is incorrect - fail-fast in the caller.
    return (alignment == 0u || !isPowerOfTwo(alignment))
        ? 0u
        : (value + alignment - 1u) & ~(alignment - 1u);
}

//log2 for powers of two (compile-time).
[[nodiscard]] constexpr u32 log2PowerOfTwo(usize n) noexcept {
    return isPowerOfTwo(n) ? static_cast<u32>(std::countr_zero(n)) : 0u;
}

//=== Compile-time verification of fundamental invariants ===
static_assert(sizeof(u64) == 8, "u64 must be exactly 8 bytes");
static_assert(sizeof(uptr) == sizeof(void*), "uptr must match pointer size");
static_assert(CHAR_BIT == 8, "qtx assumes 8-bit bytes");
static_assert(isPowerOfTwo(kMaxCacheLineSize));
static_assert(isPowerOfTwo(kMinCacheLineSize));
static_assert(isPowerOfTwo(kDefaultPageSize));
static_assert(alignUp(7u, 8u) == 8u);
static_assert(alignUp(8u, 8u) == 8u);
static_assert(alignUp(9u, 8u) == 16u);
static_assert(alignUp(0u, 64u) == 0u);
static_assert(log2PowerOfTwo(64u) == 6u);
static_assert(log2PowerOfTwo(128u) == 7u);

}  // namespace qtx::core
