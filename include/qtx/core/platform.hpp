// ============================================================================
// @file        platform.hpp
// @brief       Compile-time platform and compiler detection.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// HA-LAYER: This file is part of the formally-safe core.
// No runtime logic, pure preprocessor definitions and constexpr constants.
//
// Detected properties:
//   - Operating system (QTX_OS_WINDOWS, QTX_OS_LINUX, QTX_OS_MACOS)
//   - Compiler (QTX_COMPILER_MSVC, QTX_COMPILER_GCC, QTX_COMPILER_CLANG)
//   - Architecture (QTX_ARCH_X86_64, QTX_ARCH_ARM64)
//   - Compiler intrinsics (QTX_FORCE_INLINE, QTX_RESTRICT, etc.)
//
// ============================================================================

#pragma once

// ============================================================================
// === Operating System Detection ===
// ============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define QTX_OS_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
    #define QTX_OS_MACOS 1
#elif defined(__linux__)
    #define QTX_OS_LINUX 1
#else
    #define QTX_OS_UNKNOWN 1
#endif

// ============================================================================
// === Compiler Detection ===
// ============================================================================

// Note: Clang-cl on Windows defines both __clang__ and _MSC_VER.
// We detect Clang first to handle this case correctly.
#if defined(__clang__)
    #define QTX_COMPILER_CLANG 1
    #define QTX_COMPILER_VERSION_MAJOR __clang_major__
    #define QTX_COMPILER_VERSION_MINOR __clang_minor__
#elif defined(__GNUC__)
    #define QTX_COMPILER_GCC 1
    #define QTX_COMPILER_VERSION_MAJOR __GNUC__
    #define QTX_COMPILER_VERSION_MINOR __GNUC_MINOR__
#elif defined(_MSC_VER)
    #define QTX_COMPILER_MSVC 1
    #define QTX_COMPILER_VERSION_MAJOR (_MSC_VER / 100)
    #define QTX_COMPILER_VERSION_MINOR (_MSC_VER % 100)
#else
    #define QTX_COMPILER_UNKNOWN 1
#endif

// ============================================================================
// === Architecture Detection ===
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)
    #define QTX_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define QTX_ARCH_ARM64 1
#elif defined(__riscv) && (__riscv_xlen == 64)
    #define QTX_ARCH_RISCV64 1
#else
    #define QTX_ARCH_UNKNOWN 1
#endif

// ============================================================================
// === Compiler Intrinsic Abstractions ===
// ============================================================================

// Force-inline: stronger than 'inline', compiler-specific.
#if defined(QTX_COMPILER_MSVC)
    #define QTX_FORCE_INLINE __forceinline
#elif defined(QTX_COMPILER_GCC) || defined(QTX_COMPILER_CLANG)
    #define QTX_FORCE_INLINE inline __attribute__((always_inline))
#else
    #define QTX_FORCE_INLINE inline
#endif

// Restrict pointer aliasing hint.
#if defined(QTX_COMPILER_MSVC)
    #define QTX_RESTRICT __restrict
#elif defined(QTX_COMPILER_GCC) || defined(QTX_COMPILER_CLANG)
    #define QTX_RESTRICT __restrict__
#else
    #define QTX_RESTRICT
#endif

// No-inline: prevent inlining (useful for error paths).
#if defined(QTX_COMPILER_MSVC)
    #define QTX_NOINLINE __declspec(noinline)
#elif defined(QTX_COMPILER_GCC) || defined(QTX_COMPILER_CLANG)
    #define QTX_NOINLINE __attribute__((noinline))
#else
    #define QTX_NOINLINE
#endif

// Debug break: platform-specific trap instruction.
#if defined(QTX_COMPILER_MSVC)
    #define QTX_DEBUG_BREAK() __debugbreak()
#elif defined(QTX_COMPILER_GCC) || defined(QTX_COMPILER_CLANG)
    #if defined(QTX_ARCH_X86_64)
        #define QTX_DEBUG_BREAK() __asm__ volatile("int $3")
    #elif defined(QTX_ARCH_ARM64)
        #define QTX_DEBUG_BREAK() __builtin_trap()
    #else
        #define QTX_DEBUG_BREAK() __builtin_trap()
    #endif
#else
    #define QTX_DEBUG_BREAK() ((void)0)
#endif

// ============================================================================
// === MSVC Warning Suppression Helpers ===
// ============================================================================

// Usage:
//   QTX_MSVC_WARNINGS_PUSH()
//   QTX_MSVC_WARNINGS_DISABLE(4200 4324)
//   ... code with suppressed warnings ...
//   QTX_MSVC_WARNINGS_POP()
#if defined(QTX_COMPILER_MSVC)
    #define QTX_MSVC_WARNINGS_PUSH()      __pragma(warning(push))
    #define QTX_MSVC_WARNINGS_POP()       __pragma(warning(pop))
    #define QTX_MSVC_WARNINGS_DISABLE(x)  __pragma(warning(disable: x))
#else
    #define QTX_MSVC_WARNINGS_PUSH()
    #define QTX_MSVC_WARNINGS_POP()
    #define QTX_MSVC_WARNINGS_DISABLE(x)
#endif

// ============================================================================
// === DLL Export/Import (for future shared library builds) ===
// ============================================================================

#if defined(QTX_OS_WINDOWS)
    #define QTX_EXPORT __declspec(dllexport)
    #define QTX_IMPORT __declspec(dllimport)
#elif defined(QTX_COMPILER_GCC) || defined(QTX_COMPILER_CLANG)
    #define QTX_EXPORT __attribute__((visibility("default")))
    #define QTX_IMPORT
#else
    #define QTX_EXPORT
    #define QTX_IMPORT
#endif
