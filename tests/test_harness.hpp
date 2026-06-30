// ============================================================================
// @file        test_harness.hpp
// @brief       Minimalistic test framework with no external dependencies.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================
//
// HA-LAYER: test infrastructure, not for production. Uses stdout
// and std::exit for reporting. fail-fast on first error with --strict.
//
// ============================================================================

#pragma once

#include "qtx/core/platform.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <string_view>
#include <vector>

#if defined(QTX_OS_WINDOWS)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif


namespace qtx::testing {

struct TestCase {
    std::string_view suite;
    std::string_view name;
    std::function<void()> fn;
};

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
};

//Global Test Registry.
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline TestStats& stats() {
    static TestStats s;
    return s;
}

struct TestRegistrar {
    TestRegistrar(std::string_view suite,
                  std::string_view name,
                  std::function<void()> fn) {
        registry().push_back({suite, name, std::move(fn)});
    }
};

//Exception to immediately exit the test on FAIL.
struct AssertionFailure : std::exception {
    const char* what() const noexcept override { return "assertion failed"; }
};

inline void reportFail(const char* file, int line, const char* expr) {
    std::fprintf(stderr, "  \x1b[31mFAIL\x1b[0m at %s:%d: %s\n",
                 file, line, expr);
    throw AssertionFailure{};
}

inline int run(int argc, char** argv) {
    // Enable ANSI escape codes on Windows 10+ terminals.
#if defined(QTX_OS_WINDOWS)
    {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            if (GetConsoleMode(hOut, &dwMode)) {
                dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(hOut, dwMode);
            }
        }
        HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
        if (hErr != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            if (GetConsoleMode(hErr, &dwMode)) {
                dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(hErr, dwMode);
            }
        }
    }
#endif

    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0 ||
            std::strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
    }
    auto& reg = registry();
    std::printf("Running %zu tests...\n", reg.size());

    std::string_view last_suite;
    for (auto& tc : reg) {
        if (tc.suite != last_suite) {
            std::printf("\n[%.*s]\n",
                        static_cast<int>(tc.suite.size()), tc.suite.data());
            last_suite = tc.suite;
        }
        if (verbose) {
            std::printf("  - %.*s ... ",
                        static_cast<int>(tc.name.size()), tc.name.data());
            std::fflush(stdout);
        }
        ++stats().total;
        try {
            tc.fn();
            ++stats().passed;
            if (verbose) std::printf("\x1b[32mok\x1b[0m\n");
        } catch (const AssertionFailure&) {
            ++stats().failed;
            std::printf("  \x1b[31m✗ %.*s\x1b[0m\n",
                        static_cast<int>(tc.name.size()), tc.name.data());
        } catch (const std::exception& e) {
            ++stats().failed;
            std::printf("  \x1b[31m✗ %.*s — exception: %s\x1b[0m\n",
                        static_cast<int>(tc.name.size()), tc.name.data(),
                        e.what());
        }
    }
    std::printf("\n========================================\n");
    std::printf("Total:  %d\n", stats().total);
    std::printf("Passed: \x1b[32m%d\x1b[0m\n", stats().passed);
    std::printf("Failed: \x1b[%sm%d\x1b[0m\n",
                stats().failed == 0 ? "32" : "31", stats().failed);
    return stats().failed == 0 ? 0 : 1;
}

}  // namespace qtx::testing

//=== Registration macros (minimal, no pain) ===
#define QTX_TEST(suite_name, test_name)                              \
    static void qtx_test_##suite_name##_##test_name();               \
    static ::qtx::testing::TestRegistrar                             \
        qtx_reg_##suite_name##_##test_name{                          \
            #suite_name, #test_name,                                    \
            qtx_test_##suite_name##_##test_name};                    \
    static void qtx_test_##suite_name##_##test_name()

#define QTX_EXPECT(cond)                                             \
    do {                                                                \
        if (!(cond)) {                                                  \
            ::qtx::testing::reportFail(__FILE__, __LINE__, #cond);   \
        }                                                               \
    } while (0)

#define QTX_EXPECT_EQ(a, b)                                          \
    do {                                                                \
        if (!((a) == (b))) {                                            \
            ::qtx::testing::reportFail(                              \
                __FILE__, __LINE__, #a " == " #b);                      \
        }                                                               \
    } while (0)

#define QTX_EXPECT_NE(a, b)                                          \
    do {                                                                \
        if (!((a) != (b))) {                                            \
            ::qtx::testing::reportFail(                              \
                __FILE__, __LINE__, #a " != " #b);                      \
        }                                                               \
    } while (0)
