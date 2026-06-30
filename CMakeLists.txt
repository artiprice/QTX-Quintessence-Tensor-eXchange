# ============================================================================
# CMakeLists.txt for qtx
#
# Two build modes:
#   1. Tests (QTX_BUILD_TESTS=ON, default): uses ggml_test_shim.c
#   2. Production (QTX_BUILD_WITH_GGML=ON): links real ggml runtime
#
# Quick start:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#   ./qtx_tests
#   ./oom_survival
#
# Production with real ggml:
#   cmake .. -DQTX_BUILD_WITH_GGML=ON
#   make -j$(nproc)
#   ./examples/real_ggml_demo
#
# Windows (Visual Studio / Ninja):
#   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build .
#   .\qtx_tests.exe
#
# macOS (Xcode / Ninja):
#   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build .
#   ./qtx_tests
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(qtx
    VERSION 0.3.0
    LANGUAGES C CXX
    DESCRIPTION "Software Lithography Framework for AI agent memory orchestration")

# === Configuration options ===
option(QTX_BUILD_TESTS    "Build unit tests"             ON)
option(QTX_BUILD_BENCH    "Build OOM Survival benchmark" ON)
option(QTX_BUILD_WITH_GGML "Link against real ggml (not test shim)" OFF)
option(QTX_STRICT_WARNINGS "Treat warnings as errors"    OFF)

# === Global compiler settings ===
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# ============================================================================
# === Compiler-specific flags ===
#
# GCC/Clang: explicit warning flags, hardening flags.
# MSVC:      /W4 + /permissive- for strictness, no GCC-specific flags.
# AppleClang: same as Clang, minus GCC-only flags.
# ============================================================================

if(MSVC)
    # --- HA flags for MSVC (applied to safe-core, not to bridge) ---
    set(QTX_HA_FLAGS
        /W4                          # High warning level (≈ -Wall -Wextra)
        /permissive-                 # Strict standard conformance
        /Zc:__cplusplus              # Report correct __cplusplus value
        /Zc:preprocessor             # Standard-conformant preprocessor
        /EHsc                        # C++ exception handling model
        /utf-8                       # Source and execution charset = UTF-8
        /volatile:iso                # ISO volatile semantics
        /wd4324                      # Structure padded due to alignas (expected)
    )

    # Relaxed flags for bridge (C-ABI with ggml).
    set(QTX_BRIDGE_FLAGS
        /W3
        /EHsc
        /utf-8
        /wd4324
        /wd4200                      # Zero-sized array in struct (ggml headers)
    )

    if(QTX_STRICT_WARNINGS)
        list(APPEND QTX_HA_FLAGS /WX)
    endif()

else()
    # --- HA flags for GCC/Clang (applied to safe-core, not to bridge) ---
    set(QTX_HA_FLAGS
        -Wall -Wextra -Wpedantic
        -Wconversion -Wshadow
        -Wnon-virtual-dtor -Wold-style-cast
        -Woverloaded-virtual -Wnull-dereference
        -Wcast-align -Wdouble-promotion
        -fno-strict-aliasing
        -fno-delete-null-pointer-checks
    )

    # -ftrivial-auto-var-init=zero: GCC 12+ and Clang 14+.
    # Check support before adding.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-ftrivial-auto-var-init=zero
        QTX_HAS_TRIVIAL_AUTO_VAR_INIT)
    if(QTX_HAS_TRIVIAL_AUTO_VAR_INIT)
        list(APPEND QTX_HA_FLAGS -ftrivial-auto-var-init=zero)
    endif()

    if(QTX_STRICT_WARNINGS)
        list(APPEND QTX_HA_FLAGS -Werror)
    endif()

    # Relaxed flags for bridge (C-ABI with ggml).
    set(QTX_BRIDGE_FLAGS
        -Wall -Wextra
        -fno-strict-aliasing
        -fno-delete-null-pointer-checks
    )
endif()

# ============================================================================
# === qtx_core: header-only HA-clean part ===
# ============================================================================
add_library(qtx_core INTERFACE)
target_include_directories(qtx_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qtx_core INTERFACE cxx_std_23)

# ============================================================================
# === qtx_codebooks: fixed I-Quant lookup tables (ggml MIT codebooks) ===
# ============================================================================
add_library(qtx_codebooks STATIC src/iq_codebooks.cpp)
target_include_directories(qtx_codebooks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qtx_codebooks PUBLIC cxx_std_23)
# Make qtx_core users transitively pick up the codebooks.
target_link_libraries(qtx_core INTERFACE qtx_codebooks)

# ============================================================================
# === ggml dependency: either real from third_party/ggml_src, or headers-only ===
# ============================================================================
set(GGML_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml_src")
set(GGML_HDR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")

if(QTX_BUILD_WITH_GGML)
    if(NOT EXISTS "${GGML_SRC_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "QTX_BUILD_WITH_GGML=ON needs the ggml sources, which live in a "
            "git submodule at ${GGML_SRC_DIR}.\n"
            "Initialize it with:\n"
            "    git submodule update --init --depth 1 third_party/ggml_src\n"
            "(or clone the repo with --recursive). Alternatively build with "
            "QTX_BUILD_WITH_GGML=OFF — the test suite does not require it.")
    endif()

    # Disable everything we don't need for production build.
    set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA           OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN         OFF CACHE BOOL "" FORCE)
    set(GGML_METAL          OFF CACHE BOOL "" FORCE)
    set(GGML_OPENCL         OFF CACHE BOOL "" FORCE)
    set(GGML_KOMPUTE        OFF CACHE BOOL "" FORCE)
    set(GGML_HIP            OFF CACHE BOOL "" FORCE)
    set(GGML_SYCL           OFF CACHE BOOL "" FORCE)
    set(GGML_RPC            OFF CACHE BOOL "" FORCE)
    set(GGML_MUSA           OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)

    add_subdirectory(${GGML_SRC_DIR} ${CMAKE_BINARY_DIR}/_ggml_src EXCLUDE_FROM_ALL)

    set(QTX_GGML_INCLUDES
        ${GGML_SRC_DIR}/include
        ${GGML_SRC_DIR}/src)
    set(QTX_GGML_LIBS ggml ggml-cpu ggml-base)
    message(STATUS "qtx: linking against REAL ggml runtime (CPU-only)")
else()
    # Headers-only mode: just ggml public headers, no runtime linkage.
    # ggml_backend_buffer_init etc. are provided via shim for tests.
    set(QTX_GGML_INCLUDES
        ${GGML_HDR_DIR}/include
        ${GGML_HDR_DIR}/src)
    set(QTX_GGML_LIBS "")  # shim will be added in tests
    message(STATUS "qtx: using ggml HEADERS-ONLY mode (test shim)")
endif()

# ============================================================================
# === qtx_bridge: unsafe FFI layer ===
# ============================================================================
add_library(qtx_bridge STATIC
    src/unsafe/ggml_bridge/bridge.cpp)

target_link_libraries(qtx_bridge PUBLIC qtx_core)
target_include_directories(qtx_bridge
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/unsafe/ggml_bridge
        ${QTX_GGML_INCLUDES})

target_compile_options(qtx_bridge PRIVATE ${QTX_BRIDGE_FLAGS})

if(QTX_BUILD_WITH_GGML)
    target_link_libraries(qtx_bridge PUBLIC ${QTX_GGML_LIBS})
endif()

# ============================================================================
# === Tests ===
# ============================================================================
if(QTX_BUILD_TESTS)
    set(TEST_SOURCES
        tests/test_main.cpp
        tests/test_gen_id.cpp
        tests/test_axiom_selector.cpp
        tests/test_fractal_arena.cpp
        tests/test_spsc_ring_buffer.cpp
        tests/test_integration.cpp
        tests/test_quantizer.cpp
        tests/test_tiered_bridge.cpp
        tests/test_cold_codec.cpp)

    # Bridge tests are built with relaxed warnings (ggml headers).
    set(BRIDGE_TEST_SOURCES
        tests/test_ggml_bridge.cpp
        tests/test_boundary_safety.cpp)

    # If building without real ggml - shim is needed.
    if(NOT QTX_BUILD_WITH_GGML)
        set(SHIM_SOURCES tests/ggml_test_shim.c)

        # MSVC: compile shim as C++ to avoid C11/C17 compatibility issues.
        if(MSVC)
            set_source_files_properties(${SHIM_SOURCES}
                PROPERTIES LANGUAGE CXX)
        endif()

        list(APPEND BRIDGE_TEST_SOURCES ${SHIM_SOURCES})
    endif()

    add_executable(qtx_tests
        ${TEST_SOURCES} ${BRIDGE_TEST_SOURCES})

    target_link_libraries(qtx_tests PRIVATE qtx_core qtx_bridge Threads::Threads)
    target_include_directories(qtx_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/unsafe/ggml_bridge
        ${QTX_GGML_INCLUDES})

    # HA flags to main tests, relaxed to bridge tests.
    set_source_files_properties(${TEST_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${QTX_HA_FLAGS}")
    set_source_files_properties(${BRIDGE_TEST_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${QTX_BRIDGE_FLAGS}")

    add_custom_target(check
        COMMAND qtx_tests
        DEPENDS qtx_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL)
endif()

# ============================================================================
# === Benchmark ===
# ============================================================================
if(QTX_BUILD_BENCH)
    add_executable(oom_survival benchmarks/oom_survival_test.cpp)
    target_link_libraries(oom_survival PRIVATE qtx_core)
    if(MSVC)
        target_compile_options(oom_survival PRIVATE /O2)
    else()
        target_compile_options(oom_survival PRIVATE -O3)
    endif()

    # Microbenchmark suite for hot-path operations
    add_executable(perf_microbench benchmarks/perf_microbench.cpp)
    target_link_libraries(perf_microbench PRIVATE qtx_core Threads::Threads)
    if(MSVC)
        target_compile_options(perf_microbench PRIVATE /O2)
    else()
        target_compile_options(perf_microbench PRIVATE -O2 -march=native)
    endif()
endif()

# ============================================================================
# === Production example (only with QTX_BUILD_WITH_GGML) ===
# ============================================================================
if(QTX_BUILD_WITH_GGML)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/real_ggml_demo.cpp)
        add_executable(real_ggml_demo examples/real_ggml_demo.cpp)
        target_link_libraries(real_ggml_demo PRIVATE
            qtx_core qtx_bridge ${QTX_GGML_LIBS} Threads::Threads)
        target_include_directories(real_ggml_demo PRIVATE
            ${QTX_GGML_INCLUDES})
        target_compile_options(real_ggml_demo PRIVATE ${QTX_BRIDGE_FLAGS})
    endif()
endif()

# ============================================================================
# === Configuration summary ===
# ============================================================================
message(STATUS "")
message(STATUS "=== qtx configuration ===")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "  System:            ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Tests:             ${QTX_BUILD_TESTS}")
message(STATUS "  Benchmark:         ${QTX_BUILD_BENCH}")
message(STATUS "  Real ggml:         ${QTX_BUILD_WITH_GGML}")
message(STATUS "  Strict warnings:   ${QTX_STRICT_WARNINGS}")
message(STATUS "")
