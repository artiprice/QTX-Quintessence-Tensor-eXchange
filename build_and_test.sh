#!/usr/bin/env bash
# ============================================================================
# @file        build_and_test.sh
# @brief       Build and run qtx unit tests (P1 + SPSC + Bridge + Quant).
# @author      QTX Project
# ============================================================================
#
# Modes:
#   ./build_and_test.sh           — release build + tests
#   ./build_and_test.sh asan      — AddressSanitizer + UBSan
#   ./build_and_test.sh tsan      — ThreadSanitizer
#   ./build_and_test.sh strict    — release + -Werror + max warnings
#   ./build_and_test.sh bench     — OOM Survival benchmark (separate binary)
#   ./build_and_test.sh all       — strict + asan + tsan + bench
#
# Build architecture:
#   - HA-core: -Werror, all warnings, C++23
#   - Bridge: relaxed warnings (interacts with ggml C-ABI)
#   - GGML shim: C11 (thin runtime shim for tests)
#   - Benchmark: separate executable file
#
# Platform support:
#   - Linux (GCC 12+ / Clang 16+)
#   - macOS (AppleClang 15+ / Homebrew Clang 16+)
#
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${1:-release}"

# === Platform and compiler detection ===
PLATFORM="$(uname -s)"

# Detect C++ and C compilers.
if [[ "$PLATFORM" == "Darwin" ]]; then
    # macOS: prefer clang++ (system default on macOS).
    CXX="${CXX:-clang++}"
    CC="${CC:-clang}"
else
    # Linux: prefer g++, fall back to clang++.
    CXX="${CXX:-g++}"
    CC="${CC:-gcc}"
fi

# === HA flags (mandatory in any mode for HA-core) ===
HA_FLAGS=(
    -std=c++23
    -Wall -Wextra -Wpedantic
    -Wconversion -Wshadow
    -Wnon-virtual-dtor -Wold-style-cast
    -Woverloaded-virtual -Wnull-dereference
    -Wcast-align -Wdouble-promotion
    -fno-strict-aliasing
    -fno-delete-null-pointer-checks
)

# -ftrivial-auto-var-init=zero: supported by GCC 12+ and Clang 14+.
if $CXX -ftrivial-auto-var-init=zero -x c++ /dev/null -c -o /dev/null 2>/dev/null; then
    HA_FLAGS+=(-ftrivial-auto-var-init=zero)
fi

# === Bridge flags (relaxed for interaction with ggml C-ABI) ===
# All unsafe operations are localized in bridge.cpp; there we INTENTIONALLY use
# C-style casts (see UNSAFE FFI BOUNDARY banner in bridge.cpp).
BRIDGE_FLAGS=(
    -std=c++23
    -Wall -Wextra
    -fno-strict-aliasing
    -fno-delete-null-pointer-checks
)

# === Shim flags (C11, minimal implementation of ggml runtime for tests) ===
SHIM_FLAGS=(
    -std=c11
    -Wall -Wextra
)

# === Include paths ===
INCLUDES=(
    -I include
    -I src/unsafe/ggml_bridge
    -I third_party/ggml/include
    -I third_party/ggml/src
)

# === Sources ===
CXX_HA_SOURCES=(
    tests/test_main.cpp
    tests/test_gen_id.cpp
    tests/test_axiom_selector.cpp
    tests/test_fractal_arena.cpp
    tests/test_spsc_ring_buffer.cpp
    tests/test_integration.cpp
    tests/test_quantizer.cpp
    tests/test_tiered_bridge.cpp
    tests/test_cold_codec.cpp
)

# ggml-bridge tests include ggml.h, which has code shadowing
# constructor warning - this is a public ggml header that we cannot
# fix. We build these tests with relaxed warnings.
CXX_BRIDGE_TEST_SOURCES=(
    tests/test_ggml_bridge.cpp
    tests/test_boundary_safety.cpp
)

CXX_BRIDGE_SOURCES=(
    src/unsafe/ggml_bridge/bridge.cpp
)

C_SHIM_SOURCES=(
    tests/ggml_test_shim.c
)

OUT_DIR="build/${MODE}"
mkdir -p "$OUT_DIR"
BIN="$OUT_DIR/qtx_tests"

# Mode-specific flags
case "$MODE" in
    release)
        OPT_FLAGS=(-O2 -DNDEBUG)
        ;;
    strict)
        OPT_FLAGS=(-O2 -DNDEBUG -Werror)
        ;;
    debug)
        OPT_FLAGS=(-O0 -g3)
        ;;
    asan)
        OPT_FLAGS=(-O1 -g3 -fsanitize=address,undefined -fno-omit-frame-pointer)
        ;;
    tsan)
        OPT_FLAGS=(-O1 -g3 -fsanitize=thread -fno-omit-frame-pointer)
        ;;
    bench)
        # Benchmark: maximum optimization, no debug info.
        OPT_FLAGS=(-O3 -DNDEBUG)
        # -march=native: works on both Linux and macOS.
        if [[ "$PLATFORM" == "Darwin" ]]; then
            # Apple Silicon: -mcpu=apple-m1 or -mcpu=native.
            OPT_FLAGS+=(-mcpu=native)
        else
            OPT_FLAGS+=(-march=native)
        fi
        BENCH_BIN="$OUT_DIR/oom_survival"
        mkdir -p "$OUT_DIR"
        echo "=== Building OOM Survival Benchmark ==="
        $CXX -std=c++23 -Wall -Wextra "${OPT_FLAGS[@]}" \
            -I include \
            benchmarks/oom_survival_test.cpp \
            -o "$BENCH_BIN"
        echo "✓ Build OK: $BENCH_BIN"
        echo
        "$BENCH_BIN"
        exit $?
        ;;
    micro-demo)
        # Phase 2 Block C.2 micro-tenant packing demo.
        OPT_FLAGS=(-O3 -DNDEBUG)
        if [[ "$PLATFORM" == "Darwin" ]]; then
            OPT_FLAGS+=(-mcpu=native)
        else
            OPT_FLAGS+=(-march=native)
        fi
        DEMO_BIN="$OUT_DIR/micro_tenant_demo"
        mkdir -p "$OUT_DIR"
        echo "=== Building Micro-Tenant Packing Demo ==="
        $CXX -std=c++23 -Wall -Wextra "${OPT_FLAGS[@]}" \
            -I include \
            benchmarks/micro_tenant_demo.cpp \
            -lpthread \
            -o "$DEMO_BIN"
        echo "✓ Build OK: $DEMO_BIN"
        echo
        "$DEMO_BIN"
        exit $?
        ;;
    all)
        # Run all modes sequentially.
        echo "=== Running all checks ==="
        for m in strict asan tsan bench; do
            echo
            echo "##### MODE: $m #####"
            "$0" "$m"
            if [[ $? -ne 0 ]]; then
                echo "FAILED in mode $m"
                exit 1
            fi
        done
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [release|strict|debug|asan|tsan|bench|all]"
        exit 1
        ;;
esac

echo "=== Building in mode: $MODE (${CXX}, ${PLATFORM}) ==="
echo

# Step 1: compile bridge.cpp with relaxed warnings.
BRIDGE_OBJ="$OUT_DIR/bridge.o"
echo "→ Compiling bridge..."
$CXX "${BRIDGE_FLAGS[@]}" "${OPT_FLAGS[@]}" "${INCLUDES[@]}" \
    -c "${CXX_BRIDGE_SOURCES[@]}" -o "$BRIDGE_OBJ"

# Step 1b: compile iq_codebooks.cpp (large data tables; no warnings).
IQCB_OBJ="$OUT_DIR/iq_codebooks.o"
echo "→ Compiling iq_codebooks..."
$CXX "${BRIDGE_FLAGS[@]}" "${OPT_FLAGS[@]}" "${INCLUDES[@]}" \
    -c src/iq_codebooks.cpp -o "$IQCB_OBJ"

# Step 2: compile shim (C).
SHIM_OBJ="$OUT_DIR/shim.o"
echo "→ Compiling ggml shim..."
$CC "${SHIM_FLAGS[@]}" "${OPT_FLAGS[@]}" "${INCLUDES[@]}" \
    -c "${C_SHIM_SOURCES[@]}" -o "$SHIM_OBJ"

# Step 3: compile bridge tests (relaxed warnings - ggml header issue).
BRIDGE_TEST_OBJS=()
for src in "${CXX_BRIDGE_TEST_SOURCES[@]}"; do
    obj="$OUT_DIR/$(basename "${src%.cpp}").o"
    $CXX "${BRIDGE_FLAGS[@]}" "${OPT_FLAGS[@]}" "${INCLUDES[@]}" \
        -c "$src" -o "$obj"
    BRIDGE_TEST_OBJS+=("$obj")
done
echo "→ Compiled bridge tests..."

# Step 4: compile HA tests + link everything together.
echo "→ Compiling HA tests + linking..."
LINK_FLAGS=(-lpthread)

# macOS: no -lpthread needed (pthreads built into system library).
if [[ "$PLATFORM" == "Darwin" ]]; then
    LINK_FLAGS=()
fi

$CXX "${HA_FLAGS[@]}" "${OPT_FLAGS[@]}" "${INCLUDES[@]}" \
    "${CXX_HA_SOURCES[@]}" \
    "${BRIDGE_TEST_OBJS[@]}" \
    "$BRIDGE_OBJ" "$SHIM_OBJ" "$IQCB_OBJ" \
    "${LINK_FLAGS[@]}" \
    -o "$BIN"

echo "✓ Build OK: $BIN"
echo
echo "=== Running tests ==="
"$BIN" "${@:2}"
