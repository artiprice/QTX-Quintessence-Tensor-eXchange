// ============================================================================
// @file        real_ggml_demo.cpp
//@brief End-to-end demo: qtx backend serves the real ggml runtime.
// @author      QTX Project
// @date        2026-05-13
//
//Demonstrates that qtx::ArenaBridge integrates correctly
//with real ggml v0.11.1 - not shim, but production runtime.
//
//Assembly:
//                cmake .. -DQTX_BUILD_WITH_GGML=ON
//                make real_ggml_demo
//                ./real_ggml_demo
//
// ============================================================================

#include "qtx/unsafe/ggml_bridge/bridge.hpp"
#include "qtx/arena/fractal_arena.hpp"

extern "C" {
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
}

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace qtx;

int main() {
    std::printf("==============================================================\n");
    std::printf("  qtx × ggml — end-to-end integration demo\n");
    std::printf("==============================================================\n\n");

    //Step 1. Create a qtx arena.
    using Arena = arena::FractalArena<256, 4096>;
    auto arena = std::make_unique<Arena>();
    std::printf("[1/5] FractalArena created: %zu slots × %zu bytes = %zu KiB\n",
        Arena::kSlotCount, Arena::kSlotSize, Arena::kTotalSize / 1024);

    //Step 2. Create a bridge over the arena.
    unsafe::ggml_bridge::ArenaBridge<Arena> bridge(
        arena.get(),
        unsafe::ggml_bridge::defaultConfig(Arena::kSlotSize));
    auto* buft = bridge.bufferType();
    if (!buft) {
        std::printf("FAIL: bridge.bufferType() returned null\n");
        return 1;
    }
    std::printf("[2/5] Bridge initialized: %s, align=%zu, max_size=%zu\n",
        ggml_backend_buft_name(buft),
        ggml_backend_buft_get_alignment(buft),
        ggml_backend_buft_get_max_size(buft));

    //Step 3. Initialize the ggml context (for tensor metadata).
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL: ggml_init\n");
        return 1;
    }
    std::printf("[3/5] ggml_context created\n");

    //Step 4. Create several tensors and allocate buffers for them
    //THROUGH OUR BRIDGE. Each tensor → one slot in our arena.
    constexpr int kTensorCount = 5;
    std::vector<struct ggml_tensor*>      tensors(kTensorCount);
    std::vector<ggml_backend_buffer_t>    buffers(kTensorCount);

    for (int i = 0; i < kTensorCount; ++i) {
        tensors[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        const size_t need = ggml_nbytes(tensors[i]);
        buffers[i] = ggml_backend_buft_alloc_buffer(buft, need);
        if (!buffers[i]) {
            std::printf("FAIL: alloc_buffer for tensor %d\n", i);
            return 1;
        }
        tensors[i]->data   = ggml_backend_buffer_get_base(buffers[i]);
        tensors[i]->buffer = buffers[i];
    }
    std::printf("[4/5] %d tensors allocated through bridge: arena uses %zu/%zu slots\n",
        kTensorCount, arena->usedSlots(), Arena::kSlotCount);

    //Step 5. Write unique data to each tensor, read back,
    //checking integrity.
    int total_errors = 0;
    for (int i = 0; i < kTensorCount; ++i) {
        std::vector<float> input(32 * 32);
        for (size_t k = 0; k < input.size(); ++k) {
            input[k] = static_cast<float>(i * 1000 + static_cast<int>(k));
        }
        ggml_backend_tensor_set(tensors[i], input.data(), 0, ggml_nbytes(tensors[i]));

        std::vector<float> output(32 * 32);
        ggml_backend_tensor_get(tensors[i], output.data(), 0, ggml_nbytes(tensors[i]));

        int errors = 0;
        for (size_t k = 0; k < input.size(); ++k) {
            if (input[k] != output[k]) ++errors;
        }
        total_errors += errors;
    }
    std::printf("[5/5] Round-trip verification: %d errors across %d tensors × 1024 floats\n",
        total_errors, kTensorCount);

    // Cleanup.
    for (auto* buf : buffers) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::printf("\n   Cleanup: arena uses %zu/%zu slots\n",
        arena->usedSlots(), Arena::kSlotCount);

    std::printf("\n==============================================================\n");
    if (total_errors == 0 && arena->usedSlots() == 0) {
        std::printf("  STATUS: PASS — qtx bridge serves real ggml correctly\n");
    } else {
        std::printf("  STATUS: FAIL — verification or cleanup issue\n");
    }
    std::printf("==============================================================\n");
    return (total_errors == 0 && arena->usedSlots() == 0) ? 0 : 1;
}
