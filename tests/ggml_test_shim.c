// ============================================================================
// @file        ggml_test_shim.c
//@brief Minimal implementation of ggml runtime for bridge unit tests.
//
//Implements ONLY ggml_backend_buffer_init - a function that
//needed to create ggml_backend_buffer from vtable. Rest
//ggml runtime functions are not called in tests.
//
//This file is NOT used in the production build - it is linked
//real ggml runtime.
// @author      QTX Project
// @date        2026-05-12
// ============================================================================

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <stdlib.h>
#include <string.h>

//The implementation is the same as ggml/src/ggml-backend.cpp:ggml_backend_buffer_init,
//simplified for tests: we simply allocate the struct and fill in the fields.
//
//In real ggml runtime this function also increments the allocated counter
//buffers and registers buffer with device, which is not needed for unit tests.
ggml_backend_buffer_t ggml_backend_buffer_init(
        ggml_backend_buffer_type_t      buft,
        struct ggml_backend_buffer_i    iface,
        void *                          context,
        size_t                          size)
{
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)malloc(sizeof(struct ggml_backend_buffer));
    if (!buffer) {
        return NULL;
    }

    buffer->iface   = iface;
    buffer->buft    = buft;
    buffer->context = context;
    buffer->size    = size;
    buffer->usage   = GGML_BACKEND_BUFFER_USAGE_ANY;

    return buffer;
}

//ggml_backend_buffer_free calls iface.free_buffer and frees the struct.
//Used in tests for cleanup.
void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (!buffer) return;
    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    free(buffer);
}

//Simple getters - needed for tests to check what the bridge returns
//correct values ​​via iface.
void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (!buffer) return NULL;
    return buffer->iface.get_base(buffer);
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    if (!buffer) return 0;
    return buffer->size;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->iface.clear) return;
    buffer->iface.clear(buffer, value);
}

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    if (!buft) return NULL;
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size)
{
    if (!buft) return NULL;
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    if (!buft) return 0;
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    if (!buft) return 0;
    if (buft->iface.get_max_size == NULL) return (size_t)-1;
    return buft->iface.get_max_size(buft);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.is_host == NULL) return false;
    return buft->iface.is_host(buft);
}
