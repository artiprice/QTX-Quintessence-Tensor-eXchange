// ============================================================================
// @file        quantize_microbench.cpp
// @brief       Microbenchmark for all four FP32 <-> quantized hot paths.
// ============================================================================

#include "qtx/quantize/quantizer.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>

using namespace qtx::quantize;
using qtx::core::usize;

template <typename Fn>
static double measureNs(Fn fn, int iters) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count()
         / static_cast<double>(iters);
}

int main() {
    constexpr usize kElements   = 1024;
    constexpr usize kBytes      = kElements * sizeof(float);
    constexpr usize kInt8Bytes  = (kElements / kBlockElements) * 36u;
    constexpr usize kInt4Bytes  = (kElements / kBlockElements) * 20u;
    constexpr usize kBf16Bytes  = kElements * 2u;
    constexpr usize kFp16Bytes  = kElements * 2u;
    constexpr usize kFp8Bytes   = kElements * 1u;
    constexpr usize kNVFP4Bytes = 4u + (kElements / kNVFP4BlockElements) * kNVFP4BlockBytes;
    constexpr usize kMXFP4Bytes = (kElements / kMXFP4BlockElements) * kMXFP4BlockBytes;
    constexpr usize kQ4_1Bytes  = (kElements / kGGML_BlockElements) * kQ4_1BlockBytes;
    constexpr usize kQ5_0Bytes  = (kElements / kGGML_BlockElements) * kQ5_0BlockBytes;
    constexpr usize kQ5_1Bytes  = (kElements / kGGML_BlockElements) * kQ5_1BlockBytes;
    constexpr usize kQ2_KBytes  = (kElements / kKBlockElements) * kQ2_K_BlockBytes;
    constexpr usize kQ3_KBytes  = (kElements / kKBlockElements) * kQ3_K_BlockBytes;
    constexpr usize kQ4_KBytes  = (kElements / kKBlockElements) * kQ4_K_BlockBytes;
    constexpr usize kQ5_KBytes  = (kElements / kKBlockElements) * kQ5_K_BlockBytes;
    constexpr usize kQ6_KBytes  = (kElements / kKBlockElements) * kQ6_K_BlockBytes;
    constexpr usize kIQ4_NLBytes = (kElements / kIQ4_NLBlockElements) * kIQ4_NLBlockBytes;
    constexpr usize kIQ4_XSBytes = (kElements / kKBlockElements) * kIQ4_XSBlockBytes;
    constexpr usize kIQ1_SBytes   = (kElements / kKBlockElements) * kIQ1_S_BlockBytes;
    constexpr usize kIQ1_MBytes   = (kElements / kKBlockElements) * kIQ1_M_BlockBytes;
    constexpr usize kIQ2_XXSBytes = (kElements / kKBlockElements) * kIQ2_XXS_BlockBytes;
    constexpr usize kIQ2_XSBytes  = (kElements / kKBlockElements) * kIQ2_XS_BlockBytes;
    constexpr usize kIQ2_SBytes   = (kElements / kKBlockElements) * kIQ2_S_BlockBytes_v2;
    constexpr usize kIQ3_XXSBytes = (kElements / kKBlockElements) * kIQ3_XXS_BlockBytes;
    constexpr usize kIQ3_SBytes   = (kElements / kKBlockElements) * kIQ3_S_BlockBytes;
    constexpr int kIters = 100'000;

    alignas(64) std::array<float, kElements>  src_floats{};
    alignas(64) std::array<std::byte, kBytes> src{};
    alignas(64) std::array<std::byte, kInt8Bytes>  dst_int8{};
    alignas(64) std::array<std::byte, kInt4Bytes>  dst_int4{};
    alignas(64) std::array<std::byte, kBf16Bytes>  dst_bf16{};
    alignas(64) std::array<std::byte, kFp16Bytes>  dst_fp16{};
    alignas(64) std::array<std::byte, kFp8Bytes>   dst_e4m3{};
    alignas(64) std::array<std::byte, kFp8Bytes>   dst_e5m2{};
    alignas(64) std::array<std::byte, kNVFP4Bytes> dst_nvfp4{};
    alignas(64) std::array<std::byte, kMXFP4Bytes> dst_mxfp4{};
    alignas(64) std::array<std::byte, kQ4_1Bytes>  dst_q4_1{};
    alignas(64) std::array<std::byte, kQ5_0Bytes>  dst_q5_0{};
    alignas(64) std::array<std::byte, kQ5_1Bytes>  dst_q5_1{};
    alignas(64) std::array<std::byte, kQ2_KBytes>  dst_q2_k{};
    alignas(64) std::array<std::byte, kQ3_KBytes>  dst_q3_k{};
    alignas(64) std::array<std::byte, kQ4_KBytes>  dst_q4_k{};
    alignas(64) std::array<std::byte, kQ5_KBytes>  dst_q5_k{};
    alignas(64) std::array<std::byte, kQ6_KBytes>  dst_q6_k{};
    alignas(64) std::array<std::byte, kIQ4_NLBytes> dst_iq4_nl{};
    alignas(64) std::array<std::byte, kIQ4_XSBytes> dst_iq4_xs{};
    alignas(64) std::array<std::byte, kIQ1_SBytes>   dst_iq1_s{};
    alignas(64) std::array<std::byte, kIQ1_MBytes>   dst_iq1_m{};
    alignas(64) std::array<std::byte, kIQ2_XXSBytes> dst_iq2_xxs{};
    alignas(64) std::array<std::byte, kIQ2_XSBytes>  dst_iq2_xs{};
    alignas(64) std::array<std::byte, kIQ2_SBytes>   dst_iq2_s{};
    alignas(64) std::array<std::byte, kIQ3_XXSBytes> dst_iq3_xxs{};
    alignas(64) std::array<std::byte, kIQ3_SBytes>   dst_iq3_s{};
    alignas(64) std::array<std::byte, kBytes>      dst_fp32{};

    for (usize i = 0; i < kElements; ++i) {
        src_floats[i] = static_cast<float>((i * 7919u) % 1024u) / 64.0f - 8.0f;
    }
    std::memcpy(src.data(), src_floats.data(), kBytes);

    // Pre-compress for the decompress benchmarks.
    (void)compressFP32ToINT8(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_int8});
    (void)compressFP32ToINT4(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_int4});
    (void)compressFP32ToBF16(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_bf16});
    (void)compressFP32ToFP16(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_fp16});
    (void)compressFP32ToFP8_E4M3(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_e4m3});
    (void)compressFP32ToFP8_E5M2(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_e5m2});
    (void)compressFP32ToNVFP4(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_nvfp4});
    (void)compressFP32ToMXFP4(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_mxfp4});
    (void)compressFP32ToQ4_1(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q4_1});
    (void)compressFP32ToQ5_0(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q5_0});
    (void)compressFP32ToQ5_1(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q5_1});
    (void)compressFP32ToQ2_K(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q2_k});
    (void)compressFP32ToQ3_K(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q3_k});
    (void)compressFP32ToQ4_K(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q4_k});
    (void)compressFP32ToQ5_K(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q5_k});
    (void)compressFP32ToQ6_K(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_q6_k});
    (void)compressFP32ToIQ4_NL(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq4_nl});
    (void)compressFP32ToIQ4_XS(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq4_xs});
    (void)compressFP32ToIQ1_S(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq1_s});
    (void)compressFP32ToIQ1_M(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq1_m});
    (void)compressFP32ToIQ2_XXS(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq2_xxs});
    (void)compressFP32ToIQ2_XS(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq2_xs});
    (void)compressFP32ToIQ2_S(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq2_s});
    (void)compressFP32ToIQ3_XXS(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq3_xxs});
    (void)compressFP32ToIQ3_S(
        std::span<const std::byte>{src}, std::span<std::byte>{dst_iq3_s});

    // Sink that prevents the compiler from eliding the kernel calls.
    // We checksum the first few bytes of dst_fp32 after each measurement.
    volatile std::byte sink{};

    // Warmup.
    for (int i = 0; i < 1000; ++i) {
        (void)compressFP32ToINT8(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_int8});
    }

    const double ns_c_int8 = measureNs([&]() {
        (void)compressFP32ToINT8(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_int8});
        sink = dst_int8[0];
    }, kIters);
    const double ns_d_int8 = measureNs([&]() {
        (void)decompressINT8ToFP32(
            std::span<const std::byte>{dst_int8}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_int4 = measureNs([&]() {
        (void)compressFP32ToINT4(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_int4});
        sink = dst_int4[0];
    }, kIters);
    const double ns_d_int4 = measureNs([&]() {
        (void)decompressINT4ToFP32(
            std::span<const std::byte>{dst_int4}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_bf16 = measureNs([&]() {
        (void)compressFP32ToBF16(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_bf16});
        sink = dst_bf16[0];
    }, kIters);
    const double ns_d_bf16 = measureNs([&]() {
        (void)decompressBF16ToFP32(
            std::span<const std::byte>{dst_bf16}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_fp16 = measureNs([&]() {
        (void)compressFP32ToFP16(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_fp16});
        sink = dst_fp16[0];
    }, kIters);
    const double ns_d_fp16 = measureNs([&]() {
        (void)decompressFP16ToFP32(
            std::span<const std::byte>{dst_fp16}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_e4m3 = measureNs([&]() {
        (void)compressFP32ToFP8_E4M3(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_e4m3});
        sink = dst_e4m3[0];
    }, kIters);
    const double ns_d_e4m3 = measureNs([&]() {
        (void)decompressFP8_E4M3_ToFP32(
            std::span<const std::byte>{dst_e4m3}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_e5m2 = measureNs([&]() {
        (void)compressFP32ToFP8_E5M2(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_e5m2});
        sink = dst_e5m2[0];
    }, kIters);
    const double ns_d_e5m2 = measureNs([&]() {
        (void)decompressFP8_E5M2_ToFP32(
            std::span<const std::byte>{dst_e5m2}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_nvfp4 = measureNs([&]() {
        (void)compressFP32ToNVFP4(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_nvfp4});
        sink = dst_nvfp4[0];
    }, kIters);
    const double ns_d_nvfp4 = measureNs([&]() {
        (void)decompressNVFP4ToFP32(
            std::span<const std::byte>{dst_nvfp4}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_mxfp4 = measureNs([&]() {
        (void)compressFP32ToMXFP4(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_mxfp4});
        sink = dst_mxfp4[0];
    }, kIters);
    const double ns_d_mxfp4 = measureNs([&]() {
        (void)decompressMXFP4ToFP32(
            std::span<const std::byte>{dst_mxfp4}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q4_1 = measureNs([&]() {
        (void)compressFP32ToQ4_1(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q4_1});
        sink = dst_q4_1[0];
    }, kIters);
    const double ns_d_q4_1 = measureNs([&]() {
        (void)decompressQ4_1ToFP32(
            std::span<const std::byte>{dst_q4_1}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q5_0 = measureNs([&]() {
        (void)compressFP32ToQ5_0(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q5_0});
        sink = dst_q5_0[0];
    }, kIters);
    const double ns_d_q5_0 = measureNs([&]() {
        (void)decompressQ5_0ToFP32(
            std::span<const std::byte>{dst_q5_0}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q5_1 = measureNs([&]() {
        (void)compressFP32ToQ5_1(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q5_1});
        sink = dst_q5_1[0];
    }, kIters);
    const double ns_d_q5_1 = measureNs([&]() {
        (void)decompressQ5_1ToFP32(
            std::span<const std::byte>{dst_q5_1}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q2_k = measureNs([&]() {
        (void)compressFP32ToQ2_K(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q2_k});
        sink = dst_q2_k[0];
    }, kIters);
    const double ns_d_q2_k = measureNs([&]() {
        (void)decompressQ2_K_ToFP32(
            std::span<const std::byte>{dst_q2_k}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q3_k = measureNs([&]() {
        (void)compressFP32ToQ3_K(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q3_k});
        sink = dst_q3_k[0];
    }, kIters);
    const double ns_d_q3_k = measureNs([&]() {
        (void)decompressQ3_K_ToFP32(
            std::span<const std::byte>{dst_q3_k}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q4_k = measureNs([&]() {
        (void)compressFP32ToQ4_K(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q4_k});
        sink = dst_q4_k[0];
    }, kIters);
    const double ns_d_q4_k = measureNs([&]() {
        (void)decompressQ4_K_ToFP32(
            std::span<const std::byte>{dst_q4_k}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q5_k = measureNs([&]() {
        (void)compressFP32ToQ5_K(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q5_k});
        sink = dst_q5_k[0];
    }, kIters);
    const double ns_d_q5_k = measureNs([&]() {
        (void)decompressQ5_K_ToFP32(
            std::span<const std::byte>{dst_q5_k}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_q6_k = measureNs([&]() {
        (void)compressFP32ToQ6_K(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_q6_k});
        sink = dst_q6_k[0];
    }, kIters);
    const double ns_d_q6_k = measureNs([&]() {
        (void)decompressQ6_K_ToFP32(
            std::span<const std::byte>{dst_q6_k}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq4_nl = measureNs([&]() {
        (void)compressFP32ToIQ4_NL(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq4_nl});
        sink = dst_iq4_nl[0];
    }, kIters);
    const double ns_d_iq4_nl = measureNs([&]() {
        (void)decompressIQ4_NL_ToFP32(
            std::span<const std::byte>{dst_iq4_nl}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq4_xs = measureNs([&]() {
        (void)compressFP32ToIQ4_XS(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq4_xs});
        sink = dst_iq4_xs[0];
    }, kIters);
    const double ns_d_iq4_xs = measureNs([&]() {
        (void)decompressIQ4_XS_ToFP32(
            std::span<const std::byte>{dst_iq4_xs}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);

    // ---- Codebook IQ family (note: lower iteration count due to NN-search cost) ----
    constexpr int kIters_CB = 1000;
    const double ns_c_iq1_s = measureNs([&]() {
        (void)compressFP32ToIQ1_S(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq1_s});
        sink = dst_iq1_s[0];
    }, kIters_CB);
    const double ns_d_iq1_s = measureNs([&]() {
        (void)decompressIQ1_S_ToFP32(
            std::span<const std::byte>{dst_iq1_s}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq1_m = measureNs([&]() {
        (void)compressFP32ToIQ1_M(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq1_m});
        sink = dst_iq1_m[0];
    }, kIters_CB);
    const double ns_d_iq1_m = measureNs([&]() {
        (void)decompressIQ1_M_ToFP32(
            std::span<const std::byte>{dst_iq1_m}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq2_xxs = measureNs([&]() {
        (void)compressFP32ToIQ2_XXS(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq2_xxs});
        sink = dst_iq2_xxs[0];
    }, kIters_CB);
    const double ns_d_iq2_xxs = measureNs([&]() {
        (void)decompressIQ2_XXS_ToFP32(
            std::span<const std::byte>{dst_iq2_xxs}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq2_xs = measureNs([&]() {
        (void)compressFP32ToIQ2_XS(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq2_xs});
        sink = dst_iq2_xs[0];
    }, kIters_CB);
    const double ns_d_iq2_xs = measureNs([&]() {
        (void)decompressIQ2_XS_ToFP32(
            std::span<const std::byte>{dst_iq2_xs}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq2_s = measureNs([&]() {
        (void)compressFP32ToIQ2_S(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq2_s});
        sink = dst_iq2_s[0];
    }, kIters_CB);
    const double ns_d_iq2_s = measureNs([&]() {
        (void)decompressIQ2_S_ToFP32(
            std::span<const std::byte>{dst_iq2_s}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq3_xxs = measureNs([&]() {
        (void)compressFP32ToIQ3_XXS(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq3_xxs});
        sink = dst_iq3_xxs[0];
    }, kIters_CB);
    const double ns_d_iq3_xxs = measureNs([&]() {
        (void)decompressIQ3_XXS_ToFP32(
            std::span<const std::byte>{dst_iq3_xxs}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    const double ns_c_iq3_s = measureNs([&]() {
        (void)compressFP32ToIQ3_S(
            std::span<const std::byte>{src}, std::span<std::byte>{dst_iq3_s});
        sink = dst_iq3_s[0];
    }, kIters_CB);
    const double ns_d_iq3_s = measureNs([&]() {
        (void)decompressIQ3_S_ToFP32(
            std::span<const std::byte>{dst_iq3_s}, std::span<std::byte>{dst_fp32});
        sink = dst_fp32[0];
    }, kIters);
    (void)sink;

    auto row = [](const char* name, double ns, double in_bytes) {
        const double gbps = in_bytes / ns;  // bytes/ns == GB/s
        std::printf("  %-30s %7.0f ns   %7.1f ns/block   %5.2f GB/s\n",
                    name, ns, ns / 32.0, gbps);
    };

    std::printf("Quantize hot-path benchmark (1024 elements / call)\n");
    std::printf("  %-30s %10s   %16s   %10s\n",
                "kernel", "ns/call", "ns/block", "throughput");
    std::printf("  ─────────────────────────────────────────────────────────────────\n");
    row("compressFP32ToINT8",        ns_c_int8,  static_cast<double>(kBytes));
    row("decompressINT8ToFP32",      ns_d_int8,  static_cast<double>(kInt8Bytes));
    row("compressFP32ToINT4",        ns_c_int4,  static_cast<double>(kBytes));
    row("decompressINT4ToFP32",      ns_d_int4,  static_cast<double>(kInt4Bytes));
    row("compressFP32ToBF16",        ns_c_bf16,  static_cast<double>(kBytes));
    row("decompressBF16ToFP32",      ns_d_bf16,  static_cast<double>(kBf16Bytes));
    row("compressFP32ToFP16",        ns_c_fp16,  static_cast<double>(kBytes));
    row("decompressFP16ToFP32",      ns_d_fp16,  static_cast<double>(kFp16Bytes));
    row("compressFP32ToFP8_E4M3",    ns_c_e4m3,  static_cast<double>(kBytes));
    row("decompressFP8_E4M3_ToFP32", ns_d_e4m3,  static_cast<double>(kFp8Bytes));
    row("compressFP32ToFP8_E5M2",    ns_c_e5m2,  static_cast<double>(kBytes));
    row("decompressFP8_E5M2_ToFP32", ns_d_e5m2,  static_cast<double>(kFp8Bytes));
    row("compressFP32ToNVFP4",       ns_c_nvfp4, static_cast<double>(kBytes));
    row("decompressNVFP4ToFP32",     ns_d_nvfp4, static_cast<double>(kNVFP4Bytes));
    row("compressFP32ToMXFP4",       ns_c_mxfp4, static_cast<double>(kBytes));
    row("decompressMXFP4ToFP32",     ns_d_mxfp4, static_cast<double>(kMXFP4Bytes));
    row("compressFP32ToQ4_1",        ns_c_q4_1,  static_cast<double>(kBytes));
    row("decompressQ4_1ToFP32",      ns_d_q4_1,  static_cast<double>(kQ4_1Bytes));
    row("compressFP32ToQ5_0",        ns_c_q5_0,  static_cast<double>(kBytes));
    row("decompressQ5_0ToFP32",      ns_d_q5_0,  static_cast<double>(kQ5_0Bytes));
    row("compressFP32ToQ5_1",        ns_c_q5_1,  static_cast<double>(kBytes));
    row("decompressQ5_1ToFP32",      ns_d_q5_1,  static_cast<double>(kQ5_1Bytes));
    row("compressFP32ToQ2_K",        ns_c_q2_k,  static_cast<double>(kBytes));
    row("decompressQ2_K_ToFP32",     ns_d_q2_k,  static_cast<double>(kQ2_KBytes));
    row("compressFP32ToQ3_K",        ns_c_q3_k,  static_cast<double>(kBytes));
    row("decompressQ3_K_ToFP32",     ns_d_q3_k,  static_cast<double>(kQ3_KBytes));
    row("compressFP32ToQ4_K",        ns_c_q4_k,  static_cast<double>(kBytes));
    row("decompressQ4_K_ToFP32",     ns_d_q4_k,  static_cast<double>(kQ4_KBytes));
    row("compressFP32ToQ5_K",        ns_c_q5_k,  static_cast<double>(kBytes));
    row("decompressQ5_K_ToFP32",     ns_d_q5_k,  static_cast<double>(kQ5_KBytes));
    row("compressFP32ToQ6_K",        ns_c_q6_k,  static_cast<double>(kBytes));
    row("decompressQ6_K_ToFP32",     ns_d_q6_k,  static_cast<double>(kQ6_KBytes));
    row("compressFP32ToIQ4_NL",      ns_c_iq4_nl, static_cast<double>(kBytes));
    row("decompressIQ4_NL_ToFP32",   ns_d_iq4_nl, static_cast<double>(kIQ4_NLBytes));
    row("compressFP32ToIQ4_XS",      ns_c_iq4_xs, static_cast<double>(kBytes));
    row("decompressIQ4_XS_ToFP32",   ns_d_iq4_xs, static_cast<double>(kIQ4_XSBytes));
    row("compressFP32ToIQ1_S",       ns_c_iq1_s,   static_cast<double>(kBytes));
    row("decompressIQ1_S_ToFP32",    ns_d_iq1_s,   static_cast<double>(kIQ1_SBytes));
    row("compressFP32ToIQ1_M",       ns_c_iq1_m,   static_cast<double>(kBytes));
    row("decompressIQ1_M_ToFP32",    ns_d_iq1_m,   static_cast<double>(kIQ1_MBytes));
    row("compressFP32ToIQ2_XXS",     ns_c_iq2_xxs, static_cast<double>(kBytes));
    row("decompressIQ2_XXS_ToFP32",  ns_d_iq2_xxs, static_cast<double>(kIQ2_XXSBytes));
    row("compressFP32ToIQ2_XS",      ns_c_iq2_xs,  static_cast<double>(kBytes));
    row("decompressIQ2_XS_ToFP32",   ns_d_iq2_xs,  static_cast<double>(kIQ2_XSBytes));
    row("compressFP32ToIQ2_S",       ns_c_iq2_s,   static_cast<double>(kBytes));
    row("decompressIQ2_S_ToFP32",    ns_d_iq2_s,   static_cast<double>(kIQ2_SBytes));
    row("compressFP32ToIQ3_XXS",     ns_c_iq3_xxs, static_cast<double>(kBytes));
    row("decompressIQ3_XXS_ToFP32",  ns_d_iq3_xxs, static_cast<double>(kIQ3_XXSBytes));
    row("compressFP32ToIQ3_S",       ns_c_iq3_s,   static_cast<double>(kBytes));
    row("decompressIQ3_S_ToFP32",    ns_d_iq3_s,   static_cast<double>(kIQ3_SBytes));
    return 0;
}

