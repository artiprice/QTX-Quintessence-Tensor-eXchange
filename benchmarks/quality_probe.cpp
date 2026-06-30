// Quick-and-dirty quality probe: measure quantization fidelity on
// a sin/cos workload that mimics neural-network activations.
// Reports per-format: max-abs-error, mean-abs-error, signal-to-noise (dB),
// and storage ratio.

#include "qtx/quantize/quantizer.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace qtx;

static void probe(const char* name, arena::QuantFormat fmt,
                  const std::vector<float>& src,
                  std::vector<float>& out)
{
    const core::usize n = src.size();
    const core::usize compressed = quantize::compressedSize(fmt, n);

    std::vector<std::byte> buf(compressed);
    auto src_bytes  = std::as_bytes(std::span<const float>(src.data(), n));
    auto buf_span   = std::span<std::byte>(buf.data(), compressed);
    auto out_bytes  = std::as_writable_bytes(std::span<float>(out.data(), n));

    auto cw = quantize::compress(fmt, src_bytes, buf_span);
    auto dw = quantize::decompress(fmt, std::span<const std::byte>(buf_span), out_bytes);

    if (cw == 0u || dw == 0u) {
        std::printf("  %-8s FAILED (compress/decompress returned 0)\n", name);
        return;
    }

    double sum_sq_signal = 0.0;
    double sum_sq_error  = 0.0;
    double sum_abs_error = 0.0;
    double max_abs_error = 0.0;
    for (core::usize i = 0; i < n; ++i) {
        const double e = static_cast<double>(out[i]) - src[i];
        sum_sq_signal += static_cast<double>(src[i]) * src[i];
        sum_sq_error  += e * e;
        const double a = std::fabs(e);
        sum_abs_error += a;
        if (a > max_abs_error) max_abs_error = a;
    }

    const double mae = sum_abs_error / static_cast<double>(n);
    const double snr_db = 10.0 * std::log10(
        sum_sq_signal / (sum_sq_error + 1e-30));
    const double ratio = static_cast<double>(n * sizeof(float)) /
                         static_cast<double>(cw);

    std::printf("  %-8s  bytes=%6zu  ratio=%5.2fx  max_abs=%.5f  MAE=%.5f  SNR=%6.2f dB\n",
                name, cw, ratio, max_abs_error, mae, snr_db);
}

int main() {
    constexpr core::usize kN = 4096u;
    std::vector<float> src(kN), out(kN);
    // sin/cos mix in [-1, 1] — mimics layer-norm activations
    for (core::usize i = 0; i < kN; ++i) {
        const float t = static_cast<float>(i) * 0.01f;
        src[i] = 0.7f * std::sin(t) + 0.3f * std::cos(3.0f * t);
    }

    std::printf("Quantization fidelity (%zu elements, sin/cos workload, range [-1,1])\n", kN);
    std::printf("  format   bytes  ratio  max_abs    MAE       SNR\n");
    std::printf("  ──────────────────────────────────────────────────────\n");
    probe("BF16",  arena::QuantFormat::kBF16, src, out);
    probe("INT8",  arena::QuantFormat::kINT8, src, out);
    probe("INT4",  arena::QuantFormat::kINT4, src, out);
    return 0;
}
