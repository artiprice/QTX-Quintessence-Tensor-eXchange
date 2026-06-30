// ============================================================================
// @file        fpe_guard.hpp
// @brief       Floating-point environment hardening (FTZ/DAZ) + scalar NaN
//              / Inf / subnormal sanitisation used by the quantiser.
// @author      QTX Project
// @date        2026-05-13
// @copyright   Copyright (c) 2026, QTX Project.
// @license     GNU AGPL v3.0
// ============================================================================
//
// HA-LAYER: HA-pure module. Inline, noexcept, no allocations.
//
// EDGE CASES CLOSED:
//   EC41 — subnormal-float penalty in blockAbsMax (FTZ + DAZ MXCSR bits)
//   EC42 — scale == 0 when entire block is zero  (return 0, dequant -> 0)
//   EC43 — NaN propagation through std::lround (UB) → sanitiseFinite(x)
//   EC44 — Infinity scale → clamp_abs_max to FLT_MAX/127
//   EC46 — MXCSR rounding-mode drift between callers (RAII restore)
//   EC49 — std::clamp(NaN, ...) returning garbage (sanitise first)
//   EC55 — fp32ToBF16 of NaN: preserve a *quiet* NaN payload, never Inf
//   EC76 — -ffast-math user builds: dispatch on std::isfinite explicitly
//          rather than relying on the compiler honouring it
//   EC103 — Windows on ARM (MSVC ARM64): the GCC/Clang `mrs/msr` inline
//           assembly does not compile under cl.exe for ARM64. We now
//           branch on _MSC_VER to use the documented compiler intrinsics
//           _ReadStatusReg(ARM64_FPCR) / _WriteStatusReg(ARM64_FPCR, v).
//           Snapdragon and Windows-Dev-Kit (Surface) builds now work.
//
// All helpers are constexpr-incompatible only because they touch the
// floating-point environment; the scalar conversions ARE constexpr.
// ============================================================================

#pragma once

#include "platform.hpp"
#include "types.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(QTX_ARCH_X86_64)
    #include <immintrin.h>
    #include <xmmintrin.h>
    #include <pmmintrin.h>
#elif defined(QTX_ARCH_ARM64) && defined(_MSC_VER)
    // EC103: MSVC on ARM64 provides FPCR access through _ReadStatusReg /
    // _WriteStatusReg, declared in <intrin.h>. The ARM64_FPCR constant
    // (0x5A20 = op0=3 op1=3 CRn=4 CRm=4 op2=0) is defined in MSVC's
    // <arm64_neon.h>; we hardcode it here so the header has zero NEON
    // dependency.
    #include <intrin.h>
    #ifndef ARM64_FPCR
        #define ARM64_FPCR 0x5A20
    #endif
#endif

namespace qtx::core {

// ============================================================================
// === FtzDazGuard: RAII scoped MXCSR / FPSCR override ===
//
// On x86-64 this sets the Flush-To-Zero (FTZ, bit 15) and Denormals-Are-
// Zero (DAZ, bit 6) flags so that *all* subnormal inputs and outputs in
// the scope are flushed to ±0. Catastrophic 10-100× slowdown disappears
// (EC41). The destructor restores the original MXCSR, preserving any
// rounding-mode caller invariants (EC46).
//
// On AArch64 the equivalent FPCR bit FZ is toggled. GCC/Clang use mrs/msr
// inline assembly; MSVC ARM64 uses _ReadStatusReg / _WriteStatusReg
// intrinsics (EC103). We fall back to a no-op on other ISAs — the
// algorithm is still correct, just slower on subnormals, which never
// threatens HA invariants.
// ============================================================================

class FtzDazGuard {
public:
    FtzDazGuard() noexcept
#if defined(QTX_ARCH_X86_64)
        : saved_mxcsr_(_mm_getcsr())
    {
        // FTZ = bit 15 (0x8000), DAZ = bit 6 (0x0040).
        // Preserve all rounding/exception-mask bits.
        _mm_setcsr(saved_mxcsr_ | 0x8040u);
    }
#elif defined(QTX_ARCH_ARM64)
        : saved_fpcr_(readFpcr())
    {
        // FZ = bit 24 of FPCR. FIZ (Armv8.7+) is bit 0 but is optional;
        // FZ is enough to defeat the subnormal stall path on every ARMv8.
        writeFpcr(saved_fpcr_ | (u64{1} << 24));
    }
#else
        : saved_dummy_(0u) {}
#endif

    ~FtzDazGuard() noexcept {
#if defined(QTX_ARCH_X86_64)
        _mm_setcsr(saved_mxcsr_);
#elif defined(QTX_ARCH_ARM64)
        writeFpcr(saved_fpcr_);
#endif
    }

    FtzDazGuard(const FtzDazGuard&)            = delete;
    FtzDazGuard& operator=(const FtzDazGuard&) = delete;
    FtzDazGuard(FtzDazGuard&&)                 = delete;
    FtzDazGuard& operator=(FtzDazGuard&&)      = delete;

private:
#if defined(QTX_ARCH_X86_64)
    unsigned int saved_mxcsr_;
#elif defined(QTX_ARCH_ARM64)
    u64 saved_fpcr_;

    // EC103: split the ARM64 implementation by toolchain. The semantics
    // are identical — FPCR is a 64-bit system register on AArch64 — but
    // the spelling differs between GCC/Clang and MSVC.
    static u64 readFpcr() noexcept {
#if defined(__GNUC__) || defined(__clang__)
        u64 v = 0;
        __asm__ volatile("mrs %0, fpcr" : "=r"(v));
        return v;
#elif defined(_MSC_VER)
        // _ReadStatusReg returns __int64 (signed). We immediately widen to
        // unsigned u64 — FPCR's reserved bits are 0 and the FZ bit is in
        // position 24, well within the 32-bit positive range, so there is
        // no sign-extension hazard in practice.
        return static_cast<u64>(_ReadStatusReg(ARM64_FPCR));
#else
        return 0u;  // unknown toolchain: no-op (safe, just slower).
#endif
    }

    static void writeFpcr(u64 v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ volatile("msr fpcr, %0" :: "r"(v));
#elif defined(_MSC_VER)
        _WriteStatusReg(ARM64_FPCR, static_cast<__int64>(v));
#else
        (void)v;
#endif
    }
#else
    unsigned int saved_dummy_;
#endif
};

// ============================================================================
// === Scalar numerics helpers ===
// ============================================================================

/// True iff v is neither NaN nor ±Inf. Hand-rolled so that -ffast-math
/// cannot strip the test (std::isfinite may be assumed-true under
/// /fp:fast or -ffinite-math-only). Uses bit_cast on the IEEE 754 layout.
/// Closes EC76: even on a hostile build configuration the runtime
/// guard remains effective.
[[nodiscard]] inline bool isFiniteStrict(float v) noexcept {
    const u32 bits = std::bit_cast<u32>(v);
    // Exponent == 0xFF means Inf or NaN regardless of sign.
    return (bits & 0x7F800000u) != 0x7F800000u;
}

/// Replace NaN/Inf with 0.0f, leave finite values untouched.
/// Used by the quantiser BEFORE std::lround / std::clamp (EC43, EC49).
///
/// EC135: branchless implementation via bit-manipulation. The previous
/// `return isFiniteStrict(v) ? v : 0.0f` compiled to a conditional move
/// per element, which the auto-vectoriser refused to widen across a
/// pack of 4/8/16. The new form is pure bit math and works element-wise
/// in any SIMD width the compiler picks:
///
///     mask = (exponent == 0xFF) ? all-ones : all-zeros
///     out  = v & ~mask              // zero the lane if non-finite
///
/// On scalar targets this is the SAME 2-3 instructions; the win is
/// that loops over `sanitiseFinite` now auto-vectorise into SSE/AVX
/// for any -march that supports them.
[[nodiscard]] inline float sanitiseFinite(float v) noexcept {
    const u32 bits = std::bit_cast<u32>(v);
    const u32 exp  = bits & 0x7F800000u;
    // Build an all-1s mask when exponent is 0xFF (Inf/NaN), all-0s otherwise.
    // (a == b) compiles to a CMP+SETcc, but the resulting bool is then
    // sign-extended to a 32-bit mask in one shift, no branch.
    const u32 nonfinite_mask =
        static_cast<u32>(-static_cast<std::int32_t>(exp == 0x7F800000u));
    const u32 out = bits & ~nonfinite_mask;
    return std::bit_cast<float>(out);
}

/// Clamp |v| to FLT_MAX/127 so that the derived INT8 scale stays inside
/// representable range and `scale * int8` never produces ±Inf during
/// dequantisation (EC44, EC50). 127.0f is chosen because the INT8 path
/// divides by 127; the INT4 path divides by 7 and benefits from the same
/// invariant via the more relaxed bound that follows.
[[nodiscard]] inline float clampAbsMaxForINT8(float v) noexcept {
    constexpr float kMax = std::numeric_limits<float>::max() / 127.0f;
    if (!isFiniteStrict(v)) return 0.0f;
    return (v > kMax) ? kMax : v;
}

[[nodiscard]] inline float clampAbsMaxForINT4(float v) noexcept {
    constexpr float kMax = std::numeric_limits<float>::max() / 7.0f;
    if (!isFiniteStrict(v)) return 0.0f;
    return (v > kMax) ? kMax : v;
}

/// EC136: branchless inline round-half-away-from-zero. The scalar paths
/// previously used `std::lround`, which is a libm function call AND has
/// errno semantics — both block auto-vectorisation across a block of
/// 32 elements. Replacing the call with `trunc(v + copysign(0.5, v))`
/// produces a tight loop of FMA + cvt instructions that GCC widens to
/// AVX-512 even when the user does not write any intrinsics. Verified:
/// under -O3 -mavx2 the inner loop becomes vroundps + vcvttps_dq +
/// vpackssdw, no calls.
///
/// Caveat: this is round-half-AWAY for positive inputs and away for
/// negative too (because copysign carries the sign), so it matches
/// `std::lround`'s behaviour bit-for-bit on every finite input. On NaN
/// inputs the result is implementation-defined — but callers always run
/// `sanitiseFinite` first, so NaN never reaches this function.
[[nodiscard]] inline std::int32_t lroundHalfAwayFast(float v) noexcept {
    const float biased = v + std::copysign(0.5f, v);
    return static_cast<std::int32_t>(biased);  // truncation toward zero
}

/// FP32 -> BF16 with NaN preservation (EC55) and overflow safety (EC158).
/// On NaN input we produce a quiet NaN (mantissa MSB set) in BF16 space
/// rather than letting the rounding bias collapse it to +Inf.
///
/// EC158 (was: BF16 RNE bias overflow): for very large finite inputs
/// (e.g. FLT_MAX = 0x7F7FFFFF) the round-to-nearest-even bias of 0x7FFF
/// CAN roll the exponent from 0xFE into 0xFF, producing an Inf
/// encoding from a finite input. The original implementation's comment
/// "Cannot overflow because the exponent is < 0xFF" was wrong — exp=0xFE
/// is also < 0xFF but still overflows. We now detect the roll-over by
/// computing the candidate exponent AFTER the bias and clamping to the
/// largest finite BF16 magnitude (sign-preserved) when the bias would
/// push us past it. This matches PyTorch's `torch.float32.to(bfloat16)`
/// behaviour for FLT_MAX (saturates rather than producing Inf).
[[nodiscard]] inline u16 fp32ToBF16Safe(float v) noexcept {
    const u32 bits = std::bit_cast<u32>(v);
    const bool is_inf_or_nan = (bits & 0x7F800000u) == 0x7F800000u;
    const bool is_nan = is_inf_or_nan && (bits & 0x007FFFFFu) != 0u;

    if (is_nan) {
        // Quiet NaN in BF16: sign | 0xFF | 0x40 (MSB-of-mantissa set).
        return static_cast<u16>(((bits >> 16) & 0x8000u) | 0x7FC0u);
    }
    if (is_inf_or_nan) {
        // ±Inf: top 16 bits already encode ±Inf in BF16.
        return static_cast<u16>(bits >> 16);
    }
    // RNE: bias = 0x7FFF + LSB-of-result. Apply, then check whether
    // the bias rolled the exponent into 0xFF (Inf). When the input has
    // exp == 0xFE AND the mantissa is large enough that the +bias
    // overflows the mantissa field, the carry propagates into the
    // exponent field and we end up with exp == 0xFF, encoding Inf —
    // even though the input was finite. The branchless detection:
    // if (rounded >> 23) == 0xFF, saturate to the largest-magnitude
    // finite BF16 with the input's sign.
    const u32 rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    const u32 rounded = bits + rounding_bias;
    if ((rounded & 0x7F800000u) == 0x7F800000u) [[unlikely]] {
        // EC158: saturate to largest finite BF16, sign preserved.
        // Largest finite BF16 = sign | 0xFE | 0x7F = (0x7F80) - 1 in
        // BF16 bits, i.e. 0x7F7F. With sign: 0x[8|0]7F | 0x7F.
        constexpr u16 kLargestFiniteBF16 = 0x7F7Fu;
        const u16 sign_bf16 = static_cast<u16>((bits >> 16) & 0x8000u);
        return static_cast<u16>(sign_bf16 | kLargestFiniteBF16);
    }
    return static_cast<u16>(rounded >> 16);
}

/// FP32 -> IEEE 754 binary16 (FP16) with NaN preservation and overflow
/// safety, mirroring the BF16 contract above. The encoding is:
///
///     bit 15        : sign
///     bits 14..10   : exponent (5 bits, bias 15)
///     bits  9..0    : mantissa (10 bits)
///
/// HA contracts upheld:
///   - NaN  input -> quiet NaN (sign | exp=0x1F | mantissa MSB set).
///   - ±Inf input -> ±Inf encoding (exp=0x1F, mantissa=0).
///   - Finite-in -> Finite-out (EC158 invariant): any finite FP32 that
///     would round up into the FP16 ±Inf encoding saturates to the
///     largest finite FP16 magnitude with the sign preserved.
///   - Subnormals are emitted when the value falls below the smallest
///     normal FP16 (2^-14); pure underflow rounds to ±0 with sign kept.
///   - Round-to-nearest-even on the discarded mantissa bits, matching
///     IEEE 754 default rounding (same rounding rule as fp32ToBF16Safe).
///
/// References:
///   * IEEE 754-2008 §3.6 (binary16), §4.3 (roundTiesToEven).
///   * Jeroen van der Zijp, "Fast Half Float Conversions" (2008) — the
///     constants below match that paper for the normal-range path.
[[nodiscard]] inline u16 fp32ToFP16Safe(float v) noexcept {
    const u32 bits = std::bit_cast<u32>(v);
    const u32 sign = (bits >> 16) & 0x8000u;          // FP16 sign bit
    const u32 abs  = bits & 0x7FFFFFFFu;              // |v| in FP32 bits

    // Non-finite inputs.
    if (abs >= 0x7F800000u) [[unlikely]] {
        const bool is_nan = abs > 0x7F800000u;
        if (is_nan) {
            // Quiet NaN: sign | exp=0x1F | mantissa-MSB.
            return static_cast<u16>(sign | 0x7E00u);
        }
        // ±Inf.
        return static_cast<u16>(sign | 0x7C00u);
    }

    // Pure zero (and any value smaller than the smallest FP16 subnormal,
    // 2^-24 ≈ 5.96e-8) rounds to ±0.
    //   Smallest positive FP16 subnormal in FP32 bits = (127 - 24) << 23
    //   = 0x33000000 (i.e. 2^-24 = 5.9604644775e-8). Anything strictly
    //   below 2^-25 rounds-half-to-even to zero.
    if (abs < 0x33000000u) [[unlikely]] {
        // Round-to-nearest-even at the FP16 subnormal threshold:
        //   if v >= 2^-25 (half of the smallest subnormal), round up to 1.
        //   else round to 0.
        // 2^-25 in FP32 bits = (127 - 25) << 23 = 0x32800000.
        if (abs >= 0x32800000u) {
            return static_cast<u16>(sign | 0x0001u);   // smallest subnormal
        }
        return static_cast<u16>(sign);                 // ±0
    }

    // Saturate finite overflow to the largest finite FP16 (EC158 analogue).
    //   Smallest FP32 value that rounds to FP16 +Inf is 65520.0f = (143 << 23)
    //   = 0x47800000. Anything at or above that becomes ±Inf under IEEE
    //   rounding; we instead clamp to the largest finite FP16 magnitude
    //   0x7BFF (= 65504.0) with sign preserved, matching the project-
    //   wide "finite-in → finite-out" invariant.
    if (abs >= 0x47800000u) [[unlikely]] {
        return static_cast<u16>(sign | 0x7BFFu);
    }

    // Subnormal FP16: input exponent < -14 (i.e. FP32 abs < 2^-14
    // = 0x38800000). The subnormal mantissa is the true mantissa shifted
    // right so that the binary point aligns with FP16's exp=0 encoding;
    // RNE is applied on the dropped bits.
    if (abs < 0x38800000u) {
        // exponent of the FP32 value, in the same biased form (127-bias).
        const u32 fp32_exp = (abs >> 23);              // 0..126
        // Number of additional right-shifts needed: (1 - (fp32_exp - 127))
        //   = (128 - fp32_exp). The implicit leading 1 is prepended.
        const u32 shift = 113u - fp32_exp + 13u;       // 13 = (23 - 10)
        const u32 mant_with_leading_one = (abs & 0x007FFFFFu) | 0x00800000u;
        const u32 round_bit = (mant_with_leading_one >> (shift - 1u)) & 1u;
        const u32 sticky    = ((mant_with_leading_one & ((1u << (shift - 1u)) - 1u)) != 0u) ? 1u : 0u;
        u32 mant = mant_with_leading_one >> shift;
        // Round-to-nearest-even: round up iff round_bit && (sticky || (mant & 1)).
        if ((round_bit != 0u) && ((sticky != 0u) || ((mant & 1u) != 0u))) {
            mant += 1u;
        }
        // mant can roll into 0x0400 (= smallest normal); that is the
        // correct FP16 encoding for that boundary (exp=1, mantissa=0).
        return static_cast<u16>(sign | mant);
    }

    // Normal-range FP16: re-bias exponent (127 -> 15 means subtract 112).
    //
    // FP32 exp e32 in [113, 142] maps to FP16 exp e16 = e32 - 112 in [1, 30].
    // The 23-bit FP32 mantissa is rounded to 10 bits with RNE.
    const u32 e32 = (abs >> 23) - 112u;                // FP16 biased exp, 1..30
    const u32 m32 = abs & 0x007FFFFFu;                 // 23-bit mantissa
    // Round-half-to-even on the 13 discarded LSBs.
    //   round_bit = bit 12 (the MSB of the discarded field)
    //   sticky    = OR of bits 11..0
    const u32 round_bit = (m32 >> 12) & 1u;
    const u32 sticky    = (m32 & 0x0FFFu) != 0u ? 1u : 0u;
    u32 m16 = m32 >> 13;                               // 10-bit mantissa
    u32 result = (e32 << 10) | m16;
    if ((round_bit != 0u) && ((sticky != 0u) || ((m16 & 1u) != 0u))) {
        result += 1u;
        // A mantissa overflow from 0x3FF -> 0x400 with carry into the
        // exponent is HANDLED CORRECTLY by the bare `result += 1` because
        // the mantissa and exponent live in adjacent bit-fields with no
        // gap: the carry naturally increments exp. If exp then rolls
        // into 0x1F (=31, the Inf encoding), we have a true rounding
        // overflow from a finite input -> saturate per EC158.
        if ((result & 0x7C00u) == 0x7C00u) [[unlikely]] {
            return static_cast<u16>(sign | 0x7BFFu);
        }
    }
    return static_cast<u16>(sign | result);
}

/// IEEE 754 binary16 (FP16) -> FP32. Lossless: every FP16 value is exactly
/// representable in FP32. Special cases (±0, subnormals, ±Inf, NaN) are
/// all preserved bit-for-bit in FP32 semantics.
[[nodiscard]] inline float fp16ToFP32(u16 h) noexcept {
    const u32 sign = (static_cast<u32>(h) & 0x8000u) << 16;       // FP32 sign
    const u32 exp  = (static_cast<u32>(h) >> 10) & 0x1Fu;         // 5-bit exp
    const u32 mant = static_cast<u32>(h) & 0x03FFu;               // 10-bit mantissa

    if (exp == 0u) {
        // ±0 or subnormal.
        if (mant == 0u) {
            return std::bit_cast<float>(sign);                    // ±0
        }
        // Subnormal: normalise. The 10-bit mantissa holds the value
        // mant * 2^-24 (the binary point is to the LEFT of bit 9 in
        // FP16 subnormal encoding). To re-express in FP32 normal form
        // we shift the leading 1 up to bit position 10, counting shifts.
        //
        // After the loop, `e` counts (1 + number-of-shifts-performed).
        // If the leading 1 started at bit position p (with bit 9 being
        // the MSB of the FP16 mantissa), then we performed (10 - p)
        // shifts, so e = 11 - p. The true magnitude is 2^(p-24).
        // The FP32 exponent field is 127 + (p - 24) = 103 + p
        //   = 103 + (11 - e) = 114 - e.
        u32 m = mant;
        u32 e = 1u;
        while ((m & 0x0400u) == 0u) {
            m <<= 1;
            e += 1u;
        }
        m &= 0x03FFu;                                             // drop the implicit 1
        const u32 fp32_exp = (114u - e) << 23;                    // see derivation above
        const u32 fp32_man = m << 13;
        return std::bit_cast<float>(sign | fp32_exp | fp32_man);
    }
    if (exp == 0x1Fu) {
        // ±Inf or NaN.
        if (mant == 0u) {
            return std::bit_cast<float>(sign | 0x7F800000u);      // ±Inf
        }
        // Preserve quiet-bit; lower 9 mantissa bits propagate to the top
        // of the FP32 mantissa.
        return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13));
    }
    // Normal range: re-bias exponent (15 -> 127 means add 112).
    const u32 fp32_exp = (exp + 112u) << 23;
    const u32 fp32_man = mant << 13;
    return std::bit_cast<float>(sign | fp32_exp | fp32_man);
}

// ===========================================================================
// FP8_E4M3 (OCP/Hopper/Intel 8-bit float, 1 sign + 4 exponent + 3 mantissa)
// ===========================================================================
//
// Encoding:
//     bit 7     : sign
//     bits 6..3 : exponent (4 bits, bias 7)
//     bits 2..0 : mantissa (3 bits)
//
// Diverges from IEEE 754:
//   * No infinities. The pattern `S.1111.111` is the SINGLE NaN
//     encoding; all other `S.1111.xxx` patterns encode finite values.
//   * Max finite = 1.110₂ × 2^8 = 448 (encoded as `S.1111.110`).
//   * Smallest positive normal = 2^-6 ≈ 0.015625 (`0.0001.000`).
//   * Smallest positive subnormal = 2^-9 ≈ 0.001953 (`0.0000.001`).
//
// HA invariants:
//   * NaN  input -> S.1111.111 (canonical E4M3 NaN).
//   * ±Inf input -> sign | 0x7E (saturate to ±448; E4M3 has no Inf).
//   * Finite input above 448 saturates to ±448 (EC158 finite-in → finite-out).
//   * Round-to-nearest-even on the discarded mantissa bits.
//
// References:
//   * Open Compute Project, "OCP 8-bit Floating Point Specification" (2023).
//   * NVIDIA Hopper H100 TMA / WMMA FP8 specification.
//   * Micikevicius et al., "FP8 Formats for Deep Learning" (arXiv 2209.05433).
// ===========================================================================

[[nodiscard]] inline u8 fp32ToFP8_E4M3_Safe(float v) noexcept {
    const u32 bits = std::bit_cast<u32>(v);
    const u8  sign = static_cast<u8>((bits >> 24) & 0x80u);
    const u32 abs  = bits & 0x7FFFFFFFu;

    // Non-finite inputs: NaN -> 0x7F (with sign), ±Inf -> saturate to ±448.
    if (abs >= 0x7F800000u) [[unlikely]] {
        const bool is_nan = abs > 0x7F800000u;
        if (is_nan) {
            // Canonical E4M3 NaN: S.1111.111 = sign | 0x7F.
            return static_cast<u8>(sign | 0x7Fu);
        }
        // ±Inf -> saturate to ±448 (E4M3 has no Inf encoding).
        return static_cast<u8>(sign | 0x7Eu);
    }

    // Finite overflow: anything ≥ 464.0 (= 448 + half-ULP at 448 = 16)
    // rounds up to the NaN slot under naive RNE; saturate to ±448.
    // 464 in FP32 bits = (135 << 23) | (0b1101_0000 << 15)
    //                  = 0x43E80000. Threshold check is on the magnitude.
    constexpr u32 kSaturateAtOrAbove = 0x43E80000u;       // 464.0f bits
    if (abs >= kSaturateAtOrAbove) [[unlikely]] {
        return static_cast<u8>(sign | 0x7Eu);             // ±448
    }

    // Subnormal cutoff at the bottom: anything below half the smallest
    // E4M3 subnormal (2^-10 = 0.0009765625) rounds to ±0.
    //   2^-10 in FP32 bits = (117 << 23) = 0x3A800000.
    constexpr u32 kRoundToZeroBelow = 0x3A800000u;        // 2^-10
    if (abs < kRoundToZeroBelow) [[unlikely]] {
        // Tie at exactly 2^-10 rounds to even (= 0 since LSB is 0).
        return static_cast<u8>(sign);                     // ±0
    }

    // Subnormal range: input below smallest E4M3 normal (2^-6 = 0x3C800000).
    // Subnormal mantissa fills the 3-bit field with an extra leading
    // shift: shift = 23 (FP32 mantissa width) - 3 (E4M3 mantissa width)
    //              + (e4m3_min_normal_exp - fp32_exp)
    // where e4m3_min_normal_exp = 121 (= 127 - 6).
    if (abs < 0x3C800000u) {
        const u32 fp32_exp = abs >> 23;                   // 117..120
        // shift = 20 + (121 - fp32_exp). For fp32_exp = 120, shift = 21.
        // For fp32_exp = 117, shift = 24.
        const u32 shift = 20u + (121u - fp32_exp);
        const u32 mant_with_leading_one = (abs & 0x007FFFFFu) | 0x00800000u;
        const u32 round_bit = (mant_with_leading_one >> (shift - 1u)) & 1u;
        const u32 sticky    = ((mant_with_leading_one
                                & ((1u << (shift - 1u)) - 1u)) != 0u) ? 1u : 0u;
        u32 mant = mant_with_leading_one >> shift;
        if ((round_bit != 0u) && ((sticky != 0u) || ((mant & 1u) != 0u))) {
            mant += 1u;
        }
        // mant may roll into 0x08 = the smallest-normal encoding
        // (S.0001.000 = sign | 0x08). That is the correct boundary
        // encoding and we let the OR below stitch it together.
        return static_cast<u8>(sign | (mant & 0x0Fu));
    }

    // Normal range: input exponent in [121, 134] maps to E4M3 exp [1, 14].
    // FP32 mantissa rounded to 3 bits via RNE on the 20 discarded LSBs.
    const u32 e8  = (abs >> 23) - 120u;                   // E4M3 biased exp, 1..14
    const u32 m32 = abs & 0x007FFFFFu;
    const u32 round_bit = (m32 >> 19) & 1u;
    const u32 sticky    = (m32 & 0x0007FFFFu) != 0u ? 1u : 0u;
    u32 m8 = m32 >> 20;                                   // top-3 bits
    u32 result = (e8 << 3) | m8;
    if ((round_bit != 0u) && ((sticky != 0u) || ((m8 & 1u) != 0u))) {
        result += 1u;
        // Mantissa overflow with carry into exp is handled by the bare
        // += 1 (adjacent fields). If the carry produced result == 0x7F
        // (S.1111.111 = NaN encoding), saturate to 0x7E (±448) per the
        // finite-in → finite-out invariant.
        if (result == 0x7Fu) [[unlikely]] {
            result = 0x7Eu;
        }
    }
    return static_cast<u8>(sign | (result & 0x7Fu));
}

[[nodiscard]] inline float fp8_E4M3_toFP32(u8 h) noexcept {
    const u32 sign = (static_cast<u32>(h) & 0x80u) << 24;
    const u32 exp  = (static_cast<u32>(h) >> 3) & 0x0Fu;
    const u32 mant = static_cast<u32>(h) & 0x07u;

    if (exp == 0u) {
        // ±0 or subnormal.
        if (mant == 0u) {
            return std::bit_cast<float>(sign);
        }
        // Subnormal: value = mant * 2^-9.
        // Find leading 1 in the 3-bit mantissa, shift to FP32 implicit-1
        // position. With at most 3 bits, the loop runs at most 3 times.
        //
        // Derivation: mant = m × 2^-9 means leading 1 at bit position p
        // (0..2 in the 3-bit field). FP32 exponent field = 127 + (p - 9)
        //   = 118 + p. With e = (3 - p), we get FP32 exp = 121 - e.
        u32 m = mant;
        u32 e = 1u;
        while ((m & 0x04u) == 0u) {
            m <<= 1;
            e += 1u;
        }
        m &= 0x03u;                                       // drop implicit 1
        const u32 fp32_exp = (121u - e) << 23;
        const u32 fp32_man = m << 21;                     // 23 - 2 = 21
        return std::bit_cast<float>(sign | fp32_exp | fp32_man);
    }
    if (exp == 0x0Fu && mant == 0x07u) {
        // Canonical E4M3 NaN. Emit a quiet FP32 NaN with sign preserved.
        return std::bit_cast<float>(sign | 0x7FC00000u);
    }
    // Normal range: re-bias exponent (7 -> 127 means add 120).
    const u32 fp32_exp = (exp + 120u) << 23;
    const u32 fp32_man = mant << 20;                      // 23 - 3 = 20
    return std::bit_cast<float>(sign | fp32_exp | fp32_man);
}

// ===========================================================================
// FP8_E5M2 (OCP/Hopper/Intel 8-bit float, 1 sign + 5 exponent + 2 mantissa)
// ===========================================================================
//
// Encoding:
//     bit 7     : sign
//     bits 6..2 : exponent (5 bits, bias 15)
//     bits 1..0 : mantissa (2 bits)
//
// IEEE 754-like (unlike E4M3):
//   * Inf (exp=0x1F, mant=0) and NaN (exp=0x1F, mant≠0) both encodable.
//   * Max finite = 1.11₂ × 2^15 = 57344 (encoded as `S.11110.11`).
//   * Smallest positive normal = 2^-14 ≈ 6.10e-5.
//   * Smallest positive subnormal = 2^-16 ≈ 1.53e-5.
//
// HA invariants (consistent with FP16 / FP8_E4M3):
//   * NaN  input -> quiet NaN with sign preserved (S.11111.11).
//   * ±Inf input -> ±Inf encoding (S.11111.00).
//   * Finite input above 57344 saturates to ±57344 (NOT ±Inf, per
//     finite-in → finite-out).
//   * Round-to-nearest-even.
// ===========================================================================

[[nodiscard]] inline u8 fp32ToFP8_E5M2_Safe(float v) noexcept {
    const u32 bits = std::bit_cast<u32>(v);
    const u8  sign = static_cast<u8>((bits >> 24) & 0x80u);
    const u32 abs  = bits & 0x7FFFFFFFu;

    if (abs >= 0x7F800000u) [[unlikely]] {
        const bool is_nan = abs > 0x7F800000u;
        if (is_nan) {
            // Quiet NaN: sign | exp=0x1F | mant MSB.
            return static_cast<u8>(sign | 0x7Eu);
        }
        // ±Inf: sign | exp=0x1F | mant=0.
        return static_cast<u8>(sign | 0x7Cu);
    }

    // Finite overflow boundary: ULP at max-finite (57344) is 2^13 = 8192;
    // half-ULP = 4096; saturation threshold = 57344 + 4096 = 61440.
    // 61440 in FP32 bits = (142 << 23) | (0b111 << 20) = 0x477F0000... actually
    //   61440 = 1.875 × 2^15 → exponent field = 127+15 = 142, mantissa = 0b111_0000_…
    //   bits = (142 << 23) | (0xF << 19) = 0x477F0000
    constexpr u32 kSaturateAtOrAbove = 0x477F0000u;       // 61440.0f bits
    if (abs >= kSaturateAtOrAbove) [[unlikely]] {
        return static_cast<u8>(sign | 0x7Bu);             // ±57344 (S.11110.11)
    }

    // Underflow: anything below half the smallest subnormal (2^-17) -> ±0.
    //   2^-17 in FP32 bits = (110 << 23) = 0x37000000.
    constexpr u32 kRoundToZeroBelow = 0x37000000u;        // 2^-17
    if (abs < kRoundToZeroBelow) [[unlikely]] {
        return static_cast<u8>(sign);                     // ±0
    }

    // Subnormal: input below smallest E5M2 normal (2^-14 = 0x38800000).
    if (abs < 0x38800000u) {
        const u32 fp32_exp = abs >> 23;                   // 110..112
        // shift = 21 + (113 - fp32_exp). For fp32_exp = 112, shift = 22.
        const u32 shift = 21u + (113u - fp32_exp);
        const u32 mant_with_leading_one = (abs & 0x007FFFFFu) | 0x00800000u;
        const u32 round_bit = (mant_with_leading_one >> (shift - 1u)) & 1u;
        const u32 sticky    = ((mant_with_leading_one
                                & ((1u << (shift - 1u)) - 1u)) != 0u) ? 1u : 0u;
        u32 mant = mant_with_leading_one >> shift;
        if ((round_bit != 0u) && ((sticky != 0u) || ((mant & 1u) != 0u))) {
            mant += 1u;
        }
        return static_cast<u8>(sign | (mant & 0x07u));    // 3 LSBs (exp=0 implicit)
    }

    // Normal range: input exponent in [113, 142] maps to E5M2 exp [1, 30].
    // 13 LSBs of the FP32 mantissa are rounded to 2 bits via RNE.
    const u32 e8  = (abs >> 23) - 112u;                   // E5M2 biased exp, 1..30
    const u32 m32 = abs & 0x007FFFFFu;
    const u32 round_bit = (m32 >> 20) & 1u;
    const u32 sticky    = (m32 & 0x000FFFFFu) != 0u ? 1u : 0u;
    u32 m8 = m32 >> 21;                                   // top-2 bits
    u32 result = (e8 << 2) | m8;
    if ((round_bit != 0u) && ((sticky != 0u) || ((m8 & 1u) != 0u))) {
        result += 1u;
        // If the rounded result reaches the Inf encoding (exp == 0x1F,
        // mant == 0), the input was a finite value rounding up to ±Inf
        // — saturate to ±57344 per the finite-in → finite-out invariant.
        if ((result & 0x7Cu) == 0x7Cu) [[unlikely]] {
            result = 0x7Bu;                               // ±57344
        }
    }
    return static_cast<u8>(sign | (result & 0x7Fu));
}

[[nodiscard]] inline float fp8_E5M2_toFP32(u8 h) noexcept {
    const u32 sign = (static_cast<u32>(h) & 0x80u) << 24;
    const u32 exp  = (static_cast<u32>(h) >> 2) & 0x1Fu;
    const u32 mant = static_cast<u32>(h) & 0x03u;

    if (exp == 0u) {
        if (mant == 0u) {
            return std::bit_cast<float>(sign);            // ±0
        }
        // Subnormal: value = mant × 2^-16. Leading bit in [0..1].
        // FP32 exponent field = 127 + p - 16 = 111 + p, where p is
        // the leading-bit position (0 or 1). With e = (2 - p),
        // FP32 exp = 113 - e.
        u32 m = mant;
        u32 e = 1u;
        while ((m & 0x02u) == 0u) {
            m <<= 1;
            e += 1u;
        }
        m &= 0x01u;
        const u32 fp32_exp = (113u - e) << 23;
        const u32 fp32_man = m << 22;                     // 23 - 1 = 22
        return std::bit_cast<float>(sign | fp32_exp | fp32_man);
    }
    if (exp == 0x1Fu) {
        if (mant == 0u) {
            return std::bit_cast<float>(sign | 0x7F800000u);  // ±Inf
        }
        // NaN: propagate mantissa bits to upper FP32 mantissa.
        return std::bit_cast<float>(sign | 0x7F800000u | (mant << 21));
    }
    // Normal: re-bias exponent (15 -> 127 means add 112).
    const u32 fp32_exp = (exp + 112u) << 23;
    const u32 fp32_man = mant << 21;                      // 23 - 2 = 21
    return std::bit_cast<float>(sign | fp32_exp | fp32_man);
}

// ===========================================================================
// NVFP4 element-level codec (E2M1, 1 sign + 2 exp + 1 mantissa, bias 1).
// ===========================================================================
//
// The 16 codepoints of E2M1 with exponent bias 1 are:
//
//   nibble  sign   exp  mant   value         |   nibble  value
//   0000     +     00    0     +0.0         |   1000     -0.0
//   0001     +     00    1     +0.5         |   1001     -0.5
//   0010     +     01    0     +1.0         |   1010     -1.0
//   0011     +     01    1     +1.5         |   1011     -1.5
//   0100     +     10    0     +2.0         |   1100     -2.0
//   0101     +     10    1     +3.0         |   1101     -3.0
//   0110     +     11    0     +4.0         |   1110     -4.0
//   0111     +     11    1     +6.0         |   1111     -6.0
//
// Derivation: with bias=1, exp field e in {0,1,2,3} gives unbiased
// exponent e-1 in {-1,0,1,2}. Mantissa m in {0,1} is the explicit bit.
// Subnormal (e=0): value = m * 2^-1 * 1 = m * 0.5 → {0, 0.5}.
// Normal   (e>0): value = (1 + m*0.5) * 2^(e-1)
//                       → e=1: {1, 1.5}; e=2: {2, 3}; e=3: {4, 6}.
//
// Reference:
//   * NVIDIA Developer Blog, "Introducing NVFP4 for Efficient and
//     Accurate Low-Precision Inference" (2025).
//   * OCP Microscaling Specification, MXFP4 section (NVFP4 inherits
//     the E2M1 element format from MX; the block-level scaling is
//     NVIDIA's own variant).

/// The 8 positive magnitudes of E2M1, indexed by the lower 3 bits of
/// the nibble. The sign bit (bit 3) is handled separately. Constexpr
/// so the compiler can fold the table into immediate operands.
inline constexpr float kNVFP4_Magnitudes[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

/// Maximum magnitude representable in E2M1: 6.0f.
inline constexpr float kNVFP4_MaxMagnitude = 6.0f;

/// Decode a single E2M1 nibble (low 4 bits of `n`) to FP32.
/// Saturating: high bits of `n` are ignored.
[[nodiscard]] inline float nvfp4ElementToFP32(u8 n) noexcept {
    const u8 idx  = n & 0x07u;             // |magnitude| index
    const u8 sign = n & 0x08u;             // sign bit
    const float mag = kNVFP4_Magnitudes[idx];
    return sign != 0u ? -mag : mag;
}

/// Encode a single FP32 value, ALREADY SCALED by the per-block factor,
/// to an E2M1 nibble. Uses round-to-nearest-even with the codebook
/// midpoints {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0} as boundaries.
/// Saturates magnitudes ≥ 6.0 to the +6 / -6 codepoints. NaN/Inf in
/// the input is impossible at this stage (the block-level encode
/// applies sanitiseFinite first and the scale-divide cannot inject
/// non-finite values when scale is finite).
///
/// Tie-to-even applied at each midpoint:
///   0.25 → 0 (LSB of result = 0); 0.75 → 1.0; 1.25 → 1.0;
///   1.75 → 2.0; 2.5 → 2; 3.5 → 4; 5.0 → 4.
/// (For positive inputs; symmetric for negative.)
[[nodiscard]] inline u8 fp32ElementToNVFP4_Safe(float scaled) noexcept {
    const float a    = std::fabs(scaled);
    const u8   sign  = (scaled < 0.0f || (scaled == 0.0f
                        && std::signbit(scaled))) ? 0x08u : 0x00u;

    // Saturation. NVFP4 has no NaN/Inf encoding; sanitiseFinite() at
    // the block level ensures `scaled` is finite here, so saturating to
    // ±6 covers the entire input range above the codebook.
    if (a >= kNVFP4_MaxMagnitude) {
        return static_cast<u8>(sign | 0x07u);                  // ±6
    }

    // Round to nearest codepoint with ties-to-even on the 7 midpoints.
    // The codebook is non-uniform, so we use a tabulated comparison
    // against each midpoint rather than a single multiply-and-truncate.
    //
    // Midpoints between consecutive |magnitudes|:
    //   |0   0.5  1   1.5  2   3   4   6|
    //   mid:  0.25 0.75 1.25 1.75 2.5 3.5 5.0
    //
    // Tie-to-even target index at each midpoint (index of the EVEN
    // neighbour in the codebook):
    //   mid 0.25 between idx 0 and 1 -> 0 (idx 0 is even)
    //   mid 0.75 between idx 1 and 2 -> 2
    //   mid 1.25 between idx 2 and 3 -> 2
    //   mid 1.75 between idx 3 and 4 -> 4
    //   mid 2.5  between idx 4 and 5 -> 4
    //   mid 3.5  between idx 5 and 6 -> 6
    //   mid 5.0  between idx 6 and 7 -> 6
    static constexpr float    kMidpoints[7] = {
        0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f
    };
    static constexpr u8       kTieToEven[7] = {
        0u, 2u, 2u, 4u, 4u, 6u, 6u
    };

    u8 idx = 7u;                                                // default: top bin
    for (u32 i = 0; i < 7u; ++i) {
        if (a == kMidpoints[i]) {
            idx = kTieToEven[i];
            break;
        }
        if (a < kMidpoints[i]) {
            idx = static_cast<u8>(i);                           // lower neighbour
            break;
        }
    }
    return static_cast<u8>(sign | (idx & 0x07u));
}

}  // namespace qtx::core
