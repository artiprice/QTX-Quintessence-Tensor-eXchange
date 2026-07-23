<div align="center">

# QTX — Quintessence Tensor eXchange

**A high-assurance C++23 tiered memory engine that dynamically quantizes
"sleeping" tensors on the fly — so consumer hardware can hold far more
concurrent agents than RAM should allow.**

*Hardware-sympathetic. Zero syscalls on the hot path. Pluggable codec for
any quantization format, including your own.*

[![tests](https://img.shields.io/badge/tests-389%2F389-success)](#testing)
[![QA gates](https://img.shields.io/badge/QA%20gates-release%20%C2%B7%20strict%20%C2%B7%20asan%20%C2%B7%20tsan-success)](#testing)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)](#build--test)
[![license](https://img.shields.io/badge/license-AGPL%20v3.0-blue)](LICENSE)

</div>

---

## The problem

When you run many models or agents on one machine, most of their context
sits idle most of the time. Keeping every tensor in full FP32 burns RAM
you don't have; quantizing everything up front destroys precision on the
tensors you're actively using. Garbage-collected orchestration runtimes
(Python, Java) make it worse — just *instantiating* a thousand idle agents
can devour gigabytes before a single inference runs.

## What QTX does

QTX keeps a **hot tier** in FP32 and a **cold tier** of compressed
tensors. When the hot tier fills, the least-recently-used tensor is
automatically evicted to cold — **compressed on the way down** — and
transparently **decompressed on the way back up** the instant something
touches it. Your code only ever asks for an FP32 view; the tiering,
eviction, compression and promotion happen underneath.

```
   acquireFP32(handle)                       release(handle)
          │                                         │
          ▼                                         ▼
   ┌──────────────┐   hot tier full   ┌──────────────────────────┐
   │   HOT (FP32) │ ────────────────► │  evict LRU → COMPRESS →   │
   │              │ ◄──────────────── │  COLD tier (quantized)    │
   └──────────────┘   touched again   └──────────────────────────┘
          ▲          DECOMPRESS → promote
          │
     your code sees FP32, always
```

The cold-tier format is **not hard-wired**. It's chosen through a **codec
adapter** — at compile time for zero overhead, at runtime from a flag, or
by plugging in a **third-party quantizer** QTX knows nothing about.

```cpp
// Compile-time: zero-overhead, format fixed at build.
using Bridge = tiered::TieredArenaBridge<HotArena, ColdArena, 4096, codec::Int4ColdCodec>;

// Runtime: pick from a CLI flag.
RuntimeColdCodec<>::setAdapter(codec::adapterByName(argv_flag));  // "int4", "fp16", ...

// Foreign codec: bring your own (an AWQ/GPTQ kernel, anything).
codec::ForeignCodecAdapter awq{&awq_bound, &awq_compress, &awq_decompress, "awq"};
RuntimeColdCodec<>::setAdapter(&awq);
```

The bridge touches the codec in exactly **one seam**, so a new cold format
— or an external quantizer — never touches the concurrency-sensitive hot
path.

---

## Status (honest)

The **engine and the adapter work and are tested end-to-end**: the dynamic
FP32 ↔ quantized round-trip through real eviction/promotion passes under
AddressSanitizer *and* ThreadSanitizer (the bridge is concurrent —
lock-free hot path, single mutex for structural changes).

| Area | State |
|---|---|
| Tiered dynamic quantization (hot/cold, auto-evict, auto-promote) | ✅ working, tested |
| Pluggable cold-codec adapter (compile-time / runtime / foreign) | ✅ working, tested |
| Micro-tenant packing (32 small agents per 4 KiB page) | ✅ working, tested |
| Lock-free SPSC queue + generational arena + O(1) selector | ✅ working, tested |
| C-ABI safety boundary (magic + install-token + bounds check) | ✅ working, tested |
| 25 quantization formats (FP / INT / K-Quant / I-Quant) | ✅ decode is bit-exact ggml-compatible |
| Quantization **encoders** | ⚠️ single-pass *reference quality* — correct & ggml-decodable, not yet perplexity-tuned vs offline encoders |
| Drop-in `ggml` backend C-ABI (`ggml_backend_qtx_*`) | 🚧 designed, not yet implemented (Roadmap) |
| Calibration formats (GPTQ, AWQ, EXL2, BitNet) | 🚧 planned (Roadmap) |

Quantization is **lossy** (that's the point of the cold tier): INT8 keeps
the signal within a few percent; INT4 trades more error for ~2× the
density. The bridge guarantees the *round-trip mechanics* are correct and
race-free, not that compression is free of error.

> **Claude for Open Source — Ecosystem Impact Track.** QTX is applying to
> the [Claude for Open Source](https://claude.com/contact-sales/claude-for-oss)
> program via its **Ecosystem Impact Track** — the path for projects that
> serve the software ecosystem rather than ones already at the
> 5,000-star / 1M-download maintainer threshold.
>
> **The ecosystem case:** the local-inference stack (`llama.cpp`, `ggml`,
> and everything built on them) is bottlenecked by memory when running
> many models or agents on one machine. QTX attacks that directly — a
> tiered engine that dynamically quantizes idle tensors to fit **2.35×+**
> more concurrent agents in the same RAM, with a decoder layer that is
> already **bit-exact compatible with ggml's 25 formats**, so it slots
> into that ecosystem rather than competing with it. Acceptance is at
> Anthropic's discretion and not guaranteed; if approved, the Claude Max
> grant goes straight to the Roadmap below — the `ggml` backend C-ABI
> (the integration that makes the impact real), SIMD-accelerated
> encoders, the calibration formats, and contributor-grade docs.

---

## Architecture

**FractalArena** — a contiguous, pre-allocated memory die. After init it
does **zero dynamic allocations** on the hot path; fixed-size,
128-byte-aligned slots eliminate external fragmentation. (Micro-tenant
sub-slabbing packs up to 32 lightweight agents into one 4 KiB page.)

**AxiomSelector** — an O(1) slot picker using CPU bitscan intrinsics
(`tzcnt` / `std::countr_zero`) over 64-bit occupancy bitmasks, with a
thread-local stochastic start index to avoid cache-line contention under
parallel load.

**SPSCRingBuffer** — a wait-free single-producer/single-consumer queue
built on HFT patterns, with `alignas(128)` read/write heads to kill false
sharing.

**Quantizer** — a branchless, block-based compression engine. On entry it
installs a `FtzDazGuard` (flush-to-zero / denormals-are-zero) to avoid
denormal stalls, and sanitizes `NaN`/`Inf` lanes to `0.0` before scaling.
AVX-512 fast paths (with an AVX2 fallback and a portable scalar oracle)
for the linear formats; scalar reference encoders for the K-/I-Quants.

**Safety boundary** — the high-assurance C++23 core is separated from the
untrusted C-ABI by a chain of trust: a magic signature, a per-instance
`install_token` (u64 entropy, defeats magic-only forgery), and an address
bounds check.

A hard rule across the codebase: the `GenID` C-ABI never changes its bit
layout (pinned by `static_assert`), so the binary interface stays stable
as formats are added.

---

## Measured numbers

*Intel® Xeon® @ 2.80 GHz, single vCPU sandbox, GCC 13.3, `-O3 -DNDEBUG
-march=native`. Re-run them yourself with `build_and_test.sh bench`.*

### Memory density — OOM survival

Physical budget **544 KiB** (256 KiB hot + 288 KiB cold). A naive FP32
allocator runs out at 64 agents. QTX hosts **320 concurrent agents** at
full saturation with **0 verification failures**:

| Active / Dormant | Memory used | Bytes / agent | Density |
|---|---|---|---|
| 64 / 0 (hot only) | 256 KiB | 4096 | 1.00× |
| 64 / 64 | 328 KiB | 2624 | **1.56×** |
| 64 / 192 | 472 KiB | 1888 | **2.17×** |
| 64 / 256 (OOM edge) | 544 KiB | 1741 | **2.35×** |

Micro-tenant packing pushes small agents (128 B contexts) to a **32× density** (128 tenants in 16 KiB vs 512 KiB naive), 0 leaks.

### Latency / throughput

| Operation | Latency | Throughput |
|---|---|---|
| AxiomSelector acquire+release | 18.5 ns | 54 M-op/s |
| FractalArena allocate+release | 37.9 ns | 26 M-op/s |
| SPSC push+pop (same thread) | 2.4 ns | 424 M-op/s |
| TieredBridge full lifecycle (create→acquire→release→destroy) | 170 ns | 5.9 M-op/s |
| FP32 → INT8 compress | — | ~8 GB/s |
| INT8 → FP32 decompress | — | ~45 GB/s |

*(Throughput depends on format and machine; the build runs the full
benchmark and prints current numbers.)*

---

## Build & test

Zero runtime dependencies. Needs a C++23 compiler (GCC 13+, Clang 16+,
MSVC 19.34+).

The full ggml repository (used only by the optional real-ggml demo) is a
**git submodule** under `third_party/ggml_src/`. The test suite does **not**
need it — a small set of ggml headers + a test shim is committed directly
under `third_party/ggml/` (MIT, used to validate bit-exact decode), so a
plain clone builds and tests out of the box.

```bash
git clone <repo> qtx && cd qtx          # tests build immediately
# optional — only for the real-ggml demo:
git submodule update --init --depth 1 third_party/ggml_src

bash build_and_test.sh release   # build + run 389 tests
bash build_and_test.sh strict    # -Werror, -Wconversion, max warnings
bash build_and_test.sh asan      # AddressSanitizer + UBSan
bash build_and_test.sh tsan      # ThreadSanitizer — proves the concurrency
bash build_and_test.sh bench     # OOM-survival + perf benchmarks
bash build_and_test.sh all       # strict + asan + tsan + bench
```

Windows: `.\build_and_test.ps1`. CMake (all platforms):

```bash
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j
./qtx_tests
```

> **Note:** QTX is currently a **library + test/benchmark binaries**, not
> a standalone CLI tool. The runnable artifacts today are `qtx_tests`,
> `oom_survival`, and the perf benchmarks. A `qtx` command-line front-end
> (`--cold-format int4`, round-trip/inspect modes) is a small, planned
> addition on top of the existing codec-adapter registry.

---

## 30-second API tour

```cpp
#include "qtx/tiered/tiered_bridge.hpp"
#include "qtx/arena/fractal_arena.hpp"
using namespace qtx;

using HotArena  = arena::FractalArena<64, 4096>;    // 64 slots × 4 KiB FP32
using ColdArena = arena::FractalArena<512, 1152>;   // 512 compressed slots
using Bridge    = tiered::TieredArenaBridge<HotArena, ColdArena>;  // default: INT8 cold

auto hot = std::make_unique<HotArena>();
auto cold = std::make_unique<ColdArena>();
Bridge bridge(hot.get(), cold.get());

auto h = bridge.createTenant();
auto view = bridge.acquireFP32(h);   // writable FP32 span
fill(view, my_weights);
bridge.release(h);                   // now eligible for eviction
// ... pressure builds, h is evicted to cold (compressed) ...
auto back = bridge.acquireFP32(h);   // transparently decompressed to FP32
```

`Bridge::coldCodecName()` reports the active format;
`Bridge::coldCompressedBytes()` reports its per-tenant footprint.

### Formats usable as cold codecs

- **Float:** FP16, BF16, FP8 (E4M3 / E5M2), NVFP4, MXFP4
- **Integer:** INT8, INT4
- **GGML legacy:** Q4_1, Q5_0, Q5_1
- **K-Quants:** Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- **I-Quants:** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS

All decoders are bit-for-bit ggml-compatible (IQ4_NL's encoder is too), so
QTX consumes tensors any ggml-based tool produced.

---

## Design principle

A strict, never-violated priority order — when goals conflict, the higher
wins:

**Safety (no UB) > latency > data locality > template machinery > raw throughput.**

Every codec is `noexcept`, sanitizes non-finite inputs, rejects corrupted
scales on decode, and uses `std::bit_cast` over type-punning. The hot path
allocates nothing. The whole suite is clean under UBSan, ASan, and TSan.

## Testing

389 tests across four independent QA gates:

| Gate | Proves |
|---|---|
| `release` | functional correctness |
| `strict` | `-Werror` + `-Wconversion -Wshadow -Wold-style-cast -Wdouble-promotion` … all clean |
| `asan` | no memory errors, no UB |
| `tsan` | no data races in the concurrent tiering engine |

Groups: Quant 242, GenID 26, Arena 25, Tiered 21, Bridge 21, Selector 15,
SPSC 14, Boundary 13, ColdCodec 9, Integration 3.

## Roadmap

1. **SIMD-accelerate the codebook encoders** (AVX-512 / NEON in-register
   LUT) — the I-Quant encoders currently brute-force the codebook search.
2. **Drop-in `ggml` backend C-ABI** so QTX is a pluggable KV-cache /
   tensor buffer behind `llama.cpp` without changing call sites.
3. **Calibration formats: GPTQ, AWQ, EXL2, BitNet** — needs a
   calibration-data input on compress (per-channel salience / Hessian).
4. **Perplexity-tune the single-pass encoders** to match offline ggml
   quality, plus real `.gguf` round-trip integration tests and a `docs/`
   set for contributors.
5. **Virtualized paged KV-cache** — a page-table of GenID pages so an
   agent's context can grow to gigabytes with zero external fragmentation.

The codec adapter is the natural extension point — contributions welcome.

## License

QTX is available under a dual-license model:

- **Open source:** [GNU AGPL v3.0](LICENSE) (`AGPL-3.0-only`).
- **Commercial:** a separate commercial license may be available for
  proprietary products and use cases where AGPL compliance is not
  suitable.

See [LICENSING.md](LICENSING.md) for details and contact instructions.
No commercial rights are granted without a separate written agreement.

© 2026 Mikhail Melik-Kazarian. Third-party ggml materials under
`third_party/` remain under the MIT License.
