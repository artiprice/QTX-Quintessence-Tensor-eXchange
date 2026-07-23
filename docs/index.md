---
layout: article
title: I built a memory tier that quantizes tensors while they sleep
description: QTX keeps active tensors in FP32 and compresses dormant state under memory pressure. This is how the project—and its trade-offs—came together.
---

QTX started for a fairly ordinary reason: there was an application
deadline.

I found [Anthropic's Claude for Open Source
program](https://claude.com/contact-sales/claude-for-oss) and wanted to
submit something through its ecosystem path. I spent a long time looking
for a project idea before settling on memory orchestration for local AI
agents. At first, I did not attach much importance to it. It was simply
the topic I had chosen.

Then the project became more interesting than the application.

The more I worked on its memory model, the more the parts started to fit
together. Each new constraint made the design harder, but also made the
system more coherent. Somewhere in that process I stopped treating QTX
as an entry for a program and became genuinely attached to it.

I began around May 2026 and worked on it intensely for somewhere between
one and three weeks. I then left it alone for a while. Shortly before the
application window I was targeting ended, I returned to it. I do not
remember one grand reason for returning. It was closer to the usual
question that keeps unfinished projects alive: *what if?*

QTX is not affiliated with or endorsed by Anthropic. The program was the
reason I began; it is not the reason I finished.

## The problem

Running one model locally is mostly a compute problem. Running many
models or agents on the same machine quickly becomes a memory problem.

An agent may spend most of its lifetime waiting for input, a tool call,
another agent, or a scheduler. Its state still occupies memory at full
precision. With enough inactive agents, RAM fills with tensors that are
not currently doing useful work.

QTX explores a simple policy:

- keep active tensors in a hot FP32 tier;
- when that tier fills, select an inactive tensor and quantize it into a
  cold tier;
- if the tensor becomes active again, restore it transparently.

The caller only asks for an FP32 view:

```cpp
auto handle = bridge.createTenant();
auto view = bridge.acquireFP32(handle);
fill(view, data);
bridge.release(handle);

// Memory pressure may move this tenant into the quantized cold tier.

auto restored = bridge.acquireFP32(handle);
```

Behind this API are two fixed-capacity arenas. The hot arena stores FP32
slots. The cold arena stores compressed slots. When the hot arena is
full, QTX finds an unpinned least-recently-used tenant, compresses its
state, and releases the hot slot. Acquiring the tenant later reverses
the operation.

Structural changes use one mutex. The common acquire and release path
uses atomics and does not allocate.

QTX is currently a C++23 library with tests and benchmarks. It is not a
finished model server or CLI. A drop-in ggml backend that could place it
behind llama.cpp is still roadmap work.

## Why not quantize everything in advance?

Offline quantization is already a good answer for model weights. QTX is
aimed at state whose activity changes while a system is running.

Keeping every tenant in FP32 wastes memory when most of them are
dormant. Keeping every tenant permanently quantized adds reconstruction
error to the active working set as well. A hot/cold design pays the
compression cost only under memory pressure and keeps active state
uncompressed.

This is still a trade-off, not free memory. Quantization is lossy.
Promotion has a cost. A workload with poor locality can repeatedly evict
and restore the same tensors. QTX does not remove those costs; it turns
them into an explicit memory policy.

## The part of the design I like most

The memory mechanism is the part of QTX I am happiest with.

The underlying `FractalArena` is preallocated and divided into
fixed-size, 128-byte-aligned slots. Once an arena exists, allocation
from it does not call the system allocator. Slot selection uses
occupancy bitmaps and a count-trailing-zero operation instead of
scanning every slot.

A handle contains both a slot index and a generation. Reusing a slot
increments its generation, so an old handle cannot silently refer to a
new tenant that happens to occupy the same address. The handle layout is
pinned by compile-time assertions because it also crosses the C ABI.

The project also includes an SPSC queue whose producer and consumer
state are separated across cache lines, and a micro-tenant allocator
that packs small states into sub-slots.

None of these techniques is individually new. What I value is how they
fit together: fixed storage, stale-handle detection, eviction,
promotion, quantization, and the unsafe C boundary all have to preserve
the same invariants.

I tried to follow high-assurance principles without treating performance
as an afterthought. My priority order was:

```text
safety > latency > data locality > template machinery > raw throughput
```

That means avoiding undefined behavior, keeping allocation out of the
hot path, validating data at the C boundary, sanitizing non-finite
quantizer inputs, using `std::bit_cast` instead of type punning, and
testing concurrent paths under ThreadSanitizer.

It does not mean that the implementation is beyond criticism. It means
that safety and performance were design inputs rather than cleanup
tasks.

## The codec is replaceable

I did not want the tiering engine to know the details of every
quantization format. Compression is therefore behind one codec seam:

```cpp
using Bridge = qtx::tiered::TieredArenaBridge<
    HotArena,
    ColdArena,
    4096,
    qtx::codec::Int4ColdCodec>;
```

A codec provides a size bound, compression, and decompression. It can be
selected at compile time, selected from the runtime registry, or supplied
through an adapter for an external implementation.

QTX currently includes 25 FP, integer, K-Quant, I-Quant, and legacy ggml
formats. Decoder compatibility is the stronger part: the decoders are
tested against ggml wire layouts.

I spent a disproportionate amount of time on the quantization formats.
That was probably the most frustrating part of development, although I
would not call it one specific bug. Not every encoder is ideal, and that
is not the central claim of the project. Most of the encoders are
single-pass reference implementations. They produce decodable output,
but still need model-level perplexity work before they should be
compared with mature offline quantizers.

The formats are useful, but QTX is primarily about deciding when and
where tensor state should live.

## Small tenants are a different problem

Compressing a 128-byte state inside its own 4 KiB slot does not solve
much because internal fragmentation dominates.

QTX therefore has a separate micro-tenant path. It divides one 4 KiB
slot into 32 cells and tracks their ownership with one atomic 32-bit
bitmap. Clearing the final occupied bit releases the parent slot.

In the included benchmark, 128 tenants with 128 bytes of state occupy
16 KiB rather than 512 KiB with one 4 KiB page per tenant. This is 32x
packing, not 32x quantization, and I think the distinction is important.

Micro-tenants currently remain in the hot tier. They are not migrated
to cold storage. Combining sub-slot ownership with migration would add
another state machine, and I did not want to pretend that complexity was
already solved.

## What the 2.35x number means

The repository includes a synthetic OOM-survival benchmark with a fixed
physical budget:

- 256 KiB of hot storage;
- 288 KiB of cold storage;
- 4 KiB of FP32 state per full-size tenant.

A naive FP32 allocator fits 64 tenants. In the benchmark configuration,
QTX fits 64 active and 256 dormant tenants in the same 544 KiB budget:
320 tenants total, or 1,741 bytes per tenant on average. That is the
source of the reported 2.35x density figure.

It is not a claim that arbitrary LLM applications will use 2.35x less
total RAM. Results depend on tensor sizes, codec choice, the size and
stability of the hot set, access locality, and the amount of error a
workload can tolerate.

The published microbenchmarks were measured in a single-vCPU virtualized
Intel Xeon environment using GCC 13.3 with:

```text
-O3 -DNDEBUG -march=native
```

The benchmark code is in the repository because I would rather see
results from other machines than present one environment as universal.

## A bug that sanitizers could not find

One early version of the micro-tenant allocator used compare-and-swap
incorrectly. Instead of packing 32 tenants into a shared slot, it opened
a new slot for each tenant.

The program was race-free. The algorithm was simply wrong.
ThreadSanitizer could not tell me that. A packing-invariant test did.

That failure influenced how I think about high-assurance code.
Sanitizers can detect important classes of errors, but they cannot prove
that a state machine implements the policy you intended. Some properties
need direct invariant tests.

The repository now contains 389 tests covering quantization, arena
boundaries, stale handles, eviction and promotion, codec adapters,
micro-tenant ownership, the SPSC queue, and the ggml-facing boundary.
The intended gates are a release build, strict compiler warnings,
AddressSanitizer/UBSan, and ThreadSanitizer.

## How AI was used

AI was used extensively during QTX development, including code
generation.

I made the architectural decisions and chose the invariants, trade-offs,
and direction of the project. AI often acted as connective tissue
between ideas: turning a design into a first implementation, helping
enumerate edge cases, and accelerating repetitive work.

I do not think “AI-generated” is either a quality guarantee or a useful
substitute for reviewing the result. Generated code was treated as code
that still had to survive the same boundaries, compiler warnings,
tests, sanitizers, compatibility checks, and benchmarks as everything
else.

Quality was always the priority. The repository is open so that this
claim does not have to be accepted on trust.

## What is unfinished

The largest missing part is practical integration.

QTX does not yet provide the planned drop-in
`ggml_backend_qtx_*` backend. There is a ggml-compatible bridge and
format-validation code, but using QTX transparently as a llama.cpp
KV-cache or tensor buffer still requires work.

Other open areas include:

- SIMD acceleration for codebook encoders;
- perplexity and end-to-end model-quality evaluation;
- real GGUF round-trip tests;
- calibration-aware formats such as GPTQ and AWQ;
- a design for tenants spanning multiple slots;
- benchmarks on multi-core consumer hardware;
- benchmarks based on real agent workloads rather than synthetic
  tenants.

## Why I am publishing it

The honest answer is that I do not know exactly what I want from Hacker
News.

This is my first experience opening a project to a large technical
community. Of course I am interested in users, criticism, contributors,
and possibly commercial users. But I am also curious about the process
itself.

What changes after strangers read the code? What does it feel like to
hear opinions from people who have no reason to be polite? What happens
when somebody opens an issue, proposes a different architecture, or
sends an email about something I made?

I am interested not only in how people react to QTX, but in how I will
change after publishing it.

The questions I would especially like feedback on are:

1. Should lossy eviction require explicit opt-in for every tensor?
2. For a future ggml backend, is preserving a contiguous buffer
   interface worth the complexity, or should QTX expose paged views?
3. What would be the most convincing first real workload: KV-cache
   pages, dormant LoRA adapters, agent state, or something else?
4. Should micro-tenants eventually participate in cold-tier migration,
   or should the two ownership models remain separate?

QTX is written in C++23. The open-source version is AGPLv3, with a
separate commercial license available for proprietary use.

You can [read the source, run the benchmarks, and inspect every open
edge on GitHub](https://github.com/artiprice/QTX-Quintessence-Tensor-eXchange).

