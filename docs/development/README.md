# Development reports

Engineering journals from QTX's build-out, kept for transparency: each
documents the design decisions, edge cases, and verification for one
slice of the project. They are historical records, not user
documentation — for usage, see the [top-level README](../../README.md).

## Implementation epics

| Report | Covers |
|---|---|
| [EPIC1_FP8_REPORT.md](EPIC1_FP8_REPORT.md) | FP8 (E4M3 / E5M2), NVFP4, MXFP4 — hardware float formats |
| [EPIC2_GGML_LEGACY_REPORT.md](EPIC2_GGML_LEGACY_REPORT.md) | Q4_1, Q5_0, Q5_1, INT8 — ggml legacy formats |
| [EPIC3_K_QUANTS_REPORT.md](EPIC3_K_QUANTS_REPORT.md) | Q2_K…Q6_K — hierarchical super-block quantization |
| [EPIC4_I_QUANTS_REPORT.md](EPIC4_I_QUANTS_REPORT.md) | IQ1_S…IQ4_XS — codebook-based importance quantization |

## Hardening & performance

| Report | Covers |
|---|---|
| [EDGE_CASES_REPORT.md](EDGE_CASES_REPORT.md) | EC-100 edge-case hardening catalog |
| [PHASE2_HARDENING_REPORT.md](PHASE2_HARDENING_REPORT.md) | EC121–EC200 threat catalog & C-ABI boundary hardening |
| [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) | Throughput, latency, and compression benchmarks |
