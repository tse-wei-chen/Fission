# NVIDIA FP16 qualification evidence — 2026-10-03

This note records the RTX 3060 + SmolLM2-135M-Instruct FP16 CUDA/GQA
qualification sequence. The final qualification and memory gates both passed.

## Final qualification status

The final unprofiled qualification wrapper completed every expected
workload/concurrency/repetition row:

```text
expected:  30
validated: 30
issues:     0
```

The run covered 1,056 measured requests with zero failures. Every emitted row
reported exact usage. `gpu-long-context` completed C=1/4/8/16 with three
repetitions each, closing the nine rows that were missing from the earlier
partial run.

The two CUDA correctness failures found during qualification are no longer
present:

- multi-row continuation prefill no longer reaches the CUDA
  `GroupQueryAttention` boundary that requires batch size one when a non-empty
  past is combined with more than one new token;
- the D2D gathered past-KV path now accepts FP16 KV tensors instead of failing
  on its former FP32-only guard.

Long-context concurrency exposes an intentional latency trade-off from the GQA
safety policy. Throughput at C=8/C=16 remains approximately 102–106 output
tok/s, while TPOT p50 rises to approximately 64.16/119.45 ms. This is recorded
as a performance optimization target, not a correctness/completeness failure:
the rows complete successfully, preserve exact usage, and do not crash the
server.

## Memory gate

A separate qualification run with GPU telemetry also completed every expected
row:

```text
expected:  10
validated: 10
```

Telemetry sampled the GPU 198 times at a 500 ms interval.

| Metric | Peak |
| --- | ---: |
| VRAM used | 3,479 MiB / 12,288 MiB |
| VRAM headroom | 8,809 MiB |
| GPU utilization | 86% |
| Memory-controller utilization | 26% |
| Temperature | 68 C |

The measured workload therefore remained far from VRAM exhaustion on the
validated RTX 3060. The low peak memory-controller utilization also means the
qualification result does not indicate device-memory-controller saturation as
the primary explanation for the long-context singleton latency trade-off.

The memory-gate serving rows were:

| Workload | C | Max tokens | Output tok/s | TTFT p50 ms | TPOT p50 ms | E2E p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-decode-extended | 8 | 256 | 511.19 | 42.40 | 15.43 | 4348.17 |
| gpu-decode-extended | 16 | 256 | 764.83 | 56.82 | 20.76 | 5523.16 |
| gpu-decode-extended | 32 | 256 | 1113.12 | 72.64 | 28.57 | 7359.24 |
| gpu-long-context | 1 | 64 | 110.08 | 53.38 | 8.23 | 631.84 |
| gpu-long-context | 4 | 64 | 94.42 | 798.44 | 30.10 | 2760.63 |
| gpu-long-context | 8 | 64 | 94.30 | 1249.14 | 66.48 | 5563.23 |
| gpu-long-context | 16 | 64 | 98.08 | 2399.03 | 125.01 | 10437.78 |
| gpu-short-high-concurrency | 8 | 32 | 485.19 | 33.24 | 15.53 | 583.40 |
| gpu-short-high-concurrency | 16 | 32 | 869.27 | 43.25 | 17.56 | 661.09 |
| gpu-short-high-concurrency | 32 | 32 | 1834.50 | 50.36 | 16.31 | 569.40 |

The telemetry run used one repetition per row and is evidence for memory and
utilization headroom, not a replacement for the three-repetition performance
qualification.

## Earlier partial evidence

Before the GQA continuation-prefill and FP16 gathered-KV fixes, the first
high-concurrency run already showed useful C=32 scaling but emitted only the
C=1 long-context row. That partial result was intentionally not accepted as a
qualification pass. The completeness validator correctly reported 21/30 rows
and forced the missing C=4/8/16 long-context cases to be fixed and rerun.

## Remaining promotion gates

The high-concurrency/long-context completeness gate and the memory-headroom
gate are now closed for this hardware/model/runtime tuple. Two promotion gates
remain:

1. complete the CUDA-only `GroupQueryAttention` structural profile assertion;
2. compare FP32 and FP16 deterministic generations with the checked-in semantic
   parity corpus using raw generated token IDs and decoded text.

Keep the FP32 page-locked path selectable as a compatibility fallback. New GPU
models, ORT/CUDA versions, model families, and precision formats remain separate
qualification targets rather than inheriting this RTX 3060 result automatically.
