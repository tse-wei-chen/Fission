# NVIDIA real-model smoke and performance evidence

This document records the real NVIDIA validation path for Fission. Hosted CI remains self-contained and does not claim real GPU validation.

## Validated host

The current hardware evidence was collected on:

- NVIDIA GeForce RTX 3060
- 12,288 MiB VRAM
- NVIDIA driver 610.74
- CUDA 13.4 runtime (`cudart64_13.dll`)
- SmolLM2-135M-Instruct
- 30 transformer layers
- 3 KV heads
- head dimension 64
- vocabulary size 49,152
- EOS token id 2

The one-shot startup probe covers tokenizer -> server worker -> scheduler/runtime/KV -> ONNX Runtime backend -> request decoder and requires at least one generated token plus a terminal reason.

## Initial FP32 CUDA validation

The first functional CUDA startup probe completed successfully with one prompt token and one generated token. ONNX Runtime reported that 121 Memcpy nodes were inserted into the graph and that some shape/non-preferred nodes executed on CPU. That established functional CUDA execution for this exact host/model stack but did not establish performance qualification.

The first serving benchmark showed that continuous batching was working: output throughput scaled with concurrency instead of degrading into scalar decode. At `gpu-short` concurrency 8 the engine reached roughly 173 output tok/s, while `gpu-decode` concurrency 8 reached roughly 124 output tok/s. This moved the investigation from scheduler batch formation to work inside the batched ORT/CUDA step.

`Fission:CudaPageLockedDecodeLogits` experiment and
`-PageLockedDecodeLogits` benchmark switch exist to measure whether replacing
the pageable decode-logits destination with reusable CUDA page-locked memory
improves TPOT on this host.

## Page-locked logits A/B validation

A three-repetition A/B run on the same RTX 3060 validated the page-locked
decode-logits experiment. The table below reports mean output throughput and
mean-of-run TPOT p50 values:

| Workload | C | Baseline tok/s | Pinned tok/s | Delta | Baseline TPOT p50 ms | Pinned TPOT p50 ms | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-short | 1 | 52.97 | 59.01 | +11.41% | 18.26 | 16.31 | -10.66% |
| gpu-short | 4 | 97.58 | 111.07 | +13.83% | 38.88 | 34.25 | -11.91% |
| gpu-short | 8 | 168.59 | 196.23 | +16.39% | 45.05 | 38.72 | -14.05% |
| gpu-decode | 1 | 47.66 | 51.86 | +8.82% | 20.73 | 19.03 | -8.22% |
| gpu-decode | 4 | 84.41 | 91.47 | +8.36% | 46.78 | 42.95 | -8.19% |
| gpu-decode | 8 | 140.08 | 140.51 | +0.31% | 56.07 | 55.80 | -0.48% |

All 288 measured requests succeeded and every run returned exact usage
accounting. The page-locked path therefore has a repeatable TPOT/throughput
benefit at low and medium concurrency and for the short workload through
concurrency 8. The long decode-heavy workload at concurrency 8 is effectively
flat, indicating that full-batch long-context execution is dominated elsewhere.

TTFT deltas from this experiment are not attributed to page-locked decode
logits. The optimization is entered only after prefill has produced the first
token, so it cannot causally change the first-token path; observed TTFT movement
is treated as run-to-run scheduling/system variation.

The next transfer-reduction experiment is graph-side greedy sampling. For this
model and vocabulary, an eight-row FP32 decode step exposes about 1.5 MiB of
logits to the host path. A graph-side `Gather -> ArgMax` output reduces the
requested host result to eight int64 token ids while leaving the logits tensor
inside ONNX Runtime.

## Graph-side greedy A/B result

A three-repetition comparison against the validated page-locked baseline showed
that returning only graph-side greedy token ids does **not** improve this model
consistently on the RTX 3060:

| Workload | C | Pinned tok/s | Graph greedy tok/s | Throughput delta | Pinned TPOT p50 ms | Graph TPOT p50 ms | TPOT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-short | 1 | 59.01 | 55.29 | -6.30% | 16.31 | 17.47 | +7.10% |
| gpu-short | 4 | 111.07 | 105.76 | -4.78% | 34.25 | 36.11 | +5.43% |
| gpu-short | 8 | 196.23 | 202.57 | +3.23% | 38.72 | 37.65 | -2.78% |
| gpu-decode | 1 | 51.86 | 48.55 | -6.38% | 19.03 | 20.43 | +7.34% |
| gpu-decode | 4 | 91.47 | 84.38 | -7.75% | 42.95 | 46.59 | +8.46% |
| gpu-decode | 8 | 140.51 | 136.82 | -2.63% | 55.80 | 57.67 | +3.35% |

All measured requests succeeded with exact usage accounting. The graph-side path
therefore remains an experiment rather than the default. Removing the full
logits host result saves transfer volume, but the added graph operations and/or
provider scheduling cost more at C=1/4 and on the long C=8 decode workload.
Only the short C=8 row shows a modest win.

This result makes the existing 121 Memcpy nodes and CPU/CUDA graph partition the
next attribution target. Use the ORT profiling gate before changing provider
options or graph placement.

### Checker boundary for ORT-optimized graphs

The validated model contains `SimplifiedLayerNormalization`, which the generic
ONNX checker may not recognize even though the target ONNX Runtime build accepts
and executes the graph. The graph rewrite tool therefore supports
`--skip-check`, **disabled by default**.

Use `--skip-check` only when the checker fails on an operator already accepted
by the target runtime, then require the normal Fission NVIDIA one-shot smoke
before benchmarking the rewritten model. Skipping the checker is not runtime
validation.

## ORT profile attribution: FP32 GQA CPU island

The first pinned-logits ORT profile reached ORT's event limit, so it is a
partial trace. Even with that limitation, the provider/operator distribution is
decisive:

| Signal | Partial-trace value |
| --- | ---: |
| Node execution events | 996,376 |
| Summed node duration | 40,311.79 ms |
| CUDA provider share | 92.17% |
| CPU provider share | 7.83% |
| Memcpy events | 219,204 |
| Memcpy duration share | 18.57% |
| CPU GroupQueryAttention events | 54,348 |
| CPU GroupQueryAttention share | 7.67% |
| MemcpyFromHost events | 54,348 |
| MemcpyToHost events | 164,856 |
| CUDA MatMul share | 44.75% |

The exact equality between CPU `GroupQueryAttention` and
`MemcpyFromHost` event counts, together with roughly three
`MemcpyToHost` events per GQA invocation, strongly identifies GQA as a CPU
island inside the otherwise CUDA-heavy graph. Treat that event-count relationship
as attribution evidence, not a proof that every Memcpy belongs exclusively to
GQA.

The reason is visible in ONNX Runtime 1.30 itself: the CUDA
`GroupQueryAttention` contrib kernel is registered for FP16/BF16 (plus
quantized-KV variants), while the current Fission/SmolLM2 validation graph is
FP32. The CPU provider has an FP32 GQA kernel, so ORT partitions those nodes to
CPU and inserts transfers around them.

For reference, inspect the ONNX Runtime 1.30 source:

- `onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc`
- `onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc`

The next validated path is therefore FP16 CUDA execution rather than another
scheduler or logits-transfer tweak.

### Constrained FP16 path

Fission's first FP16 CUDA path intentionally supports only:

- FP16 model/KV tensors;
- CUDA execution provider;
- graph-side sampled token ids;
- CUDA-resident FP16 KV state.

It does **not** yet add an FP16 pageable/pinned host-logits sampler. This keeps
the first experiment narrow enough to answer whether moving GQA off CPU removes
the measured transfer island. Existing FP32 page-locked serving remains the
validated default until the FP16 hardware gate passes.

## FP16 GQA unprofiled performance validation

A three-repetition, unprofiled serving run on the same RTX 3060 established a
large repeatable win for the FP16 GQA path over the previously validated FP32
page-locked baseline. All 288 measured requests succeeded and all six rows
reported exact usage for all three repetitions.

| Workload | C | FP32 pinned tok/s | FP16 GQA tok/s | Throughput delta | FP32 TTFT p50 ms | FP16 TTFT p50 ms | TTFT delta | FP32 TPOT p50 ms | FP16 TPOT p50 ms | TPOT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-decode | 1 | 51.86 | 135.63 | +161.53% | 42.15 | 15.04 | -64.32% | 19.03 | 7.35 | -61.38% |
| gpu-decode | 4 | 91.47 | 239.61 | +161.95% | 119.40 | 45.33 | -62.04% | 42.95 | 16.44 | -61.72% |
| gpu-decode | 8 | 140.51 | 443.44 | +215.59% | 193.13 | 58.99 | -69.46% | 55.80 | 17.75 | -68.19% |
| gpu-short | 1 | 59.01 | 129.23 | +119.00% | 34.96 | 14.31 | -59.07% | 16.31 | 7.27 | -55.43% |
| gpu-short | 4 | 111.07 | 237.35 | +113.69% | 86.06 | 38.34 | -55.45% | 34.25 | 15.98 | -53.34% |
| gpu-short | 8 | 196.23 | 508.62 | +159.20% | 104.56 | 39.03 | -62.67% | 38.72 | 15.28 | -60.54% |

The improvement is too large and too consistent to treat as run-to-run noise.
It also changes both prefill/first-token and decode behavior, unlike the earlier
page-locked decode-only experiment. On this host/model stack, FP16 is now the
**preferred NVIDIA performance candidate**.

This does not yet make FP16 the unconditional production default. Remaining
qualification gates are:

- rerun the ORT structural gate with `-RequireCudaOp GroupQueryAttention` after
  the empty-provider StrictMode fix, confirming no observed CPU GQA events;
- high-concurrency serving beyond C=8;
- extended 256-token decode under C=8/16/32;
- a substantially longer prompt/context at C=1/4/8/16;
- semantic/token-output comparison on a fixed prompt corpus before claiming
  precision parity or equivalent model quality.

The production qualification workload manifest is
`benchmarks/serving/workloads.gpu-qualification.json`. Until those gates are
closed, keep the FP32 page-locked path as the compatibility fallback even though
FP16 is the preferred performance path for this validated NVIDIA stack.
