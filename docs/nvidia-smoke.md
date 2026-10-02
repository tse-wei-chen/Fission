# NVIDIA real-model smoke

This smoke is the first hardware gate for the ONNX Runtime CUDA serving path. It
runs one real prompt through the same production composition used by
`Fission.Server` and exits before opening an HTTP listener.

## What it verifies

The one-shot probe covers:

```text
tokenizer.json
  -> Hugging Face prompt encoding
  -> InferenceWorker
  -> F# scheduling / C# runtime
  -> KV reservation and lifecycle
  -> ONNX Runtime CUDA Execution Provider
  -> CUDA-resident decoder binding
  -> generated token history
  -> request-scoped text decoder
```

A zero exit code means the configured model generated at least one token, reached
a terminal inference reason, released its runtime/KV ownership, and returned
control to the process. It is deliberately stronger than checking that an ONNX
session can be constructed.

## Host prerequisites

- .NET 10 SDK.
- An NVIDIA GPU visible to `nvidia-smi`.
- NVIDIA CUDA and cuDNN libraries compatible with the
  `Microsoft.ML.OnnxRuntime.Gpu` version referenced by the repository.
- A decoder-with-past ONNX graph matching Fission's current Optimum legacy FP32
  decoder contract.
- The matching Hugging Face `tokenizer.json`.
- Correct decoder geometry: hidden-layer count, KV-head count, head dimension,
  vocabulary size, and model EOS token ids when applicable.

ONNX Runtime changes its CUDA/cuDNN package matrix over time. Check the official
CUDA Execution Provider compatibility table for the exact package version in
`src/Fission.Backends.OnnxRuntime/Fission.Backends.OnnxRuntime.csproj` instead
of assuming a CUDA major version from the driver alone:

<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>

## Run

Build and execute the one-shot probe from the repository root:

```powershell
pwsh ./eng/run-nvidia-smoke.ps1 `
  -ModelPath /models/model/decoder_with_past_model.onnx `
  -TokenizerPath /models/model/tokenizer.json `
  -ModelId model-smoke `
  -NumHiddenLayers <layers> `
  -NumKvHeads <kv-heads> `
  -HeadDim <head-dim> `
  -VocabularySize <vocabulary-size> `
  -EosTokenIds "<comma-separated-token-ids>"
```

Optional controls include:

- `-CudaDeviceId` (default `0`)
- `-Prompt` (default `Hello`)
- `-MaxTokens` (default `1`)
- `-TimeoutSeconds` (default `120`)
- `-ChatTemplate none|chatml|qwen2|llama3`
- `-CudaRuntimeLibraryPath` when the CUDA Runtime cannot be resolved normally. When supplied, its parent directory is also prepended to the child process library search path (`PATH` on Windows, `LD_LIBRARY_PATH` elsewhere).
- `-NoBuild` when `Fission.Server` is already built

The script never logs generated text. `Fission.Server` logs model id, prompt and
generated token counts, finish reason, and elapsed time.

## Success criteria

The command must:

1. show the intended NVIDIA device through `nvidia-smi`;
2. initialize the CUDA Runtime and ONNX Runtime CUDA Execution Provider;
3. load the configured decoder-with-past model and tokenizer;
4. complete at least one generated token before the timeout;
5. log `Startup inference probe succeeded`;
6. log the one-shot completion message and exit with code `0`.

Any model contract mismatch, missing provider library, tokenizer mismatch,
scheduler/runtime failure, timeout, or decode failure must leave a non-zero
process exit.

## CI boundary

Hosted CI does not claim NVIDIA hardware validation. The container workflow runs
the same one-shot exit path with the deterministic backend so the control-flow
contract cannot silently regress. Real CUDA validation evidence must come from an
NVIDIA host using the command above.

## Recorded hardware validation

The first real NVIDIA validation was recorded on 2026-10-02 with this exact
functional matrix:

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 |
| Device memory | 12288 MiB |
| NVIDIA driver | 610.74 |
| CUDA toolkit/runtime | CUDA 13.4 with explicit `cudart64_13.dll` |
| ONNX Runtime GPU package | 1.30.0 |
| Model | `SmolLM2-135M-Instruct` |
| ONNX graph | `artifacts/models/SmolLM2-135M-Instruct/onnx/model.onnx` |
| Tokenizer | matching Hugging Face `tokenizer.json` |
| Geometry | 30 layers, 3 KV heads, head dim 64, vocabulary 49152 |
| EOS token ids | `2` |
| Probe result | promptTokens=1, generatedTokens=1, finishReason=Length |
| Probe elapsed | 335.1 ms |

This establishes a **functionally validated** CUDA path for that exact
model/runtime combination. It is not yet a performance qualification.

The same run reported two ONNX Runtime optimization warnings:

- 121 `Memcpy` nodes were inserted for the CUDA execution provider;
- some graph nodes were assigned outside the preferred execution provider.

Those warnings did not prevent correct inference, but they are performance
signals. Do not suppress them as noise. First record TTFT, TPOT, throughput, and
concurrency behavior with the serving benchmark gate; use that evidence to decide
whether node placement or transfer reduction is worth changing.

## Next gate: real serving benchmark

Run the hardware-controlled serving gate after the one-shot probe succeeds:

```powershell
pwsh ./eng/run-nvidia-serving-benchmark.ps1 `
  -ModelPath artifacts/models/SmolLM2-135M-Instruct/onnx/model.onnx `
  -TokenizerPath artifacts/models/SmolLM2-135M-Instruct/tokenizer.json `
  -ModelId SmolLM2-135M-Instruct `
  -NumHiddenLayers 30 `
  -NumKvHeads 3 `
  -HeadDim 64 `
  -VocabularySize 49152 `
  -EosTokenIds 2
```

The runner starts `Fission.Server` with the same CUDA/model/tokenizer
configuration, keeps the startup probe enabled as a pre-serving gate, waits for
`/healthz`, runs `workloads.gpu-smoke.json`, writes environment metadata and
server logs beside the benchmark artifacts, then produces Markdown and CSV
reports.

Use `-Manifest benchmarks/serving/workloads.json` to move from the small gate
to the full concurrency suite.


## First serving baseline

The first RTX 3060 serving benchmark on 2026-10-02 used
`workloads.gpu-smoke.json` with one repetition and established this baseline:

| Workload | C | Output tok/s | TTFT p50 ms | TPOT p50 ms | E2E p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpu-short | 1 | 55.55 | 35.98 | 17.16 | 672.90 |
| gpu-short | 4 | 94.96 | 100.70 | 39.97 | 1429.17 |
| gpu-short | 8 | 173.06 | 121.16 | 43.52 | 1522.96 |
| gpu-decode | 1 | 47.46 | 46.65 | 20.83 | 2800.20 |
| gpu-decode | 4 | 78.32 | 136.73 | 50.30 | 6608.85 |
| gpu-decode | 8 | 123.50 | 211.14 | 63.63 | 8319.86 |

All requests succeeded and usage accounting was exact. Output throughput still
increased through concurrency 8, while TPOT rose by roughly 2.5x for
`gpu-short` and 3.1x for `gpu-decode` versus concurrency 1. That makes the
batched CUDA step itself, including host/device transfer and logits handling, the
next optimization target rather than the scheduler's ability to form a batch.

The CUDA binding currently keeps KV device-resident but leaves decode logits
host-backed. For vocabulary 49,152, an 8-row FP32 decode batch writes about
1.5 MiB of logits to host memory per token step. The
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
