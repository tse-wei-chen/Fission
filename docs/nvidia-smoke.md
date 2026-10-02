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
