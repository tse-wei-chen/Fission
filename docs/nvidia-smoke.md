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
pwsh ./eng/run-nvidia-smoke.ps1 \
  -ModelPath /models/model/decoder_with_past_model.onnx \
  -TokenizerPath /models/model/tokenizer.json \
  -ModelId model-smoke \
  -NumHiddenLayers <layers> \
  -NumKvHeads <kv-heads> \
  -HeadDim <head-dim> \
  -VocabularySize <vocabulary-size> \
  -EosTokenIds "<comma-separated-token-ids>"
```

Optional controls include:

- `-CudaDeviceId` (default `0`)
- `-Prompt` (default `Hello`)
- `-MaxTokens` (default `1`)
- `-TimeoutSeconds` (default `120`)
- `-ChatTemplate none|chatml|qwen2|llama3`
- `-CudaRuntimeLibraryPath` when the CUDA Runtime cannot be resolved normally
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
