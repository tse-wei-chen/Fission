# OpenAI-compatible serving invariants

`Fission.Server` is the HTTP/SSE transport boundary above `InferenceWorker`.

## Request path

```text
HTTP request
  -> OpenAI request DTO
  -> ITextTokenCodec
  -> InferenceWorker.SubmitAsync
  -> InferenceEngine scheduling cycles
  -> decode token stream
  -> OpenAI JSON or SSE response
```

The current transport supports the common subset used by:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- non-stream JSON responses
- `stream=true` SSE responses terminated by `data: [DONE]`
- `max_tokens` and `max_completion_tokens`

## Tokenizer boundary

The server does not own model tokenization semantics. `ITextTokenCodec` owns prompt encoding, chat-template encoding, and creation of a request-scoped `ITextTokenDecoder`. `DeterministicTextTokenCodec` exists only for the zero-model deterministic backend and transport specifications. Real model integrations must replace it with the tokenizer/chat template that matches the loaded model.

Generated text is decoded with request-local state rather than by decoding each token id independently. `ITextTokenDecoder.Append` may return an empty string when a tokenizer needs later ids to complete a stable text fragment; streaming endpoints suppress those empty fragments. `Complete` flushes any remaining stable text before the terminal OpenAI chunk. The decoder is request-scoped and disposable so production codecs may lease native tokenizer state without globally serializing concurrent requests. This boundary is required by byte-level/fallback tokenizers where one Unicode fragment may span multiple generated token ids.


### Hugging Face tokenizer composition

`Fission.Server` keeps the deterministic codec as the default for zero-model
smoke tests. Set `Fission:Tokenizer=huggingface` to load a local
`tokenizer.json` through the native Hugging Face tokenizers binding.

Required:

- `Fission:TokenizerPath`: local `tokenizer.json` path.

Optional:

- `Fission:AddPromptSpecialTokens`: whether completion prompts use the tokenizer
  post-processor; defaults to `true`.
- `Fission:SkipSpecialTokensOnDecode`: whether generated-text decoding suppresses
  tokenizer special tokens; defaults to `true`.
- `Fission:ChatTemplate`: `chatml`/`qwen2`, `llama3`, or `none`. It defaults to
  `none`.

Chat serving deliberately requires an explicit model template. With
`ChatTemplate=none`, completion requests still work, while chat encoding fails
with a request error instead of silently applying the wrong prompt format.

The built-in templates currently cover:

- `chatml` / `qwen2`: Qwen/ChatML-style `<|im_start|>role ... <|im_end|>` framing.
- `llama3`: Llama-3-style begin/header/eot framing.

The production codec shares one initialized tokenizer across concurrent requests
and keeps only streaming decode state per request. Added special-token extraction
is explicitly enabled by leaving Hugging Face `encode_special_tokens=false`;
control strings such as `<|im_start|>` therefore map directly to their configured
added-token ids instead of being split by the normal tokenizer model.

Its incremental decoder ports the Hugging Face `DecodeStream` state machine:
it retains the minimum token context needed for decoder continuity, buffers
incomplete byte-fallback UTF-8, and never re-decodes the complete generation
history on every token. Chat roles are restricted to the supported OpenAI-style
role set (`system`, `developer`, `user`, `assistant`, `tool`) so arbitrary
role strings cannot silently become control-role text.

## Cancellation ownership

HTTP request threads never dispose runtime sequences or KV pages directly.

A client disconnect causes the endpoint to abandon the stream and call `InferenceStream.CancelAsync`. The worker places that cancellation on its control channel. The single worker actor then calls `InferenceEngine.CancelAsync`, which shares the same serialization gate as scheduler cycles. Runtime sequence cancellation and KV release therefore happen only at a scheduler-cycle boundary, never concurrently with backend execution.

Cancellation before the first prefill creates no runtime sequence and only terminates engine request state. Cancellation after prefill/decode transitions the live sequence to `Cancelled`, removes it from `ExecutionPlanExecutor`, and returns its KV leases to the shared pool.

## Finish reasons

Engine snapshots carry an explicit terminal reason:

- `Stop`: backend EOS/stop condition
- `Length`: `max_new_tokens` reached
- `Cancelled`: caller/client cancellation

The OpenAI transport maps these to `stop`, `length`, and `cancelled` respectively. A disconnected client normally does not receive the final cancelled chunk; the reason remains available in engine state for tracing/metrics.

## Backpressure

Admission remains bounded at `InferenceWorker`. Per-request output channels remain unbounded for now so a slow network consumer cannot block the scheduler actor or device loop. A later serving policy may add bounded output buffering plus explicit slow-consumer cancellation/drop rules.


## Backend composition

`Fission.Server` defaults to the deterministic backend so CI, transport specs,
and zero-model smoke runs remain self-contained. Set `Fission:Backend=onnx` to
compose the existing stateful Optimum legacy decoder stack:

```text
OnnxRuntimeModelSource
  -> OptimumLegacyFloatDecoderBinding
  -> DecoderOnlyOnnxExecutionAdapter
  -> OnnxRuntimeBackend
  -> ContinuousBatchExecutor
  -> InferenceEngine
```

The initial real-model composition requires:

- `Fission:ModelPath`: local decoder-with-past ONNX file.
- `Fission:ModelId`: model id expected by serving requests.
- `Fission:NumHiddenLayers`: positive decoder layer count.
- `Fission:NumKvHeads`: positive KV-head count.
- `Fission:HeadDim`: positive attention head dimension.
- `Fission:VocabularySize`: positive vocabulary size.
- `Fission:EosTokenIds`: optional comma/semicolon/space-separated EOS token ids.

The server validates configuration and model-file existence before runtime
construction. `ContinuousBatchExecutor.CreateAsync` initializes the backend
before Kestrel starts serving, so an invalid ONNX graph/session contract fails
startup rather than the first inference request.

### ONNX execution provider

`Fission:ExecutionProvider` selects the ONNX Runtime execution provider for an
ONNX backend:

- `cpu` (default): uses the existing FP32 Optimum decoder binding.
- `cuda`: appends the ONNX Runtime CUDA Execution Provider and uses the
  CUDA-resident Optimum FP32 decoder binding.

CUDA mode accepts:

- `Fission:CudaDeviceId`: non-negative CUDA ordinal; defaults to `0`.
- `Fission:CudaRuntimeLibraryPath`: optional explicit CUDA Runtime library.
- `Fission:CudaPageLockedDecodeLogits`: experimental opt-in decode hot path.
  When `true`, decode logits are written into reusable CUDA page-locked host
  buffers before CPU greedy sampling. Prefill logits remain on the existing
  pageable scratch path. Defaults to `false`.
- `Fission:SampledTokenIdsOutput`: optional CUDA-only rank-2 int64 graph
  output containing one greedy token id per batch row. When configured, the
  CUDA binding fetches this small output instead of the full logits tensor for
  both prefill and decode. The graph output must have shape `[batch, 1]`.
  `eng/add-greedy-argmax-output.py` can append the matching
  `Gather(last sequence position) -> ArgMax(vocabulary)` path to a compatible
  decoder graph.
- `Fission:OrtProfileOutputPathPrefix`: optional CUDA ONNX Runtime profile
  path prefix. When set, the CUDA session enables ORT profiling and writes a
  Chrome-trace JSON file when the session shuts down cleanly. Profiling adds
  overhead and is intended for diagnosis rather than throughput comparison.
- `Fission:CudaPoolMaxRetainedBytes`: maximum idle KV/device-buffer bytes
  retained for exact-size reuse; defaults to 256 MiB. Set `0` to disable idle
  retention.
- `Fission:CudaPoolMaxRetainedBuffersPerSize`: exact-size idle buffer count
  limit; defaults to `8`. Set `0` to disable idle retention.

When `Fission:Device` is not explicitly set, CUDA mode derives the logical
runtime device as `cuda:<CudaDeviceId>`; CPU and deterministic modes default to
`cpu:0`.

CUDA composition couples three existing layers:

```text
ONNX Runtime CUDA EP
  + OptimumLegacyCudaFloatDecoderBinding
  + CudaPooledDeviceMemoryAllocator
  + CudaDeviceMemoryPressureMonitor
        |
        v
ContinuousBatchExecutor / runtime memory admission
```

The device-wide pressure monitor uses `cudaMemGetInfo`, so scheduling sees
model weights, ONNX Runtime workspaces, resident KV, allocator caches, and other
CUDA allocations as one physical VRAM budget. Idle bytes retained by the Fission
CUDA pool are classified as reclaimable and may be synchronously trimmed through
the runtime memory-reclaim capability.

The server now uses the `Microsoft.ML.OnnxRuntime.Gpu` package for ONNX hosting.
CPU remains the default execution provider; CUDA native/provider initialization
is only requested when `ExecutionProvider=cuda`.

Tokenizer composition is independently selectable, so an ONNX model can be
paired with a matching Hugging Face `tokenizer.json` and explicit chat-template
profile. CPU and deterministic serving remain the hardware-independent CI paths.
CUDA composition is now wired, while the next milestone is a real NVIDIA
hardware smoke using an exported decoder-with-past model before CUDA mode is
treated as production-validated.


## Startup inference probe

Real-model hosts can opt into a full-stack startup inference before Kestrel begins
accepting traffic:

- `Fission:StartupProbeEnabled=true`
- `Fission:StartupProbePrompt=<non-empty prompt>`
- `Fission:StartupProbeModelId=<model id>` (optional; falls back to `Fission:ModelId`)
- `Fission:StartupProbeMaxTokens=<positive integer>` (default `1`)
- `Fission:StartupProbeTimeoutSeconds=<positive integer>` (default `60`)
- `Fission:StartupProbeExitAfterSuccess=true` (optional one-shot mode)

The probe deliberately uses the same configured `ITextTokenCodec`,
`InferenceWorker`, scheduler/runtime, backend, KV lifecycle, and request-scoped
decoder as normal serving:

```text
configured prompt
  -> production tokenizer
  -> InferenceWorker
  -> scheduler / runtime / KV
  -> ONNX Runtime CPU or CUDA backend
  -> generated token history
  -> production streaming decoder
```

Startup succeeds only after the request reaches a terminal inference reason and
produces at least one generated token. The server logs model id, token counts,
finish reason, and elapsed time, but not the generated text. A model/tokenizer
mismatch, invalid ONNX graph, CUDA/provider failure, runtime scheduling failure,
or decode error therefore fails process startup instead of leaving an HTTP server
that becomes unhealthy on its first real request.

The probe is disabled by default so deterministic CI/container smoke paths do not
perform extra inference. For NVIDIA real-model validation, enable it together
with `Backend=onnx`, `ExecutionProvider=cuda`, and
`Tokenizer=huggingface`.

With `StartupProbeExitAfterSuccess=true`, the server becomes a one-shot
validation executable: a successful full-stack probe logs its summary and exits
with code 0 before Kestrel begins listening. Composition, CUDA/provider, model
contract, tokenizer, scheduling, KV, timeout, or decode failures still fail the
process. This mode is used by [the NVIDIA real-model smoke runner](nvidia-smoke.md).


## Opt-in graceful control endpoint

`Fission:ControlToken` is intended for controlled benchmark/process shutdown.
When it is unset, no control route is mapped. When it is set, the server exposes:

```text
POST /internal/control/shutdown
X-Fission-Control-Token: <configured token>
```

An invalid or missing token returns `401`. A valid token returns `202` and
requests graceful application shutdown after the response completes. The
NVIDIA benchmark runner generates an ephemeral random token automatically so
ONNX Runtime profiling can flush during normal session disposal.
