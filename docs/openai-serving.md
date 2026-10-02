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

- `Fission:AddSpecialTokens`: whether completion prompts use the tokenizer
  post-processor; defaults to `true`.
- `Fission:SkipSpecialTokens`: whether generated-text decoding suppresses
  tokenizer special tokens; defaults to `true`.
- `Fission:ChatTemplate`: `chatml`, `llama3`, or `none`. It defaults to
  `none`.

Chat serving deliberately requires an explicit model template. With
`ChatTemplate=none`, completion requests still work, while chat encoding fails
with a request error instead of silently applying the wrong prompt format.

The built-in templates currently cover:

- `chatml`: Qwen/ChatML-style `<|im_start|>role ... <|im_end|>` framing.
- `llama3`: Llama-3-style begin/header/eot framing.

The production codec shares one initialized tokenizer across concurrent requests
and keeps only streaming decode state per request. Its incremental decoder ports
the Hugging Face `DecodeStream` state machine: it retains the minimum token
context needed for decoder continuity, buffers incomplete byte-fallback UTF-8,
and never re-decodes the complete generation history on every token.

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

This composition currently uses the CPU FP32 Optimum legacy binding. Tokenizer
composition is independently selectable, so an ONNX model can be paired with a
matching Hugging Face `tokenizer.json` and explicit chat-template profile.
CUDA Execution Provider composition remains a separate hardware-tested
milestone. The deterministic backend and tokenizer both remain defaults for CI
and zero-model smoke paths.
