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

The server does not own model tokenization semantics. `ITextTokenCodec` owns prompt encoding, chat-template encoding, and creation of a request-scoped `ITextTokenDecoder`. `DeterministicTextTokenCodec` exists only for the zero-model deterministic backend and transport specifications.

Set `Fission:Tokenizer=huggingface` to load a local Hugging Face `tokenizer.json` through the Rust-backed `Tokenizers.HuggingFace` binding. The tokenizer is loaded once at server startup and shared read-only across requests; each response still owns independent managed decode-stream state. The required setting is `Fission:TokenizerPath`. Optional settings are:

- `Fission:ChatTemplate=none|qwen2|llama3` (default `none`).
- `Fission:AddPromptSpecialTokens=true|false` (default `true`).
- `Fission:SkipSpecialTokensOnDecode=true|false` (default `true`).

Qwen2 chat rendering emits the standard `<|im_start|>...<|im_end|>` turn structure and an assistant generation prefix. Llama 3 rendering emits `<|begin_of_text|>`, header tokens, `<|eot_id|>` turn terminators, and the assistant header prefix. These control strings must exist as added special tokens in the loaded tokenizer. They are extracted to their configured ids rather than re-tokenized as ordinary text. `none` keeps completion tokenization available but makes chat requests fail clearly until a model-appropriate template is selected.

Generated text is decoded with request-local state rather than by decoding each token id independently. The Hugging Face codec reproduces the prefix/context algorithm used by tokenizers 0.23 `DecodeStream`: pending ids are retained when byte-fallback or contextual decoding has not produced stable Unicode yet, stable prefixes are emitted once, and already-stable decode context is discarded. `ITextTokenDecoder.Append` may therefore return an empty string; streaming endpoints suppress those empty fragments. `Complete` flushes any remaining text before the terminal OpenAI chunk. This avoids the classic byte-level/metaspace error where independently decoding token ids loses cross-token Unicode or whitespace context.

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

This composition currently uses the CPU FP32 Optimum legacy binding. A
production Hugging Face tokenizer and Qwen2/Llama3 chat-template path are
available independently through `Fission:Tokenizer=huggingface`; deterministic
tokenization remains the default for zero-model smoke runs. CUDA Execution
Provider composition is still a separate hardware-tested milestone.
