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

The server does not own model tokenization semantics. `ITextTokenCodec` owns prompt encoding, chat-template encoding, and token-to-text decoding. `DeterministicTextTokenCodec` exists only for the zero-model deterministic backend and transport specifications. Real model integrations must replace it with the tokenizer/chat template that matches the loaded model.

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
