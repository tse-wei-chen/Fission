# Async inference worker

`InferenceWorker` is the serving-side actor above `InferenceEngine`.

## Concurrency model

- Many producers may call `SubmitAsync` concurrently.
- Submissions enter a bounded `Channel`, so admission pressure propagates to producers instead of creating unbounded request tasks.
- Exactly one worker pump advances scheduler cycles. Requests never create independent loops that race for the same device/runtime state.
- The worker drains newly queued requests between cycles so continuous batching can absorb new arrivals while existing requests decode.

## Streaming

Each accepted request receives an `InferenceStream`:

- `SequenceId` identifies the engine request.
- `ReadTokensAsync()` exposes only decode tokens as `IAsyncEnumerable<int>`.
- `Completion` resolves to the final immutable `InferenceRequestSnapshot` after runtime sequence ownership and KV pages have been released.

Prefill backend results are deliberately not streamed. Token order is taken from the engine's committed request history, keeping the worker downstream of runtime state transitions.

## Ownership and shutdown

Once a submission has entered the bounded admission channel, the worker owns it. Producer cancellation can prevent queue admission, but does not abandon a request after ownership transfer.

`DisposeAsync` closes admission and drains already-active requests before the pump exits. Failures are propagated to queued submissions, open token streams, and completion tasks.

Per-request token channels are currently unbounded. This prevents one slow consumer from head-of-line blocking device scheduling; a later serving policy can add bounded output buffers together with explicit slow-client cancellation/eviction semantics.
