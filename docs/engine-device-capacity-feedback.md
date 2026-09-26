# Engine device-capacity feedback

`ContinuousBatchExecutor.capacity` is the maximum number of outstanding inference items for one execution target. Atomic scheduler envelopes are charged by item count, and `ScheduledBatchExecutor` rejects an envelope larger than that capacity before runtime state is materialized.

`InferenceEngine` therefore feeds the same limit back into scheduling rather than relying on that rejection as normal flow control. For each cycle it uses:

```text
max scheduled sequences = min(configured MaxBatchSequences, runtime MaxInferenceItems)
```

This keeps the configured policy ceiling while respecting the concrete execution target. The runtime exposes only `RuntimeExecutionCapacity`; the engine does not depend on the device actor implementation.

The execution-time weighted credit gate remains necessary because multiple producers or non-engine callers can still contend for the target after scheduling. Capacity feedback prevents structurally oversized engine batches; credits provide live backpressure.
