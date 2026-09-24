# Backend sequence lifecycle invariants

A stateful inference backend may retain physical state after prefill and across decode iterations. Examples include model KV tensors, pending logits, sampler state, decoder buffers, or native handles.

The runtime metadata KV page table is not a substitute for that backend-owned state. Both lifecycles must converge when a sequence finishes or is cancelled.

## Release ordering

Terminal sequence release follows this order:

```text
sequence -> Finished / Cancelled
        -> ContinuousBatchExecutor release work item
        -> IInferenceBackend.ReleaseSequenceAsync
        -> backend-owned state released
        -> ExecutionPlanExecutor removes sequence metadata
        -> metadata KV page leases disposed / returned to pool
```

`ReleaseSequenceAsync` is queued on the same single-device actor as prefill and decode. A caller never invokes backend release concurrently with backend execution.

If backend release fails, the runtime sequence remains registered. Metadata and KV leases are intentionally retained so the failure can be observed, retried, or recovered without losing the identity of the backend state that may still exist.

## Backend contract

`IInferenceBackend.ReleaseSequenceAsync` has a default no-op implementation so stateless backends do not need boilerplate. Stateful backends must override it.

A stateful implementation should make release idempotent where practical. The runtime normally invokes it exactly once for a terminal sequence, but idempotence simplifies recovery from process-level retries and partial backend failures.

`IInferenceBackend.DisposeAsync` remains the global shutdown fallback. It must release any backend-owned resources that remain when the whole device executor is disposed, including sequence state that did not reach the normal terminal release path.

## Why release is device work

Model backends commonly use native runtimes, GPU streams, device memory, and asynchronous execution. Releasing those resources from an HTTP thread or scheduler thread could race with an in-flight prefill/decode operation. Keeping release on `ContinuousBatchExecutor` preserves one serialization domain for all device-facing state mutations.

This boundary is required before ONNX Runtime or custom CUDA backends can safely keep per-sequence state across decode steps.
