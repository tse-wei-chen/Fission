# Scheduler-driven inference engine loop

`Fission.Engine` is the request-level orchestration layer above the pure F# scheduling policy and the C# stateful runtime.

## Cycle

One engine cycle performs the following steps:

1. Snapshot active request metadata without holding a lock across device work.
2. Read `ExecutionPlanExecutor.KvCapacity` for the live KV page budget and block size.
3. Derive `SchedulingCandidate` values from request state plus the live runtime sequence position/phase.
4. Invoke `ISchedulingKernel`, implemented by the F# `SchedulingKernel` adapter.
5. Bind selected prefill work to the full prompt; `ScheduledBatchExecutor` slices the current chunk from runtime position.
6. Execute the selected batch through the stateful runtime and `ContinuousBatchExecutor`.
7. Commit backend tokens only for decode work.
8. When decode reaches backend EOS or `max_new_tokens`, transition the runtime sequence to terminal state, remove it from runtime ownership, and return its KV page leases to the pool.

Only one scheduling cycle may commit at a time. Producers may still submit requests concurrently, and device work inside a selected cycle can be coalesced by `ContinuousBatchExecutor`.

## Boundary rules

- F# remains the policy owner. `ISchedulingKernel` is a DTO/interface boundary so the C# engine does not depend on F# union representation.
- Runtime state is authoritative for sequence phase, position, KV availability, and token-block size.
- Request state is authoritative for prompt data, priority/deadline, generated-token history, and generation limits.
- Prefill backend outputs are not user-visible generated tokens.
- Terminal engine-owned sequences must release runtime ownership; completed requests remain queryable as immutable snapshots while their KV memory is reusable.
- An empty scheduled batch while active requests remain is treated as a no-progress condition by `RunUntilCompleteAsync` rather than spinning indefinitely.

This layer is intentionally backend-agnostic. Replacing the deterministic backend with ONNX/CUDA should not change scheduler or request lifecycle semantics.
