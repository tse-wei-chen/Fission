# Backend state transactions

Fission treats model-owned inference state as part of the same logical state machine as runtime KV metadata.

A stateful backend may retain logits, physical KV tensors, decoder buffers, or other per-sequence objects between prefill and decode. Snapshot, fork, restore, and release therefore cannot update only `KvPageTable`; the backend must observe the same transaction in the same order.

## Device-actor ordering

`ContinuousBatchExecutor` is the serialization boundary for backend work.

Inference work may be micro-batched, but transaction controls are barriers. Given queue order:

```text
decode(A)
snapshot(A)
decode(A)
```

the executor flushes the first inference segment, executes the snapshot control, then starts the later inference segment. The two decodes cannot be coalesced ahead of the snapshot.

The serialized backend controls are:

- `SnapshotSequenceAsync`
- `ForkSequenceAsync`
- `RestoreSequenceAsync`
- `ReleaseSnapshotAsync`
- `ReleaseSequenceAsync`

Stateless `IInferenceBackend` implementations may use the default no-op transaction hooks. Stateful adapters must implement the operations they support.

## Runtime reservations

A device barrier alone is not sufficient. Two concurrent plans targeting the same sequence could otherwise sample metadata at one position while their backend work is queued in a different order.

`ExecutionPlanExecutor` therefore reserves every sequence targeted by a plan for the full plan lifetime. Restore plans also reserve the referenced snapshot. A conflicting operation is rejected before it can enqueue backend work.

This does not serialize the whole runtime: plans for different sequences still execute concurrently and can converge into the same device micro-batch.

Sequence and snapshot release use the same reservation mechanism, so a state object cannot be released while another plan is mutating or restoring it.

## Transaction ordering

For snapshot creation:

1. Create/acquire the metadata snapshot.
2. Execute the backend snapshot barrier.
3. Publish the snapshot into runtime ownership.
4. If backend snapshot creation fails, dispose the unpublished metadata snapshot.

For fork:

1. Materialize unpublished metadata branches.
2. Execute the backend fork barrier for all branch IDs.
3. Publish metadata branches.
4. If publication fails, dispose metadata branches and best-effort release backend branch state.

For restore:

1. Validate runtime ownership of sequence and snapshot.
2. Execute the backend restore barrier.
3. Restore metadata KV and position.

For release:

1. Reserve the state object.
2. Release backend-owned state on the device actor.
3. Remove runtime ownership.
4. Dispose metadata KV leases.

Backend release happens before metadata removal. If backend cleanup fails, runtime metadata remains available for diagnostics or retry rather than creating orphaned native state.

## ONNX Runtime adapters

`OnnxRuntimeBackend` forwards transaction controls to `IOnnxRuntimeExecutionAdapter`.

The ONNX adapter boundary owns model/export-specific physical state semantics. An adapter that does not implement snapshot/fork/restore throws `NotSupportedException`; Fission must not silently pretend metadata-only branching is valid for a stateful model.

A future causal-LM adapter can implement snapshots with immutable/shared OrtValue state, copy-on-write KV pages, or another representation without changing scheduler, engine, or serving contracts.

## Current scope

This protocol covers snapshot, fork, restore, snapshot release, and sequence release. Backend-aware device migration is intentionally separate work; `MigrateKvExecutionStep` must not be treated as transparent physical-state migration for a stateful backend until a migration transaction hook is added.
