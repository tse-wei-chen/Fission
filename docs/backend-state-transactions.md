# Backend state transactions

Fission treats model-owned inference state as part of the same logical state machine as runtime KV metadata.

A stateful backend may retain logits, physical KV tensors, decoder buffers, or other per-sequence objects between prefill and decode. Snapshot, fork, restore, migrate, and release therefore cannot update only `KvPageTable` or `SequenceProcess.Device`; the backend must observe the same transaction in the same order.

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
- `MigrateSequenceAsync`
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

For migration:

1. Reserve the sequence for the plan lifetime.
2. Execute the backend migration barrier with the target `DeviceId`.
3. Commit `SequenceProcess.Device` only after backend migration succeeds.
4. If the backend migration fails, leave runtime device metadata and sequence version unchanged.

A migration-capable backend owns the physical transfer or rerouting semantics. The current runtime still submits subsequent sequence work through the same device actor, so a backend that reports migration success must continue to route that sequence correctly after the move. The in-process ONNX Runtime adapter does not yet implement this and therefore rejects migration explicitly.

For release:

1. Reserve the state object.
2. Release backend-owned state on the device actor.
3. Remove runtime ownership.
4. Dispose metadata KV leases.

Backend release happens before metadata removal. If backend cleanup fails, runtime metadata remains available for diagnostics or retry rather than creating orphaned native state.

## ONNX Runtime adapters

`OnnxRuntimeBackend` forwards transaction controls to `IOnnxRuntimeExecutionAdapter`.

The ONNX adapter boundary owns model/export-specific physical state semantics. An adapter that does not implement snapshot/fork/restore/migrate throws `NotSupportedException`; Fission must not silently pretend metadata-only state changes are valid for a stateful model.

A future causal-LM adapter can implement snapshots with immutable/shared OrtValue state, copy-on-write KV pages, migration to another execution provider/device, or another representation without changing scheduler, engine, or serving contracts.

## Current scope

This protocol covers snapshot, fork, restore, migration commit ordering, snapshot release, and sequence release. It does not yet provide a multi-device actor fabric or a concrete ONNX/CUDA KV transfer mechanism. Those are the next layer required for transparent cross-device process migration.
