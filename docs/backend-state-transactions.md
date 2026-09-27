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

The serialized backend controls include:

- `SnapshotSequenceAsync`
- `ForkSequenceAsync`
- `RestoreSequenceAsync`
- legacy `MigrateSequenceAsync`
- transactional migration prepare/import/commit/abort
- `ReleaseSnapshotAsync`
- `ReleaseSequenceAsync`

Transaction-control exceptions fail only the submitted control and do not terminate the device actor. This is required so migration rollback can enqueue abort work after an import or commit failure. Inference execution failures remain actor-fatal because a failed backend batch may have partially advanced model-owned state.

Stateless `IInferenceBackend` implementations may use the default no-op transaction hooks. Stateful adapters must implement the operations they support.

## Runtime reservations

A device barrier alone is not sufficient. Two concurrent plans targeting the same sequence could otherwise sample metadata at one position while their backend work is queued in a different order.

`ExecutionPlanExecutor` therefore reserves every sequence targeted by a plan for the full plan lifetime. Restore plans also reserve the referenced snapshot. A conflicting operation is rejected before it can enqueue backend work.

This does not serialize the whole runtime: plans for different sequences still execute concurrently and can converge into the same device micro-batch.

Sequence and snapshot release use the same reservation mechanism, so a state object cannot be released while another plan is mutating or restoring it.

## Transaction ordering

For snapshot creation:

1. Create/acquire the metadata snapshot.
2. Execute the backend snapshot barrier on the sequence's current actor.
3. Publish the snapshot into runtime ownership together with its device placement.
4. If backend snapshot creation fails, dispose the unpublished metadata snapshot.

For fork:

1. Materialize unpublished metadata branches.
2. Execute the backend fork barrier for all branch IDs on the parent sequence actor.
3. Publish metadata branches with the parent's current placement.
4. If publication fails, dispose metadata branches and best-effort release backend branch state.

For restore:

1. Validate runtime ownership of sequence and snapshot.
2. Resolve both sequence placement and snapshot ownership to execution actors.
3. Reject the restore if they belong to different actors.
4. Execute the backend restore barrier.
5. Restore metadata KV and position.

For migration, the preferred cross-actor path is the optional `ISequenceMigrationBackend` protocol:

1. Reserve the sequence for the plan lifetime.
2. Validate that the target `DeviceId` resolves to a registered actor.
3. `PrepareSequenceMigrationAsync` on the source actor and obtain an opaque `SequenceMigrationTransfer`.
4. `ImportSequenceMigrationAsync` on the target actor.
5. `CommitSequenceMigrationAsync` on the source actor.
6. Commit `SequenceProcess.Device` only after source commit succeeds.
7. Route later inference and release to the target actor.

The transfer token carries transaction, sequence, source-device, and target-device identity. Its backend-specific payload is represented by the concrete transfer type; the runtime does not inspect physical transport state.

If import or commit fails, the runtime keeps `SequenceProcess.Device` and sequence version unchanged, then invokes `AbortSequenceMigrationAsync` on target and source using `CancellationToken.None`. Target abort must remove any partially imported state. Source abort must restore source state even when commit failed after destructive work. Abort should therefore be idempotent for a transfer that may only have been partially applied.

Prepare failure is different: because no transfer token reaches the runtime, a backend prepare operation must leave the source usable before propagating its exception.

When source and target resolve to the same actor, or when both actors do not advertise `ISequenceMigrationBackend`, Fission keeps the existing legacy `MigrateSequenceAsync` hook. This preserves one-actor backend-internal routing and existing migration-capable fabrics while allowing newer backends to opt into explicit rollback semantics.

For release:

1. Reserve the state object.
2. Resolve the actor that owns the sequence or snapshot.
3. Release backend-owned state on that actor.
4. Remove runtime ownership.
5. Dispose metadata KV leases.

Backend release happens before metadata removal. If backend cleanup fails, runtime metadata remains available for diagnostics or retry rather than creating orphaned native state.

## ONNX Runtime adapters

`OnnxRuntimeBackend` forwards the legacy transaction controls to `IOnnxRuntimeExecutionAdapter`.

The current ONNX adapter does not implement physical cross-device migration and therefore still rejects migration. A future causal-LM adapter may implement `ISequenceMigrationBackend` at the backend/fabric layer and use immutable/shared OrtValue state, CUDA IPC, staged host transfer, NIXL/RDMA, or another transport without changing scheduler or plan contracts.

## Current scope

The runtime supports multiple registered device actors, placement-aware routing, snapshot locality, per-device scheduler envelopes, explicit source/target migration transactions, rollback after partial target import or source commit failure, and control-failure isolation. It still does not provide a concrete CUDA/NIXL/RDMA transfer implementation, topology-aware placement policy, bandwidth admission, or distributed recovery after process/node loss. See `multi-device-runtime.md` and `migration-transfer-protocol.md` for the current boundaries.