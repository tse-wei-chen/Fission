# Migration transfer protocol

Fission's cross-device runtime distinguishes logical placement from physical model state. A `SequenceProcess.Device` change is committed only after the backend has established runnable state on the target actor.

## Why a multi-phase protocol is required

A single migration callback is sufficient when a backend owns the entire device fabric internally. It is not sufficient when source and target are independent device actors and failures can happen after the target starts importing state.

The optional `ISequenceMigrationBackend` contract introduces an opaque `SequenceMigrationTransfer` and four backend phases:

```text
source actor                  target actor
     |                             |
     | prepare(sequence, target)   |
     |----------------------------> transfer token
     |                             |
     |                 import(transfer)
     |                             |
     | commit(transfer)            |
     |                             |
     | runtime placement commit    |
```

`SequenceMigrationTransfer` identifies one migration transaction and carries sequence/source/target identity. Concrete backends derive their own transfer type and may attach host buffers, CUDA IPC handles, NIXL descriptors, RDMA registrations, immutable-state references, or other transport-specific state.

Transport-aware backends additionally attach the runtime-selected `SequenceMigrationTransportPlan` to the transfer token. Legacy protocol transfers leave that field null.

## Success path

For a sequence moving from actor A to actor B:

1. The plan reserves the sequence, preventing concurrent runtime operations on that sequence.
2. If both actors are transport-aware, the runtime serializes physical-byte estimation and peer-capability discovery through the actors, selects one transport plan, and acquires migration admission.
3. A prepare control is serialized through actor A. Transport-aware prepare receives the exact selected plan.
4. Actor A returns a transfer token while retaining enough source state for rollback.
5. For transport-aware migration, the runtime verifies that the transfer token attests the selected plan.
6. An import control is serialized through actor B.
7. Actor B creates runnable target state without requiring runtime placement to change yet.
8. A commit control is serialized through actor A.
9. Actor A may release or retire source state and transport resources.
10. Only after commit returns does the runtime update `SequenceProcess.Device`.
11. Later decode/release operations resolve to actor B.
12. Transport admission is released after the physical transaction is complete.

The runtime does not publish an intermediate placement.

## Failure path

Import and commit are allowed to fail after partial physical work. When either fails, the runtime keeps logical placement on the source and executes rollback with caller cancellation suppressed:

```text
failure
  |
  +--> abort(target)  // remove partial imported state
  |
  +--> abort(source)  // restore source after partial/destructive commit
```

Target abort is attempted whenever target import was attempted, even if import itself threw. Backends should make abort idempotent because the target may have applied none, some, or all of the import before reporting failure.

Source abort must restore a runnable source sequence when commit throws after destructive work. This lets the original sequence continue decoding after rollback.

A transport-plan attestation mismatch is detected before target import. The source transfer is aborted, target abort is skipped, logical placement remains unchanged, and migration admission is held until source abort finishes.

If rollback itself fails, `ExecutionPlanExecutor` raises an `AggregateException` containing the original migration failure plus rollback failures. Runtime placement is not advanced.

Prepare failure has no transfer token available to the runtime, so prepare must be self-rollbacking: a thrown prepare operation must leave source state usable.

## Device-actor failure isolation

Migration rollback requires the actor to survive control-operation exceptions. `ContinuousBatchExecutor` therefore treats transaction-control failure differently from inference failure:

- control exception: complete that control with the exception and continue pumping later work;
- inference exception: fail the actor because a partially executed batch may have ambiguous model state.

Transport capability discovery, byte estimation, planned prepare, import, commit, and abort all execute through the device actor rather than touching backend state out of band.

## Compatibility paths

There are three migration levels:

1. **Transport-aware two-actor protocol**: source and target both implement `ISequenceMigrationTransportBackend`; runtime plans/admit the physical transfer and then runs prepare/import/commit/abort.
2. **Transactional two-actor protocol**: source and target both implement `ISequenceMigrationBackend`, but one or both do not expose transport planning; runtime runs the #49 prepare/import/commit/abort transaction without an explicit transport plan.
3. **Legacy/internal routing**: one actor owns the fabric or the two-actor transaction protocol is unavailable; runtime uses `IInferenceBackend.MigrateSequenceAsync`.

The legacy hook remains responsible for making the logical target usable before it reports success, but it does not provide runtime-visible prepare/import/commit rollback phases.

## Transport mapping

The protocol intentionally does not prescribe a byte format. Transport-aware backends can advertise paths such as:

- same-node CUDA peer-to-peer copies;
- CUDA IPC or shared GPU-memory handles;
- pinned-host staging when direct P2P is unavailable;
- NIXL or RDMA descriptors for cross-process/cross-node migration;
- immutable/shared KV representations when devices can reference common storage.

The runtime planner selects among mutually supported capabilities without changing `ExecutionPlan`, sequence metadata, or scheduler contracts.

## First concrete transport: ONNX host staging

`OnnxRuntimeMigratableBackend` and `DecoderOnlyOnnxExecutionAdapter` now provide the first concrete physical implementation of this protocol. A decoder binding that implements `IDecoderOrtHostStagingBinding` can export backend-owned decoder state into a managed host payload and reconstruct a distinct target `DecoderOrtState` from that payload.

`OptimumLegacyFloatHostStagingBinding` supplies an FP32 implementation for the existing Optimum legacy decoder binding. Prepare deep-copies K/V tensors and the causal frontier, import creates target-owned `OrtValue` views over the staged arrays, commit releases source state, and abort can remove target state or reconstruct source state after destructive commit failure. The imported state retains a lifetime anchor to the managed payload until its `OrtValue` views are disposed.

Source and target advertise a model/format-specific host-staging transport id. Incompatible codecs therefore fail mutual transport negotiation before physical export. See `docs/onnx-host-staging-migration.md` for the concrete data path and executable-spec coverage.

## Migration observability

Transport-aware migration emits first-class `ExecutionTraceEvent` records correlated with the originating plan and `MigrateKvExecutionStep`.

The lifecycle is represented explicitly:

```text
MigrationStarted
      |
MigrationPlanned
      |
      +--> MigrationCommitted
      |
      +--> MigrationFailed          // no transfer token existed
      |
      +--> MigrationRolledBack      // transfer existed, abort succeeded
      |
      +--> MigrationRollbackFailed  // one or more abort operations failed
```

Trace metadata includes source and target placement, backend transaction id when one exists, selected transport id and kind, estimated transfer bytes, planner-estimated duration, measured end-to-end elapsed time, triggering exception type, and rollback-failure count.

`MigrationPlanned` is recorded before admission acquisition. This means an admission wait that is later cancelled still leaves enough trace data to explain which physical route and byte budget the runtime intended to consume. Terminal elapsed time includes planning, admission waiting, transfer phases, and rollback when rollback is required.

The existing metadata replay remains driven by `StepCompleted`. A successful migration therefore replays the new target placement, while a failed migration never publishes a completed migration step and leaves replayed placement on the source. The richer migration events are additive evidence for later time-travel debugging and scheduler simulation rather than a replacement for deterministic state reconstruction.

## Remaining work

The control plane now includes transactional rollback, deterministic transport planning, transfer-plan attestation, byte/concurrency admission, one correctness-first ONNX managed-host transport, and plan-correlated migration tracing. Production GPU migration still needs pinned/asynchronous host staging, concrete CUDA P2P/IPC and NIXL/RDMA implementations, measured topology/bandwidth inputs, timeout/health classification, process/node failure recovery, and durable/exportable trace storage. The current protocol handles synchronous runtime rollback while both device actors remain alive.
