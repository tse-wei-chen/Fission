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

## Success path

For a sequence moving from actor A to actor B:

1. The plan reserves the sequence, preventing concurrent runtime operations on that sequence.
2. A prepare control is serialized through actor A.
3. Actor A returns a transfer token while retaining enough source state for rollback.
4. An import control is serialized through actor B.
5. Actor B creates runnable target state without requiring runtime placement to change yet.
6. A commit control is serialized through actor A.
7. Actor A may release or retire source state and transport resources.
8. Only after commit returns does the runtime update `SequenceProcess.Device`.
9. Later decode/release operations resolve to actor B.

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

If rollback itself fails, `ExecutionPlanExecutor` raises an `AggregateException` containing the original migration failure plus rollback failures. Runtime placement is not advanced.

Prepare failure has no transfer token available to the runtime, so prepare must be self-rollbacking: a thrown prepare operation must leave source state usable.

## Device-actor failure isolation

Migration rollback requires the actor to survive control-operation exceptions. `ContinuousBatchExecutor` therefore treats transaction-control failure differently from inference failure:

- control exception: complete that control with the exception and continue pumping later work;
- inference exception: fail the actor because a partially executed batch may have ambiguous model state.

This means a failed target import can be followed by target abort, and a failed source commit can be followed by source abort on the same actors.

## Compatibility path

The explicit protocol is used only when source and target are different actors and both backends implement `ISequenceMigrationBackend`.

Otherwise the runtime falls back to `IInferenceBackend.MigrateSequenceAsync`. This preserves existing one-actor internal routing and older fabric backends. The legacy hook remains responsible for making the logical target usable before it reports success, but it does not provide the runtime-visible prepare/import/commit rollback phases.

## Transport mapping

The protocol intentionally does not prescribe a byte format. Likely transport implementations include:

- same-node CUDA peer-to-peer copies;
- CUDA IPC or shared GPU-memory handles;
- pinned-host staging when direct P2P is unavailable;
- NIXL or RDMA descriptors for cross-process/cross-node migration;
- immutable/shared KV representations when devices can reference common storage.

A future topology-aware migration planner can choose among these transports without changing `ExecutionPlan`, sequence metadata, or scheduler contracts.

## Remaining work

The transaction boundary is now explicit, but production migration still needs concrete transfer implementations, topology and bandwidth admission, timeout/health classification, observability for migration bytes and latency, and process/node failure recovery. The current protocol handles synchronous runtime rollback while both device actors remain alive.