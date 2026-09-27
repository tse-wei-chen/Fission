# Migration transport planning and admission

Fission separates migration transaction correctness from physical transport selection.

The transaction protocol in `migration-transfer-protocol.md` defines when source state is prepared, target state is imported, source state is committed, and rollback runs. This layer defines how transport-aware backends describe candidate physical paths and how the runtime bounds migration pressure before starting a transfer.

## Transport-aware backend contract

`ISequenceMigrationTransportBackend` extends `ISequenceMigrationBackend` with three transport-specific operations:

- peer-specific `SequenceMigrationTransportCapability` records;
- authoritative live-state byte estimation through `EstimateSequenceMigrationBytesAsync`;
- planned prepare through `PrepareSequenceMigrationAsync(..., SequenceMigrationTransportPlan, ...)`.

The runtime does not infer physical transfer bytes from KV page count or scheduler accounting. Concrete backends know tensor geometry, element widths, retained decoder state, compression/packing, and transport-specific metadata, so the source backend is the byte-estimation authority.

A capability contains:

- `TransportId`: compatibility key shared by source and target;
- `Kind`: broad family such as direct device, shared memory, host staging, or remote memory;
- optional per-transfer byte limit (`0` means no backend-imposed cap);
- estimated bandwidth;
- fixed latency;
- backend preference used only as a deterministic tie-break after estimated duration.

Both peers must advertise the same `TransportId` and `Kind`. Matching only the broad transport kind is intentionally insufficient because two implementations of, for example, remote memory may not share descriptor or registration semantics.

## Planner

`SequenceMigrationTransportPlanner` takes the source capabilities, target capabilities, and backend-provided estimated bytes.

For each compatible path it computes conservative effective values:

- max transfer bytes: minimum non-zero peer limit;
- bandwidth: minimum peer estimate;
- fixed latency: maximum peer estimate;
- preference: minimum peer preference.

Paths that cannot carry the estimated bytes are discarded. Remaining paths are ordered by:

1. shortest estimated completion time;
2. highest effective preference;
3. ordinal `TransportId` for deterministic tie-breaking.

The result is a `SequenceMigrationTransportPlan`. The planner has no CUDA/NIXL-specific logic; new transport implementations participate by advertising capabilities rather than changing runtime policy code.

## Runtime transaction integration

When source and target are different actors and both implement `ISequenceMigrationTransportBackend`, `ExecutionPlanExecutor` now executes this sequence:

1. reserve the sequence for the plan lifetime;
2. serialize source physical-byte estimation through the source actor;
3. serialize peer-capability discovery through the source and target actors;
4. select one `SequenceMigrationTransportPlan`;
5. acquire a `SequenceMigrationAdmissionController` lease for the estimated bytes;
6. run planned prepare on the source actor with the selected plan;
7. require the returned `SequenceMigrationTransfer.TransportPlan` to equal the selected plan;
8. import on the target actor;
9. commit on the source actor;
10. publish `SequenceProcess.Device` only after commit succeeds;
11. release admission only after success or rollback fully finishes.

If only the transaction protocol is supported, but one side does not expose transport planning, the runtime keeps the #49 prepare/import/commit/abort path unchanged. One-actor/internal-routing migration still uses the legacy `IInferenceBackend.MigrateSequenceAsync` compatibility path.

Transport capability discovery, byte estimation, and planned prepare are device-actor controls. The runtime therefore does not call backend transport APIs out of band or concurrently with actor-owned backend state.

## Transfer-plan attestation

`SequenceMigrationTransfer` now optionally carries a `TransportPlan`.

Legacy protocol transfers leave it null. A transport-aware planned prepare must return a transfer whose `TransportPlan` equals the runtime-selected plan. This prevents a backend from silently substituting another physical path after admission and scheduling decisions were made.

A missing or mismatched attestation is treated as a migration transaction failure before target import. The runtime keeps logical placement on the source, source-aborts the prepared transfer, and keeps the admission lease until that abort completes.

## Admission

`SequenceMigrationAdmissionController` bounds two independent resources:

- total estimated bytes currently in migration transactions;
- maximum concurrent migration transactions.

`AcquireAsync` returns a lease. The runtime holds that lease across planned prepare, target import, source commit, and target/source rollback. Reserved migration pressure is therefore not released while target cleanup or source restoration is still active.

A migration larger than the total configured byte budget is rejected immediately. A migration that fits the total budget but would exceed current in-flight pressure waits for capacity and observes caller cancellation. Cancellation while waiting occurs before prepare, so no backend transfer state is allocated and runtime placement remains unchanged.

`ExecutionPlanExecutor` uses an effectively unbounded private admission controller by default to preserve existing behavior. Callers that need real migration pressure limits can inject a shared `SequenceMigrationAdmissionController` together with a planner. The injected admission controller is caller-owned and can coordinate multiple runtime executors under one budget.

## Current physical boundary

The runtime now has an end-to-end transport-aware control plane:

```text
estimate bytes
  -> discover source/target capabilities
  -> select transport plan
  -> acquire admission
  -> prepare(source, plan)
  -> verify transfer attestation
  -> import(target)
  -> commit(source)
  -> publish placement
  -> release admission
```

Failures after prepare run target/source abort as appropriate while the admission lease is still held.

What remains is a concrete physical transport implementation. A CUDA backend can now map `DirectDevice` to peer copies or CUDA IPC, `HostStaging` to pinned-host buffers, and a distributed backend can map `RemoteMemory` to NIXL/RDMA descriptors without changing `ExecutionPlanExecutor` transaction ordering.
