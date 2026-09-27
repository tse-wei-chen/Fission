# Migration transport planning and admission

Fission now separates migration transaction correctness from physical transport selection.

The transaction protocol in `migration-transfer-protocol.md` defines when source state is prepared, target state is imported, source state is committed, and rollback runs. This layer defines how a future transport-aware backend/fabric can describe candidate physical paths and how the runtime can bound migration pressure before starting a transfer.

## Transport-aware backend contract

`ISequenceMigrationTransportBackend` extends `ISequenceMigrationBackend` with two planning inputs:

- peer-specific `SequenceMigrationTransportCapability` records;
- authoritative live-state byte estimation through `EstimateSequenceMigrationBytesAsync`.

The runtime must not infer physical transfer bytes from KV page count or scheduler accounting. Concrete backends know tensor geometry, element widths, retained decoder state, compression/packing, and transport-specific metadata, so the source backend is the byte-estimation authority.

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

The result is a `SequenceMigrationTransportPlan`. The planner has no CUDA/NIXL-specific logic; new transport implementations can participate by advertising capabilities rather than changing runtime policy code.

## Admission

`SequenceMigrationAdmissionController` bounds two independent resources:

- total estimated bytes currently in migration transactions;
- maximum concurrent migration transactions.

`AcquireAsync` returns a lease. The lease is intended to cover the entire physical transaction, including rollback, so reserved bandwidth/memory pressure is not released while target cleanup or source restoration is still active.

A migration larger than the total configured byte budget is rejected immediately. A migration that fits the total budget but would exceed current in-flight pressure waits for capacity and observes caller cancellation.

## Current integration boundary

This milestone defines and tests the transport planning/admission primitives but does **not yet wire them into `ExecutionPlanExecutor`**. Existing cross-actor migration continues to use the prepare/import/commit/abort protocol from #49.

The next integration step is:

1. detect that both source and target implement `ISequenceMigrationTransportBackend`;
2. serialize source byte estimation with sequence state;
3. obtain peer capabilities;
4. create a transport plan;
5. acquire a migration-admission lease;
6. pass the selected plan into a transport-aware prepare operation;
7. hold the lease through commit or abort.

That integration also needs an explicit way for a backend transfer token to attest which transport plan it materialized. A concrete ONNX/CUDA implementation should follow after that identity boundary is in place.
