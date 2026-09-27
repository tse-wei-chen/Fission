# Multi-device runtime routing

Fission can register more than one `ContinuousBatchExecutor` behind one `ExecutionPlanExecutor` by using `ExecutionDeviceRegistry`.

Each registered executor remains a single-reader device actor. The registry adds placement-aware routing; it does not merge backend mutation across actors.

## Placement

A new sequence starts on the registry's default device. After that, `SequenceProcess.Device` determines which actor receives its prefill continuation, decode, snapshot, fork, restore, and release work.

Migration is a two-layer transaction:

1. reserve the sequence for the plan lifetime,
2. validate the target actor when the runtime has multiple registered actors,
3. execute `MigrateSequenceAsync` on the current/source actor,
4. require the backend/fabric to make the target actor able to continue the sequence before the hook completes,
5. commit `SequenceProcess.Device`,
6. route subsequent inference and release work to the target actor.

The runtime does not serialize or copy backend-specific KV itself. A migration-capable backend may use P2P copy, RDMA, shared host memory, a distributed cache service, or another transfer mechanism behind the migration hook.

## Single-actor compatibility

A registry containing one actor preserves the earlier backend-internal migration model. The logical `DeviceId` may change even when that target is not separately registered; later work still enters the same actor and the backend is responsible for internal routing.

With two or more actors, a migration target must be explicitly registered. This prevents runtime metadata from committing placement to an actor that cannot receive future work.

## Snapshot locality

Runtime snapshot ownership records the logical device where backend snapshot state was created.

Restore is allowed only when the snapshot and sequence resolve to the same execution actor. Migrating a sequence to another registered actor does not silently migrate old snapshot state. Snapshot release always returns to the actor that owns that snapshot.

Cross-device snapshot transfer is a separate future transaction.

## Scheduler envelopes

`ScheduledBatchExecutor` forms one atomic submission envelope per execution actor. A scheduling batch may therefore contain work for multiple actors while preserving deterministic scheduler order independently inside each actor.

Capacity preflight is also per actor. For example, two actors with capacity 1 may execute a two-sequence scheduling batch when one sequence is placed on each actor.

The current request-level scheduler still uses the minimum registered actor capacity as its conservative global feedback. Placement-aware scheduling and aggregate multi-device capacity are later policy work.

## Current boundary

This milestone provides runtime actor registration, placement-aware routing, target admission, snapshot locality, and per-device scheduled batching. It does not provide a concrete CUDA/NIXL/RDMA KV transfer implementation, automatic load balancing, or scheduler-driven device placement.