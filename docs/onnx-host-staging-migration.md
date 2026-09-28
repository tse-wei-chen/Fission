# ONNX Runtime host-staging migration

Fission has a concrete physical implementation of the cross-device migration protocol for decoder state owned by the ONNX Runtime backend.

The implementation is intentionally opt-in. `OnnxRuntimeBackend` keeps its existing non-transactional behavior, while `OnnxRuntimeMigratableBackend` requires an execution adapter that implements `IOnnxRuntimeSequenceMigrationAdapter` and advertises an actual physical-state codec.

## Data path

For `DecoderOnlyOnnxExecutionAdapter`, migration is enabled when its binding implements `IDecoderOrtHostStagingBinding`.

```text
source DecoderOrtState
        |
        | deep-copy K/V tensors
        v
host-staging payload
        |
        | transport-plan attestation
        v
target binding import
        |
        | create target-owned OrtValue views
        v
target DecoderOrtState
```

Prepare exports a deep copy of every K/V tensor plus the decoder causal frontier (`Position` and `NextTokenId`). Import reconstructs a distinct target `DecoderOrtState`. Commit releases source physical state only after target import succeeds. Abort removes target state and, if a destructive source commit failed, can reconstruct the source state from the staged payload.

## Payload ownership

`DecoderOrtHostStagingPayload` is ref-counted. The migration transfer starts with one owner reference. A binding whose imported `OrtValue` instances are views over payload memory calls `payload.Retain()` and passes that lease as the imported `DecoderOrtState` lifetime anchor.

The source-side terminal operation owns release of the transfer reference:

- successful source commit releases the transfer owner after source state is retired;
- source abort releases it immediately when source state never disappeared;
- destructive-commit rollback first reconstructs source state with its own retained payload lease, then releases the transfer owner.

`DecoderOrtState` releases Fission-owned host-staging lifetime leases only after all of its `OrtValue` instances are disposed. Ordinary caller-supplied lifetime-anchor objects remain reference anchors only and are not automatically disposed. This keeps public state-anchor ownership compatible while allowing pooled staging memory to be reclaimed deterministically.

Prepare also disposes a newly exported payload if format/frontier/byte validation fails before a transfer token can be published.

## Optimum legacy FP32 codec

`OptimumLegacyFloatHostStagingBinding` decorates `OptimumLegacyFloatDecoderBinding` and provides the first production-shaped codec. It:

- preserves the existing prefill/decode/batched execution behavior;
- estimates transfer bytes from decoder geometry;
- deep-copies FP32 key/value tensors into exact-length GC-pinned managed arrays on export;
- pools returned staging arrays by exact element length so repeated same-shape migrations avoid fresh pinned allocations;
- bounds retention by both buffers-per-length and total retained bytes;
- clears returned buffers by default before retaining them in the pool;
- validates layer count, geometry, byte count, format id, position, and next-token frontier;
- creates distinct target `OrtValue` instances over the staged arrays on import;
- advertises a format-specific host-staging transport id so incompatible source/target bindings fail transport negotiation before export.

`PinnedHostStagingPoolOptions` controls retention and clear-on-return policy. `PinnedHostStagingPoolStatistics` exposes allocation, reuse, return/drop, and retained-buffer counters so production tests and telemetry can distinguish allocation pressure from transport bytes.

The transport id includes model identity and decoder geometry/format. Host staging therefore participates in the same deterministic transport planner, byte/concurrency admission control, timeout classification, and migration tracing as future CUDA P2P, IPC, NIXL, or RDMA transports.

## What “pinned” means here

The current pool uses `.NET` GC-pinned managed arrays (`GC.AllocateUninitializedArray<T>(..., pinned: true)`). The array object has a stable managed address for its lifetime and can back existing `OrtValue.CreateTensorValueFromMemory` calls without another managed copy.

This is **not** yet CUDA page-locked host memory allocated with CUDA host-allocation/registration APIs, and this milestone does not claim asynchronous GPU DMA. It establishes the ownership, pooling, stable-address, retention, and rollback lifetime model that a later CUDA-specific allocator can implement behind the same staging boundary.

## Scope and performance

This path remains a correctness-first physical transport, but repeated Optimum same-shape migrations no longer require fresh managed K/V arrays for every staged transfer. GC-pinned pooling also avoids moving the backing arrays while imported `OrtValue` views are alive.

The next GPU-specific step is to introduce a native page-locked allocator / registration layer and asynchronous device-copy primitive, then advertise a measured transport capability. CUDA P2P/IPC, NIXL/RDMA, topology measurement, and cross-process failure recovery remain separate transport milestones.

## Executable specs

`tests/Fission.OnnxRuntime.Migration.Specs` continues to cover:

- end-to-end migration through `ExecutionPlanExecutor` and two device actors;
- source-state release only after successful target import/commit;
- target decode continuing from the migrated causal frontier;
- byte/concurrency admission release after migration;
- negotiation failure for incompatible codec ids before physical export;
- Optimum FP32 host-staging export/import round-trip and source/target memory independence.

`tests/Fission.OnnxRuntime.PinnedStaging.Specs` additionally covers:

- transfer-owner release while an imported state still retains the payload;
- deterministic final payload reclamation after `OrtValue` disposal;
- exact-length pinned buffer allocation and same-shape reuse;
- retained-byte/buffer accounting;
- bounded pool retention and dropped-buffer accounting.
