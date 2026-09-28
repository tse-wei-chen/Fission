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
- deep-copies FP32 key/value tensors into exact-length pooled host buffers on export;
- uses GC-pinned managed arrays by default;
- pools returned staging buffers by exact element length so repeated same-shape migrations avoid fresh physical allocations;
- bounds retention by both buffers-per-length and total retained bytes;
- clears returned buffers by default before retaining them in the pool;
- validates layer count, geometry, byte count, format id, position, and next-token frontier;
- creates distinct target `OrtValue` instances over the staged `Memory<float>` on import;
- advertises a format-specific host-staging transport id so incompatible source/target bindings fail transport negotiation before export.

`PinnedHostStagingPoolOptions` controls retention and clear-on-return policy. `PinnedHostStagingPoolStatistics` exposes allocation, reuse, return/drop, and retained-buffer counters so production tests and telemetry can distinguish allocation pressure from transport bytes.

The transport id includes model identity and decoder geometry/format. Host staging therefore participates in the same deterministic transport planner, byte/concurrency admission control, timeout classification, and migration tracing as future CUDA P2P, IPC, NIXL, or RDMA transports.

## Physical host-allocation boundary

The pool does not assume that a staging allocation is a managed `float[]`.

`IHostStagingFloatBufferAllocator` creates exact-length `IHostStagingFloatBuffer` instances, and each buffer exposes a `Memory<float>` whose lifetime is controlled by the existing payload lease. `OptimumLegacyFloatHostStagingBinding` imports that memory through ONNX Runtime's `OrtValue.CreateTensorValueFromMemory(Memory<T>, ...)` path.

Allocator implementations own only individual physical buffers. The pool owns reuse policy; the migration payload owns active buffer leases. This separation prevents physical allocation/release rules from leaking into the transaction protocol.

## GC-pinned default allocator

`GcPinnedHostStagingFloatBufferAllocator` remains the default. It uses `.NET` pinned-object-heap arrays (`GC.AllocateUninitializedArray<T>(..., pinned: true)`) so existing users do not acquire a CUDA runtime dependency.

This allocator gives the managed array a stable address but does not make the allocation CUDA page-locked and does not by itself enable asynchronous GPU DMA.

## CUDA page-locked allocator

`CudaPageLockedHostStagingFloatBufferAllocator` is an opt-in physical allocator backed by the CUDA Runtime API:

- `cudaHostAlloc` allocates exact-length page-locked host memory;
- `cudaFreeHost` releases each physical allocation;
- a custom `MemoryManager<float>` exposes the native allocation as `Memory<float>` without an intermediate managed array;
- a `SafeHandle` owns the native allocation so deterministic payload teardown is backed by a finalizer-safe native resource boundary;
- `CudaHostAllocationFlags.Portable` is the default because migration staging can cross CUDA contexts/devices;
- `Mapped` and `WriteCombined` remain explicit opt-ins;
- `TryCreate` provides a non-throwing cudart availability probe;
- native allocation failures surface as `CudaRuntimeException` with the CUDA error code and, when available, CUDA error text.

The allocator dynamically loads the CUDA Runtime instead of adding a mandatory CUDA package/runtime dependency to normal Fission builds. Callers can also provide an explicit runtime-library path/name through `CudaPageLockedHostStagingAllocatorOptions`.

Page-locked memory is a scarce system resource. The same exact-length pool and `MaxRetainedBytes` bound therefore remain in force for CUDA-backed staging, rather than retaining native pinned allocations without limit.

This milestone changes physical host memory only. Export still copies K/V bytes synchronously from the current `OrtValue` state into host staging, and import still constructs CPU-memory `OrtValue` views. It does **not** yet call `cudaMemcpyAsync`, manage CUDA streams/events, or claim overlapped GPU DMA.

## Scope and performance

Repeated same-shape migrations can now reuse either GC-pinned or CUDA page-locked physical staging allocations through the same pool. Payload reference counting still prevents a returned buffer from being reused while any imported `OrtValue` aliases it.

The next GPU-specific step is an asynchronous copy primitive (`cudaMemcpyAsync` plus stream/event completion ownership) and a transport binding that uses it without weakening the current transaction/rollback semantics. CUDA P2P/IPC, NIXL/RDMA, topology measurement, and cross-process failure recovery remain separate transport milestones.

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
- bounded pool retention and dropped-buffer accounting;
- custom allocator injection and deterministic physical-buffer disposal;
- CUDA page-locked allocator byte/flag forwarding through a fake CUDA Runtime boundary;
- native `Memory<float>` import through ONNX Runtime while payload ownership remains alive;
- deterministic `cudaFreeHost` lifetime after final imported-state disposal;
- CUDA allocation-error propagation and non-throwing cudart availability probing.
