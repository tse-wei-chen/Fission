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

For asynchronous CUDA DMA, a staging allocation must additionally implement `ICudaPageLockedHostStagingFloatBuffer`. That capability exposes the stable native host pointer only for buffers whose allocator explicitly promises CUDA page-locked semantics. The ordinary GC-pinned allocator deliberately does not implement it: a stable managed address is not sufficient proof that `cudaMemcpyAsync` may safely use the allocation as an asynchronous host endpoint.

## CUDA asynchronous copy primitive

`CudaAsyncCopyEngine` provides the low-level asynchronous DMA ownership primitive needed by a GPU-resident KV transport.

The engine:

- owns a bounded pool of reusable `cudaStreamNonBlocking` streams;
- submits `cudaMemcpyAsync` with explicit CUDA memcpy direction;
- records `cudaEventDisableTiming` completion events;
- uses one completion pump and `cudaEventQuery` instead of blocking one managed thread per transfer;
- treats `cudaErrorNotReady` as an in-flight state rather than a failure;
- holds a stream slot until the completion event reaches a terminal state;
- if caller cancellation arrives after native submission, delays caller-visible cancellation until the CUDA work has completed, preserving source/destination lifetime safety;
- synchronizes an already-submitted stream when event completion tracking itself fails before allowing stream reuse;
- drains all in-flight work before `DisposeAsync` destroys reusable streams;
- dynamically loads cudart and offers `TryCreate` for a non-throwing runtime availability probe.

The copy primitive is now used by the source-side CUDA D2H staging exporter described below. It still does not, by itself, make a decoder binding GPU-resident.

## CUDA-resident decoder-state boundary

`IDecoderOrtCudaResidentStateBinding` is the explicit capability boundary for bindings that truly own CUDA device-resident K/V tensors. A binding must not implement this interface unless it can expose stable raw CUDA tensor addresses and retain their underlying native allocations independently for the duration of a borrow.

The capability returns a `DecoderOrtCudaResidentStateLease`. Each layer contains `CudaDeviceTensorView` key/value descriptors with:

- the exact CUDA device pointer for the start of the tensor region;
- exact logical byte length;
- tensor element type and shape;
- CUDA device ordinal.

`DecoderOrtCudaResidentStateLease.Create` validates every descriptor against the matching `DecoderOrtState` `OrtValue` before any physical transport is allowed to consume the pointer. Validation requires:

- ONNX Runtime allocator name `Cuda`, rejecting CPU and `CudaPinned` host allocations;
- the same CUDA device ordinal in ORT memory metadata and the raw-pointer descriptor;
- device-default ORT memory rather than host-accessible/pinned memory;
- identical tensor element type and shape;
- exact tensor byte length;
- the same decoder layer count and a non-disposed source state.

The binding transfers an independent `IDisposable` native-allocation retain into the CUDA-resident lease. Validation failure releases that retain immediately. Successful disposal releases it exactly once. The device pointers are only valid while the lease remains alive.

This boundary deliberately does not attempt to recover a CUDA address by reflecting into ONNX Runtime internals. The binding that allocated or otherwise owns the device memory must provide the pointer explicitly; ONNX Runtime's public tensor-memory metadata is used to attest that the associated `OrtValue` is actually a compatible CUDA device tensor.

The GPU-free specs construct synthetic `OrtValue` tensors over unmanaged memory carrying CUDA memory metadata. They validate the contract only; they do not claim the host allocations are real GPU memory and do not execute CUDA inference.

## Asynchronous CUDA device-to-host staging export

`CudaDeviceToHostStagingExporter` is the first physical bridge from validated device-resident KV state into the existing host-staging transaction protocol.

```text
source DecoderOrtState
        |
        | IDecoderOrtCudaResidentStateBinding.AcquireCudaResidentState
        v
validated CUDA K/V pointers + native allocation retain
        |
        | cudaMemcpyAsync(DeviceToHost)
        | bounded non-blocking streams + completion events
        v
exact-length CUDA page-locked pooled host buffers
        |
        | after every DMA reaches terminal completion
        v
DecoderOrtCudaHostStagingPayload
```

The exporter currently supports FP32 KV tensors. It first rents every destination buffer before submitting any native copy, so allocation or page-lock capability failures cannot leave a partially submitted transfer. Every destination must expose `ICudaPageLockedHostStagingFloatBuffer`; stable but ordinary pinned managed memory is rejected before the first `cudaMemcpyAsync` call.

The binding-owned `DecoderOrtCudaResidentStateLease` remains alive until every submitted key/value copy has reached its CUDA completion event. `DecoderOrtCudaHostStagingPayload` is published only after `Task.WhenAll` reaches a terminal state for all copies. This preserves pointer and buffer lifetime on success, native failure, and cancellation. In particular, cancellation after CUDA submission is not allowed to return source device allocations or destination staging buffers while the GPU may still access them.

The payload retains the source CUDA layout format id, exact K/V shapes, causal frontier, and exact transfer-byte accounting. Its page-locked buffers use the existing exact-length staging pool, so repeated same-shape D2H exports can reuse physical host allocations.

`IDecoderOrtAsyncHostStagingBinding` extends the existing host-staging binding contract with an awaitable source export. `DecoderOnlyOnnxExecutionAdapter` detects this capability during migration prepare and awaits it before validating/publishing the transfer. Existing synchronous CPU host-staging bindings continue to use `ExportHostStagingState` unchanged.

This milestone is source-side only. A complete GPU-to-GPU-via-host migration still requires a target-side CUDA H2D importer that allocates target device KV storage, copies the staged buffers with `cudaMemcpyAsync(HostToDevice)`, and publishes a target `DecoderOrtState` only after all H2D completions are terminal.

## Scope and performance

Repeated same-shape migrations can reuse either GC-pinned or CUDA page-locked physical staging allocations through the same pool. Payload reference counting still prevents a returned buffer from being reused while any imported `OrtValue` aliases it.

Fission now has all three source-side primitives for an asynchronous GPU host-staging path: a validated CUDA-resident state borrow, bounded event-driven async copy ownership, and a D2H exporter that does not publish partially copied payloads. The next integration step is the symmetric target-side H2D allocation/import boundary. CUDA P2P/IPC, NIXL/RDMA, topology measurement, and cross-process failure recovery remain separate transport milestones.

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

`tests/Fission.OnnxRuntime.CudaCopy.Specs` covers the GPU-free control/lifetime contract for the async copy engine:

- non-blocking stream creation and disable-timing completion events;
- `cudaErrorNotReady` polling before completion;
- bounded stream concurrency and queued-copy admission;
- cancellation after submission remaining pending until native completion;
- synchronous `cudaMemcpyAsync` failure cleanup and stream reuse;
- event-query failure synchronization/cleanup and stream reuse;
- `DisposeAsync` draining in-flight work before stream destruction;
- non-throwing cudart availability probing.

`tests/Fission.OnnxRuntime.CudaResidentState.Specs` covers the GPU-free device-state capability contract:

- validated CUDA allocator/device/type/shape/byte metadata;
- causal-frontier, device-id, layer-count, and total-byte preservation;
- rejection of `CudaPinned` host memory as device-resident KV;
- rejection of device-ordinal, shape, byte-length, and element-type mismatches;
- deterministic native lifetime release on both validation failure and successful lease disposal;
- disposed-state and disposed-lease guards;
- defensive copying of shape metadata.

`tests/Fission.OnnxRuntime.CudaD2HStaging.Specs` covers the source-side asynchronous staging contract without requiring a GPU:

- D2H export remains incomplete until both key/value CUDA completion events are terminal;
- every native copy is submitted as `cudaMemcpyDeviceToHost`;
- the binding-owned source CUDA allocation retain remains alive for the entire in-flight DMA interval;
- destination page-locked buffers cannot return to the pool while native copies are in flight;
- completed payloads preserve KV bytes, tensor shape, source CUDA format, and causal frontier;
- payload disposal returns exact-length page-locked buffers and repeated same-shape export reuses them;
- cancellation after submission waits native completion before releasing source or destination lifetimes;
- a stable host allocation without the explicit CUDA page-locked capability is rejected before any DMA submission.
