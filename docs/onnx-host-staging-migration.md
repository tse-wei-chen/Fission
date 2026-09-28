# ONNX Runtime host-staging migration

Fission has a concrete host-staging transaction path for decoder state owned by the ONNX Runtime backend, plus reusable CUDA primitives for asynchronously moving real device-resident KV state through page-locked host memory.

The feature remains opt-in. `OnnxRuntimeBackend` keeps its existing non-transactional behavior. `OnnxRuntimeMigratableBackend` requires an execution adapter that implements `IOnnxRuntimeSequenceMigrationAdapter`, and the decoder adapter enables physical migration only when its binding implements `IDecoderOrtHostStagingBinding`.

## Transaction data path

The generic migration protocol remains:

```text
source DecoderOrtState
        |
        | prepare / physical export
        v
host-staging payload
        |
        | transport-plan attestation
        v
target binding import
        |
        | publish distinct target state
        v
target DecoderOrtState
```

Prepare exports a physical copy plus the decoder causal frontier (`Position` and `NextTokenId`). Target import must finish before the target state is published. Source commit retires source physical state only after target import succeeds. Abort removes target state and, if a destructive source commit needs rollback, reconstructs source state from the staged payload.

`IDecoderOrtAsyncHostStagingBinding` gives prepare an awaitable export boundary. `IDecoderOrtAsyncHostStagingImportBinding` gives target import and destructive-source rollback an awaitable import boundary. Existing synchronous CPU bindings continue to use the original methods unchanged.

## Payload ownership

`DecoderOrtHostStagingPayload` is reference-counted. The migration transfer owns the initial reference. A binding whose imported `OrtValue` objects alias payload memory retains the payload and passes the returned lease to `DecoderOrtState` as a Fission-owned lifetime anchor.

The source-side terminal operation owns release of the transfer reference:

- successful source commit releases it after source state is retired;
- source abort releases it immediately when source state never disappeared;
- destructive-commit rollback reconstructs source state first, then releases the transfer owner.

`DecoderOrtState` disposes its `OrtValue` objects before releasing a Fission-owned lifetime anchor. This ordering is important both for host-memory views and for CUDA device allocations whose raw memory must outlive the ORT tensor wrappers.

## Optimum legacy FP32 CPU codec

`OptimumLegacyFloatHostStagingBinding` decorates `OptimumLegacyFloatDecoderBinding` and remains the first production-shaped CPU codec. It:

- preserves existing prefill/decode/batched execution behavior;
- estimates KV transfer bytes from decoder geometry;
- deep-copies FP32 key/value tensors into exact-length pooled host buffers;
- uses GC-pinned managed arrays by default;
- validates geometry, layer count, byte count, format id and causal frontier;
- creates distinct target `OrtValue` objects over staged memory;
- keeps payload memory alive until all imported ORT views are gone;
- advertises a model/format-specific host-staging transport id.

`PinnedHostStagingPoolOptions` bounds retained buffers and bytes, and `PinnedHostStagingPoolStatistics` exposes allocation/reuse/return/drop counters.

## Host allocation boundary

`IHostStagingFloatBufferAllocator` owns physical host allocation only. `PinnedFloatBufferPool` owns reuse policy. `DecoderOrtHostStagingPayload` owns active pool leases.

The default `GcPinnedHostStagingFloatBufferAllocator` uses pinned-object-heap arrays. A stable managed address is useful for ORT host views but is not proof of CUDA page-locked memory.

`CudaPageLockedHostStagingFloatBufferAllocator` uses `cudaHostAlloc`/`cudaFreeHost`, exposes the allocation through a custom `MemoryManager<float>`, and implements `ICudaPageLockedHostStagingFloatBuffer`. That explicit capability is required for asynchronous CUDA DMA. Ordinary GC-pinned buffers deliberately do not implement it.

Page-locked memory remains bounded by the same exact-length pool because it is a scarce system resource.

## CUDA-resident decoder-state boundary

`IDecoderOrtCudaResidentStateBinding` is the explicit capability for a binding that truly owns device-resident decoder KV state. The binding must provide stable raw CUDA tensor addresses and an independent native-allocation retain.

`DecoderOrtCudaResidentStateLease.Create` validates every `CudaDeviceTensorView` against the corresponding `OrtValue` before a transport can consume the pointer. Validation requires:

- ORT allocator name `Cuda`, rejecting CPU and `CudaPinned` host allocations;
- matching CUDA device ordinal;
- `OrtMemType.Default` and default device-memory type;
- identical tensor element type and shape;
- exact byte length;
- matching decoder layer count and a live source state.

The raw pointers remain valid only while the lease is alive. Fission does not reflect into ONNX Runtime internals to discover pointers; the binding that owns the CUDA allocation must provide them explicitly.

## Event-driven CUDA copy engine

`CudaAsyncCopyEngine` owns a bounded pool of reusable non-blocking CUDA streams and completion events. It submits `cudaMemcpyAsync`, records disable-timing events, and uses one completion pump with `cudaEventQuery` instead of blocking one managed thread per transfer.

Its lifetime rule is strict: cancellation after native submission is reported only after the corresponding CUDA event reaches a terminal state. Therefore callers may safely keep source and destination allocations alive until the returned operation completes, including cancellation and failure paths.

If completion tracking itself fails, the engine synchronizes the submitted stream before reuse. `DisposeAsync` drains all in-flight work before destroying streams.

## Explicit CUDA device ordinal

Multi-GPU code must not depend on the ambient CUDA device of whichever .NET thread happens to run a continuation.

`CudaDeviceBoundAsyncCopyEngine` wraps the async copy engine with an explicit `DeviceId`. Every stream, event and memcpy runtime call enters that CUDA device and restores the thread's previous device before returning.

`CudaDeviceMemoryAllocator` applies the same rule to `cudaMalloc` and `cudaFree`. `CudaDeviceMemoryAllocation` exposes the raw pointer and owns it through a `SafeHandle`; final release runs `cudaFree` under the allocation's original device ordinal.

The device-bound D2H exporter overload also checks the source `DecoderOrtCudaResidentStateLease.DeviceId` before allocating staging buffers or submitting DMA. A source state from a different GPU is rejected before physical work begins.

## Asynchronous CUDA device-to-host export

`CudaDeviceToHostStagingExporter` bridges validated device-resident KV state into the host-staging transaction protocol:

```text
source DecoderOrtState
        |
        | AcquireCudaResidentState
        v
validated CUDA K/V pointers + native retain
        |
        | cudaMemcpyAsync(DeviceToHost)
        v
CUDA page-locked pooled buffers
        |
        | all completion events terminal
        v
DecoderOrtCudaHostStagingPayload
```

The current bridge supports FP32 KV tensors. It rents every destination buffer before submitting any native copy, so allocation or page-lock-capability failures cannot leave a partially submitted transfer.

The source CUDA allocation retain remains alive until all submitted K/V copies are terminal. `DecoderOrtCudaHostStagingPayload` is published only after that point. Cancellation after submission therefore cannot reclaim source device memory or destination page-locked buffers while CUDA may still access them.

The payload preserves:

- source CUDA physical-layout format id;
- exact key/value shapes;
- decoder position and next-token frontier;
- exact transfer-byte accounting;
- page-locked host buffers owned by the existing exact-length pool.

## Asynchronous CUDA host-to-device import

`CudaHostToDeviceStateImporter` reconstructs a target `DecoderOrtState` on one explicit CUDA device:

```text
DecoderOrtCudaHostStagingPayload
        |
        | validate full payload geometry + total bytes
        v
cudaMalloc all target K/V buffers
        |
        | pin host payload for DMA lifetime
        | cudaMemcpyAsync(HostToDevice)
        v
target CUDA K/V buffers
        |
        | all completion events terminal
        v
CUDA OrtValue views + DecoderOrtState
```

The importer deliberately has three ordered phases.

First, it validates the complete payload before touching target device memory: host format id, source CUDA format id, every tensor shape, FP32 element count, and total `ByteLength` including metadata.

Second, it completes every target `cudaMalloc` before submitting the first H2D copy. A later allocation failure therefore cannot leave earlier DMA in flight. All host `Memory<float>` objects remain pinned until every submitted H2D event is terminal.

Third, after H2D completion it creates ORT tensor wrappers with:

- `OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default)`;
- `OrtValue.CreateTensorValueWithData` over the target CUDA pointers;
- the exact staged shape and byte geometry.

The returned `DecoderOrtState` owns a Fission lifetime anchor containing the target CUDA allocations and ORT memory-info object. State disposal first disposes the `OrtValue` wrappers and only then releases the lifetime anchor and calls `cudaFree`. The host-staging payload can be released after import because the target state no longer aliases host memory.

Cancellation after H2D submission follows the same native-lifetime rule as D2H: caller-visible cancellation waits for every submitted copy to become terminal, after which failed/canceled imports release target allocations. No device pointer is freed while DMA may still reference it.

`DecoderOnlyOnnxExecutionAdapter` detects `IDecoderOrtAsyncHostStagingImportBinding` and awaits it both for normal target import and for destructive-source rollback. Target state is not inserted into `DecoderStateStore` until the async import returns successfully.

## What is complete and what is not

The reusable physical bridge is now present in both directions:

```text
CUDA source KV
   -> validated CUDA pointer lease
   -> async D2H
   -> CUDA page-locked host payload
   -> async H2D
   -> target CUDA allocations + CUDA OrtValues
```

This does **not** yet mean Fission has a complete production GPU decoder binding. A real model/export binding still has to own CUDA-resident inference state, implement `IDecoderOrtCudaResidentStateBinding`, and compose the D2H exporter/H2D importer through the async host-staging binding interfaces.

Direct CUDA P2P/IPC, NIXL/RDMA, topology measurement, peer-access policy, GPU allocator pooling, and cross-process failure recovery remain separate transport/runtime milestones.

## Executable specs

`tests/Fission.OnnxRuntime.Migration.Specs` covers the generic end-to-end host-staging transaction: transport negotiation, source commit ordering, target decode continuation, admission release, incompatible codec rejection, and CPU Optimum round-trip behavior.

`tests/Fission.OnnxRuntime.PinnedStaging.Specs` covers payload reference counting, exact-length pool reuse/bounds, custom physical allocators, CUDA page-locked allocation, deterministic `cudaFreeHost`, and CUDA runtime availability/error handling.

`tests/Fission.OnnxRuntime.CudaCopy.Specs` covers bounded stream concurrency, event polling, post-submit cancellation, memcpy/query failure cleanup, stream reuse and dispose draining.

`tests/Fission.OnnxRuntime.CudaResidentState.Specs` covers CUDA memory metadata attestation, device/type/shape/byte validation, `CudaPinned` rejection, and independent native lifetime retention.

`tests/Fission.OnnxRuntime.CudaD2HStaging.Specs` covers source-side D2H publication ordering, page-locked destination requirements, payload bytes/shapes, pool reuse, and cancellation lifetime safety.

`tests/Fission.OnnxRuntime.CudaH2DState.Specs` covers target-side reconstruction without requiring a GPU:

- explicit CUDA device scoping and restoration;
- `cudaMalloc`/`cudaFree` target lifetime;
- `cudaMemcpyAsync(HostToDevice)` direction and copied bytes;
- CUDA `OrtValue` memory metadata, shape and byte geometry;
- host-payload release independence after import;
- target allocation lifetime extending until `DecoderOrtState.Dispose`;
- cancellation after submission draining before `cudaFree`;
- allocation failure before any H2D submission;
- format and allocator/copy-engine device mismatch rejection;
- adapter dispatch to `IDecoderOrtAsyncHostStagingImportBinding`, including proof that migration import remains incomplete until the asynchronous binding releases its completion gate.
