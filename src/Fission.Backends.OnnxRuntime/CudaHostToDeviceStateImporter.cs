using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Reconstructs a decoder state on one explicit CUDA device from a page-locked
/// host-staging payload. Every target allocation is created before any H2D copy is
/// submitted, and the returned state is published only after every CUDA completion
/// event has reached a terminal state.
/// </summary>
public sealed class CudaHostToDeviceStateImporter
{
    private const long MetadataBytes = sizeof(int) * 2L;
    private readonly CudaDeviceMemoryAllocator _allocator;
    private readonly CudaDeviceBoundAsyncCopyEngine _copyEngine;

    public CudaHostToDeviceStateImporter(
        CudaDeviceMemoryAllocator allocator,
        CudaDeviceBoundAsyncCopyEngine copyEngine)
    {
        ArgumentNullException.ThrowIfNull(allocator);
        ArgumentNullException.ThrowIfNull(copyEngine);
        if (allocator.DeviceId != copyEngine.DeviceId)
        {
            throw new ArgumentException(
                $"CUDA allocator targets device {allocator.DeviceId}, but copy engine targets device {copyEngine.DeviceId}.");
        }

        _allocator = allocator;
        _copyEngine = copyEngine;
    }

    public int DeviceId => _allocator.DeviceId;

    public async ValueTask<DecoderOrtState> ImportAsync(
        DecoderOrtCudaHostStagingPayload payload,
        string expectedCudaFormatId,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(payload);
        if (string.IsNullOrWhiteSpace(expectedCudaFormatId))
        {
            throw new ArgumentException(
                "Expected CUDA state format id cannot be empty.",
                nameof(expectedCudaFormatId));
        }

        cancellationToken.ThrowIfCancellationRequested();
        ObjectDisposedException.ThrowIf(payload.IsDisposed, payload);

        if (!StringComparer.Ordinal.Equals(
                payload.SourceCudaFormatId,
                expectedCudaFormatId))
        {
            throw new InvalidOperationException(
                $"CUDA host-staging payload was exported from format '{payload.SourceCudaFormatId}', " +
                $"but target expects '{expectedCudaFormatId}'.");
        }

        var expectedHostFormat =
            CudaDeviceToHostStagingExporter.BuildHostStagingFormatId(
                expectedCudaFormatId);
        if (!StringComparer.Ordinal.Equals(payload.FormatId, expectedHostFormat))
        {
            throw new InvalidOperationException(
                $"CUDA host-staging payload format '{payload.FormatId}' does not match '{expectedHostFormat}'.");
        }

        using var payloadLifetime = payload.Retain();
        var allocations = new List<CudaDeviceMemoryAllocation>(
            checked(payload.LayerCount * 2));
        var hostPins = new List<MemoryHandle>(checked(payload.LayerCount * 2));
        var copies = new List<CopyPlan>(checked(payload.LayerCount * 2));
        var keyMemories = new Memory<float>[payload.LayerCount];
        var valueMemories = new Memory<float>[payload.LayerCount];
        var keyShapes = new long[payload.LayerCount][];
        var valueShapes = new long[payload.LayerCount][];
        var keyBytes = new long[payload.LayerCount];
        var valueBytes = new long[payload.LayerCount];
        var ownedOrtValues = new List<OrtValue>(checked(payload.LayerCount * 2));
        OrtMemoryInfo? memoryInfo = null;
        CudaImportedStateLifetime? lifetimeAnchor = null;

        try
        {
            // Validate the entire payload before touching target device memory. A
            // malformed later layer must not leave earlier cudaMalloc allocations.
            var validatedBytes = MetadataBytes;
            for (var layer = 0; layer < payload.LayerCount; layer++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                keyMemories[layer] = payload.GetKeyMemory(layer);
                valueMemories[layer] = payload.GetValueMemory(layer);
                keyShapes[layer] = payload.GetKeyShape(layer).ToArray();
                valueShapes[layer] = payload.GetValueShape(layer).ToArray();
                keyBytes[layer] = ValidateFp32HostTensor(
                    keyMemories[layer],
                    keyShapes[layer],
                    layer,
                    "key");
                valueBytes[layer] = ValidateFp32HostTensor(
                    valueMemories[layer],
                    valueShapes[layer],
                    layer,
                    "value");
                validatedBytes = checked(
                    validatedBytes + keyBytes[layer] + valueBytes[layer]);
            }

            if (payload.ByteLength != validatedBytes)
            {
                throw new InvalidOperationException(
                    $"CUDA host-staging payload reports {payload.ByteLength} byte(s), " +
                    $"but tensor geometry plus metadata requires {validatedBytes}.");
            }

            // Complete all cudaMalloc operations before submitting any native copy.
            // This keeps allocation failure out of the partially-submitted DMA state.
            for (var layer = 0; layer < payload.LayerCount; layer++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var keyAllocation = _allocator.Allocate(keyBytes[layer]);
                allocations.Add(keyAllocation);
                var valueAllocation = _allocator.Allocate(valueBytes[layer]);
                allocations.Add(valueAllocation);

                var keyPin = keyMemories[layer].Pin();
                hostPins.Add(keyPin);
                var keyPointer = GetPointer(keyPin);
                if (keyPointer == 0)
                {
                    throw new InvalidOperationException(
                        $"CUDA host-staging layer {layer} key exposed a null host pointer.");
                }

                var valuePin = valueMemories[layer].Pin();
                hostPins.Add(valuePin);
                var valuePointer = GetPointer(valuePin);
                if (valuePointer == 0)
                {
                    throw new InvalidOperationException(
                        $"CUDA host-staging layer {layer} value exposed a null host pointer.");
                }

                copies.Add(new CopyPlan(
                    keyAllocation.Pointer,
                    keyPointer,
                    checked((nuint)keyBytes[layer])));
                copies.Add(new CopyPlan(
                    valueAllocation.Pointer,
                    valuePointer,
                    checked((nuint)valueBytes[layer])));
            }

            var pending = new Task[copies.Count];
            for (var index = 0; index < copies.Count; index++)
            {
                var copy = copies[index];
                pending[index] = _copyEngine.CopyAsync(
                    copy.Destination,
                    copy.Source,
                    copy.ByteLength,
                    CudaMemcpyKind.HostToDevice,
                    cancellationToken).AsTask();
            }

            // Task.WhenAll observes terminal completion of every submitted copy.
            // CudaAsyncCopyEngine itself delays post-submit cancellation until the
            // native event completes, so allocations and host pins remain valid.
            await Task.WhenAll(pending).ConfigureAwait(false);

            memoryInfo = new OrtMemoryInfo(
                "Cuda",
                OrtAllocatorType.DeviceAllocator,
                DeviceId,
                OrtMemType.Default);

            var layers = new DecoderOrtLayerState[payload.LayerCount];
            var residentLayers = new DecoderOrtCudaLayerView[payload.LayerCount];
            for (var layer = 0; layer < payload.LayerCount; layer++)
            {
                var keyAllocation = allocations[layer * 2];
                var valueAllocation = allocations[(layer * 2) + 1];

                var key = OrtValue.CreateTensorValueWithData(
                    memoryInfo,
                    TensorElementType.Float,
                    keyShapes[layer],
                    keyAllocation.Pointer,
                    keyBytes[layer]);
                ownedOrtValues.Add(key);

                var value = OrtValue.CreateTensorValueWithData(
                    memoryInfo,
                    TensorElementType.Float,
                    valueShapes[layer],
                    valueAllocation.Pointer,
                    valueBytes[layer]);
                ownedOrtValues.Add(value);
                layers[layer] = new DecoderOrtLayerState(key, value);
                residentLayers[layer] = new DecoderOrtCudaLayerView(
                    new CudaDeviceTensorView(
                        keyAllocation.Pointer,
                        keyBytes[layer],
                        TensorElementType.Float,
                        keyShapes[layer],
                        DeviceId),
                    new CudaDeviceTensorView(
                        valueAllocation.Pointer,
                        valueBytes[layer],
                        TensorElementType.Float,
                        valueShapes[layer],
                        DeviceId));
            }

            lifetimeAnchor = new CudaImportedStateLifetime(
                expectedCudaFormatId,
                DeviceId,
                memoryInfo,
                allocations.ToArray(),
                residentLayers);

            // Transfer native allocation/memory-info ownership into the lifetime
            // anchor before constructing the state. If state construction fails,
            // the outer catch disposes OrtValue wrappers first, then the anchor.
            memoryInfo = null;
            allocations.Clear();

            var state = new DecoderOrtState(
                payload.Position,
                layers,
                payload.NextTokenId,
                lifetimeAnchor);
            lifetimeAnchor = null;
            ownedOrtValues.Clear();
            return state;
        }
        catch
        {
            for (var index = ownedOrtValues.Count - 1; index >= 0; index--)
            {
                ownedOrtValues[index].Dispose();
            }

            lifetimeAnchor?.Dispose();
            memoryInfo?.Dispose();
            for (var index = allocations.Count - 1; index >= 0; index--)
            {
                allocations[index].Dispose();
            }

            throw;
        }
        finally
        {
            for (var index = hostPins.Count - 1; index >= 0; index--)
            {
                hostPins[index].Dispose();
            }
        }
    }

    private static long ValidateFp32HostTensor(
        Memory<float> memory,
        IReadOnlyList<long> shape,
        int layer,
        string slot)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Count == 0)
        {
            throw new InvalidOperationException(
                $"CUDA host-staging layer {layer} {slot} shape is empty.");
        }

        var elementCount = 1L;
        for (var index = 0; index < shape.Count; index++)
        {
            var dimension = shape[index];
            if (dimension <= 0)
            {
                throw new InvalidOperationException(
                    $"CUDA host-staging layer {layer} {slot} shape contains non-positive dimension {dimension}.");
            }

            elementCount = checked(elementCount * dimension);
        }

        if (elementCount != memory.Length)
        {
            throw new InvalidOperationException(
                $"CUDA host-staging layer {layer} {slot} shape requires {elementCount} FP32 element(s), " +
                $"but payload contains {memory.Length}.");
        }

        return checked(elementCount * sizeof(float));
    }

    private static unsafe nint GetPointer(MemoryHandle handle) =>
        (nint)handle.Pointer;

    private readonly record struct CopyPlan(
        nint Destination,
        nint Source,
        nuint ByteLength);

    private sealed class CudaImportedStateLifetime :
        IDisposable,
        IDecoderOrtOwnedLifetimeAnchor,
        IDecoderOrtCudaResidentStateSource
    {
        private readonly string _cudaFormatId;
        private readonly int _deviceId;
        private readonly DecoderOrtCudaLayerView[] _residentLayers;
        private OrtMemoryInfo? _memoryInfo;
        private CudaDeviceMemoryAllocation[]? _allocations;
        private int _referenceCount = 1;
        private int _ownerReleased;

        public CudaImportedStateLifetime(
            string cudaFormatId,
            int deviceId,
            OrtMemoryInfo memoryInfo,
            CudaDeviceMemoryAllocation[] allocations,
            DecoderOrtCudaLayerView[] residentLayers)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(cudaFormatId);
            ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
            ArgumentNullException.ThrowIfNull(memoryInfo);
            ArgumentNullException.ThrowIfNull(allocations);
            ArgumentNullException.ThrowIfNull(residentLayers);
            if (allocations.Length != checked(residentLayers.Length * 2))
            {
                throw new ArgumentException(
                    "Imported CUDA allocation count must equal two buffers per decoder layer.",
                    nameof(allocations));
            }

            _cudaFormatId = cudaFormatId;
            _deviceId = deviceId;
            _memoryInfo = memoryInfo;
            _allocations = allocations;
            _residentLayers = residentLayers.ToArray();
        }

        DecoderOrtCudaResidentStateLease IDecoderOrtCudaResidentStateSource.AcquireCudaResidentState(
            string formatId,
            DecoderOrtState state)
        {
            if (!StringComparer.Ordinal.Equals(formatId, _cudaFormatId))
            {
                throw new InvalidOperationException(
                    $"Imported CUDA state format '{_cudaFormatId}' is incompatible with requested format '{formatId}'.");
            }

            var retain = new RetainedImportedStateLifetime(this);
            return DecoderOrtCudaResidentStateLease.Create(
                _cudaFormatId,
                state,
                _deviceId,
                _residentLayers,
                retain);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _ownerReleased, 1) == 0)
            {
                ReleaseReference();
            }
        }

        void IDecoderOrtOwnedLifetimeAnchor.Release() => Dispose();

        private void Retain()
        {
            while (true)
            {
                var current = Volatile.Read(ref _referenceCount);
                if (current <= 0 || Volatile.Read(ref _allocations) is null)
                {
                    throw new ObjectDisposedException(
                        nameof(CudaImportedStateLifetime));
                }

                if (Interlocked.CompareExchange(
                        ref _referenceCount,
                        checked(current + 1),
                        current) == current)
                {
                    return;
                }
            }
        }

        private void ReleaseReference()
        {
            var remaining = Interlocked.Decrement(ref _referenceCount);
            if (remaining < 0)
            {
                throw new InvalidOperationException(
                    "Imported CUDA state lifetime reference count underflowed.");
            }

            if (remaining != 0)
            {
                return;
            }

            var allocations = Interlocked.Exchange(ref _allocations, null);
            if (allocations is not null)
            {
                for (var index = allocations.Length - 1; index >= 0; index--)
                {
                    allocations[index].Dispose();
                }
            }

            Interlocked.Exchange(ref _memoryInfo, null)?.Dispose();
        }

        private sealed class RetainedImportedStateLifetime : IDisposable
        {
            private CudaImportedStateLifetime? _owner;

            public RetainedImportedStateLifetime(
                CudaImportedStateLifetime owner)
            {
                owner.Retain();
                _owner = owner;
            }

            public void Dispose()
            {
                Interlocked.Exchange(ref _owner, null)?.ReleaseReference();
            }
        }
    }
}
