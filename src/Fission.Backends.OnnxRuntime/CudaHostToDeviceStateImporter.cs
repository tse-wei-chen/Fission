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
            }

            var lifetimeAnchor = new CudaImportedStateLifetime(
                memoryInfo,
                allocations.ToArray());
            DecoderOrtState state;
            try
            {
                state = new DecoderOrtState(
                    payload.Position,
                    layers,
                    payload.NextTokenId,
                    lifetimeAnchor);
            }
            catch
            {
                lifetimeAnchor.Dispose();
                throw;
            }

            memoryInfo = null;
            allocations.Clear();
            ownedOrtValues.Clear();
            return state;
        }
        catch
        {
            for (var index = ownedOrtValues.Count - 1; index >= 0; index--)
            {
                ownedOrtValues[index].Dispose();
            }

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
        IDecoderOrtOwnedLifetimeAnchor
    {
        private OrtMemoryInfo? _memoryInfo;
        private CudaDeviceMemoryAllocation[]? _allocations;

        public CudaImportedStateLifetime(
            OrtMemoryInfo memoryInfo,
            CudaDeviceMemoryAllocation[] allocations)
        {
            ArgumentNullException.ThrowIfNull(memoryInfo);
            ArgumentNullException.ThrowIfNull(allocations);
            _memoryInfo = memoryInfo;
            _allocations = allocations;
        }

        public void Dispose()
        {
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

        void IDecoderOrtOwnedLifetimeAnchor.Release() => Dispose();
    }
}
