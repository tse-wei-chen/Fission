using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Host-staging payload produced by asynchronous CUDA device-to-host copies.
/// The payload owns exact-length CUDA page-locked FP32 buffers until the transfer
/// owner and every retained imported-state lease have released them.
/// </summary>
public sealed class DecoderOrtCudaHostStagingPayload : DecoderOrtHostStagingPayload
{
    private readonly PinnedFloatBufferPool.PinnedFloatBufferLease[] _keys;
    private readonly PinnedFloatBufferPool.PinnedFloatBufferLease[] _values;
    private readonly ReadOnlyCollection<long>[] _keyShapes;
    private readonly ReadOnlyCollection<long>[] _valueShapes;

    internal DecoderOrtCudaHostStagingPayload(
        string formatId,
        string sourceCudaFormatId,
        int position,
        int? nextTokenId,
        long byteLength,
        PinnedFloatBufferPool.PinnedFloatBufferLease[] keys,
        PinnedFloatBufferPool.PinnedFloatBufferLease[] values,
        IReadOnlyList<long>[] keyShapes,
        IReadOnlyList<long>[] valueShapes)
        : base(formatId, position, nextTokenId, byteLength)
    {
        if (string.IsNullOrWhiteSpace(sourceCudaFormatId))
        {
            throw new ArgumentException(
                "Source CUDA state format id cannot be empty.",
                nameof(sourceCudaFormatId));
        }

        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(keyShapes);
        ArgumentNullException.ThrowIfNull(valueShapes);
        if (keys.Length == 0 ||
            values.Length != keys.Length ||
            keyShapes.Length != keys.Length ||
            valueShapes.Length != keys.Length)
        {
            throw new ArgumentException(
                "CUDA host-staging payload arrays must contain the same non-zero layer count.");
        }

        SourceCudaFormatId = sourceCudaFormatId;
        _keys = keys;
        _values = values;
        _keyShapes = new ReadOnlyCollection<long>[keyShapes.Length];
        _valueShapes = new ReadOnlyCollection<long>[valueShapes.Length];
        for (var layer = 0; layer < keyShapes.Length; layer++)
        {
            _keyShapes[layer] = Array.AsReadOnly(keyShapes[layer].ToArray());
            _valueShapes[layer] = Array.AsReadOnly(valueShapes[layer].ToArray());
        }
    }

    public string SourceCudaFormatId { get; }
    public int LayerCount => _keys.Length;

    public Memory<float> GetKeyMemory(int layer)
    {
        ValidateLayer(layer);
        return _keys[layer].Memory;
    }

    public Memory<float> GetValueMemory(int layer)
    {
        ValidateLayer(layer);
        return _values[layer].Memory;
    }

    public IReadOnlyList<long> GetKeyShape(int layer)
    {
        ValidateLayer(layer);
        return _keyShapes[layer];
    }

    public IReadOnlyList<long> GetValueShape(int layer)
    {
        ValidateLayer(layer);
        return _valueShapes[layer];
    }

    private void ValidateLayer(int layer)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        if (layer >= _keys.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }
    }

    protected override void DisposeCore()
    {
        for (var layer = _values.Length - 1; layer >= 0; layer--)
        {
            _values[layer].Dispose();
        }

        for (var layer = _keys.Length - 1; layer >= 0; layer--)
        {
            _keys[layer].Dispose();
        }
    }
}

/// <summary>
/// Asynchronously deep-copies validated CUDA device-resident decoder K/V tensors
/// into exact-length CUDA page-locked host buffers.
///
/// The exporter deliberately owns only its staging-buffer pool. The supplied
/// <see cref="CudaAsyncCopyEngine"/> is caller-owned and can be shared across
/// exporters/transfers. A payload is returned only after every submitted D2H copy
/// has reached terminal CUDA event completion.
/// </summary>
public sealed class CudaDeviceToHostStagingExporter : IDisposable
{
    private const long MetadataBytes = sizeof(int) * 2L;
    private readonly CudaAsyncCopyEngine _copyEngine;
    private readonly PinnedFloatBufferPool _stagingPool;
    private int _disposed;

    public CudaDeviceToHostStagingExporter(
        CudaAsyncCopyEngine copyEngine,
        CudaPageLockedHostStagingFloatBufferAllocator stagingAllocator,
        PinnedHostStagingPoolOptions? stagingPoolOptions = null)
        : this(
            copyEngine,
            (IHostStagingFloatBufferAllocator)stagingAllocator,
            stagingPoolOptions)
    {
    }

    internal CudaDeviceToHostStagingExporter(
        CudaAsyncCopyEngine copyEngine,
        IHostStagingFloatBufferAllocator stagingAllocator,
        PinnedHostStagingPoolOptions? stagingPoolOptions = null)
    {
        ArgumentNullException.ThrowIfNull(copyEngine);
        ArgumentNullException.ThrowIfNull(stagingAllocator);
        _copyEngine = copyEngine;
        _stagingPool = new PinnedFloatBufferPool(
            stagingPoolOptions,
            stagingAllocator);
    }

    public PinnedHostStagingPoolStatistics HostStagingPoolStatistics =>
        _stagingPool.GetStatistics();

    public static string BuildHostStagingFormatId(string sourceCudaFormatId)
    {
        if (string.IsNullOrWhiteSpace(sourceCudaFormatId))
        {
            throw new ArgumentException(
                "Source CUDA state format id cannot be empty.",
                nameof(sourceCudaFormatId));
        }

        return $"cuda-d2h-host-fp32-v1:{sourceCudaFormatId}";
    }

    public static long EstimateHostStagingBytes(
        DecoderOrtCudaResidentStateLease residentState)
    {
        ArgumentNullException.ThrowIfNull(residentState);
        ObjectDisposedException.ThrowIf(residentState.IsDisposed, residentState);
        return checked(residentState.ByteLength + MetadataBytes);
    }

    /// <summary>
    /// Acquires the binding's independent CUDA-allocation retain, keeps it alive
    /// through every D2H completion event, and releases it only after export has
    /// completed or fully unwound.
    /// </summary>
    public async ValueTask<DecoderOrtCudaHostStagingPayload> ExportAsync(
        IDecoderOrtCudaResidentStateBinding binding,
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(binding);
        ArgumentNullException.ThrowIfNull(state);
        cancellationToken.ThrowIfCancellationRequested();

        using var residentState = binding.AcquireCudaResidentState(
            state,
            cancellationToken);
        ArgumentNullException.ThrowIfNull(residentState);
        if (!StringComparer.Ordinal.Equals(
                residentState.FormatId,
                binding.CudaResidentStateFormatId))
        {
            throw new InvalidOperationException(
                $"CUDA-resident binding '{binding.Name}' acquired format " +
                $"'{residentState.FormatId}', but advertises '{binding.CudaResidentStateFormatId}'.");
        }

        if (residentState.Position != state.Position ||
            residentState.NextTokenId != state.NextTokenId)
        {
            throw new InvalidOperationException(
                "CUDA-resident state acquisition changed the decoder causal frontier.");
        }

        return await ExportAsync(
            residentState,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Copies an already-retained CUDA state into page-locked host staging. The
    /// caller must keep <paramref name="residentState"/> alive until this operation
    /// returns. The binding overload above is preferred when the exporter should
    /// own the complete borrow lifetime.
    /// </summary>
    public async ValueTask<DecoderOrtCudaHostStagingPayload> ExportAsync(
        DecoderOrtCudaResidentStateLease residentState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(residentState);
        ObjectDisposedException.ThrowIf(residentState.IsDisposed, residentState);
        cancellationToken.ThrowIfCancellationRequested();

        var layerCount = residentState.LayerCount;
        var keys = new PinnedFloatBufferPool.PinnedFloatBufferLease[layerCount];
        var values = new PinnedFloatBufferPool.PinnedFloatBufferLease[layerCount];
        var keyShapes = new IReadOnlyList<long>[layerCount];
        var valueShapes = new IReadOnlyList<long>[layerCount];
        var acquired = new List<PinnedFloatBufferPool.PinnedFloatBufferLease>(
            checked(layerCount * 2));
        var copies = new List<CopyPlan>(checked(layerCount * 2));

        try
        {
            // Allocate every destination before submitting native work. This keeps
            // allocation failure out of the partially-submitted DMA state space.
            for (var layer = 0; layer < layerCount; layer++)
            {
                var source = residentState.GetLayer(layer);
                var keyElements = ValidateFp32Tensor(source.Key, layer, "key");
                var valueElements = ValidateFp32Tensor(source.Value, layer, "value");

                var keyBuffer = _stagingPool.Rent(keyElements);
                acquired.Add(keyBuffer);
                keys[layer] = keyBuffer;
                keyShapes[layer] = source.Key.Shape;
                copies.Add(new CopyPlan(
                    keyBuffer.GetCudaPageLockedPointer(),
                    source.Key.DevicePointer,
                    checked((nuint)source.Key.ByteLength)));

                var valueBuffer = _stagingPool.Rent(valueElements);
                acquired.Add(valueBuffer);
                values[layer] = valueBuffer;
                valueShapes[layer] = source.Value.Shape;
                copies.Add(new CopyPlan(
                    valueBuffer.GetCudaPageLockedPointer(),
                    source.Value.DevicePointer,
                    checked((nuint)source.Value.ByteLength)));
            }

            // Submit all copies so CudaAsyncCopyEngine's bounded stream pool can
            // overlap them. Task.WhenAll reaches a terminal state only after every
            // submitted copy has reached its CUDA event, including cancellation and
            // failure paths, so destination/source lifetimes remain safe in finally.
            var pending = new Task[copies.Count];
            for (var index = 0; index < copies.Count; index++)
            {
                var copy = copies[index];
                pending[index] = _copyEngine.CopyAsync(
                    copy.Destination,
                    copy.Source,
                    copy.ByteLength,
                    CudaMemcpyKind.DeviceToHost,
                    cancellationToken).AsTask();
            }

            await Task.WhenAll(pending).ConfigureAwait(false);

            var payload = new DecoderOrtCudaHostStagingPayload(
                BuildHostStagingFormatId(residentState.FormatId),
                residentState.FormatId,
                residentState.Position,
                residentState.NextTokenId,
                EstimateHostStagingBytes(residentState),
                keys,
                values,
                keyShapes,
                valueShapes);
            acquired.Clear();
            return payload;
        }
        catch
        {
            for (var index = acquired.Count - 1; index >= 0; index--)
            {
                acquired[index].Dispose();
            }

            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _stagingPool.Dispose();
    }

    private static int ValidateFp32Tensor(
        CudaDeviceTensorView tensor,
        int layer,
        string slot)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.ElementType != TensorElementType.Float)
        {
            throw new NotSupportedException(
                $"CUDA D2H host staging currently supports FP32 KV only; " +
                $"layer {layer} {slot} is {tensor.ElementType}.");
        }

        if (tensor.ByteLength % sizeof(float) != 0)
        {
            throw new InvalidOperationException(
                $"CUDA decoder layer {layer} {slot} byte length {tensor.ByteLength} is not FP32-aligned.");
        }

        var elements = tensor.ByteLength / sizeof(float);
        if (elements <= 0 || elements > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"CUDA decoder layer {layer} {slot} contains {elements} FP32 elements, outside the supported staging range.");
        }

        return checked((int)elements);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    private readonly record struct CopyPlan(
        nint Destination,
        nint Source,
        nuint ByteLength);
}
