using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Immutable non-owning view over one CUDA device-resident tensor region.
///
/// The pointer is meaningful only while the owning
/// <see cref="DecoderOrtCudaResidentStateLease"/> remains alive. Bindings must
/// expose the exact tensor start address, not a host pointer, CUDA-pinned host
/// allocation, or an opaque allocation handle.
/// </summary>
public sealed class CudaDeviceTensorView
{
    private readonly ReadOnlyCollection<long> _shape;

    public CudaDeviceTensorView(
        nint devicePointer,
        long byteLength,
        TensorElementType elementType,
        IReadOnlyList<long> shape,
        int deviceId)
    {
        if (devicePointer == 0)
        {
            throw new ArgumentException(
                "CUDA device tensor pointer cannot be null.",
                nameof(devicePointer));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Count == 0)
        {
            throw new ArgumentException(
                "CUDA device tensor shape must contain at least one dimension.",
                nameof(shape));
        }

        var copiedShape = new long[shape.Count];
        for (var index = 0; index < shape.Count; index++)
        {
            var dimension = shape[index];
            ArgumentOutOfRangeException.ThrowIfNegative(
                dimension,
                nameof(shape));
            copiedShape[index] = dimension;
        }

        DevicePointer = devicePointer;
        ByteLength = byteLength;
        ElementType = elementType;
        DeviceId = deviceId;
        _shape = Array.AsReadOnly(copiedShape);
    }

    public nint DevicePointer { get; }
    public long ByteLength { get; }
    public TensorElementType ElementType { get; }
    public IReadOnlyList<long> Shape => _shape;
    public int DeviceId { get; }

    internal void ValidateAgainst(
        OrtValue value,
        int layerIndex,
        string slotName)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (!value.IsTensor)
        {
            throw new InvalidOperationException(
                $"CUDA-resident decoder layer {layerIndex} {slotName} OrtValue is not a tensor.");
        }

        using (var memoryInfo = value.GetTensorMemoryInfo())
        {
            if (!StringComparer.Ordinal.Equals(memoryInfo.Name, "Cuda"))
            {
                throw new InvalidOperationException(
                    $"CUDA-resident decoder layer {layerIndex} {slotName} is backed by " +
                    $"allocator '{memoryInfo.Name}', not CUDA device memory.");
            }

            if (memoryInfo.Id != DeviceId)
            {
                throw new InvalidOperationException(
                    $"CUDA-resident decoder layer {layerIndex} {slotName} reports device " +
                    $"{memoryInfo.Id}, but its pointer descriptor reports device {DeviceId}.");
            }

            if (memoryInfo.GetMemoryType() != OrtMemType.Default)
            {
                throw new InvalidOperationException(
                    $"CUDA-resident decoder layer {layerIndex} {slotName} uses ORT memory type " +
                    $"{memoryInfo.GetMemoryType()}, not device-default memory.");
            }

            if (memoryInfo.GetDeviceMemoryType() != OrtDeviceMemoryType.DEFAULT)
            {
                throw new InvalidOperationException(
                    $"CUDA-resident decoder layer {layerIndex} {slotName} is not device-default memory.");
            }
        }

        var typeAndShape = value.GetTensorTypeAndShape();
        if (typeAndShape.ElementDataType != ElementType)
        {
            throw new InvalidOperationException(
                $"CUDA-resident decoder layer {layerIndex} {slotName} element type " +
                $"{typeAndShape.ElementDataType} does not match descriptor type {ElementType}.");
        }

        var actualShape = typeAndShape.Shape;
        if (actualShape.Length != _shape.Count ||
            !actualShape.SequenceEqual(_shape))
        {
            throw new InvalidOperationException(
                $"CUDA-resident decoder layer {layerIndex} {slotName} shape does not match its pointer descriptor.");
        }

        var actualBytes = value.GetTensorSizeInBytes();
        if (actualBytes != ByteLength)
        {
            throw new InvalidOperationException(
                $"CUDA-resident decoder layer {layerIndex} {slotName} contains {actualBytes} byte(s), " +
                $"but its pointer descriptor reports {ByteLength}.");
        }
    }
}

/// <summary>
/// CUDA device-resident key/value pointer views for one decoder layer.
/// </summary>
public readonly record struct DecoderOrtCudaLayerView(
    CudaDeviceTensorView Key,
    CudaDeviceTensorView Value);

/// <summary>
/// Lifetime guard for a validated CUDA-resident decoder state.
///
/// The binding transfers ownership of an independent native-allocation retain
/// lease into <see cref="Create"/>. That retain lease is released only when this
/// instance is disposed. Callers must not use any returned device pointer after
/// disposal.
/// </summary>
public sealed class DecoderOrtCudaResidentStateLease : IDisposable
{
    private readonly DecoderOrtCudaLayerView[] _layers;
    private IDisposable? _lifetimeLease;
    private int _disposed;

    private DecoderOrtCudaResidentStateLease(
        string formatId,
        int position,
        int? nextTokenId,
        int deviceId,
        DecoderOrtCudaLayerView[] layers,
        long byteLength,
        IDisposable lifetimeLease)
    {
        FormatId = formatId;
        Position = position;
        NextTokenId = nextTokenId;
        DeviceId = deviceId;
        _layers = layers;
        ByteLength = byteLength;
        _lifetimeLease = lifetimeLease;
    }

    public string FormatId { get; }
    public int Position { get; }
    public int? NextTokenId { get; }
    public int DeviceId { get; }
    public int LayerCount => _layers.Length;
    public long ByteLength { get; }
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>
    /// Validates an externally-owned CUDA pointer set against the corresponding
    /// decoder OrtValues and transfers ownership of <paramref name="lifetimeLease"/>
    /// into the returned lease. If validation fails, the supplied lifetime lease
    /// is disposed before the exception escapes.
    /// </summary>
    public static DecoderOrtCudaResidentStateLease Create(
        string formatId,
        DecoderOrtState state,
        int deviceId,
        IReadOnlyList<DecoderOrtCudaLayerView> layers,
        IDisposable lifetimeLease)
    {
        if (string.IsNullOrWhiteSpace(formatId))
        {
            throw new ArgumentException(
                "CUDA-resident state format id cannot be empty.",
                nameof(formatId));
        }

        ArgumentNullException.ThrowIfNull(state);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentNullException.ThrowIfNull(lifetimeLease);

        var transferred = false;
        try
        {
            if (state.IsDisposed)
            {
                throw new ObjectDisposedException(
                    nameof(state),
                    "Cannot acquire CUDA-resident pointers from a disposed decoder state.");
            }

            if (layers.Count != state.LayerCount)
            {
                throw new InvalidOperationException(
                    $"CUDA-resident pointer set contains {layers.Count} layer(s), but decoder state contains {state.LayerCount}.");
            }

            var validatedLayers = new DecoderOrtCudaLayerView[layers.Count];
            var totalBytes = 0L;
            for (var layerIndex = 0; layerIndex < layers.Count; layerIndex++)
            {
                var view = layers[layerIndex];
                ArgumentNullException.ThrowIfNull(view.Key);
                ArgumentNullException.ThrowIfNull(view.Value);

                if (view.Key.DeviceId != deviceId || view.Value.DeviceId != deviceId)
                {
                    throw new InvalidOperationException(
                        $"CUDA-resident decoder layer {layerIndex} belongs to device " +
                        $"{view.Key.DeviceId}/{view.Value.DeviceId}, but the state lease targets device {deviceId}.");
                }

                var ortLayer = state.GetLayer(layerIndex);
                view.Key.ValidateAgainst(ortLayer.Key, layerIndex, "key");
                view.Value.ValidateAgainst(ortLayer.Value, layerIndex, "value");

                totalBytes = checked(totalBytes + view.Key.ByteLength + view.Value.ByteLength);
                validatedLayers[layerIndex] = view;
            }

            var lease = new DecoderOrtCudaResidentStateLease(
                formatId,
                state.Position,
                state.NextTokenId,
                deviceId,
                validatedLayers,
                totalBytes,
                lifetimeLease);
            transferred = true;
            return lease;
        }
        finally
        {
            if (!transferred)
            {
                lifetimeLease.Dispose();
            }
        }
    }

    public DecoderOrtCudaLayerView GetLayer(int layer)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        if (layer >= _layers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }

        return _layers[layer];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        Interlocked.Exchange(ref _lifetimeLease, null)?.Dispose();
    }
}

/// <summary>
/// Optional decoder-binding capability for exposing actual CUDA device-resident
/// KV tensor pointers to physical transports.
///
/// The returned lease must own an independent retain on the underlying device
/// allocations; merely keeping the <see cref="DecoderOrtState"/> object strongly
/// reachable is not sufficient. Bindings that cannot provide a stable raw CUDA
/// pointer and an independent lifetime retain must not implement this capability.
/// </summary>
public interface IDecoderOrtCudaResidentStateBinding : IDecoderOrtModelBinding
{
    /// <summary>
    /// Versioned compatibility key for the physical CUDA-resident KV layout.
    /// </summary>
    string CudaResidentStateFormatId { get; }

    DecoderOrtCudaResidentStateLease AcquireCudaResidentState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default);
}
