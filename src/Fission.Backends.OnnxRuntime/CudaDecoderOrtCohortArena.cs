using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Ref-counted FP32 CUDA backing storage for one dense decoder cohort frontier.
///
/// The arena owns one batched key/value CUDA allocation per layer. Its initial
/// builder reference keeps those allocations alive while ORT output views are in
/// use. Every row DecoderOrtState and every resident-state borrow then retains the
/// arena independently. Device memory is released only after the builder, all row
/// states and all outstanding borrows have released their references.
/// </summary>
internal sealed class CudaDecoderOrtCohortArena
{
    private readonly CudaDeviceMemoryAllocation[] _keyAllocations;
    private readonly CudaDeviceMemoryAllocation[] _valueAllocations;
    private readonly long[] _perSequenceShape;
    private readonly long[] _batchedShape;
    private readonly ReadOnlyCollection<long> _perSequenceShapeView;
    private readonly ReadOnlyCollection<long> _batchedShapeView;
    private OrtMemoryInfo? _memoryInfo;
    private int _referenceCount = 1;
    private int _released;

    public CudaDecoderOrtCohortArena(
        int position,
        int batchSize,
        int layerCount,
        IReadOnlyList<long> perSequenceShape,
        CudaDeviceMemoryAllocator allocator)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(layerCount, 1);
        ArgumentNullException.ThrowIfNull(perSequenceShape);
        ArgumentNullException.ThrowIfNull(allocator);
        if (perSequenceShape.Count == 0)
        {
            throw new ArgumentException(
                "CUDA cohort per-sequence shape cannot be empty.",
                nameof(perSequenceShape));
        }

        _perSequenceShape = perSequenceShape.ToArray();
        if (_perSequenceShape[0] != 1)
        {
            throw new ArgumentException(
                "CUDA cohort per-sequence shape must have batch dimension 1.",
                nameof(perSequenceShape));
        }

        var elementCount = 1L;
        for (var index = 0; index < _perSequenceShape.Length; index++)
        {
            var dimension = _perSequenceShape[index];
            if (dimension <= 0)
            {
                throw new ArgumentException(
                    "CUDA cohort tensor dimensions must be positive.",
                    nameof(perSequenceShape));
            }

            elementCount = checked(elementCount * dimension);
        }

        _batchedShape = _perSequenceShape.ToArray();
        _batchedShape[0] = batchSize;
        _perSequenceShapeView = Array.AsReadOnly(_perSequenceShape);
        _batchedShapeView = Array.AsReadOnly(_batchedShape);
        PerSequenceByteLength = checked(elementCount * sizeof(float));
        BatchedByteLength = checked(PerSequenceByteLength * batchSize);
        Position = position;
        BatchSize = batchSize;
        DeviceId = allocator.DeviceId;
        _keyAllocations = new CudaDeviceMemoryAllocation[layerCount];
        _valueAllocations = new CudaDeviceMemoryAllocation[layerCount];

        var keys = 0;
        var values = 0;
        try
        {
            for (var layer = 0; layer < layerCount; layer++)
            {
                _keyAllocations[layer] = allocator.Allocate(BatchedByteLength);
                keys++;
                _valueAllocations[layer] = allocator.Allocate(BatchedByteLength);
                values++;
            }

            _memoryInfo = new OrtMemoryInfo(
                "Cuda",
                OrtAllocatorType.DeviceAllocator,
                DeviceId,
                OrtMemType.Default);
        }
        catch
        {
            for (var layer = values - 1; layer >= 0; layer--)
            {
                _valueAllocations[layer].Dispose();
            }

            for (var layer = keys - 1; layer >= 0; layer--)
            {
                _keyAllocations[layer].Dispose();
            }

            throw;
        }
    }

    public int Position { get; }
    public int BatchSize { get; }
    public int LayerCount => _keyAllocations.Length;
    public int DeviceId { get; }
    public long PerSequenceByteLength { get; }
    public long BatchedByteLength { get; }
    public bool IsReleased => Volatile.Read(ref _released) != 0;
    public IReadOnlyList<long> PerSequenceShape => _perSequenceShapeView;
    public IReadOnlyList<long> BatchedShape => _batchedShapeView;

    /// <summary>
    /// Creates caller-owned batched ORT output wrappers over one layer's complete
    /// CUDA allocations. The builder reference must stay alive until these wrappers
    /// are disposed.
    /// </summary>
    public DecoderOrtLayerState CreateBatchedLayer(int layer)
    {
        ThrowIfReleased();
        ValidateLayer(layer);
        var memoryInfo = _memoryInfo ??
            throw new ObjectDisposedException(nameof(CudaDecoderOrtCohortArena));
        OrtValue? key = null;
        try
        {
            key = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                _batchedShape,
                _keyAllocations[layer].Pointer,
                BatchedByteLength);
            var value = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                _batchedShape,
                _valueAllocations[layer].Pointer,
                BatchedByteLength);
            return new DecoderOrtLayerState(key, value);
        }
        catch
        {
            key?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates one immutable row state whose CUDA OrtValues are byte-offset views
    /// into this cohort's layer allocations. State construction retains the arena.
    /// </summary>
    public DecoderOrtState CreateRowState(
        int row,
        int? nextTokenId)
    {
        ThrowIfReleased();
        ValidateRow(row);
        var memoryInfo = _memoryInfo ??
            throw new ObjectDisposedException(nameof(CudaDecoderOrtCohortArena));
        var layers = new DecoderOrtLayerState[LayerCount];
        var owned = new List<OrtValue>(checked(LayerCount * 2));
        try
        {
            for (var layer = 0; layer < LayerCount; layer++)
            {
                var key = OrtValue.CreateTensorValueWithData(
                    memoryInfo,
                    TensorElementType.Float,
                    _perSequenceShape,
                    GetRowPointer(_keyAllocations[layer], row),
                    PerSequenceByteLength);
                owned.Add(key);
                var value = OrtValue.CreateTensorValueWithData(
                    memoryInfo,
                    TensorElementType.Float,
                    _perSequenceShape,
                    GetRowPointer(_valueAllocations[layer], row),
                    PerSequenceByteLength);
                owned.Add(value);
                layers[layer] = new DecoderOrtLayerState(key, value);
            }

            var state = new DecoderOrtState(
                Position,
                layers,
                nextTokenId,
                new DecoderOrtCudaCohortSlice(this, row));
            owned.Clear();
            return state;
        }
        catch
        {
            for (var index = owned.Count - 1; index >= 0; index--)
            {
                owned[index].Dispose();
            }

            throw;
        }
    }

    /// <summary>
    /// Creates an independently retained raw-pointer lease for a row state. The
    /// returned lease may outlive the DecoderOrtState owner that requested it.
    /// </summary>
    public DecoderOrtCudaResidentStateLease AcquireResidentState(
        string formatId,
        DecoderOrtState state,
        int row)
    {
        ThrowIfReleased();
        ArgumentNullException.ThrowIfNull(state);
        ValidateRow(row);

        if (!state.TryGetCudaCohortSlice(out var slice) ||
            !ReferenceEquals(slice.Arena, this) ||
            slice.Row != row)
        {
            throw new InvalidOperationException(
                "Decoder state is not the requested row of this CUDA cohort arena.");
        }

        var layers = new DecoderOrtCudaLayerView[LayerCount];
        for (var layer = 0; layer < LayerCount; layer++)
        {
            layers[layer] = new DecoderOrtCudaLayerView(
                new CudaDeviceTensorView(
                    GetRowPointer(_keyAllocations[layer], row),
                    PerSequenceByteLength,
                    TensorElementType.Float,
                    _perSequenceShape,
                    DeviceId),
                new CudaDeviceTensorView(
                    GetRowPointer(_valueAllocations[layer], row),
                    PerSequenceByteLength,
                    TensorElementType.Float,
                    _perSequenceShape,
                    DeviceId));
        }

        return DecoderOrtCudaResidentStateLease.Create(
            formatId,
            state,
            DeviceId,
            layers,
            new RetainedArenaLease(this));
    }

    public void Retain()
    {
        while (true)
        {
            var current = Volatile.Read(ref _referenceCount);
            if (current <= 0 || IsReleased)
            {
                throw new ObjectDisposedException(
                    nameof(CudaDecoderOrtCohortArena));
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

    /// <summary>
    /// Releases one builder/state/borrow reference. The final release disposes ORT
    /// memory metadata and every layer CUDA allocation exactly once.
    /// </summary>
    public void Release()
    {
        var remaining = Interlocked.Decrement(ref _referenceCount);
        if (remaining < 0)
        {
            throw new InvalidOperationException(
                "CUDA decoder cohort arena was released more times than retained.");
        }

        if (remaining != 0)
        {
            return;
        }

        if (Interlocked.Exchange(ref _released, 1) != 0)
        {
            throw new InvalidOperationException(
                "CUDA decoder cohort arena was already released.");
        }

        Interlocked.Exchange(ref _memoryInfo, null)?.Dispose();
        for (var layer = LayerCount - 1; layer >= 0; layer--)
        {
            _valueAllocations[layer].Dispose();
            _keyAllocations[layer].Dispose();
        }
    }

    private nint GetRowPointer(
        CudaDeviceMemoryAllocation allocation,
        int row)
    {
        var offset = checked(PerSequenceByteLength * row);
        return checked(allocation.Pointer + (nint)offset);
    }

    private void ValidateLayer(int layer)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        if (layer >= LayerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }
    }

    private void ValidateRow(int row)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        if (row >= BatchSize)
        {
            throw new ArgumentOutOfRangeException(nameof(row));
        }
    }

    private void ThrowIfReleased() =>
        ObjectDisposedException.ThrowIf(IsReleased, this);

    private sealed class RetainedArenaLease : IDisposable
    {
        private CudaDecoderOrtCohortArena? _arena;

        public RetainedArenaLease(CudaDecoderOrtCohortArena arena)
        {
            arena.Retain();
            _arena = arena;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _arena, null)?.Release();
        }
    }
}

internal readonly record struct DecoderOrtCudaCohortSlice(
    CudaDecoderOrtCohortArena Arena,
    int Row);
