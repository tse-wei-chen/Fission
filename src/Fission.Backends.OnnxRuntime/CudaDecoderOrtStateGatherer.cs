using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Owns one dense CUDA cohort produced by <see cref="CudaDecoderOrtStateGatherer"/>.
/// Each exposed row state retains the shared target arena independently. Disposing
/// this batch releases every row state; callers must keep the batch alive while a
/// decoder run consumes <see cref="States"/>.
/// </summary>
public sealed class CudaGatheredDecoderStateBatch : IDisposable
{
    private readonly DecoderOrtState[] _states;
    private readonly ReadOnlyCollection<DecoderOrtState> _statesView;
    private int _disposed;

    internal CudaGatheredDecoderStateBatch(
        int position,
        int deviceId,
        DecoderOrtState[] states)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        ArgumentNullException.ThrowIfNull(states);
        if (states.Length == 0)
        {
            throw new ArgumentException(
                "Gathered CUDA decoder batch must contain at least one row.",
                nameof(states));
        }

        Position = position;
        DeviceId = deviceId;
        _states = states;
        _statesView = Array.AsReadOnly(states);
    }

    public int Position { get; }
    public int DeviceId { get; }
    public int BatchSize => _states.Length;
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public IReadOnlyList<DecoderOrtState> States
    {
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _statesView;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        for (var index = _states.Length - 1; index >= 0; index--)
        {
            _states[index].Dispose();
        }
    }
}

/// <summary>
/// Asynchronously gathers arbitrary same-format CUDA decoder rows into a new
/// dense CUDA cohort using device-to-device copies only.
///
/// Source rows may come from different cohort arenas, standalone migration
/// imports, subsets, reordered rows, or duplicate/forked rows. Every source and
/// target native allocation remains independently retained until all submitted
/// CUDA completion events reach a terminal state. No host KV staging is used.
/// </summary>
public sealed class CudaDecoderOrtStateGatherer
{
    private readonly DecoderOrtGeometry _geometry;
    private readonly string _cudaFormatId;
    private readonly CudaDeviceMemoryAllocator _allocator;
    private readonly CudaDeviceBoundAsyncCopyEngine _copyEngine;

    public CudaDecoderOrtStateGatherer(
        DecoderOrtGeometry geometry,
        string cudaFormatId,
        CudaDeviceMemoryAllocator allocator,
        CudaDeviceBoundAsyncCopyEngine copyEngine)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentException.ThrowIfNullOrWhiteSpace(cudaFormatId);
        ArgumentNullException.ThrowIfNull(allocator);
        ArgumentNullException.ThrowIfNull(copyEngine);

        if (geometry.KvElementType != TensorElementType.Float)
        {
            throw new ArgumentException(
                "CUDA decoder gather currently supports FP32 KV tensors only.",
                nameof(geometry));
        }

        if (allocator.DeviceId != copyEngine.DeviceId)
        {
            throw new ArgumentException(
                $"CUDA gather allocator targets device {allocator.DeviceId}, but copy engine targets device {copyEngine.DeviceId}.");
        }

        _geometry = geometry;
        _cudaFormatId = cudaFormatId;
        _allocator = allocator;
        _copyEngine = copyEngine;
    }

    public int DeviceId => _allocator.DeviceId;
    public string CudaFormatId => _cudaFormatId;

    public async ValueTask<CudaGatheredDecoderStateBatch> GatherAsync(
        IReadOnlyList<DecoderOrtState> sourceStates,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(sourceStates);
        if (sourceStates.Count == 0)
        {
            throw new ArgumentException(
                "CUDA decoder gather requires at least one source row.",
                nameof(sourceStates));
        }

        cancellationToken.ThrowIfCancellationRequested();
        var position = sourceStates[0]?.Position
            ?? throw new ArgumentException(
                "CUDA decoder gather source row cannot be null.",
                nameof(sourceStates));
        if (position == 0)
        {
            throw new InvalidOperationException(
                "CUDA decoder gather requires a non-empty past frontier.");
        }

        var expectedShape = _geometry.GetPastKvShape(
            batchSize: 1,
            pastSequenceLength: position);
        var expectedBytes = CheckedFp32ByteLength(expectedShape);

        var sourceLeases = new DecoderOrtCudaResidentStateLease[sourceStates.Count];
        var targetLeases = new DecoderOrtCudaResidentStateLease[sourceStates.Count];
        var targetStates = new DecoderOrtState[sourceStates.Count];
        CudaDecoderOrtCohortArena? targetArena = null;
        var success = false;

        try
        {
            // Validate and retain every source before allocating target memory. A
            // bad later row cannot leave a partially-created target cohort.
            for (var row = 0; row < sourceStates.Count; row++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var state = sourceStates[row] ?? throw new ArgumentException(
                    $"CUDA decoder gather source row {row} is null.",
                    nameof(sourceStates));
                if (state.IsDisposed)
                {
                    throw new ObjectDisposedException(
                        nameof(sourceStates),
                        $"CUDA decoder gather source row {row} is disposed.");
                }

                if (state.Position != position)
                {
                    throw new InvalidOperationException(
                        $"CUDA decoder gather row {row} is at position {state.Position}, not {position}.");
                }

                if (state.LayerCount != _geometry.NumHiddenLayers)
                {
                    throw new InvalidOperationException(
                        $"CUDA decoder gather row {row} has {state.LayerCount} layer(s); geometry requires {_geometry.NumHiddenLayers}.");
                }

                var lease = state.AcquireCudaResidentState(_cudaFormatId);
                try
                {
                    ValidateSourceLease(
                        lease,
                        state,
                        row,
                        expectedShape,
                        expectedBytes);
                    sourceLeases[row] = lease;
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }

            cancellationToken.ThrowIfCancellationRequested();
            targetArena = new CudaDecoderOrtCohortArena(
                position,
                sourceStates.Count,
                _geometry.NumHiddenLayers,
                expectedShape,
                _allocator);

            for (var row = 0; row < sourceStates.Count; row++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                targetStates[row] = targetArena.CreateRowState(
                    row,
                    sourceStates[row].NextTokenId);
                targetLeases[row] = targetStates[row].AcquireCudaResidentState(
                    _cudaFormatId);
            }

            var copies = new Task[checked(
                sourceStates.Count * _geometry.NumHiddenLayers * 2)];
            var copyIndex = 0;
            for (var row = 0; row < sourceStates.Count; row++)
            {
                var source = sourceLeases[row];
                var target = targetLeases[row];
                for (var layer = 0; layer < _geometry.NumHiddenLayers; layer++)
                {
                    var sourceLayer = source.GetLayer(layer);
                    var targetLayer = target.GetLayer(layer);
                    copies[copyIndex++] = _copyEngine.CopyAsync(
                        targetLayer.Key.DevicePointer,
                        sourceLayer.Key.DevicePointer,
                        checked((nuint)expectedBytes),
                        CudaMemcpyKind.DeviceToDevice,
                        cancellationToken).AsTask();
                    copies[copyIndex++] = _copyEngine.CopyAsync(
                        targetLayer.Value.DevicePointer,
                        sourceLayer.Value.DevicePointer,
                        checked((nuint)expectedBytes),
                        CudaMemcpyKind.DeviceToDevice,
                        cancellationToken).AsTask();
                }
            }

            // Task.WhenAll does not return until every submitted copy is terminal.
            // Post-submit cancellation is surfaced by CudaAsyncCopyEngine only
            // after the corresponding native event completes, so all pointers stay
            // valid throughout cleanup even on cancellation/failure.
            await Task.WhenAll(copies).ConfigureAwait(false);

            var batch = new CudaGatheredDecoderStateBatch(
                position,
                DeviceId,
                targetStates);
            success = true;
            return batch;
        }
        finally
        {
            for (var row = targetLeases.Length - 1; row >= 0; row--)
            {
                targetLeases[row]?.Dispose();
            }

            for (var row = sourceLeases.Length - 1; row >= 0; row--)
            {
                sourceLeases[row]?.Dispose();
            }

            if (!success)
            {
                for (var row = targetStates.Length - 1; row >= 0; row--)
                {
                    targetStates[row]?.Dispose();
                }
            }

            // The arena's builder reference survives target-state construction so
            // target pointers stay valid even if a later row or copy setup fails.
            // On success, each returned row state independently retains the arena.
            targetArena?.Release();
        }
    }

    private void ValidateSourceLease(
        DecoderOrtCudaResidentStateLease lease,
        DecoderOrtState state,
        int row,
        IReadOnlyList<long> expectedShape,
        long expectedBytes)
    {
        if (!StringComparer.Ordinal.Equals(lease.FormatId, _cudaFormatId))
        {
            throw new InvalidOperationException(
                $"CUDA decoder gather row {row} exposed format '{lease.FormatId}', not '{_cudaFormatId}'.");
        }

        if (lease.DeviceId != DeviceId)
        {
            throw new InvalidOperationException(
                $"CUDA decoder gather row {row} belongs to device {lease.DeviceId}, not {DeviceId}.");
        }

        if (lease.Position != state.Position ||
            lease.NextTokenId != state.NextTokenId ||
            lease.LayerCount != _geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                $"CUDA decoder gather row {row} resident lease does not match its decoder-state frontier.");
        }

        for (var layer = 0; layer < lease.LayerCount; layer++)
        {
            var resident = lease.GetLayer(layer);
            ValidateTensor(
                resident.Key,
                row,
                layer,
                "key",
                expectedShape,
                expectedBytes);
            ValidateTensor(
                resident.Value,
                row,
                layer,
                "value",
                expectedShape,
                expectedBytes);
        }
    }

    private void ValidateTensor(
        CudaDeviceTensorView tensor,
        int row,
        int layer,
        string slot,
        IReadOnlyList<long> expectedShape,
        long expectedBytes)
    {
        if (tensor.DeviceId != DeviceId ||
            tensor.ElementType != TensorElementType.Float ||
            tensor.ByteLength != expectedBytes ||
            !tensor.Shape.SequenceEqual(expectedShape))
        {
            throw new InvalidOperationException(
                $"CUDA decoder gather row {row} layer {layer} {slot} does not match expected FP32 geometry/device.");
        }
    }

    private static long CheckedFp32ByteLength(IReadOnlyList<long> shape)
    {
        long elements = 1;
        foreach (var dimension in shape)
        {
            if (dimension < 0)
            {
                throw new InvalidOperationException(
                    "CUDA decoder gather cannot materialize a dynamic negative tensor dimension.");
            }

            elements = checked(elements * dimension);
        }

        return checked(elements * sizeof(float));
    }
}
