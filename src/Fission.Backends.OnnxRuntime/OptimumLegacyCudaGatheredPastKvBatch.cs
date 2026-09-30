using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Device-only gather transaction for an arbitrary ordered set of CUDA-resident
/// Optimum legacy decoder states.
///
/// Every source row is independently retained through the resident-state
/// capability while asynchronous device-to-device copies are in flight. The
/// destination is a new dense CUDA cohort arena whose batched OrtValues can be
/// appended directly to the next ORT past-KV input list. No KV bytes pass through
/// host memory.
/// </summary>
internal sealed class OptimumLegacyCudaGatheredPastKvBatch : IDisposable
{
    private readonly CudaDecoderOrtCohortArena _arena;
    private readonly OrtValue[] _inputValues;
    private readonly ReadOnlyCollection<string> _inputNamesView;
    private readonly ReadOnlyCollection<OrtValue> _inputValuesView;
    private int _disposed;

    private OptimumLegacyCudaGatheredPastKvBatch(
        CudaDecoderOrtCohortArena arena,
        string[] inputNames,
        OrtValue[] inputValues,
        long copiedBytes)
    {
        _arena = arena;
        _inputValues = inputValues;
        _inputNamesView = Array.AsReadOnly(inputNames);
        _inputValuesView = Array.AsReadOnly(inputValues);
        PastSequenceLength = arena.Position;
        BatchSize = arena.BatchSize;
        DeviceId = arena.DeviceId;
        CopiedBytes = copiedBytes;
    }

    public int PastSequenceLength { get; }
    public int BatchSize { get; }
    public int DeviceId { get; }
    public long CopiedBytes { get; }
    public IReadOnlyList<string> InputNames => _inputNamesView;

    public IReadOnlyList<OrtValue> InputValues
    {
        get
        {
            ThrowIfDisposed();
            return _inputValuesView;
        }
    }

    public static async ValueTask<OptimumLegacyCudaGatheredPastKvBatch> CreateAsync(
        OptimumLegacyDecoderProfile profile,
        IReadOnlyList<DecoderOrtState> states,
        string cudaFormatId,
        CudaDeviceMemoryAllocator allocator,
        CudaDeviceBoundAsyncCopyEngine copyEngine,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(states);
        ArgumentException.ThrowIfNullOrWhiteSpace(cudaFormatId);
        ArgumentNullException.ThrowIfNull(allocator);
        ArgumentNullException.ThrowIfNull(copyEngine);
        cancellationToken.ThrowIfCancellationRequested();

        if (states.Count == 0)
        {
            throw new ArgumentException(
                "CUDA gathered past-KV input requires at least one source state.",
                nameof(states));
        }

        if (allocator.DeviceId != copyEngine.DeviceId)
        {
            throw new ArgumentException(
                $"CUDA gather allocator targets device {allocator.DeviceId}, but copy engine targets device {copyEngine.DeviceId}.",
                nameof(copyEngine));
        }

        var geometry = profile.Geometry;
        var contract = profile.Contract;
        if (geometry.KvElementType != TensorElementType.Float)
        {
            throw new ArgumentException(
                "CUDA gathered past-KV input currently supports FP32 KV tensors only.",
                nameof(profile));
        }

        if (string.IsNullOrWhiteSpace(contract.PastKeyNames) ||
            string.IsNullOrWhiteSpace(contract.PastValueNames))
        {
            throw new ArgumentException(
                "Optimum legacy profile must declare past key/value input names.",
                nameof(profile));
        }

        var first = states[0] ??
            throw new ArgumentException(
                "CUDA gathered past-KV source collection contains null.",
                nameof(states));
        ObjectDisposedException.ThrowIf(first.IsDisposed, first);
        var position = first.Position;
        if (position <= 0)
        {
            throw new InvalidOperationException(
                "CUDA gathered past-KV input requires a positive causal frontier.");
        }

        var perSequenceShape = geometry.GetPastKvShape(
            batchSize: 1,
            position);
        var expectedBytes = CheckedTensorByteLength(perSequenceShape);

        // Resolve all model-contract names before acquiring native ownership.
        var inputNames = new string[checked(geometry.NumHiddenLayers * 2)];
        for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
        {
            var offset = checked(layer * 2);
            inputNames[offset] = DecoderOnlyOnnxContract.ExpandLayerName(
                contract.PastKeyNames,
                layer);
            inputNames[offset + 1] = DecoderOnlyOnnxContract.ExpandLayerName(
                contract.PastValueNames,
                layer);
        }

        var sourceLeases = new DecoderOrtCudaResidentStateLease[states.Count];
        CudaDecoderOrtCohortArena? destinationArena = null;
        var targetStates = new DecoderOrtState?[states.Count];
        var targetLeases = new DecoderOrtCudaResidentStateLease?[states.Count];
        OrtValue[]? inputValues = null;
        var producedValues = 0;
        var transferred = false;

        try
        {
            // Validate and retain every source before allocating the destination.
            // This permits duplicate rows and mixed source arenas as long as every
            // state has the same physical format, geometry, device and frontier.
            for (var row = 0; row < states.Count; row++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var state = states[row] ??
                    throw new ArgumentException(
                        $"CUDA gathered past-KV source state at index {row} is null.",
                        nameof(states));
                ObjectDisposedException.ThrowIf(state.IsDisposed, state);
                if (state.Position != position)
                {
                    throw new InvalidOperationException(
                        "CUDA gathered past-KV input cannot combine different causal positions.");
                }

                if (state.LayerCount != geometry.NumHiddenLayers)
                {
                    throw new InvalidOperationException(
                        $"CUDA source row {row} contains {state.LayerCount} layer(s), but profile requires {geometry.NumHiddenLayers}.");
                }

                var lease = state.AcquireCudaResidentState(cudaFormatId);
                sourceLeases[row] = lease;
                ValidateLease(
                    lease,
                    row,
                    position,
                    allocator.DeviceId,
                    geometry.NumHiddenLayers,
                    perSequenceShape,
                    expectedBytes);
            }

            destinationArena = new CudaDecoderOrtCohortArena(
                position,
                states.Count,
                geometry.NumHiddenLayers,
                perSequenceShape,
                allocator);

            // Temporary row states/leases give the gather transaction validated
            // raw destination pointers while the arena's builder reference owns
            // the final dense allocation lifetime.
            for (var row = 0; row < states.Count; row++)
            {
                var targetState = destinationArena.CreateRowState(row, nextTokenId: null);
                targetStates[row] = targetState;
                targetLeases[row] = targetState.AcquireCudaResidentState(cudaFormatId);
            }

            // Adjacent rows of one CUDA cohort arena are physically contiguous in
            // every layer allocation. Coalesce maximal forward-contiguous source
            // runs so one cudaMemcpyAsync/event pair moves many rows at once. A
            // duplicate, reorder, mixed arena, or standalone imported state starts
            // a new run and therefore retains the exact arbitrary-gather semantics.
            var copyRuns = BuildCopyRuns(states);
            var copyTasks = new Task[checked(copyRuns.Count * geometry.NumHiddenLayers * 2)];
            var copyIndex = 0;
            foreach (var run in copyRuns)
            {
                var source = sourceLeases[run.StartRow];
                var target = targetLeases[run.StartRow]
                    ?? throw new InvalidOperationException("CUDA gather target lease was not created.");
                var runBytes = checked(expectedBytes * run.RowCount);
                for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
                {
                    var sourceLayer = source.GetLayer(layer);
                    var targetLayer = target.GetLayer(layer);
                    copyTasks[copyIndex++] = copyEngine.CopyAsync(
                            targetLayer.Key.DevicePointer,
                            sourceLayer.Key.DevicePointer,
                            checked((nuint)runBytes),
                            CudaMemcpyKind.DeviceToDevice,
                            cancellationToken)
                        .AsTask();
                    copyTasks[copyIndex++] = copyEngine.CopyAsync(
                            targetLayer.Value.DevicePointer,
                            sourceLayer.Value.DevicePointer,
                            checked((nuint)runBytes),
                            CudaMemcpyKind.DeviceToDevice,
                            cancellationToken)
                        .AsTask();
                }
            }

            // Task.WhenAll does not return until every submitted copy is terminal.
            // CudaAsyncCopyEngine itself delays post-submission cancellation until
            // its native completion event fires, so source and destination leases
            // remain valid across cancellation and failure cleanup.
            await Task.WhenAll(copyTasks).ConfigureAwait(false);

            inputValues = new OrtValue[inputNames.Length];
            for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
            {
                var pair = destinationArena.CreateBatchedLayer(layer);
                var offset = checked(layer * 2);
                inputValues[offset] = pair.Key;
                producedValues++;
                inputValues[offset + 1] = pair.Value;
                producedValues++;
            }

            var result = new OptimumLegacyCudaGatheredPastKvBatch(
                destinationArena,
                inputNames,
                inputValues,
                checked((long)states.Count * geometry.NumHiddenLayers * 2L * expectedBytes));
            transferred = true;
            destinationArena = null;
            inputValues = null;
            return result;
        }
        finally
        {
            if (!transferred && inputValues is not null)
            {
                for (var index = producedValues - 1; index >= 0; index--)
                {
                    inputValues[index].Dispose();
                }
            }

            for (var row = targetLeases.Length - 1; row >= 0; row--)
            {
                targetLeases[row]?.Dispose();
                targetStates[row]?.Dispose();
            }

            destinationArena?.Release();

            for (var row = sourceLeases.Length - 1; row >= 0; row--)
            {
                sourceLeases[row]?.Dispose();
            }
        }
    }

    /// <summary>
    /// Creates one independently retained row view over the gathered dense arena.
    /// This is used by gather-aware execution wrappers to feed the existing CUDA
    /// binding without copying KV again. The returned state may outlive this batch.
    /// </summary>
    public DecoderOrtState CreateRowState(int row, int? nextTokenId)
    {
        ThrowIfDisposed();
        return _arena.CreateRowState(row, nextTokenId);
    }

    public void AppendInputs(
        ICollection<string> inputNames,
        ICollection<OrtValue> inputValues)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(inputNames);
        ArgumentNullException.ThrowIfNull(inputValues);
        for (var index = 0; index < _inputValues.Length; index++)
        {
            inputNames.Add(_inputNamesView[index]);
            inputValues.Add(_inputValues[index]);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        for (var index = _inputValues.Length - 1; index >= 0; index--)
        {
            _inputValues[index].Dispose();
        }

        _arena.Release();
    }

    private static List<CopyRun> BuildCopyRuns(IReadOnlyList<DecoderOrtState> states)
    {
        var runs = new List<CopyRun>(states.Count);
        var startRow = 0;
        while (startRow < states.Count)
        {
            var rowCount = 1;
            if (states[startRow].TryGetCudaCohortSlice(out var firstSlice))
            {
                var arena = firstSlice.Arena;
                var expectedSourceRow = checked(firstSlice.Row + 1);
                while (startRow + rowCount < states.Count &&
                       states[startRow + rowCount].TryGetCudaCohortSlice(out var nextSlice) &&
                       ReferenceEquals(nextSlice.Arena, arena) &&
                       nextSlice.Row == expectedSourceRow)
                {
                    rowCount++;
                    expectedSourceRow = checked(expectedSourceRow + 1);
                }
            }

            runs.Add(new CopyRun(startRow, rowCount));
            startRow += rowCount;
        }

        return runs;
    }

    private static void ValidateLease(
        DecoderOrtCudaResidentStateLease lease,
        int row,
        int position,
        int deviceId,
        int layerCount,
        IReadOnlyList<long> expectedShape,
        long expectedBytes)
    {
        if (lease.Position != position ||
            lease.DeviceId != deviceId ||
            lease.LayerCount != layerCount)
        {
            throw new InvalidOperationException(
                $"CUDA source row {row} resident lease does not match the gather frontier/device/layer geometry.");
        }

        for (var layer = 0; layer < layerCount; layer++)
        {
            var pair = lease.GetLayer(layer);
            ValidateTensor(pair.Key, row, layer, "key", deviceId, expectedShape, expectedBytes);
            ValidateTensor(pair.Value, row, layer, "value", deviceId, expectedShape, expectedBytes);
        }
    }

    private static void ValidateTensor(
        CudaDeviceTensorView tensor,
        int row,
        int layer,
        string slot,
        int deviceId,
        IReadOnlyList<long> expectedShape,
        long expectedBytes)
    {
        if (tensor.DeviceId != deviceId ||
            tensor.ElementType != TensorElementType.Float ||
            tensor.ByteLength != expectedBytes ||
            !tensor.Shape.SequenceEqual(expectedShape))
        {
            throw new InvalidOperationException(
                $"CUDA source row {row} layer {layer} {slot} does not match the requested FP32 gather geometry.");
        }
    }

    private static long CheckedTensorByteLength(IReadOnlyList<long> shape)
    {
        var elementCount = 1L;
        foreach (var dimension in shape)
        {
            if (dimension <= 0)
            {
                throw new InvalidOperationException(
                    "CUDA gathered past-KV shape must contain only positive dimensions.");
            }

            elementCount = checked(elementCount * dimension);
        }

        return checked(elementCount * sizeof(float));
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);

    private readonly record struct CopyRun(int StartRow, int RowCount);
}
