using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Caller-owned present-KV output transaction for one Optimum legacy decoder
/// cohort run.
///
/// The transaction allocates one CUDA key/value tensor per layer for the complete
/// batch, exposes those tensors with the exact Optimum present-KV output names, and
/// materializes per-row DecoderOrtState views after the run. Disposing the batch
/// first disposes all batched OrtValue wrappers and only then releases the arena's
/// builder reference. Row states created before disposal retain the CUDA arena
/// independently.
/// </summary>
internal sealed class OptimumLegacyCudaPresentKvBatch : IDisposable
{
    private readonly CudaDecoderOrtCohortArena _arena;
    private readonly OrtValue[] _outputValues;
    private readonly ReadOnlyCollection<string> _outputNamesView;
    private readonly ReadOnlyCollection<OrtValue> _outputValuesView;
    private int _disposed;

    public OptimumLegacyCudaPresentKvBatch(
        OptimumLegacyDecoderProfile profile,
        int batchSize,
        int pastSequenceLength,
        int sequenceLength,
        CudaDeviceMemoryAllocator allocator)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(allocator);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(pastSequenceLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(sequenceLength, 1);

        var geometry = profile.Geometry;
        var contract = profile.Contract;
        if (string.IsNullOrWhiteSpace(contract.PresentKeyNames) ||
            string.IsNullOrWhiteSpace(contract.PresentValueNames))
        {
            throw new ArgumentException(
                "Optimum legacy profile must declare present key/value output names.",
                nameof(profile));
        }

        Position = checked(pastSequenceLength + sequenceLength);
        BatchSize = batchSize;
        DeviceId = allocator.DeviceId;
        var perSequenceShape = geometry.GetPresentKvShape(
            batchSize: 1,
            pastSequenceLength,
            sequenceLength);
        _arena = new CudaDecoderOrtCohortArena(
            Position,
            batchSize,
            geometry.NumHiddenLayers,
            perSequenceShape,
            allocator,
            geometry.KvElementType);

        var outputNames = new string[checked(geometry.NumHiddenLayers * 2)];
        _outputValues = new OrtValue[outputNames.Length];
        var produced = 0;
        try
        {
            for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
            {
                var offset = checked(layer * 2);
                outputNames[offset] = DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PresentKeyNames,
                    layer);
                outputNames[offset + 1] = DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PresentValueNames,
                    layer);

                // Resolve all contract metadata before creating the caller-owned
                // CUDA views so name-format failure cannot strand untracked OrtValues.
                var layerState = _arena.CreateBatchedLayer(layer);
                _outputValues[offset] = layerState.Key;
                produced++;
                _outputValues[offset + 1] = layerState.Value;
                produced++;
            }
        }
        catch
        {
            for (var index = produced - 1; index >= 0; index--)
            {
                _outputValues[index].Dispose();
            }

            _arena.Release();
            throw;
        }

        _outputNamesView = Array.AsReadOnly(outputNames);
        _outputValuesView = Array.AsReadOnly(_outputValues);
    }

    public int Position { get; }
    public int BatchSize { get; }
    public int DeviceId { get; }
    public IReadOnlyList<string> OutputNames => _outputNamesView;
    public IReadOnlyList<OrtValue> OutputValues
    {
        get
        {
            ThrowIfDisposed();
            return _outputValuesView;
        }
    }

    public DecoderOrtState CreateRowState(
        int row,
        int? nextTokenId)
    {
        ThrowIfDisposed();
        return _arena.CreateRowState(row, nextTokenId);
    }

    public void AppendOutputs(
        ICollection<string> outputNames,
        ICollection<OrtValue> outputValues)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(outputValues);
        for (var index = 0; index < _outputValues.Length; index++)
        {
            outputNames.Add(_outputNamesView[index]);
            outputValues.Add(_outputValues[index]);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        for (var index = _outputValues.Length - 1; index >= 0; index--)
        {
            _outputValues[index].Dispose();
        }

        _arena.Release();
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}
