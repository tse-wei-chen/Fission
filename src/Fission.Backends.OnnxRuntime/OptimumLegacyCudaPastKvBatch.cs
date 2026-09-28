using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Caller-owned zero-copy past-KV input transaction for one dense Optimum legacy
/// decoder cohort.
///
/// This transaction deliberately accepts only the full row set of one
/// CudaDecoderOrtCohortArena in natural row order. That invariant lets the next ORT
/// run bind the arena's existing batched allocations directly, with no device
/// gather, host repack, D2H or H2D transfer. Subsets, row reordering and mixed
/// arenas are rejected explicitly until a device-side gather path exists.
/// </summary>
internal sealed class OptimumLegacyCudaPastKvBatch : IDisposable
{
    private readonly CudaDecoderOrtCohortArena _arena;
    private readonly OrtValue[] _inputValues;
    private readonly ReadOnlyCollection<string> _inputNamesView;
    private readonly ReadOnlyCollection<OrtValue> _inputValuesView;
    private int _disposed;

    public OptimumLegacyCudaPastKvBatch(
        OptimumLegacyDecoderProfile profile,
        IReadOnlyList<DecoderOrtState> states)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(states);
        if (states.Count == 0)
        {
            throw new ArgumentException(
                "CUDA past-KV input transaction requires at least one state.",
                nameof(states));
        }

        var geometry = profile.Geometry;
        var contract = profile.Contract;
        if (geometry.KvElementType != TensorElementType.Float)
        {
            throw new ArgumentException(
                "CUDA past-KV input binding currently supports FP32 KV tensors only.",
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
                "CUDA past-KV input state collection contains null.",
                nameof(states));
        ObjectDisposedException.ThrowIf(first.IsDisposed, first);
        if (!first.TryGetCudaCohortSlice(out var firstSlice))
        {
            throw new InvalidOperationException(
                "CUDA past-KV input state is not backed by a CUDA cohort arena.");
        }

        _arena = firstSlice.Arena;
        PastSequenceLength = first.Position;
        BatchSize = states.Count;
        DeviceId = _arena.DeviceId;

        if (firstSlice.Row != 0)
        {
            throw new InvalidOperationException(
                "Zero-copy CUDA past-KV input requires row 0 to be the first state.");
        }

        if (_arena.BatchSize != BatchSize)
        {
            throw new InvalidOperationException(
                $"Zero-copy CUDA past-KV input requires the complete arena batch of {_arena.BatchSize} row(s), " +
                $"but received {BatchSize} state(s).");
        }

        if (_arena.Position != PastSequenceLength)
        {
            throw new InvalidOperationException(
                "CUDA cohort arena position does not match the past-KV state frontier.");
        }

        if (_arena.LayerCount != geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                $"CUDA cohort arena contains {_arena.LayerCount} layer(s), but the profile requires " +
                $"{geometry.NumHiddenLayers}.");
        }

        var expectedPerSequenceShape = geometry.GetPastKvShape(
            batchSize: 1,
            PastSequenceLength);
        if (!_arena.PerSequenceShape.SequenceEqual(expectedPerSequenceShape))
        {
            throw new InvalidOperationException(
                "CUDA cohort arena tensor geometry does not match the requested Optimum past-KV shape.");
        }

        for (var index = 0; index < states.Count; index++)
        {
            var state = states[index] ??
                throw new ArgumentException(
                    $"CUDA past-KV input state at index {index} is null.",
                    nameof(states));
            ObjectDisposedException.ThrowIf(state.IsDisposed, state);
            if (state.Position != PastSequenceLength)
            {
                throw new InvalidOperationException(
                    "Zero-copy CUDA past-KV input cannot combine different causal positions.");
            }

            if (!state.TryGetCudaCohortSlice(out var slice) ||
                !ReferenceEquals(slice.Arena, _arena))
            {
                throw new InvalidOperationException(
                    "Zero-copy CUDA past-KV input requires every state to share one CUDA cohort arena.");
            }

            if (slice.Row != index)
            {
                throw new InvalidOperationException(
                    $"Zero-copy CUDA past-KV input requires natural row order 0..{BatchSize - 1}; " +
                    $"state index {index} refers to arena row {slice.Row}.");
            }
        }

        var inputNames = new string[checked(geometry.NumHiddenLayers * 2)];
        _inputValues = new OrtValue[inputNames.Length];
        _arena.Retain();
        var produced = 0;
        try
        {
            for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
            {
                var offset = checked(layer * 2);
                inputNames[offset] = DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PastKeyNames,
                    layer);
                inputNames[offset + 1] = DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PastValueNames,
                    layer);

                var layerState = _arena.CreateBatchedLayer(layer);
                _inputValues[offset] = layerState.Key;
                produced++;
                _inputValues[offset + 1] = layerState.Value;
                produced++;
            }
        }
        catch
        {
            for (var index = produced - 1; index >= 0; index--)
            {
                _inputValues[index].Dispose();
            }

            _arena.Release();
            throw;
        }

        _inputNamesView = Array.AsReadOnly(inputNames);
        _inputValuesView = Array.AsReadOnly(_inputValues);
    }

    public int PastSequenceLength { get; }
    public int BatchSize { get; }
    public int DeviceId { get; }
    public IReadOnlyList<string> InputNames => _inputNamesView;
    public IReadOnlyList<OrtValue> InputValues
    {
        get
        {
            ThrowIfDisposed();
            return _inputValuesView;
        }
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

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}
