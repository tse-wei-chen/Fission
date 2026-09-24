using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

public sealed record DecoderKvLayerGeometry(
    IReadOnlyList<long> KeyShape,
    IReadOnlyList<long> ValueShape);

public sealed record DecoderOutputGeometry(
    IReadOnlyList<long> LogitsShape,
    IReadOnlyList<DecoderKvLayerGeometry> Layers);

/// <summary>
/// Owns the caller-preallocated outputs for one decoder model step.
/// Logits remain arena-owned for sampling. Present K/V tensors may be detached
/// exactly once into a DecoderOrtState after a successful run.
/// </summary>
public sealed class DecoderOutputArena<T> : IDisposable
    where T : unmanaged
{
    private readonly OwnedOrtTensor<T> _logits;
    private readonly OwnedOrtTensor<T>[] _keys;
    private readonly OwnedOrtTensor<T>[] _values;
    private readonly string[] _outputNames;
    private int _kvTransferred;
    private int _disposed;

    public DecoderOutputArena(
        DecoderOnlyOnnxContract contract,
        DecoderOutputGeometry geometry)
    {
        ArgumentNullException.ThrowIfNull(contract);
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(geometry.LogitsShape);
        ArgumentNullException.ThrowIfNull(geometry.Layers);

        // Reuse contract validation for complete cache-pattern and layer-count
        // invariants before allocating any native/pinned resources.
        _ = contract.ToSessionContract();
        if (!contract.UsesPastKeyValues)
        {
            throw new InvalidOperationException(
                "Decoder output arena requires a contract with present key/value cache tensors.");
        }

        if (geometry.Layers.Count != contract.NumHiddenLayers)
        {
            throw new ArgumentException(
                $"Decoder output geometry contains {geometry.Layers.Count} layers but contract declares {contract.NumHiddenLayers}.",
                nameof(geometry));
        }

        _logits = new OwnedOrtTensor<T>(geometry.LogitsShape.ToArray());
        _keys = new OwnedOrtTensor<T>[geometry.Layers.Count];
        _values = new OwnedOrtTensor<T>[geometry.Layers.Count];
        _outputNames = new string[1 + (geometry.Layers.Count * 2)];
        _outputNames[0] = contract.Logits;

        var allocatedLayers = 0;
        try
        {
            for (var layer = 0; layer < geometry.Layers.Count; layer++)
            {
                var layerGeometry = geometry.Layers[layer]
                    ?? throw new ArgumentException(
                        $"Decoder output geometry layer {layer} is null.",
                        nameof(geometry));

                _keys[layer] = new OwnedOrtTensor<T>(layerGeometry.KeyShape.ToArray());
                _values[layer] = new OwnedOrtTensor<T>(layerGeometry.ValueShape.ToArray());
                _outputNames[1 + (layer * 2)] =
                    DecoderOnlyOnnxContract.ExpandLayerName(contract.PresentKeyNames!, layer);
                _outputNames[2 + (layer * 2)] =
                    DecoderOnlyOnnxContract.ExpandLayerName(contract.PresentValueNames!, layer);
                allocatedLayers++;
            }
        }
        catch
        {
            for (var layer = allocatedLayers; layer >= 0; layer--)
            {
                if (layer < _values.Length)
                {
                    _values[layer]?.Dispose();
                    _keys[layer]?.Dispose();
                }
            }

            _logits.Dispose();
            throw;
        }
    }

    public int LayerCount => _keys.Length;
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;
    public bool IsKvTransferred => Volatile.Read(ref _kvTransferred) != 0;
    public IReadOnlyList<string> OutputNames => _outputNames;

    public IReadOnlyList<OrtValue> OutputValues
    {
        get
        {
            ThrowIfDisposed();
            if (IsKvTransferred)
            {
                throw new InvalidOperationException(
                    "Decoder output KV tensors have already been transferred to a DecoderOrtState.");
            }

            var outputs = new OrtValue[_outputNames.Length];
            outputs[0] = _logits.Value;
            for (var layer = 0; layer < _keys.Length; layer++)
            {
                outputs[1 + (layer * 2)] = _keys[layer].Value;
                outputs[2 + (layer * 2)] = _values[layer].Value;
            }

            return outputs;
        }
    }

    public ReadOnlySpan<T> Logits
    {
        get
        {
            ThrowIfDisposed();
            return _logits.ReadOnlySpan;
        }
    }

    public Span<T> MutableLogits
    {
        get
        {
            ThrowIfDisposed();
            return _logits.Span;
        }
    }

    public DecoderOrtState DetachKvState(int position)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (Interlocked.CompareExchange(ref _kvTransferred, 1, 0) != 0)
        {
            throw new InvalidOperationException(
                "Decoder output KV tensors have already been transferred.");
        }

        var detached = new List<OrtValue>(_keys.Length * 2);
        try
        {
            var layers = new DecoderOrtLayerState[_keys.Length];
            for (var layer = 0; layer < _keys.Length; layer++)
            {
                var key = _keys[layer].DetachValue();
                detached.Add(key);
                var value = _values[layer].DetachValue();
                detached.Add(value);
                layers[layer] = new DecoderOrtLayerState(key, value);
            }

            var state = new DecoderOrtState(position, layers);
            detached.Clear(); // DecoderOrtState now owns every detached value.
            return state;
        }
        catch
        {
            foreach (var value in detached)
            {
                value.Dispose();
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

        for (var layer = _keys.Length - 1; layer >= 0; layer--)
        {
            _values[layer]?.Dispose();
            _keys[layer]?.Dispose();
        }

        _logits.Dispose();
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(IsDisposed, this);
}
