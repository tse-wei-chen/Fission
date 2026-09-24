using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

public readonly record struct DecoderOrtLayerState(
    OrtValue Key,
    OrtValue Value);

/// <summary>
/// Immutable owned physical decoder state for one version.
///
/// Besides KV tensors, the state may retain NextTokenId: the sampled token that
/// must be fed as input to the next one-token decode step. Keeping that token in
/// the same immutable object as KV means snapshot/fork/restore moves the complete
/// causal frontier rather than only the cache payload.
///
/// Construction transfers ownership of every layer Key/Value OrtValue to this
/// instance only after the complete payload validates. The same OrtValue instance
/// cannot appear in more than one owned slot. DecoderStateStore may then share the
/// state object across snapshots/branches and the OrtValues are disposed exactly
/// once when the final state owner disappears.
/// </summary>
public sealed class DecoderOrtState : IDisposable
{
    private readonly DecoderOrtLayerState[] _layers;
    private int _disposed;

    public DecoderOrtState(
        int position,
        IReadOnlyList<DecoderOrtLayerState> layers,
        int? nextTokenId = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (nextTokenId is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(nextTokenId));
        }

        ArgumentNullException.ThrowIfNull(layers);
        if (layers.Count == 0)
        {
            throw new ArgumentException(
                "Decoder KV state must contain at least one layer.",
                nameof(layers));
        }

        var owned = new HashSet<OrtValue>(ReferenceEqualityComparer.Instance);
        var validated = new DecoderOrtLayerState[layers.Count];

        for (var index = 0; index < layers.Count; index++)
        {
            var layer = layers[index];
            ArgumentNullException.ThrowIfNull(layer.Key);
            ArgumentNullException.ThrowIfNull(layer.Value);

            if (!layer.Key.IsTensor)
            {
                throw new ArgumentException(
                    $"Decoder layer {index} key OrtValue is not a tensor.",
                    nameof(layers));
            }

            if (!layer.Value.IsTensor)
            {
                throw new ArgumentException(
                    $"Decoder layer {index} value OrtValue is not a tensor.",
                    nameof(layers));
            }

            if (!owned.Add(layer.Key))
            {
                throw new ArgumentException(
                    $"Decoder layer {index} key OrtValue is already owned by another KV slot.",
                    nameof(layers));
            }

            if (!owned.Add(layer.Value))
            {
                throw new ArgumentException(
                    $"Decoder layer {index} value OrtValue is already owned by another KV slot.",
                    nameof(layers));
            }

            validated[index] = layer;
        }

        Position = position;
        NextTokenId = nextTokenId;
        _layers = validated;
    }

    public int Position { get; }

    /// <summary>
    /// Sampled token to feed into the next decode step. It is null for payloads
    /// created outside the causal-LM execution protocol (for example low-level
    /// ownership tests).
    /// </summary>
    public int? NextTokenId { get; }

    public int LayerCount => _layers.Length;
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public DecoderOrtLayerState GetLayer(int layer)
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

        // Dispose in reverse ownership order, mirroring ORT's disposable output
        // collection behavior and preserving dependencies if a provider ever
        // layers value resources internally.
        for (var index = _layers.Length - 1; index >= 0; index--)
        {
            _layers[index].Value.Dispose();
            _layers[index].Key.Dispose();
        }
    }
}
