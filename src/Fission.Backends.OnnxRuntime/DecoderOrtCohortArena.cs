namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Managed backing storage for one dense decoder cohort frontier.
///
/// Rows are stable for the lifetime of this arena. Individual DecoderOrtState
/// instances own independently disposable OrtValue slices while this object keeps
/// the complete batched key/value buffers available for the next decode step.
/// </summary>
internal sealed class DecoderOrtCohortArena
{
    private readonly float[][] _keyBuffers;
    private readonly float[][] _valueBuffers;

    public DecoderOrtCohortArena(
        int position,
        int batchSize,
        float[][] keyBuffers,
        float[][] valueBuffers)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (batchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        }

        ArgumentNullException.ThrowIfNull(keyBuffers);
        ArgumentNullException.ThrowIfNull(valueBuffers);

        if (keyBuffers.Length == 0 || keyBuffers.Length != valueBuffers.Length)
        {
            throw new ArgumentException(
                "A cohort arena requires matching non-empty key/value layer buffers.");
        }

        for (var layer = 0; layer < keyBuffers.Length; layer++)
        {
            ArgumentNullException.ThrowIfNull(keyBuffers[layer]);
            ArgumentNullException.ThrowIfNull(valueBuffers[layer]);
            if (keyBuffers[layer].Length != valueBuffers[layer].Length)
            {
                throw new ArgumentException(
                    $"Cohort arena layer {layer} key/value buffers have different lengths.");
            }
        }

        Position = position;
        BatchSize = batchSize;
        _keyBuffers = keyBuffers;
        _valueBuffers = valueBuffers;
    }

    public int Position { get; }
    public int BatchSize { get; }
    public int LayerCount => _keyBuffers.Length;

    public float[] GetKeyBuffer(int layer) => _keyBuffers[layer];
    public float[] GetValueBuffer(int layer) => _valueBuffers[layer];
}

internal readonly record struct DecoderOrtCohortSlice(
    DecoderOrtCohortArena Arena,
    int Row);
