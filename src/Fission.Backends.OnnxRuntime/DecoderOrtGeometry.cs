using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Physical tensor geometry for one decoder export family.
///
/// The initial supported KV layout is the common legacy cache layout
/// [batch, kv_heads, sequence, head_dim]. Keeping geometry explicit prevents the
/// generic execution adapter from guessing model-specific tensor shapes.
/// </summary>
public sealed class DecoderOrtGeometry
{
    public DecoderOrtGeometry(
        int numHiddenLayers,
        int numKvHeads,
        int headDim,
        int vocabularySize,
        TensorElementType kvElementType)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numHiddenLayers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numKvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabularySize);

        if (kvElementType is not (
            TensorElementType.Float or
            TensorElementType.Float16 or
            TensorElementType.BFloat16))
        {
            throw new ArgumentOutOfRangeException(
                nameof(kvElementType),
                kvElementType,
                "Decoder KV element type must be Float, Float16, or BFloat16.");
        }

        NumHiddenLayers = numHiddenLayers;
        NumKvHeads = numKvHeads;
        HeadDim = headDim;
        VocabularySize = vocabularySize;
        KvElementType = kvElementType;
    }

    public int NumHiddenLayers { get; }
    public int NumKvHeads { get; }
    public int HeadDim { get; }
    public int VocabularySize { get; }
    public TensorElementType KvElementType { get; }

    public int KvElementSizeBytes => KvElementType switch
    {
        TensorElementType.Float => sizeof(float),
        TensorElementType.Float16 => sizeof(ushort),
        TensorElementType.BFloat16 => sizeof(ushort),
        _ => throw new InvalidOperationException(
            $"Unsupported decoder KV element type {KvElementType}.")
    };

    public long[] GetInputIdsShape(int batchSize, int sequenceLength)
    {
        ValidatePositive(batchSize, nameof(batchSize));
        ValidatePositive(sequenceLength, nameof(sequenceLength));
        return new long[] { batchSize, sequenceLength };
    }

    public long[] GetAttentionMaskShape(
        int batchSize,
        int pastSequenceLength,
        int sequenceLength)
    {
        ValidatePositive(batchSize, nameof(batchSize));
        ValidateNonNegative(pastSequenceLength, nameof(pastSequenceLength));
        ValidatePositive(sequenceLength, nameof(sequenceLength));
        return new long[]
        {
            batchSize,
            checked((long)pastSequenceLength + sequenceLength)
        };
    }

    public long[] GetPositionIdsShape(int batchSize, int sequenceLength) =>
        GetInputIdsShape(batchSize, sequenceLength);

    public long[] GetPastKvShape(int batchSize, int pastSequenceLength)
    {
        ValidatePositive(batchSize, nameof(batchSize));
        ValidateNonNegative(pastSequenceLength, nameof(pastSequenceLength));
        return new long[]
        {
            batchSize,
            NumKvHeads,
            pastSequenceLength,
            HeadDim
        };
    }

    public long[] GetPresentKvShape(
        int batchSize,
        int pastSequenceLength,
        int sequenceLength)
    {
        ValidatePositive(batchSize, nameof(batchSize));
        ValidateNonNegative(pastSequenceLength, nameof(pastSequenceLength));
        ValidatePositive(sequenceLength, nameof(sequenceLength));
        return new long[]
        {
            batchSize,
            NumKvHeads,
            checked((long)pastSequenceLength + sequenceLength),
            HeadDim
        };
    }

    public long[] GetLogitsShape(int batchSize, int sequenceLength)
    {
        ValidatePositive(batchSize, nameof(batchSize));
        ValidatePositive(sequenceLength, nameof(sequenceLength));
        return new long[] { batchSize, sequenceLength, VocabularySize };
    }

    public long GetKvElementCountPerSequence(int sequenceLength)
    {
        ValidateNonNegative(sequenceLength, nameof(sequenceLength));
        return checked(
            (long)NumHiddenLayers *
            2L *
            NumKvHeads *
            sequenceLength *
            HeadDim);
    }

    public long GetKvBytesPerSequence(int sequenceLength) =>
        checked(GetKvElementCountPerSequence(sequenceLength) * KvElementSizeBytes);

    private static void ValidatePositive(int value, string parameterName)
    {
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static void ValidateNonNegative(int value, string parameterName)
    {
        if (value < 0)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }
}
