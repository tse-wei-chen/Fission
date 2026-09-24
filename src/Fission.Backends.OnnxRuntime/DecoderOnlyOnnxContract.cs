using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Export manifest for a decoder-only causal language model. Tensor names remain
/// model/export specific while the runtime consumes a stable logical vocabulary.
/// Cache name patterns may use either "%d" or "{0}" as the zero-based layer slot.
/// </summary>
public sealed record DecoderOnlyOnnxContract(
    int NumHiddenLayers,
    string InputIds,
    string Logits,
    string? AttentionMask = null,
    string? PositionIds = null,
    string? PastKeyNames = null,
    string? PastValueNames = null,
    string? PresentKeyNames = null,
    string? PresentValueNames = null,
    int InputIdsRank = 2,
    int? LogitsRank = 3,
    TensorElementType InputIdsElementType = TensorElementType.Int64,
    IReadOnlyList<OnnxTensorContract>? AdditionalInputs = null,
    IReadOnlyList<OnnxTensorContract>? AdditionalOutputs = null)
{
    public bool UsesPastKeyValues =>
        PastKeyNames is not null ||
        PastValueNames is not null ||
        PresentKeyNames is not null ||
        PresentValueNames is not null;

    public OnnxSessionContract ToSessionContract()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(NumHiddenLayers);
        ValidateName(InputIds, nameof(InputIds));
        ValidateName(Logits, nameof(Logits));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(InputIdsRank);
        if (LogitsRank is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(LogitsRank));
        }

        ValidateCachePatterns();

        var inputs = new List<OnnxTensorContract>
        {
            new("input_ids", InputIds, InputIdsRank, InputIdsElementType)
        };

        if (!string.IsNullOrWhiteSpace(AttentionMask))
        {
            inputs.Add(new OnnxTensorContract("attention_mask", AttentionMask!, Rank: 2));
        }

        if (!string.IsNullOrWhiteSpace(PositionIds))
        {
            inputs.Add(new OnnxTensorContract(
                "position_ids",
                PositionIds!,
                Rank: 2,
                ElementType: TensorElementType.Int64));
        }

        var outputs = new List<OnnxTensorContract>
        {
            new("logits", Logits, LogitsRank)
        };

        if (UsesPastKeyValues)
        {
            for (var layer = 0; layer < NumHiddenLayers; layer++)
            {
                inputs.Add(new OnnxTensorContract(
                    $"past_key[{layer}]",
                    ExpandLayerName(PastKeyNames!, layer)));
                inputs.Add(new OnnxTensorContract(
                    $"past_value[{layer}]",
                    ExpandLayerName(PastValueNames!, layer)));
                outputs.Add(new OnnxTensorContract(
                    $"present_key[{layer}]",
                    ExpandLayerName(PresentKeyNames!, layer)));
                outputs.Add(new OnnxTensorContract(
                    $"present_value[{layer}]",
                    ExpandLayerName(PresentValueNames!, layer)));
            }
        }

        if (AdditionalInputs is not null)
        {
            inputs.AddRange(AdditionalInputs);
        }

        if (AdditionalOutputs is not null)
        {
            outputs.AddRange(AdditionalOutputs);
        }

        return new OnnxSessionContract(inputs, outputs);
    }

    public static string ExpandLayerName(string pattern, int layer)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pattern);
        ArgumentOutOfRangeException.ThrowIfNegative(layer);

        if (pattern.Contains("%d", StringComparison.Ordinal))
        {
            return pattern.Replace(
                "%d",
                layer.ToString(CultureInfo.InvariantCulture),
                StringComparison.Ordinal);
        }

        if (pattern.Contains("{0}", StringComparison.Ordinal))
        {
            return string.Format(CultureInfo.InvariantCulture, pattern, layer);
        }

        throw new InvalidOperationException(
            $"Layer tensor name pattern '{pattern}' must contain either '%d' or '{{0}}'.");
    }

    private void ValidateCachePatterns()
    {
        if (!UsesPastKeyValues)
        {
            return;
        }

        var patterns = new Dictionary<string, string?>
        {
            [nameof(PastKeyNames)] = PastKeyNames,
            [nameof(PastValueNames)] = PastValueNames,
            [nameof(PresentKeyNames)] = PresentKeyNames,
            [nameof(PresentValueNames)] = PresentValueNames
        };

        foreach (var (name, value) in patterns)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new InvalidOperationException(
                    $"Decoder cache contract is incomplete: {name} is required when any past/present cache pattern is configured.");
            }

            _ = ExpandLayerName(value!, 0);
        }
    }

    private static void ValidateName(string value, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException(
                "ONNX tensor name cannot be empty.",
                parameterName);
        }
    }
}
