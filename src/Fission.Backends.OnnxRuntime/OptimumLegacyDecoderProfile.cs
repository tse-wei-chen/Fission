using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Concrete profile for Optimum decoder-with-past exports that use the legacy
/// per-layer 4-D KV cache convention.
///
/// This profile intentionally targets the separate decoder-with-past graph, not
/// Optimum's merged decoder graph with a use_cache_branch selector.
/// </summary>
public sealed record OptimumLegacyDecoderProfile(
    DecoderOrtGeometry Geometry,
    DecoderOnlyOnnxContract Contract)
{
    public static OptimumLegacyDecoderProfile CreateLlamaLike(
        int numHiddenLayers,
        int numKvHeads,
        int headDim,
        int vocabularySize,
        TensorElementType kvElementType = TensorElementType.Float,
        TensorElementType logitsElementType = TensorElementType.Float,
        string inputIds = "input_ids",
        string attentionMask = "attention_mask",
        string positionIds = "position_ids",
        string logits = "logits",
        string pastKeyNames = "past_key_values.{0}.key",
        string pastValueNames = "past_key_values.{0}.value",
        string presentKeyNames = "present.{0}.key",
        string presentValueNames = "present.{0}.value")
    {
        var geometry = new DecoderOrtGeometry(
            numHiddenLayers,
            numKvHeads,
            headDim,
            vocabularySize,
            kvElementType);

        var contract = new DecoderOnlyOnnxContract(
            NumHiddenLayers: numHiddenLayers,
            InputIds: inputIds,
            Logits: logits,
            AttentionMask: attentionMask,
            PositionIds: positionIds,
            PastKeyNames: pastKeyNames,
            PastValueNames: pastValueNames,
            PresentKeyNames: presentKeyNames,
            PresentValueNames: presentValueNames,
            InputIdsRank: 2,
            LogitsRank: 3,
            InputIdsElementType: TensorElementType.Int64,
            AttentionMaskElementType: TensorElementType.Int64,
            PositionIdsElementType: TensorElementType.Int64,
            KvRank: 4,
            KvElementType: kvElementType,
            LogitsElementType: logitsElementType);

        return new OptimumLegacyDecoderProfile(geometry, contract);
    }

    public OnnxSessionContract SessionContract => Contract.ToSessionContract();
}
