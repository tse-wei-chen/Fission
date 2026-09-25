using Fission.Abstractions;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 2,
    numKvHeads: 8,
    headDim: 128,
    vocabularySize: 32_000,
    kvElementType: TensorElementType.Float16,
    logitsElementType: TensorElementType.Float);

var geometry = profile.Geometry;
Require(geometry.NumHiddenLayers == 2, "Geometry must retain the decoder layer count.");
Require(geometry.NumKvHeads == 8, "Geometry must retain GQA KV head count independently from attention heads.");
Require(geometry.HeadDim == 128, "Geometry must retain attention head dimension.");
Require(geometry.VocabularySize == 32_000, "Geometry must retain vocabulary size.");
Require(geometry.KvElementSizeBytes == 2, "Float16 KV elements must occupy two bytes.");

Require(
    geometry.GetInputIdsShape(1, 7).SequenceEqual(new long[] { 1, 7 }),
    "input_ids shape must be [batch, sequence].");
Require(
    geometry.GetPositionIdsShape(1, 7).SequenceEqual(new long[] { 1, 7 }),
    "position_ids shape must match input_ids.");
Require(
    geometry.GetAttentionMaskShape(1, 5, 7).SequenceEqual(new long[] { 1, 12 }),
    "attention_mask must cover past plus current sequence length.");
Require(
    geometry.GetPastKvShape(1, 5).SequenceEqual(new long[] { 1, 8, 5, 128 }),
    "Past KV must use [batch, kv_heads, past_sequence, head_dim].");
Require(
    geometry.GetPresentKvShape(1, 5, 7).SequenceEqual(new long[] { 1, 8, 12, 128 }),
    "Present KV sequence axis must equal past plus current sequence length.");
Require(
    geometry.GetLogitsShape(1, 7).SequenceEqual(new long[] { 1, 7, 32_000 }),
    "Logits must use [batch, sequence, vocabulary].");

Require(
    geometry.GetKvElementCountPerSequence(10) == 40_960,
    "Two-layer KV element count must include key and value tensors for every layer.");
Require(
    geometry.GetKvBytesPerSequence(10) == 81_920,
    "KV byte accounting must include Float16 element width.");

var modelId = new ModelId("geometry-model");
var memoryProfile = new DecoderOrtGeometryKvMemoryProfile(modelId, geometry);
Require(
    memoryProfile.GetKvBytesPerToken(modelId) == 8_192,
    "Geometry-backed memory profile must expose the full per-token KV byte cost.");

var wrongModelRejected = false;
try
{
    _ = memoryProfile.GetKvBytesPerToken(new ModelId("other-model"));
}
catch (InvalidOperationException)
{
    wrongModelRejected = true;
}
Require(wrongModelRejected, "Geometry-backed memory profile must reject a different model id.");

var sessionContract = profile.SessionContract;
Require(sessionContract.Inputs.Count == 7, "Two-layer profile must declare input_ids, mask, position_ids, and four past-KV inputs.");
Require(sessionContract.Outputs.Count == 5, "Two-layer profile must declare logits and four present-KV outputs.");
Require(
    sessionContract.Inputs.Select(static tensor => tensor.TensorName).SequenceEqual(new[]
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "past_key_values.1.key",
        "past_key_values.1.value"
    }),
    "Optimum legacy decoder input names must expand deterministically.");
Require(
    sessionContract.Outputs.Select(static tensor => tensor.TensorName).SequenceEqual(new[]
    {
        "logits",
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value"
    }),
    "Optimum legacy decoder output names must expand deterministically.");

var kvInputs = sessionContract.Inputs.Where(static tensor => tensor.LogicalName.StartsWith("past_", StringComparison.Ordinal)).ToArray();
var kvOutputs = sessionContract.Outputs.Where(static tensor => tensor.LogicalName.StartsWith("present_", StringComparison.Ordinal)).ToArray();
Require(
    kvInputs.Concat(kvOutputs).All(static tensor => tensor.Rank == 4),
    "Every legacy cache tensor must require rank 4.");
Require(
    kvInputs.Concat(kvOutputs).All(static tensor => tensor.ElementType == TensorElementType.Float16),
    "Every legacy cache tensor must require the configured KV element type.");
Require(
    sessionContract.Outputs[0].ElementType == TensorElementType.Float,
    "Logits contract must retain its independent element type.");
Require(
    sessionContract.Inputs[1].ElementType == TensorElementType.Int64 &&
    sessionContract.Inputs[2].ElementType == TensorElementType.Int64,
    "Optimum attention mask and position ids must be Int64 in this profile.");

var custom = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 1,
    numKvHeads: 4,
    headDim: 64,
    vocabularySize: 8_000,
    inputIds: "tokens",
    attentionMask: "mask",
    positionIds: "positions",
    logits: "scores",
    pastKeyNames: "cache.%d.k",
    pastValueNames: "cache.%d.v",
    presentKeyNames: "next.%d.k",
    presentValueNames: "next.%d.v");
Require(
    custom.SessionContract.Inputs.Select(static tensor => tensor.TensorName).SequenceEqual(new[]
    {
        "tokens", "mask", "positions", "cache.0.k", "cache.0.v"
    }),
    "Profile must preserve exporter-specific names instead of hard-coding one graph signature.");
Require(
    custom.SessionContract.Outputs.Select(static tensor => tensor.TensorName).SequenceEqual(new[]
    {
        "scores", "next.0.k", "next.0.v"
    }),
    "Custom present-KV names must remain configurable.");

var invalidElementTypeRejected = false;
try
{
    _ = new DecoderOrtGeometry(
        numHiddenLayers: 1,
        numKvHeads: 1,
        headDim: 64,
        vocabularySize: 1_000,
        kvElementType: TensorElementType.Int8);
}
catch (ArgumentOutOfRangeException)
{
    invalidElementTypeRejected = true;
}
Require(invalidElementTypeRejected, "Non-floating decoder KV element types must be rejected.");

var negativePastRejected = false;
try
{
    _ = geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: -1);
}
catch (ArgumentOutOfRangeException)
{
    negativePastRejected = true;
}
Require(negativePastRejected, "Negative past sequence length must be rejected.");

Console.WriteLine(
    $"Fission decoder geometry specs passed: kvShape=[{string.Join(',', geometry.GetPastKvShape(1, 5))}], " +
    $"kvBytesAt10={geometry.GetKvBytesPerSequence(10)}, kvBytesPerToken={memoryProfile.GetKvBytesPerToken(modelId)}, " +
    $"inputs={sessionContract.Inputs.Count}, outputs={sessionContract.Outputs.Count}.");
