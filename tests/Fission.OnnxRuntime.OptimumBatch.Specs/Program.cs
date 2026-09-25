using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static DecoderOrtState CreateState(
    int position,
    int nextToken,
    float keySeed,
    float valueSeed)
{
    var key = OrtValue.CreateTensorValueFromMemory(
        new[] { keySeed },
        new long[] { 1, 1, 1, 1 });
    var value = OrtValue.CreateTensorValueFromMemory(
        new[] { valueSeed },
        new long[] { 1, 1, 1, 1 });
    return new DecoderOrtState(
        position,
        new[] { new DecoderOrtLayerState(key, value) },
        nextTokenId: nextToken);
}

// Verify the exact ORT primitive used by cohort state splitting. Each OrtValue
// pins only its Memory<T> slice; it must observe backing-array mutations without a
// copy, and disposing one slice handle must not invalidate a sibling slice.
var sliceBacking = new[] { 1f, 2f, 3f, 4f };
using var rightSlice = OrtValue.CreateTensorValueFromMemory(
    OrtMemoryInfo.DefaultInstance,
    sliceBacking.AsMemory(2, 2),
    new long[] { 1, 1, 2, 1 });
using (var leftSlice = OrtValue.CreateTensorValueFromMemory(
    OrtMemoryInfo.DefaultInstance,
    sliceBacking.AsMemory(0, 2),
    new long[] { 1, 1, 2, 1 }))
{
    sliceBacking[1] = 20f;
    Require(
        leftSlice.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 1f, 20f }),
        "Memory-backed OrtValue slices must observe backing-array mutations without a copy.");
}
Require(
    rightSlice.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 3f, 4f }),
    "Disposing one Memory-backed OrtValue slice must not invalidate a sibling slice.");

// Batch=2 variant of the tiny decoder-shaped Optimum legacy fixture. The graph
// appends the current token to each row's KV and maps 0->1->2->3->0 through its
// logits table. Every graph input/output has a fixed batch dimension of two so a
// successful run proves the concrete binding is issuing one physical batched ORT
// invocation rather than looping over scalar sessions.
const string BatchTwoDecoderBase64 =
    "CAcSB0Zpc3Npb246hQgKRQoMbG9naXRzX3RhYmxlCglpbnB1dF9pZHMSBmxvZ2l0cxoNZ2F0aGVyX2xvZ2l0cyIG" +
    "R2F0aGVyKgsKBGF4aXMYAKABAgoyCglpbnB1dF9pZHMSCHRva2VuXzJkGgpjYXN0X3Rva2VuIgRDYXN0KgkKAnRv" +
    "GAGgAQIKOQoIdG9rZW5fMmQSCHRva2VuXzNkGgp1bnNxdWVlemUzIglVbnNxdWVlemUqDAoEYXhlc0IBAqABBwo5" +
    "Cgh0b2tlbl8zZBIIdG9rZW5fNGQaCnVuc3F1ZWV6ZTQiCVVuc3F1ZWV6ZSoMCgRheGVzQgEDoAEHClEKFXBhc3Rf" +
    "a2V5X3ZhbHVlcy4wLmtleQoIdG9rZW5fNGQSDXByZXNlbnQuMC5rZXkaCmFwcGVuZF9rZXkiBkNvbmNhdCoLCgRh" +
    "eGlzGAKgAQIKNgoIdG9rZW5fNGQKC3ZhbHVlX3NjYWxlEgt0b2tlbl92YWx1ZRoLc2NhbGVfdmFsdWUiA011bApa" +
    "ChdwYXN0X2tleV92YWx1ZXMuMC52YWx1ZQoLdG9rZW5fdmFsdWUSD3ByZXNlbnQuMC52YWx1ZRoMYXBwZW5kX3Zh" +
    "bHVlIgZDb25jYXQqCwoEYXhpcxgCoAECEhRmaXNzaW9uIHRpbnkgZGVjb2RlcipWCAQIBBABIkAAAAAAAAAgQQAA" +
    "AAAAAAAAAAAAAAAAAAAAACBBAAAAAAAAAAAAAAAAAAAAAAAAIEEAACBBAAAAAAAAAAAAAAAAQgxsb2dpdHNfdGFi" +
    "bGUqHQgBCAEIAQgBEAEiBAAAQEBCC3ZhbHVlX3NjYWxlWhsKCWlucHV0X2lkcxIOCgwIBxIICgIIAgoCCAFaNQoO" +
    "YXR0ZW50aW9uX21hc2sSIwohCAcSHQoCCAIKFxIVdG90YWxfc2VxdWVuY2VfbGVuZ3RoWh4KDHBvc2l0aW9uX2lk" +
    "cxIOCgwIBxIICgIIAgoCCAFaQwoVcGFzdF9rZXlfdmFsdWVzLjAua2V5EioKKAgBEiQKAggCCgIIAQoWEhRwYXN0" +
    "X3NlcXVlbmNlX2xlbmd0aAoCCAFaRQoXcGFzdF9rZXlfdmFsdWVzLjAudmFsdWUSKgooCAESJAoCCAIKAggBChYS" +
    "FHBhc3Rfc2VxdWVuY2VfbGVuZ3RoCgIIAWIcCgZsb2dpdHMSEgoQCAESDAoCCAIKAggBCgIIBGI+Cg1wcmVzZW50" +
    "LjAua2V5Ei0KKwgBEicKAggCCgIIAQoZEhdwcmVzZW50X3NlcXVlbmNlX2xlbmd0aAoCCAFiQAoPcHJlc2VudC4w" +
    "LnZhbHVlEi0KKwgBEicKAggCCgIIAQoZEhdwcmVzZW50X3NlcXVlbmNlX2xlbmd0aAoCCAFCBAoAEAs=";

var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 1,
    numKvHeads: 1,
    headDim: 1,
    vocabularySize: 4,
    kvElementType: TensorElementType.Float,
    logitsElementType: TensorElementType.Float);
using var binding = new OptimumLegacyFloatDecoderBinding(profile);
using var sessionOptions = new SessionOptions
{
    IntraOpNumThreads = 1,
    InterOpNumThreads = 1
};
using var session = new InferenceSession(
    Convert.FromBase64String(BatchTwoDecoderBase64),
    sessionOptions);
profile.SessionContract.Validate(session);

var modelId = new ModelId("tiny-optimum-batch2");
var firstId = SequenceId.New();
var secondId = SequenceId.New();
using var firstPrior = CreateState(
    position: 1,
    nextToken: 0,
    keySeed: 30f,
    valueSeed: 31f);
using var secondPrior = CreateState(
    position: 1,
    nextToken: 1,
    keySeed: 40f,
    valueSeed: 41f);

var decoded = binding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(firstId, modelId, Position: 1),
        new DecodeItem(secondId, modelId, Position: 1)
    },
    new[] { firstPrior, secondPrior });

Require(decoded.Count == 2, "A two-sequence cohort must return two decoder results.");
Require(binding.OrtRunCount == 1, "A same-position two-sequence cohort must execute exactly one ORT run.");
Require(decoded[0].TokenId == 1, "First sequence must map token frontier 0 -> 1.");
Require(decoded[1].TokenId == 2, "Second sequence must map token frontier 1 -> 2.");
Require(decoded[0].State.Position == 2 && decoded[1].State.Position == 2, "Both states must advance one decode position.");
Require(decoded[0].State.NextTokenId == 1 && decoded[1].State.NextTokenId == 2, "Split states must retain their own sampled token frontiers.");

var firstKey = decoded[0].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().ToArray();
var secondKey = decoded[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().ToArray();
Require(firstKey.SequenceEqual(new[] { 30f, 0f }), "First row must preserve its own past KV and append token 0.");
Require(secondKey.SequenceEqual(new[] { 40f, 1f }), "Second row must preserve its own past KV and append token 1.");

// Row states use independent OrtValue handles over non-overlapping slices of the
// same batched managed output arrays. Releasing one handle must not unpin or
// invalidate the sibling row.
decoded[0].State.Dispose();
Require(decoded[0].State.IsDisposed, "Disposed first split state must report terminal ownership.");
Require(!decoded[1].State.IsDisposed, "Disposing one split state must not dispose its sibling.");
Require(
    decoded[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 40f, 1f }),
    "Second split state must remain readable after the first state is disposed.");
decoded[1].State.Dispose();

Console.WriteLine(
    $"Fission zero-copy Optimum cohort specs passed: ortRuns={binding.OrtRunCount}, " +
    $"tokens={decoded[0].TokenId},{decoded[1].TokenId}, " +
    $"positions={decoded[0].State.Position},{decoded[1].State.Position}.");
