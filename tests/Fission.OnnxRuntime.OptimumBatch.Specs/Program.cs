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
using var firstPrior = CreateState(1, 0, 30f, 31f);
using var secondPrior = CreateState(1, 1, 40f, 41f);

var decoded = binding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(firstId, modelId, Position: 1),
        new DecodeItem(secondId, modelId, Position: 1)
    },
    new[] { firstPrior, secondPrior });

Require(decoded.Count == 2, "A two-sequence cohort must return two decoder results.");
Require(binding.OrtRunCount == 1, "A two-sequence cohort must execute one ORT run.");
Require(binding.PastKvPackCount == 1, "External row states must be packed once to establish a cohort arena.");
Require(binding.PastKvArenaReuseCount == 0, "The first external cohort has no reusable arena yet.");
Require(binding.PastKvCopiedElementCount == 4, "The first 2x1x1x1 key/value frontier should copy four FP32 elements.");
Require(decoded[0].TokenId == 1 && decoded[1].TokenId == 2, "First decode tokens must preserve row identity.");
Require(decoded[0].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 30f, 0f }), "First row KV is incorrect.");
Require(decoded[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 40f, 1f }), "Second row KV is incorrect.");

var decodedAgain = binding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(secondId, modelId, Position: 2),
        new DecodeItem(firstId, modelId, Position: 2)
    },
    new[] { decoded[1].State, decoded[0].State });

Require(binding.OrtRunCount == 2, "Two cohort steps must execute exactly two ORT runs.");
Require(binding.PastKvPackCount == 1, "A complete stable arena must not trigger a second past-KV pack.");
Require(binding.PastKvArenaReuseCount == 1, "The second cohort step must reuse the complete prior arena.");
Require(binding.PastKvCopiedElementCount == 4, "Arena reuse must not copy additional past-KV elements.");
Require(decodedAgain[0].TokenId == 3, "Reversed request row for the second sequence must map token 2 -> 3.");
Require(decodedAgain[1].TokenId == 2, "Reversed request row for the first sequence must map token 1 -> 2.");
Require(
    decodedAgain[0].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 40f, 1f, 2f }),
    "Second sequence must retain stable physical row history across arena reuse.");
Require(
    decodedAgain[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 30f, 0f, 1f }),
    "First sequence must retain stable physical row history across arena reuse.");

decoded[0].State.Dispose();
decoded[1].State.Dispose();
Require(
    decodedAgain[0].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 40f, 1f, 2f }),
    "Releasing the prior arena must not invalidate the next frontier.");

decodedAgain[0].State.Dispose();
Require(!decodedAgain[1].State.IsDisposed, "Sibling row handles in the new arena must remain independently disposable.");
Require(
    decodedAgain[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 30f, 0f, 1f }),
    "Disposing one new-arena row must not invalidate its sibling.");
decodedAgain[1].State.Dispose();

Console.WriteLine(
    $"Fission persistent Optimum cohort specs passed: ortRuns={binding.OrtRunCount}, " +
    $"packs={binding.PastKvPackCount}, reuses={binding.PastKvArenaReuseCount}, " +
    $"copiedElements={binding.PastKvCopiedElementCount}.");
