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

var modelId = new ModelId("tiny-optimum-prefill-batch2");
var firstId = SequenceId.New();
var secondId = SequenceId.New();

var initial = binding.ExecutePrefillBatch(
    session,
    new[]
    {
        new PrefillItem(
            firstId,
            modelId,
            new ReadOnlyMemory<int>(new[] { 3 }),
            Position: 0),
        new PrefillItem(
            secondId,
            modelId,
            new ReadOnlyMemory<int>(new[] { 0 }),
            Position: 0)
    },
    new DecoderOrtState?[] { null, null });

Require(initial.Count == 2, "Two initial prefill rows must return two decoder states.");
Require(binding.OrtRunCount == 1, "Two equal-shape initial prefills must execute one ORT run.");
Require(binding.PastKvPackCount == 0, "Initial prefill must not pack empty past KV.");
Require(binding.PastKvArenaReuseCount == 0, "Initial prefill has no prior arena to reuse.");
Require(binding.PastKvCopiedElementCount == 0, "Initial prefill must not copy past KV elements.");
Require(initial[0].TokenId == 0, "Prompt token 3 must map to sampled frontier 0.");
Require(initial[1].TokenId == 1, "Prompt token 0 must map to sampled frontier 1.");
Require(initial[0].State.Position == 1 && initial[1].State.Position == 1, "Initial batch must advance both positions to one.");
Require(
    initial[0].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 3f }),
    "First prefill row must own its prompt token in KV.");
Require(
    initial[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 0f }),
    "Second prefill row must own its prompt token in KV.");

// Reverse logical request order on continuation. A complete two-row arena exists,
// so the binding must restore physical row order before the ORT run and map the
// results back to the caller's logical order without packing/copying prior KV.
var continued = binding.ExecutePrefillBatch(
    session,
    new[]
    {
        new PrefillItem(
            secondId,
            modelId,
            new ReadOnlyMemory<int>(new[] { 1 }),
            Position: 1),
        new PrefillItem(
            firstId,
            modelId,
            new ReadOnlyMemory<int>(new[] { 0 }),
            Position: 1)
    },
    new DecoderOrtState?[] { initial[1].State, initial[0].State });

Require(binding.OrtRunCount == 2, "Initial plus continuation prefill cohorts must use exactly two ORT runs.");
Require(binding.PastKvPackCount == 0, "Complete prefill arena continuation must stay on zero-copy reuse.");
Require(binding.PastKvArenaReuseCount == 1, "Continuation prefill must reuse the complete prior arena once.");
Require(binding.PastKvCopiedElementCount == 0, "Arena reuse must not copy prior KV elements.");
Require(continued[0].TokenId == 2, "Reversed second sequence continuation token 1 must map to frontier 2.");
Require(continued[1].TokenId == 1, "Reversed first sequence continuation token 0 must map to frontier 1.");
Require(continued[0].State.Position == 2 && continued[1].State.Position == 2, "Continuation batch must advance both positions to two.");
Require(
    continued[0].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 0f, 1f }),
    "Second sequence must keep its physical KV row after reversed logical ordering.");
Require(
    continued[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 3f, 0f }),
    "First sequence must keep its physical KV row after reversed logical ordering.");

initial[0].State.Dispose();
Require(
    continued[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 3f, 0f }),
    "Releasing one prior row must not invalidate the successor arena.");
initial[1].State.Dispose();
continued[0].State.Dispose();
continued[1].State.Dispose();

Console.WriteLine(
    $"Fission Optimum prefill cohort specs passed: ortRuns={binding.OrtRunCount}, " +
    $"packs={binding.PastKvPackCount}, reuses={binding.PastKvArenaReuseCount}, " +
    $"copiedElements={binding.PastKvCopiedElementCount}.");
