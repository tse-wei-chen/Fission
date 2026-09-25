using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Fission.Runtime.Execution;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

// Tiny decoder-shaped ONNX graph, generated specifically for this executable
// spec. It exposes the Optimum legacy one-layer schema:
//   input_ids, attention_mask, position_ids, past_key_values.0.key/value
//   -> logits, present.0.key/value
// The graph appends the current input token to KV and uses a logits table that
// produces the deterministic token transition 0->1->2->3->0.
const string TinyDecoderBase64 =
    "CAcSB0Zpc3Npb246hQgKRQoMbG9naXRzX3RhYmxlCglpbnB1dF9pZHMSBmxvZ2l0cxoNZ2F0aGVyX2xv" +
    "Z2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAgoyCglpbnB1dF9pZHMSCHRva2VuXzJkGgpjYXN0X3Rva2Vu" +
    "IgRDYXN0KgkKAnRvGAGgAQIKOQoIdG9rZW5fMmQSCHRva2VuXzNkGgp1bnNxdWVlemUzIglVbnNxdWVl" +
    "emUqDAoEYXhlc0IBAqABBwo5Cgh0b2tlbl8zZBIIdG9rZW5fNGQaCnVuc3F1ZWV6ZTQiCVVuc3F1ZWV6" +
    "ZSoMCgRheGVzQgEDoAEHClEKFXBhc3Rfa2V5X3ZhbHVlcy4wLmtleQoIdG9rZW5fNGQSDXByZXNlbnQu" +
    "MC5rZXkaCmFwcGVuZF9rZXkiBkNvbmNhdCoLCgRheGlzGAKgAQIKNgoIdG9rZW5fNGQKC3ZhbHVlX3Nj" +
    "YWxlEgt0b2tlbl92YWx1ZRoLc2NhbGVfdmFsdWUiA011bApaChdwYXN0X2tleV92YWx1ZXMuMC52YWx1" +
    "ZQoLdG9rZW5fdmFsdWUSD3ByZXNlbnQuMC52YWx1ZRoMYXBwZW5kX3ZhbHVlIgZDb25jYXQqCwoEYXhp" +
    "cxgCoAECEhRmaXNzaW9uIHRpbnkgZGVjb2RlcipWCAQIBBABIkAAAAAAAAAgQQAAAAAAAAAAAAAAAAAA" +
    "AAAAACBBAAAAAAAAAAAAAAAAAAAAAAAAIEEAACBBAAAAAAAAAAAAAAAAQgxsb2dpdHNfdGFibGUqHQgB" +
    "CAEIAQgBEAEiBAAAQEBCC3ZhbHVlX3NjYWxlWhsKCWlucHV0X2lkcxIOCgwIBxIICgIIAQoCCAFaNQoO" +
    "YXR0ZW50aW9uX21hc2sSIwohCAcSHQoCCAEKFxIVdG90YWxfc2VxdWVuY2VfbGVuZ3RoWh4KDHBvc2l0" +
    "aW9uX2lkcxIOCgwIBxIICgIIAQoCCAFaQwoVcGFzdF9rZXlfdmFsdWVzLjAua2V5EioKKAgBEiQKAggB" +
    "CgIIAQoWEhRwYXN0X3NlcXVlbmNlX2xlbmd0aAoCCAFaRQoXcGFzdF9rZXlfdmFsdWVzLjAudmFsdWUS" +
    "KgooCAESJAoCCAEKAggBChYSFHBhc3Rfc2VxdWVuY2VfbGVuZ3RoCgIIAWIcCgZsb2dpdHMSEgoQCAES" +
    "DAoCCAEKAggBCgIIBGI+Cg1wcmVzZW50LjAua2V5Ei0KKwgBEicKAggBCgIIAQoZEhdwcmVzZW50X3Nl" +
    "cXVlbmNlX2xlbmd0aAoCCAFiQAoPcHJlc2VudC4wLnZhbHVlEi0KKwgBEicKAggBCgIIAQoZEhdwcmVz" +
    "ZW50X3NlcXVlbmNlX2xlbmd0aAoCCAFCBAoAEAs=";

var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 1,
    numKvHeads: 1,
    headDim: 1,
    vocabularySize: 4,
    kvElementType: TensorElementType.Float,
    logitsElementType: TensorElementType.Float);
var binding = new OptimumLegacyFloatDecoderBinding(
    profile,
    eosTokenIds: new[] { 3 });
var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);
var modelId = new ModelId("tiny-optimum-decoder");
var backend = new OnnxRuntimeBackend(
    new OnnxRuntimeBackendOptions(
        modelId,
        new DeviceId("cpu:optimum-spec"),
        OnnxRuntimeModelSource.FromBytes(
            Convert.FromBase64String(TinyDecoderBase64)),
        IntraOpNumThreads: 1,
        InterOpNumThreads: 1),
    adapter);

await using var executor = await ContinuousBatchExecutor.CreateAsync(
    backend,
    capacity: 32,
    maxBatchSize: 8);

Require(
    executor.BackendName == "onnxruntime/decoder/optimum-legacy-fp32-greedy",
    "Concrete Optimum decoder binding must surface through the generic backend.");

var parent = SequenceId.New();
var branch = SequenceId.New();
var snapshot = KvSnapshotId.New();

// Prompt token 3 maps to sampled token 0. Prefill uses zero-length past KV and
// creates present KV with sequence length 1.
var prefill = await executor.SubmitPrefillAsync(
    new PrefillItem(
        parent,
        modelId,
        new ReadOnlyMemory<int>(new[] { 3 })));
Require(prefill.TokenId == 0, "Tiny decoder logits table must map prompt token 3 to token 0.");
Require(!prefill.IsFinished, "Token 0 is not configured as EOS.");

await executor.SnapshotSequenceAsync(parent, snapshot);
await executor.ForkSequenceAsync(parent, new[] { branch });

// The first branch decode must consume state.NextTokenId=0, not derive a token
// from position. The fixture maps 0 -> 1.
var firstBranchDecode = await executor.SubmitDecodeAsync(
    new DecodeItem(branch, modelId, Position: 1));
Require(
    firstBranchDecode.TokenId == 1,
    "First decode must consume the sampled token frontier 0 and produce token 1.");

// The next immutable state carries NextTokenId=1, so the following step maps
// 1 -> 2 and grows present KV from length 2 to length 3.
var secondBranchDecode = await executor.SubmitDecodeAsync(
    new DecodeItem(branch, modelId, Position: 2));
Require(
    secondBranchDecode.TokenId == 2,
    "Second decode must consume the updated token frontier 1 and produce token 2.");

// Restore must rewind both KV and the sampled token frontier to the state directly
// after prefill. Decoding again from position 1 must therefore reproduce 0 -> 1.
await executor.RestoreSequenceAsync(branch, snapshot);
var restoredDecode = await executor.SubmitDecodeAsync(
    new DecodeItem(branch, modelId, Position: 1));
Require(
    restoredDecode.TokenId == 1,
    "Restore must rewind token frontier together with KV state.");

// Continue 1 -> 2 -> 3. Token 3 is configured as EOS and must flow through the
// generic BackendStepResult termination flag.
var afterRestoreSecond = await executor.SubmitDecodeAsync(
    new DecodeItem(branch, modelId, Position: 2));
Require(afterRestoreSecond.TokenId == 2, "Restored chain must advance 1 -> 2.");
Require(!afterRestoreSecond.IsFinished, "Token 2 is not EOS.");

var eosStep = await executor.SubmitDecodeAsync(
    new DecodeItem(branch, modelId, Position: 3));
Require(eosStep.TokenId == 3, "Restored chain must advance 2 -> 3.");
Require(eosStep.IsFinished, "Configured EOS token must mark the backend step finished.");

await executor.ReleaseSnapshotAsync(snapshot);
await executor.ReleaseSequenceAsync(parent);
await executor.ReleaseSequenceAsync(branch);

// Stateful chunked prefill must append prompt tokens to the existing immutable KV
// frontier. The second PrefillItem intentionally omits Position, matching the
// current runtime/engine call path; the backend must infer position 1 from state.
var chunked = SequenceId.New();
var firstChunk = await executor.SubmitPrefillAsync(
    new PrefillItem(
        chunked,
        modelId,
        new ReadOnlyMemory<int>(new[] { 3 })));
Require(firstChunk.TokenId == 0, "First prompt chunk must establish the 3 -> 0 frontier.");

var secondChunk = await executor.SubmitPrefillAsync(
    new PrefillItem(
        chunked,
        modelId,
        new ReadOnlyMemory<int>(new[] { 0, 1 })));
Require(
    secondChunk.TokenId == 2,
    "Second prompt chunk must append to prior KV and sample from its final prompt token 1.");
Require(
    !secondChunk.IsFinished,
    "Chunked prompt continuation ending at token 1 must not report EOS.");

var afterChunkedPrefill = await executor.SubmitDecodeAsync(
    new DecodeItem(chunked, modelId, Position: 3));
Require(
    afterChunkedPrefill.TokenId == 3 && afterChunkedPrefill.IsFinished,
    "Decode after chunked prefill must consume frontier 2 and produce configured EOS token 3.");
await executor.ReleaseSequenceAsync(chunked);

// Greedy sampler must inspect only the final sequence position, matching causal
// generation semantics for prefill outputs shaped [B, S, V].
var sampled = OptimumLegacyFloatDecoderBinding.GreedySampleLastPosition(
    new float[]
    {
        100f, 0f, 0f, 0f,
        0f, 1f, 9f, 2f
    },
    sequenceLength: 2,
    vocabularySize: 4);
Require(sampled == 2, "Greedy sampler must use logits from the final sequence position.");

Console.WriteLine(
    $"Fission Optimum decoder specs passed: prefill={prefill.TokenId}, " +
    $"branch={firstBranchDecode.TokenId}->{secondBranchDecode.TokenId}, " +
    $"restored={restoredDecode.TokenId}->{afterRestoreSecond.TokenId}->{eosStep.TokenId}, " +
    $"chunked={firstChunk.TokenId}->{secondChunk.TokenId}->{afterChunkedPrefill.TokenId}, " +
    $"eos={eosStep.IsFinished}.");
