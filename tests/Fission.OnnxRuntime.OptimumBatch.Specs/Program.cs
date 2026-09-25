using System.Buffers;
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

// Simulate a fork: two logical sequences reference the same immutable prior
// state, so both requests carry the same arena row. Reusing the whole arena would
// incorrectly substitute the missing sibling row. The binding must instead pack
// both logical branches and establish a fresh two-row arena for their outputs.
var forked = binding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(SequenceId.New(), modelId, Position: 3),
        new DecodeItem(SequenceId.New(), modelId, Position: 3)
    },
    new[] { decodedAgain[1].State, decodedAgain[1].State });

Require(binding.OrtRunCount == 3, "Fork fallback must still execute one physical ORT run.");
Require(binding.PastKvPackCount == 2, "Duplicate arena rows must force a gather/pack fallback.");
Require(binding.PastKvArenaReuseCount == 1, "Duplicate arena rows must not count as an arena reuse.");
Require(binding.PastKvCopiedElementCount == 16, "Fork fallback should copy two three-token key/value rows exactly once.");
Require(forked[0].TokenId == 3 && forked[1].TokenId == 3, "Forked branches must preserve the shared token frontier independently.");
Require(!ReferenceEquals(forked[0].State, forked[1].State), "Fork fallback outputs must own distinct decoder state objects.");
Require(
    forked[0].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 30f, 0f, 1f, 2f }) &&
    forked[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 30f, 0f, 1f, 2f }),
    "Fork fallback must materialize both logical branches from the shared prior row.");

forked[0].State.Dispose();
Require(!forked[1].State.IsDisposed, "Fork fallback outputs must remain independently disposable.");
Require(
    forked[1].State.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 30f, 0f, 1f, 2f }),
    "Disposing one fork output must not invalidate its sibling branch.");
forked[1].State.Dispose();

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

// Verify pooled output ownership independently from the semantic cohort tests.
// The pool deliberately returns 64-element arrays for every tiny KV request so a
// frontier returned after step two can satisfy the larger logical tensor needed by
// step three. Exact Memory<T> slicing in the binding keeps ORT shapes independent
// from rented array capacity.
var bufferPool = new ReusingFloatPool(bufferLength: 64);
using var pooledBinding = new OptimumLegacyFloatDecoderBinding(
    profile,
    cohortBufferPool: bufferPool);
using var pooledFirstPrior = CreateState(1, 0, 50f, 51f);
using var pooledSecondPrior = CreateState(1, 1, 60f, 61f);
var pooledFirstId = SequenceId.New();
var pooledSecondId = SequenceId.New();

var pooledFirst = pooledBinding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(pooledFirstId, modelId, Position: 1),
        new DecodeItem(pooledSecondId, modelId, Position: 1)
    },
    new[] { pooledFirstPrior, pooledSecondPrior });
Require(bufferPool.RentCount == 2 && bufferPool.AllocationCount == 2, "First pooled frontier must rent two newly allocated key/value buffers.");
Require(bufferPool.ReturnCount == 0, "Live pooled row states must keep their arena buffers rented.");

var pooledSecond = pooledBinding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(pooledFirstId, modelId, Position: 2),
        new DecodeItem(pooledSecondId, modelId, Position: 2)
    },
    new[] { pooledFirst[0].State, pooledFirst[1].State });
Require(bufferPool.RentCount == 4 && bufferPool.AllocationCount == 4, "Second frontier must rent another pair while the first frontier is still live.");
Require(bufferPool.ReturnCount == 0, "Prior arena buffers cannot return while row states still own them.");

pooledFirst[0].State.Dispose();
Require(bufferPool.ReturnCount == 0, "One live sibling row must retain the complete pooled arena.");
pooledFirst[1].State.Dispose();
Require(bufferPool.ReturnCount == 2, "The final row release must return both layer buffers exactly once.");

var pooledThird = pooledBinding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(pooledFirstId, modelId, Position: 3),
        new DecodeItem(pooledSecondId, modelId, Position: 3)
    },
    new[] { pooledSecond[0].State, pooledSecond[1].State });
Require(bufferPool.RentCount == 6, "Three cohort frontiers must perform six key/value rents for one layer.");
Require(bufferPool.AllocationCount == 4, "Third frontier must reuse the two buffers returned by the first frontier.");
Require(bufferPool.ReuseCount == 2, "Third frontier must observe two physical buffer reuses.");
Require(pooledBinding.OrtRunCount == 3, "Pooled backing storage must not change physical ORT run count.");
Require(pooledThird[0].State.Position == 4 && pooledThird[1].State.Position == 4, "Pooled frontier states must advance normally.");

pooledSecond[0].State.Dispose();
pooledSecond[1].State.Dispose();
Require(bufferPool.ReturnCount == 4, "Releasing the second frontier must return its two buffers.");
pooledThird[0].State.Dispose();
Require(bufferPool.ReturnCount == 4, "One third-frontier sibling must keep reused buffers checked out.");
pooledThird[1].State.Dispose();
Require(bufferPool.ReturnCount == 6, "All three pooled frontiers must balance six rents with six returns.");

Console.WriteLine(
    $"Fission persistent Optimum cohort specs passed: ortRuns={binding.OrtRunCount}, " +
    $"packs={binding.PastKvPackCount}, reuses={binding.PastKvArenaReuseCount}, " +
    $"copiedElements={binding.PastKvCopiedElementCount}, " +
    $"poolAllocations={bufferPool.AllocationCount}, poolReuses={bufferPool.ReuseCount}.");

sealed class ReusingFloatPool : ArrayPool<float>
{
    private readonly object _gate = new();
    private readonly Stack<float[]> _available = new();
    private readonly HashSet<float[]> _availableSet = new(ReferenceEqualityComparer.Instance);
    private readonly int _bufferLength;

    public ReusingFloatPool(int bufferLength)
    {
        if (bufferLength < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(bufferLength));
        }

        _bufferLength = bufferLength;
    }

    public int RentCount { get; private set; }
    public int ReturnCount { get; private set; }
    public int AllocationCount { get; private set; }
    public int ReuseCount { get; private set; }

    public override float[] Rent(int minimumLength)
    {
        if (minimumLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumLength));
        }

        lock (_gate)
        {
            RentCount++;
            if (_available.Count > 0)
            {
                var reused = _available.Pop();
                _availableSet.Remove(reused);
                if (reused.Length >= minimumLength)
                {
                    ReuseCount++;
                    return reused;
                }
            }

            AllocationCount++;
            return new float[Math.Max(_bufferLength, minimumLength)];
        }
    }

    public override void Return(float[] array, bool clearArray = false)
    {
        ArgumentNullException.ThrowIfNull(array);
        lock (_gate)
        {
            if (!_availableSet.Add(array))
            {
                throw new InvalidOperationException("The same pooled buffer was returned more than once.");
            }

            if (clearArray)
            {
                Array.Clear(array);
            }

            ReturnCount++;
            _available.Push(array);
        }
    }
}
