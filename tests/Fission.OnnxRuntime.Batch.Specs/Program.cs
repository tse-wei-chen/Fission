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

const string MulModelBase64 =
    "CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0" +
    "KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4K" +
    "DAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==";

var modelId = new ModelId("batch-toy");
var binding = new BatchedToyBinding();
var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);
await using var backend = new OnnxRuntimeBackend(
    new OnnxRuntimeBackendOptions(
        modelId,
        new DeviceId("cpu:batch-spec"),
        OnnxRuntimeModelSource.FromBytes(Convert.FromBase64String(MulModelBase64))),
    adapter);

await backend.InitializeAsync();

var first = SequenceId.New();
var second = SequenceId.New();
var prefill = await backend.PrefillAsync(new PrefillBatch(new[]
{
    new PrefillItem(first, modelId, new ReadOnlyMemory<int>(new[] { 2 })),
    new PrefillItem(second, modelId, new ReadOnlyMemory<int>(new[] { 3 }))
}));
Require(prefill.Count == 2, "Two prefills must establish two decoder states.");
Require(binding.BatchPrefillCalls == 1, "One PrefillBatch must invoke the batch prefill binding exactly once.");
Require(binding.ScalarPrefillCalls == 0, "Batch-capable prefill bindings must bypass scalar prefill execution.");
Require(binding.OrtRunCount == 1, "Two prefill sequences must be evaluated by one real InferenceSession.Run call.");
Require(prefill[0].TokenId == 10, "First prefill must map through column 0 of the shared ORT run (2*5=10).");
Require(prefill[1].TokenId == 18, "Second prefill must map through column 1 of the shared ORT run (3*6=18).");

var decoded = await backend.DecodeAsync(new DecodeBatch(new[]
{
    new DecodeItem(first, modelId, Position: 1),
    new DecodeItem(second, modelId, Position: 1)
}));

Require(binding.BatchDecodeCalls == 1, "One DecodeBatch must invoke the batch decode binding exactly once.");
Require(binding.ScalarDecodeCalls == 0, "Batch-capable bindings must bypass scalar decode execution.");
Require(binding.OrtRunCount == 2, "Prefill plus decode must execute exactly two shared ORT runs.");
Require(decoded[0].TokenId == 50, "First decode must consume prefill frontier 10 and map 10*5=50.");
Require(decoded[1].TokenId == 108, "Second decode must consume prefill frontier 18 and map 18*6=108.");

await backend.ReleaseSequenceAsync(first);
await backend.ReleaseSequenceAsync(second);

Console.WriteLine(
    $"Fission decoder batch specs passed: prefillBatchCalls={binding.BatchPrefillCalls}, " +
    $"decodeBatchCalls={binding.BatchDecodeCalls}, scalarPrefill={binding.ScalarPrefillCalls}, " +
    $"scalarDecode={binding.ScalarDecodeCalls}, ortRuns={binding.OrtRunCount}, " +
    $"prefillTokens={prefill[0].TokenId},{prefill[1].TokenId}, " +
    $"decodeTokens={decoded[0].TokenId},{decoded[1].TokenId}.");

sealed class BatchedToyBinding :
    IDecoderOrtBatchModelBinding,
    IDecoderOrtBatchPrefillModelBinding
{
    private int _disposed;

    public string Name => "batched-toy-mul";
    public int ScalarPrefillCalls { get; private set; }
    public int BatchPrefillCalls { get; private set; }
    public int ScalarDecodeCalls { get; private set; }
    public int BatchDecodeCalls { get; private set; }
    public int OrtRunCount { get; private set; }

    public OnnxSessionContract SessionContract { get; } = new(
        Inputs: new[]
        {
            new OnnxTensorContract("input", "X", Rank: 2, ElementType: TensorElementType.Float)
        },
        Outputs: new[]
        {
            new OnnxTensorContract("output", "Y", Rank: 2, ElementType: TensorElementType.Float)
        });

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        ScalarPrefillCalls++;
        throw new InvalidOperationException(
            "Scalar prefill must not be called when the batch prefill capability is available.");
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecutePrefillBatch(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default)
    {
        EnsureAlive();
        cancellationToken.ThrowIfCancellationRequested();
        if (items.Count != 2 || priorStates.Count != 2)
        {
            throw new InvalidOperationException("Toy batch spec requires exactly two prefill items.");
        }

        if (priorStates.Any(static state => state is not null))
        {
            throw new InvalidOperationException("Toy batch prefill spec only exercises initial prefill.");
        }

        BatchPrefillCalls++;
        var firstToken = items[0].Tokens.Span[^1];
        var secondToken = items[1].Tokens.Span[^1];
        var output = ExecuteSharedRun(session, firstToken, secondToken, cancellationToken);
        var firstResult = checked((int)MathF.Round(output[4]));
        var secondResult = checked((int)MathF.Round(output[5]));

        return new[]
        {
            new DecoderOrtStepResult(
                firstResult,
                CreateState(checked(items[0].Position!.Value + items[0].Tokens.Length), firstResult)),
            new DecoderOrtStepResult(
                secondResult,
                CreateState(checked(items[1].Position!.Value + items[1].Tokens.Length), secondResult))
        };
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ScalarDecodeCalls++;
        throw new InvalidOperationException(
            "Scalar decode must not be called when the batch capability is available.");
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecuteDecodeBatch(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default)
    {
        EnsureAlive();
        cancellationToken.ThrowIfCancellationRequested();
        if (items.Count != 2 || priorStates.Count != 2)
        {
            throw new InvalidOperationException("Toy batch spec requires exactly two decode items.");
        }

        BatchDecodeCalls++;
        var firstToken = priorStates[0].NextTokenId ?? throw new InvalidOperationException("Missing first token frontier.");
        var secondToken = priorStates[1].NextTokenId ?? throw new InvalidOperationException("Missing second token frontier.");
        var output = ExecuteSharedRun(session, firstToken, secondToken, cancellationToken);
        var token0 = checked((int)MathF.Round(output[4]));
        var token1 = checked((int)MathF.Round(output[5]));

        return new[]
        {
            new DecoderOrtStepResult(token0, CreateState(items[0].Position + 1, token0)),
            new DecoderOrtStepResult(token1, CreateState(items[1].Position + 1, token1))
        };
    }

    private float[] ExecuteSharedRun(
        InferenceSession session,
        int firstToken,
        int secondToken,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // mul_1.onnx consumes a fixed [3,2] tensor. Treat each column as one
        // logical sequence so both sequences share one real ORT invocation.
        var inputBuffer = new float[]
        {
            firstToken, secondToken,
            firstToken, secondToken,
            firstToken, secondToken
        };
        var outputBuffer = new float[6];
        using var input = OrtValue.CreateTensorValueFromMemory(
            inputBuffer,
            new long[] { 3, 2 });
        using var output = OrtValue.CreateTensorValueFromMemory(
            outputBuffer,
            new long[] { 3, 2 });
        using var runOptions = new RunOptions();
        CallerOwnedOrtRun.Execute(
            session,
            runOptions,
            new[] { "X" },
            new[] { input },
            new[] { "Y" },
            new[] { output });
        OrtRunCount++;
        return outputBuffer;
    }

    private static DecoderOrtState CreateState(int position, int nextToken)
    {
        var key = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)nextToken },
            new long[] { 1, 1, 1, 1 });
        var value = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)nextToken },
            new long[] { 1, 1, 1, 1 });
        return new DecoderOrtState(
            position,
            new[] { new DecoderOrtLayerState(key, value) },
            nextTokenId: nextToken);
    }

    private void EnsureAlive() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
