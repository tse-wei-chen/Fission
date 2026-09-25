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
Require(binding.ScalarPrefillCalls == 2, "Prefill remains on the scalar binding path in this milestone.");

var decoded = await backend.DecodeAsync(new DecodeBatch(new[]
{
    new DecodeItem(first, modelId, Position: 1),
    new DecodeItem(second, modelId, Position: 1)
}));

Require(binding.BatchDecodeCalls == 1, "One DecodeBatch must invoke the batch binding exactly once.");
Require(binding.ScalarDecodeCalls == 0, "Batch-capable bindings must bypass scalar decode execution.");
Require(binding.OrtRunCount == 1, "Two sequences must be evaluated by one real InferenceSession.Run call.");
Require(decoded[0].TokenId == 10, "First sequence must map through column 0 of the shared ORT run (2*5=10).");
Require(decoded[1].TokenId == 18, "Second sequence must map through column 1 of the shared ORT run (3*6=18).");

await backend.ReleaseSequenceAsync(first);
await backend.ReleaseSequenceAsync(second);

Console.WriteLine(
    $"Fission decoder batch specs passed: batchCalls={binding.BatchDecodeCalls}, " +
    $"scalarCalls={binding.ScalarDecodeCalls}, ortRuns={binding.OrtRunCount}, " +
    $"tokens={decoded[0].TokenId},{decoded[1].TokenId}.");

sealed class BatchedToyBinding : IDecoderOrtBatchModelBinding
{
    private int _disposed;

    public string Name => "batched-toy-mul";
    public int ScalarPrefillCalls { get; private set; }
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
        EnsureAlive();
        cancellationToken.ThrowIfCancellationRequested();
        ScalarPrefillCalls++;
        var token = item.Tokens.Span[^1];
        return new DecoderOrtStepResult(
            token,
            CreateState(item.Tokens.Length, token));
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

        var token0 = checked((int)MathF.Round(outputBuffer[4]));
        var token1 = checked((int)MathF.Round(outputBuffer[5]));
        return new[]
        {
            new DecoderOrtStepResult(token0, CreateState(items[0].Position + 1, token0)),
            new DecoderOrtStepResult(token1, CreateState(items[1].Position + 1, token1))
        };
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
