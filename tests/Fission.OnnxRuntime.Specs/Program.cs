using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Fission.Runtime.Execution;
using Microsoft.ML.OnnxRuntime;

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

var modelBytes = Convert.FromBase64String(MulModelBase64);
var modelId = new ModelId("ort-mul-spec");
var device = new DeviceId("cpu:ort");
var adapter = new MulSpecAdapter();
var backend = new OnnxRuntimeBackend(
    new OnnxRuntimeBackendOptions(
        modelId,
        device,
        OnnxRuntimeModelSource.FromBytes(modelBytes),
        IntraOpNumThreads: 1,
        InterOpNumThreads: 1),
    adapter);

await using var executor = await ContinuousBatchExecutor.CreateAsync(
    backend,
    capacity: 16,
    maxBatchSize: 4);

Require(backend.IsInitialized, "ContinuousBatchExecutor initialization must create the ORT session.");
Require(backend.Name == "onnxruntime/mul-spec", "Backend name must include the model adapter identity.");
Require(adapter.Initialized, "Adapter must validate the live InferenceSession during backend initialization.");

var sequence = SequenceId.New();
var prefill = await executor.SubmitPrefillAsync(
    new PrefillItem(
        sequence,
        modelId,
        new ReadOnlyMemory<int>(new[] { 1, 2, 3, 4, 5, 6 })));

Require(prefill.SequenceId == sequence, "ORT prefill result must preserve sequence identity.");
Require(prefill.TokenId == 36, "mul_1.onnx must execute real tensor multiplication (last output 6*6=36).");
Require(adapter.ActiveSequenceCount == 1, "Model adapter must retain per-sequence state after prefill.");

var decode = await executor.SubmitDecodeAsync(
    new DecodeItem(sequence, modelId, Position: 6));
Require(decode.TokenId == 42, "ORT decode adapter must run the session again (position+1=7; last output 7*6=42).");
Require(adapter.RunCount == 2, "Prefill and decode must each execute the ONNX Runtime session.");

await executor.ReleaseSequenceAsync(sequence);
Require(adapter.ActiveSequenceCount == 0, "Device-actor release must remove adapter-owned sequence state.");
Require(adapter.ReleaseCount == 1, "Backend release hook must run exactly once.");

Console.WriteLine(
    $"Fission ONNX Runtime specs passed: backend={backend.Name}, prefill={prefill.TokenId}, " +
    $"decode={decode.TokenId}, runs={adapter.RunCount}, releases={adapter.ReleaseCount}.");

sealed class MulSpecAdapter : IOnnxRuntimeExecutionAdapter
{
    private readonly HashSet<SequenceId> _activeSequences = new();

    public string Name => "mul-spec";
    public bool Initialized { get; private set; }
    public int ActiveSequenceCount => _activeSequences.Count;
    public int RunCount { get; private set; }
    public int ReleaseCount { get; private set; }

    public ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (!session.InputNames.SequenceEqual(new[] { "X" }) ||
            !session.OutputNames.SequenceEqual(new[] { "Y" }))
        {
            throw new InvalidOperationException(
                $"Unexpected mul_1 model signature: inputs=[{string.Join(',', session.InputNames)}], " +
                $"outputs=[{string.Join(',', session.OutputNames)}].");
        }

        Initialized = true;
        return ValueTask.CompletedTask;
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        InferenceSession session,
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        var results = new BackendStepResult[batch.Items.Count];

        for (var index = 0; index < batch.Items.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var item = batch.Items[index];
            if (item.Tokens.Length != 6)
            {
                throw new InvalidOperationException("mul_1 spec adapter requires exactly six prompt tokens.");
            }

            var input = item.Tokens.Span.ToArray();
            var tensor = Array.ConvertAll(input, static value => (float)value);
            var token = Run(session, tensor);
            _activeSequences.Add(item.SequenceId);
            results[index] = new BackendStepResult(item.SequenceId, token);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        InferenceSession session,
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        var results = new BackendStepResult[batch.Items.Count];

        for (var index = 0; index < batch.Items.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var item = batch.Items[index];
            if (!_activeSequences.Contains(item.SequenceId))
            {
                throw new InvalidOperationException(
                    $"Decode reached ORT adapter without sequence state for {item.SequenceId}.");
            }

            var value = checked((float)(item.Position + 1));
            var tensor = new[] { value, value, value, value, value, value };
            results[index] = new BackendStepResult(item.SequenceId, Run(session, tensor));
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();

        if (!_activeSequences.Remove(sequenceId))
        {
            throw new InvalidOperationException(
                $"Release reached ORT adapter without sequence state for {sequenceId}.");
        }

        ReleaseCount++;
        return ValueTask.CompletedTask;
    }

    public void Dispose()
    {
        _activeSequences.Clear();
        Initialized = false;
    }

    private int Run(InferenceSession session, float[] values)
    {
        using var input = OrtValue.CreateTensorValueFromMemory(
            values,
            new long[] { 3, 2 });
        using var runOptions = new RunOptions();
        var inputs = new Dictionary<string, OrtValue>
        {
            ["X"] = input
        };

        using var outputs = session.Run(
            runOptions,
            inputs,
            new[] { "Y" });

        var output = outputs[0].GetTensorDataAsSpan<float>();
        RunCount++;
        return checked((int)MathF.Round(output[^1]));
    }

    private void EnsureInitialized()
    {
        if (!Initialized)
        {
            throw new InvalidOperationException("mul_1 ORT adapter has not been initialized.");
        }
    }
}
