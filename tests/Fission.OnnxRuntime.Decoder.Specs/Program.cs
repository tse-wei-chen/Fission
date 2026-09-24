using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Fission.Runtime.Execution;
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

var modelBytes = Convert.FromBase64String(MulModelBase64);
var modelId = new ModelId("toy-decoder");
var device = new DeviceId("cpu:decoder-spec");

// Adapter-provided graph contracts must be validated by the generic backend
// before the model binding initializes any state.
var rejectedBinding = new ToyDecoderBinding(brokenContract: true);
var rejectedAdapter = new DecoderOnlyOnnxExecutionAdapter(rejectedBinding);
var rejectedBackend = new OnnxRuntimeBackend(
    new OnnxRuntimeBackendOptions(
        modelId,
        device,
        OnnxRuntimeModelSource.FromBytes(modelBytes)),
    rejectedAdapter);
var rejected = false;
try
{
    _ = await ContinuousBatchExecutor.CreateAsync(rejectedBackend);
}
catch (InvalidOperationException exception)
    when (exception.Message.Contains("missing_X", StringComparison.Ordinal))
{
    rejected = true;
}
finally
{
    await rejectedBackend.DisposeAsync();
}

Require(rejected, "Adapter-provided session contract must reject a mismatched live graph.");
Require(!rejectedBinding.EverInitialized, "Binding initialization must not run after contract validation fails.");

var binding = new ToyDecoderBinding();
var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);
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
    capacity: 32,
    maxBatchSize: 8);

Require(binding.Initialized, "Decoder model binding must initialize after its live graph contract passes.");
Require(executor.BackendName == "onnxruntime/decoder/toy-mul", "Backend name must identify the decoder binding.");

var parent = SequenceId.New();
var branch = SequenceId.New();
var snapshot = KvSnapshotId.New();

var prefill = await executor.SubmitPrefillAsync(
    new PrefillItem(
        parent,
        modelId,
        new ReadOnlyMemory<int>(new[] { 1, 2, 3, 4, 5, 6 })));
Require(prefill.TokenId == 36, "Toy decoder prefill must execute the live ONNX graph.");
Require(binding.PrefillStates.Count == 1, "Prefill must publish one immutable decoder state version.");
var sharedState = binding.PrefillStates[0];
Require(sharedState.Position == 6, "Prefill state position must equal the prompt token count.");

await executor.SnapshotSequenceAsync(parent, snapshot);
await executor.ForkSequenceAsync(parent, new[] { branch });

var branchDecode = await executor.SubmitDecodeAsync(
    new DecodeItem(branch, modelId, Position: 6));
Require(branchDecode.TokenId == 42, "Branch decode must execute from restored position 6 and produce token 42.");
Require(binding.DecodeStates.Count == 1, "Branch decode must publish a new immutable state version.");
var divergentState = binding.DecodeStates[0];
Require(divergentState.Position == 7, "Decode must advance physical decoder state by one position.");
Require(!sharedState.IsDisposed, "Parent/snapshot owners must keep the shared prefill state alive after branch divergence.");
Require(!divergentState.IsDisposed, "Divergent branch state must remain alive while the branch owns it.");

await executor.RestoreSequenceAsync(branch, snapshot);
Require(divergentState.IsDisposed, "Restoring the branch must release its unshared divergent state.");
Require(!sharedState.IsDisposed, "Restore must repoint the branch to the shared snapshot state without disposing it.");

await executor.ReleaseSnapshotAsync(snapshot);
await executor.ReleaseSequenceAsync(parent);
Require(!sharedState.IsDisposed, "Restored branch must remain the final owner of the shared prefill state.");

var resumedDecode = await executor.SubmitDecodeAsync(
    new DecodeItem(branch, modelId, Position: 6));
Require(resumedDecode.TokenId == 42, "Restored branch must decode deterministically from the snapshot position.");
Require(binding.DecodeStates.Count == 2, "Resumed decode must publish a second immutable state version.");
var resumedState = binding.DecodeStates[1];
Require(sharedState.IsDisposed, "Replacing the final shared owner must dispose the old prefill state.");
Require(!resumedState.IsDisposed, "New resumed state must remain alive until sequence release.");

await executor.ReleaseSequenceAsync(branch);
Require(resumedState.IsDisposed, "Final sequence release must dispose the latest physical decoder state.");
Require(binding.RunCount == 6, "Prefill plus two decode steps must each run key/value outputs through ORT.");

Console.WriteLine(
    $"Fission decoder adapter specs passed: backend={executor.BackendName}, " +
    $"prefill={prefill.TokenId}, branchDecode={branchDecode.TokenId}, resumed={resumedDecode.TokenId}, " +
    $"runs={binding.RunCount}, sharedDisposed={sharedState.IsDisposed}, resumedDisposed={resumedState.IsDisposed}.");

sealed class ToyDecoderBinding : IDecoderOrtModelBinding
{
    private readonly bool _brokenContract;
    private int _disposed;

    public ToyDecoderBinding(bool brokenContract = false)
    {
        _brokenContract = brokenContract;
        SessionContract = new OnnxSessionContract(
            Inputs: new[]
            {
                new OnnxTensorContract(
                    "decoder_input",
                    brokenContract ? "missing_X" : "X",
                    Rank: 2,
                    ElementType: TensorElementType.Float)
            },
            Outputs: new[]
            {
                new OnnxTensorContract(
                    "decoder_output",
                    "Y",
                    Rank: 2,
                    ElementType: TensorElementType.Float)
            });
    }

    public string Name => "toy-mul";
    public OnnxSessionContract SessionContract { get; }
    public bool Initialized { get; private set; }
    public bool EverInitialized { get; private set; }
    public int RunCount { get; private set; }
    public List<DecoderOrtState> PrefillStates { get; } = new();
    public List<DecoderOrtState> DecodeStates { get; } = new();

    public ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        if (_brokenContract)
        {
            throw new InvalidOperationException(
                "Broken toy binding must never initialize because its session contract should fail first.");
        }

        Initialized = true;
        EverInitialized = true;
        return ValueTask.CompletedTask;
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (item.Tokens.Length != 6)
        {
            throw new InvalidOperationException(
                "Toy decoder prefill requires exactly six prompt tokens.");
        }

        var values = Array.ConvertAll(
            item.Tokens.Span.ToArray(),
            static value => (float)value);
        var step = ExecuteStep(
            session,
            values,
            nextPosition: item.Tokens.Length,
            cancellationToken);
        PrefillStates.Add(step.State);
        return step;
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(priorState);
        if (priorState.IsDisposed)
        {
            throw new InvalidOperationException(
                "Toy decoder received a disposed prior state.");
        }

        if (priorState.Position != item.Position)
        {
            throw new InvalidOperationException(
                $"Toy decoder prior state position {priorState.Position} does not match decode position {item.Position}.");
        }

        // Touch the prior physical payload so this spec proves the adapter only
        // supplies live state versions to model bindings.
        _ = priorState.GetLayer(0).Key.GetTensorDataAsSpan<float>()[^1];

        var nextValue = checked((float)(item.Position + 1));
        var values = new[]
        {
            nextValue,
            nextValue,
            nextValue,
            nextValue,
            nextValue,
            nextValue
        };
        var step = ExecuteStep(
            session,
            values,
            nextPosition: checked(item.Position + 1),
            cancellationToken);
        DecodeStates.Add(step.State);
        return step;
    }

    private DecoderOrtStepResult ExecuteStep(
        InferenceSession session,
        float[] values,
        int nextPosition,
        CancellationToken cancellationToken)
    {
        var keyBuffer = new float[6];
        var valueBuffer = new float[6];
        var valueInputBuffer = values.Select(static value => value + 1f).ToArray();

        var keyOutput = OrtValue.CreateTensorValueFromMemory(
            keyBuffer,
            new long[] { 3, 2 });
        var valueOutput = OrtValue.CreateTensorValueFromMemory(
            valueBuffer,
            new long[] { 3, 2 });

        try
        {
            using var runOptions = new RunOptions();
            using var keyInput = OrtValue.CreateTensorValueFromMemory(
                values,
                new long[] { 3, 2 });
            using var valueInput = OrtValue.CreateTensorValueFromMemory(
                valueInputBuffer,
                new long[] { 3, 2 });

            cancellationToken.ThrowIfCancellationRequested();
            CallerOwnedOrtRun.Execute(
                session,
                runOptions,
                new[] { "X" },
                new[] { keyInput },
                new[] { "Y" },
                new[] { keyOutput });
            RunCount++;

            cancellationToken.ThrowIfCancellationRequested();
            CallerOwnedOrtRun.Execute(
                session,
                runOptions,
                new[] { "X" },
                new[] { valueInput },
                new[] { "Y" },
                new[] { valueOutput });
            RunCount++;

            var tokenId = checked((int)MathF.Round(keyBuffer[^1]));
            var state = new DecoderOrtState(
                nextPosition,
                new[] { new DecoderOrtLayerState(keyOutput, valueOutput) });
            return new DecoderOrtStepResult(tokenId, state);
        }
        catch
        {
            keyOutput.Dispose();
            valueOutput.Dispose();
            throw;
        }
    }

    private void EnsureInitialized()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!Initialized)
        {
            throw new InvalidOperationException(
                "Toy decoder binding has not been initialized.");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        Initialized = false;
    }
}
