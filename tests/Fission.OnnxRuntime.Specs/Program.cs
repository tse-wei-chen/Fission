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
var modelId = new ModelId("ort-mul-spec");
var device = new DeviceId("cpu:ort");
var mulContract = new OnnxSessionContract(
    Inputs: new[]
    {
        new OnnxTensorContract("multiplicand", "X", Rank: 2, ElementType: TensorElementType.Float)
    },
    Outputs: new[]
    {
        new OnnxTensorContract("product", "Y", Rank: 2, ElementType: TensorElementType.Float)
    });

// A live graph mismatch must fail before adapter initialization, otherwise the
// adapter could allocate model state against the wrong export signature.
var rejectedAdapter = new MulSpecAdapter();
var rejectedBackend = new OnnxRuntimeBackend(
    new OnnxRuntimeBackendOptions(
        modelId,
        device,
        OnnxRuntimeModelSource.FromBytes(modelBytes),
        IntraOpNumThreads: 1,
        InterOpNumThreads: 1,
        SessionContract: new OnnxSessionContract(
            Inputs: new[] { new OnnxTensorContract("multiplicand", "missing_input") },
            Outputs: new[] { new OnnxTensorContract("product", "Y") })),
    rejectedAdapter);

var contractRejected = false;
try
{
    await rejectedBackend.InitializeAsync();
}
catch (InvalidOperationException exception)
    when (exception.Message.Contains("missing_input", StringComparison.Ordinal))
{
    contractRejected = true;
}
finally
{
    await rejectedBackend.DisposeAsync();
}

Require(contractRejected, "A mismatched ONNX session contract must fail backend initialization.");
Require(!rejectedAdapter.EverInitialized, "Session contract validation must happen before adapter initialization.");

// Decoder-only manifests stay export-specific while expanding to a stable
// logical contract used by the model adapter.
var decoderManifest = new DecoderOnlyOnnxContract(
    NumHiddenLayers: 2,
    InputIds: "input_ids",
    Logits: "logits",
    AttentionMask: "attention_mask",
    PositionIds: "position_ids",
    PastKeyNames: "past_key_values.%d.key",
    PastValueNames: "past_key_values.%d.value",
    PresentKeyNames: "present.{0}.key",
    PresentValueNames: "present.{0}.value");
var decoderSessionContract = decoderManifest.ToSessionContract();

Require(decoderManifest.UsesPastKeyValues, "Decoder manifest must detect configured KV cache tensors.");
Require(
    decoderSessionContract.Inputs.Select(static input => input.TensorName).SequenceEqual(new[]
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "past_key_values.1.key",
        "past_key_values.1.value"
    }),
    "Decoder manifest must expand layer-indexed past KV input names deterministically.");
Require(
    decoderSessionContract.Outputs.Select(static output => output.TensorName).SequenceEqual(new[]
    {
        "logits",
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value"
    }),
    "Decoder manifest must expand layer-indexed present KV output names deterministically.");
Require(
    DecoderOnlyOnnxContract.ExpandLayerName("cache.%d.key", 7) == "cache.7.key" &&
    DecoderOnlyOnnxContract.ExpandLayerName("cache.{0}.value", 7) == "cache.7.value",
    "Decoder layer name patterns must support both %d and {0} conventions.");

var incompleteCacheRejected = false;
try
{
    _ = new DecoderOnlyOnnxContract(
        NumHiddenLayers: 2,
        InputIds: "input_ids",
        Logits: "logits",
        PastKeyNames: "past.%d.key")
        .ToSessionContract();
}
catch (InvalidOperationException)
{
    incompleteCacheRejected = true;
}
Require(incompleteCacheRejected, "Partial past/present cache manifests must be rejected.");

// Caller-owned/preallocated output proof. The same output OrtValue is reused
// across runs and remains valid after RunOptions disposal because ORT only writes
// into the caller-supplied handle; it does not return/own an output collection.
using (var sessionOptions = new SessionOptions())
using (var directSession = new InferenceSession(modelBytes, sessionOptions))
using (var preallocatedInput = new OwnedOrtTensor<float>(
    new[] { 1f, 2f, 3f, 4f, 5f, 6f },
    new long[] { 3, 2 }))
using (var preallocatedOutput = new OwnedOrtTensor<float>(new long[] { 3, 2 }))
{
    using (var runOptions = new RunOptions())
    {
        directSession.Run(
            runOptions,
            new[] { "X" },
            new[] { preallocatedInput.Value },
            new[] { "Y" },
            new[] { preallocatedOutput.Value });
    }

    Require(preallocatedOutput.ReadOnlySpan[^1] == 36f, "Preallocated ORT output buffer must receive the first graph result (6*6=36).");
    Require(!preallocatedOutput.IsDisposed, "Run/RunOptions disposal must not transfer or release caller-owned output OrtValue ownership.");
    Require(preallocatedOutput.Value.GetTensorDataAsSpan<float>()[^1] == 36f, "Caller-owned output OrtValue must remain readable after the run returns.");

    preallocatedInput.Span.CopyTo(new float[0]);
    var secondInput = preallocatedInput.Span;
    secondInput[0] = 2f;
    secondInput[1] = 3f;
    secondInput[2] = 4f;
    secondInput[3] = 5f;
    secondInput[4] = 6f;
    secondInput[5] = 7f;

    using (var runOptions = new RunOptions())
    {
        directSession.Run(
            runOptions,
            new[] { "X" },
            new[] { preallocatedInput.Value },
            new[] { "Y" },
            new[] { preallocatedOutput.Value });
    }

    Require(preallocatedOutput.ReadOnlySpan[^1] == 42f, "The same preallocated output buffer must be reusable across model steps (7*6=42).");
}

var invalidShapeRejected = false;
try
{
    using var invalidTensor = new OwnedOrtTensor<float>(new float[5], new long[] { 3, 2 });
}
catch (ArgumentException)
{
    invalidShapeRejected = true;
}
Require(invalidShapeRejected, "OwnedOrtTensor must reject buffer/shape element-count mismatches before creating an OrtValue.");

var adapter = new MulSpecAdapter();
var backend = new OnnxRuntimeBackend(
    new OnnxRuntimeBackendOptions(
        modelId,
        device,
        OnnxRuntimeModelSource.FromBytes(modelBytes),
        IntraOpNumThreads: 1,
        InterOpNumThreads: 1,
        SessionContract: mulContract),
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
    $"decode={decode.TokenId}, runs={adapter.RunCount}, releases={adapter.ReleaseCount}, " +
    $"decoderInputs={decoderSessionContract.Inputs.Count}, decoderOutputs={decoderSessionContract.Outputs.Count}.");

sealed class MulSpecAdapter : IOnnxRuntimeExecutionAdapter
{
    private readonly HashSet<SequenceId> _activeSequences = new();

    public string Name => "mul-spec";
    public bool Initialized { get; private set; }
    public bool EverInitialized { get; private set; }
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
        EverInitialized = true;
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
