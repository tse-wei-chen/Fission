using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static ExecutionBindings EmptyBindings() =>
    new(new Dictionary<SequenceId, ReadOnlyMemory<int>>());

static async Task PrefillAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId,
    ModelId modelId)
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new PrefillExecutionStep(sequenceId, modelId, 6, CompletesPrefill: true)
            }),
        new ExecutionBindings(
            new Dictionary<SequenceId, ReadOnlyMemory<int>>
            {
                [sequenceId] = new ReadOnlyMemory<int>(new[] { 1, 2, 3, 4, 5, 6 })
            }));
}

static async Task MigrateAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId,
    DeviceId targetDevice)
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new MigrateKvExecutionStep(sequenceId, targetDevice)
            }),
        EmptyBindings());
}

static async Task<BackendStepResult> DecodeOneAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId)
{
    var result = await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new DecodeExecutionStep(sequenceId, 1)
            }),
        EmptyBindings());
    return result.BackendResults.Single();
}

static SequenceProcess GetSequence(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId)
{
    if (!runtime.TryGetSequence(sequenceId, out var sequence) || sequence is null)
    {
        throw new InvalidOperationException($"Sequence {sequenceId} is missing.");
    }

    return sequence;
}

const string MulModelBase64 =
    "CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0" +
    "KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4K" +
    "DAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==";

var modelBytes = Convert.FromBase64String(MulModelBase64);
await RunEndToEndHostStagingAsync(modelBytes);
await RunIncompatibleCodecAsync(modelBytes);
RunOptimumCodecRoundTrip();

Console.WriteLine("Fission ONNX Runtime host-staging migration specs passed.");

static async Task RunEndToEndHostStagingAsync(byte[] modelBytes)
{
    var modelId = new ModelId("toy-host-staging");
    var sourceId = new DeviceId("cpu:onnx-source");
    var targetId = new DeviceId("cpu:onnx-target");
    var sourceBinding = new ToyHostStagingDecoderBinding("toy-fp32-v1");
    var targetBinding = new ToyHostStagingDecoderBinding("toy-fp32-v1");
    var sourceAdapter = new DecoderOnlyOnnxExecutionAdapter(sourceBinding);
    var targetAdapter = new DecoderOnlyOnnxExecutionAdapter(targetBinding);

    var sourceBackend = new OnnxRuntimeMigratableBackend(
        new OnnxRuntimeBackendOptions(
            modelId,
            sourceId,
            OnnxRuntimeModelSource.FromBytes(modelBytes),
            IntraOpNumThreads: 1,
            InterOpNumThreads: 1),
        sourceAdapter);
    var targetBackend = new OnnxRuntimeMigratableBackend(
        new OnnxRuntimeBackendOptions(
            modelId,
            targetId,
            OnnxRuntimeModelSource.FromBytes(modelBytes),
            IntraOpNumThreads: 1,
            InterOpNumThreads: 1),
        targetAdapter);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(
        sourceBackend,
        capacity: 8,
        maxBatchSize: 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(
        targetBackend,
        capacity: 8,
        maxBatchSize: 8);
    using var admission = new SequenceMigrationAdmissionController(
        maxInflightBytes: ToyHostStagingDecoderBinding.EstimatedBytes,
        maxConcurrentTransfers: 1);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, modelId);
    var sourceState = sourceBinding.LastPrefillState ??
        throw new InvalidOperationException("Source prefill did not publish physical decoder state.");
    var sourceKey = sourceState.GetLayer(0).Key;
    var sourceKeyData = sourceKey.GetTensorDataAsSpan<float>().ToArray();
    var before = GetSequence(runtime, sequenceId);
    var versionBefore = before.Version;

    await MigrateAsync(runtime, sequenceId, targetId);

    var migrated = GetSequence(runtime, sequenceId);
    Require(migrated.Device == targetId, "Host-staging migration must publish target placement.");
    Require(migrated.Version == versionBefore + 1, "Successful host-staging migration must advance version once.");
    Require(sourceBinding.ExportCount == 1, "Source must export exactly one staged physical payload.");
    Require(targetBinding.ImportCount == 1, "Target must import exactly one staged physical payload.");
    Require(sourceState.IsDisposed, "Source sequence state must be released after successful commit.");
    Require(admission.InflightBytes == 0, "Migration admission bytes must be released after commit.");
    Require(admission.ActiveTransfers == 0, "Migration transfer slot must be released after commit.");

    var imported = targetBinding.LastImportedState ??
        throw new InvalidOperationException("Target import did not publish decoder state.");
    Require(!imported.IsDisposed, "Imported target state must remain alive after migration commit.");
    var importedKey = imported.GetLayer(0).Key;
    Require(!ReferenceEquals(sourceKey, importedKey), "Target must own a distinct OrtValue, not alias source KV state.");
    Require(
        importedKey.GetTensorDataAsSpan<float>().SequenceEqual(sourceKeyData),
        "Target host-staged KV bytes must equal the source payload copied during prepare.");

    var decode = await DecodeOneAsync(runtime, sequenceId);
    Require(decode.TokenId == 42, "Target must continue decoding from migrated position 6.");
    Require(sourceBinding.DecodeCount == 0, "Source binding must not decode after placement moves to target.");
    Require(targetBinding.DecodeCount == 1, "Target binding must execute the first decode after migration.");
    Require(imported.IsDisposed, "Target decode replacement must release the imported immutable state version.");

    migrated.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), "Migrated target sequence must be releasable.");
}

static async Task RunIncompatibleCodecAsync(byte[] modelBytes)
{
    var modelId = new ModelId("toy-host-staging-incompatible");
    var sourceId = new DeviceId("cpu:onnx-source-incompatible");
    var targetId = new DeviceId("cpu:onnx-target-incompatible");
    var sourceBinding = new ToyHostStagingDecoderBinding("toy-fp32-v1");
    var targetBinding = new ToyHostStagingDecoderBinding("toy-fp32-v2");
    var sourceBackend = new OnnxRuntimeMigratableBackend(
        new OnnxRuntimeBackendOptions(
            modelId,
            sourceId,
            OnnxRuntimeModelSource.FromBytes(modelBytes)),
        new DecoderOnlyOnnxExecutionAdapter(sourceBinding));
    var targetBackend = new OnnxRuntimeMigratableBackend(
        new OnnxRuntimeBackendOptions(
            modelId,
            targetId,
            OnnxRuntimeModelSource.FromBytes(modelBytes)),
        new DecoderOnlyOnnxExecutionAdapter(targetBinding));

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, modelId);
    var before = GetSequence(runtime, sequenceId);
    var versionBefore = before.Version;
    var sourceState = sourceBinding.LastPrefillState!;

    var rejected = false;
    try
    {
        await MigrateAsync(runtime, sequenceId, targetId);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("No mutually supported migration transport", StringComparison.Ordinal))
    {
        rejected = true;
    }

    Require(rejected, "Different decoder host-staging codec ids must not negotiate a physical transport.");
    var after = GetSequence(runtime, sequenceId);
    Require(after.Device == sourceId, "Rejected transport negotiation must leave source placement unchanged.");
    Require(after.Version == versionBefore, "Rejected transport negotiation must not advance sequence version.");
    Require(sourceBinding.ExportCount == 0, "Transport negotiation must fail before source physical export.");
    Require(targetBinding.ImportCount == 0, "Transport negotiation failure must not create target state.");
    Require(!sourceState.IsDisposed, "Rejected migration must leave source physical state alive.");

    _ = await DecodeOneAsync(runtime, sequenceId);
    Require(sourceBinding.DecodeCount == 1, "Source actor must remain usable after transport negotiation failure.");
    Require(targetBinding.DecodeCount == 0, "Target must not decode after failed migration.");

    after.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), "Source sequence must remain releasable after failed negotiation.");
}

static void RunOptimumCodecRoundTrip()
{
    var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
        numHiddenLayers: 1,
        numKvHeads: 1,
        headDim: 2,
        vocabularySize: 16,
        kvElementType: TensorElementType.Float);
    using var binding = new OptimumLegacyFloatHostStagingBinding(
        new OptimumLegacyFloatDecoderBinding(profile));

    var keyBuffer = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
    var valueBuffer = new[] { 11f, 12f, 13f, 14f, 15f, 16f };
    var shape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 3);
    var sourceKey = OrtValue.CreateTensorValueFromMemory(keyBuffer, shape);
    var sourceValue = OrtValue.CreateTensorValueFromMemory(valueBuffer, shape);
    using var source = new DecoderOrtState(
        position: 3,
        new[] { new DecoderOrtLayerState(sourceKey, sourceValue) },
        nextTokenId: 7);

    var estimated = binding.EstimateHostStagingBytes(source);
    Require(estimated == 56, $"One 1x1x3x2 FP32 K/V state plus metadata must stage 56 bytes, got {estimated}.");
    var payload = binding.ExportHostStagingState(source);
    Require(payload.ByteLength == estimated, "Optimum staged payload byte length must match its estimate.");
    var imported = binding.ImportHostStagingState(payload);
    try
    {
        Require(imported.Position == 3, "Optimum import must preserve decoder position.");
        Require(imported.NextTokenId == 7, "Optimum import must preserve next-token frontier.");
        Require(
            !ReferenceEquals(source.GetLayer(0).Key, imported.GetLayer(0).Key),
            "Optimum import must construct a distinct target key OrtValue.");
        Require(
            !ReferenceEquals(source.GetLayer(0).Value, imported.GetLayer(0).Value),
            "Optimum import must construct a distinct target value OrtValue.");
        Require(
            imported.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(keyBuffer),
            "Optimum imported key bytes must equal the exported source key bytes.");
        Require(
            imported.GetLayer(0).Value.GetTensorDataAsSpan<float>().SequenceEqual(valueBuffer),
            "Optimum imported value bytes must equal the exported source value bytes.");

        source.Dispose();
        Array.Fill(keyBuffer, -1f);
        Array.Fill(valueBuffer, -2f);
        Require(
            imported.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            "Imported Optimum KV must remain independent after source buffers are mutated and disposed.");
    }
    finally
    {
        imported.Dispose();
    }
}

sealed class ToyHostStagingDecoderBinding : IDecoderOrtHostStagingBinding
{
    public const long EstimatedBytes = 56;
    private readonly string _formatId;
    private int _disposed;

    public ToyHostStagingDecoderBinding(string formatId)
    {
        _formatId = formatId;
        SessionContract = new OnnxSessionContract(
            Inputs: new[]
            {
                new OnnxTensorContract(
                    "decoder_input",
                    "X",
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

    public string Name => "toy-host-staging";
    public string HostStagingFormatId => _formatId;
    public OnnxSessionContract SessionContract { get; }
    public bool Initialized { get; private set; }
    public int ExportCount { get; private set; }
    public int ImportCount { get; private set; }
    public int DecodeCount { get; private set; }
    public DecoderOrtState? LastPrefillState { get; private set; }
    public DecoderOrtState? LastImportedState { get; private set; }

    public ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        Initialized = true;
        return ValueTask.CompletedTask;
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        if (item.Tokens.Length != 6)
        {
            throw new InvalidOperationException("Toy host-staging prefill requires exactly six tokens.");
        }

        var values = item.Tokens.Span.ToArray().Select(static value => (float)value).ToArray();
        var result = ExecuteStep(session, values, item.Tokens.Length, cancellationToken);
        LastPrefillState = result.State;
        return result;
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (priorState.Position != item.Position)
        {
            throw new InvalidOperationException("Toy host-staging decode position does not match physical state.");
        }

        _ = priorState.GetLayer(0).Key.GetTensorDataAsSpan<float>()[^1];
        DecodeCount++;
        var next = checked((float)(item.Position + 1));
        return ExecuteStep(
            session,
            new[] { next, next, next, next, next, next },
            checked(item.Position + 1),
            cancellationToken);
    }

    public long EstimateHostStagingBytes(DecoderOrtState state)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(state);
        return EstimatedBytes;
    }

    public DecoderOrtHostStagingPayload ExportHostStagingState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        var layer = state.GetLayer(0);
        ExportCount++;
        return new ToyPayload(
            HostStagingFormatId,
            state.Position,
            state.NextTokenId,
            layer.Key.GetTensorDataAsSpan<float>().ToArray(),
            layer.Value.GetTensorDataAsSpan<float>().ToArray());
    }

    public DecoderOrtState ImportHostStagingState(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        var staged = payload as ToyPayload ??
            throw new InvalidOperationException("Toy host-staging payload type mismatch.");
        if (!StringComparer.Ordinal.Equals(staged.FormatId, HostStagingFormatId))
        {
            throw new InvalidOperationException("Toy host-staging payload format mismatch.");
        }

        var key = OrtValue.CreateTensorValueFromMemory(staged.Key, new long[] { 3, 2 });
        var value = OrtValue.CreateTensorValueFromMemory(staged.Value, new long[] { 3, 2 });
        try
        {
            var state = new DecoderOrtState(
                staged.Position,
                new[] { new DecoderOrtLayerState(key, value) },
                staged.NextTokenId,
                staged);
            ImportCount++;
            LastImportedState = state;
            return state;
        }
        catch
        {
            value.Dispose();
            key.Dispose();
            throw;
        }
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
        var keyOutput = OrtValue.CreateTensorValueFromMemory(keyBuffer, new long[] { 3, 2 });
        var valueOutput = OrtValue.CreateTensorValueFromMemory(valueBuffer, new long[] { 3, 2 });

        try
        {
            using var runOptions = new RunOptions();
            using var keyInput = OrtValue.CreateTensorValueFromMemory(values, new long[] { 3, 2 });
            using var valueInput = OrtValue.CreateTensorValueFromMemory(valueInputBuffer, new long[] { 3, 2 });
            cancellationToken.ThrowIfCancellationRequested();
            CallerOwnedOrtRun.Execute(
                session,
                runOptions,
                new[] { "X" },
                new[] { keyInput },
                new[] { "Y" },
                new[] { keyOutput });
            CallerOwnedOrtRun.Execute(
                session,
                runOptions,
                new[] { "X" },
                new[] { valueInput },
                new[] { "Y" },
                new[] { valueOutput });

            var tokenId = checked((int)MathF.Round(keyBuffer[^1]));
            var state = new DecoderOrtState(
                nextPosition,
                new[] { new DecoderOrtLayerState(keyOutput, valueOutput) },
                tokenId);
            return new DecoderOrtStepResult(tokenId, state);
        }
        catch
        {
            valueOutput.Dispose();
            keyOutput.Dispose();
            throw;
        }
    }

    private void EnsureInitialized()
    {
        ThrowIfDisposed();
        if (!Initialized)
        {
            throw new InvalidOperationException("Toy host-staging binding is not initialized.");
        }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Initialized = false;
        }
    }

    private sealed class ToyPayload : DecoderOrtHostStagingPayload
    {
        public ToyPayload(
            string formatId,
            int position,
            int? nextTokenId,
            float[] key,
            float[] value)
            : base(formatId, position, nextTokenId, EstimatedBytes)
        {
            Key = key;
            Value = value;
        }

        public float[] Key { get; }
        public float[] Value { get; }
    }
}
