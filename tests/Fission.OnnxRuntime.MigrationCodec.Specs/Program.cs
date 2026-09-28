using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
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

using var session = new InferenceSession(
    Convert.FromBase64String(MulModelBase64));
using var sourceBinding = new BatchProbeBinding("source-batch");
using var targetBinding = new BatchProbeBinding("target-batch");
var sourceCodec = new StandaloneAsyncCodec("codec-source");
var targetCodec = new StandaloneAsyncCodec("codec-target");
using var sourceAdapter = new DecoderOnlyOnnxExecutionAdapter(
    sourceBinding,
    sourceCodec);
using var targetAdapter = new DecoderOnlyOnnxExecutionAdapter(
    targetBinding,
    targetCodec);

await sourceAdapter.InitializeAsync(session);
await targetAdapter.InitializeAsync(session);

var modelId = new ModelId("migration-codec-batch");
var sourceDevice = new DeviceId("cuda:0");
var targetDevice = new DeviceId("cuda:1");
var first = SequenceId.New();
var second = SequenceId.New();

var prefill = await sourceAdapter.PrefillAsync(
    session,
    new PrefillBatch(new[]
    {
        new PrefillItem(first, modelId, new ReadOnlyMemory<int>(new[] { 3 })),
        new PrefillItem(second, modelId, new ReadOnlyMemory<int>(new[] { 7 }))
    }));
Require(prefill.Count == 2, "Standalone migration codec must not interfere with model prefill results.");
Require(sourceBinding.BatchPrefillCalls == 1, "Separate migration codec must preserve the original batch-prefill binding capability.");
Require(sourceBinding.ScalarPrefillCalls == 0, "Separate migration codec must not force scalar prefill fallback.");

var decoded = await sourceAdapter.DecodeAsync(
    session,
    new DecodeBatch(new[]
    {
        new DecodeItem(first, modelId, Position: 1),
        new DecodeItem(second, modelId, Position: 1)
    }));
Require(decoded.Count == 2, "Standalone migration codec must not interfere with model decode results.");
Require(sourceBinding.BatchDecodeCalls == 1, "Separate migration codec must preserve the original batch-decode binding capability.");
Require(sourceBinding.ScalarDecodeCalls == 0, "Separate migration codec must not force scalar decode fallback.");

var sourceMigration = (IOnnxRuntimeSequenceMigrationAdapter)sourceAdapter;
var targetMigration = (IOnnxRuntimeSequenceMigrationAdapter)targetAdapter;
Require(sourceMigration.SupportsSequenceMigration, "A separately injected codec must enable decoder migration.");
Require(targetMigration.SupportsSequenceMigration, "Target adapter must advertise separately injected migration codec support.");

var capability = sourceMigration
    .GetSequenceMigrationTransportCapabilities(modelId, sourceDevice, targetDevice)
    .Single();
Require(
    capability.TransportId.Contains(StandaloneAsyncCodec.FormatId, StringComparison.Ordinal),
    "Transport identity must come from the standalone codec format rather than the execution binding type.");

var estimated = await sourceMigration.EstimateSequenceMigrationBytesAsync(
    modelId,
    sourceDevice,
    first,
    targetDevice);
Require(estimated == StandaloneAsyncCodec.PayloadBytes, "Standalone codec must own migration byte estimation.");

var transfer = await sourceMigration.PrepareSequenceMigrationAsync(
    modelId,
    sourceDevice,
    first,
    targetDevice,
    transportPlan: null);
Require(sourceCodec.AsyncExportCalls == 1, "Adapter must dispatch standalone async export capability.");
Require(sourceCodec.SyncExportCalls == 0, "Async standalone codec must bypass synchronous export fallback.");

await targetMigration.ImportSequenceMigrationAsync(
    modelId,
    targetDevice,
    transfer);
Require(targetCodec.AsyncImportCalls == 1, "Adapter must dispatch standalone async import capability.");
Require(targetCodec.SyncImportCalls == 0, "Async standalone codec must bypass synchronous import fallback.");

var targetDecoded = await targetAdapter.DecodeAsync(
    session,
    new DecodeBatch(new[]
    {
        new DecodeItem(first, modelId, Position: 2)
    }));
Require(targetDecoded.Count == 1, "Imported state must remain usable by the original target execution binding.");
Require(targetBinding.BatchDecodeCalls == 1, "Imported state must continue through target batch-decode capability.");
Require(targetBinding.ScalarDecodeCalls == 0, "Migration codec injection must not hide target batch capability.");

await targetMigration.AbortSequenceMigrationAsync(
    modelId,
    targetDevice,
    transfer);
await sourceMigration.AbortSequenceMigrationAsync(
    modelId,
    sourceDevice,
    transfer);
Require(sourceCodec.LastPayload is { IsDisposed: true }, "Source abort must release the standalone codec payload owner.");

await sourceAdapter.ReleaseSequenceAsync(first);
await sourceAdapter.ReleaseSequenceAsync(second);

Console.WriteLine(
    $"Fission decoder migration-codec specs passed: " +
    $"sourceBatchPrefill={sourceBinding.BatchPrefillCalls}, " +
    $"sourceBatchDecode={sourceBinding.BatchDecodeCalls}, " +
    $"targetBatchDecode={targetBinding.BatchDecodeCalls}, " +
    $"asyncExport={sourceCodec.AsyncExportCalls}, asyncImport={targetCodec.AsyncImportCalls}.");

sealed class BatchProbeBinding :
    IDecoderOrtBatchModelBinding,
    IDecoderOrtBatchPrefillModelBinding
{
    private int _disposed;

    public BatchProbeBinding(string name)
    {
        Name = name;
    }

    public string Name { get; }
    public int ScalarPrefillCalls { get; private set; }
    public int BatchPrefillCalls { get; private set; }
    public int ScalarDecodeCalls { get; private set; }
    public int BatchDecodeCalls { get; private set; }

    public OnnxSessionContract SessionContract { get; } = new(
        Array.Empty<OnnxTensorContract>(),
        Array.Empty<OnnxTensorContract>());

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        ScalarPrefillCalls++;
        throw new InvalidOperationException("Scalar prefill must not run for the batch probe binding.");
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecutePrefillBatch(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        BatchPrefillCalls++;
        var results = new DecoderOrtStepResult[items.Count];
        for (var index = 0; index < items.Count; index++)
        {
            if (priorStates[index] is not null)
            {
                throw new InvalidOperationException("Probe spec only uses initial batch prefill.");
            }

            var token = checked(items[index].Tokens.Span[^1] * 10);
            results[index] = new DecoderOrtStepResult(
                token,
                CreateState(
                    checked(items[index].Position!.Value + items[index].Tokens.Length),
                    token));
        }

        return results;
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ScalarDecodeCalls++;
        throw new InvalidOperationException("Scalar decode must not run for the batch probe binding.");
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecuteDecodeBatch(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        BatchDecodeCalls++;
        var results = new DecoderOrtStepResult[items.Count];
        for (var index = 0; index < items.Count; index++)
        {
            var token = checked((priorStates[index].NextTokenId ?? 0) + 1);
            results[index] = new DecoderOrtStepResult(
                token,
                CreateState(items[index].Position + 1, token));
        }

        return results;
    }

    private static DecoderOrtState CreateState(int position, int token)
    {
        var key = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)token },
            new long[] { 1, 1, 1, 1 });
        var value = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)token },
            new long[] { 1, 1, 1, 1 });
        try
        {
            return new DecoderOrtState(
                position,
                new[] { new DecoderOrtLayerState(key, value) },
                token);
        }
        catch
        {
            value.Dispose();
            key.Dispose();
            throw;
        }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose() => Interlocked.Exchange(ref _disposed, 1);
}

sealed class StandaloneAsyncCodec :
    IDecoderOrtAsyncHostStagingCodec,
    IDecoderOrtAsyncHostStagingImportCodec
{
    public const string FormatId = "standalone-host-codec-v1";
    public const long PayloadBytes = 8;

    public StandaloneAsyncCodec(string name)
    {
        Name = name;
    }

    public string Name { get; }
    public string HostStagingFormatId => FormatId;
    public int SyncExportCalls { get; private set; }
    public int AsyncExportCalls { get; private set; }
    public int SyncImportCalls { get; private set; }
    public int AsyncImportCalls { get; private set; }
    public ProbePayload? LastPayload { get; private set; }

    public long EstimateHostStagingBytes(DecoderOrtState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(state.IsDisposed, state);
        return PayloadBytes;
    }

    public DecoderOrtHostStagingPayload ExportHostStagingState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        SyncExportCalls++;
        throw new InvalidOperationException("Synchronous export fallback must not run for async standalone codec.");
    }

    public ValueTask<DecoderOrtHostStagingPayload> ExportHostStagingStateAsync(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        AsyncExportCalls++;
        LastPayload = new ProbePayload(
            HostStagingFormatId,
            state.Position,
            state.NextTokenId,
            PayloadBytes);
        return ValueTask.FromResult<DecoderOrtHostStagingPayload>(LastPayload);
    }

    public DecoderOrtState ImportHostStagingState(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default)
    {
        SyncImportCalls++;
        throw new InvalidOperationException("Synchronous import fallback must not run for async standalone codec.");
    }

    public ValueTask<DecoderOrtState> ImportHostStagingStateAsync(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        AsyncImportCalls++;
        var token = payload.NextTokenId ?? throw new InvalidOperationException("Probe payload is missing next-token frontier.");
        var key = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)token },
            new long[] { 1, 1, 1, 1 });
        var value = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)token },
            new long[] { 1, 1, 1, 1 });
        try
        {
            return ValueTask.FromResult(
                new DecoderOrtState(
                    payload.Position,
                    new[] { new DecoderOrtLayerState(key, value) },
                    token));
        }
        catch
        {
            value.Dispose();
            key.Dispose();
            throw;
        }
    }

    public sealed class ProbePayload : DecoderOrtHostStagingPayload
    {
        public ProbePayload(
            string formatId,
            int position,
            int? nextTokenId,
            long byteLength)
            : base(formatId, position, nextTokenId, byteLength)
        {
        }
    }
}
