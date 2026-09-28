using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

internal static class AsyncImportDispatchSpec
{
    private const string MulModelBase64 =
        "CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0" +
        "KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4K" +
        "DAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==";

    [ModuleInitializer]
    internal static void Run()
    {
        using var session = new InferenceSession(
            Convert.FromBase64String(MulModelBase64));
        using var binding = new AsyncImportProbeBinding();
        using var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);
        adapter.InitializeAsync(session).GetAwaiter().GetResult();

        var modelId = new ModelId("async-import-dispatch");
        var sequenceId = SequenceId.New();
        var sourceDevice = new DeviceId("cuda:source");
        var targetDevice = new DeviceId("cuda:target");
        var payload = new ProbePayload(
            binding.HostStagingFormatId,
            position: 3,
            nextTokenId: 17);
        var transfer = new DecoderOrtHostStagingTransfer(
            Guid.NewGuid(),
            sequenceId,
            sourceDevice,
            targetDevice,
            modelId,
            payload,
            transportPlan: null);

        var migration = (IOnnxRuntimeSequenceMigrationAdapter)adapter;
        migration.ImportSequenceMigrationAsync(
                modelId,
                targetDevice,
                transfer)
            .GetAwaiter()
            .GetResult();

        Require(
            binding.AsyncImportCount == 1 && binding.SyncImportCount == 0,
            "Migration adapter must dispatch to the async import capability instead of the synchronous fallback.");

        migration.AbortSequenceMigrationAsync(
                modelId,
                targetDevice,
                transfer)
            .GetAwaiter()
            .GetResult();
        transfer.ReleasePayloadOwner();
        Require(
            payload.IsDisposed,
            "Async import dispatch spec must deterministically release the transfer payload owner.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static DecoderOrtState CreateState(
        int position,
        int? nextTokenId)
    {
        var key = OrtValue.CreateTensorValueFromMemory(
            new[] { 1f, 2f },
            new long[] { 1, 2 });
        var value = OrtValue.CreateTensorValueFromMemory(
            new[] { 3f, 4f },
            new long[] { 1, 2 });
        try
        {
            return new DecoderOrtState(
                position,
                new[] { new DecoderOrtLayerState(key, value) },
                nextTokenId);
        }
        catch
        {
            value.Dispose();
            key.Dispose();
            throw;
        }
    }

    private sealed class ProbePayload : DecoderOrtHostStagingPayload
    {
        public ProbePayload(string formatId, int position, int? nextTokenId)
            : base(formatId, position, nextTokenId, byteLength: 8)
        {
        }
    }

    private sealed class AsyncImportProbeBinding :
        IDecoderOrtAsyncHostStagingImportBinding
    {
        private int _asyncImportCount;
        private int _syncImportCount;

        public string Name => "async-import-probe";
        public string HostStagingFormatId => "async-import-probe-v1";
        public OnnxSessionContract SessionContract { get; } = new(
            Array.Empty<OnnxTensorContract>(),
            Array.Empty<OnnxTensorContract>());
        public int AsyncImportCount => Volatile.Read(ref _asyncImportCount);
        public int SyncImportCount => Volatile.Read(ref _syncImportCount);

        public long EstimateHostStagingBytes(DecoderOrtState state) => 8;

        public DecoderOrtHostStagingPayload ExportHostStagingState(
            DecoderOrtState state,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public DecoderOrtState ImportHostStagingState(
            DecoderOrtHostStagingPayload payload,
            CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _syncImportCount);
            throw new InvalidOperationException(
                "Synchronous import fallback must not run when async import capability is present.");
        }

        public ValueTask<DecoderOrtState> ImportHostStagingStateAsync(
            DecoderOrtHostStagingPayload payload,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref _asyncImportCount);
            return ValueTask.FromResult(
                CreateState(payload.Position, payload.NextTokenId));
        }

        public DecoderOrtStepResult ExecutePrefill(
            InferenceSession session,
            PrefillItem item,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public DecoderOrtStepResult ExecuteDecode(
            InferenceSession session,
            DecodeItem item,
            DecoderOrtState priorState,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public void Dispose()
        {
        }
    }
}
