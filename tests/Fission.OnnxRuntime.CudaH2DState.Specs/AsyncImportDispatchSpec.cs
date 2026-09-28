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
    internal static void Run() => RunAsync().GetAwaiter().GetResult();

    private static async Task RunAsync()
    {
        using var session = new InferenceSession(
            Convert.FromBase64String(MulModelBase64));
        using var binding = new AsyncImportProbeBinding();
        using var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);
        await adapter.InitializeAsync(session);

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
        var importTask = migration.ImportSequenceMigrationAsync(
            modelId,
            targetDevice,
            transfer).AsTask();

        await binding.Started.Task.WaitAsync(TimeSpan.FromSeconds(3));
        Require(
            !importTask.IsCompleted,
            "Migration adapter must await the asynchronous host-staging import before publishing target state.");
        Require(
            binding.AsyncImportCount == 1 && binding.SyncImportCount == 0,
            "Migration adapter must dispatch to the async import capability instead of the synchronous fallback.");

        binding.Release.TrySetResult();
        await importTask;
        Require(
            binding.AsyncImportCount == 1 && binding.SyncImportCount == 0,
            "Successful target import must use exactly one asynchronous import call.");

        await migration.AbortSequenceMigrationAsync(
            modelId,
            targetDevice,
            transfer);
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
        public TaskCompletionSource Started { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource Release { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

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

        public async ValueTask<DecoderOrtState> ImportHostStagingStateAsync(
            DecoderOrtHostStagingPayload payload,
            CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _asyncImportCount);
            Started.TrySetResult();
            await Release.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();

            var key = OrtValue.CreateTensorValueFromMemory(
                new[] { 1f, 2f },
                new long[] { 1, 2 });
            var value = OrtValue.CreateTensorValueFromMemory(
                new[] { 3f, 4f },
                new long[] { 1, 2 });
            try
            {
                return new DecoderOrtState(
                    payload.Position,
                    new[] { new DecoderOrtLayerState(key, value) },
                    payload.NextTokenId);
            }
            catch
            {
                value.Dispose();
                key.Dispose();
                throw;
            }
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
            Release.TrySetResult();
        }
    }
}
