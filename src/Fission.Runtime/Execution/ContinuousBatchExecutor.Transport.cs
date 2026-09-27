using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    internal bool SupportsTransportAwareMigration =>
        _backend is ISequenceMigrationTransportBackend;

    internal async ValueTask<IReadOnlyList<SequenceMigrationTransportCapability>>
        GetSequenceMigrationTransportCapabilitiesAsync(
            DeviceId peerDevice,
            CancellationToken cancellationToken = default)
    {
        var work = new PendingGetMigrationTransportCapabilities(peerDevice);
        await SubmitControlAsync(work, cancellationToken).ConfigureAwait(false);
        return work.Capabilities ?? throw new InvalidOperationException(
            $"Backend {BackendName} completed migration capability discovery without a result.");
    }

    internal async ValueTask<long> EstimateSequenceMigrationBytesAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        var work = new PendingEstimateMigrationBytes(sequenceId, targetDevice);
        await SubmitControlAsync(work, cancellationToken).ConfigureAwait(false);
        if (work.EstimatedBytes <= 0)
        {
            throw new InvalidOperationException(
                $"Backend {BackendName} estimated {work.EstimatedBytes} migration byte(s) for sequence {sequenceId}; " +
                "transport-aware migration estimates must be positive.");
        }

        return work.EstimatedBytes;
    }

    internal async ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transportPlan);
        var work = new PendingPreparePlannedMigration(
            sequenceId,
            targetDevice,
            transportPlan);
        await SubmitControlAsync(work, cancellationToken).ConfigureAwait(false);
        return work.Transfer ?? throw new InvalidOperationException(
            $"Backend {BackendName} completed planned migration prepare without a transfer token.");
    }

    private sealed class PendingGetMigrationTransportCapabilities(
        DeviceId peerDevice) : PendingControl
    {
        public IReadOnlyList<SequenceMigrationTransportCapability>? Capabilities { get; private set; }

        public override ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            var transport = RequireTransportMigration(backend);
            var capabilities = transport.GetSequenceMigrationTransportCapabilities(peerDevice);
            ArgumentNullException.ThrowIfNull(capabilities);
            Capabilities = capabilities.ToArray();
            return ValueTask.CompletedTask;
        }
    }

    private sealed class PendingEstimateMigrationBytes(
        SequenceId sequenceId,
        DeviceId targetDevice) : PendingControl
    {
        public long EstimatedBytes { get; private set; }

        public override async ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            EstimatedBytes = await RequireTransportMigration(backend)
                .EstimateSequenceMigrationBytesAsync(sequenceId, targetDevice)
                .ConfigureAwait(false);
        }
    }

    private sealed class PendingPreparePlannedMigration(
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan) : PendingControl
    {
        public SequenceMigrationTransfer? Transfer { get; private set; }

        public override async ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            Transfer = await RequireTransportMigration(backend)
                .PrepareSequenceMigrationAsync(
                    sequenceId,
                    targetDevice,
                    transportPlan)
                .ConfigureAwait(false);
        }
    }

    private static ISequenceMigrationTransportBackend RequireTransportMigration(
        IInferenceBackend backend) =>
        backend as ISequenceMigrationTransportBackend ??
        throw new NotSupportedException(
            $"Backend '{backend.Name}' does not implement transport-aware sequence migration.");
}
