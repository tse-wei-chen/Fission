using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;
using Fission.Runtime.Tracing;

namespace Fission.Runtime.Execution;

public sealed partial class ExecutionPlanExecutor
{
    private SequenceMigrationTransportPlanner _migrationTransportPlanner = new();
    private SequenceMigrationAdmissionController _migrationAdmission =
        new(long.MaxValue, int.MaxValue);
    private bool _ownsMigrationAdmission = true;

    /// <summary>
    /// Creates a runtime with explicit physical-migration planning and admission
    /// policy. The admission controller is caller-owned and may be shared across
    /// executors that should consume one migration-pressure budget.
    /// </summary>
    public ExecutionPlanExecutor(
        ExecutionDeviceRegistry devices,
        SequenceMigrationTransportPlanner migrationTransportPlanner,
        SequenceMigrationAdmissionController migrationAdmission,
        IExecutionTraceSink? trace = null,
        KvPagePool? kvPagePool = null)
        : this(devices, trace, kvPagePool)
    {
        ArgumentNullException.ThrowIfNull(migrationTransportPlanner);
        ArgumentNullException.ThrowIfNull(migrationAdmission);

        _migrationAdmission.Dispose();
        _migrationTransportPlanner = migrationTransportPlanner;
        _migrationAdmission = migrationAdmission;
        _ownsMigrationAdmission = false;
    }

    public long MigrationInflightBytes => _migrationAdmission.InflightBytes;
    public int ActiveMigrations => _migrationAdmission.ActiveTransfers;

    private async ValueTask ExecuteTransportAwareTransactionalMigrationAsync(
        SequenceProcess sequence,
        ContinuousBatchExecutor sourceDevice,
        ContinuousBatchExecutor targetDevice,
        DeviceId targetPlacement,
        CancellationToken cancellationToken)
    {
        var estimatedBytes = await sourceDevice.EstimateSequenceMigrationBytesAsync(
                sequence.Id,
                targetPlacement,
                cancellationToken)
            .ConfigureAwait(false);

        var sourceCapabilities = await sourceDevice
            .GetSequenceMigrationTransportCapabilitiesAsync(
                targetDevice.Device,
                cancellationToken)
            .ConfigureAwait(false);
        var targetCapabilities = await targetDevice
            .GetSequenceMigrationTransportCapabilitiesAsync(
                sourceDevice.Device,
                cancellationToken)
            .ConfigureAwait(false);

        var transportPlan = _migrationTransportPlanner.Plan(
            estimatedBytes,
            sourceCapabilities,
            targetCapabilities);

        using var admission = await _migrationAdmission
            .AcquireAsync(transportPlan.EstimatedBytes, cancellationToken)
            .ConfigureAwait(false);

        SequenceMigrationTransfer? transfer = null;
        var importAttempted = false;

        try
        {
            transfer = await sourceDevice.PrepareSequenceMigrationAsync(
                    sequence.Id,
                    targetPlacement,
                    transportPlan,
                    cancellationToken)
                .ConfigureAwait(false);

            ValidatePlannedTransfer(
                transfer,
                sequence.Id,
                sourceDevice.Device,
                targetPlacement,
                transportPlan,
                sourceDevice.BackendName);

            importAttempted = true;
            await targetDevice.ImportSequenceMigrationAsync(transfer, cancellationToken)
                .ConfigureAwait(false);

            await sourceDevice.CommitSequenceMigrationAsync(transfer, cancellationToken)
                .ConfigureAwait(false);

            sequence.MigrateTo(targetPlacement);
        }
        catch (Exception failure)
        {
            if (transfer is null)
            {
                throw;
            }

            await RollBackMigrationAsync(
                    transfer,
                    sourceDevice,
                    targetDevice,
                    importAttempted,
                    failure)
                .ConfigureAwait(false);
        }
    }

    private static void ValidatePlannedTransfer(
        SequenceMigrationTransfer transfer,
        SequenceId sequenceId,
        DeviceId sourceDevice,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan,
        string backendName)
    {
        if (transfer.SequenceId != sequenceId ||
            transfer.SourceDevice != sourceDevice ||
            transfer.TargetDevice != targetDevice)
        {
            throw new InvalidOperationException(
                $"Backend {backendName} returned migration transfer {transfer.TransactionId} " +
                "with sequence or device identity that does not match the planned prepare request.");
        }

        if (transfer.TransportPlan is null)
        {
            throw new InvalidOperationException(
                $"Backend {backendName} returned transport-aware migration transfer {transfer.TransactionId} " +
                "without a transport plan attestation.");
        }

        if (transfer.TransportPlan != transportPlan)
        {
            throw new InvalidOperationException(
                $"Backend {backendName} returned migration transfer {transfer.TransactionId} " +
                $"for transport '{transfer.TransportPlan.TransportId}', but runtime selected '{transportPlan.TransportId}'.");
        }
    }

    private static async ValueTask RollBackMigrationAsync(
        SequenceMigrationTransfer transfer,
        ContinuousBatchExecutor sourceDevice,
        ContinuousBatchExecutor targetDevice,
        bool importAttempted,
        Exception failure)
    {
        var rollbackFailures = new List<Exception>();

        if (importAttempted)
        {
            try
            {
                await targetDevice.AbortSequenceMigrationAsync(
                        transfer,
                        CancellationToken.None)
                    .ConfigureAwait(false);
            }
            catch (Exception rollbackFailure)
            {
                rollbackFailures.Add(rollbackFailure);
            }
        }

        try
        {
            await sourceDevice.AbortSequenceMigrationAsync(
                    transfer,
                    CancellationToken.None)
                .ConfigureAwait(false);
        }
        catch (Exception rollbackFailure)
        {
            rollbackFailures.Add(rollbackFailure);
        }

        if (rollbackFailures.Count == 0)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failure)
                .Throw();
        }

        throw new AggregateException(
            $"Sequence migration transaction {transfer.TransactionId} failed and rollback also encountered errors.",
            new[] { failure }.Concat(rollbackFailures));
    }

    private void DisposeMigrationTransportResources()
    {
        if (_ownsMigrationAdmission)
        {
            _migrationAdmission.Dispose();
        }
    }
}
