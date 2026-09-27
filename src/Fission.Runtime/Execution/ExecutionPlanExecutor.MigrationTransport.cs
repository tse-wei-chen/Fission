using System.Diagnostics;
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
        Guid planId,
        int stepIndex,
        CancellationToken cancellationToken)
    {
        var startedAt = Stopwatch.GetTimestamp();
        var sourcePlacement = sequence.Device;
        SequenceMigrationTransportPlan? transportPlan = null;
        SequenceMigrationTransfer? transfer = null;
        SequenceMigrationAdmissionController.Lease? admission = null;
        var importAttempted = false;

        RecordMigrationTrace(
            planId,
            stepIndex,
            ExecutionTraceKind.MigrationStarted,
            sequence,
            sourcePlacement,
            targetPlacement);

        try
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

            transportPlan = _migrationTransportPlanner.Plan(
                estimatedBytes,
                sourceCapabilities,
                targetCapabilities);

            RecordMigrationTrace(
                planId,
                stepIndex,
                ExecutionTraceKind.MigrationPlanned,
                sequence,
                sourcePlacement,
                targetPlacement,
                transportPlan: transportPlan);

            admission = await _migrationAdmission
                .AcquireAsync(transportPlan.EstimatedBytes, cancellationToken)
                .ConfigureAwait(false);

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

            RecordMigrationTrace(
                planId,
                stepIndex,
                ExecutionTraceKind.MigrationCommitted,
                sequence,
                sourcePlacement,
                targetPlacement,
                transportPlan,
                transfer,
                elapsed: Stopwatch.GetElapsedTime(startedAt));
        }
        catch (Exception failure)
        {
            if (transfer is null)
            {
                RecordMigrationTrace(
                    planId,
                    stepIndex,
                    ExecutionTraceKind.MigrationFailed,
                    sequence,
                    sourcePlacement,
                    targetPlacement,
                    transportPlan,
                    failure: failure,
                    elapsed: Stopwatch.GetElapsedTime(startedAt));
                throw;
            }

            var rollbackFailures = await RollBackMigrationAsync(
                    transfer,
                    sourceDevice,
                    targetDevice,
                    importAttempted)
                .ConfigureAwait(false);

            var traceKind = rollbackFailures.Count == 0
                ? ExecutionTraceKind.MigrationRolledBack
                : ExecutionTraceKind.MigrationRollbackFailed;
            RecordMigrationTrace(
                planId,
                stepIndex,
                traceKind,
                sequence,
                sourcePlacement,
                targetPlacement,
                transportPlan,
                transfer,
                failure,
                Stopwatch.GetElapsedTime(startedAt),
                rollbackFailures.Count);

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
        finally
        {
            admission?.Dispose();
        }
    }

    private void RecordMigrationTrace(
        Guid planId,
        int stepIndex,
        ExecutionTraceKind kind,
        SequenceProcess sequence,
        DeviceId sourcePlacement,
        DeviceId targetPlacement,
        SequenceMigrationTransportPlan? transportPlan = null,
        SequenceMigrationTransfer? transfer = null,
        Exception? failure = null,
        TimeSpan? elapsed = null,
        int? rollbackFailureCount = null)
    {
        Record(new ExecutionTraceEvent(
            planId,
            kind,
            stepIndex,
            nameof(MigrateKvExecutionStep),
            sequence.Id,
            Position: sequence.Position,
            KvPageCount: sequence.Kv.Count,
            Device: sourcePlacement,
            TargetDevice: targetPlacement,
            TransactionId: transfer?.TransactionId,
            TransportId: transportPlan?.TransportId,
            TransportKind: transportPlan?.Kind,
            TransferBytes: transportPlan?.EstimatedBytes,
            EstimatedDuration: transportPlan?.EstimatedDuration,
            Elapsed: elapsed,
            FailureType: failure?.GetType().FullName,
            RollbackFailureCount: rollbackFailureCount));
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

    private static async ValueTask<IReadOnlyList<Exception>> RollBackMigrationAsync(
        SequenceMigrationTransfer transfer,
        ContinuousBatchExecutor sourceDevice,
        ContinuousBatchExecutor targetDevice,
        bool importAttempted)
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

        return rollbackFailures;
    }

    private void DisposeMigrationTransportResources()
    {
        if (_ownsMigrationAdmission)
        {
            _migrationAdmission.Dispose();
        }
    }
}
