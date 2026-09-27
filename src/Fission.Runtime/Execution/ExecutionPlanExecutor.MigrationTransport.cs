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
    private SequenceMigrationTimeoutPolicy _migrationTimeouts =
        SequenceMigrationTimeoutPolicy.Disabled;
    private bool _ownsMigrationAdmission = true;

    /// <summary>
    /// Creates a runtime with explicit physical-migration planning, admission,
    /// and optional cooperative phase deadlines. The admission controller is
    /// caller-owned and may be shared across executors that should consume one
    /// migration-pressure budget.
    /// </summary>
    public ExecutionPlanExecutor(
        ExecutionDeviceRegistry devices,
        SequenceMigrationTransportPlanner migrationTransportPlanner,
        SequenceMigrationAdmissionController migrationAdmission,
        IExecutionTraceSink? trace = null,
        KvPagePool? kvPagePool = null,
        SequenceMigrationTimeoutPolicy? migrationTimeouts = null)
        : this(devices, trace, kvPagePool)
    {
        ArgumentNullException.ThrowIfNull(migrationTransportPlanner);
        ArgumentNullException.ThrowIfNull(migrationAdmission);

        _migrationTimeouts = migrationTimeouts ?? SequenceMigrationTimeoutPolicy.Disabled;
        _migrationTimeouts.Validate();

        _migrationAdmission.Dispose();
        _migrationTransportPlanner = migrationTransportPlanner;
        _migrationAdmission = migrationAdmission;
        _ownsMigrationAdmission = false;
    }

    public long MigrationInflightBytes => _migrationAdmission.InflightBytes;
    public int ActiveMigrations => _migrationAdmission.ActiveTransfers;
    public SequenceMigrationTimeoutPolicy MigrationTimeouts => _migrationTimeouts;

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
        var currentPhase = SequenceMigrationPhase.EstimateBytes;
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
            currentPhase = SequenceMigrationPhase.EstimateBytes;
            var estimatedBytes = await RunMigrationPhaseAsync(
                    currentPhase,
                    token => sourceDevice.EstimateSequenceMigrationBytesAsync(
                        sequence.Id,
                        targetPlacement,
                        token),
                    cancellationToken)
                .ConfigureAwait(false);

            currentPhase = SequenceMigrationPhase.SourceCapabilityDiscovery;
            var sourceCapabilities = await RunMigrationPhaseAsync(
                    currentPhase,
                    token => sourceDevice.GetSequenceMigrationTransportCapabilitiesAsync(
                        targetDevice.Device,
                        token),
                    cancellationToken)
                .ConfigureAwait(false);

            currentPhase = SequenceMigrationPhase.TargetCapabilityDiscovery;
            var targetCapabilities = await RunMigrationPhaseAsync(
                    currentPhase,
                    token => targetDevice.GetSequenceMigrationTransportCapabilitiesAsync(
                        sourceDevice.Device,
                        token),
                    cancellationToken)
                .ConfigureAwait(false);

            currentPhase = SequenceMigrationPhase.Planning;
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

            currentPhase = SequenceMigrationPhase.Admission;
            admission = await RunMigrationPhaseAsync(
                    currentPhase,
                    token => _migrationAdmission.AcquireAsync(
                        transportPlan.EstimatedBytes,
                        token),
                    cancellationToken)
                .ConfigureAwait(false);

            currentPhase = SequenceMigrationPhase.Prepare;
            transfer = await RunMigrationPhaseAsync(
                    currentPhase,
                    token => sourceDevice.PrepareSequenceMigrationAsync(
                        sequence.Id,
                        targetPlacement,
                        transportPlan,
                        token),
                    cancellationToken)
                .ConfigureAwait(false);

            currentPhase = SequenceMigrationPhase.Attestation;
            ValidatePlannedTransfer(
                transfer,
                sequence.Id,
                sourceDevice.Device,
                targetPlacement,
                transportPlan,
                sourceDevice.BackendName);

            currentPhase = SequenceMigrationPhase.Import;
            importAttempted = true;
            await RunMigrationPhaseAsync(
                    currentPhase,
                    token => targetDevice.ImportSequenceMigrationCancellableAsync(
                        transfer,
                        token),
                    cancellationToken)
                .ConfigureAwait(false);

            currentPhase = SequenceMigrationPhase.Commit;
            await RunMigrationPhaseAsync(
                    currentPhase,
                    token => sourceDevice.CommitSequenceMigrationCancellableAsync(
                        transfer,
                        token),
                    cancellationToken)
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
            var classification = SequenceMigrationFailureClassifier.Classify(
                failure,
                currentPhase,
                cancellationToken);

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
                    elapsed: Stopwatch.GetElapsedTime(startedAt),
                    classification: classification);
                throw;
            }

            var rollbackFailures = await RollBackMigrationAsync(
                    transfer,
                    sourceDevice,
                    targetDevice,
                    importAttempted)
                .ConfigureAwait(false);

            var combinedHealthImpact = classification.HealthImpact;
            foreach (var rollbackFailure in rollbackFailures)
            {
                combinedHealthImpact |= rollbackFailure.Classification.HealthImpact;
            }

            classification = classification with
            {
                HealthImpact = combinedHealthImpact
            };

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
                rollbackFailures.Count,
                classification);

            if (rollbackFailures.Count == 0)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo
                    .Capture(failure)
                    .Throw();
            }

            throw new AggregateException(
                $"Sequence migration transaction {transfer.TransactionId} failed and rollback also encountered errors.",
                new[] { failure }.Concat(rollbackFailures.Select(static item => item.Failure)));
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
        int? rollbackFailureCount = null,
        SequenceMigrationFailureClassification? classification = null)
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
            RollbackFailureCount: rollbackFailureCount,
            MigrationPhase: classification?.Phase,
            MigrationFailureClass: classification?.FailureClass,
            MigrationHealthImpact: classification?.HealthImpact,
            MigrationTimeout: classification?.Timeout));
    }

    private async ValueTask<T> RunMigrationPhaseAsync<T>(
        SequenceMigrationPhase phase,
        Func<CancellationToken, ValueTask<T>> operation,
        CancellationToken cancellationToken)
    {
        var timeout = _migrationTimeouts.GetTimeout(phase);
        if (timeout == Timeout.InfiniteTimeSpan)
        {
            return await operation(cancellationToken).ConfigureAwait(false);
        }

        using var timeoutCancellation = new CancellationTokenSource();
        timeoutCancellation.CancelAfter(timeout);
        using var linkedCancellation = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            timeoutCancellation.Token);

        try
        {
            return await operation(linkedCancellation.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException exception)
            when (!cancellationToken.IsCancellationRequested &&
                  timeoutCancellation.IsCancellationRequested)
        {
            throw new SequenceMigrationTimeoutException(phase, timeout, exception);
        }
    }

    private async ValueTask RunMigrationPhaseAsync(
        SequenceMigrationPhase phase,
        Func<CancellationToken, ValueTask> operation,
        CancellationToken cancellationToken)
    {
        var timeout = _migrationTimeouts.GetTimeout(phase);
        if (timeout == Timeout.InfiniteTimeSpan)
        {
            await operation(cancellationToken).ConfigureAwait(false);
            return;
        }

        using var timeoutCancellation = new CancellationTokenSource();
        timeoutCancellation.CancelAfter(timeout);
        using var linkedCancellation = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            timeoutCancellation.Token);

        try
        {
            await operation(linkedCancellation.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException exception)
            when (!cancellationToken.IsCancellationRequested &&
                  timeoutCancellation.IsCancellationRequested)
        {
            throw new SequenceMigrationTimeoutException(phase, timeout, exception);
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

    private async ValueTask<IReadOnlyList<MigrationRollbackFailure>> RollBackMigrationAsync(
        SequenceMigrationTransfer transfer,
        ContinuousBatchExecutor sourceDevice,
        ContinuousBatchExecutor targetDevice,
        bool importAttempted)
    {
        var rollbackFailures = new List<MigrationRollbackFailure>();

        if (importAttempted)
        {
            try
            {
                await RunMigrationPhaseAsync(
                        SequenceMigrationPhase.TargetRollback,
                        token => targetDevice.AbortSequenceMigrationCancellableAsync(
                            transfer,
                            token),
                        CancellationToken.None)
                    .ConfigureAwait(false);
            }
            catch (Exception rollbackFailure)
            {
                rollbackFailures.Add(new MigrationRollbackFailure(
                    rollbackFailure,
                    SequenceMigrationFailureClassifier.Classify(
                        rollbackFailure,
                        SequenceMigrationPhase.TargetRollback)));
            }
        }

        try
        {
            await RunMigrationPhaseAsync(
                    SequenceMigrationPhase.SourceRollback,
                    token => sourceDevice.AbortSequenceMigrationCancellableAsync(
                        transfer,
                        token),
                    CancellationToken.None)
                .ConfigureAwait(false);
        }
        catch (Exception rollbackFailure)
        {
            rollbackFailures.Add(new MigrationRollbackFailure(
                rollbackFailure,
                SequenceMigrationFailureClassifier.Classify(
                    rollbackFailure,
                    SequenceMigrationPhase.SourceRollback)));
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

    private sealed record MigrationRollbackFailure(
        Exception Failure,
        SequenceMigrationFailureClassification Classification);
}
