using System.Collections.Concurrent;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;
using Fission.Runtime.Tracing;

namespace Fission.Runtime.Execution;

public sealed class ExecutionBindings
{
    private readonly IReadOnlyDictionary<SequenceId, ReadOnlyMemory<int>> _prefillTokens;

    public ExecutionBindings(IReadOnlyDictionary<SequenceId, ReadOnlyMemory<int>> prefillTokens)
    {
        _prefillTokens = prefillTokens;
    }

    public ReadOnlyMemory<int> ResolvePrefill(SequenceId sequenceId, int expectedTokenCount)
    {
        if (!_prefillTokens.TryGetValue(sequenceId, out var tokens))
        {
            throw new KeyNotFoundException($"No prefill token binding exists for sequence {sequenceId}.");
        }

        if (tokens.Length != expectedTokenCount)
        {
            throw new InvalidOperationException(
                $"Plan expects {expectedTokenCount} prefill tokens for {sequenceId}, but binding contains {tokens.Length}.");
        }

        return tokens;
    }
}

public sealed record ForkExecutionResult(
    SequenceId Parent,
    IReadOnlyList<SequenceId> Branches);

public sealed record ExecutionPlanResult(
    Guid PlanId,
    IReadOnlyList<BackendStepResult> BackendResults,
    IReadOnlyList<KvSnapshotId> Snapshots,
    IReadOnlyList<ForkExecutionResult> Forks);

/// <summary>
/// Interprets a compiled inference plan against the stateful runtime.
/// Plans targeting different sequences may execute concurrently and converge on
/// one of the registered single-device continuous batch executors. A sequence,
/// and any snapshot used by restore, is reserved for one plan at a time so
/// metadata state cannot race backend transaction barriers.
/// </summary>
public sealed class ExecutionPlanExecutor : IDisposable
{
    private readonly ExecutionDeviceRegistry _devices;
    private readonly IExecutionTraceSink? _trace;
    private readonly KvPagePool _kvPagePool;
    private readonly ConcurrentDictionary<SequenceId, SequenceProcess> _sequences = new();
    private readonly ConcurrentDictionary<KvSnapshotId, RuntimeOwnedSnapshot> _snapshots = new();
    private readonly ConcurrentDictionary<SequenceId, byte> _sequenceReservations = new();
    private readonly ConcurrentDictionary<KvSnapshotId, byte> _snapshotReservations = new();
    private int _disposed;

    public ExecutionPlanExecutor(
        ContinuousBatchExecutor device,
        IExecutionTraceSink? trace = null,
        KvPagePool? kvPagePool = null)
        : this(new ExecutionDeviceRegistry(device), trace, kvPagePool)
    {
    }

    public ExecutionPlanExecutor(
        ExecutionDeviceRegistry devices,
        IExecutionTraceSink? trace = null,
        KvPagePool? kvPagePool = null)
    {
        ArgumentNullException.ThrowIfNull(devices);
        _devices = devices;
        _trace = trace;
        _kvPagePool = kvPagePool ?? new KvPagePool(int.MaxValue);
    }

    public int SequenceCount => _sequences.Count;
    public int SnapshotCount => _snapshots.Count;
    internal int DeviceInferenceCapacity => _devices.MinimumInferenceCapacity;
    public KvPagePool KvPages => _kvPagePool;
    public RuntimeKvCapacity KvCapacity => new(
        _kvPagePool.Capacity,
        _kvPagePool.AllocatedPages,
        _kvPagePool.AvailablePages,
        _kvPagePool.TokensPerPage);

    public bool TryGetSequence(SequenceId sequenceId, out SequenceProcess? sequence) =>
        _sequences.TryGetValue(sequenceId, out sequence);

    internal DeviceId ResolveExecutionDevice(SequenceId sequenceId) =>
        _sequences.TryGetValue(sequenceId, out var sequence)
            ? _devices.ResolvePlacement(sequence.Device).Device
            : _devices.DefaultDevice;

    internal int GetDeviceInferenceCapacity(DeviceId device) =>
        _devices.GetInferenceCapacity(device);

    public async ValueTask<bool> ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ReserveSequence(sequenceId, "sequence release");

        try
        {
            if (!_sequences.TryGetValue(sequenceId, out var sequence))
            {
                return false;
            }

            if (sequence.Status is not SequenceStatus.Finished and not SequenceStatus.Cancelled)
            {
                throw new InvalidOperationException(
                    $"Cannot release sequence {sequenceId} while it is {sequence.Status}.");
            }

            var device = _devices.ResolvePlacement(sequence.Device);
            await device.ReleaseSequenceAsync(sequenceId, cancellationToken)
                .ConfigureAwait(false);

            if (!_sequences.TryRemove(sequenceId, out var removed))
            {
                return false;
            }

            removed.Dispose();
            return true;
        }
        finally
        {
            _sequenceReservations.TryRemove(sequenceId, out _);
        }
    }

    public async ValueTask<bool> ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ReserveSnapshot(snapshotId, "snapshot release");

        try
        {
            if (!_snapshots.TryGetValue(snapshotId, out var snapshot))
            {
                return false;
            }

            var device = _devices.ResolvePlacement(snapshot.Device);
            await device.ReleaseSnapshotAsync(snapshotId, cancellationToken)
                .ConfigureAwait(false);

            if (!_snapshots.TryRemove(snapshotId, out var removed))
            {
                return false;
            }

            removed.Snapshot.Dispose();
            return true;
        }
        finally
        {
            _snapshotReservations.TryRemove(snapshotId, out _);
        }
    }

    public async ValueTask<ExecutionPlanResult> ExecuteAsync(
        CompiledExecutionPlan plan,
        ExecutionBindings bindings,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(bindings);

        using var reservation = ReservePlan(plan);

        var backendResults = new List<BackendStepResult>();
        var snapshotIds = new List<KvSnapshotId>();
        var forks = new List<ForkExecutionResult>();

        Record(new ExecutionTraceEvent(
            plan.PlanId,
            ExecutionTraceKind.PlanStarted,
            -1,
            "Plan"));

        for (var stepIndex = 0; stepIndex < plan.Steps.Count; stepIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var step = plan.Steps[stepIndex];
            var operation = step.GetType().Name;
            RecordSequenceState(
                plan.PlanId,
                ExecutionTraceKind.StepStarted,
                stepIndex,
                operation,
                step.SequenceId);

            switch (step)
            {
                case PrefillExecutionStep prefill:
                    await ExecutePrefillAsync(prefill, bindings, backendResults, cancellationToken)
                        .ConfigureAwait(false);
                    break;

                case DecodeExecutionStep decode:
                    await ExecuteDecodeAsync(decode, backendResults, cancellationToken)
                        .ConfigureAwait(false);
                    break;

                case SnapshotKvExecutionStep snapshot:
                {
                    var sequence = GetSequence(snapshot.SequenceId);
                    var device = _devices.ResolvePlacement(sequence.Device);
                    var state = sequence.Snapshot();

                    try
                    {
                        await device.SnapshotSequenceAsync(
                                sequence.Id,
                                state.Id,
                                cancellationToken)
                            .ConfigureAwait(false);

                        var ownedSnapshot = new RuntimeOwnedSnapshot(state, sequence.Device);
                        if (!_snapshots.TryAdd(state.Id, ownedSnapshot))
                        {
                            await device.ReleaseSnapshotAsync(state.Id, cancellationToken)
                                .ConfigureAwait(false);
                            throw new InvalidOperationException($"Duplicate snapshot id {state.Id}.");
                        }
                    }
                    catch
                    {
                        state.Dispose();
                        throw;
                    }

                    snapshotIds.Add(state.Id);
                    Record(new ExecutionTraceEvent(
                        plan.PlanId,
                        ExecutionTraceKind.SnapshotCreated,
                        stepIndex,
                        operation,
                        sequence.Id,
                        SnapshotId: state.Id,
                        Position: sequence.Position,
                        KvPageCount: sequence.Kv.Count,
                        Device: sequence.Device));
                    break;
                }

                case ForkKvExecutionStep fork:
                {
                    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fork.Branches);
                    var parent = GetSequence(fork.SequenceId);
                    var device = _devices.ResolvePlacement(parent.Device);
                    var branches = new SequenceProcess[fork.Branches];
                    var branchIds = new SequenceId[fork.Branches];

                    for (var index = 0; index < fork.Branches; index++)
                    {
                        var branch = parent.Fork();
                        branches[index] = branch;
                        branchIds[index] = branch.Id;
                    }

                    try
                    {
                        await device.ForkSequenceAsync(
                                parent.Id,
                                branchIds,
                                cancellationToken)
                            .ConfigureAwait(false);
                    }
                    catch
                    {
                        foreach (var branch in branches)
                        {
                            branch.Dispose();
                        }

                        throw;
                    }

                    var addedCount = 0;
                    try
                    {
                        for (var index = 0; index < branches.Length; index++)
                        {
                            var branch = branches[index];
                            if (!_sequences.TryAdd(branch.Id, branch))
                            {
                                throw new InvalidOperationException(
                                    $"Duplicate forked sequence id {branch.Id}.");
                            }

                            addedCount++;
                            Record(new ExecutionTraceEvent(
                                plan.PlanId,
                                ExecutionTraceKind.SequenceForked,
                                stepIndex,
                                operation,
                                branch.Id,
                                RelatedSequenceId: parent.Id,
                                Position: branch.Position,
                                KvPageCount: branch.Kv.Count,
                                Device: branch.Device));
                        }
                    }
                    catch
                    {
                        for (var index = 0; index < branches.Length; index++)
                        {
                            if (index < addedCount)
                            {
                                _sequences.TryRemove(branches[index].Id, out _);
                            }

                            branches[index].Dispose();
                            try
                            {
                                await device.ReleaseSequenceAsync(
                                        branches[index].Id,
                                        CancellationToken.None)
                                    .ConfigureAwait(false);
                            }
                            catch
                            {
                                // Preserve the registration failure. Backend-wide DisposeAsync
                                // remains the final cleanup path if rollback release also fails.
                            }
                        }

                        throw;
                    }

                    forks.Add(new ForkExecutionResult(parent.Id, branchIds));
                    break;
                }

                case RestoreKvExecutionStep restore:
                {
                    var sequence = GetSequence(restore.SequenceId);
                    if (!_snapshots.TryGetValue(restore.SnapshotId, out var ownedSnapshot))
                    {
                        throw new KeyNotFoundException(
                            $"Snapshot {restore.SnapshotId} is not owned by this executor.");
                    }

                    var sequenceDevice = _devices.ResolvePlacement(sequence.Device);
                    var snapshotDevice = _devices.ResolvePlacement(ownedSnapshot.Device);
                    if (!ReferenceEquals(sequenceDevice, snapshotDevice))
                    {
                        throw new InvalidOperationException(
                            $"Cannot restore snapshot {restore.SnapshotId} on sequence {sequence.Id}: " +
                            $"snapshot state belongs to device {ownedSnapshot.Device}, but the sequence is on {sequence.Device}.");
                    }

                    await sequenceDevice.RestoreSequenceAsync(
                            sequence.Id,
                            restore.SnapshotId,
                            cancellationToken)
                        .ConfigureAwait(false);
                    sequence.Restore(ownedSnapshot.Snapshot);
                    break;
                }

                case MigrateKvExecutionStep migrate:
                {
                    var sequence = GetSequence(migrate.SequenceId);
                    if (sequence.Device == migrate.TargetDevice)
                    {
                        break;
                    }

                    _devices.ValidateMigrationTarget(migrate.TargetDevice);
                    var sourceDevice = _devices.ResolvePlacement(sequence.Device);
                    var targetDevice = _devices.ResolvePlacement(migrate.TargetDevice);

                    if (!ReferenceEquals(sourceDevice, targetDevice) &&
                        sourceDevice.SupportsTransactionalMigration &&
                        targetDevice.SupportsTransactionalMigration)
                    {
                        await ExecuteTransactionalMigrationAsync(
                                sequence,
                                sourceDevice,
                                targetDevice,
                                migrate.TargetDevice,
                                cancellationToken)
                            .ConfigureAwait(false);
                    }
                    else
                    {
                        await sourceDevice.MigrateSequenceAsync(
                                sequence.Id,
                                migrate.TargetDevice,
                                cancellationToken)
                            .ConfigureAwait(false);
                        sequence.MigrateTo(migrate.TargetDevice);
                    }

                    break;
                }

                default:
                    throw new NotSupportedException($"Unsupported execution step: {step.GetType().Name}.");
            }

            RecordSequenceState(
                plan.PlanId,
                ExecutionTraceKind.StepCompleted,
                stepIndex,
                operation,
                step.SequenceId);
        }

        Record(new ExecutionTraceEvent(
            plan.PlanId,
            ExecutionTraceKind.PlanCompleted,
            plan.Steps.Count,
            "Plan"));

        return new ExecutionPlanResult(plan.PlanId, backendResults, snapshotIds, forks);
    }

    private static async ValueTask ExecuteTransactionalMigrationAsync(
        SequenceProcess sequence,
        ContinuousBatchExecutor sourceDevice,
        ContinuousBatchExecutor targetDevice,
        DeviceId targetPlacement,
        CancellationToken cancellationToken)
    {
        SequenceMigrationTransfer? transfer = null;
        var importAttempted = false;

        try
        {
            transfer = await sourceDevice.PrepareSequenceMigrationAsync(
                    sequence.Id,
                    targetPlacement,
                    cancellationToken)
                .ConfigureAwait(false);

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
                throw;
            }

            throw new AggregateException(
                $"Sequence migration transaction {transfer.TransactionId} failed and rollback also encountered errors.",
                new[] { failure }.Concat(rollbackFailures));
        }
    }

    private PlanReservation ReservePlan(CompiledExecutionPlan plan)
    {
        var sequenceIds = plan.Steps
            .Select(static step => step.SequenceId)
            .Distinct()
            .ToArray();
        var snapshotIds = plan.Steps
            .OfType<RestoreKvExecutionStep>()
            .Select(static step => step.SnapshotId)
            .Distinct()
            .ToArray();

        var reservedSequences = new List<SequenceId>(sequenceIds.Length);
        var reservedSnapshots = new List<KvSnapshotId>(snapshotIds.Length);

        try
        {
            foreach (var sequenceId in sequenceIds)
            {
                ReserveSequence(sequenceId, $"plan {plan.PlanId}");
                reservedSequences.Add(sequenceId);
            }

            foreach (var snapshotId in snapshotIds)
            {
                ReserveSnapshot(snapshotId, $"plan {plan.PlanId}");
                reservedSnapshots.Add(snapshotId);
            }

            return new PlanReservation(
                _sequenceReservations,
                _snapshotReservations,
                reservedSequences.ToArray(),
                reservedSnapshots.ToArray());
        }
        catch
        {
            foreach (var snapshotId in reservedSnapshots)
            {
                _snapshotReservations.TryRemove(snapshotId, out _);
            }

            foreach (var sequenceId in reservedSequences)
            {
                _sequenceReservations.TryRemove(sequenceId, out _);
            }

            throw;
        }
    }

    private void ReserveSequence(SequenceId sequenceId, string operation)
    {
        if (!_sequenceReservations.TryAdd(sequenceId, 0))
        {
            throw new InvalidOperationException(
                $"Sequence {sequenceId} is already reserved by another runtime operation; cannot start {operation}.");
        }
    }

    private void ReserveSnapshot(KvSnapshotId snapshotId, string operation)
    {
        if (!_snapshotReservations.TryAdd(snapshotId, 0))
        {
            throw new InvalidOperationException(
                $"Snapshot {snapshotId} is already reserved by another runtime operation; cannot start {operation}.");
        }
    }

    private async ValueTask ExecutePrefillAsync(
        PrefillExecutionStep step,
        ExecutionBindings bindings,
        List<BackendStepResult> results,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(step.TokenCount);

        var sequence = _sequences.GetOrAdd(
            step.SequenceId,
            id => SequenceProcess.Create(id, step.ModelId, _devices.DefaultDevice, _kvPagePool));

        if (sequence.Model != step.ModelId)
        {
            throw new InvalidOperationException(
                $"Sequence {step.SequenceId} is already bound to model {sequence.Model}, not {step.ModelId}.");
        }

        if (sequence.Status == SequenceStatus.Waiting || sequence.Status == SequenceStatus.Suspended)
        {
            sequence.TransitionTo(SequenceStatus.Prefilling);
        }
        else if (sequence.Status != SequenceStatus.Prefilling)
        {
            throw new InvalidOperationException(
                $"Cannot prefill sequence {step.SequenceId} while it is {sequence.Status}.");
        }

        var tokens = bindings.ResolvePrefill(step.SequenceId, step.TokenCount);
        var device = _devices.ResolvePlacement(sequence.Device);
        var result = await device.SubmitPrefillAsync(
            new PrefillItem(step.SequenceId, step.ModelId, tokens),
            cancellationToken).ConfigureAwait(false);

        sequence.RecordPrefill(step.TokenCount);
        if (step.CompletesPrefill)
        {
            sequence.TransitionTo(SequenceStatus.Decoding);
        }

        results.Add(result);
    }

    private async ValueTask ExecuteDecodeAsync(
        DecodeExecutionStep step,
        List<BackendStepResult> results,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(step.MaxTokens);
        var sequence = GetSequence(step.SequenceId);

        if (sequence.Status == SequenceStatus.Suspended)
        {
            sequence.TransitionTo(SequenceStatus.Decoding);
        }

        if (sequence.Status != SequenceStatus.Decoding)
        {
            throw new InvalidOperationException(
                $"Cannot decode sequence {step.SequenceId} while it is {sequence.Status}.");
        }

        var device = _devices.ResolvePlacement(sequence.Device);
        for (var index = 0; index < step.MaxTokens; index++)
        {
            var result = await device.SubmitDecodeAsync(
                new DecodeItem(sequence.Id, sequence.Model, sequence.Position),
                cancellationToken).ConfigureAwait(false);

            sequence.RecordDecode();
            results.Add(result);

            if (result.IsFinished)
            {
                sequence.TransitionTo(SequenceStatus.Finished);
                break;
            }
        }
    }

    private SequenceProcess GetSequence(SequenceId sequenceId)
    {
        if (_sequences.TryGetValue(sequenceId, out var sequence))
        {
            return sequence;
        }

        throw new KeyNotFoundException($"Sequence {sequenceId} does not exist in this executor.");
    }

    private void RecordSequenceState(
        Guid planId,
        ExecutionTraceKind kind,
        int stepIndex,
        string operation,
        SequenceId sequenceId)
    {
        if (_trace is null)
        {
            return;
        }

        if (!_sequences.TryGetValue(sequenceId, out var sequence))
        {
            Record(new ExecutionTraceEvent(
                planId,
                kind,
                stepIndex,
                operation,
                sequenceId));
            return;
        }

        Record(new ExecutionTraceEvent(
            planId,
            kind,
            stepIndex,
            operation,
            sequence.Id,
            Position: sequence.Position,
            KvPageCount: sequence.Kv.Count,
            Device: sequence.Device));
    }

    private void Record(ExecutionTraceEvent traceEvent) =>
        _trace?.Record(traceEvent);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var snapshot in _snapshots.Values)
        {
            snapshot.Snapshot.Dispose();
        }

        _snapshots.Clear();

        foreach (var sequence in _sequences.Values)
        {
            sequence.Dispose();
        }

        _sequences.Clear();
        _sequenceReservations.Clear();
        _snapshotReservations.Clear();
    }

    private sealed record RuntimeOwnedSnapshot(
        KvSnapshot Snapshot,
        DeviceId Device);

    private sealed class PlanReservation : IDisposable
    {
        private readonly ConcurrentDictionary<SequenceId, byte> _sequenceReservations;
        private readonly ConcurrentDictionary<KvSnapshotId, byte> _snapshotReservations;
        private readonly SequenceId[] _sequenceIds;
        private readonly KvSnapshotId[] _snapshotIds;
        private int _disposed;

        public PlanReservation(
            ConcurrentDictionary<SequenceId, byte> sequenceReservations,
            ConcurrentDictionary<KvSnapshotId, byte> snapshotReservations,
            SequenceId[] sequenceIds,
            KvSnapshotId[] snapshotIds)
        {
            _sequenceReservations = sequenceReservations;
            _snapshotReservations = snapshotReservations;
            _sequenceIds = sequenceIds;
            _snapshotIds = snapshotIds;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            foreach (var snapshotId in _snapshotIds)
            {
                _snapshotReservations.TryRemove(snapshotId, out _);
            }

            foreach (var sequenceId in _sequenceIds)
            {
                _sequenceReservations.TryRemove(sequenceId, out _);
            }
        }
    }
}
