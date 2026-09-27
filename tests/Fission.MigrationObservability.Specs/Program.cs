using System.Collections.Concurrent;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Tracing;

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
    ModelId model)
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new PrefillExecutionStep(sequenceId, model, 4, CompletesPrefill: true)
            }),
        new ExecutionBindings(
            new Dictionary<SequenceId, ReadOnlyMemory<int>>
            {
                [sequenceId] = new ReadOnlyMemory<int>(new[] { 1, 2, 3, 4 })
            }));
}

static Task MigrateAsync(
    ExecutionPlanExecutor runtime,
    Guid planId,
    SequenceId sequenceId,
    DeviceId targetDevice) =>
    runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            planId,
            0,
            new ExecutionStep[]
            {
                new MigrateKvExecutionStep(sequenceId, targetDevice)
            }),
        EmptyBindings()).AsTask();

static ExecutionTraceEvent[] MigrationEvents(
    InMemoryExecutionTraceSink trace,
    Guid planId) =>
    trace.Snapshot()
        .Select(static item => item.Event)
        .Where(item => item.PlanId == planId && item.Kind is
            ExecutionTraceKind.MigrationStarted or
            ExecutionTraceKind.MigrationPlanned or
            ExecutionTraceKind.MigrationCommitted or
            ExecutionTraceKind.MigrationRolledBack or
            ExecutionTraceKind.MigrationRollbackFailed or
            ExecutionTraceKind.MigrationFailed)
        .ToArray();

static void RequireKinds(
    IReadOnlyList<ExecutionTraceEvent> events,
    params ExecutionTraceKind[] expected)
{
    Require(
        events.Select(static item => item.Kind).SequenceEqual(expected),
        $"Expected migration trace kinds [{string.Join(',', expected)}], got [{string.Join(',', events.Select(static item => item.Kind))}].");
}

static async Task RunCommittedAsync()
{
    var sourceId = new DeviceId("gpu:trace-source");
    var targetId = new DeviceId("gpu:trace-target");
    var sourceBackend = new TraceTransportBackend(sourceId);
    var targetBackend = new TraceTransportBackend(targetId);
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    var model = new ModelId("trace-commit");
    await PrefillAsync(runtime, sequenceId, model);

    var migrationPlanId = Guid.NewGuid();
    await MigrateAsync(runtime, migrationPlanId, sequenceId, targetId);

    var events = MigrationEvents(trace, migrationPlanId);
    RequireKinds(
        events,
        ExecutionTraceKind.MigrationStarted,
        ExecutionTraceKind.MigrationPlanned,
        ExecutionTraceKind.MigrationCommitted);

    var started = events[0];
    Require(started.StepIndex == 0, "Migration trace must preserve the execution-plan step index.");
    Require(started.SequenceId == sequenceId, "Migration trace must preserve sequence identity.");
    Require(started.Device == sourceId && started.TargetDevice == targetId, "Migration trace must preserve source/target placement.");

    var planned = events[1];
    Require(planned.TransportId == TraceTransportBackend.TransportId, "Planned trace must expose selected transport id.");
    Require(planned.TransportKind == SequenceMigrationTransportKind.HostStaging, "Planned trace must expose transport kind.");
    Require(planned.TransferBytes == TraceTransportBackend.EstimatedBytes, "Planned trace must expose physical byte estimate.");
    Require(planned.EstimatedDuration is { } estimate && estimate > TimeSpan.Zero, "Planned trace must expose planner duration estimate.");
    Require(planned.TransactionId is null, "Planning happens before a backend transfer transaction exists.");

    var committed = events[2];
    Require(committed.TransactionId is { } transactionId && transactionId != Guid.Empty, "Committed trace must expose transaction id.");
    Require(committed.TransportId == TraceTransportBackend.TransportId, "Committed trace must retain selected transport id.");
    Require(committed.TransferBytes == TraceTransportBackend.EstimatedBytes, "Committed trace must retain transfer byte estimate.");
    Require(committed.Elapsed is { } elapsed && elapsed >= TimeSpan.Zero, "Committed trace must expose measured end-to-end elapsed time.");
    Require(committed.FailureType is null, "Committed migration must not report a failure type.");
    Require(targetBackend.GetPosition(sequenceId) == 4, "Committed migration must leave runnable target state.");

    var replay = ExecutionTraceReplay.Replay(trace.Snapshot());
    Require(replay.Sequences[sequenceId].Device == targetId, "Replay must observe target placement after successful migration step completion.");
}

static async Task RunPrepareFailureAsync()
{
    var sourceId = new DeviceId("gpu:prepare-source");
    var targetId = new DeviceId("gpu:prepare-target");
    var sourceBackend = new TraceTransportBackend(sourceId) { FailPrepare = true };
    var targetBackend = new TraceTransportBackend(targetId);
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("trace-prepare-failure"));

    var migrationPlanId = Guid.NewGuid();
    var failed = false;
    try
    {
        await MigrateAsync(runtime, migrationPlanId, sequenceId, targetId);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("prepare-failed", StringComparison.Ordinal))
    {
        failed = true;
    }

    Require(failed, "Prepare failure must propagate.");
    var events = MigrationEvents(trace, migrationPlanId);
    RequireKinds(
        events,
        ExecutionTraceKind.MigrationStarted,
        ExecutionTraceKind.MigrationPlanned,
        ExecutionTraceKind.MigrationFailed);

    var terminal = events[^1];
    Require(terminal.TransactionId is null, "Pre-transfer failure must not invent a transaction id.");
    Require(terminal.TransportId == TraceTransportBackend.TransportId, "Pre-transfer failure after planning must retain transport metadata.");
    Require(terminal.FailureType == typeof(InvalidOperationException).FullName, "Pre-transfer failure must expose exception type.");
    Require(terminal.Elapsed is { } elapsed && elapsed >= TimeSpan.Zero, "Pre-transfer failure must expose elapsed time.");
    Require(sourceBackend.GetPosition(sequenceId) == 4, "Prepare failure must keep source state runnable.");
}

static async Task RunRollbackAsync(bool failAbort)
{
    var sourceId = new DeviceId(failAbort ? "gpu:rollback-fail-source" : "gpu:rollback-source");
    var targetId = new DeviceId(failAbort ? "gpu:rollback-fail-target" : "gpu:rollback-target");
    var sourceBackend = new TraceTransportBackend(sourceId);
    var targetBackend = new TraceTransportBackend(targetId)
    {
        FailImportAfterStaging = true,
        FailAbort = failAbort
    };
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId(failAbort ? "trace-rollback-failure" : "trace-rollback"));

    var migrationPlanId = Guid.NewGuid();
    Exception? observed = null;
    try
    {
        await MigrateAsync(runtime, migrationPlanId, sequenceId, targetId);
    }
    catch (Exception exception)
    {
        observed = exception;
    }

    Require(observed is not null, "Import failure must propagate.");
    if (failAbort)
    {
        Require(observed is AggregateException, "Rollback abort failure must aggregate the original and rollback failures.");
    }
    else
    {
        Require(observed is InvalidOperationException, "Successful rollback must preserve the original import exception.");
    }

    var events = MigrationEvents(trace, migrationPlanId);
    RequireKinds(
        events,
        ExecutionTraceKind.MigrationStarted,
        ExecutionTraceKind.MigrationPlanned,
        failAbort
            ? ExecutionTraceKind.MigrationRollbackFailed
            : ExecutionTraceKind.MigrationRolledBack);

    var terminal = events[^1];
    Require(terminal.TransactionId is { } transactionId && transactionId != Guid.Empty, "Rollback trace must expose backend transaction id.");
    Require(terminal.FailureType == typeof(InvalidOperationException).FullName, "Rollback trace must preserve the triggering failure type.");
    Require(terminal.RollbackFailureCount == (failAbort ? 1 : 0), "Rollback trace must report abort failure count.");
    Require(terminal.Elapsed is { } elapsed && elapsed >= TimeSpan.Zero, "Rollback trace must expose elapsed time including rollback.");
    Require(sourceBackend.GetPosition(sequenceId) == 4, "Rollback must preserve runnable source state.");

    if (!failAbort)
    {
        Require(!targetBackend.Contains(sequenceId), "Successful target abort must remove partially imported target state.");
    }
}

await RunCommittedAsync();
await RunPrepareFailureAsync();
await RunRollbackAsync(failAbort: false);
await RunRollbackAsync(failAbort: true);

Console.WriteLine("Fission migration observability specs passed.");

sealed class TraceMigrationTransfer : SequenceMigrationTransfer
{
    public TraceMigrationTransfer(
        Guid transactionId,
        SequenceId sequenceId,
        DeviceId sourceDevice,
        DeviceId targetDevice,
        int position,
        SequenceMigrationTransportPlan transportPlan)
        : base(transactionId, sequenceId, sourceDevice, targetDevice, transportPlan)
    {
        Position = position;
    }

    public int Position { get; }
}

sealed class TraceTransportBackend : IInferenceBackend, ISequenceMigrationTransportBackend
{
    public const string TransportId = "trace-host-v1";
    public const long EstimatedBytes = 128;
    private readonly ConcurrentDictionary<SequenceId, int> _states = new();
    private int _initialized;

    public TraceTransportBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => $"trace-transport:{Device}";
    public DeviceId Device { get; }
    public bool FailPrepare { get; init; }
    public bool FailImportAfterStaging { get; init; }
    public bool FailAbort { get; init; }

    public bool Contains(SequenceId sequenceId) => _states.ContainsKey(sequenceId);

    public int GetPosition(SequenceId sequenceId) =>
        _states.TryGetValue(sequenceId, out var position)
            ? position
            : throw new InvalidOperationException($"Sequence {sequenceId} has no state on {Device}.");

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Volatile.Write(ref _initialized, 1);
        return ValueTask.CompletedTask;
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var item = batch.Items[index];
            var position = checked((item.Position ?? 0) + item.Tokens.Length);
            _states[item.SequenceId] = position;
            results[index] = new BackendStepResult(item.SequenceId, position);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var item = batch.Items[index];
            var current = GetPosition(item.SequenceId);
            if (current != item.Position)
            {
                throw new InvalidOperationException("Decode position does not match backend state.");
            }

            var next = checked(current + 1);
            _states[item.SequenceId] = next;
            results[index] = new BackendStepResult(item.SequenceId, next);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public IReadOnlyList<SequenceMigrationTransportCapability> GetSequenceMigrationTransportCapabilities(
        DeviceId peerDevice) =>
        new[]
        {
            new SequenceMigrationTransportCapability(
                TransportId,
                SequenceMigrationTransportKind.HostStaging,
                MaxTransferBytes: 4096,
                EstimatedBandwidthBytesPerSecond: 1_000_000_000,
                EstimatedFixedLatency: TimeSpan.FromMilliseconds(1))
        };

    public ValueTask<long> EstimateSequenceMigrationBytesAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _ = GetPosition(sequenceId);
        return ValueTask.FromResult(EstimatedBytes);
    }

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default) =>
        ValueTask.FromException<SequenceMigrationTransfer>(
            new NotSupportedException("Trace backend requires an explicit transport plan."));

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (FailPrepare)
        {
            throw new InvalidOperationException("prepare-failed");
        }

        var position = GetPosition(sequenceId);
        return ValueTask.FromResult<SequenceMigrationTransfer>(
            new TraceMigrationTransfer(
                Guid.NewGuid(),
                sequenceId,
                Device,
                targetDevice,
                position,
                transportPlan));
    }

    public ValueTask ImportSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var staged = transfer as TraceMigrationTransfer ??
            throw new InvalidOperationException("Trace transfer type mismatch.");
        if (Device != staged.TargetDevice)
        {
            throw new InvalidOperationException("Import reached the wrong device.");
        }

        _states[staged.SequenceId] = staged.Position;
        if (FailImportAfterStaging)
        {
            throw new InvalidOperationException("import-failed-after-staging");
        }

        return ValueTask.CompletedTask;
    }

    public ValueTask CommitSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var staged = transfer as TraceMigrationTransfer ??
            throw new InvalidOperationException("Trace transfer type mismatch.");
        if (Device != staged.SourceDevice)
        {
            throw new InvalidOperationException("Commit reached the wrong device.");
        }

        _states.TryRemove(staged.SequenceId, out _);
        return ValueTask.CompletedTask;
    }

    public ValueTask AbortSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var staged = transfer as TraceMigrationTransfer ??
            throw new InvalidOperationException("Trace transfer type mismatch.");

        if (Device == staged.TargetDevice)
        {
            if (FailAbort)
            {
                throw new InvalidOperationException("abort-failed");
            }

            _states.TryRemove(staged.SequenceId, out _);
        }
        else if (Device == staged.SourceDevice)
        {
            _states[staged.SequenceId] = staged.Position;
        }
        else
        {
            throw new InvalidOperationException("Abort reached an unrelated device.");
        }

        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _states.TryRemove(sequenceId, out _);
        return ValueTask.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        _states.Clear();
        Volatile.Write(ref _initialized, 0);
        return ValueTask.CompletedTask;
    }

    private void EnsureInitialized()
    {
        if (Volatile.Read(ref _initialized) == 0)
        {
            throw new InvalidOperationException("Trace transport backend is not initialized.");
        }
    }
}
