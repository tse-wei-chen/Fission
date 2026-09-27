using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Tracing;

static void Require(
    [DoesNotReturnIf(false)] bool condition,
    string message)
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
    DeviceId targetDevice,
    CancellationToken cancellationToken = default) =>
    runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            planId,
            0,
            new ExecutionStep[]
            {
                new MigrateKvExecutionStep(sequenceId, targetDevice)
            }),
        EmptyBindings(),
        cancellationToken).AsTask();

static ExecutionTraceEvent TerminalMigrationEvent(
    InMemoryExecutionTraceSink trace,
    Guid planId) =>
    trace.Snapshot()
        .Select(static item => item.Event)
        .Where(item => item.PlanId == planId && item.Kind is
            ExecutionTraceKind.MigrationCommitted or
            ExecutionTraceKind.MigrationRolledBack or
            ExecutionTraceKind.MigrationRollbackFailed or
            ExecutionTraceKind.MigrationFailed)
        .Single();

static async Task RunPrepareTimeoutAsync()
{
    var sourceId = new DeviceId("gpu:timeout-prepare-source");
    var targetId = new DeviceId("gpu:timeout-prepare-target");
    var source = new TimeoutBackend(sourceId) { PrepareDelay = TimeSpan.FromMilliseconds(250) };
    var target = new TimeoutBackend(targetId);
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(source, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(target, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4),
        new SequenceMigrationTimeoutPolicy
        {
            PrepareTimeout = TimeSpan.FromMilliseconds(25)
        });

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("prepare-timeout"));

    var planId = Guid.NewGuid();
    SequenceMigrationTimeoutException? observed = null;
    try
    {
        await MigrateAsync(runtime, planId, sequenceId, targetId);
    }
    catch (SequenceMigrationTimeoutException exception)
    {
        observed = exception;
    }

    Require(observed is not null, "Prepare phase must surface a migration timeout exception.");
    Require(observed.Phase == SequenceMigrationPhase.Prepare, "Prepare timeout must identify the prepare phase.");
    Require(source.GetPosition(sequenceId) == 4, "Timed-out prepare must leave source state runnable.");
    Require(!target.Contains(sequenceId), "Timed-out prepare must not create target state.");
    Require(runtime.ActiveMigrations == 0 && runtime.MigrationInflightBytes == 0, "Prepare timeout must release migration admission.");

    var terminal = TerminalMigrationEvent(trace, planId);
    Require(terminal.Kind == ExecutionTraceKind.MigrationFailed, "Prepare timeout occurs before a transfer token and must trace as MigrationFailed.");
    Require(terminal.MigrationPhase == SequenceMigrationPhase.Prepare, "Trace must preserve the timed-out phase.");
    Require(terminal.MigrationFailureClass == SequenceMigrationFailureClass.TimedOut, "Trace must classify deadline expiry as TimedOut.");
    Require(terminal.MigrationHealthImpact == SequenceMigrationHealthImpact.SourceSuspect, "Prepare timeout must conservatively flag the source.");
    Require(terminal.MigrationTimeout == TimeSpan.FromMilliseconds(25), "Trace must preserve the configured deadline.");
    Require(terminal.FailureType == typeof(SequenceMigrationTimeoutException).FullName, "Trace must expose the timeout exception type.");
}

static async Task RunImportTimeoutAndPersistenceAsync()
{
    var sourceId = new DeviceId("gpu:timeout-import-source");
    var targetId = new DeviceId("gpu:timeout-import-target");
    var source = new TimeoutBackend(sourceId);
    var target = new TimeoutBackend(targetId) { ImportDelay = TimeSpan.FromMilliseconds(250) };
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(source, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(target, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4),
        new SequenceMigrationTimeoutPolicy
        {
            ImportTimeout = TimeSpan.FromMilliseconds(25),
            RollbackTimeout = TimeSpan.FromSeconds(1)
        });

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("import-timeout"));

    var planId = Guid.NewGuid();
    SequenceMigrationTimeoutException? observed = null;
    try
    {
        await MigrateAsync(runtime, planId, sequenceId, targetId);
    }
    catch (SequenceMigrationTimeoutException exception)
    {
        observed = exception;
    }

    Require(observed is not null && observed.Phase == SequenceMigrationPhase.Import, "Import phase must surface a classified timeout.");
    Require(source.GetPosition(sequenceId) == 4, "Import timeout rollback must preserve source state.");
    Require(!target.Contains(sequenceId), "Import timeout rollback must remove staged target state.");
    Require(target.AbortCalls == 1 && source.AbortCalls == 1, "Import timeout must abort both target and source.");

    var terminal = TerminalMigrationEvent(trace, planId);
    Require(terminal.Kind == ExecutionTraceKind.MigrationRolledBack, "Import timeout with successful aborts must trace as rolled back.");
    Require(terminal.TransactionId is { } transactionId && transactionId != Guid.Empty, "Import timeout trace must retain transaction identity.");
    Require(terminal.MigrationPhase == SequenceMigrationPhase.Import, "Import timeout trace must preserve failure phase.");
    Require(terminal.MigrationFailureClass == SequenceMigrationFailureClass.TimedOut, "Import timeout trace must preserve timeout classification.");
    Require(terminal.MigrationHealthImpact == SequenceMigrationHealthImpact.TargetSuspect, "Import timeout must conservatively flag the target.");
    Require(terminal.MigrationTimeout == TimeSpan.FromMilliseconds(25), "Import timeout trace must preserve deadline duration.");

    await using var stream = new MemoryStream();
    await ExecutionTraceJsonLines.WriteAsync(stream, trace.Snapshot());
    stream.Position = 0;
    var restored = await ExecutionTraceJsonLines.ReadAsync(stream);
    var restoredTerminal = restored
        .Select(static item => item.Event)
        .Single(item => item.PlanId == planId && item.Kind == ExecutionTraceKind.MigrationRolledBack);
    Require(restoredTerminal.MigrationPhase == terminal.MigrationPhase, "Trace codec must preserve migration failure phase.");
    Require(restoredTerminal.MigrationFailureClass == terminal.MigrationFailureClass, "Trace codec must preserve migration failure class.");
    Require(restoredTerminal.MigrationHealthImpact == terminal.MigrationHealthImpact, "Trace codec must preserve migration health impact.");
    Require(restoredTerminal.MigrationTimeout == terminal.MigrationTimeout, "Trace codec must preserve migration timeout duration.");
}

static async Task RunCallerCancellationAsync()
{
    var sourceId = new DeviceId("gpu:cancel-source");
    var targetId = new DeviceId("gpu:cancel-target");
    var source = new TimeoutBackend(sourceId) { PrepareDelay = TimeSpan.FromMilliseconds(250) };
    var target = new TimeoutBackend(targetId);
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(source, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(target, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4),
        new SequenceMigrationTimeoutPolicy
        {
            PrepareTimeout = TimeSpan.FromSeconds(5)
        });

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("caller-cancel"));

    using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(25));
    var planId = Guid.NewGuid();
    var canceled = false;
    try
    {
        await MigrateAsync(runtime, planId, sequenceId, targetId, cancellation.Token);
    }
    catch (OperationCanceledException)
    {
        canceled = true;
    }

    Require(canceled, "Caller cancellation must propagate as cancellation rather than timeout.");
    var terminal = TerminalMigrationEvent(trace, planId);
    Require(terminal.MigrationFailureClass == SequenceMigrationFailureClass.CallerCanceled, "Caller cancellation must have its own failure class.");
    Require(terminal.MigrationPhase == SequenceMigrationPhase.Prepare, "Caller cancellation must preserve the active phase.");
    Require(terminal.MigrationHealthImpact == SequenceMigrationHealthImpact.None, "Caller cancellation must not mark a device suspect.");
    Require(terminal.MigrationTimeout is null, "Caller cancellation must not invent a timeout duration.");
    Require(source.GetPosition(sequenceId) == 4 && !target.Contains(sequenceId), "Caller cancellation before transfer creation must leave source placement usable.");
}

static async Task RunBackendFailureClassificationAsync()
{
    var sourceId = new DeviceId("gpu:backend-failure-source");
    var targetId = new DeviceId("gpu:backend-failure-target");
    var source = new TimeoutBackend(sourceId);
    var target = new TimeoutBackend(targetId) { FailImportAfterStaging = true };
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(source, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(target, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("backend-failure"));

    var planId = Guid.NewGuid();
    var failed = false;
    try
    {
        await MigrateAsync(runtime, planId, sequenceId, targetId);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("import-failed", StringComparison.Ordinal))
    {
        failed = true;
    }

    Require(failed, "Backend import failure must propagate.");
    var terminal = TerminalMigrationEvent(trace, planId);
    Require(terminal.Kind == ExecutionTraceKind.MigrationRolledBack, "Backend import failure must roll back.");
    Require(terminal.MigrationPhase == SequenceMigrationPhase.Import, "Backend failure trace must identify import.");
    Require(terminal.MigrationFailureClass == SequenceMigrationFailureClass.BackendFailure, "Backend exception must classify separately from timeout/cancellation.");
    Require(terminal.MigrationHealthImpact == SequenceMigrationHealthImpact.TargetSuspect, "Import backend failure must flag target health evidence.");
    Require(terminal.MigrationTimeout is null, "Ordinary backend failure must not report a timeout.");
}

static async Task RunRollbackTimeoutAsync()
{
    var sourceId = new DeviceId("gpu:rollback-timeout-source");
    var targetId = new DeviceId("gpu:rollback-timeout-target");
    var source = new TimeoutBackend(sourceId);
    var target = new TimeoutBackend(targetId)
    {
        FailImportAfterStaging = true,
        AbortDelay = TimeSpan.FromMilliseconds(250)
    };
    var trace = new InMemoryExecutionTraceSink();
    using var admission = new SequenceMigrationAdmissionController(4096, 1);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(source, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(target, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        trace,
        new KvPagePool(16, 4),
        new SequenceMigrationTimeoutPolicy
        {
            RollbackTimeout = TimeSpan.FromMilliseconds(25)
        });

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("rollback-timeout"));

    var planId = Guid.NewGuid();
    AggregateException? observed = null;
    try
    {
        await MigrateAsync(runtime, planId, sequenceId, targetId);
    }
    catch (AggregateException exception)
    {
        observed = exception;
    }

    Require(observed is not null, "Rollback timeout must aggregate with the triggering migration failure.");
    Require(observed.InnerExceptions.Any(static item => item is SequenceMigrationTimeoutException timeout && timeout.Phase == SequenceMigrationPhase.TargetRollback), "Aggregate must expose target rollback timeout.");
    Require(source.GetPosition(sequenceId) == 4, "Source rollback must still run after target rollback timeout.");

    var terminal = TerminalMigrationEvent(trace, planId);
    Require(terminal.Kind == ExecutionTraceKind.MigrationRollbackFailed, "Rollback timeout must trace as rollback failure.");
    Require(terminal.RollbackFailureCount == 1, "Rollback timeout must count as one rollback failure.");
    Require(terminal.MigrationFailureClass == SequenceMigrationFailureClass.BackendFailure, "Terminal classification must preserve the triggering import failure class.");
    Require(terminal.MigrationHealthImpact == SequenceMigrationHealthImpact.TargetSuspect, "Rollback timeout must contribute target health evidence.");
    Require(runtime.ActiveMigrations == 0 && runtime.MigrationInflightBytes == 0, "Admission must release after rollback timeout handling completes.");
}

await RunPrepareTimeoutAsync();
await RunImportTimeoutAndPersistenceAsync();
await RunCallerCancellationAsync();
await RunBackendFailureClassificationAsync();
await RunRollbackTimeoutAsync();

Console.WriteLine("Fission migration timeout and health-classification specs passed.");

sealed class TimeoutTransfer : SequenceMigrationTransfer
{
    public TimeoutTransfer(
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

sealed class TimeoutBackend : IInferenceBackend, ISequenceMigrationTransportBackend
{
    private const string TransportId = "timeout-host-v1";
    private readonly ConcurrentDictionary<SequenceId, int> _states = new();
    private int _initialized;
    private int _abortCalls;

    public TimeoutBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => $"timeout-backend:{Device}";
    public DeviceId Device { get; }
    public TimeSpan PrepareDelay { get; init; }
    public TimeSpan ImportDelay { get; init; }
    public TimeSpan CommitDelay { get; init; }
    public TimeSpan AbortDelay { get; init; }
    public bool FailImportAfterStaging { get; init; }
    public int AbortCalls => Volatile.Read(ref _abortCalls);

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
        var result = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var item = batch.Items[index];
            var position = checked((item.Position ?? 0) + item.Tokens.Length);
            _states[item.SequenceId] = position;
            result[index] = new BackendStepResult(item.SequenceId, position);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(result);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        var result = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var item = batch.Items[index];
            var current = GetPosition(item.SequenceId);
            var next = checked(current + 1);
            _states[item.SequenceId] = next;
            result[index] = new BackendStepResult(item.SequenceId, next);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(result);
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
        return ValueTask.FromResult(128L);
    }

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default) =>
        ValueTask.FromException<SequenceMigrationTransfer>(
            new NotSupportedException("Timeout backend requires an explicit transport plan."));

    public async ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan,
        CancellationToken cancellationToken = default)
    {
        await DelayAsync(PrepareDelay, cancellationToken);
        return new TimeoutTransfer(
            Guid.NewGuid(),
            sequenceId,
            Device,
            targetDevice,
            GetPosition(sequenceId),
            transportPlan);
    }

    public async ValueTask ImportSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        var typed = RequireTransfer(transfer);
        _states[typed.SequenceId] = typed.Position;
        await DelayAsync(ImportDelay, cancellationToken);
        if (FailImportAfterStaging)
        {
            throw new InvalidOperationException("import-failed-after-staging");
        }
    }

    public async ValueTask CommitSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        var typed = RequireTransfer(transfer);
        await DelayAsync(CommitDelay, cancellationToken);
        _states.TryRemove(typed.SequenceId, out _);
    }

    public async ValueTask AbortSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        var typed = RequireTransfer(transfer);
        Interlocked.Increment(ref _abortCalls);
        await DelayAsync(AbortDelay, cancellationToken);

        if (Device == typed.TargetDevice)
        {
            _states.TryRemove(typed.SequenceId, out _);
        }
        else if (Device == typed.SourceDevice)
        {
            _states[typed.SequenceId] = typed.Position;
        }
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
        return ValueTask.CompletedTask;
    }

    private TimeoutTransfer RequireTransfer(SequenceMigrationTransfer transfer) =>
        transfer as TimeoutTransfer ??
        throw new InvalidOperationException("Timeout transfer type mismatch.");

    private static async ValueTask DelayAsync(
        TimeSpan delay,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (delay > TimeSpan.Zero)
        {
            await Task.Delay(delay, cancellationToken);
        }
    }

    private void EnsureInitialized()
    {
        if (Volatile.Read(ref _initialized) == 0)
        {
            throw new InvalidOperationException("Backend is not initialized.");
        }
    }
}
