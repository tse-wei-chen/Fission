using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static SequenceProcess GetSequence(ExecutionPlanExecutor runtime, SequenceId sequenceId)
{
    if (!runtime.TryGetSequence(sequenceId, out var sequence) || sequence is null)
    {
        throw new InvalidOperationException($"Sequence {sequenceId} is missing.");
    }

    return sequence;
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

static async Task MigrateAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId,
    DeviceId targetDevice)
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new MigrateKvExecutionStep(sequenceId, targetDevice)
            }),
        EmptyBindings());
}

static async Task DecodeOneAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId)
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new DecodeExecutionStep(sequenceId, 1)
            }),
        EmptyBindings());
}

static async Task RunSuccessAsync()
{
    var sourceId = new DeviceId("gpu:source");
    var targetId = new DeviceId("gpu:target");
    var model = new ModelId("migration-protocol-success");
    var fabric = new ProtocolFabric(sourceId, targetId);
    var sourceBackend = new ProtocolBackend(sourceId, fabric);
    var targetBackend = new ProtocolBackend(targetId, fabric);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(
        sourceBackend,
        capacity: 8,
        maxBatchSize: 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(
        targetBackend,
        capacity: 8,
        maxBatchSize: 8);

    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, model);
    var before = GetSequence(runtime, sequenceId);
    var versionBefore = before.Version;

    await MigrateAsync(runtime, sequenceId, targetId);

    var migrated = GetSequence(runtime, sequenceId);
    Require(migrated.Device == targetId, "Successful protocol migration must publish target placement.");
    Require(migrated.Version == versionBefore + 1, "Successful protocol migration must advance sequence version once.");
    Require(!fabric.Contains(sourceId, sequenceId), "Successful commit must remove source backend state.");
    Require(fabric.GetPosition(targetId, sequenceId) == 4, "Successful import must make target state runnable.");

    await DecodeOneAsync(runtime, sequenceId);
    Require(fabric.GetPosition(targetId, sequenceId) == 5, "Decode after migration must execute against target state.");
    Require(
        fabric.Events.SequenceEqual(new[]
        {
            "prefill:4@gpu:source",
            "prepare:gpu:source->gpu:target:4",
            "import:gpu:source->gpu:target:4",
            "commit:gpu:source->gpu:target:4",
            "decode:5@gpu:target"
        }),
        $"Successful migration protocol order is incorrect: [{string.Join(',', fabric.Events)}].");

    migrated.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), "Successful migrated sequence must be releasable on target actor.");
}

static async Task RunImportFailureAsync()
{
    var sourceId = new DeviceId("gpu:source");
    var targetId = new DeviceId("gpu:target");
    var model = new ModelId("migration-protocol-import-failure");
    var fabric = new ProtocolFabric(sourceId, targetId);
    var sourceBackend = new ProtocolBackend(sourceId, fabric);
    var targetBackend = new ProtocolBackend(targetId, fabric)
    {
        FailImportAfterStaging = true
    };

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(
        sourceBackend,
        capacity: 8,
        maxBatchSize: 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(
        targetBackend,
        capacity: 8,
        maxBatchSize: 8);

    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, model);
    var before = GetSequence(runtime, sequenceId);
    var versionBefore = before.Version;

    var failed = false;
    try
    {
        await MigrateAsync(runtime, sequenceId, targetId);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("import-failed", StringComparison.Ordinal))
    {
        failed = true;
    }

    Require(failed, "Target import failure must propagate to the plan caller.");
    var after = GetSequence(runtime, sequenceId);
    Require(after.Device == sourceId, "Import failure must not publish target runtime placement.");
    Require(after.Version == versionBefore, "Import failure must not advance runtime sequence version.");
    Require(fabric.GetPosition(sourceId, sequenceId) == 4, "Import rollback must keep source state intact.");
    Require(!fabric.Contains(targetId, sequenceId), "Import rollback must remove partially staged target state.");

    await DecodeOneAsync(runtime, sequenceId);
    Require(fabric.GetPosition(sourceId, sequenceId) == 5, "Source actor must remain usable after target control failure and rollback.");
    Require(
        fabric.Events.SequenceEqual(new[]
        {
            "prefill:4@gpu:source",
            "prepare:gpu:source->gpu:target:4",
            "import:gpu:source->gpu:target:4",
            "abort-target:gpu:source->gpu:target:4",
            "abort-source:gpu:source->gpu:target:4",
            "decode:5@gpu:source"
        }),
        $"Import rollback order is incorrect: [{string.Join(',', fabric.Events)}].");

    after.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), "Rolled-back source sequence must remain releasable.");
}

static async Task RunCommitFailureAsync()
{
    var sourceId = new DeviceId("gpu:source");
    var targetId = new DeviceId("gpu:target");
    var model = new ModelId("migration-protocol-commit-failure");
    var fabric = new ProtocolFabric(sourceId, targetId);
    var sourceBackend = new ProtocolBackend(sourceId, fabric)
    {
        FailCommitAfterSourceRemoval = true
    };
    var targetBackend = new ProtocolBackend(targetId, fabric);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(
        sourceBackend,
        capacity: 8,
        maxBatchSize: 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(
        targetBackend,
        capacity: 8,
        maxBatchSize: 8);

    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, model);
    var before = GetSequence(runtime, sequenceId);
    var versionBefore = before.Version;

    var failed = false;
    try
    {
        await MigrateAsync(runtime, sequenceId, targetId);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("commit-failed", StringComparison.Ordinal))
    {
        failed = true;
    }

    Require(failed, "Source commit failure must propagate to the plan caller.");
    var after = GetSequence(runtime, sequenceId);
    Require(after.Device == sourceId, "Commit failure must not publish target runtime placement.");
    Require(after.Version == versionBefore, "Commit failure must not advance runtime sequence version.");
    Require(fabric.GetPosition(sourceId, sequenceId) == 4, "Source abort must restore state removed by a partial commit.");
    Require(!fabric.Contains(targetId, sequenceId), "Commit rollback must remove imported target state.");

    await DecodeOneAsync(runtime, sequenceId);
    Require(fabric.GetPosition(sourceId, sequenceId) == 5, "Source actor must continue decoding after failed commit rollback.");
    Require(
        fabric.Events.SequenceEqual(new[]
        {
            "prefill:4@gpu:source",
            "prepare:gpu:source->gpu:target:4",
            "import:gpu:source->gpu:target:4",
            "commit:gpu:source->gpu:target:4",
            "abort-target:gpu:source->gpu:target:4",
            "abort-source:gpu:source->gpu:target:4",
            "decode:5@gpu:source"
        }),
        $"Commit rollback order is incorrect: [{string.Join(',', fabric.Events)}].");

    after.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), "Commit-rolled-back source sequence must remain releasable.");
}

await RunSuccessAsync();
await RunImportFailureAsync();
await RunCommitFailureAsync();

Console.WriteLine("Fission transactional migration protocol specs passed.");

enum MigrationFailurePoint
{
    None,
    ImportAfterStaging,
    CommitAfterSourceRemoval
}

sealed class ProtocolTransfer : SequenceMigrationTransfer
{
    public ProtocolTransfer(
        Guid transactionId,
        SequenceId sequenceId,
        DeviceId sourceDevice,
        DeviceId targetDevice,
        int position)
        : base(transactionId, sequenceId, sourceDevice, targetDevice)
    {
        Position = position;
    }

    public int Position { get; }
}

sealed class ProtocolFabric
{
    private readonly object _gate = new();
    private readonly Dictionary<DeviceId, Dictionary<SequenceId, int>> _states;
    private readonly List<string> _events = new();

    public ProtocolFabric(params DeviceId[] devices)
    {
        _states = devices.ToDictionary(
            static device => device,
            static _ => new Dictionary<SequenceId, int>());
    }

    public IReadOnlyList<string> Events
    {
        get
        {
            lock (_gate)
            {
                return _events.ToArray();
            }
        }
    }

    public bool Contains(DeviceId device, SequenceId sequenceId)
    {
        lock (_gate)
        {
            return _states[device].ContainsKey(sequenceId);
        }
    }

    public int GetPosition(DeviceId device, SequenceId sequenceId)
    {
        lock (_gate)
        {
            if (_states[device].TryGetValue(sequenceId, out var position))
            {
                return position;
            }

            throw new InvalidOperationException($"No state for {sequenceId} on {device}.");
        }
    }

    public void SetPosition(DeviceId device, SequenceId sequenceId, int position)
    {
        lock (_gate)
        {
            _states[device][sequenceId] = position;
        }
    }

    public bool Remove(DeviceId device, SequenceId sequenceId)
    {
        lock (_gate)
        {
            return _states[device].Remove(sequenceId);
        }
    }

    public void Record(string message)
    {
        lock (_gate)
        {
            _events.Add(message);
        }
    }
}

sealed class ProtocolBackend : IInferenceBackend, ISequenceMigrationBackend
{
    private readonly ProtocolFabric _fabric;
    private bool _initialized;

    public ProtocolBackend(DeviceId device, ProtocolFabric fabric)
    {
        Device = device;
        _fabric = fabric;
    }

    public string Name => $"migration-protocol:{Device}";
    public DeviceId Device { get; }
    public bool FailImportAfterStaging { get; init; }
    public bool FailCommitAfterSourceRemoval { get; init; }

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _initialized = true;
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
            var position = item.Tokens.Length;
            _fabric.SetPosition(Device, item.SequenceId, position);
            _fabric.Record($"prefill:{position}@{Device}");
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
            var current = _fabric.GetPosition(Device, item.SequenceId);
            if (current != item.Position)
            {
                throw new InvalidOperationException(
                    $"Decode position {item.Position} does not match backend position {current} on {Device}.");
            }

            var next = checked(current + 1);
            _fabric.SetPosition(Device, item.SequenceId, next);
            _fabric.Record($"decode:{next}@{Device}");
            results[index] = new BackendStepResult(item.SequenceId, next);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default) =>
        throw new InvalidOperationException(
            "legacy-migration-hook-used-by-transactional-backend");

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var position = _fabric.GetPosition(Device, sequenceId);
        var transfer = new ProtocolTransfer(
            Guid.NewGuid(),
            sequenceId,
            Device,
            targetDevice,
            position);
        _fabric.Record($"prepare:{Device}->{targetDevice}:{position}");
        return ValueTask.FromResult<SequenceMigrationTransfer>(transfer);
    }

    public ValueTask ImportSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var protocol = RequireTransfer(transfer);
        if (protocol.TargetDevice != Device)
        {
            throw new InvalidOperationException(
                $"Transfer target {protocol.TargetDevice} does not match backend device {Device}.");
        }

        _fabric.SetPosition(Device, protocol.SequenceId, protocol.Position);
        _fabric.Record($"import:{protocol.SourceDevice}->{Device}:{protocol.Position}");

        if (FailImportAfterStaging)
        {
            throw new InvalidOperationException("import-failed-after-target-staging");
        }

        return ValueTask.CompletedTask;
    }

    public ValueTask CommitSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var protocol = RequireTransfer(transfer);
        if (protocol.SourceDevice != Device)
        {
            throw new InvalidOperationException(
                $"Transfer source {protocol.SourceDevice} does not match backend device {Device}.");
        }

        _fabric.Remove(Device, protocol.SequenceId);
        _fabric.Record($"commit:{Device}->{protocol.TargetDevice}:{protocol.Position}");

        if (FailCommitAfterSourceRemoval)
        {
            throw new InvalidOperationException("commit-failed-after-source-removal");
        }

        return ValueTask.CompletedTask;
    }

    public ValueTask AbortSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var protocol = RequireTransfer(transfer);

        if (Device == protocol.TargetDevice)
        {
            _fabric.Remove(Device, protocol.SequenceId);
            _fabric.Record($"abort-target:{protocol.SourceDevice}->{Device}:{protocol.Position}");
            return ValueTask.CompletedTask;
        }

        if (Device == protocol.SourceDevice)
        {
            _fabric.SetPosition(Device, protocol.SequenceId, protocol.Position);
            _fabric.Record($"abort-source:{Device}->{protocol.TargetDevice}:{protocol.Position}");
            return ValueTask.CompletedTask;
        }

        throw new InvalidOperationException(
            $"Backend {Device} does not participate in migration {protocol.TransactionId}.");
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _fabric.Remove(Device, sequenceId);
        return ValueTask.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private static ProtocolTransfer RequireTransfer(SequenceMigrationTransfer transfer) =>
        transfer as ProtocolTransfer ??
        throw new InvalidOperationException(
            $"Unsupported migration transfer type {transfer.GetType().Name}.");

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException($"Backend {Device} is not initialized.");
        }
    }
}
