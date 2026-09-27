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

var model = new ModelId("migration-model");
var sourceDevice = new DeviceId("gpu:source");
var targetDevice = new DeviceId("gpu:target");

// Device actor barrier: decode -> migrate -> decode must preserve queue order.
var barrierBackend = new MigrationBackend(sourceDevice);
await using (var barrierDevice = await ContinuousBatchExecutor.CreateAsync(
    barrierBackend,
    capacity: 16,
    maxBatchSize: 16))
{
    var sequence = SequenceId.New();
    await barrierDevice.SubmitPrefillAsync(
        new PrefillItem(sequence, model, new ReadOnlyMemory<int>(new[] { 1, 2, 3, 4 })));

    var firstDecode = barrierDevice.SubmitDecodeAsync(
        new DecodeItem(sequence, model, Position: 4)).AsTask();
    var migration = barrierDevice.MigrateSequenceAsync(sequence, targetDevice).AsTask();
    var secondDecode = barrierDevice.SubmitDecodeAsync(
        new DecodeItem(sequence, model, Position: 5)).AsTask();

    await Task.WhenAll(firstDecode, migration, secondDecode);

    Require(
        barrierBackend.Events.SequenceEqual(new[]
        {
            "prefill:4@gpu:source",
            "decode:5@gpu:source",
            "migrate:gpu:source->gpu:target",
            "decode:6@gpu:target"
        }),
        $"Migration barrier order is incorrect: [{string.Join(',', barrierBackend.Events)}].");
    Require(
        barrierBackend.Placements[sequence] == targetDevice,
        "Backend migration must publish the target placement before later decode work executes.");

    await barrierDevice.ReleaseSequenceAsync(sequence);
}

// Runtime commit ordering: backend migration completes before Device metadata changes.
var successBackend = new MigrationBackend(sourceDevice);
await using (var successDevice = await ContinuousBatchExecutor.CreateAsync(
    successBackend,
    capacity: 16,
    maxBatchSize: 8))
{
    using var runtime = new ExecutionPlanExecutor(
        successDevice,
        kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));
    var sequenceId = SequenceId.New();

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
                [sequenceId] = new ReadOnlyMemory<int>(new[] { 10, 11, 12, 13 })
            }));

    var beforeMigration = GetSequence(runtime, sequenceId);
    Require(beforeMigration.Device == sourceDevice, "New runtime sequence must start on the backend device.");
    var versionBeforeMigration = beforeMigration.Version;

    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[] { new MigrateKvExecutionStep(sequenceId, targetDevice) }),
        new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));

    var migrated = GetSequence(runtime, sequenceId);
    Require(migrated.Device == targetDevice, "Runtime metadata must commit the target device after backend migration succeeds.");
    Require(migrated.Version == versionBeforeMigration + 1, "Successful migration must advance the sequence version exactly once.");
    Require(successBackend.Placements[sequenceId] == targetDevice, "Backend and runtime placement must converge after migration.");

    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[] { new DecodeExecutionStep(sequenceId, 1) }),
        new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));

    Require(
        successBackend.Events.TakeLast(2).SequenceEqual(new[]
        {
            "migrate:gpu:source->gpu:target",
            "decode:5@gpu:target"
        }),
        "Decode after migration must observe the backend's migrated placement.");

    migrated.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), "Migrated sequence cleanup must release backend and runtime state.");
    Require(runtime.SequenceCount == 0, "Successful migration spec must leave no runtime sequences.");
}

// Failure ordering: a backend migration fault must not publish target metadata.
var failingBackend = new MigrationBackend(sourceDevice) { FailMigration = true };
var failingDevice = await ContinuousBatchExecutor.CreateAsync(
    failingBackend,
    capacity: 16,
    maxBatchSize: 8);
var failingRuntime = new ExecutionPlanExecutor(
    failingDevice,
    kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));
var failingSequenceId = SequenceId.New();

try
{
    await failingRuntime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new PrefillExecutionStep(failingSequenceId, model, 4, CompletesPrefill: true)
            }),
        new ExecutionBindings(
            new Dictionary<SequenceId, ReadOnlyMemory<int>>
            {
                [failingSequenceId] = new ReadOnlyMemory<int>(new[] { 20, 21, 22, 23 })
            }));

    var beforeFailure = GetSequence(failingRuntime, failingSequenceId);
    var originalDevice = beforeFailure.Device;
    var originalVersion = beforeFailure.Version;

    var failed = false;
    try
    {
        await failingRuntime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                0,
                new ExecutionStep[] { new MigrateKvExecutionStep(failingSequenceId, targetDevice) }),
            new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("migration-failed", StringComparison.Ordinal))
    {
        failed = true;
    }

    Require(failed, "Backend migration failure must propagate to the plan caller.");

    var afterFailure = GetSequence(failingRuntime, failingSequenceId);
    Require(afterFailure.Device == originalDevice, "Failed migration must leave runtime Device metadata unchanged.");
    Require(afterFailure.Version == originalVersion, "Failed migration must leave the runtime sequence version unchanged.");
    Require(
        failingBackend.Placements[failingSequenceId] == sourceDevice,
        "Failed migration must leave backend placement unchanged.");
}
finally
{
    failingRuntime.Dispose();
    try
    {
        await failingDevice.DisposeAsync();
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("migration-failed", StringComparison.Ordinal))
    {
        // Control-operation failures are fatal to the single-device pump today.
        // Disposal must preserve that original pump failure.
    }
}

Console.WriteLine("Fission backend migration transaction specs passed.");

sealed class MigrationBackend : IInferenceBackend
{
    private bool _initialized;

    public MigrationBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => "migration-spec";
    public DeviceId Device { get; }
    public bool FailMigration { get; init; }
    public Dictionary<SequenceId, int> SequenceStates { get; } = new();
    public Dictionary<SequenceId, DeviceId> Placements { get; } = new();
    public List<string> Events { get; } = new();

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
            var next = SequenceStates.TryGetValue(item.SequenceId, out var current)
                ? checked(current + item.Tokens.Length)
                : item.Tokens.Length;
            SequenceStates[item.SequenceId] = next;
            if (!Placements.ContainsKey(item.SequenceId))
            {
                Placements[item.SequenceId] = Device;
            }

            Events.Add($"prefill:{next}@{Placements[item.SequenceId]}");
            results[index] = new BackendStepResult(item.SequenceId, next);
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
            if (!SequenceStates.TryGetValue(item.SequenceId, out var current))
            {
                throw new InvalidOperationException($"Missing backend state for {item.SequenceId}.");
            }

            if (item.Position != current)
            {
                throw new InvalidOperationException(
                    $"Decode position {item.Position} does not match backend state {current} for {item.SequenceId}.");
            }

            var next = checked(current + 1);
            SequenceStates[item.SequenceId] = next;
            Events.Add($"decode:{next}@{Placements[item.SequenceId]}");
            results[index] = new BackendStepResult(item.SequenceId, next);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();

        if (!SequenceStates.ContainsKey(sequenceId) || !Placements.TryGetValue(sequenceId, out var source))
        {
            throw new InvalidOperationException($"Missing backend state for migration sequence {sequenceId}.");
        }

        if (FailMigration)
        {
            throw new InvalidOperationException("migration-failed");
        }

        Placements[sequenceId] = targetDevice;
        Events.Add($"migrate:{source}->{targetDevice}");
        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        SequenceStates.Remove(sequenceId);
        Placements.Remove(sequenceId);
        Events.Add("release-sequence");
        return ValueTask.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        SequenceStates.Clear();
        Placements.Clear();
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Migration backend has not been initialized.");
        }
    }
}
