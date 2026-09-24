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

var model = new ModelId("transaction-model");

// Device actor barrier spec: decode -> snapshot -> decode must not be reordered
// into one decode micro-batch ahead of the snapshot control operation.
var barrierBackend = new TransactionalBackend(new DeviceId("cpu:barrier"));
await using (var barrierDevice = await ContinuousBatchExecutor.CreateAsync(
    barrierBackend,
    capacity: 16,
    maxBatchSize: 16))
{
    var sequence = SequenceId.New();
    await barrierDevice.SubmitPrefillAsync(
        new PrefillItem(sequence, model, new ReadOnlyMemory<int>(new[] { 1, 2, 3, 4 })));

    var snapshot = KvSnapshotId.New();
    var firstDecode = barrierDevice.SubmitDecodeAsync(
        new DecodeItem(sequence, model, Position: 4)).AsTask();
    var snapshotBarrier = barrierDevice.SnapshotSequenceAsync(sequence, snapshot).AsTask();
    var secondDecode = barrierDevice.SubmitDecodeAsync(
        new DecodeItem(sequence, model, Position: 5)).AsTask();

    await Task.WhenAll(firstDecode, snapshotBarrier, secondDecode);

    Require(barrierBackend.SnapshotStates[snapshot] == 5, "Snapshot barrier must observe state after only the first decode.");
    Require(barrierBackend.SequenceStates[sequence] == 6, "Second decode must execute after the snapshot barrier.");
    Require(
        barrierBackend.Events.SequenceEqual(new[]
        {
            "prefill:4",
            "decode:5",
            "snapshot:5",
            "decode:6"
        }),
        $"Device actor transaction order is incorrect: [{string.Join(',', barrierBackend.Events)}].");

    await barrierDevice.ReleaseSnapshotAsync(snapshot);
    await barrierDevice.ReleaseSequenceAsync(sequence);
    Require(barrierBackend.SequenceStates.Count == 0 && barrierBackend.SnapshotStates.Count == 0, "Barrier backend cleanup must release sequence and snapshot state.");
}

// Runtime transaction spec: backend state must follow metadata snapshot/restore/fork.
var backend = new TransactionalBackend(new DeviceId("cpu:transactions"));
var kvPool = new KvPagePool(capacity: 16, tokensPerPage: 4);
await using var device = await ContinuousBatchExecutor.CreateAsync(
    backend,
    capacity: 32,
    maxBatchSize: 8);
using var runtime = new ExecutionPlanExecutor(device, kvPagePool: kvPool);
var parentId = SequenceId.New();
var emptyBindings = new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>());

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[]
        {
            new PrefillExecutionStep(parentId, model, 4)
        }),
    new ExecutionBindings(
        new Dictionary<SequenceId, ReadOnlyMemory<int>>
        {
            [parentId] = new ReadOnlyMemory<int>(new[] { 10, 11, 12, 13 })
        }));
Require(backend.SequenceStates[parentId] == 4, "Prefill must materialize backend sequence state.");

var snapshotResult = await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new SnapshotKvExecutionStep(parentId) }),
    emptyBindings);
var snapshotId = snapshotResult.Snapshots.Single();
Require(backend.SnapshotStates[snapshotId] == 4, "Backend snapshot must capture the same logical position as metadata.");
Require(runtime.SnapshotCount == 1, "Runtime must own one metadata snapshot.");

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new DecodeExecutionStep(parentId, 2) }),
    emptyBindings);
Require(GetSequence(runtime, parentId).Position == 6 && backend.SequenceStates[parentId] == 6, "Backend and metadata must advance together before restore.");

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new RestoreKvExecutionStep(parentId, snapshotId) }),
    emptyBindings);
Require(GetSequence(runtime, parentId).Position == 4, "Metadata restore must rewind token position.");
Require(backend.SequenceStates[parentId] == 4, "Backend restore must rewind physical/model state to the snapshot.");

var forkResult = await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new ForkKvExecutionStep(parentId, 2) }),
    emptyBindings);
var branchAId = forkResult.Forks.Single().Branches[0];
var branchBId = forkResult.Forks.Single().Branches[1];
Require(backend.SequenceStates[branchAId] == 4 && backend.SequenceStates[branchBId] == 4, "Backend fork must clone/share the parent logical state for every branch.");

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new DecodeExecutionStep(branchAId, 1) }),
    emptyBindings);
Require(backend.SequenceStates[parentId] == 4, "Parent backend state must remain unchanged when a branch diverges.");
Require(backend.SequenceStates[branchAId] == 5, "Decoded branch must advance independently.");
Require(backend.SequenceStates[branchBId] == 4, "Sibling branch backend state must remain shared/logically unchanged.");
Require(GetSequence(runtime, branchAId).Position == 5 && GetSequence(runtime, branchBId).Position == 4, "Metadata branches must mirror backend divergence.");

Require(await runtime.ReleaseSnapshotAsync(snapshotId), "Explicit snapshot release must succeed.");
Require(runtime.SnapshotCount == 0 && backend.SnapshotStates.Count == 0, "Snapshot release must converge backend and metadata ownership.");

foreach (var sequenceId in new[] { parentId, branchAId, branchBId })
{
    var sequence = GetSequence(runtime, sequenceId);
    sequence.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), $"Terminal sequence {sequenceId} must release successfully.");
}

Require(runtime.SequenceCount == 0, "All runtime sequence metadata must be released.");
Require(backend.SequenceStates.Count == 0, "All backend sequence state must be released.");
Require(kvPool.AllocatedPages == 0, "All metadata KV leases must return to the pool after branch cleanup.");

Console.WriteLine(
    $"Fission backend transaction specs passed: events={backend.Events.Count}, sequences={runtime.SequenceCount}, " +
    $"snapshots={runtime.SnapshotCount}, kv={kvPool.AllocatedPages}/{kvPool.Capacity}.");

sealed class TransactionalBackend : IInferenceBackend
{
    private bool _initialized;

    public TransactionalBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => "transactional-spec";
    public DeviceId Device { get; }
    public Dictionary<SequenceId, int> SequenceStates { get; } = new();
    public Dictionary<KvSnapshotId, int> SnapshotStates { get; } = new();
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
            var state = SequenceStates.TryGetValue(item.SequenceId, out var current)
                ? checked(current + item.Tokens.Length)
                : item.Tokens.Length;
            SequenceStates[item.SequenceId] = state;
            Events.Add($"prefill:{state}");
            results[index] = new BackendStepResult(item.SequenceId, state);
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
                throw new InvalidOperationException($"Missing backend state for decode sequence {item.SequenceId}.");
            }

            var state = checked(current + 1);
            SequenceStates[item.SequenceId] = state;
            Events.Add($"decode:{state}");
            results[index] = new BackendStepResult(item.SequenceId, state);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (!SequenceStates.TryGetValue(sequenceId, out var state))
        {
            throw new InvalidOperationException($"Missing backend state for snapshot sequence {sequenceId}.");
        }

        SnapshotStates.Add(snapshotId, state);
        Events.Add($"snapshot:{state}");
        return ValueTask.CompletedTask;
    }

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (!SequenceStates.TryGetValue(parentSequenceId, out var state))
        {
            throw new InvalidOperationException($"Missing backend state for fork parent {parentSequenceId}.");
        }

        foreach (var branchId in branchSequenceIds)
        {
            SequenceStates.Add(branchId, state);
        }

        Events.Add($"fork:{state}:{branchSequenceIds.Count}");
        return ValueTask.CompletedTask;
    }

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (!SnapshotStates.TryGetValue(snapshotId, out var state))
        {
            throw new InvalidOperationException($"Missing backend snapshot state {snapshotId}.");
        }

        SequenceStates[sequenceId] = state;
        Events.Add($"restore:{state}");
        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (!SnapshotStates.Remove(snapshotId))
        {
            throw new InvalidOperationException($"Missing backend snapshot state {snapshotId} during release.");
        }

        Events.Add("release-snapshot");
        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (!SequenceStates.Remove(sequenceId))
        {
            throw new InvalidOperationException($"Missing backend sequence state {sequenceId} during release.");
        }

        Events.Add("release-sequence");
        return ValueTask.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        SequenceStates.Clear();
        SnapshotStates.Clear();
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Transactional backend has not been initialized.");
        }
    }
}
