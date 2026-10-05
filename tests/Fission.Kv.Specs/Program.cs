using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Backends;
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

static ExecutionBindings Bind(SequenceId sequenceId, params int[] tokens) =>
    new(new Dictionary<SequenceId, ReadOnlyMemory<int>>
    {
        [sequenceId] = new ReadOnlyMemory<int>(tokens)
    });

static ExecutionBindings EmptyBindings() =>
    new(new Dictionary<SequenceId, ReadOnlyMemory<int>>());

static async Task VerifyPartialForkCopyOnWriteAsync()
{
    var model = new ModelId("kv-cow-model");
    var device = new DeviceId("cpu:kv-cow-fork");
    var pool = new KvPagePool(capacity: 8, tokensPerPage: 4);

    await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
        new DeterministicBackend(device),
        capacity: 16,
        maxBatchSize: 4);

    using (var runtime = new ExecutionPlanExecutor(deviceExecutor, kvPagePool: pool))
    {
        var parentId = SequenceId.New();
        var setup = await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                32,
                new ExecutionStep[]
                {
                    new PrefillExecutionStep(parentId, model, 2, CompletesPrefill: true),
                    new ForkKvExecutionStep(parentId, 1)
                }),
            Bind(parentId, 10, 11));

        Require(setup.Forks.Count == 1 && setup.Forks[0].Branches.Count == 1,
            "Expected one forked branch.");
        Require(pool.AllocatedPages == 1,
            "Forking a two-token partial page must keep one physical shared page.");

        var parent = GetSequence(runtime, parentId);
        var branchId = setup.Forks[0].Branches[0];
        var branch = GetSequence(runtime, branchId);

        var sharedPageId = parent.Kv.PageIds.Single();
        Require(branch.Kv.PageIds.SequenceEqual(new[] { sharedPageId }),
            "Forked branch must initially share the parent's partial tail page.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                8,
                new ExecutionStep[] { new DecodeExecutionStep(branchId, 1) }),
            EmptyBindings());

        var privateTailId = branch.Kv.PageIds.Single();
        Require(privateTailId != sharedPageId,
            "First branch write into a shared partial tail must copy the page before mutation.");
        Require(parent.Kv.PageIds.SequenceEqual(new[] { sharedPageId }),
            "Parent must retain the original shared page after child copy-on-write.");
        Require(branch.Kv.Count == 1,
            "Copy-on-write must replace the partial tail instead of appending a logical page.");
        Require(pool.AllocatedPages == 2,
            "Copy-on-write must consume exactly one additional physical page.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                8,
                new ExecutionStep[] { new DecodeExecutionStep(branchId, 1) }),
            EmptyBindings());

        Require(branch.Kv.PageIds.SequenceEqual(new[] { privateTailId }),
            "A now-private tail page must be reused until its token block is full.");
        Require(pool.AllocatedPages == 2,
            "Writing the remaining private tail slack must not allocate another page.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                8,
                new ExecutionStep[] { new DecodeExecutionStep(branchId, 1) }),
            EmptyBindings());

        Require(branch.Kv.Count == 2,
            "Decode after the private tail reaches the page boundary must append a new page.");
        Require(branch.Kv.PageIds[0] == privateTailId,
            "Boundary growth must preserve the private copied prefix page.");
        Require(pool.AllocatedPages == 3,
            "Boundary growth after copy-on-write must consume one additional physical page.");
    }

    Require(pool.AllocatedPages == 0,
        "Runtime disposal must release parent and fork copy-on-write page leases.");
}

static async Task VerifyPartialSnapshotCopyOnWriteAsync()
{
    var model = new ModelId("kv-cow-model");
    var device = new DeviceId("cpu:kv-cow-snapshot");
    var pool = new KvPagePool(capacity: 8, tokensPerPage: 4);

    await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
        new DeterministicBackend(device),
        capacity: 16,
        maxBatchSize: 4);

    using (var runtime = new ExecutionPlanExecutor(deviceExecutor, kvPagePool: pool))
    {
        var sequenceId = SequenceId.New();
        var setup = await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                32,
                new ExecutionStep[]
                {
                    new PrefillExecutionStep(sequenceId, model, 3, CompletesPrefill: true),
                    new SnapshotKvExecutionStep(sequenceId)
                }),
            Bind(sequenceId, 20, 21, 22));

        Require(setup.Snapshots.Count == 1, "Expected one KV snapshot.");
        var sequence = GetSequence(runtime, sequenceId);

        var snapshotId = setup.Snapshots[0];
        var snapshotPageId = sequence.Kv.PageIds.Single();
        Require(pool.AllocatedPages == 1,
            "Snapshotting a partial page must acquire a reference without allocating a page.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                8,
                new ExecutionStep[] { new DecodeExecutionStep(sequenceId, 1) }),
            EmptyBindings());

        var mutatedPageId = sequence.Kv.PageIds.Single();
        Require(mutatedPageId != snapshotPageId,
            "Writing after a partial-page snapshot must preserve the snapshot through copy-on-write.");
        Require(sequence.Kv.Count == 1,
            "Snapshot copy-on-write must replace, not append, the logical tail page.");
        Require(pool.AllocatedPages == 2,
            "Snapshot copy-on-write must retain the snapshot page and one private successor page.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                8,
                new ExecutionStep[] { new RestoreKvExecutionStep(sequenceId, snapshotId) }),
            EmptyBindings());

        Require(sequence.Position == 3,
            "Restore must rewind the sequence to the partial-page snapshot position.");
        Require(sequence.Kv.PageIds.SequenceEqual(new[] { snapshotPageId }),
            "Restore must recover the original snapshot page after copy-on-write mutation.");
        Require(pool.AllocatedPages == 1,
            "Restore must release the discarded private successor page.");
    }

    Require(pool.AllocatedPages == 0,
        "Runtime disposal must release partial snapshot page leases.");
}

await VerifyPartialForkCopyOnWriteAsync();
await VerifyPartialSnapshotCopyOnWriteAsync();

Console.WriteLine("Fission KV specs passed: partial-tail fork/snapshot copy-on-write is isolated and reclaimable.");
