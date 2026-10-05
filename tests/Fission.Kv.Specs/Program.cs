using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
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

static ScheduledExecutionBindings EmptyScheduledBindings() =>
    new(new Dictionary<SequenceId, ScheduledPrefillBinding>());

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

static async Task VerifyScheduledCopyOnWriteGrantValidationAsync()
{
    var model = new ModelId("kv-cow-model");
    var device = new DeviceId("cpu:kv-cow-scheduled");
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
            Bind(parentId, 30, 31));

        var branchId = setup.Forks.Single().Branches.Single();
        var branch = GetSequence(runtime, branchId);
        var scheduled = new ScheduledBatchExecutor(runtime);
        var bindings = EmptyScheduledBindings();

        var staleBatch = new ScheduledBatch(
            Guid.NewGuid(),
            new[]
            {
                new ScheduledWorkItem(
                    branchId,
                    ScheduledWorkKind.Decode,
                    TokenGrant: 1,
                    KvPageGrant: 0,
                    Priority: 0,
                    CompletesPrefill: false)
            },
            ConsumedTokens: 1,
            ConsumedKvPages: 0);

        var rejected = false;
        try
        {
            await scheduled.ExecuteAsync(staleBatch, bindings);
        }
        catch (InvalidOperationException)
        {
            rejected = true;
        }

        Require(rejected,
            "Scheduled preflight must reject a stale grant that omits shared-tail COW demand.");
        Require(branch.Position == 2 && pool.AllocatedPages == 1,
            "Rejected COW grants must not dispatch backend work or mutate runtime KV state.");

        var validBatch = new ScheduledBatch(
            Guid.NewGuid(),
            new[]
            {
                new ScheduledWorkItem(
                    branchId,
                    ScheduledWorkKind.Decode,
                    TokenGrant: 1,
                    KvPageGrant: 1,
                    Priority: 0,
                    CompletesPrefill: false)
            },
            ConsumedTokens: 1,
            ConsumedKvPages: 1);

        await scheduled.ExecuteAsync(validBatch, bindings);

        Require(branch.Position == 3,
            "A COW-aware scheduled grant must advance the branch by one decode token.");
        Require(pool.AllocatedPages == 2,
            "A valid scheduled shared-tail write must consume exactly one COW page.");
    }

    Require(pool.AllocatedPages == 0,
        "Scheduled COW validation must not leak page leases after runtime disposal.");
}

static async Task VerifyBackendCowWriteIntentAsync()
{
    var model = new ModelId("kv-cow-intent-model");
    var device = new DeviceId("cpu:kv-cow-intent");
    var pool = new KvPagePool(capacity: 12, tokensPerPage: 4);
    var backend = new KvWriteIntentBackend(device);

    await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
        backend,
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
            Bind(parentId, 40, 41));

        Require(backend.Prefills.Count == 1,
            "Setup prefill must reach the tracking backend exactly once.");
        Require(!backend.Prefills[0].KvWrite.RequiresMaterialization,
            "Initial private prefill must not report a COW materialization intent.");
        Require(backend.Prefills[0].Position == 0,
            "Runtime must carry the explicit committed position on generic prefill items.");

        var branchId = setup.Forks.Single().Branches.Single();
        var branch = GetSequence(runtime, branchId);
        Require(branch.Position == 2 && pool.AllocatedPages == 1,
            "Fork setup must leave one shared partial tail at position two.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                8,
                new ExecutionStep[] { new DecodeExecutionStep(branchId, 1) }),
            EmptyBindings());

        Require(backend.Decodes.Count == 1,
            "First branch decode must reach the tracking backend exactly once.");
        var firstDecode = backend.Decodes[0];
        Require(firstDecode.SequenceId == branchId && firstDecode.Position == 2,
            "Backend COW intent must be attached to the exact divergent branch position.");
        Require(firstDecode.KvWrite.RequiresMaterialization,
            "First write into a shared partial tail must require backend materialization.");
        Require(firstDecode.KvWrite.CopyOnWritePages == 1,
            "Backend contract must report the one physical COW page charged by admission.");
        Require(firstDecode.KvWrite.TokensPerPage == 4,
            "Backend contract must preserve runtime KV token-block geometry.");
        Require(firstDecode.KvWrite.TailTokenCount == 2,
            "Backend contract must identify how much of the shared tail is already committed.");
        Require(branch.Position == 3 && pool.AllocatedPages == 2,
            "Successful divergent decode must commit one private runtime COW page.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                8,
                new ExecutionStep[] { new DecodeExecutionStep(branchId, 1) }),
            EmptyBindings());

        Require(backend.Decodes.Count == 2,
            "Second branch decode must reach the backend.");
        Require(!backend.Decodes[1].KvWrite.RequiresMaterialization,
            "Once the branch owns a private tail, later writes into its slack must not repeat COW materialization.");
        Require(branch.Position == 4 && pool.AllocatedPages == 2,
            "Filling the private tail must not allocate another physical page.");
    }

    Require(pool.AllocatedPages == 0,
        "Backend write-intent decode coverage must release all runtime page leases.");

    var prefillPool = new KvPagePool(capacity: 12, tokensPerPage: 4);
    var prefillBackend = new KvWriteIntentBackend(new DeviceId("cpu:kv-cow-intent-prefill"));
    await using var prefillDevice = await ContinuousBatchExecutor.CreateAsync(
        prefillBackend,
        capacity: 16,
        maxBatchSize: 4);

    using (var runtime = new ExecutionPlanExecutor(prefillDevice, kvPagePool: prefillPool))
    {
        var parentId = SequenceId.New();
        var setup = await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                32,
                new ExecutionStep[]
                {
                    new PrefillExecutionStep(parentId, model, 2, CompletesPrefill: false),
                    new ForkKvExecutionStep(parentId, 1)
                }),
            Bind(parentId, 50, 51));

        var branchId = setup.Forks.Single().Branches.Single();
        var branch = GetSequence(runtime, branchId);
        var scheduled = new ScheduledBatchExecutor(runtime);
        var bindings = new ScheduledExecutionBindings(
            new Dictionary<SequenceId, ScheduledPrefillBinding>
            {
                [branchId] = new(model, new[] { 50, 51, 52, 53 })
            });
        var batch = new ScheduledBatch(
            Guid.NewGuid(),
            new[]
            {
                new ScheduledWorkItem(
                    branchId,
                    ScheduledWorkKind.Prefill,
                    TokenGrant: 1,
                    KvPageGrant: 1,
                    Priority: 0,
                    CompletesPrefill: false)
            },
            ConsumedTokens: 1,
            ConsumedKvPages: 1);

        await scheduled.ExecuteAsync(batch, bindings);

        Require(prefillBackend.Prefills.Count == 2,
            "Tracking backend must observe initial and fork-continuation prefill calls.");
        var continuation = prefillBackend.Prefills[1];
        Require(continuation.SequenceId == branchId && continuation.Position == 2,
            "Scheduled prefill continuation must carry the branch's live committed position.");
        Require(continuation.Tokens.Span.SequenceEqual(new[] { 52 }),
            "Scheduled prefill must preserve the exact granted prompt slice.");
        Require(continuation.KvWrite.RequiresMaterialization,
            "Partial-prefill branch divergence must carry the same physical COW contract as decode.");
        Require(
            continuation.KvWrite.CopyOnWritePages == 1 &&
            continuation.KvWrite.TokensPerPage == 4 &&
            continuation.KvWrite.TailTokenCount == 2,
            "Scheduled prefill COW intent must match runtime page geometry and admission cost.");
        Require(branch.Position == 3 && prefillPool.AllocatedPages == 2,
            "Scheduled prefill continuation must commit its private logical tail after backend success.");
    }

    Require(prefillPool.AllocatedPages == 0,
        "Backend write-intent prefill coverage must release all runtime page leases.");
}

await VerifyPartialForkCopyOnWriteAsync();
await VerifyPartialSnapshotCopyOnWriteAsync();
await VerifyScheduledCopyOnWriteGrantValidationAsync();
await VerifyBackendCowWriteIntentAsync();

Console.WriteLine("Fission KV specs passed: partial-tail COW is isolated, backend write intent is explicit, scheduler grants are revalidated, and pages are reclaimable.");

sealed class KvWriteIntentBackend(DeviceId device) : IInferenceBackend
{
    public string Name => "kv-write-intent";
    public DeviceId Device { get; } = device;
    public List<PrefillItem> Prefills { get; } = [];
    public List<DecodeItem> Decodes { get; } = [];

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            Prefills.Add(item);
            results[index] = new BackendStepResult(item.SequenceId, 1000 + item.Tokens.Length);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            Decodes.Add(item);
            results[index] = new BackendStepResult(item.SequenceId, 2000 + item.Position);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
