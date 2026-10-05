using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;
using Fission.Scheduler;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static async Task RequireThrowsAsync<TException>(Func<Task> action, string message)
    where TException : Exception
{
    try
    {
        await action().ConfigureAwait(false);
    }
    catch (TException)
    {
        return;
    }

    throw new InvalidOperationException(message);
}

static InferenceEngineOptions CreateOptions() =>
    new(
        MaxBatchTokens: 4,
        MaxBatchSequences: 4,
        Scheduling: new SchedulingPolicyOptions(
            DecodeTokenReserve: 0,
            MaxPrefillChunkTokens: 4,
            DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50)));

static async Task RunForkLifecycleAsync()
{
    var device = new DeviceId("cpu:fork-lifecycle");
    var model = new ModelId("fork-lifecycle-model");
    var backend = new TrackingForkBackend(device);
    var kvPool = new KvPagePool(capacity: 16, tokensPerPage: 4);

    await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
        backend,
        capacity: 32,
        maxBatchSize: 8);
    using var runtime = new ExecutionPlanExecutor(
        deviceExecutor,
        kvPagePool: kvPool);
    using var engine = new InferenceEngine(
        runtime,
        new SchedulingKernel(),
        CreateOptions());

    var enqueuedAt = new DateTimeOffset(2026, 10, 5, 6, 0, 0, TimeSpan.Zero);
    var deadline = enqueuedAt.AddSeconds(10);
    var parentId = engine.Submit(
        model,
        new[] { 11, 12 },
        maxNewTokens: 3,
        priority: 7,
        deadline: deadline,
        enqueuedAt: enqueuedAt);

    var prefill = await engine.RunCycleAsync(enqueuedAt.AddMilliseconds(1));
    Require(
        prefill.Batch.Items.Count == 1 &&
        prefill.Batch.Items[0].SequenceId.Equals(parentId) &&
        prefill.Batch.Items[0].Kind == ScheduledWorkKind.Prefill,
        "Fork lifecycle setup must materialize the parent with one prefill quantum.");
    Require(
        runtime.TryGetSequence(parentId, out var materializedParent) &&
        materializedParent is not null &&
        materializedParent.Status == SequenceStatus.Decoding &&
        materializedParent.Position == 2,
        "Completed prompt prefill must leave the parent in decoding state at position two.");

    var firstDecode = await engine.RunCycleAsync(enqueuedAt.AddMilliseconds(2));
    Require(
        firstDecode.Batch.Items.Count == 1 &&
        firstDecode.Batch.Items[0].SequenceId.Equals(parentId) &&
        firstDecode.Batch.Items[0].Kind == ScheduledWorkKind.Decode,
        "Fork lifecycle setup must commit one parent decode token before branching.");

    var parentBeforeFork = engine.GetSnapshot(parentId);
    Require(
        parentBeforeFork.GeneratedTokens.Count == 1,
        "Parent must have exactly one generated token before fork.");
    Require(
        runtime.TryGetSequence(parentId, out materializedParent) &&
        materializedParent is not null &&
        materializedParent.Position == 3,
        "Parent runtime position must include the first generated token before fork.");
    Require(
        kvPool.AllocatedPages == 1,
        "Three committed tokens must still fit in one physical KV page before fork.");

    var fork = await engine.ForkAsync(parentId, branches: 2);
    Require(fork.Parent.Equals(parentId), "Fork result must identify the requested parent.");
    Require(fork.Branches.Count == 2, "Fork must return exactly two branch ids.");
    Require(
        fork.Branches[0] != fork.Branches[1] &&
        !fork.Branches.Contains(parentId),
        "Forked branches must receive distinct sequence ids.");
    Require(engine.RequestCount == 3, "Parent plus two branches must be retained in engine history.");
    Require(engine.ActiveRequestCount == 3, "Parent plus two branches must immediately become active scheduler candidates.");
    Require(runtime.SequenceCount == 3, "Runtime must own parent plus both forked sequences.");
    Require(
        kvPool.AllocatedPages == 1,
        "Forking a partial tail must share the existing physical page until a branch writes.");
    Require(backend.ForkCount == 1, "Backend fork hook must run exactly once.");
    Require(backend.LastForkParent.Equals(parentId), "Backend fork hook must receive the parent sequence id.");
    Require(
        backend.LastForkBranches.SequenceEqual(fork.Branches),
        "Backend fork hook must receive the exact branch ids returned by the engine.");

    foreach (var branchId in fork.Branches)
    {
        var branchSnapshot = engine.GetSnapshot(branchId);
        Require(branchSnapshot.ModelId.Equals(parentBeforeFork.ModelId), "Branch must inherit the parent model.");
        Require(branchSnapshot.PromptTokenCount == parentBeforeFork.PromptTokenCount, "Branch must inherit prompt history.");
        Require(
            branchSnapshot.GeneratedTokens.SequenceEqual(parentBeforeFork.GeneratedTokens),
            "Branch must inherit committed generated-token history.");
        Require(branchSnapshot.Priority == parentBeforeFork.Priority, "Branch must inherit priority.");
        Require(branchSnapshot.Deadline == parentBeforeFork.Deadline, "Branch must inherit deadline.");
        Require(branchSnapshot.EnqueuedAt == parentBeforeFork.EnqueuedAt, "Branch must inherit enqueue age.");
        Require(!branchSnapshot.IsCompleted, "Fresh fork branch must remain active.");

        Require(
            runtime.TryGetSequence(branchId, out var branchSequence) &&
            branchSequence is not null &&
            branchSequence.Status == SequenceStatus.Decoding &&
            branchSequence.Position == 3,
            "Forked runtime branch must inherit parent decoding state and position.");
        Require(
            branchSequence.KvPageWriteOverhead == 1,
            "Each branch must expose shared partial-tail COW pressure before its first write.");
    }

    await engine.RunUntilCompleteAsync(maxCycles: 20);

    var allIds = new[] { parentId }.Concat(fork.Branches).ToArray();
    foreach (var sequenceId in allIds)
    {
        var snapshot = engine.GetSnapshot(sequenceId);
        Require(snapshot.IsCompleted, "Every fork family member must reach a terminal request snapshot.");
        Require(snapshot.FinishReason == InferenceFinishReason.Length, "Fork family members must preserve max-new-token length semantics.");
        Require(snapshot.GeneratedTokens.Count == 3, "Each fork family member must stop at the inherited max-new-token budget.");
        Require(
            snapshot.GeneratedTokens[0] == parentBeforeFork.GeneratedTokens[0],
            "Every branch must retain the token generated before fork.");
    }

    Require(engine.ActiveRequestCount == 0, "All fork family members must leave the active scheduler set after completion.");
    Require(runtime.SequenceCount == 0, "Completion must release all fork family runtime sequences.");
    Require(kvPool.AllocatedPages == 0, "Completion must return all fork/COW KV pages to the shared pool.");
    Require(backend.ReleaseCount == 3, "Backend sequence release must run once for parent and each branch.");
}

static async Task RunForkFailureAsync()
{
    var device = new DeviceId("cpu:fork-failure");
    var model = new ModelId("fork-failure-model");
    var backend = new TrackingForkBackend(device, rejectFork: true);
    var kvPool = new KvPagePool(capacity: 8, tokensPerPage: 4);

    await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
        backend,
        capacity: 16,
        maxBatchSize: 4);
    using var runtime = new ExecutionPlanExecutor(
        deviceExecutor,
        kvPagePool: kvPool);
    using var engine = new InferenceEngine(
        runtime,
        new SchedulingKernel(),
        CreateOptions());

    var enqueuedAt = new DateTimeOffset(2026, 10, 5, 7, 0, 0, TimeSpan.Zero);
    var parentId = engine.Submit(
        model,
        new[] { 21, 22 },
        maxNewTokens: 2,
        enqueuedAt: enqueuedAt);

    await RequireThrowsAsync<InvalidOperationException>(
        () => engine.ForkAsync(parentId, branches: 2).AsTask(),
        "Forking before runtime materialization must fail.");
    Require(backend.ForkCount == 0, "Pre-materialization rejection must not reach the backend fork hook.");
    Require(engine.RequestCount == 1 && engine.ActiveRequestCount == 1, "Pre-materialization rejection must not add engine requests.");
    Require(runtime.SequenceCount == 0 && kvPool.AllocatedPages == 0, "Pre-materialization rejection must not allocate runtime KV state.");

    var prefill = await engine.RunCycleAsync(enqueuedAt.AddMilliseconds(1));
    Require(prefill.Batch.Items.Count == 1, "Failure spec must materialize its parent before backend fork rejection.");
    Require(runtime.SequenceCount == 1 && kvPool.AllocatedPages == 1, "Materialized parent must own one runtime sequence and one KV page.");

    await RequireThrowsAsync<NotSupportedException>(
        () => engine.ForkAsync(parentId, branches: 2).AsTask(),
        "Backend fork rejection must propagate to the caller.");

    Require(backend.ForkCount == 1, "Materialized fork attempt must reach the backend exactly once.");
    Require(engine.RequestCount == 1 && engine.ActiveRequestCount == 1, "Rejected backend fork must not register branch requests.");
    Require(runtime.SequenceCount == 1, "Rejected backend fork must dispose provisional runtime branches.");
    Require(kvPool.AllocatedPages == 1, "Rejected backend fork must return provisional shared KV leases.");

    var cancelled = await engine.CancelAsync(parentId);
    Require(cancelled.FinishReason == InferenceFinishReason.Cancelled, "Failure-spec parent must remain cancellable after fork rejection.");
    Require(engine.ActiveRequestCount == 0, "Cancellation must clear the final active parent request.");
    Require(runtime.SequenceCount == 0 && kvPool.AllocatedPages == 0, "Cancellation must release the surviving parent runtime/KV state.");
    Require(backend.ReleaseCount == 1, "Only the materialized parent should require backend release after rejected fork.");
}

await RunForkLifecycleAsync();
await RunForkFailureAsync();
Console.WriteLine("Fission.Engine.Fork.Specs passed.");

sealed class TrackingForkBackend : IInferenceBackend
{
    private readonly bool _rejectFork;
    private bool _initialized;

    public TrackingForkBackend(DeviceId device, bool rejectFork = false)
    {
        Device = device;
        _rejectFork = rejectFork;
    }

    public string Name => "tracking-fork";
    public DeviceId Device { get; }
    public int ForkCount { get; private set; }
    public int ReleaseCount { get; private set; }
    public SequenceId LastForkParent { get; private set; }
    public IReadOnlyList<SequenceId> LastForkBranches { get; private set; } = Array.Empty<SequenceId>();

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
        cancellationToken.ThrowIfCancellationRequested();

        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            results[index] = new BackendStepResult(item.SequenceId, 10_000 + item.Tokens.Length);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();

        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            results[index] = new BackendStepResult(item.SequenceId, 20_000 + item.Position);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        ForkCount++;
        LastForkParent = parentSequenceId;
        LastForkBranches = branchSequenceIds.ToArray();

        return _rejectFork
            ? ValueTask.FromException(new NotSupportedException("Fork state is intentionally unsupported by this test backend."))
            : ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        ReleaseCount++;
        return ValueTask.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Tracking fork backend is not initialized.");
        }
    }
}
