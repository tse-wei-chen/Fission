using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Runtime.Backends;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;
using Fission.Runtime.Tracing;

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

static ReplaySequenceState GetReplaySequence(ExecutionReplayResult replay, SequenceId sequenceId)
{
    if (!replay.Sequences.TryGetValue(sequenceId, out var sequence))
    {
        throw new InvalidOperationException($"Replay is missing sequence {sequenceId}.");
    }

    return sequence;
}

var model = new ModelId("spec-model");
var device = new DeviceId("cpu:0");
var parentId = SequenceId.New();
var trace = new InMemoryExecutionTraceSink();
var kvPool = new KvPagePool(capacity: 7, tokensPerPage: 4);

await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(device),
    capacity: 128,
    maxBatchSize: 16);

using var runtime = new ExecutionPlanExecutor(deviceExecutor, trace, kvPool);
Require(runtime.KvPages == kvPool, "Runtime must expose its configured KV page pool.");
Require(runtime.KvCapacity.TokensPerPage == 4, "Runtime KV feedback must expose the page block size.");
Require(runtime.KvCapacity.AvailablePages == 7, "Runtime KV feedback must expose live available capacity.");

var initialPlan = new CompiledExecutionPlan(
    Guid.NewGuid(),
    100,
    new ExecutionStep[]
    {
        new PrefillExecutionStep(parentId, model, 4, CompletesPrefill: true),
        new ForkKvExecutionStep(parentId, 2)
    });

var bindings = new ExecutionBindings(
    new Dictionary<SequenceId, ReadOnlyMemory<int>>
    {
        [parentId] = new ReadOnlyMemory<int>(new[] { 10, 11, 12, 13 })
    });

var initialResult = await runtime.ExecuteAsync(initialPlan, bindings);
Require(initialResult.Forks.Count == 1, "Expected one fork result.");
Require(initialResult.Forks[0].Branches.Count == 2, "Expected two forked branches.");
Require(kvPool.AllocatedPages == 1, "Forking shared history must not allocate additional physical KV pages.");

var branchAId = initialResult.Forks[0].Branches[0];
var branchBId = initialResult.Forks[0].Branches[1];
var parent = GetSequence(runtime, parentId);
var branchA = GetSequence(runtime, branchAId);
var branchB = GetSequence(runtime, branchBId);

var sharedPages = parent.Kv.PageIds.ToArray();
Require(sharedPages.Length == 1, "Four prefill tokens at block size four must create one KV page.");
Require(branchA.Kv.PageIds.SequenceEqual(sharedPages), "Branch A must share the parent's KV history after fork.");
Require(branchB.Kv.PageIds.SequenceEqual(sharedPages), "Branch B must share the parent's KV history after fork.");
Require(branchA.Position == parent.Position && branchB.Position == parent.Position, "Forked branches must inherit token position.");

var branchDecode = new CompiledExecutionPlan(
    Guid.NewGuid(),
    50,
    new ExecutionStep[]
    {
        new DecodeExecutionStep(branchAId, 1)
    });

await runtime.ExecuteAsync(
    branchDecode,
    new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));

var branchAPages = branchA.Kv.PageIds.ToArray();
Require(branchAPages.Length == sharedPages.Length + 1, "Decode at a page boundary must allocate a new KV page.");
Require(branchAPages.Take(sharedPages.Length).SequenceEqual(sharedPages), "Decoded branch must preserve shared KV prefix.");
Require(branchB.Kv.PageIds.SequenceEqual(sharedPages), "Sibling branch must remain unchanged when another branch decodes.");
Require(parent.Kv.PageIds.SequenceEqual(sharedPages), "Parent must remain unchanged when a child branch decodes.");
Require(kvPool.AllocatedPages == 2, "Branch divergence must allocate exactly one new physical KV page.");

var snapshotPlan = new CompiledExecutionPlan(
    Guid.NewGuid(),
    50,
    new ExecutionStep[]
    {
        new SnapshotKvExecutionStep(parentId)
    });

var snapshotResult = await runtime.ExecuteAsync(
    snapshotPlan,
    new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));

Require(snapshotResult.Snapshots.Count == 1, "Expected one snapshot id.");
Require(kvPool.AllocatedPages == 2, "Snapshot references must not consume additional physical KV capacity.");
var snapshotId = snapshotResult.Snapshots[0];
var snapshotPosition = parent.Position;
var snapshotPages = parent.Kv.PageIds.ToArray();

var mutateParentPlan = new CompiledExecutionPlan(
    Guid.NewGuid(),
    50,
    new ExecutionStep[]
    {
        new DecodeExecutionStep(parentId, 2)
    });

await runtime.ExecuteAsync(
    mutateParentPlan,
    new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));

Require(parent.Position == snapshotPosition + 2, "Parent decode should advance token position.");
Require(parent.Kv.PageIds.Count == snapshotPages.Length + 1, "Two decodes inside one token block must materialize only one new page.");
Require(kvPool.AllocatedPages == 3, "Parent mutations must consume one block-sized KV page.");

var rollbackPlan = new CompiledExecutionPlan(
    Guid.NewGuid(),
    50,
    new ExecutionStep[]
    {
        new RestoreKvExecutionStep(parentId, snapshotId)
    });

await runtime.ExecuteAsync(
    rollbackPlan,
    new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));

Require(parent.Position == snapshotPosition, "Restore must rewind token position to the snapshot.");
Require(parent.Kv.PageIds.SequenceEqual(snapshotPages), "Restore must rewind KV page history to the snapshot.");
Require(branchA.Kv.PageIds.Take(sharedPages.Length).SequenceEqual(parent.Kv.PageIds), "Forked branch must retain the original shared prefix after parent rollback.");
Require(kvPool.AllocatedPages == 2, "Rollback must return the discarded mutation block to the pool.");

var recordedTrace = trace.Snapshot();
Require(recordedTrace.Count > 0, "Execution trace must contain events.");
Require(
    recordedTrace.Select(static item => item.Ordinal).SequenceEqual(Enumerable.Range(1, recordedTrace.Count).Select(static value => (long)value)),
    "Execution trace ordinals must be contiguous and deterministic for this single-threaded spec.");

var replay = ExecutionTraceReplay.Replay(recordedTrace);
Require(replay.Snapshots.Contains(snapshotId), "Replay must retain snapshot creation metadata.");
Require(replay.Sequences.Count == runtime.SequenceCount, "Replay must reconstruct every live sequence.");

foreach (var sequence in new[] { parent, branchA, branchB })
{
    var replayed = GetReplaySequence(replay, sequence.Id);
    Require(replayed.Position == sequence.Position, $"Replay position mismatch for {sequence.Id}.");
    Require(replayed.KvPageCount == sequence.Kv.Count, $"Replay KV page count mismatch for {sequence.Id}.");
    Require(replayed.Device == sequence.Device, $"Replay device mismatch for {sequence.Id}.");
}

Require(GetReplaySequence(replay, parentId).ParentSequenceId is null, "Parent sequence must have no replay parent.");
Require(GetReplaySequence(replay, branchAId).ParentSequenceId == parentId, "Branch A replay must preserve fork ancestry.");
Require(GetReplaySequence(replay, branchBId).ParentSequenceId == parentId, "Branch B replay must preserve fork ancestry.");

var scheduledExecutor = new ScheduledBatchExecutor(runtime);
var scheduledPrefillId = SequenceId.New();
var branchBPositionBeforeScheduledDecode = branchB.Position;
var validSchedule = new ScheduledBatch(
    Guid.NewGuid(),
    new ScheduledWorkItem[]
    {
        new(scheduledPrefillId, ScheduledWorkKind.Prefill, 3, 1, 25, CompletesPrefill: true),
        new(branchBId, ScheduledWorkKind.Decode, 1, 1, 10, CompletesPrefill: false)
    },
    ConsumedTokens: 4,
    ConsumedKvPages: 2);

var scheduledBindings = new ScheduledExecutionBindings(
    new Dictionary<SequenceId, ScheduledPrefillBinding>
    {
        [scheduledPrefillId] = new(
            model,
            new ReadOnlyMemory<int>(new[] { 20, 21, 22 }))
    });

var scheduledResult = await scheduledExecutor.ExecuteAsync(validSchedule, scheduledBindings);
Require(scheduledResult.ScheduleId == validSchedule.ScheduleId, "Scheduled execution must preserve schedule id.");
Require(scheduledResult.ItemResults.Count == 2, "Scheduled execution must return one result per work item.");
Require(kvPool.AllocatedPages == 4, "Scheduled prefill plus boundary decode must consume two physical KV pages.");

var scheduledPrefill = GetSequence(runtime, scheduledPrefillId);
Require(scheduledPrefill.Position == 3, "Scheduled prefill must advance the new sequence by its token grant.");
Require(scheduledPrefill.Status == SequenceStatus.Decoding, "Final scheduled prefill must leave the sequence ready to decode.");
Require(branchB.Position == branchBPositionBeforeScheduledDecode + 1, "Scheduled decode must advance the existing sequence by one token.");

var chunkedId = SequenceId.New();
var fullPrompt = new ReadOnlyMemory<int>(new[] { 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 });
var chunkBindings = new ScheduledExecutionBindings(
    new Dictionary<SequenceId, ScheduledPrefillBinding>
    {
        [chunkedId] = new(model, fullPrompt)
    });

async Task RunChunkAsync(int tokenGrant, int kvGrant, bool completesPrefill)
{
    var batch = new ScheduledBatch(
        Guid.NewGuid(),
        new[]
        {
            new ScheduledWorkItem(
                chunkedId,
                ScheduledWorkKind.Prefill,
                tokenGrant,
                kvGrant,
                Priority: 5,
                CompletesPrefill: completesPrefill)
        },
        ConsumedTokens: tokenGrant,
        ConsumedKvPages: kvGrant);

    await scheduledExecutor.ExecuteAsync(batch, chunkBindings);
}

await RunChunkAsync(4, 1, completesPrefill: false);
var chunked = GetSequence(runtime, chunkedId);
Require(chunked.Position == 4 && chunked.Status == SequenceStatus.Prefilling, "First partial chunk must keep the sequence in prefill state.");
Require(kvPool.AllocatedPages == 5, "First four-token chunk must allocate one page.");

await RunChunkAsync(4, 1, completesPrefill: false);
Require(chunked.Position == 8 && chunked.Status == SequenceStatus.Prefilling, "Second partial chunk must keep the sequence in prefill state.");
Require(kvPool.AllocatedPages == 6, "Second four-token chunk must allocate one additional page.");

await RunChunkAsync(2, 1, completesPrefill: true);
Require(chunked.Position == 10 && chunked.Status == SequenceStatus.Decoding, "Final chunk must transition the sequence to decode state.");
Require(chunked.Kv.PageIds.Count == 3, "Ten tokens at block size four must occupy three KV pages.");
Require(kvPool.AllocatedPages == 7 && kvPool.AvailablePages == 0, "Chunked prefill must consume the final three KV blocks exactly.");

var sequenceCountBeforeInvalidSchedule = runtime.SequenceCount;
var invalidPrefillId = SequenceId.New();
var invalidSchedule = new ScheduledBatch(
    Guid.NewGuid(),
    new ScheduledWorkItem[]
    {
        new(invalidPrefillId, ScheduledWorkKind.Prefill, 2, 1, 0, CompletesPrefill: true),
        new(SequenceId.New(), ScheduledWorkKind.Prefill, 2, 1, 0, CompletesPrefill: true)
    },
    ConsumedTokens: 5,
    ConsumedKvPages: 2);

var invalidBindings = new ScheduledExecutionBindings(
    new Dictionary<SequenceId, ScheduledPrefillBinding>
    {
        [invalidPrefillId] = new(model, new ReadOnlyMemory<int>(new[] { 40, 41 })),
        [invalidSchedule.Items[1].SequenceId] = new(model, new ReadOnlyMemory<int>(new[] { 50, 51 }))
    });

var invalidRejected = false;
try
{
    await scheduledExecutor.ExecuteAsync(invalidSchedule, invalidBindings);
}
catch (InvalidOperationException)
{
    invalidRejected = true;
}

Require(invalidRejected, "Invalid scheduled resource accounting must be rejected.");
Require(runtime.SequenceCount == sequenceCountBeforeInvalidSchedule, "Invalid schedule validation must complete before runtime side effects begin.");
Require(!runtime.TryGetSequence(invalidPrefillId, out _), "Invalid scheduled prefill must not create a sequence.");
Require(kvPool.AllocatedPages == 7, "Rejected scheduled work must not leak or consume KV pages.");

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new DecodeExecutionStep(branchAId, 3) }),
    new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));
Require(branchA.Position == 8, "Existing KV page slack must permit decoding while the global page pool is full.");
Require(kvPool.AllocatedPages == 7, "Decode within an already materialized page must not consume capacity.");

var exhaustedPosition = branchA.Position;
var capacityRejected = false;
try
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[] { new DecodeExecutionStep(branchAId, 1) }),
        new ExecutionBindings(new Dictionary<SequenceId, ReadOnlyMemory<int>>()));
}
catch (KvPageCapacityExceededException)
{
    capacityRejected = true;
}

Require(capacityRejected, "Runtime must reject a boundary decode when KV page capacity is exhausted.");
Require(branchA.Position == exhaustedPosition, "Capacity rejection must not advance sequence metadata.");
Require(kvPool.AllocatedPages == 7, "Capacity rejection must not leak page accounting.");

var finalBranchAPageCount = branchA.Kv.PageIds.Count;
runtime.Dispose();
Require(kvPool.AllocatedPages == 0 && kvPool.AvailablePages == kvPool.Capacity, "Runtime disposal must release every physical KV page lease.");

Console.WriteLine(
    $"Fission runtime specs passed: block={kvPool.TokensPerPage}, shared={sharedPages.Length}, branchA={finalBranchAPageCount}, chunks=3, traceEvents={recordedTrace.Count}, kvCapacity={kvPool.Capacity}.");
