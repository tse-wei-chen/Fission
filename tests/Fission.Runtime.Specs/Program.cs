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
var kvPool = new KvPagePool(capacity: 4);

await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(device),
    capacity: 128,
    maxBatchSize: 16);

using var runtime = new ExecutionPlanExecutor(deviceExecutor, trace, kvPool);
Require(runtime.KvPages == kvPool, "Runtime must expose its configured KV page pool.");
Require(kvPool.AllocatedPages == 0 && kvPool.AvailablePages == 4, "KV pool must start empty.");

var initialPlan = new CompiledExecutionPlan(
    Guid.NewGuid(),
    100,
    new ExecutionStep[]
    {
        new PrefillExecutionStep(parentId, model, 4),
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
Require(sharedPages.Length == 1, "Prefill should create one logical KV page in the metadata backend.");
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
Require(branchAPages.Length == sharedPages.Length + 1, "Decoded branch must append its own KV page.");
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
Require(parent.Kv.PageIds.Count == snapshotPages.Length + 2, "Parent decode should append logical KV pages.");
Require(kvPool.AllocatedPages == 4 && kvPool.AvailablePages == 0, "Parent mutations must consume the remaining two KV pages.");

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
Require(kvPool.AllocatedPages == 2 && kvPool.AvailablePages == 2, "Rollback must return discarded mutation pages to the pool.");

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
        new(scheduledPrefillId, ScheduledWorkKind.Prefill, 3, 1, 25),
        new(branchBId, ScheduledWorkKind.Decode, 1, 1, 10)
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
Require(kvPool.AllocatedPages == 4 && kvPool.AvailablePages == 0, "Scheduled prefill/decode must consume their physical KV page grants.");

var scheduledPrefill = GetSequence(runtime, scheduledPrefillId);
Require(scheduledPrefill.Position == 3, "Scheduled prefill must advance the new sequence by its token grant.");
Require(scheduledPrefill.Status == SequenceStatus.Decoding, "Scheduled prefill must leave the sequence ready to decode.");
Require(branchB.Position == branchBPositionBeforeScheduledDecode + 1, "Scheduled decode must advance the existing sequence by one token.");

var sequenceCountBeforeInvalidSchedule = runtime.SequenceCount;
var invalidPrefillId = SequenceId.New();
var invalidSchedule = new ScheduledBatch(
    Guid.NewGuid(),
    new ScheduledWorkItem[]
    {
        new(invalidPrefillId, ScheduledWorkKind.Prefill, 2, 1, 0),
        new(SequenceId.New(), ScheduledWorkKind.Prefill, 2, 1, 0)
    },
    ConsumedTokens: 5,
    ConsumedKvPages: 2);

var invalidBindings = new ScheduledExecutionBindings(
    new Dictionary<SequenceId, ScheduledPrefillBinding>
    {
        [invalidPrefillId] = new(model, new ReadOnlyMemory<int>(new[] { 30, 31 })),
        [invalidSchedule.Items[1].SequenceId] = new(model, new ReadOnlyMemory<int>(new[] { 40, 41 }))
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
Require(kvPool.AllocatedPages == 4, "Rejected scheduled work must not leak or consume KV pages.");

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

Require(capacityRejected, "Runtime must reject KV allocation when page capacity is exhausted.");
Require(branchA.Position == exhaustedPosition, "Capacity rejection must not advance sequence state.");
Require(kvPool.AllocatedPages == 4, "Capacity rejection must not leak page accounting.");

var finalBranchAPageCount = branchA.Kv.PageIds.Count;
runtime.Dispose();
Require(kvPool.AllocatedPages == 0 && kvPool.AvailablePages == kvPool.Capacity, "Runtime disposal must release every physical KV page lease.");

Console.WriteLine(
    $"Fission runtime specs passed: shared={sharedPages.Length}, branchA={finalBranchAPageCount}, rollback={snapshotPages.Length}, traceEvents={recordedTrace.Count}, scheduledItems={scheduledResult.ItemResults.Count}, kvCapacity={kvPool.Capacity}.");
