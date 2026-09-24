using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Backends;
using Fission.Runtime.Execution;
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

var model = new ModelId("spec-model");
var device = new DeviceId("cpu:0");
var parentId = SequenceId.New();

await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(device),
    capacity: 128,
    maxBatchSize: 16);

using var runtime = new ExecutionPlanExecutor(deviceExecutor);

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

Console.WriteLine(
    $"Fission runtime specs passed: shared={sharedPages.Length}, branchA={branchA.Kv.PageIds.Count}, rollback={parent.Kv.PageIds.Count}.");
