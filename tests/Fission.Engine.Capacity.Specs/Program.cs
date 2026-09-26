using Fission.Abstractions;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Backends;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

var device = new DeviceId("cpu:engine-capacity");
var model = new ModelId("engine-capacity-model");
var kvPool = new KvPagePool(capacity: 32, tokensPerPage: 4);

await using var executor = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(device),
    capacity: 2,
    maxBatchSize: 8);
using var runtime = new ExecutionPlanExecutor(executor, kvPagePool: kvPool);
using var engine = new InferenceEngine(
    runtime,
    new SchedulingKernel(),
    new InferenceEngineOptions(
        MaxBatchTokens: 32,
        MaxBatchSequences: 8,
        Scheduling: new SchedulingPolicyOptions(
            DecodeTokenReserve: 0,
            MaxPrefillChunkTokens: 8,
            DeadlineUrgencyWindow: TimeSpan.Zero)));

Require(
    runtime.GetExecutionCapacity().MaxInferenceItems == 2,
    "Runtime execution-capacity feedback must expose the device inference-item limit.");

var baseTime = new DateTimeOffset(2026, 9, 26, 13, 0, 0, TimeSpan.Zero);
var requests = new[]
{
    engine.Submit(model, new[] { 11 }, maxNewTokens: 1, priority: 3, enqueuedAt: baseTime),
    engine.Submit(model, new[] { 12 }, maxNewTokens: 1, priority: 2, enqueuedAt: baseTime.AddMilliseconds(1)),
    engine.Submit(model, new[] { 13 }, maxNewTokens: 1, priority: 1, enqueuedAt: baseTime.AddMilliseconds(2))
};

var firstCycle = await engine.RunCycleAsync(baseTime.AddMilliseconds(3));
Require(
    firstCycle.Batch.Items.Count == 2,
    "Engine must clamp scheduler sequence budget to device capacity even when MaxBatchSequences is larger.");
Require(
    firstCycle.Batch.Items.All(static item => item.Kind == ScheduledWorkKind.Prefill),
    "First capacity-feedback cycle must consist of the two admitted prefills.");
Require(
    runtime.SequenceCount == 2 && kvPool.AllocatedPages == 2,
    "Only the two device-capacity-admitted requests may materialize runtime/KV state in the first cycle.");
Require(
    !runtime.TryGetSequence(requests[2], out _),
    "The third request must remain scheduler-pending instead of reaching oversized-envelope preflight.");

foreach (var request in requests)
{
    var cancelled = await engine.CancelAsync(request);
    Require(
        cancelled.FinishReason == InferenceFinishReason.Cancelled,
        "Capacity-feedback test requests must remain safely cancellable.");
}

Require(engine.ActiveRequestCount == 0, "Capacity-feedback cleanup must leave no active engine requests.");
Require(runtime.SequenceCount == 0, "Capacity-feedback cleanup must release all runtime sequences.");
Require(kvPool.AllocatedPages == 0, "Capacity-feedback cleanup must return all KV pages.");

Console.WriteLine(
    $"Fission engine capacity specs passed: configured=8, device={runtime.GetExecutionCapacity().MaxInferenceItems}, " +
    $"selected={firstCycle.Batch.Items.Count}, kv={kvPool.AllocatedPages}/{kvPool.Capacity}.");
