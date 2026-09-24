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

var device = new DeviceId("cpu:0");
var model = new ModelId("engine-spec-model");
var kvPool = new KvPagePool(capacity: 16, tokensPerPage: 4);

await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(device),
    capacity: 128,
    maxBatchSize: 16);

using var runtime = new ExecutionPlanExecutor(deviceExecutor, kvPagePool: kvPool);
ISchedulingKernel scheduler = new SchedulingKernel();
using var engine = new InferenceEngine(
    runtime,
    scheduler,
    new InferenceEngineOptions(
        MaxBatchTokens: 5,
        MaxBatchSequences: 2,
        Scheduling: new SchedulingPolicyOptions(
            DecodeTokenReserve: 1,
            MaxPrefillChunkTokens: 4,
            DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50))));

var baseTime = new DateTimeOffset(2026, 9, 24, 14, 0, 0, TimeSpan.Zero);
var requestA = engine.Submit(
    model,
    new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    maxNewTokens: 3,
    priority: 10,
    enqueuedAt: baseTime);

var requestB = engine.Submit(
    model,
    new[] { 11, 12, 13, 14, 15, 16 },
    maxNewTokens: 2,
    priority: 0,
    enqueuedAt: baseTime.AddMilliseconds(1));

var cycles = new List<InferenceCycleResult>();
for (var index = 0; index < 20 && engine.ActiveRequestCount != 0; index++)
{
    cycles.Add(await engine.RunCycleAsync(baseTime.AddMilliseconds(index + 2)));
}

Require(engine.ActiveRequestCount == 0, "Both requests must complete through repeated scheduler cycles.");
Require(cycles.Count > 1, "Engine must require multiple cycles for chunked prefill and decode.");
Require(
    cycles.SelectMany(static cycle => cycle.Batch.Items)
        .Where(static item => item.Kind == ScheduledWorkKind.Prefill)
        .All(static item => item.TokenGrant <= 4),
    "Every prefill grant must respect MaxPrefillChunkTokens.");
Require(
    cycles.SelectMany(static cycle => cycle.Batch.Items)
        .Any(static item => item.Kind == ScheduledWorkKind.Prefill && !item.CompletesPrefill),
    "At least one long prompt must be split into a partial prefill chunk.");
Require(
    cycles.Any(static cycle =>
        cycle.Batch.Items.Any(static item => item.Kind == ScheduledWorkKind.Decode)
        && cycle.Batch.Items.Any(static item => item.Kind == ScheduledWorkKind.Prefill)),
    "At least one scheduler cycle must interleave decode and remaining prefill work.");

var snapshotA = engine.GetSnapshot(requestA);
var snapshotB = engine.GetSnapshot(requestB);
Require(snapshotA.IsCompleted && snapshotA.GeneratedTokens.Count == 3, "Request A must stop exactly at max_new_tokens=3.");
Require(snapshotB.IsCompleted && snapshotB.GeneratedTokens.Count == 2, "Request B must stop exactly at max_new_tokens=2.");
Require(runtime.SequenceCount == 0, "Completed engine requests must release runtime sequence ownership.");
Require(kvPool.AllocatedPages == 0, "Completed requests must return every KV page to the shared pool.");
Require(kvPool.AvailablePages == kvPool.Capacity, "KV capacity must be fully reusable after request completion.");
Require(cycles[^1].KvCapacity.AllocatedPages == 0, "Post-cycle KV feedback must observe released terminal pages.");

var prefillGrants = cycles
    .SelectMany(static cycle => cycle.Batch.Items)
    .Where(static item => item.Kind == ScheduledWorkKind.Prefill)
    .Select(static item => item.TokenGrant)
    .ToArray();

Console.WriteLine(
    $"Fission engine specs passed: cycles={cycles.Count}, prefill=[{string.Join(',', prefillGrants)}], " +
    $"generatedA={snapshotA.GeneratedTokens.Count}, generatedB={snapshotB.GeneratedTokens.Count}, kv={kvPool.AllocatedPages}/{kvPool.Capacity}.");
