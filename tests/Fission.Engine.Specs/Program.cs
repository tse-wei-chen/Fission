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

static async Task<int[]> CollectAsync(InferenceStream stream)
{
    var tokens = new List<int>();
    await foreach (var token in stream.ReadTokensAsync())
    {
        tokens.Add(token);
    }

    return tokens.ToArray();
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

await using var worker = new InferenceWorker(
    engine,
    new InferenceWorkerOptions(AdmissionCapacity: 4));

var submitC = worker.SubmitAsync(
    model,
    new[] { 101, 102, 103, 104, 105, 106, 107, 108, 109 },
    maxNewTokens: 4,
    priority: 5,
    enqueuedAt: baseTime.AddSeconds(1)).AsTask();

var submitD = worker.SubmitAsync(
    model,
    new[] { 201, 202, 203, 204, 205 },
    maxNewTokens: 3,
    priority: 1,
    enqueuedAt: baseTime.AddSeconds(1).AddMilliseconds(1)).AsTask();

var streams = await Task.WhenAll(submitC, submitD);
Require(streams[0].SequenceId != streams[1].SequenceId, "Concurrent submissions must receive distinct sequence ids.");

var collectC = CollectAsync(streams[0]);
var collectD = CollectAsync(streams[1]);
var streamed = await Task.WhenAll(collectC, collectD);
var completions = await Task.WhenAll(streams[0].Completion, streams[1].Completion);

Require(streamed[0].Length == 4, "Worker stream C must publish exactly max_new_tokens=4 decode tokens.");
Require(streamed[1].Length == 3, "Worker stream D must publish exactly max_new_tokens=3 decode tokens.");
Require(completions[0].IsCompleted && completions[1].IsCompleted, "Worker completion tasks must finish with terminal snapshots.");
Require(streamed[0].SequenceEqual(completions[0].GeneratedTokens), "Stream C token order must match engine request history.");
Require(streamed[1].SequenceEqual(completions[1].GeneratedTokens), "Stream D token order must match engine request history.");
Require(runtime.SequenceCount == 0, "Async worker must release all terminal runtime sequences.");
Require(kvPool.AllocatedPages == 0, "Async worker completion must return all KV pages to the pool.");

Console.WriteLine(
    $"Fission engine specs passed: cycles={cycles.Count}, prefill=[{string.Join(',', prefillGrants)}], " +
    $"manual={snapshotA.GeneratedTokens.Count + snapshotB.GeneratedTokens.Count}, " +
    $"streamed={streamed[0].Length + streamed[1].Length}, kv={kvPool.AllocatedPages}/{kvPool.Capacity}.");
