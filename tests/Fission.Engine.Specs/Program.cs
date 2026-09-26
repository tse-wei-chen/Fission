using Fission.Abstractions;
using Fission.Abstractions.Execution;
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
Require(snapshotA.FinishReason == InferenceFinishReason.Length, "Request A must report a length finish reason.");
Require(snapshotB.FinishReason == InferenceFinishReason.Length, "Request B must report a length finish reason.");
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
Require(completions[0].FinishReason == InferenceFinishReason.Length, "Worker stream C must finish by length.");
Require(completions[1].FinishReason == InferenceFinishReason.Length, "Worker stream D must finish by length.");
Require(streamed[0].SequenceEqual(completions[0].GeneratedTokens), "Stream C token order must match engine request history.");
Require(streamed[1].SequenceEqual(completions[1].GeneratedTokens), "Stream D token order must match engine request history.");
Require(runtime.SequenceCount == 0, "Async worker must release all terminal runtime sequences.");
Require(kvPool.AllocatedPages == 0, "Async worker completion must return all KV pages to the pool.");

var trackingBackend = new TrackingStateBackend(new DeviceId("cpu:tracking"));
var trackingKvPool = new KvPagePool(capacity: 8, tokensPerPage: 4);
await using var trackingDevice = await ContinuousBatchExecutor.CreateAsync(
    trackingBackend,
    capacity: 16,
    maxBatchSize: 4);
using var trackingRuntime = new ExecutionPlanExecutor(
    trackingDevice,
    kvPagePool: trackingKvPool);
using var trackingEngine = new InferenceEngine(
    trackingRuntime,
    new SchedulingKernel(),
    new InferenceEngineOptions(
        MaxBatchTokens: 4,
        MaxBatchSequences: 1,
        Scheduling: new SchedulingPolicyOptions(
            DecodeTokenReserve: 1,
            MaxPrefillChunkTokens: 4,
            DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50))));

var normalStateful = trackingEngine.Submit(
    new ModelId("tracking-model"),
    new[] { 1, 2, 3, 4 },
    maxNewTokens: 2,
    enqueuedAt: baseTime.AddSeconds(2));
await trackingEngine.RunUntilCompleteAsync();
Require(trackingBackend.ActiveSequenceCount == 0, "Normal completion must release backend-owned sequence state.");
Require(trackingBackend.ReleaseCount == 1, "Normal completion must release backend state exactly once.");
Require(trackingBackend.ReleasedSequences.Contains(normalStateful), "Normal sequence id must reach backend release hook.");
Require(trackingRuntime.SequenceCount == 0 && trackingKvPool.AllocatedPages == 0, "Normal stateful completion must release runtime and KV ownership.");

var cancelledStateful = trackingEngine.Submit(
    new ModelId("tracking-model"),
    new[] { 10, 11, 12, 13 },
    maxNewTokens: 100,
    enqueuedAt: baseTime.AddSeconds(3));
var prefillCycle = await trackingEngine.RunCycleAsync(baseTime.AddSeconds(3).AddMilliseconds(1));
Require(prefillCycle.Batch.Items.Count == 1 && prefillCycle.Batch.Items[0].Kind == ScheduledWorkKind.Prefill, "Cancellation spec must materialize backend state with one prefill cycle.");
Require(trackingBackend.ActiveSequenceCount == 1, "Prefill must create backend-owned per-sequence state.");
Require(trackingRuntime.SequenceCount == 1 && trackingKvPool.AllocatedPages == 1, "Prefill must create runtime and KV ownership before cancellation.");

var cancelledSnapshot = await trackingEngine.CancelAsync(cancelledStateful);
Require(cancelledSnapshot.FinishReason == InferenceFinishReason.Cancelled, "Cancellation must preserve cancelled finish reason.");
Require(trackingBackend.ActiveSequenceCount == 0, "Cancellation must release backend-owned sequence state.");
Require(trackingBackend.ReleaseCount == 2, "Cancellation must invoke backend release exactly once in addition to normal completion.");
Require(trackingBackend.ReleasedSequences.Contains(cancelledStateful), "Cancelled sequence id must reach backend release hook.");
Require(trackingRuntime.SequenceCount == 0 && trackingKvPool.AllocatedPages == 0, "Cancellation must release runtime and KV ownership after backend release.");

var byteBackend = new TrackingStateBackend(
    new DeviceId("cpu:byte-budget"),
    kvBytesPerToken: 128);
var byteKvPool = new KvPagePool(capacity: 16, tokensPerPage: 16);
await using var byteDevice = await ContinuousBatchExecutor.CreateAsync(
    byteBackend,
    capacity: 16,
    maxBatchSize: 4);
using var byteRuntime = new ExecutionPlanExecutor(
    byteDevice,
    kvPagePool: byteKvPool);
using var byteEngine = new InferenceEngine(
    byteRuntime,
    new SchedulingKernel(),
    new InferenceEngineOptions(
        MaxBatchTokens: 16,
        MaxBatchSequences: 1,
        Scheduling: new SchedulingPolicyOptions(
            DecodeTokenReserve: 0,
            MaxPrefillChunkTokens: 16,
            DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50)),
        MaxKvBytes: 512),
    kvMemoryProfile: byteBackend);

var byteLimited = byteEngine.Submit(
    new ModelId("byte-model"),
    new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    maxNewTokens: 1,
    enqueuedAt: baseTime.AddSeconds(4));

var byteFirstCycle = await byteEngine.RunCycleAsync(baseTime.AddSeconds(4).AddMilliseconds(1));
Require(byteFirstCycle.Batch.Items.Count == 1, "Byte-budget spec must select one partial prefill.");
Require(byteFirstCycle.Batch.Items[0].TokenGrant == 4, "512 bytes at 128 bytes/token must admit exactly four prompt tokens.");
Require(byteFirstCycle.Batch.Items[0].KvByteGrant == 512, "Scheduled work must preserve its physical KV byte grant.");
Require(byteFirstCycle.Batch.ConsumedKvBytes == 512, "Scheduled batch must account for consumed physical KV bytes.");
Require(
    byteRuntime.TryGetSequence(byteLimited, out var byteSequence) && byteSequence?.Position == 4,
    "Executed partial prefill must advance the live sequence position to four tokens.");

var byteSecondCycle = await byteEngine.RunCycleAsync(baseTime.AddSeconds(4).AddMilliseconds(2));
Require(byteSecondCycle.Batch.Items.Count == 0, "A fully retained byte budget must block the next prefill chunk.");
Require(
    byteSecondCycle.Deferred.Count == 1 &&
    byteSecondCycle.Deferred[0].Reason == SchedulingDeferralReason.KvByteBudget,
    "Live retained KV bytes must feed back into the next scheduler cycle.");

var byteCancelled = await byteEngine.CancelAsync(byteLimited);
Require(byteCancelled.FinishReason == InferenceFinishReason.Cancelled, "Byte-budget test request must be cancellable after partial prefill.");
Require(byteRuntime.SequenceCount == 0 && byteKvPool.AllocatedPages == 0, "Byte-budget cancellation must release runtime KV ownership.");

var capacityKvPool = new KvPagePool(capacity: 16, tokensPerPage: 4);
await using var capacityDevice = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(new DeviceId("cpu:engine-capacity")),
    capacity: 2,
    maxBatchSize: 8);
using var capacityRuntime = new ExecutionPlanExecutor(
    capacityDevice,
    kvPagePool: capacityKvPool);
using var capacityEngine = new InferenceEngine(
    capacityRuntime,
    new SchedulingKernel(),
    new InferenceEngineOptions(
        MaxBatchTokens: 8,
        MaxBatchSequences: 4,
        Scheduling: new SchedulingPolicyOptions(
            DecodeTokenReserve: 0,
            MaxPrefillChunkTokens: 4,
            DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50))));

var capacityModel = new ModelId("capacity-model");
var capacityRequests = new[]
{
    capacityEngine.Submit(capacityModel, new[] { 1 }, maxNewTokens: 1, priority: 3, enqueuedAt: baseTime.AddSeconds(5)),
    capacityEngine.Submit(capacityModel, new[] { 2 }, maxNewTokens: 1, priority: 2, enqueuedAt: baseTime.AddSeconds(5).AddMilliseconds(1)),
    capacityEngine.Submit(capacityModel, new[] { 3 }, maxNewTokens: 1, priority: 1, enqueuedAt: baseTime.AddSeconds(5).AddMilliseconds(2))
};

var capacityCycles = new List<InferenceCycleResult>();
for (var index = 0; index < 8 && capacityEngine.ActiveRequestCount != 0; index++)
{
    capacityCycles.Add(await capacityEngine.RunCycleAsync(baseTime.AddSeconds(5).AddMilliseconds(10 + index)));
}

Require(capacityCycles.Count > 0, "Capacity feedback spec must execute scheduler cycles.");
Require(
    capacityCycles.All(static cycle => cycle.Batch.Items.Count <= 2),
    "Engine must clamp every scheduler batch to the device's two inference-item credits.");
Require(
    capacityCycles[0].Batch.Items.Count == 2,
    "A wider MaxBatchSequences setting must still use the device capacity instead of triggering an oversized envelope.");
Require(capacityEngine.ActiveRequestCount == 0, "Requests deferred by device capacity must make progress in later cycles.");
Require(
    capacityRequests.All(id => capacityEngine.GetSnapshot(id).IsCompleted),
    "All requests must complete after capacity-clamped cycles.");
Require(capacityRuntime.SequenceCount == 0, "Capacity-clamped completion must release runtime sequences.");
Require(capacityKvPool.AllocatedPages == 0, "Capacity-clamped completion must release KV pages.");

Console.WriteLine(
    $"Fission engine specs passed: cycles={cycles.Count}, prefill=[{string.Join(',', prefillGrants)}], " +
    $"manual={snapshotA.GeneratedTokens.Count + snapshotB.GeneratedTokens.Count}, " +
    $"streamed={streamed[0].Length + streamed[1].Length}, backendReleases={trackingBackend.ReleaseCount}, " +
    $"byteGrant={byteFirstCycle.Batch.ConsumedKvBytes}, deviceClamp={capacityCycles.Max(static cycle => cycle.Batch.Items.Count)}, " +
    $"kv={kvPool.AllocatedPages}/{kvPool.Capacity}.");

sealed class TrackingStateBackend : IInferenceBackend, IInferenceKvMemoryProfile
{
    private readonly HashSet<SequenceId> _active = new();
    private readonly HashSet<SequenceId> _released = new();
    private readonly long _kvBytesPerToken;
    private bool _initialized;

    public TrackingStateBackend(DeviceId device, long kvBytesPerToken = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(kvBytesPerToken);
        Device = device;
        _kvBytesPerToken = kvBytesPerToken;
    }

    public string Name => "tracking-state";
    public DeviceId Device { get; }
    public int ActiveSequenceCount => _active.Count;
    public int ReleaseCount { get; private set; }
    public IReadOnlySet<SequenceId> ReleasedSequences => _released;

    public long GetKvBytesPerToken(ModelId modelId) => _kvBytesPerToken;

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
            _active.Add(item.SequenceId);
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
            if (!_active.Contains(item.SequenceId))
            {
                throw new InvalidOperationException($"Decode reached backend without state for {item.SequenceId}.");
            }

            results[index] = new BackendStepResult(item.SequenceId, 20_000 + item.Position);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();

        if (!_active.Remove(sequenceId))
        {
            throw new InvalidOperationException($"Backend release called without active state for {sequenceId}.");
        }

        _released.Add(sequenceId);
        ReleaseCount++;
        return ValueTask.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        _active.Clear();
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Tracking backend has not been initialized.");
        }
    }
}
