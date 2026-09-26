using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
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

var deviceId = new DeviceId("cpu:cancellation-boundary");
var model = new ModelId("cancellation-boundary-model");
var backend = new BlockingBackend(deviceId);
await using var device = await ContinuousBatchExecutor.CreateAsync(
    backend,
    capacity: 2,
    maxBatchSize: 8);

// Cancellation before queue acceptance must still withdraw work completely.
using (var cancelledBeforeAdmission = new CancellationTokenSource())
{
    cancelledBeforeAdmission.Cancel();
    var cancelled = false;
    try
    {
        await device.SubmitDecodeAsync(
            new DecodeItem(SequenceId.New(), model, Position: 0),
            cancelledBeforeAdmission.Token);
    }
    catch (OperationCanceledException)
    {
        cancelled = true;
    }

    Require(cancelled, "Pre-admission cancellation must still cancel the submitter.");
    Require(backend.DecodeCallCount == 0, "Pre-admission cancellation must not reach the backend.");
}

var kvPool = new KvPagePool(capacity: 8, tokensPerPage: 4);
using var runtime = new ExecutionPlanExecutor(device, kvPagePool: kvPool);
var scheduled = new ScheduledBatchExecutor(runtime);
var first = SequenceId.New();
var second = SequenceId.New();
var schedule = new ScheduledBatch(
    Guid.NewGuid(),
    new ScheduledWorkItem[]
    {
        new(first, ScheduledWorkKind.Prefill, 1, 1, 10, CompletesPrefill: true),
        new(second, ScheduledWorkKind.Prefill, 1, 1, 9, CompletesPrefill: true)
    },
    ConsumedTokens: 2,
    ConsumedKvPages: 2);
var bindings = new ScheduledExecutionBindings(
    new Dictionary<SequenceId, ScheduledPrefillBinding>
    {
        [first] = new(model, new ReadOnlyMemory<int>(new[] { 11 })),
        [second] = new(model, new ReadOnlyMemory<int>(new[] { 12 }))
    });

// Once the atomic envelope has reached backend execution, cancelling the caller
// must not let runtime metadata return early while backend state advances alone.
using (var cancelledAfterAdmission = new CancellationTokenSource())
{
    var execution = scheduled.ExecuteAsync(
        schedule,
        bindings,
        cancelledAfterAdmission.Token).AsTask();

    var physicalBatchSize = await backend.PrefillStarted.Task
        .WaitAsync(TimeSpan.FromSeconds(5));
    Require(physicalBatchSize == 2, "Both scheduled prefills must reach one physical backend batch.");

    cancelledAfterAdmission.Cancel();
    backend.AllowPrefill();

    var result = await execution.WaitAsync(TimeSpan.FromSeconds(5));
    Require(result.ItemResults.Count == 2, "Accepted atomic inference must complete all runtime item results after caller cancellation.");
}

foreach (var sequenceId in new[] { first, second })
{
    Require(
        runtime.TryGetSequence(sequenceId, out var sequence) && sequence is not null,
        $"Accepted sequence {sequenceId} must remain registered after its prefill commits.");
    Require(sequence!.Position == 1, $"Accepted sequence {sequenceId} must commit its backend frontier to runtime position 1.");
    Require(sequence.Status == SequenceStatus.Decoding, $"Accepted sequence {sequenceId} must transition to decoding after final prefill.");
}
Require(kvPool.AllocatedPages == 2, "Both accepted prefills must commit their metadata KV pages.");

// Stateful controls use the same transaction boundary: once the actor starts the
// snapshot barrier, caller cancellation cannot leave a backend snapshot without
// the corresponding runtime snapshot registration.
using (var cancelledSnapshot = new CancellationTokenSource())
{
    var snapshotPlan = new CompiledExecutionPlan(
        Guid.NewGuid(),
        Priority: 0,
        new ExecutionStep[] { new SnapshotKvExecutionStep(first) });
    var emptyBindings = new ExecutionBindings(
        new Dictionary<SequenceId, ReadOnlyMemory<int>>());

    var snapshotExecution = runtime.ExecuteAsync(
        snapshotPlan,
        emptyBindings,
        cancelledSnapshot.Token).AsTask();

    var backendSnapshotId = await backend.SnapshotStarted.Task
        .WaitAsync(TimeSpan.FromSeconds(5));
    cancelledSnapshot.Cancel();
    backend.AllowSnapshot();

    var snapshotResult = await snapshotExecution.WaitAsync(TimeSpan.FromSeconds(5));
    Require(snapshotResult.Snapshots.Count == 1, "Accepted snapshot control must commit one runtime snapshot after caller cancellation.");
    Require(snapshotResult.Snapshots[0] == backendSnapshotId, "Runtime and backend must commit the same snapshot id.");
    Require(runtime.SnapshotCount == 1, "Accepted snapshot control must remain registered in runtime ownership.");

    Require(
        await runtime.ReleaseSnapshotAsync(snapshotResult.Snapshots[0]),
        "Committed snapshot must remain releasable after post-admission cancellation.");
}

foreach (var sequenceId in new[] { first, second })
{
    Require(runtime.TryGetSequence(sequenceId, out var sequence) && sequence is not null, "Cleanup sequence must still exist.");
    sequence!.TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), "Cleanup must release committed sequence ownership.");
}

Require(runtime.SequenceCount == 0, "Cancellation-boundary cleanup must leave no runtime sequences.");
Require(runtime.SnapshotCount == 0, "Cancellation-boundary cleanup must leave no runtime snapshots.");
Require(kvPool.AllocatedPages == 0, "Cancellation-boundary cleanup must return all metadata KV pages.");

Console.WriteLine(
    $"Fission device cancellation specs passed: preAdmissionDecodeCalls={backend.DecodeCallCount}, " +
    $"acceptedPrefillBatch=2, snapshots={backend.SnapshotCallCount}, kv={kvPool.AllocatedPages}/{kvPool.Capacity}.");

sealed class BlockingBackend : IInferenceBackend
{
    private readonly TaskCompletionSource<int> _prefillStarted =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource<bool> _allowPrefill =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource<KvSnapshotId> _snapshotStarted =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource<bool> _allowSnapshot =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private bool _initialized;
    private int _decodeCallCount;
    private int _snapshotCallCount;

    public BlockingBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => "blocking-cancellation-spec";
    public DeviceId Device { get; }
    public TaskCompletionSource<int> PrefillStarted => _prefillStarted;
    public TaskCompletionSource<KvSnapshotId> SnapshotStarted => _snapshotStarted;
    public int DecodeCallCount => Volatile.Read(ref _decodeCallCount);
    public int SnapshotCallCount => Volatile.Read(ref _snapshotCallCount);

    public void AllowPrefill() => _allowPrefill.TrySetResult(true);
    public void AllowSnapshot() => _allowSnapshot.TrySetResult(true);

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _initialized = true;
        return ValueTask.CompletedTask;
    }

    public async ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _prefillStarted.TrySetResult(batch.Items.Count);
        await _allowPrefill.Task.ConfigureAwait(false);

        return batch.Items
            .Select(static item => new BackendStepResult(item.SequenceId, 100 + item.Tokens.Length))
            .ToArray();
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _decodeCallCount);
        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
            batch.Items.Select(static item => new BackendStepResult(item.SequenceId, 200 + item.Position)).ToArray());
    }

    public async ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _snapshotCallCount);
        _snapshotStarted.TrySetResult(snapshotId);
        await _allowSnapshot.Task.ConfigureAwait(false);
    }

    public ValueTask DisposeAsync()
    {
        _allowPrefill.TrySetResult(true);
        _allowSnapshot.TrySetResult(true);
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Backend is not initialized.");
        }
    }
}
