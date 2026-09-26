using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static async Task RequirePrefillFaultAsync(
    ContinuousBatchExecutor executor,
    ModelId modelId)
{
    var sequenceId = SequenceId.New();
    var faulted = false;

    try
    {
        await executor.SubmitPrefillAsync(
                new PrefillItem(
                    sequenceId,
                    modelId,
                    new ReadOnlyMemory<int>(new[] { 1, 2, 3 })))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));
    }
    catch (InvalidOperationException exception) when (exception.Message == FailingBackend.PumpFailureMessage)
    {
        faulted = true;
    }

    Require(
        faulted,
        "A backend execution fault must complete the current request with the same exception instead of hanging its ticket.");
}

var model = new ModelId("device-actor-failure-spec");

var cleanDisposeBackend = new FailingBackend(
    new DeviceId("cpu:failure-cleanup"),
    throwOnDispose: false);
var cleanDisposeExecutor = await ContinuousBatchExecutor.CreateAsync(
    cleanDisposeBackend,
    capacity: 8,
    maxBatchSize: 4);

await RequirePrefillFaultAsync(cleanDisposeExecutor, model);

var pumpFailurePropagated = false;
try
{
    await cleanDisposeExecutor.DisposeAsync();
}
catch (InvalidOperationException exception) when (exception.Message == FailingBackend.PumpFailureMessage)
{
    pumpFailurePropagated = true;
}

Require(pumpFailurePropagated, "Disposal must preserve the original device-pump failure.");
Require(cleanDisposeBackend.DisposeCount == 1, "Backend cleanup must run even after the device pump faults.");
Require(cleanDisposeBackend.IsDisposed, "Backend cleanup must complete before DisposeAsync returns the pump failure.");

// DisposeAsync is idempotent even when its first invocation reported the pump
// failure; backend resources must not be released twice.
await cleanDisposeExecutor.DisposeAsync();
Require(cleanDisposeBackend.DisposeCount == 1, "Repeated executor disposal must not invoke backend cleanup twice.");

var doubleFailureBackend = new FailingBackend(
    new DeviceId("cpu:failure-aggregate"),
    throwOnDispose: true);
var doubleFailureExecutor = await ContinuousBatchExecutor.CreateAsync(
    doubleFailureBackend,
    capacity: 8,
    maxBatchSize: 4);

await RequirePrefillFaultAsync(doubleFailureExecutor, model);

AggregateException? aggregateFailure = null;
try
{
    await doubleFailureExecutor.DisposeAsync();
}
catch (AggregateException exception)
{
    aggregateFailure = exception.Flatten();
}

Require(aggregateFailure is not null, "Pump and backend cleanup faults must be reported together.");
Require(
    aggregateFailure!.InnerExceptions.Any(static exception =>
        exception is InvalidOperationException && exception.Message == FailingBackend.PumpFailureMessage),
    "Aggregate disposal failure must preserve the original device-pump exception.");
Require(
    aggregateFailure.InnerExceptions.Any(static exception =>
        exception is InvalidOperationException && exception.Message == FailingBackend.DisposeFailureMessage),
    "Aggregate disposal failure must preserve the backend cleanup exception.");
Require(doubleFailureBackend.DisposeCount == 1, "Failing backend cleanup must still be attempted exactly once.");
Require(doubleFailureBackend.IsDisposed, "Backend must record the cleanup attempt before its disposal exception escapes.");

await doubleFailureExecutor.DisposeAsync();
Require(doubleFailureBackend.DisposeCount == 1, "Repeated disposal after an aggregate failure must remain idempotent.");

// Scheduler-selected mixed work reaches the actor in queue order. The actor may
// coalesce adjacent work of one kind, but it must not move a later prefill in
// front of an earlier decode just to form a larger homogeneous batch.
var orderingBackend = new RecordingBackend(new DeviceId("cpu:ordering"));
await using (var orderingExecutor = await ContinuousBatchExecutor.CreateAsync(
    orderingBackend,
    capacity: 16,
    maxBatchSize: 16))
{
    var firstDecodeId = SequenceId.New();
    var prefillId = SequenceId.New();
    var secondDecodeId = SequenceId.New();

    var firstDecode = orderingExecutor.SubmitDecodeAsync(
        new DecodeItem(firstDecodeId, model, Position: 1)).AsTask();
    var prefill = orderingExecutor.SubmitPrefillAsync(
        new PrefillItem(
            prefillId,
            model,
            new ReadOnlyMemory<int>(new[] { 7, 8 }),
            Position: 0)).AsTask();
    var secondDecode = orderingExecutor.SubmitDecodeAsync(
        new DecodeItem(secondDecodeId, model, Position: 4)).AsTask();

    await Task.WhenAll(firstDecode, prefill, secondDecode)
        .WaitAsync(TimeSpan.FromSeconds(5));
}

var orderedKinds = orderingBackend.Events
    .Select(static entry => entry.Kind)
    .ToArray();
Require(
    orderedKinds.SequenceEqual(new[] { "decode", "prefill", "decode" }),
    "Mixed device inference must preserve queue order instead of moving prefills ahead of reserved decodes.");
Require(
    orderingBackend.Events[0].SequenceIds.Count == 1 &&
    orderingBackend.Events[1].SequenceIds.Count == 1 &&
    orderingBackend.Events[2].SequenceIds.Count == 1,
    "Kind switches must form separate contiguous backend batches.");

// ScheduledBatchExecutor validates all work first, then registers each one-step
// runtime plan into a fixed scheduler-order slot. Device capacity is now measured
// in inference-item credits, so this executor grants enough credits for the full
// scheduled envelope while still proving deterministic batch membership/order.
var atomicBackend = new RecordingBackend(new DeviceId("cpu:atomic-schedule"));
await using (var atomicDevice = await ContinuousBatchExecutor.CreateAsync(
    atomicBackend,
    capacity: 4,
    maxBatchSize: 16))
{
    var kvPool = new KvPagePool(capacity: 16, tokensPerPage: 4);
    using var atomicRuntime = new ExecutionPlanExecutor(
        atomicDevice,
        kvPagePool: kvPool);
    var scheduled = new ScheduledBatchExecutor(atomicRuntime);

    var first = SequenceId.New();
    var second = SequenceId.New();
    var third = SequenceId.New();
    var initial = new ScheduledBatch(
        Guid.NewGuid(),
        new ScheduledWorkItem[]
        {
            new(first, ScheduledWorkKind.Prefill, 1, 1, 10, CompletesPrefill: true),
            new(second, ScheduledWorkKind.Prefill, 1, 1, 9, CompletesPrefill: true),
            new(third, ScheduledWorkKind.Prefill, 1, 1, 8, CompletesPrefill: true)
        },
        ConsumedTokens: 3,
        ConsumedKvPages: 3);
    var initialBindings = new ScheduledExecutionBindings(
        new Dictionary<SequenceId, ScheduledPrefillBinding>
        {
            [first] = new(model, new ReadOnlyMemory<int>(new[] { 11 })),
            [second] = new(model, new ReadOnlyMemory<int>(new[] { 12 })),
            [third] = new(model, new ReadOnlyMemory<int>(new[] { 13 }))
        });

    var initialResult = await scheduled.ExecuteAsync(initial, initialBindings)
        .AsTask()
        .WaitAsync(TimeSpan.FromSeconds(5));
    Require(initialResult.ItemResults.Count == 3, "Atomic scheduled prefill must return one runtime result per item.");
    Require(atomicBackend.Events.Count == 1, "Three scheduled prefills must reach the backend as one physical call.");
    Require(
        atomicBackend.Events[0].Kind == "prefill" &&
        atomicBackend.Events[0].SequenceIds.SequenceEqual(new[] { first, second, third }),
        "Atomic scheduled prefill must preserve scheduler order and batch membership.");

    atomicBackend.Events.Clear();
    var fourth = SequenceId.New();
    var fifth = SequenceId.New();
    var mixed = new ScheduledBatch(
        Guid.NewGuid(),
        new ScheduledWorkItem[]
        {
            new(first, ScheduledWorkKind.Decode, 1, 0, 10, CompletesPrefill: false),
            new(fourth, ScheduledWorkKind.Prefill, 1, 1, 7, CompletesPrefill: true),
            new(fifth, ScheduledWorkKind.Prefill, 1, 1, 6, CompletesPrefill: true),
            new(second, ScheduledWorkKind.Decode, 1, 0, 5, CompletesPrefill: false)
        },
        ConsumedTokens: 4,
        ConsumedKvPages: 2);
    var mixedBindings = new ScheduledExecutionBindings(
        new Dictionary<SequenceId, ScheduledPrefillBinding>
        {
            [fourth] = new(model, new ReadOnlyMemory<int>(new[] { 21 })),
            [fifth] = new(model, new ReadOnlyMemory<int>(new[] { 22 }))
        });

    await scheduled.ExecuteAsync(mixed, mixedBindings)
        .AsTask()
        .WaitAsync(TimeSpan.FromSeconds(5));

    Require(atomicBackend.Events.Count == 3, "Mixed scheduled envelope must execute exactly three contiguous kind segments.");
    Require(
        atomicBackend.Events[0].Kind == "decode" &&
        atomicBackend.Events[0].SequenceIds.SequenceEqual(new[] { first }),
        "Mixed scheduled envelope must start with the first reserved decode.");
    Require(
        atomicBackend.Events[1].Kind == "prefill" &&
        atomicBackend.Events[1].SequenceIds.SequenceEqual(new[] { fourth, fifth }),
        "Adjacent scheduled prefills must remain one physical backend batch inside the envelope.");
    Require(
        atomicBackend.Events[2].Kind == "decode" &&
        atomicBackend.Events[2].SequenceIds.SequenceEqual(new[] { second }),
        "Mixed scheduled envelope must preserve the final decode after the prefill segment.");
}

Console.WriteLine(
    $"Fission device actor specs passed: cleanDisposes={cleanDisposeBackend.DisposeCount}, " +
    $"aggregateDisposes={doubleFailureBackend.DisposeCount}, errors={aggregateFailure.InnerExceptions.Count}, " +
    $"mixedOrder={string.Join("->", orderedKinds)}, atomicPrefill=3, atomicMixed=decode->prefill(2)->decode.");

sealed class FailingBackend : IInferenceBackend
{
    public const string PumpFailureMessage = "backend execution boom";
    public const string DisposeFailureMessage = "backend dispose boom";

    private readonly bool _throwOnDispose;
    private bool _initialized;

    public FailingBackend(DeviceId device, bool throwOnDispose)
    {
        Device = device;
        _throwOnDispose = throwOnDispose;
    }

    public string Name => "failing-spec-backend";
    public DeviceId Device { get; }
    public int DisposeCount { get; private set; }
    public bool IsDisposed { get; private set; }

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
        throw new InvalidOperationException(PumpFailureMessage);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        throw new InvalidOperationException(PumpFailureMessage);
    }

    public ValueTask DisposeAsync()
    {
        DisposeCount++;
        IsDisposed = true;
        _initialized = false;

        if (_throwOnDispose)
        {
            throw new InvalidOperationException(DisposeFailureMessage);
        }

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

sealed class RecordingBackend : IInferenceBackend
{
    private bool _initialized;

    public RecordingBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => "recording-order-backend";
    public DeviceId Device { get; }
    public List<BackendEvent> Events { get; } = new();

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
        var ids = batch.Items.Select(static item => item.SequenceId).ToArray();
        Events.Add(new BackendEvent("prefill", ids));
        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
            ids.Select(static id => new BackendStepResult(id, TokenId: 101)).ToArray());
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var ids = batch.Items.Select(static item => item.SequenceId).ToArray();
        Events.Add(new BackendEvent("decode", ids));
        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
            ids.Select(static id => new BackendStepResult(id, TokenId: 202)).ToArray());
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
            throw new InvalidOperationException("Backend is not initialized.");
        }
    }
}

sealed record BackendEvent(string Kind, IReadOnlyList<SequenceId> SequenceIds);
