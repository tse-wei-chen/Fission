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

var deviceId = new DeviceId("cpu:identity-spec");
var modelId = new ModelId("identity-spec-model");
var backend = new MisorderedBackend(deviceId);
var device = await ContinuousBatchExecutor.CreateAsync(
    backend,
    capacity: 2,
    maxBatchSize: 2);

var kvPool = new KvPagePool(capacity: 8, tokensPerPage: 4);
using var runtime = new ExecutionPlanExecutor(device, kvPagePool: kvPool);
var scheduled = new ScheduledBatchExecutor(runtime);

var first = SequenceId.New();
var second = SequenceId.New();
var batch = new ScheduledBatch(
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
        [first] = new(modelId, new ReadOnlyMemory<int>(new[] { 11 })),
        [second] = new(modelId, new ReadOnlyMemory<int>(new[] { 12 }))
    });

InvalidOperationException? identityFailure = null;
try
{
    await scheduled.ExecuteAsync(batch, bindings)
        .AsTask()
        .WaitAsync(TimeSpan.FromSeconds(5));
}
catch (InvalidOperationException exception)
    when (exception.Message.Contains("corresponding work item belongs to", StringComparison.Ordinal))
{
    identityFailure = exception;
}

Require(identityFailure is not null, "Misordered backend result identities must fault the whole scheduled batch.");
Require(backend.PrefillCalls == 1, "The identity spec must execute exactly one physical prefill batch.");
Require(backend.LastPrefillBatchSize == 2, "Both scheduled prefills must reach the backend in the same physical batch.");
Require(runtime.SequenceCount == 2, "Both admitted runtime sequence objects should remain inspectable after backend failure.");
Require(runtime.TryGetSequence(first, out var firstSequence) && firstSequence is not null, "First runtime sequence must exist after failure.");
Require(runtime.TryGetSequence(second, out var secondSequence) && secondSequence is not null, "Second runtime sequence must exist after failure.");
Require(firstSequence!.Position == 0 && firstSequence.Kv.Count == 0, "First sequence must not commit position or KV after identity mismatch.");
Require(secondSequence!.Position == 0 && secondSequence.Kv.Count == 0, "Second sequence must not commit position or KV after identity mismatch.");
Require(kvPool.AllocatedPages == 0, "Identity mismatch must not allocate metadata KV pages.");

InvalidOperationException? disposeFailure = null;
try
{
    await device.DisposeAsync();
}
catch (InvalidOperationException exception)
    when (exception.Message.Contains("corresponding work item belongs to", StringComparison.Ordinal))
{
    disposeFailure = exception;
}

Require(disposeFailure is not null, "Device disposal must preserve the backend contract violation from the pump.");
Require(backend.DisposeCount == 1, "Backend cleanup must run exactly once after identity mismatch.");

Console.WriteLine(
    $"Fission device actor identity specs passed: batch={backend.LastPrefillBatchSize}, " +
    $"positions={firstSequence.Position},{secondSequence.Position}, kv={kvPool.AllocatedPages}.");

sealed class MisorderedBackend : IInferenceBackend
{
    private bool _initialized;

    public MisorderedBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => "misordered-result-backend";
    public DeviceId Device { get; }
    public int PrefillCalls { get; private set; }
    public int LastPrefillBatchSize { get; private set; }
    public int DisposeCount { get; private set; }

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
        PrefillCalls++;
        LastPrefillBatchSize = batch.Items.Count;

        if (batch.Items.Count != 2)
        {
            throw new InvalidOperationException("Identity spec expects exactly two prefill items.");
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
            new[]
            {
                new BackendStepResult(batch.Items[1].SequenceId, TokenId: 202),
                new BackendStepResult(batch.Items[0].SequenceId, TokenId: 101)
            });
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("Decode is not used by the identity spec.");

    public ValueTask DisposeAsync()
    {
        DisposeCount++;
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
