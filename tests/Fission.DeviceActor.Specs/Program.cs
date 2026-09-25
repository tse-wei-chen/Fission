using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

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

Console.WriteLine(
    $"Fission device actor failure specs passed: cleanDisposes={cleanDisposeBackend.DisposeCount}, " +
    $"aggregateDisposes={doubleFailureBackend.DisposeCount}, errors={aggregateFailure.InnerExceptions.Count}.");

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
