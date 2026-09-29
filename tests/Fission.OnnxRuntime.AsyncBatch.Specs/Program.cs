using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

const string MulModelBase64 =
    "CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0" +
    "KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4K" +
    "DAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==";

var modelId = new ModelId("async-batch-toy");
var binding = new AsyncPreferredBinding();
var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);
await using var backend = new OnnxRuntimeBackend(
    new OnnxRuntimeBackendOptions(
        modelId,
        new DeviceId("cpu:async-batch-spec"),
        OnnxRuntimeModelSource.FromBytes(Convert.FromBase64String(MulModelBase64))),
    adapter);

await backend.InitializeAsync();

var first = SequenceId.New();
var second = SequenceId.New();
var prefillTask = backend.PrefillAsync(new PrefillBatch(new[]
{
    new PrefillItem(first, modelId, new ReadOnlyMemory<int>(new[] { 2 })),
    new PrefillItem(second, modelId, new ReadOnlyMemory<int>(new[] { 5 }))
})).AsTask();

await binding.PrefillEntered;
Require(!prefillTask.IsCompleted,
    "Adapter must await the asynchronous batch capability instead of completing early.");
Require(binding.AsyncPrefillCalls == 1,
    "Initial prefill must invoke the async batch capability exactly once.");
Require(binding.SyncPrefillBatchCalls == 0 && binding.ScalarPrefillCalls == 0,
    "Async batch prefill must take precedence over synchronous batch/scalar fallbacks.");

binding.ReleasePrefill();
var prefill = await prefillTask;
Require(prefill.Count == 2 && prefill[0].TokenId == 12 && prefill[1].TokenId == 15,
    "Async prefill results must preserve item ordering and tokens.");

using var cancellation = new CancellationTokenSource();
binding.CancelOnNextDecode(cancellation);
var canceled = false;
try
{
    _ = await backend.DecodeAsync(
        new DecodeBatch(new[]
        {
            new DecodeItem(first, modelId, Position: 1),
            new DecodeItem(second, modelId, Position: 1)
        }),
        cancellation.Token);
}
catch (OperationCanceledException)
{
    canceled = true;
}

Require(canceled,
    "Cancellation triggered after async batch production must abort before state-store commit.");
Require(binding.AsyncDecodeCalls == 1,
    "Canceled decode must still have used the async batch capability exactly once.");
Require(binding.SyncDecodeBatchCalls == 0 && binding.ScalarDecodeCalls == 0,
    "Async decode must take precedence over synchronous batch/scalar fallbacks.");
Require(binding.LastDecodeStates is { Count: 2 } canceledStates &&
        canceledStates.All(static state => state.IsDisposed),
    "Adapter must dispose every async-produced state when cancellation wins before commit.");

// The canceled frontier must not replace the old position-1 states. Re-running
// the same decode position proves the store remained transactional.
var decoded = await backend.DecodeAsync(new DecodeBatch(new[]
{
    new DecodeItem(first, modelId, Position: 1),
    new DecodeItem(second, modelId, Position: 1)
}));
Require(decoded.Count == 2 && decoded[0].TokenId == 13 && decoded[1].TokenId == 16,
    "After canceled async decode, the same prior frontier must remain reusable.");
Require(binding.AsyncDecodeCalls == 2,
    "Successful retry must invoke async decode again.");
Require(binding.SyncDecodeBatchCalls == 0 && binding.ScalarDecodeCalls == 0,
    "Retry must continue to prefer async batch execution.");

await backend.ReleaseSequenceAsync(first);
await backend.ReleaseSequenceAsync(second);

Console.WriteLine(
    $"Fission async decoder batch specs passed: asyncPrefill={binding.AsyncPrefillCalls}, " +
    $"asyncDecode={binding.AsyncDecodeCalls}, syncPrefill={binding.SyncPrefillBatchCalls}, " +
    $"syncDecode={binding.SyncDecodeBatchCalls}.");

sealed class AsyncPreferredBinding :
    IDecoderOrtAsyncBatchPrefillModelBinding,
    IDecoderOrtAsyncBatchModelBinding,
    IDecoderOrtBatchPrefillModelBinding,
    IDecoderOrtBatchModelBinding
{
    private readonly TaskCompletionSource _prefillEntered =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _releasePrefill =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private CancellationTokenSource? _cancelDecode;
    private int _disposed;

    public string Name => "async-preferred-toy";
    public int AsyncPrefillCalls { get; private set; }
    public int SyncPrefillBatchCalls { get; private set; }
    public int ScalarPrefillCalls { get; private set; }
    public int AsyncDecodeCalls { get; private set; }
    public int SyncDecodeBatchCalls { get; private set; }
    public int ScalarDecodeCalls { get; private set; }
    public IReadOnlyList<DecoderOrtState>? LastDecodeStates { get; private set; }
    public Task PrefillEntered => _prefillEntered.Task;

    public OnnxSessionContract SessionContract { get; } = new(
        Inputs: new[]
        {
            new OnnxTensorContract("input", "X", Rank: 2, ElementType: TensorElementType.Float)
        },
        Outputs: new[]
        {
            new OnnxTensorContract("output", "Y", Rank: 2, ElementType: TensorElementType.Float)
        });

    public void ReleasePrefill() => _releasePrefill.TrySetResult();

    public void CancelOnNextDecode(CancellationTokenSource cancellation)
    {
        ArgumentNullException.ThrowIfNull(cancellation);
        _cancelDecode = cancellation;
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        ScalarPrefillCalls++;
        throw new InvalidOperationException("Scalar prefill must not run when async batch prefill is available.");
    }

    public async ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecutePrefillBatchAsync(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default)
    {
        EnsureAlive();
        AsyncPrefillCalls++;
        _prefillEntered.TrySetResult();
        await _releasePrefill.Task.WaitAsync(cancellationToken).ConfigureAwait(false);

        if (items.Count != 2 || priorStates.Count != 2 ||
            priorStates.Any(static state => state is not null))
        {
            throw new InvalidOperationException("Async batch spec expects two initial-prefill items.");
        }

        return new[]
        {
            CreateResult(items[0].Position!.Value + items[0].Tokens.Length, items[0].Tokens.Span[^1] + 10),
            CreateResult(items[1].Position!.Value + items[1].Tokens.Length, items[1].Tokens.Span[^1] + 10)
        };
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecutePrefillBatch(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default)
    {
        SyncPrefillBatchCalls++;
        throw new InvalidOperationException("Synchronous batch prefill must not run when async capability is available.");
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ScalarDecodeCalls++;
        throw new InvalidOperationException("Scalar decode must not run when async batch decode is available.");
    }

    public ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecuteDecodeBatchAsync(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default)
    {
        EnsureAlive();
        AsyncDecodeCalls++;
        if (items.Count != 2 || priorStates.Count != 2)
        {
            throw new InvalidOperationException("Async batch spec expects two decode items.");
        }

        var states = new[]
        {
            CreateState(items[0].Position + 1, checked(priorStates[0].NextTokenId!.Value + 1)),
            CreateState(items[1].Position + 1, checked(priorStates[1].NextTokenId!.Value + 1))
        };
        LastDecodeStates = states;
        var results = new DecoderOrtStepResult[]
        {
            new(states[0].NextTokenId!.Value, states[0]),
            new(states[1].NextTokenId!.Value, states[1])
        };

        Interlocked.Exchange(ref _cancelDecode, null)?.Cancel();
        return ValueTask.FromResult<IReadOnlyList<DecoderOrtStepResult>>(results);
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecuteDecodeBatch(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default)
    {
        SyncDecodeBatchCalls++;
        throw new InvalidOperationException("Synchronous batch decode must not run when async capability is available.");
    }

    private static DecoderOrtStepResult CreateResult(int position, int token) =>
        new(token, CreateState(position, token));

    private static DecoderOrtState CreateState(int position, int nextToken)
    {
        var key = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)nextToken },
            new long[] { 1, 1, 1, 1 });
        var value = OrtValue.CreateTensorValueFromMemory(
            new[] { (float)nextToken },
            new long[] { 1, 1, 1, 1 });
        return new DecoderOrtState(
            position,
            new[] { new DecoderOrtLayerState(key, value) },
            nextTokenId: nextToken);
    }

    private void EnsureAlive() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose() => Interlocked.Exchange(ref _disposed, 1);
}
