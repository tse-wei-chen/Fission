using System.Threading.Channels;
using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

/// <summary>
/// Single-device actor that owns backend execution. Producers enqueue work;
/// one consumer drains currently available work into micro-batches and invokes
/// the backend serially. This keeps request tasks from racing the device.
/// </summary>
public sealed class ContinuousBatchExecutor : IAsyncDisposable
{
    private readonly IInferenceBackend _backend;
    private readonly Channel<PendingWork> _queue;
    private readonly int _maxBatchSize;
    private readonly Task _pump;
    private int _disposed;

    private ContinuousBatchExecutor(
        IInferenceBackend backend,
        int capacity,
        int maxBatchSize)
    {
        _backend = backend;
        _maxBatchSize = maxBatchSize;
        _queue = Channel.CreateBounded<PendingWork>(new BoundedChannelOptions(capacity)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.Wait
        });
        _pump = Task.Run(PumpAsync);
    }

    public DeviceId Device => _backend.Device;
    public string BackendName => _backend.Name;

    public static async ValueTask<ContinuousBatchExecutor> CreateAsync(
        IInferenceBackend backend,
        int capacity = 4096,
        int maxBatchSize = 64,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBatchSize);

        await backend.InitializeAsync(cancellationToken).ConfigureAwait(false);
        return new ContinuousBatchExecutor(backend, capacity, maxBatchSize);
    }

    public ValueTask<BackendStepResult> SubmitPrefillAsync(
        PrefillItem item,
        CancellationToken cancellationToken = default) =>
        SubmitAsync(new PendingPrefill(item), cancellationToken);

    public ValueTask<BackendStepResult> SubmitDecodeAsync(
        DecodeItem item,
        CancellationToken cancellationToken = default) =>
        SubmitAsync(new PendingDecode(item), cancellationToken);

    private async ValueTask<BackendStepResult> SubmitAsync(
        PendingWork work,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
        return await work.Completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task PumpAsync()
    {
        var reader = _queue.Reader;

        try
        {
            while (await reader.WaitToReadAsync().ConfigureAwait(false))
            {
                var batch = new List<PendingWork>(_maxBatchSize);
                while (batch.Count < _maxBatchSize && reader.TryRead(out var work))
                {
                    batch.Add(work);
                }

                await ExecuteBatchAsync(batch).ConfigureAwait(false);
            }
        }
        catch (Exception exception)
        {
            _queue.Writer.TryComplete(exception);
            while (reader.TryRead(out var work))
            {
                work.Completion.TrySetException(exception);
            }

            throw;
        }
    }

    private async Task ExecuteBatchAsync(IReadOnlyList<PendingWork> batch)
    {
        var prefills = batch.OfType<PendingPrefill>().ToArray();
        if (prefills.Length != 0)
        {
            var input = new PrefillBatch(prefills.Select(static work => work.Item).ToArray());
            var results = await _backend.PrefillAsync(input).ConfigureAwait(false);
            Complete(prefills, results);
        }

        var decodes = batch.OfType<PendingDecode>().ToArray();
        if (decodes.Length != 0)
        {
            var input = new DecodeBatch(decodes.Select(static work => work.Item).ToArray());
            var results = await _backend.DecodeAsync(input).ConfigureAwait(false);
            Complete(decodes, results);
        }
    }

    private static void Complete<TWork>(
        IReadOnlyList<TWork> work,
        IReadOnlyList<BackendStepResult> results)
        where TWork : PendingWork
    {
        if (work.Count != results.Count)
        {
            throw new InvalidOperationException(
                $"Backend returned {results.Count} results for {work.Count} work items.");
        }

        for (var i = 0; i < work.Count; i++)
        {
            work[i].Completion.TrySetResult(results[i]);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _queue.Writer.TryComplete();
        await _pump.ConfigureAwait(false);
        await _backend.DisposeAsync().ConfigureAwait(false);
    }

    private abstract class PendingWork
    {
        protected PendingWork()
        {
            Completion = new TaskCompletionSource<BackendStepResult>(
                TaskCreationOptions.RunContinuationsAsynchronously);
        }

        public TaskCompletionSource<BackendStepResult> Completion { get; }
    }

    private sealed class PendingPrefill(PrefillItem item) : PendingWork
    {
        public PrefillItem Item { get; } = item;
    }

    private sealed class PendingDecode(DecodeItem item) : PendingWork
    {
        public DecodeItem Item { get; } = item;
    }
}
