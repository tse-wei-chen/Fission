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
        SubmitInferenceAsync(new PendingPrefill(item), cancellationToken);

    public ValueTask<BackendStepResult> SubmitDecodeAsync(
        DecodeItem item,
        CancellationToken cancellationToken = default) =>
        SubmitInferenceAsync(new PendingDecode(item), cancellationToken);

    public async ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var work = new PendingRelease(sequenceId);
        await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
        await work.Completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    private async ValueTask<BackendStepResult> SubmitInferenceAsync<TWork>(
        TWork work,
        CancellationToken cancellationToken)
        where TWork : PendingInference
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

                try
                {
                    await ExecuteBatchAsync(batch).ConfigureAwait(false);
                }
                catch (Exception exception)
                {
                    foreach (var work in batch)
                    {
                        work.Fail(exception);
                    }

                    throw;
                }
            }
        }
        catch (Exception exception)
        {
            _queue.Writer.TryComplete(exception);
            while (reader.TryRead(out var work))
            {
                work.Fail(exception);
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

        foreach (var release in batch.OfType<PendingRelease>())
        {
            await _backend.ReleaseSequenceAsync(release.SequenceId).ConfigureAwait(false);
            release.Completion.TrySetResult(true);
        }
    }

    private static void Complete<TWork>(
        IReadOnlyList<TWork> work,
        IReadOnlyList<BackendStepResult> results)
        where TWork : PendingInference
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
        public abstract void Fail(Exception exception);
    }

    private abstract class PendingInference : PendingWork
    {
        protected PendingInference()
        {
            Completion = new TaskCompletionSource<BackendStepResult>(
                TaskCreationOptions.RunContinuationsAsynchronously);
        }

        public TaskCompletionSource<BackendStepResult> Completion { get; }

        public override void Fail(Exception exception) =>
            Completion.TrySetException(exception);
    }

    private sealed class PendingPrefill(PrefillItem item) : PendingInference
    {
        public PrefillItem Item { get; } = item;
    }

    private sealed class PendingDecode(DecodeItem item) : PendingInference
    {
        public DecodeItem Item { get; } = item;
    }

    private sealed class PendingRelease(SequenceId sequenceId) : PendingWork
    {
        public SequenceId SequenceId { get; } = sequenceId;
        public TaskCompletionSource<bool> Completion { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public override void Fail(Exception exception) =>
            Completion.TrySetException(exception);
    }
}
