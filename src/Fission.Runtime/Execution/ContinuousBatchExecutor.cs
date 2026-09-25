using System.Runtime.ExceptionServices;
using System.Threading.Channels;
using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

/// <summary>
/// Single-device actor that owns backend execution. Producers enqueue work;
/// one consumer drains currently available work into micro-batches and invokes
/// the backend serially. Stateful control operations are queue-order barriers:
/// inference before a control is flushed first, the control executes, then later
/// inference may proceed.
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

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(
            new PendingSnapshot(sequenceId, snapshotId),
            cancellationToken);

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(
            new PendingFork(parentSequenceId, branchSequenceIds.ToArray()),
            cancellationToken);

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(
            new PendingRestore(sequenceId, snapshotId),
            cancellationToken);

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(new PendingReleaseSnapshot(snapshotId), cancellationToken);

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(new PendingReleaseSequence(sequenceId), cancellationToken);

    private async ValueTask<BackendStepResult> SubmitInferenceAsync<TWork>(
        TWork work,
        CancellationToken cancellationToken)
        where TWork : PendingInference
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
        return await work.Completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    private async ValueTask SubmitControlAsync(
        PendingControl work,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
        await work.Completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
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
        var inferenceSegment = new List<PendingInference>(_maxBatchSize);

        foreach (var work in batch)
        {
            if (work is PendingInference inference)
            {
                inferenceSegment.Add(inference);
                continue;
            }

            await ExecuteInferenceSegmentAsync(inferenceSegment).ConfigureAwait(false);
            inferenceSegment.Clear();

            var control = (PendingControl)work;
            await control.ExecuteAsync(_backend).ConfigureAwait(false);
            control.Completion.TrySetResult(true);
        }

        await ExecuteInferenceSegmentAsync(inferenceSegment).ConfigureAwait(false);
    }

    private async Task ExecuteInferenceSegmentAsync(
        IReadOnlyList<PendingInference> segment)
    {
        if (segment.Count == 0)
        {
            return;
        }

        var prefills = segment.OfType<PendingPrefill>().ToArray();
        if (prefills.Length != 0)
        {
            var input = new PrefillBatch(prefills.Select(static work => work.Item).ToArray());
            var results = await _backend.PrefillAsync(input).ConfigureAwait(false);
            Complete(prefills, results);
        }

        var decodes = segment.OfType<PendingDecode>().ToArray();
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

        ExceptionDispatchInfo? pumpFailure = null;
        ExceptionDispatchInfo? backendDisposeFailure = null;

        try
        {
            await _pump.ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            pumpFailure = ExceptionDispatchInfo.Capture(exception);
        }

        try
        {
            await _backend.DisposeAsync().ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            backendDisposeFailure = ExceptionDispatchInfo.Capture(exception);
        }

        if (pumpFailure is not null && backendDisposeFailure is not null)
        {
            throw new AggregateException(
                "Device actor and backend disposal both failed.",
                pumpFailure.SourceException,
                backendDisposeFailure.SourceException);
        }

        if (pumpFailure is not null)
        {
            pumpFailure.Throw();
        }

        if (backendDisposeFailure is not null)
        {
            backendDisposeFailure.Throw();
        }
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

    private abstract class PendingControl : PendingWork
    {
        protected PendingControl()
        {
            Completion = new TaskCompletionSource<bool>(
                TaskCreationOptions.RunContinuationsAsynchronously);
        }

        public TaskCompletionSource<bool> Completion { get; }
        public abstract ValueTask ExecuteAsync(IInferenceBackend backend);

        public override void Fail(Exception exception) =>
            Completion.TrySetException(exception);
    }

    private sealed class PendingSnapshot(
        SequenceId sequenceId,
        KvSnapshotId snapshotId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.SnapshotSequenceAsync(sequenceId, snapshotId);
    }

    private sealed class PendingFork(
        SequenceId parentSequenceId,
        SequenceId[] branchSequenceIds) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.ForkSequenceAsync(parentSequenceId, branchSequenceIds);
    }

    private sealed class PendingRestore(
        SequenceId sequenceId,
        KvSnapshotId snapshotId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.RestoreSequenceAsync(sequenceId, snapshotId);
    }

    private sealed class PendingReleaseSnapshot(
        KvSnapshotId snapshotId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.ReleaseSnapshotAsync(snapshotId);
    }

    private sealed class PendingReleaseSequence(
        SequenceId sequenceId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.ReleaseSequenceAsync(sequenceId);
    }
}
