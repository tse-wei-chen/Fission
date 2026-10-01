using System.Threading.Channels;
using Fission.Abstractions;
using Fission.Abstractions.Scheduling;

namespace Fission.Engine;

public sealed record InferenceWorkerOptions(int AdmissionCapacity = 1024);

public sealed class InferenceStream
{
    private readonly ChannelReader<int> _tokens;
    private readonly Func<SequenceId, CancellationToken, ValueTask<InferenceRequestSnapshot>> _cancel;

    internal InferenceStream(
        SequenceId sequenceId,
        ChannelReader<int> tokens,
        Task<InferenceRequestSnapshot> completion,
        Func<SequenceId, CancellationToken, ValueTask<InferenceRequestSnapshot>> cancel)
    {
        SequenceId = sequenceId;
        _tokens = tokens;
        Completion = completion;
        _cancel = cancel;
    }

    public SequenceId SequenceId { get; }
    public Task<InferenceRequestSnapshot> Completion { get; }

    public IAsyncEnumerable<int> ReadTokensAsync(
        CancellationToken cancellationToken = default) =>
        _tokens.ReadAllAsync(cancellationToken);

    public ValueTask<InferenceRequestSnapshot> CancelAsync(
        CancellationToken cancellationToken = default) =>
        _cancel(SequenceId, cancellationToken);
}

/// <summary>
/// Single scheduler actor for high-concurrency serving. Producers enqueue
/// admission requests through a bounded channel; one pump owns scheduler-cycle
/// progression, serialized cancellation, and per-request token fan-out.
/// </summary>
public sealed class InferenceWorker : IAsyncDisposable
{
    private readonly InferenceEngine _engine;
    private readonly Channel<PendingSubmission> _admission;
    private readonly Channel<PendingCancellation> _cancellation;
    private readonly Channel<byte> _workSignal;
    private readonly Dictionary<SequenceId, SessionState> _sessions = new();
    private readonly Task _pump;
    private int _disposed;

    public InferenceWorker(
        InferenceEngine engine,
        InferenceWorkerOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(engine);
        options ??= new InferenceWorkerOptions();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.AdmissionCapacity);

        _engine = engine;
        _admission = Channel.CreateBounded<PendingSubmission>(
            new BoundedChannelOptions(options.AdmissionCapacity)
            {
                SingleReader = true,
                SingleWriter = false,
                FullMode = BoundedChannelFullMode.Wait
            });
        _cancellation = Channel.CreateUnbounded<PendingCancellation>(
            new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false,
                AllowSynchronousContinuations = false
            });
        _workSignal = Channel.CreateBounded<byte>(
            new BoundedChannelOptions(1)
            {
                SingleReader = true,
                SingleWriter = false,
                FullMode = BoundedChannelFullMode.DropWrite,
                AllowSynchronousContinuations = false
            });
        _pump = Task.Run(PumpAsync);
    }

    public async ValueTask<InferenceStream> SubmitAsync(
        ModelId modelId,
        ReadOnlyMemory<int> promptTokens,
        int maxNewTokens,
        int priority = 0,
        DateTimeOffset? deadline = null,
        DateTimeOffset? enqueuedAt = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        var pending = new PendingSubmission(
            modelId,
            promptTokens.ToArray(),
            maxNewTokens,
            priority,
            deadline,
            enqueuedAt ?? DateTimeOffset.UtcNow);

        await _admission.Writer.WriteAsync(pending, cancellationToken)
            .ConfigureAwait(false);
        SignalWork();

        // Once admitted to the bounded queue, ownership has transferred to the
        // worker. Do not abandon the accepted task because that would create an
        // orphan request after producer-side cancellation.
        return await pending.Accepted.Task.ConfigureAwait(false);
    }

    private async Task PumpAsync()
    {
        Exception? failure = null;

        try
        {
            while (true)
            {
                if (_engine.ActiveRequestCount == 0 &&
                    !HasImmediatelyAvailableWork() &&
                    !await WaitForWorkAsync().ConfigureAwait(false))
                {
                    break;
                }

                await DrainCancellationsAsync().ConfigureAwait(false);
                DrainAdmissions();

                if (_engine.ActiveRequestCount == 0)
                {
                    continue;
                }

                var cycle = await _engine.RunCycleAsync(DateTimeOffset.UtcNow)
                    .ConfigureAwait(false);

                PublishDecodeTokens(cycle);
                CompleteFinishedSessions(cycle);

                if (cycle.Batch.Items.Count == 0 && _engine.ActiveRequestCount != 0)
                {
                    var backpressureWait = _engine.GetDeviceMemoryBackpressureWait(cycle);
                    if (backpressureWait is null)
                    {
                        throw new InvalidOperationException(
                            "Inference worker made no scheduling progress while active requests remain.");
                    }

                    await WaitForBackpressureOrWorkAsync(backpressureWait)
                        .ConfigureAwait(false);
                }
            }
        }
        catch (Exception exception)
        {
            failure = exception;
            throw;
        }
        finally
        {
            FailPendingAdmissions(failure);
            FailPendingCancellations(failure);
            CompleteOpenSessions(failure);
        }
    }

    private bool HasImmediatelyAvailableWork() =>
        _admission.Reader.TryPeek(out _) || _cancellation.Reader.TryPeek(out _);

    private async ValueTask<bool> WaitForWorkAsync(
        CancellationToken cancellationToken = default)
    {
        while (await _workSignal.Reader.WaitToReadAsync(cancellationToken)
                   .ConfigureAwait(false))
        {
            while (_workSignal.Reader.TryRead(out _))
            {
            }

            // A capacity-one signal may remain buffered after a busy pump already
            // drained all corresponding commands. Ignore that stale notification
            // and continue waiting instead of spinning one scheduler iteration.
            if (HasImmediatelyAvailableWork())
            {
                return true;
            }
        }

        return false;
    }

    private async Task WaitForBackpressureOrWorkAsync(Task backpressureWait)
    {
        if (backpressureWait.IsCompleted)
        {
            await backpressureWait.ConfigureAwait(false);
            return;
        }

        using var workWaitCancellation = new CancellationTokenSource();
        var workWait = WaitForWorkAsync(workWaitCancellation.Token).AsTask();
        var completed = await Task.WhenAny(backpressureWait, workWait)
            .ConfigureAwait(false);

        if (ReferenceEquals(completed, backpressureWait))
        {
            workWaitCancellation.Cancel();
            await backpressureWait.ConfigureAwait(false);
            await SuppressExpectedCancellationAsync(workWait).ConfigureAwait(false);
            return;
        }

        var hasWork = await workWait.ConfigureAwait(false);
        workWaitCancellation.Cancel();
        if (hasWork)
        {
            return;
        }

        // Both control channels are closed (typically worker disposal), but an
        // already-admitted request still owns the worker. Let the shared-runtime
        // reservation resolve rather than spinning on completed channel waits.
        await backpressureWait.ConfigureAwait(false);
    }

    private static async Task SuppressExpectedCancellationAsync(Task workWait)
    {
        try
        {
            await workWait.ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Cancellation only stops the losing readiness wait after the
            // reservation-release signal won Task.WhenAny.
        }
    }

    private void SignalWork() => _workSignal.Writer.TryWrite(0);

    private void DrainAdmissions()
    {
        while (_admission.Reader.TryRead(out var pending))
        {
            Admit(pending);
        }
    }

    private async ValueTask DrainCancellationsAsync()
    {
        while (_cancellation.Reader.TryRead(out var pending))
        {
            try
            {
                var snapshot = await _engine.CancelAsync(pending.SequenceId)
                    .ConfigureAwait(false);

                if (_sessions.Remove(pending.SequenceId, out var session))
                {
                    session.Writer.TryComplete();
                    session.Completion.TrySetResult(snapshot);
                }

                pending.Completed.TrySetResult(snapshot);
            }
            catch (Exception exception)
            {
                pending.Completed.TrySetException(exception);
            }
        }
    }

    private void Admit(PendingSubmission pending)
    {
        try
        {
            var sequenceId = _engine.Submit(
                pending.ModelId,
                pending.PromptTokens,
                pending.MaxNewTokens,
                pending.Priority,
                pending.Deadline,
                pending.EnqueuedAt);

            var tokenChannel = Channel.CreateUnbounded<int>(
                new UnboundedChannelOptions
                {
                    SingleReader = true,
                    SingleWriter = true,
                    AllowSynchronousContinuations = false
                });
            var completion = new TaskCompletionSource<InferenceRequestSnapshot>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            var stream = new InferenceStream(
                sequenceId,
                tokenChannel.Reader,
                completion.Task,
                RequestCancellationAsync);

            _sessions.Add(
                sequenceId,
                new SessionState(tokenChannel.Writer, completion));
            pending.Accepted.TrySetResult(stream);
        }
        catch (Exception exception)
        {
            pending.Accepted.TrySetException(exception);
        }
    }

    private async ValueTask<InferenceRequestSnapshot> RequestCancellationAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        var pending = new PendingCancellation(sequenceId);
        await _cancellation.Writer.WriteAsync(pending, cancellationToken)
            .ConfigureAwait(false);
        SignalWork();
        return await pending.Completed.Task.WaitAsync(cancellationToken)
            .ConfigureAwait(false);
    }

    private void PublishDecodeTokens(InferenceCycleResult cycle)
    {
        foreach (var item in cycle.Batch.Items)
        {
            if (item.Kind != ScheduledWorkKind.Decode ||
                !_sessions.TryGetValue(item.SequenceId, out var session))
            {
                continue;
            }

            var snapshot = _engine.GetSnapshot(item.SequenceId);
            if (snapshot.GeneratedTokens.Count <= session.PublishedTokenCount)
            {
                throw new InvalidOperationException(
                    $"Decode work for {item.SequenceId} did not produce a new token.");
            }

            while (session.PublishedTokenCount < snapshot.GeneratedTokens.Count)
            {
                var token = snapshot.GeneratedTokens[session.PublishedTokenCount];
                if (!session.Writer.TryWrite(token))
                {
                    throw new InvalidOperationException(
                        $"Token stream for {item.SequenceId} closed before request completion.");
                }

                session.PublishedTokenCount++;
            }
        }
    }

    private void CompleteFinishedSessions(InferenceCycleResult cycle)
    {
        foreach (var sequenceId in cycle.CompletedSequences)
        {
            if (!_sessions.Remove(sequenceId, out var session))
            {
                throw new InvalidOperationException(
                    $"Completed request {sequenceId} has no worker session.");
            }

            var snapshot = _engine.GetSnapshot(sequenceId);
            session.Writer.TryComplete();
            session.Completion.TrySetResult(snapshot);
        }
    }

    private void FailPendingAdmissions(Exception? failure)
    {
        while (_admission.Reader.TryRead(out var pending))
        {
            if (failure is null)
            {
                pending.Accepted.TrySetException(
                    new ObjectDisposedException(nameof(InferenceWorker)));
            }
            else
            {
                pending.Accepted.TrySetException(failure);
            }
        }
    }

    private void FailPendingCancellations(Exception? failure)
    {
        while (_cancellation.Reader.TryRead(out var pending))
        {
            if (failure is null)
            {
                pending.Completed.TrySetException(
                    new ObjectDisposedException(nameof(InferenceWorker)));
            }
            else
            {
                pending.Completed.TrySetException(failure);
            }
        }
    }

    private void CompleteOpenSessions(Exception? failure)
    {
        foreach (var session in _sessions.Values)
        {
            if (failure is null)
            {
                var exception = new ObjectDisposedException(nameof(InferenceWorker));
                session.Writer.TryComplete(exception);
                session.Completion.TrySetException(exception);
            }
            else
            {
                session.Writer.TryComplete(failure);
                session.Completion.TrySetException(failure);
            }
        }

        _sessions.Clear();
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _admission.Writer.TryComplete();
        _cancellation.Writer.TryComplete();

        // Wake the pump even when a command was accepted just before the command
        // writers were closed but its producer had not published the coalesced
        // readiness signal yet. Buffered work remains visible before shutdown.
        SignalWork();
        _workSignal.Writer.TryComplete();
        await _pump.ConfigureAwait(false);
    }

    private sealed record PendingSubmission(
        ModelId ModelId,
        int[] PromptTokens,
        int MaxNewTokens,
        int Priority,
        DateTimeOffset? Deadline,
        DateTimeOffset EnqueuedAt)
    {
        public TaskCompletionSource<InferenceStream> Accepted { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
    }

    private sealed record PendingCancellation(SequenceId SequenceId)
    {
        public TaskCompletionSource<InferenceRequestSnapshot> Completed { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
    }

    private sealed class SessionState
    {
        public SessionState(
            ChannelWriter<int> writer,
            TaskCompletionSource<InferenceRequestSnapshot> completion)
        {
            Writer = writer;
            Completion = completion;
        }

        public ChannelWriter<int> Writer { get; }
        public TaskCompletionSource<InferenceRequestSnapshot> Completion { get; }
        public int PublishedTokenCount { get; set; }
    }
}
