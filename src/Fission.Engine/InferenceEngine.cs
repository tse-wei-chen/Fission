using Fission.Abstractions;
using Fission.Abstractions.Scheduling;
using Fission.Runtime.Execution;
using Fission.Runtime.Sequences;

namespace Fission.Engine;

public enum InferenceFinishReason
{
    Stop,
    Length,
    Cancelled
}

public sealed record InferenceEngineOptions(
    int MaxBatchTokens,
    int MaxBatchSequences,
    SchedulingPolicyOptions Scheduling);

public sealed record InferenceRequestSnapshot(
    SequenceId SequenceId,
    ModelId ModelId,
    int PromptTokenCount,
    IReadOnlyList<int> GeneratedTokens,
    bool IsCompleted,
    InferenceFinishReason? FinishReason,
    DateTimeOffset EnqueuedAt,
    DateTimeOffset? Deadline,
    int Priority);

public sealed record InferenceCycleResult(
    Guid ScheduleId,
    ScheduledBatch Batch,
    IReadOnlyList<SchedulingDeferral> Deferred,
    IReadOnlyList<SchedulingRejection> Rejected,
    IReadOnlyList<SequenceId> CompletedSequences,
    RuntimeKvCapacity KvCapacity);

/// <summary>
/// Request-level orchestration loop. Each cycle snapshots live request/runtime
/// state, asks the F# scheduling kernel for one quantum, executes that quantum,
/// then commits generated tokens and releases terminal KV ownership.
/// </summary>
public sealed class InferenceEngine : IDisposable
{
    private readonly object _gate = new();
    private readonly SemaphoreSlim _cycleGate = new(1, 1);
    private readonly ExecutionPlanExecutor _runtime;
    private readonly ScheduledBatchExecutor _scheduledExecutor;
    private readonly ISchedulingKernel _scheduler;
    private readonly InferenceEngineOptions _options;
    private readonly Dictionary<SequenceId, RequestState> _requests = new();
    private int _disposed;

    public InferenceEngine(
        ExecutionPlanExecutor runtime,
        ISchedulingKernel scheduler,
        InferenceEngineOptions options)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(scheduler);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxBatchTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxBatchSequences);
        ArgumentOutOfRangeException.ThrowIfNegative(options.Scheduling.DecodeTokenReserve);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Scheduling.MaxPrefillChunkTokens);

        if (options.Scheduling.DeadlineUrgencyWindow < TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "DeadlineUrgencyWindow cannot be negative.");
        }

        _runtime = runtime;
        _scheduledExecutor = new ScheduledBatchExecutor(runtime);
        _scheduler = scheduler;
        _options = options;
    }

    public int RequestCount
    {
        get
        {
            lock (_gate)
            {
                return _requests.Count;
            }
        }
    }

    public int ActiveRequestCount
    {
        get
        {
            lock (_gate)
            {
                return _requests.Values.Count(static request => !request.IsCompleted);
            }
        }
    }

    public SequenceId Submit(
        ModelId modelId,
        ReadOnlyMemory<int> promptTokens,
        int maxNewTokens,
        int priority = 0,
        DateTimeOffset? deadline = null,
        DateTimeOffset? enqueuedAt = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxNewTokens);

        if (promptTokens.IsEmpty)
        {
            throw new ArgumentException("Prompt must contain at least one token.", nameof(promptTokens));
        }

        var sequenceId = SequenceId.New();
        var state = new RequestState(
            sequenceId,
            modelId,
            promptTokens.ToArray(),
            maxNewTokens,
            priority,
            deadline,
            enqueuedAt ?? DateTimeOffset.UtcNow);

        lock (_gate)
        {
            _requests.Add(sequenceId, state);
        }

        return sequenceId;
    }

    public InferenceRequestSnapshot GetSnapshot(SequenceId sequenceId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        lock (_gate)
        {
            if (!_requests.TryGetValue(sequenceId, out var request))
            {
                throw new KeyNotFoundException($"Request {sequenceId} does not exist.");
            }

            return request.Snapshot();
        }
    }

    /// <summary>
    /// Cancels a request at the same serialization boundary used by scheduler
    /// cycles. Callers never release live KV state concurrently with backend work.
    /// </summary>
    public async ValueTask<InferenceRequestSnapshot> CancelAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _cycleGate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            RequestState request;
            lock (_gate)
            {
                if (!_requests.TryGetValue(sequenceId, out request!))
                {
                    throw new KeyNotFoundException($"Request {sequenceId} does not exist.");
                }

                if (request.IsCompleted)
                {
                    return request.Snapshot();
                }
            }

            if (_runtime.TryGetSequence(sequenceId, out var sequence) && sequence is not null)
            {
                if (sequence.Status != SequenceStatus.Finished &&
                    sequence.Status != SequenceStatus.Cancelled)
                {
                    sequence.TransitionTo(SequenceStatus.Cancelled);
                }

                if (!_runtime.ReleaseSequence(sequenceId))
                {
                    throw new InvalidOperationException(
                        $"Runtime sequence {sequenceId} could not be released after cancellation.");
                }
            }

            lock (_gate)
            {
                request.IsCompleted = true;
                request.FinishReason = InferenceFinishReason.Cancelled;
                return request.Snapshot();
            }
        }
        finally
        {
            _cycleGate.Release();
        }
    }

    public async ValueTask<InferenceCycleResult> RunCycleAsync(
        DateTimeOffset now,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _cycleGate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            var active = SnapshotActiveRequests();
            var scheduleId = Guid.NewGuid();
            var kvBefore = _runtime.KvCapacity;

            if (active.Length == 0)
            {
                return new InferenceCycleResult(
                    scheduleId,
                    new ScheduledBatch(scheduleId, Array.Empty<ScheduledWorkItem>(), 0, 0),
                    Array.Empty<SchedulingDeferral>(),
                    Array.Empty<SchedulingRejection>(),
                    Array.Empty<SequenceId>(),
                    kvBefore);
            }

            var candidates = new SchedulingCandidate[active.Length];
            for (var index = 0; index < active.Length; index++)
            {
                candidates[index] = BuildCandidate(active[index], kvBefore.TokensPerPage);
            }

            var decision = _scheduler.Schedule(
                scheduleId,
                now,
                new SchedulingBudget(
                    _options.MaxBatchTokens,
                    kvBefore.AvailablePages,
                    _options.MaxBatchSequences),
                _options.Scheduling,
                candidates);

            var prefillBindings = new Dictionary<SequenceId, ScheduledPrefillBinding>();
            foreach (var item in decision.Batch.Items)
            {
                if (item.Kind != ScheduledWorkKind.Prefill)
                {
                    continue;
                }

                var request = active.FirstOrDefault(candidate => candidate.SequenceId == item.SequenceId);
                if (request is null)
                {
                    throw new InvalidOperationException(
                        $"Scheduler selected unknown request {item.SequenceId}.");
                }

                prefillBindings.Add(
                    item.SequenceId,
                    new ScheduledPrefillBinding(request.ModelId, request.PromptTokens));
            }

            var batchResult = await _scheduledExecutor.ExecuteAsync(
                decision.Batch,
                new ScheduledExecutionBindings(prefillBindings),
                cancellationToken).ConfigureAwait(false);

            if (batchResult.ItemResults.Count != decision.Batch.Items.Count)
            {
                throw new InvalidOperationException(
                    "Scheduled execution result count does not match selected work count.");
            }

            var completed = new List<SequenceId>();
            for (var index = 0; index < decision.Batch.Items.Count; index++)
            {
                var item = decision.Batch.Items[index];
                var itemResult = batchResult.ItemResults[index];

                if (itemResult.BackendResults.Count != 1)
                {
                    throw new InvalidOperationException(
                        $"Scheduled work {item.SequenceId} returned {itemResult.BackendResults.Count} backend results; expected one.");
                }

                var backendResult = itemResult.BackendResults[0];
                if (backendResult.SequenceId != item.SequenceId)
                {
                    throw new InvalidOperationException(
                        $"Backend result sequence {backendResult.SequenceId} does not match scheduled sequence {item.SequenceId}.");
                }

                if (item.Kind == ScheduledWorkKind.Decode)
                {
                    CommitDecodeResult(item.SequenceId, backendResult.TokenId);
                }

                var finishReason = GetFinishReason(item.SequenceId, backendResult.IsFinished);
                if (finishReason is not null)
                {
                    CompleteAndRelease(item.SequenceId, finishReason.Value);
                    completed.Add(item.SequenceId);
                }
            }

            return new InferenceCycleResult(
                scheduleId,
                decision.Batch,
                decision.Deferred,
                decision.Rejected,
                completed,
                _runtime.KvCapacity);
        }
        finally
        {
            _cycleGate.Release();
        }
    }

    public async ValueTask<IReadOnlyList<InferenceCycleResult>> RunUntilCompleteAsync(
        int maxCycles = 10_000,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxCycles);
        var cycles = new List<InferenceCycleResult>();

        for (var index = 0; index < maxCycles; index++)
        {
            if (ActiveRequestCount == 0)
            {
                return cycles;
            }

            var cycle = await RunCycleAsync(DateTimeOffset.UtcNow, cancellationToken)
                .ConfigureAwait(false);
            cycles.Add(cycle);

            if (cycle.Batch.Items.Count == 0 && ActiveRequestCount != 0)
            {
                throw new InvalidOperationException(
                    "Inference engine made no scheduling progress while active requests remain.");
            }
        }

        throw new InvalidOperationException(
            $"Inference engine exceeded the maximum cycle count ({maxCycles}).");
    }

    private RequestView[] SnapshotActiveRequests()
    {
        lock (_gate)
        {
            return _requests.Values
                .Where(static request => !request.IsCompleted)
                .Select(static request => request.View())
                .ToArray();
        }
    }

    private SchedulingCandidate BuildCandidate(RequestView request, int tokensPerKvPage)
    {
        if (!_runtime.TryGetSequence(request.SequenceId, out var sequence) || sequence is null)
        {
            return new SchedulingCandidate(
                request.SequenceId,
                SchedulingPhase.Prefilling,
                request.Deadline,
                request.EnqueuedAt,
                request.PromptTokens.Length,
                0,
                tokensPerKvPage,
                request.Priority);
        }

        return sequence.Status switch
        {
            SequenceStatus.Prefilling => new SchedulingCandidate(
                request.SequenceId,
                SchedulingPhase.Prefilling,
                request.Deadline,
                request.EnqueuedAt,
                RemainingPromptTokens(request, sequence.Position),
                sequence.Position,
                tokensPerKvPage,
                request.Priority),

            SequenceStatus.Decoding => new SchedulingCandidate(
                request.SequenceId,
                SchedulingPhase.Decoding,
                request.Deadline,
                request.EnqueuedAt,
                1,
                sequence.Position,
                tokensPerKvPage,
                request.Priority),

            SequenceStatus.Suspended => throw new InvalidOperationException(
                $"Engine-owned request {request.SequenceId} is suspended without a resume phase."),

            SequenceStatus.Finished or SequenceStatus.Cancelled => throw new InvalidOperationException(
                $"Terminal runtime sequence {request.SequenceId} remained active in engine state."),

            _ => throw new InvalidOperationException(
                $"Unexpected runtime sequence state {sequence.Status} for request {request.SequenceId}.")
        };
    }

    private static int RemainingPromptTokens(RequestView request, int position)
    {
        var remaining = request.PromptTokens.Length - position;
        if (remaining <= 0)
        {
            throw new InvalidOperationException(
                $"Prefill sequence {request.SequenceId} has position {position} beyond prompt length {request.PromptTokens.Length}.");
        }

        return remaining;
    }

    private void CommitDecodeResult(SequenceId sequenceId, int tokenId)
    {
        lock (_gate)
        {
            if (!_requests.TryGetValue(sequenceId, out var request) || request.IsCompleted)
            {
                throw new InvalidOperationException(
                    $"Cannot commit decode result for inactive request {sequenceId}.");
            }

            request.GeneratedTokens.Add(tokenId);
        }
    }

    private InferenceFinishReason? GetFinishReason(SequenceId sequenceId, bool backendFinished)
    {
        lock (_gate)
        {
            if (!_requests.TryGetValue(sequenceId, out var request) || request.IsCompleted)
            {
                return null;
            }

            if (backendFinished)
            {
                return InferenceFinishReason.Stop;
            }

            return request.GeneratedTokens.Count >= request.MaxNewTokens
                ? InferenceFinishReason.Length
                : null;
        }
    }

    private void CompleteAndRelease(
        SequenceId sequenceId,
        InferenceFinishReason finishReason)
    {
        if (!_runtime.TryGetSequence(sequenceId, out var sequence) || sequence is null)
        {
            throw new InvalidOperationException(
                $"Cannot complete request {sequenceId}; runtime sequence is missing.");
        }

        if (sequence.Status == SequenceStatus.Decoding)
        {
            sequence.TransitionTo(SequenceStatus.Finished);
        }
        else if (sequence.Status != SequenceStatus.Finished)
        {
            throw new InvalidOperationException(
                $"Cannot complete request {sequenceId} while runtime sequence is {sequence.Status}.");
        }

        if (!_runtime.ReleaseSequence(sequenceId))
        {
            throw new InvalidOperationException(
                $"Runtime sequence {sequenceId} could not be released after completion.");
        }

        lock (_gate)
        {
            if (_requests.TryGetValue(sequenceId, out var request))
            {
                request.IsCompleted = true;
                request.FinishReason = finishReason;
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _cycleGate.Dispose();
    }

    private sealed class RequestState
    {
        public RequestState(
            SequenceId sequenceId,
            ModelId modelId,
            int[] promptTokens,
            int maxNewTokens,
            int priority,
            DateTimeOffset? deadline,
            DateTimeOffset enqueuedAt)
        {
            SequenceId = sequenceId;
            ModelId = modelId;
            PromptTokens = promptTokens;
            MaxNewTokens = maxNewTokens;
            Priority = priority;
            Deadline = deadline;
            EnqueuedAt = enqueuedAt;
        }

        public SequenceId SequenceId { get; }
        public ModelId ModelId { get; }
        public int[] PromptTokens { get; }
        public int MaxNewTokens { get; }
        public int Priority { get; }
        public DateTimeOffset? Deadline { get; }
        public DateTimeOffset EnqueuedAt { get; }
        public List<int> GeneratedTokens { get; } = new();
        public bool IsCompleted { get; set; }
        public InferenceFinishReason? FinishReason { get; set; }

        public RequestView View() => new(
            SequenceId,
            ModelId,
            PromptTokens,
            MaxNewTokens,
            Priority,
            Deadline,
            EnqueuedAt,
            GeneratedTokens.Count);

        public InferenceRequestSnapshot Snapshot() => new(
            SequenceId,
            ModelId,
            PromptTokens.Length,
            GeneratedTokens.ToArray(),
            IsCompleted,
            FinishReason,
            EnqueuedAt,
            Deadline,
            Priority);
    }

    private sealed record RequestView(
        SequenceId SequenceId,
        ModelId ModelId,
        int[] PromptTokens,
        int MaxNewTokens,
        int Priority,
        DateTimeOffset? Deadline,
        DateTimeOffset EnqueuedAt,
        int GeneratedTokenCount);
}
