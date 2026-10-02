using System.Buffers;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
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
    SchedulingPolicyOptions Scheduling,
    long? MaxKvBytes = null,
    long? MaxDeviceBytes = null);

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
    RuntimeKvCapacity KvCapacity)
{
    /// <summary>
    /// Internal device-scoped reservation versions captured when an otherwise-
    /// runnable empty batch was blocked by another engine's transient physical
    /// device-memory reservation. Null means the empty batch is not known to be
    /// temporary memory-reservation backpressure.
    /// </summary>
    internal IReadOnlyList<RuntimeDeviceMemoryReservationVersion>?
        DeviceMemoryBackpressureReservations { get; init; }

    /// <summary>
    /// Internal device-scoped reservation versions captured when an otherwise-
    /// runnable empty batch was blocked by another engine's inference item-credit
    /// reservation. Null means the empty batch is not known to be temporary
    /// inference-admission backpressure.
    /// </summary>
    internal IReadOnlyList<RuntimeDeviceInferenceReservationVersion>?
        DeviceInferenceBackpressureReservations { get; init; }
}

/// <summary>
/// Request-level orchestration loop. Each cycle snapshots live request/runtime
/// state, asks the F# scheduling kernel for one quantum, executes that quantum,
/// then commits generated tokens and releases terminal KV ownership.
/// </summary>
public sealed partial class InferenceEngine : IDisposable
{
    private static readonly ScheduledExecutionBindings EmptyExecutionBindings =
        new(new Dictionary<SequenceId, ScheduledPrefillBinding>());

    private readonly object _gate = new();
    private readonly SemaphoreSlim _cycleGate = new(1, 1);
    private readonly ExecutionPlanExecutor _runtime;
    private readonly ScheduledBatchExecutor _scheduledExecutor;
    private readonly ISchedulingKernel _scheduler;
    private readonly InferenceEngineOptions _options;
    private readonly IInferenceKvMemoryProfile? _kvMemoryProfile;
    private readonly Dictionary<SequenceId, RequestState> _requests = new();
    private int _activeRequestCount;
    private int _disposed;

    public InferenceEngine(
        ExecutionPlanExecutor runtime,
        ISchedulingKernel scheduler,
        InferenceEngineOptions options,
        IInferenceKvMemoryProfile? kvMemoryProfile = null)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(scheduler);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxBatchTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxBatchSequences);
        ArgumentOutOfRangeException.ThrowIfNegative(options.Scheduling.DecodeTokenReserve);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Scheduling.MaxPrefillChunkTokens);

        if (options.MaxKvBytes is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "MaxKvBytes must be positive when specified.");
        }

        if (options.MaxDeviceBytes is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "MaxDeviceBytes must be positive when specified.");
        }

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
        _kvMemoryProfile = kvMemoryProfile;
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
                return _activeRequestCount;
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

        AddRequest(state);
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

                if (!await _runtime.ReleaseSequenceAsync(sequenceId, cancellationToken)
                        .ConfigureAwait(false))
                {
                    throw new InvalidOperationException(
                        $"Runtime sequence {sequenceId} could not be released after cancellation.");
                }
            }

            lock (_gate)
            {
                MarkRequestCompleted(request, InferenceFinishReason.Cancelled);
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
            var active = RentActiveRequestSnapshot(out var activeCount);
            var scheduleId = Guid.NewGuid();
            var kvBefore = _runtime.KvCapacity;

            if (activeCount == 0)
            {
                return new InferenceCycleResult(
                    scheduleId,
                    new ScheduledBatch(scheduleId, Array.Empty<ScheduledWorkItem>(), 0, 0),
                    Array.Empty<SchedulingDeferral>(),
                    Array.Empty<SchedulingRejection>(),
                    Array.Empty<SequenceId>(),
                    kvBefore);
            }

            var candidates = new SchedulingCandidate[activeCount];
            try
            {
                for (var index = 0; index < activeCount; index++)
                {
                    candidates[index] = BuildCandidate(
                        in active[index],
                        kvBefore.TokensPerPage);
                }
            }
            finally
            {
                ReturnActiveRequestSnapshot(active, activeCount);
            }

            var maxBatchSequences = _options.MaxBatchSequences;
            var availableKvBytes = GetAvailableKvBytes(candidates);
            var admission = await ScheduleWithDeviceAdmissionAsync(
                    scheduleId,
                    now,
                    kvBefore,
                    maxBatchSequences,
                    availableKvBytes,
                    candidates,
                    cancellationToken)
                .ConfigureAwait(false);
            using var memoryReservationLease = admission.DeviceMemoryReservation;
            using var inferenceReservationLease = admission.DeviceInferenceReservation;

            var executionBindings = BuildExecutionBindings(admission.Decision.Batch);
            BackendStepResult[] backendResults;
            if (admission.DeviceMemoryReservation is null &&
                admission.DeviceInferenceReservation is null)
            {
                backendResults = await _scheduledExecutor.ExecuteBackendAsync(
                        admission.Decision.Batch,
                        executionBindings,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
            else
            {
                backendResults = await _scheduledExecutor.ExecuteBackendAsync(
                        admission.Decision.Batch,
                        executionBindings,
                        device => ReleaseDeviceAdmissionReservations(
                            device,
                            admission.DeviceMemoryReservation,
                            admission.DeviceInferenceReservation),
                        cancellationToken)
                    .ConfigureAwait(false);
            }

            if (backendResults.Length != admission.Decision.Batch.Items.Count)
            {
                throw new InvalidOperationException(
                    "Scheduled execution result count does not match selected work count.");
            }

            List<SequenceId>? completed = null;
            for (var index = 0; index < admission.Decision.Batch.Items.Count; index++)
            {
                var item = admission.Decision.Batch.Items[index];
                var backendResult = backendResults[index];
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
                    await CompleteAndReleaseAsync(
                            item.SequenceId,
                            finishReason.Value,
                            cancellationToken)
                        .ConfigureAwait(false);
                    (completed ??= new List<SequenceId>()).Add(item.SequenceId);
                }
            }

            return new InferenceCycleResult(
                scheduleId,
                admission.Decision.Batch,
                admission.Decision.Deferred,
                admission.Decision.Rejected,
                completed is null ? Array.Empty<SequenceId>() : completed,
                _runtime.KvCapacity)
            {
                DeviceMemoryBackpressureReservations =
                    admission.DeviceMemoryBackpressureReservations,
                DeviceInferenceBackpressureReservations =
                    admission.DeviceInferenceBackpressureReservations
            };
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
                var backpressureWait = GetDeviceMemoryBackpressureWait(
                    cycle,
                    cancellationToken);
                if (backpressureWait is not null)
                {
                    await backpressureWait.ConfigureAwait(false);
                    continue;
                }

                throw new InvalidOperationException(
                    "Inference engine made no scheduling progress while active requests remain.");
            }
        }

        throw new InvalidOperationException(
            $"Inference engine exceeded the maximum cycle count ({maxCycles}).");
    }

    /// <summary>
    /// Preserves the existing internal worker-facing method name while waiting on
    /// either physical-memory or inference-item reservation relief. Both waits are
    /// device-scoped and versioned; unrelated-device releases do not wake them.
    /// </summary>
    internal Task? GetDeviceMemoryBackpressureWait(
        InferenceCycleResult cycle,
        CancellationToken cancellationToken = default)
    {
        Task? memoryWait = cycle.DeviceMemoryBackpressureReservations is { Count: > 0 } memoryObserved
            ? _runtime.WaitForDeviceMemoryReservationReleaseAsync(memoryObserved, cancellationToken)
            : null;
        Task? inferenceWait = cycle.DeviceInferenceBackpressureReservations is { Count: > 0 } inferenceObserved
            ? _runtime.WaitForDeviceInferenceReservationReleaseAsync(inferenceObserved, cancellationToken)
            : null;

        return (memoryWait, inferenceWait) switch
        {
            (null, null) => null,
            ({ } wait, null) => wait,
            (null, { } wait) => wait,
            ({ } memoryTask, { } inferenceTask) =>
                WaitForAnyAdmissionReliefAsync(memoryTask, inferenceTask),
        };
    }

    private static async Task WaitForAnyAdmissionReliefAsync(
        Task memoryWait,
        Task inferenceWait)
    {
        var completed = await Task.WhenAny(memoryWait, inferenceWait)
            .ConfigureAwait(false);
        await completed.ConfigureAwait(false);
    }

    private static void ReleaseDeviceAdmissionReservations(
        DeviceId device,
        IRuntimeDeviceMemoryReservationLease? memoryReservation,
        IRuntimeDeviceInferenceReservationLease? inferenceReservation)
    {
        try
        {
            memoryReservation?.Release(device);
        }
        finally
        {
            inferenceReservation?.Release(device);
        }
    }

    private ScheduledExecutionBindings BuildExecutionBindings(ScheduledBatch batch)
    {
        var prefillCount = 0;
        foreach (var item in batch.Items)
        {
            if (item.Kind == ScheduledWorkKind.Prefill)
            {
                prefillCount++;
            }
        }

        if (prefillCount == 0)
        {
            return EmptyExecutionBindings;
        }

        Dictionary<SequenceId, ScheduledPrefillBinding>? prefillBindings =
            prefillCount > 1
                ? new Dictionary<SequenceId, ScheduledPrefillBinding>(prefillCount)
                : null;

        lock (_gate)
        {
            foreach (var item in batch.Items)
            {
                if (item.Kind != ScheduledWorkKind.Prefill)
                {
                    continue;
                }

                if (!_requests.TryGetValue(item.SequenceId, out var request) || request.IsCompleted)
                {
                    throw new InvalidOperationException(
                        $"Scheduler selected unknown or inactive request {item.SequenceId}.");
                }

                var binding = new ScheduledPrefillBinding(
                    request.ModelId,
                    request.PromptTokens);
                if (prefillBindings is null)
                {
                    return ScheduledExecutionBindings.CreateSinglePrefill(
                        item.SequenceId,
                        binding);
                }

                prefillBindings.Add(item.SequenceId, binding);
            }
        }

        return new ScheduledExecutionBindings(prefillBindings!);
    }

    private void AddRequest(RequestState request)
    {
        lock (_gate)
        {
            _requests.Add(request.SequenceId, request);
            _activeRequestCount = checked(_activeRequestCount + 1);
        }
    }

    private void MarkRequestCompleted(
        RequestState request,
        InferenceFinishReason finishReason)
    {
        if (request.IsCompleted)
        {
            return;
        }

        request.IsCompleted = true;
        request.FinishReason = finishReason;
        _activeRequestCount = checked(_activeRequestCount - 1);
    }

    private RequestView[] RentActiveRequestSnapshot(out int count)
    {
        lock (_gate)
        {
            count = _activeRequestCount;
            if (count == 0)
            {
                return Array.Empty<RequestView>();
            }

            var active = ArrayPool<RequestView>.Shared.Rent(count);
            var index = 0;
            try
            {
                foreach (var request in _requests.Values)
                {
                    if (request.IsCompleted)
                    {
                        continue;
                    }

                    if ((uint)index >= (uint)count)
                    {
                        throw new InvalidOperationException(
                            "Active request count exceeded the tracked active-request index.");
                    }

                    active[index++] = request.View();
                }

                if (index != count)
                {
                    throw new InvalidOperationException(
                        $"Tracked active request count {count} does not match snapshot count {index}.");
                }

                return active;
            }
            catch
            {
                Array.Clear(active, 0, index);
                ArrayPool<RequestView>.Shared.Return(active);
                throw;
            }
        }
    }

    private static void ReturnActiveRequestSnapshot(
        RequestView[] active,
        int count)
    {
        Array.Clear(active, 0, count);
        ArrayPool<RequestView>.Shared.Return(active);
    }

    private SchedulingCandidate BuildCandidate(in RequestView request, int tokensPerKvPage)
    {
        var kvBytesPerToken = GetKvBytesPerToken(request.ModelId);
        var executionDevice = _runtime.GetExecutionDevice(request.SequenceId);

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
                request.Priority,
                kvBytesPerToken,
                executionDevice);
        }

        return sequence.Status switch
        {
            SequenceStatus.Prefilling => new SchedulingCandidate(
                request.SequenceId,
                SchedulingPhase.Prefilling,
                request.Deadline,
                request.EnqueuedAt,
                RemainingPromptTokens(in request, sequence.Position),
                sequence.Position,
                tokensPerKvPage,
                request.Priority,
                kvBytesPerToken,
                executionDevice),

            SequenceStatus.Decoding => new SchedulingCandidate(
                request.SequenceId,
                SchedulingPhase.Decoding,
                request.Deadline,
                request.EnqueuedAt,
                1,
                sequence.Position,
                tokensPerKvPage,
                request.Priority,
                kvBytesPerToken,
                executionDevice),

            SequenceStatus.Suspended => throw new InvalidOperationException(
                $"Engine-owned request {request.SequenceId} is suspended without a resume phase."),

            SequenceStatus.Finished or SequenceStatus.Cancelled => throw new InvalidOperationException(
                $"Terminal runtime sequence {request.SequenceId} remained active in engine state."),

            _ => throw new InvalidOperationException(
                $"Unexpected runtime sequence state {sequence.Status} for request {request.SequenceId}.")
        };
    }

    private async ValueTask<DeviceAdmissionDecision> ScheduleWithDeviceAdmissionAsync(
        Guid scheduleId,
        DateTimeOffset now,
        RuntimeKvCapacity kvCapacity,
        int maxBatchSequences,
        long availableKvBytes,
        IReadOnlyList<SchedulingCandidate> candidates,
        CancellationToken cancellationToken)
    {
        var admissionDevices = BuildAdmissionDevices(candidates);

        // The per-device gate now protects both physical-memory and inference-item
        // scheduler admission. It is required even when MaxDeviceBytes is disabled.
        using var admissionGate = await _runtime.EnterDeviceMemoryAdmissionAsync(
                admissionDevices,
                cancellationToken)
            .ConfigureAwait(false);

        DeviceMemoryBudgetSnapshot? deviceMemory = null;
        if (_options.MaxDeviceBytes is not null)
        {
            deviceMemory = GetDeviceMemoryBudgets(admissionDevices);
        }

        var inferenceState = _runtime.GetDeviceInferenceReservationState(admissionDevices);
        var deviceSequences = GetDeviceSequenceBudgets(admissionDevices, inferenceState);
        var decision = ScheduleOnce(
            scheduleId,
            now,
            kvCapacity,
            maxBatchSequences,
            availableKvBytes,
            candidates,
            deviceMemory?.Budgets,
            deviceSequences);
        var candidateBySequence = BuildCandidateIndex(candidates);

        if (_options.MaxDeviceBytes is not null &&
            await TryReclaimBlockedDeviceMemoryAsync(
                    decision,
                    candidateBySequence,
                    cancellationToken)
                .ConfigureAwait(false))
        {
            deviceMemory = GetDeviceMemoryBudgets(admissionDevices);
            inferenceState = _runtime.GetDeviceInferenceReservationState(admissionDevices);
            deviceSequences = GetDeviceSequenceBudgets(admissionDevices, inferenceState);
            decision = ScheduleOnce(
                scheduleId,
                now,
                kvCapacity,
                maxBatchSequences,
                availableKvBytes,
                candidates,
                deviceMemory.Budgets,
                deviceSequences);
        }

        var memoryBackpressure = deviceMemory is null
            ? null
            : GetDeviceMemoryBackpressureReservations(
                decision,
                candidateBySequence,
                deviceMemory.ReservationState);
        var inferenceBackpressure = GetDeviceInferenceBackpressureReservations(
            decision,
            candidateBySequence,
            inferenceState);

        IRuntimeDeviceInferenceReservationLease? inferenceReservation = null;
        var inferenceRequests = BuildDeviceInferenceReservationRequests(
            decision,
            candidateBySequence);
        if (inferenceRequests.Count != 0)
        {
            if (!_runtime.TryReserveDeviceInference(
                    inferenceRequests,
                    out inferenceReservation))
            {
                throw new InvalidOperationException(
                    "Inference item reservation changed while the per-device admission gate was held. " +
                    "All Engine scheduler admission must use the shared runtime admission gate.");
            }
        }

        try
        {
            IRuntimeDeviceMemoryReservationLease? memoryReservation = null;
            if (_options.MaxDeviceBytes is not null)
            {
                var memoryRequests = BuildDeviceMemoryReservationRequests(
                    decision,
                    candidateBySequence);
                if (memoryRequests.Count != 0)
                {
                    memoryReservation = _runtime.ReserveDeviceMemoryByDevice(memoryRequests);
                }
            }

            return new DeviceAdmissionDecision(
                decision,
                memoryReservation,
                inferenceReservation,
                memoryBackpressure,
                inferenceBackpressure);
        }
        catch
        {
            inferenceReservation?.Dispose();
            throw;
        }
    }

    private static SchedulingCandidateIndex BuildCandidateIndex(
        IReadOnlyList<SchedulingCandidate> candidates) =>
        new(candidates);

    private SchedulingKernelResult ScheduleOnce(
        Guid scheduleId,
        DateTimeOffset now,
        RuntimeKvCapacity kvCapacity,
        int maxBatchSequences,
        long availableKvBytes,
        IReadOnlyList<SchedulingCandidate> candidates,
        IReadOnlyList<SchedulingDeviceMemoryBudget>? deviceMemory,
        IReadOnlyList<SchedulingDeviceSequenceBudget> deviceSequences) =>
        _scheduler.Schedule(
            scheduleId,
            now,
            new SchedulingBudget(
                _options.MaxBatchTokens,
                kvCapacity.AvailablePages,
                maxBatchSequences,
                availableKvBytes,
                deviceMemory,
                deviceSequences),
            _options.Scheduling,
            candidates);

    private IReadOnlyList<SchedulingDeviceSequenceBudget> GetDeviceSequenceBudgets(
        IReadOnlyList<DeviceId> devices,
        RuntimeDeviceInferenceReservationState reservationState)
    {
        var budgets = new SchedulingDeviceSequenceBudget[devices.Count];
        for (var index = 0; index < devices.Count; index++)
        {
            var device = devices[index];
            var capacity = _runtime.GetDeviceInferenceCapacity(device);
            var reservedItems = GetReservedInferenceItems(
                reservationState.Reservations,
                device);
            var available = reservedItems >= capacity
                ? 0
                : capacity - reservedItems;
            budgets[index] = new SchedulingDeviceSequenceBudget(device, available);
        }

        return budgets;
    }

    private DeviceMemoryBudgetSnapshot GetDeviceMemoryBudgets(
        IReadOnlyList<DeviceId> devices)
    {
        if (_options.MaxDeviceBytes is not { } maxDeviceBytes)
        {
            throw new InvalidOperationException(
                "Device-memory budgets require MaxDeviceBytes to be configured.");
        }

        var pressures = _runtime.GetDeviceMemoryPressureCore(devices);
        var reservationState = _runtime.GetDeviceMemoryReservationState(devices);
        var budgets = new SchedulingDeviceMemoryBudget[devices.Count];

        for (var index = 0; index < devices.Count; index++)
        {
            var device = devices[index];
            if (!TryGetDeviceMemoryPressure(pressures, device, out var pressure))
            {
                throw new InvalidOperationException(
                    $"MaxDeviceBytes requires physical memory pressure from execution device {device}.");
            }

            var physicalAvailableBytes = pressure.ReservedBytes >= maxDeviceBytes
                ? 0L
                : maxDeviceBytes - pressure.ReservedBytes;
            var outstandingReservationBytes = GetReservedDeviceMemoryBytes(
                reservationState.Reservations,
                device);
            var availableBytes = outstandingReservationBytes >= physicalAvailableBytes
                ? 0L
                : physicalAvailableBytes - outstandingReservationBytes;
            budgets[index] = new SchedulingDeviceMemoryBudget(device, availableBytes);
        }

        return new DeviceMemoryBudgetSnapshot(budgets, reservationState);
    }

    // Admission snapshots are device-scoped and intentionally small. Linear scans
    // avoid rebuilding transient hash tables on every scheduling cycle.
    private static int GetReservedInferenceItems(
        IReadOnlyList<RuntimeDeviceInferenceReservationSnapshot> reservations,
        DeviceId device) =>
        TryGetDeviceInferenceReservation(reservations, device, out var reservation)
            ? reservation.Items
            : 0;

    private static long GetReservedDeviceMemoryBytes(
        IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot> reservations,
        DeviceId device) =>
        TryGetDeviceMemoryReservation(reservations, device, out var reservation)
            ? reservation.Bytes
            : 0L;

    private static bool TryGetDeviceInferenceReservation(
        IReadOnlyList<RuntimeDeviceInferenceReservationSnapshot> reservations,
        DeviceId device,
        out RuntimeDeviceInferenceReservationSnapshot reservation)
    {
        for (var index = 0; index < reservations.Count; index++)
        {
            var candidate = reservations[index];
            if (candidate.Device.Equals(device))
            {
                reservation = candidate;
                return true;
            }
        }

        reservation = default;
        return false;
    }

    private static bool TryGetDeviceMemoryReservation(
        IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot> reservations,
        DeviceId device,
        out RuntimeDeviceMemoryReservationSnapshot reservation)
    {
        for (var index = 0; index < reservations.Count; index++)
        {
            var candidate = reservations[index];
            if (candidate.Device.Equals(device))
            {
                reservation = candidate;
                return true;
            }
        }

        reservation = default;
        return false;
    }

    private static bool TryGetDeviceMemoryPressure(
        IReadOnlyList<RuntimeDeviceMemoryPressure> pressures,
        DeviceId device,
        out RuntimeDeviceMemoryPressure pressure)
    {
        for (var index = 0; index < pressures.Count; index++)
        {
            var candidate = pressures[index];
            if (candidate.Device.Equals(device))
            {
                pressure = candidate;
                return true;
            }
        }

        pressure = default;
        return false;
    }

    private static IReadOnlyList<RuntimeDeviceMemoryReservationVersion>?
        GetDeviceMemoryBackpressureReservations(
            SchedulingKernelResult decision,
            SchedulingCandidateIndex candidateBySequence,
            RuntimeDeviceMemoryReservationState reservationState)
    {
        if (decision.Batch.Items.Count != 0 ||
            reservationState.Reservations.Count == 0)
        {
            return null;
        }

        var blocked = new Dictionary<DeviceId, long>();

        foreach (var deferred in decision.Deferred)
        {
            if (deferred.Reason != SchedulingDeferralReason.DeviceMemoryBudget ||
                !candidateBySequence.TryGetValue(deferred.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device ||
                !TryGetDeviceMemoryReservation(
                    reservationState.Reservations,
                    device,
                    out var reservation) ||
                reservation.Bytes <= 0)
            {
                continue;
            }

            blocked[device] = reservation.ReleaseVersion;
        }

        return blocked.Count == 0
            ? null
            : blocked
                .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
                .Select(static pair => new RuntimeDeviceMemoryReservationVersion(
                    pair.Key,
                    pair.Value))
                .ToArray();
    }

    private static IReadOnlyList<RuntimeDeviceInferenceReservationVersion>?
        GetDeviceInferenceBackpressureReservations(
            SchedulingKernelResult decision,
            SchedulingCandidateIndex candidateBySequence,
            RuntimeDeviceInferenceReservationState reservationState)
    {
        if (decision.Batch.Items.Count != 0 ||
            reservationState.Reservations.Count == 0)
        {
            return null;
        }

        var blocked = new Dictionary<DeviceId, long>();

        foreach (var deferred in decision.Deferred)
        {
            if (deferred.Reason != SchedulingDeferralReason.DeviceSequenceBudget ||
                !candidateBySequence.TryGetValue(deferred.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device ||
                !TryGetDeviceInferenceReservation(
                    reservationState.Reservations,
                    device,
                    out var reservation) ||
                reservation.Items <= 0)
            {
                continue;
            }

            blocked[device] = reservation.ReleaseVersion;
        }

        return blocked.Count == 0
            ? null
            : blocked
                .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
                .Select(static pair => new RuntimeDeviceInferenceReservationVersion(
                    pair.Key,
                    pair.Value))
                .ToArray();
    }

    private IReadOnlyList<RuntimeDeviceMemoryReservationRequest>
        BuildDeviceMemoryReservationRequests(
            SchedulingKernelResult decision,
            SchedulingCandidateIndex candidateBySequence)
    {
        Dictionary<DeviceId, long>? bytesByDevice = null;
        DeviceId singleDevice = default;
        long singleBytes = 0;
        var hasSingleDevice = false;

        foreach (var item in decision.Batch.Items)
        {
            if (item.TransientKvByteGrant == 0 ||
                !candidateBySequence.TryGetValue(item.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            if (bytesByDevice is not null)
            {
                bytesByDevice.TryGetValue(device, out var existing);
                bytesByDevice[device] = checked(
                    existing + item.TransientKvByteGrant);
                continue;
            }

            if (!hasSingleDevice)
            {
                singleDevice = device;
                singleBytes = item.TransientKvByteGrant;
                hasSingleDevice = true;
                continue;
            }

            if (singleDevice.Equals(device))
            {
                singleBytes = checked(singleBytes + item.TransientKvByteGrant);
                continue;
            }

            bytesByDevice = new Dictionary<DeviceId, long>(2)
            {
                [singleDevice] = singleBytes,
                [device] = item.TransientKvByteGrant
            };
        }

        if (bytesByDevice is null)
        {
            return hasSingleDevice
                ? new[]
                {
                    new RuntimeDeviceMemoryReservationRequest(
                        singleDevice,
                        singleBytes)
                }
                : Array.Empty<RuntimeDeviceMemoryReservationRequest>();
        }

        return bytesByDevice
            .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
            .Select(static pair => new RuntimeDeviceMemoryReservationRequest(
                pair.Key,
                pair.Value))
            .ToArray();
    }

    private IReadOnlyList<RuntimeDeviceInferenceReservationRequest>
        BuildDeviceInferenceReservationRequests(
            SchedulingKernelResult decision,
            SchedulingCandidateIndex candidateBySequence)
    {
        Dictionary<DeviceId, int>? itemsByDevice = null;
        DeviceId singleDevice = default;
        var singleItems = 0;
        var hasSingleDevice = false;

        foreach (var item in decision.Batch.Items)
        {
            if (!candidateBySequence.TryGetValue(item.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            if (itemsByDevice is not null)
            {
                itemsByDevice.TryGetValue(device, out var existing);
                itemsByDevice[device] = checked(existing + 1);
                continue;
            }

            if (!hasSingleDevice)
            {
                singleDevice = device;
                singleItems = 1;
                hasSingleDevice = true;
                continue;
            }

            if (singleDevice.Equals(device))
            {
                singleItems = checked(singleItems + 1);
                continue;
            }

            itemsByDevice = new Dictionary<DeviceId, int>(2)
            {
                [singleDevice] = singleItems,
                [device] = 1
            };
        }

        if (itemsByDevice is null)
        {
            return hasSingleDevice
                ? new[]
                {
                    new RuntimeDeviceInferenceReservationRequest(
                        singleDevice,
                        singleItems)
                }
                : Array.Empty<RuntimeDeviceInferenceReservationRequest>();
        }

        return itemsByDevice
            .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
            .Select(static pair => new RuntimeDeviceInferenceReservationRequest(
                pair.Key,
                pair.Value))
            .ToArray();
    }

    private async ValueTask<bool> TryReclaimBlockedDeviceMemoryAsync(
        SchedulingKernelResult decision,
        SchedulingCandidateIndex candidateBySequence,
        CancellationToken cancellationToken)
    {
        if (_options.MaxDeviceBytes is not { } maxDeviceBytes)
        {
            return false;
        }

        Dictionary<DeviceId, SchedulingCandidate>? firstBlockedByDevice = null;
        foreach (var deferred in decision.Deferred)
        {
            if (deferred.Reason != SchedulingDeferralReason.DeviceMemoryBudget ||
                !candidateBySequence.TryGetValue(deferred.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            (firstBlockedByDevice ??=
                new Dictionary<DeviceId, SchedulingCandidate>())
                .TryAdd(device, candidate);
        }

        if (firstBlockedByDevice is null)
        {
            return false;
        }

        Dictionary<DeviceId, long>? selectedTransientByDevice = null;
        foreach (var item in decision.Batch.Items)
        {
            if (!candidateBySequence.TryGetValue(item.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            selectedTransientByDevice ??= new Dictionary<DeviceId, long>();
            selectedTransientByDevice.TryGetValue(device, out var usedBytes);
            selectedTransientByDevice[device] = checked(
                usedBytes + item.TransientKvByteGrant);
        }

        var blockedDevices = firstBlockedByDevice.Keys
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .ToArray();
        var pressures = _runtime.GetDeviceMemoryPressureCore(blockedDevices);
        var reservations = _runtime.GetDeviceMemoryReservations(blockedDevices);
        var shouldReschedule = false;

        foreach (var (device, blockedCandidate) in firstBlockedByDevice)
        {
            if (!TryGetDeviceMemoryPressure(pressures, device, out var pressure))
            {
                continue;
            }

            var selectedTransientBytes = 0L;
            if (selectedTransientByDevice is not null)
            {
                selectedTransientByDevice.TryGetValue(
                    device,
                    out selectedTransientBytes);
            }

            var outstandingReservationBytes = GetReservedDeviceMemoryBytes(
                reservations,
                device);
            var minimumBlockedTransientBytes = checked(
                ((long)blockedCandidate.Position + 1L) *
                blockedCandidate.KvBytesPerToken);
            var requiredTransientBytes = checked(
                selectedTransientBytes + minimumBlockedTransientBytes);
            var chargedTransientBytes = checked(
                outstandingReservationBytes + requiredTransientBytes);

            var releaseNeeded = pressure.ReservedBytes >= maxDeviceBytes
                ? checked(
                    pressure.ReservedBytes - maxDeviceBytes +
                    chargedTransientBytes)
                : Math.Max(
                    0L,
                    chargedTransientBytes -
                    (maxDeviceBytes - pressure.ReservedBytes));

            if (releaseNeeded <= 0)
            {
                // Pressure or outstanding reservations may have improved since the
                // scheduler snapshot. Re-run admission once under the shared gate.
                shouldReschedule = true;
                continue;
            }

            if (pressure.ReclaimableBytes == 0)
            {
                continue;
            }

            var targetReclaimableBytes = releaseNeeded >= pressure.ReclaimableBytes
                ? 0L
                : pressure.ReclaimableBytes - releaseNeeded;

            try
            {
                await _runtime.ReclaimDeviceMemoryAsync(
                        device,
                        targetReclaimableBytes,
                        cancellationToken)
                    .ConfigureAwait(false);
                shouldReschedule = true;
            }
            catch (NotSupportedException)
            {
                // Reclaim is optional. Keep unreleased residency charged to
                // physical headroom and preserve conservative scheduler deferral.
            }
        }

        return shouldReschedule;
    }

    private long GetAvailableKvBytes(SchedulingCandidate[] candidates)
    {
        if (_options.MaxKvBytes is not { } maxKvBytes)
        {
            return long.MaxValue;
        }

        long retainedBytes = 0;
        foreach (var candidate in candidates)
        {
            retainedBytes = checked(
                retainedBytes + checked(
                    (long)candidate.Position * candidate.KvBytesPerToken));
        }

        return retainedBytes >= maxKvBytes
            ? 0L
            : maxKvBytes - retainedBytes;
    }

    private long GetKvBytesPerToken(ModelId modelId)
    {
        var bytes = _kvMemoryProfile?.GetKvBytesPerToken(modelId) ?? 0L;
        if (bytes < 0L)
        {
            throw new InvalidOperationException(
                $"KV memory profile returned a negative byte cost ({bytes}) for model {modelId}.");
        }

        return bytes;
    }

    private static int RemainingPromptTokens(in RequestView request, int position)
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

    private async ValueTask CompleteAndReleaseAsync(
        SequenceId sequenceId,
        InferenceFinishReason finishReason,
        CancellationToken cancellationToken)
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

        if (!await _runtime.ReleaseSequenceAsync(sequenceId, cancellationToken)
                .ConfigureAwait(false))
        {
            throw new InvalidOperationException(
                $"Runtime sequence {sequenceId} could not be released after completion.");
        }

        lock (_gate)
        {
            if (_requests.TryGetValue(sequenceId, out var request))
            {
                MarkRequestCompleted(request, finishReason);
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

    private readonly struct SchedulingCandidateIndex
    {
        private readonly IReadOnlyList<SchedulingCandidate> _candidates;
        private readonly Dictionary<SequenceId, SchedulingCandidate>? _multiple;

        public SchedulingCandidateIndex(
            IReadOnlyList<SchedulingCandidate> candidates)
        {
            ArgumentNullException.ThrowIfNull(candidates);
            _candidates = candidates;

            if (candidates.Count <= 1)
            {
                _multiple = null;
                return;
            }

            var multiple =
                new Dictionary<SequenceId, SchedulingCandidate>(candidates.Count);
            for (var index = 0; index < candidates.Count; index++)
            {
                var candidate = candidates[index];
                multiple.Add(candidate.SequenceId, candidate);
            }

            _multiple = multiple;
        }

        public bool TryGetValue(
            SequenceId sequenceId,
            out SchedulingCandidate candidate)
        {
            if (_multiple is not null)
            {
                return _multiple.TryGetValue(sequenceId, out candidate);
            }

            if (_candidates.Count == 1)
            {
                var single = _candidates[0];
                if (single.SequenceId.Equals(sequenceId))
                {
                    candidate = single;
                    return true;
                }
            }

            candidate = default;
            return false;
        }
    }

    private sealed record DeviceMemoryBudgetSnapshot(
        IReadOnlyList<SchedulingDeviceMemoryBudget> Budgets,
        RuntimeDeviceMemoryReservationState ReservationState);

    private sealed record DeviceAdmissionDecision(
        SchedulingKernelResult Decision,
        IRuntimeDeviceMemoryReservationLease? DeviceMemoryReservation,
        IRuntimeDeviceInferenceReservationLease? DeviceInferenceReservation,
        IReadOnlyList<RuntimeDeviceMemoryReservationVersion>?
            DeviceMemoryBackpressureReservations,
        IReadOnlyList<RuntimeDeviceInferenceReservationVersion>?
            DeviceInferenceBackpressureReservations);

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

    private readonly record struct RequestView(
        SequenceId SequenceId,
        ModelId ModelId,
        int[] PromptTokens,
        int MaxNewTokens,
        int Priority,
        DateTimeOffset? Deadline,
        DateTimeOffset EnqueuedAt,
        int GeneratedTokenCount);
}
