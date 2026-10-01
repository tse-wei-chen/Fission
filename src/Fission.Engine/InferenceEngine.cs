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
public sealed class InferenceEngine : IDisposable
{
    private readonly object _gate = new();
    private readonly SemaphoreSlim _cycleGate = new(1, 1);
    private readonly ExecutionPlanExecutor _runtime;
    private readonly ScheduledBatchExecutor _scheduledExecutor;
    private readonly ISchedulingKernel _scheduler;
    private readonly InferenceEngineOptions _options;
    private readonly IInferenceKvMemoryProfile? _kvMemoryProfile;
    private readonly Dictionary<SequenceId, RequestState> _requests = new();
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

                if (!await _runtime.ReleaseSequenceAsync(sequenceId, cancellationToken)
                        .ConfigureAwait(false))
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

            var maxBatchSequences = _options.MaxBatchSequences;
            var availableKvBytes = GetAvailableKvBytes(active);
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

            var prefillBindings = new Dictionary<SequenceId, ScheduledPrefillBinding>();
            foreach (var item in admission.Decision.Batch.Items)
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

            var executionBindings = new ScheduledExecutionBindings(prefillBindings);
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

            var completed = new List<SequenceId>();
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
                    completed.Add(item.SequenceId);
                }
            }

            return new InferenceCycleResult(
                scheduleId,
                admission.Decision.Batch,
                admission.Decision.Deferred,
                admission.Decision.Rejected,
                completed,
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
                RemainingPromptTokens(request, sequence.Position),
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
        var admissionDevices = candidates
            .Select(static candidate => candidate.ExecutionDevice)
            .Where(static device => device.HasValue)
            .Select(static device => device!.Value)
            .Distinct()
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .ToArray();

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

        if (_options.MaxDeviceBytes is not null &&
            await TryReclaimBlockedDeviceMemoryAsync(
                    decision,
                    candidates,
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
                candidates,
                deviceMemory.ReservationState);
        var inferenceBackpressure = GetDeviceInferenceBackpressureReservations(
            decision,
            candidates,
            inferenceState);

        IRuntimeDeviceInferenceReservationLease? inferenceReservation = null;
        var inferenceRequests = BuildDeviceInferenceReservationRequests(
            decision,
            candidates);
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
                    candidates);
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
        var reservedByDevice = reservationState.Reservations
            .ToDictionary(
                static reservation => reservation.Device,
                static reservation => reservation.Items);
        var budgets = new SchedulingDeviceSequenceBudget[devices.Count];
        for (var index = 0; index < devices.Count; index++)
        {
            var device = devices[index];
            var capacity = _runtime.GetDeviceInferenceCapacity(device);
            reservedByDevice.TryGetValue(device, out var reservedItems);
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

        var pressureByDevice = _runtime.GetDeviceMemoryPressureCore(devices)
            .ToDictionary(static pressure => pressure.Device);
        var reservationState = _runtime.GetDeviceMemoryReservationState(devices);
        var reservedByDevice = reservationState.Reservations
            .ToDictionary(static reservation => reservation.Device,
                static reservation => reservation.Bytes);
        var budgets = new SchedulingDeviceMemoryBudget[devices.Count];

        for (var index = 0; index < devices.Count; index++)
        {
            var device = devices[index];
            if (!pressureByDevice.TryGetValue(device, out var pressure))
            {
                throw new InvalidOperationException(
                    $"MaxDeviceBytes requires physical memory pressure from execution device {device}.");
            }

            var physicalAvailableBytes = pressure.ReservedBytes >= maxDeviceBytes
                ? 0L
                : maxDeviceBytes - pressure.ReservedBytes;
            reservedByDevice.TryGetValue(device, out var outstandingReservationBytes);
            var availableBytes = outstandingReservationBytes >= physicalAvailableBytes
                ? 0L
                : physicalAvailableBytes - outstandingReservationBytes;
            budgets[index] = new SchedulingDeviceMemoryBudget(device, availableBytes);
        }

        return new DeviceMemoryBudgetSnapshot(budgets, reservationState);
    }

    private static IReadOnlyList<RuntimeDeviceMemoryReservationVersion>?
        GetDeviceMemoryBackpressureReservations(
            SchedulingKernelResult decision,
            IReadOnlyList<SchedulingCandidate> candidates,
            RuntimeDeviceMemoryReservationState reservationState)
    {
        if (decision.Batch.Items.Count != 0 ||
            reservationState.Reservations.Count == 0)
        {
            return null;
        }

        var candidateBySequence = candidates.ToDictionary(
            static candidate => candidate.SequenceId);
        var reservationByDevice = reservationState.Reservations
            .Where(static reservation => reservation.Bytes > 0)
            .ToDictionary(static reservation => reservation.Device);
        var blocked = new Dictionary<DeviceId, long>();

        foreach (var deferred in decision.Deferred)
        {
            if (deferred.Reason != SchedulingDeferralReason.DeviceMemoryBudget ||
                !candidateBySequence.TryGetValue(deferred.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device ||
                !reservationByDevice.TryGetValue(device, out var reservation))
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
            IReadOnlyList<SchedulingCandidate> candidates,
            RuntimeDeviceInferenceReservationState reservationState)
    {
        if (decision.Batch.Items.Count != 0 ||
            reservationState.Reservations.Count == 0)
        {
            return null;
        }

        var candidateBySequence = candidates.ToDictionary(
            static candidate => candidate.SequenceId);
        var reservationByDevice = reservationState.Reservations
            .Where(static reservation => reservation.Items > 0)
            .ToDictionary(static reservation => reservation.Device);
        var blocked = new Dictionary<DeviceId, long>();

        foreach (var deferred in decision.Deferred)
        {
            if (deferred.Reason != SchedulingDeferralReason.DeviceSequenceBudget ||
                !candidateBySequence.TryGetValue(deferred.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device ||
                !reservationByDevice.TryGetValue(device, out var reservation))
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
            IReadOnlyList<SchedulingCandidate> candidates)
    {
        var candidateBySequence = candidates.ToDictionary(
            static candidate => candidate.SequenceId);
        var bytesByDevice = new Dictionary<DeviceId, long>();

        foreach (var item in decision.Batch.Items)
        {
            if (item.TransientKvByteGrant == 0 ||
                !candidateBySequence.TryGetValue(item.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            bytesByDevice.TryGetValue(device, out var existing);
            bytesByDevice[device] = checked(
                existing + item.TransientKvByteGrant);
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
            IReadOnlyList<SchedulingCandidate> candidates)
    {
        var candidateBySequence = candidates.ToDictionary(
            static candidate => candidate.SequenceId);
        var itemsByDevice = new Dictionary<DeviceId, int>();

        foreach (var item in decision.Batch.Items)
        {
            if (!candidateBySequence.TryGetValue(item.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            itemsByDevice.TryGetValue(device, out var existing);
            itemsByDevice[device] = checked(existing + 1);
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
        IReadOnlyList<SchedulingCandidate> candidates,
        CancellationToken cancellationToken)
    {
        if (_options.MaxDeviceBytes is not { } maxDeviceBytes)
        {
            return false;
        }

        var candidateBySequence = candidates.ToDictionary(
            static candidate => candidate.SequenceId);
        var selectedTransientByDevice = new Dictionary<DeviceId, long>();
        foreach (var item in decision.Batch.Items)
        {
            if (!candidateBySequence.TryGetValue(item.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            selectedTransientByDevice.TryGetValue(device, out var usedBytes);
            selectedTransientByDevice[device] = checked(
                usedBytes + item.TransientKvByteGrant);
        }

        var firstBlockedByDevice = new Dictionary<DeviceId, SchedulingCandidate>();
        foreach (var deferred in decision.Deferred)
        {
            if (deferred.Reason != SchedulingDeferralReason.DeviceMemoryBudget ||
                !candidateBySequence.TryGetValue(deferred.SequenceId, out var candidate) ||
                candidate.ExecutionDevice is not { } device)
            {
                continue;
            }

            firstBlockedByDevice.TryAdd(device, candidate);
        }

        if (firstBlockedByDevice.Count == 0)
        {
            return false;
        }

        var blockedDevices = firstBlockedByDevice.Keys
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .ToArray();
        var pressureByDevice = _runtime.GetDeviceMemoryPressureCore(blockedDevices)
            .ToDictionary(static pressure => pressure.Device);
        var reservedByDevice = _runtime.GetDeviceMemoryReservations(blockedDevices)
            .ToDictionary(static reservation => reservation.Device,
                static reservation => reservation.Bytes);
        var shouldReschedule = false;

        foreach (var (device, blockedCandidate) in firstBlockedByDevice)
        {
            if (!pressureByDevice.TryGetValue(device, out var pressure))
            {
                continue;
            }

            selectedTransientByDevice.TryGetValue(device, out var selectedTransientBytes);
            reservedByDevice.TryGetValue(device, out var outstandingReservationBytes);
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

    private long GetAvailableKvBytes(RequestView[] active)
    {
        if (_options.MaxKvBytes is not { } maxKvBytes)
        {
            return long.MaxValue;
        }

        long retainedBytes = 0;
        foreach (var request in active)
        {
            if (!_runtime.TryGetSequence(request.SequenceId, out var sequence) || sequence is null)
            {
                continue;
            }

            var bytesPerToken = GetKvBytesPerToken(request.ModelId);
            retainedBytes = checked(
                retainedBytes + checked((long)sequence.Position * bytesPerToken));
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
