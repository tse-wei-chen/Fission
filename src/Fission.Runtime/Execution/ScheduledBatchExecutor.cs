using System.Buffers.Binary;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;

namespace Fission.Runtime.Execution;

public readonly record struct ScheduledPrefillBinding(
    ModelId ModelId,
    ReadOnlyMemory<int> Tokens);

public sealed class ScheduledExecutionBindings
{
    private readonly IReadOnlyDictionary<SequenceId, ScheduledPrefillBinding> _prefills;

    public ScheduledExecutionBindings(
        IReadOnlyDictionary<SequenceId, ScheduledPrefillBinding> prefills)
    {
        _prefills = prefills;
    }

    public ScheduledPrefillBinding ResolvePrefill(
        SequenceId sequenceId,
        int position,
        int expectedTokenCount,
        bool completesPrefill)
    {
        if (!_prefills.TryGetValue(sequenceId, out var binding))
        {
            throw new KeyNotFoundException(
                $"No scheduled prefill binding exists for sequence {sequenceId}.");
        }

        var end = checked(position + expectedTokenCount);
        if (end > binding.Tokens.Length)
        {
            throw new InvalidOperationException(
                $"Schedule grants tokens [{position}..{end}) for {sequenceId}, " +
                $"but the prompt contains only {binding.Tokens.Length} tokens.");
        }

        if (completesPrefill && end != binding.Tokens.Length)
        {
            throw new InvalidOperationException(
                $"Schedule marks the prefill chunk for {sequenceId} complete at position {end}, " +
                $"but the prompt length is {binding.Tokens.Length}.");
        }

        if (!completesPrefill && end >= binding.Tokens.Length)
        {
            throw new InvalidOperationException(
                $"Schedule marks the prefill chunk for {sequenceId} partial, " +
                "but the chunk consumes the remaining prompt.");
        }

        return binding with { Tokens = binding.Tokens.Slice(position, expectedTokenCount) };
    }
}

public sealed record ScheduledBatchResult(
    Guid ScheduleId,
    IReadOnlyList<ExecutionPlanResult> ItemResults);

/// <summary>
/// Bridges the F# scheduling policy and the C# stateful runtime. The whole
/// scheduled batch is validated first; only then are independent one-step plans
/// launched concurrently. Scheduler order is preserved independently on every
/// execution device by one atomic submission envelope per device, so producer
/// timing cannot change physical batch membership within an actor.
/// </summary>
public sealed class ScheduledBatchExecutor
{
    private static readonly ExecutionBindings EmptyExecutionBindings =
        new(new Dictionary<SequenceId, ReadOnlyMemory<int>>());

    private readonly ExecutionPlanExecutor _runtime;

    public ScheduledBatchExecutor(ExecutionPlanExecutor runtime)
    {
        _runtime = runtime;
    }

    public ValueTask<ScheduledBatchResult> ExecuteAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        CancellationToken cancellationToken = default) =>
        ExecuteCoreAsync(
            batch,
            bindings,
            onDeviceCompleted: null,
            cancellationToken);

    /// <summary>
    /// Runtime-internal execution path that reports each physical device exactly
    /// once after every scheduled plan targeting that actor reaches a terminal
    /// state. Other device groups may still be executing when the callback runs.
    /// </summary>
    internal ValueTask<ScheduledBatchResult> ExecuteAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        Action<DeviceId> onDeviceCompleted,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(onDeviceCompleted);
        return ExecuteCoreAsync(
            batch,
            bindings,
            onDeviceCompleted,
            cancellationToken);
    }

    private async ValueTask<ScheduledBatchResult> ExecuteCoreAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        Action<DeviceId>? onDeviceCompleted,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(batch);
        ArgumentNullException.ThrowIfNull(bindings);

        var countsByDevice = new Dictionary<DeviceId, int>();
        var prepared = Prepare(batch, bindings, countsByDevice);
        if (prepared.Length == 0)
        {
            return new ScheduledBatchResult(batch.ScheduleId, Array.Empty<ExecutionPlanResult>());
        }

        foreach (var (device, count) in countsByDevice)
        {
            var capacity = _runtime.GetDeviceInferenceCapacity(device);
            if (count > capacity)
            {
                throw new InvalidOperationException(
                    $"Scheduled batch {batch.ScheduleId} contains {count} inference item(s) " +
                    $"for device {device}, but that device actor capacity is {capacity}.");
            }
        }

        var submissions = new Dictionary<DeviceId, ContinuousBatchExecutor.AtomicSubmissionBatch>(
            countsByDevice.Count);
        foreach (var (device, count) in countsByDevice)
        {
            submissions.Add(
                device,
                ContinuousBatchExecutor.BeginAtomicSubmission(count));
        }

        try
        {
            var pending = new Task<ExecutionPlanResult>[prepared.Length];
            Dictionary<DeviceId, List<Task<ExecutionPlanResult>>>? pendingByDevice =
                onDeviceCompleted is null
                    ? null
                    : new Dictionary<DeviceId, List<Task<ExecutionPlanResult>>>(
                        countsByDevice.Count);

            for (var index = 0; index < prepared.Length; index++)
            {
                var task = ExecutePreparedAsync(index);
                pending[index] = task;

                if (pendingByDevice is null)
                {
                    continue;
                }

                var item = prepared[index];
                if (!pendingByDevice.TryGetValue(item.Device, out var devicePending))
                {
                    devicePending = new List<Task<ExecutionPlanResult>>(
                        countsByDevice[item.Device]);
                    pendingByDevice.Add(item.Device, devicePending);
                }

                devicePending.Add(task);
            }

            Task[]? deviceCompletionObservers = null;
            if (pendingByDevice is not null)
            {
                deviceCompletionObservers = new Task[pendingByDevice.Count];
                var observerIndex = 0;
                foreach (var (device, devicePending) in pendingByDevice)
                {
                    deviceCompletionObservers[observerIndex++] =
                        ObserveDeviceCompletionAsync(
                            device,
                            devicePending,
                            onDeviceCompleted!);
                }
            }

            try
            {
                var results = await Task.WhenAll(pending).ConfigureAwait(false);
                if (deviceCompletionObservers is not null)
                {
                    await Task.WhenAll(deviceCompletionObservers).ConfigureAwait(false);
                }

                return new ScheduledBatchResult(batch.ScheduleId, results);
            }
            catch
            {
                if (deviceCompletionObservers is not null)
                {
                    try
                    {
                        await Task.WhenAll(deviceCompletionObservers).ConfigureAwait(false);
                    }
                    catch
                    {
                        // Preserve the scheduled execution failure. Device-completion
                        // observers exist for cleanup and must not replace its cause.
                    }
                }

                throw;
            }
        }
        finally
        {
            foreach (var submission in submissions.Values)
            {
                submission.Dispose();
            }
        }

        async Task<ExecutionPlanResult> ExecutePreparedAsync(int index)
        {
            var item = prepared[index];
            var submission = submissions[item.Device];
            try
            {
                return await _runtime.ExecuteScheduledInferenceAsync(
                        item.PlanId,
                        item.Step,
                        item.Bindings,
                        submission,
                        item.Slot,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (Exception exception)
            {
                foreach (var pendingSubmission in submissions.Values)
                {
                    pendingSubmission.Abort(exception);
                }

                throw;
            }
        }
    }

    private static async Task ObserveDeviceCompletionAsync(
        DeviceId device,
        IReadOnlyList<Task<ExecutionPlanResult>> pending,
        Action<DeviceId> onDeviceCompleted)
    {
        try
        {
            await Task.WhenAll(pending).ConfigureAwait(false);
        }
        catch
        {
            // The main batch await owns execution failure propagation. The device
            // observer still runs its terminal callback so per-device resources can
            // be released even when this group faults or is cancelled.
        }
        finally
        {
            onDeviceCompleted(device);
        }
    }

    private PreparedItem[] Prepare(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        Dictionary<DeviceId, int> countsByDevice)
    {
        var prepared = new PreparedItem[batch.Items.Count];
        var sequences = new HashSet<SequenceId>();
        var consumedTokens = 0;
        var consumedKvPages = 0;
        var consumedKvBytes = 0L;
        var consumedTransientKvBytes = 0L;

        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(item.TokenGrant);
            ArgumentOutOfRangeException.ThrowIfNegative(item.KvPageGrant);
            ArgumentOutOfRangeException.ThrowIfNegative(item.KvByteGrant);
            ArgumentOutOfRangeException.ThrowIfNegative(item.TransientKvByteGrant);

            if (!sequences.Add(item.SequenceId))
            {
                throw new InvalidOperationException(
                    $"Scheduled batch {batch.ScheduleId} contains sequence {item.SequenceId} more than once.");
            }

            var hasExistingSequence = _runtime.TryGetSequence(item.SequenceId, out var existingSequence);
            var position = existingSequence?.Position ?? 0;
            var expectedKvPages = _runtime.KvPages.IncrementalPagesFor(position, item.TokenGrant);
            if (expectedKvPages != item.KvPageGrant)
            {
                throw new InvalidOperationException(
                    $"Schedule grants {item.KvPageGrant} KV page(s) for {item.SequenceId}, " +
                    $"but runtime position {position} requires {expectedKvPages} page(s) " +
                    $"at {_runtime.KvPages.TokensPerPage} tokens/page.");
            }

            consumedTokens = checked(consumedTokens + item.TokenGrant);
            consumedKvPages = checked(consumedKvPages + item.KvPageGrant);
            consumedKvBytes = checked(consumedKvBytes + item.KvByteGrant);
            consumedTransientKvBytes = checked(
                consumedTransientKvBytes + item.TransientKvByteGrant);

            ExecutionStep step;
            ExecutionBindings executionBindings;

            switch (item.Kind)
            {
                case ScheduledWorkKind.Prefill:
                {
                    var binding = bindings.ResolvePrefill(
                        item.SequenceId,
                        position,
                        item.TokenGrant,
                        item.CompletesPrefill);
                    step = new PrefillExecutionStep(
                        item.SequenceId,
                        binding.ModelId,
                        item.TokenGrant,
                        item.CompletesPrefill);
                    executionBindings = new ExecutionBindings(
                        new Dictionary<SequenceId, ReadOnlyMemory<int>>
                        {
                            [item.SequenceId] = binding.Tokens
                        });
                    break;
                }

                case ScheduledWorkKind.Decode:
                    if (!hasExistingSequence)
                    {
                        throw new KeyNotFoundException(
                            $"Scheduled decode sequence {item.SequenceId} does not exist in the runtime.");
                    }

                    if (item.TokenGrant != 1)
                    {
                        throw new InvalidOperationException(
                            $"Decode schedule for {item.SequenceId} must grant exactly one token.");
                    }

                    step = new DecodeExecutionStep(item.SequenceId, 1);
                    executionBindings = EmptyExecutionBindings;
                    break;

                default:
                    throw new NotSupportedException(
                        $"Unsupported scheduled work kind {item.Kind}.");
            }

            var device = _runtime.ResolveExecutionDevice(item.SequenceId);
            countsByDevice.TryGetValue(device, out var deviceCount);
            countsByDevice[device] = checked(deviceCount + 1);

            prepared[index] = new PreparedItem(
                DerivePlanId(batch.ScheduleId, index),
                step,
                executionBindings,
                device,
                deviceCount);
        }

        if (consumedTokens != batch.ConsumedTokens)
        {
            throw new InvalidOperationException(
                $"Scheduled batch {batch.ScheduleId} reports {batch.ConsumedTokens} consumed tokens, " +
                $"but its work items sum to {consumedTokens}.");
        }

        if (consumedKvPages != batch.ConsumedKvPages)
        {
            throw new InvalidOperationException(
                $"Scheduled batch {batch.ScheduleId} reports {batch.ConsumedKvPages} consumed KV pages, " +
                $"but its work items sum to {consumedKvPages}.");
        }

        if (consumedKvBytes != batch.ConsumedKvBytes)
        {
            throw new InvalidOperationException(
                $"Scheduled batch {batch.ScheduleId} reports {batch.ConsumedKvBytes} retained KV bytes, " +
                $"but its work items sum to {consumedKvBytes}.");
        }

        if (consumedTransientKvBytes != batch.ConsumedTransientKvBytes)
        {
            throw new InvalidOperationException(
                $"Scheduled batch {batch.ScheduleId} reports {batch.ConsumedTransientKvBytes} transient KV bytes, " +
                $"but its work items sum to {consumedTransientKvBytes}.");
        }

        if (consumedKvPages > _runtime.KvPages.AvailablePages)
        {
            throw new InvalidOperationException(
                $"Scheduled batch {batch.ScheduleId} needs {consumedKvPages} KV page(s), " +
                $"but runtime has only {_runtime.KvPages.AvailablePages} available.");
        }

        return prepared;
    }

    private static Guid DerivePlanId(Guid scheduleId, int index)
    {
        Span<byte> bytes = stackalloc byte[16];
        scheduleId.TryWriteBytes(bytes);

        var suffix = BinaryPrimitives.ReadInt32LittleEndian(bytes[12..]);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[12..], suffix ^ checked(index + 1));
        return new Guid(bytes);
    }

    private readonly record struct PreparedItem(
        Guid PlanId,
        ExecutionStep Step,
        ExecutionBindings Bindings,
        DeviceId Device,
        int Slot);
}
