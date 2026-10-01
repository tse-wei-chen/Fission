using System.Buffers.Binary;
using System.Runtime.ExceptionServices;
using Fission.Abstractions;
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

internal sealed class ScheduledBatchFailureCoordinator(
    Dictionary<DeviceId, ContinuousBatchExecutor.AtomicSubmissionBatch> submissions)
{
    internal void Abort(Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);
        foreach (var submission in submissions.Values)
        {
            submission.Abort(exception);
        }
    }
}

internal sealed class ScheduledDeviceCompletionTracker(
    DeviceId device,
    int remaining,
    Action<DeviceId> onCompleted)
{
    private int _remaining = remaining > 0
        ? remaining
        : throw new ArgumentOutOfRangeException(nameof(remaining));
    private ExceptionDispatchInfo? _failure;

    internal void Complete()
    {
        if (Interlocked.Decrement(ref _remaining) != 0)
        {
            return;
        }

        try
        {
            onCompleted(device);
        }
        catch (Exception exception)
        {
            Volatile.Write(ref _failure, ExceptionDispatchInfo.Capture(exception));
        }
    }

    internal void ThrowIfFailed() => Volatile.Read(ref _failure)?.Throw();
}

/// <summary>
/// Bridges the F# scheduling policy and the C# stateful runtime. The whole
/// scheduled batch is validated first; only then are independent one-step plans
/// launched concurrently. Scheduler order is preserved independently on every
/// execution device by one atomic submission envelope per device, so producer
/// timing cannot change physical batch membership within an actor.
/// </summary>
public sealed class ScheduledBatchExecutor
{
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
            var failureCoordinator = new ScheduledBatchFailureCoordinator(submissions);
            Dictionary<DeviceId, ScheduledDeviceCompletionTracker>? completionTrackers = null;
            if (onDeviceCompleted is not null)
            {
                completionTrackers = new Dictionary<DeviceId, ScheduledDeviceCompletionTracker>(
                    countsByDevice.Count);
                foreach (var (device, count) in countsByDevice)
                {
                    completionTrackers.Add(
                        device,
                        new ScheduledDeviceCompletionTracker(device, count, onDeviceCompleted));
                }
            }

            var pending = new ValueTask<ExecutionPlanResult>[prepared.Length];
            for (var index = 0; index < prepared.Length; index++)
            {
                var item = prepared[index];
                var completionTracker = completionTrackers is null
                    ? null
                    : completionTrackers[item.Device];

                pending[index] = _runtime.ExecuteScheduledInferenceAsync(
                    item.PlanId,
                    item.Step,
                    item.PrefillTokens,
                    submissions[item.Device],
                    item.Slot,
                    failureCoordinator,
                    completionTracker,
                    cancellationToken);
            }

            var results = await AwaitAllAsync(pending).ConfigureAwait(false);
            if (completionTrackers is not null)
            {
                foreach (var tracker in completionTrackers.Values)
                {
                    tracker.ThrowIfFailed();
                }
            }

            return new ScheduledBatchResult(batch.ScheduleId, results);
        }
        finally
        {
            foreach (var submission in submissions.Values)
            {
                submission.Dispose();
            }
        }
    }

    private static async ValueTask<ExecutionPlanResult[]> AwaitAllAsync(
        ValueTask<ExecutionPlanResult>[] pending)
    {
        var results = new ExecutionPlanResult[pending.Length];
        ExceptionDispatchInfo? firstFailure = null;
        ExceptionDispatchInfo? firstCancellation = null;

        for (var index = 0; index < pending.Length; index++)
        {
            try
            {
                results[index] = await pending[index].ConfigureAwait(false);
            }
            catch (OperationCanceledException exception)
            {
                firstCancellation ??= ExceptionDispatchInfo.Capture(exception);
            }
            catch (Exception exception)
            {
                firstFailure ??= ExceptionDispatchInfo.Capture(exception);
            }
        }

        firstFailure?.Throw();
        firstCancellation?.Throw();
        return results;
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

            ScheduledInferenceStep step;
            ReadOnlyMemory<int> prefillTokens = default;

            switch (item.Kind)
            {
                case ScheduledWorkKind.Prefill:
                {
                    var binding = bindings.ResolvePrefill(
                        item.SequenceId,
                        position,
                        item.TokenGrant,
                        item.CompletesPrefill);
                    step = ScheduledInferenceStep.Prefill(
                        item.SequenceId,
                        binding.ModelId,
                        item.TokenGrant,
                        item.CompletesPrefill);
                    prefillTokens = binding.Tokens;
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

                    step = ScheduledInferenceStep.Decode(item.SequenceId);
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
                prefillTokens,
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
        ScheduledInferenceStep Step,
        ReadOnlyMemory<int> PrefillTokens,
        DeviceId Device,
        int Slot);
}
