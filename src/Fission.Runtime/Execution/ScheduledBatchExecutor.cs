using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.ExceptionServices;
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

internal sealed class ScheduledDeviceGroup(DeviceId device)
{
    private ContinuousBatchExecutor.AtomicSubmissionBatch? _submission;
    private Action<DeviceId>? _onCompleted;
    private int _remaining;
    private ExceptionDispatchInfo? _completionFailure;

    internal DeviceId Device { get; } = device;
    internal int Count { get; private set; }

    internal ContinuousBatchExecutor.AtomicSubmissionBatch Submission =>
        _submission ?? throw new InvalidOperationException(
            $"Scheduled device group {Device} has not initialized its atomic submission.");

    internal int RegisterItem()
    {
        var slot = Count;
        Count = checked(Count + 1);
        return slot;
    }

    internal void InitializeSubmission(Action<DeviceId>? onCompleted)
    {
        if (_submission is not null)
        {
            throw new InvalidOperationException(
                $"Scheduled device group {Device} already initialized its atomic submission.");
        }

        if (Count <= 0)
        {
            throw new InvalidOperationException(
                $"Scheduled device group {Device} cannot initialize without work items.");
        }

        _submission = ContinuousBatchExecutor.BeginAtomicSubmission(Count);
        _onCompleted = onCompleted;
        _remaining = Count;
    }

    internal void CompleteItem()
    {
        var onCompleted = _onCompleted;
        if (onCompleted is null)
        {
            return;
        }

        if (Interlocked.Decrement(ref _remaining) != 0)
        {
            return;
        }

        try
        {
            onCompleted(Device);
        }
        catch (Exception exception)
        {
            Volatile.Write(
                ref _completionFailure,
                ExceptionDispatchInfo.Capture(exception));
        }
    }

    internal void ThrowIfCompletionFailed() =>
        Volatile.Read(ref _completionFailure)?.Throw();

    internal void DisposeSubmission() => _submission?.Dispose();
}

internal sealed class ScheduledDeviceGroupTable
{
    private DeviceId _singleDevice;
    private ScheduledDeviceGroup? _singleGroup;
    private Dictionary<DeviceId, ScheduledDeviceGroup>? _multipleGroups;

    internal GroupEnumerable Values => new(_singleGroup, _multipleGroups);

    internal bool TryGetValue(
        DeviceId device,
        out ScheduledDeviceGroup group)
    {
        if (_multipleGroups is not null)
        {
            return _multipleGroups.TryGetValue(device, out group!);
        }

        if (_singleGroup is { } singleGroup && _singleDevice.Equals(device))
        {
            group = singleGroup;
            return true;
        }

        group = null!;
        return false;
    }

    internal void Add(DeviceId device, ScheduledDeviceGroup group)
    {
        ArgumentNullException.ThrowIfNull(group);

        if (_multipleGroups is not null)
        {
            _multipleGroups.Add(device, group);
            return;
        }

        if (_singleGroup is null)
        {
            _singleDevice = device;
            _singleGroup = group;
            return;
        }

        var multipleGroups = new Dictionary<DeviceId, ScheduledDeviceGroup>(2)
        {
            [_singleDevice] = _singleGroup
        };
        multipleGroups.Add(device, group);

        _singleDevice = default;
        _singleGroup = null;
        _multipleGroups = multipleGroups;
    }

    internal void Abort(Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);
        foreach (var group in Values)
        {
            group.Submission.Abort(exception);
        }
    }

    internal readonly struct GroupEnumerable
    {
        private readonly ScheduledDeviceGroup? _singleGroup;
        private readonly Dictionary<DeviceId, ScheduledDeviceGroup>? _multipleGroups;

        internal GroupEnumerable(
            ScheduledDeviceGroup? singleGroup,
            Dictionary<DeviceId, ScheduledDeviceGroup>? multipleGroups)
        {
            _singleGroup = singleGroup;
            _multipleGroups = multipleGroups;
        }

        public GroupEnumerator GetEnumerator() =>
            new(_singleGroup, _multipleGroups);
    }

    internal struct GroupEnumerator
    {
        private readonly ScheduledDeviceGroup? _singleGroup;
        private Dictionary<DeviceId, ScheduledDeviceGroup>.ValueCollection.Enumerator
            _multipleEnumerator;
        private readonly bool _useMultiple;
        private bool _singlePending;
        private ScheduledDeviceGroup? _current;

        internal GroupEnumerator(
            ScheduledDeviceGroup? singleGroup,
            Dictionary<DeviceId, ScheduledDeviceGroup>? multipleGroups)
        {
            _singleGroup = singleGroup;
            _multipleEnumerator = multipleGroups is null
                ? default
                : multipleGroups.Values.GetEnumerator();
            _useMultiple = multipleGroups is not null;
            _singlePending = multipleGroups is null && singleGroup is not null;
            _current = null;
        }

        public ScheduledDeviceGroup Current => _current!;

        public bool MoveNext()
        {
            if (_useMultiple)
            {
                if (_multipleEnumerator.MoveNext())
                {
                    _current = _multipleEnumerator.Current;
                    return true;
                }

                _current = null;
                return false;
            }

            if (_singlePending)
            {
                _singlePending = false;
                _current = _singleGroup;
                return true;
            }

            _current = null;
            return false;
        }

        public void Dispose()
        {
            if (_useMultiple)
            {
                _multipleEnumerator.Dispose();
            }
        }
    }
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
        ExecutePublicAsync(
            batch,
            bindings,
            onDeviceCompleted: null,
            cancellationToken);

    /// <summary>
    /// Runtime-internal compatibility path that reports each physical device
    /// exactly once after every scheduled plan targeting that actor reaches a
    /// terminal state. Engine hot paths use ExecuteBackendAsync to avoid
    /// materializing per-item ExecutionPlanResult wrappers.
    /// </summary>
    internal ValueTask<ScheduledBatchResult> ExecuteAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        Action<DeviceId> onDeviceCompleted,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(onDeviceCompleted);
        return ExecutePublicAsync(
            batch,
            bindings,
            onDeviceCompleted,
            cancellationToken);
    }

    internal ValueTask<BackendStepResult[]> ExecuteBackendAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        CancellationToken cancellationToken = default) =>
        ExecuteBackendCoreAsync(
            batch,
            bindings,
            onDeviceCompleted: null,
            cancellationToken);

    internal ValueTask<BackendStepResult[]> ExecuteBackendAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        Action<DeviceId> onDeviceCompleted,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(onDeviceCompleted);
        return ExecuteBackendCoreAsync(
            batch,
            bindings,
            onDeviceCompleted,
            cancellationToken);
    }

    private async ValueTask<ScheduledBatchResult> ExecutePublicAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        Action<DeviceId>? onDeviceCompleted,
        CancellationToken cancellationToken)
    {
        var backendResults = await ExecuteBackendCoreAsync(
                batch,
                bindings,
                onDeviceCompleted,
                cancellationToken)
            .ConfigureAwait(false);

        if (backendResults.Length == 0)
        {
            return new ScheduledBatchResult(
                batch.ScheduleId,
                Array.Empty<ExecutionPlanResult>());
        }

        var itemResults = new ExecutionPlanResult[backendResults.Length];
        for (var index = 0; index < backendResults.Length; index++)
        {
            itemResults[index] = new ExecutionPlanResult(
                DerivePlanId(batch.ScheduleId, index),
                new[] { backendResults[index] },
                Array.Empty<KvSnapshotId>(),
                Array.Empty<ForkExecutionResult>());
        }

        return new ScheduledBatchResult(batch.ScheduleId, itemResults);
    }

    private async ValueTask<BackendStepResult[]> ExecuteBackendCoreAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        Action<DeviceId>? onDeviceCompleted,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(batch);
        ArgumentNullException.ThrowIfNull(bindings);

        var groupsByDevice = new ScheduledDeviceGroupTable();
        var itemCount = batch.Items.Count;
        var prepared = Prepare(batch, bindings, groupsByDevice);
        if (itemCount == 0)
        {
            return Array.Empty<BackendStepResult>();
        }

        try
        {
            foreach (var group in groupsByDevice.Values)
            {
                var capacity = _runtime.GetDeviceInferenceCapacity(group.Device);
                if (group.Count > capacity)
                {
                    throw new InvalidOperationException(
                        $"Scheduled batch {batch.ScheduleId} contains {group.Count} inference item(s) " +
                        $"for device {group.Device}, but that device actor capacity is {capacity}.");
                }
            }

            foreach (var group in groupsByDevice.Values)
            {
                group.InitializeSubmission(onDeviceCompleted);
            }

            for (var index = 0; index < itemCount; index++)
            {
                ref var item = ref prepared[index];
                item.Pending = _runtime.ExecuteScheduledInferenceAsync(
                    item.PlanId,
                    item.Step,
                    item.PrefillTokens,
                    item.Group,
                    item.Slot,
                    groupsByDevice,
                    cancellationToken);
            }

            var results = await AwaitAllAsync(prepared, itemCount).ConfigureAwait(false);
            if (onDeviceCompleted is not null)
            {
                foreach (var group in groupsByDevice.Values)
                {
                    group.ThrowIfCompletionFailed();
                }
            }

            return results;
        }
        finally
        {
            foreach (var group in groupsByDevice.Values)
            {
                group.DisposeSubmission();
            }

            ReturnPreparedItems(prepared, itemCount);
        }
    }

    private static async ValueTask<BackendStepResult[]> AwaitAllAsync(
        PreparedItem[] prepared,
        int itemCount)
    {
        var results = new BackendStepResult[itemCount];
        ExceptionDispatchInfo? firstFailure = null;
        ExceptionDispatchInfo? firstCancellation = null;

        for (var index = 0; index < itemCount; index++)
        {
            try
            {
                results[index] = await prepared[index].Pending.ConfigureAwait(false);
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
        ScheduledDeviceGroupTable groupsByDevice)
    {
        var itemCount = batch.Items.Count;
        if (itemCount == 0)
        {
            return Array.Empty<PreparedItem>();
        }

        var prepared = ArrayPool<PreparedItem>.Shared.Rent(itemCount);
        try
        {
            var sequences = new HashSet<SequenceId>();
            var consumedTokens = 0;
            var consumedKvPages = 0;
            var consumedKvBytes = 0L;
            var consumedTransientKvBytes = 0L;

            for (var index = 0; index < itemCount; index++)
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
                if (!groupsByDevice.TryGetValue(device, out var group))
                {
                    group = new ScheduledDeviceGroup(device);
                    groupsByDevice.Add(device, group);
                }

                var slot = group.RegisterItem();
                prepared[index] = new PreparedItem(
                    DerivePlanId(batch.ScheduleId, index),
                    step,
                    prefillTokens,
                    group,
                    slot);
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
        catch
        {
            ReturnPreparedItems(prepared, itemCount);
            throw;
        }
    }

    private static void ReturnPreparedItems(PreparedItem[] prepared, int itemCount)
    {
        Array.Clear(prepared, 0, itemCount);
        ArrayPool<PreparedItem>.Shared.Return(prepared);
    }

    private static Guid DerivePlanId(Guid scheduleId, int index)
    {
        Span<byte> bytes = stackalloc byte[16];
        scheduleId.TryWriteBytes(bytes);

        var suffix = BinaryPrimitives.ReadInt32LittleEndian(bytes[12..]);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[12..], suffix ^ checked(index + 1));
        return new Guid(bytes);
    }

    private struct PreparedItem
    {
        internal PreparedItem(
            Guid planId,
            ScheduledInferenceStep step,
            ReadOnlyMemory<int> prefillTokens,
            ScheduledDeviceGroup group,
            int slot)
        {
            PlanId = planId;
            Step = step;
            PrefillTokens = prefillTokens;
            Group = group;
            Slot = slot;
            Pending = default;
        }

        internal Guid PlanId { get; }
        internal ScheduledInferenceStep Step { get; }
        internal ReadOnlyMemory<int> PrefillTokens { get; }
        internal ScheduledDeviceGroup Group { get; }
        internal int Slot { get; }
        internal ValueTask<BackendStepResult> Pending { get; set; }
    }
}
