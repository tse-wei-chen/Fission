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
        int expectedTokenCount)
    {
        if (!_prefills.TryGetValue(sequenceId, out var binding))
        {
            throw new KeyNotFoundException(
                $"No scheduled prefill binding exists for sequence {sequenceId}.");
        }

        if (binding.Tokens.Length != expectedTokenCount)
        {
            throw new InvalidOperationException(
                $"Schedule grants {expectedTokenCount} prefill tokens for {sequenceId}, " +
                $"but binding contains {binding.Tokens.Length}.");
        }

        return binding;
    }
}

public sealed record ScheduledBatchResult(
    Guid ScheduleId,
    IReadOnlyList<ExecutionPlanResult> ItemResults);

/// <summary>
/// Bridges the F# scheduling policy and the C# stateful runtime. The whole
/// scheduled batch is validated first; only then are independent one-step
/// plans launched concurrently so ContinuousBatchExecutor can coalesce them.
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

    public async ValueTask<ScheduledBatchResult> ExecuteAsync(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(batch);
        ArgumentNullException.ThrowIfNull(bindings);

        var prepared = Prepare(batch, bindings);
        if (prepared.Length == 0)
        {
            return new ScheduledBatchResult(batch.ScheduleId, Array.Empty<ExecutionPlanResult>());
        }

        var pending = new Task<ExecutionPlanResult>[prepared.Length];
        for (var index = 0; index < prepared.Length; index++)
        {
            pending[index] = _runtime.ExecuteAsync(
                prepared[index].Plan,
                prepared[index].Bindings,
                cancellationToken).AsTask();
        }

        var results = await Task.WhenAll(pending).ConfigureAwait(false);
        return new ScheduledBatchResult(batch.ScheduleId, results);
    }

    private static PreparedItem[] Prepare(
        ScheduledBatch batch,
        ScheduledExecutionBindings bindings)
    {
        var prepared = new PreparedItem[batch.Items.Count];
        var sequences = new HashSet<SequenceId>();
        var consumedTokens = 0;
        var consumedKvPages = 0;

        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(item.TokenGrant);
            ArgumentOutOfRangeException.ThrowIfNegative(item.KvPageGrant);

            if (!sequences.Add(item.SequenceId))
            {
                throw new InvalidOperationException(
                    $"Scheduled batch {batch.ScheduleId} contains sequence {item.SequenceId} more than once.");
            }

            consumedTokens = checked(consumedTokens + item.TokenGrant);
            consumedKvPages = checked(consumedKvPages + item.KvPageGrant);

            ExecutionStep step;
            ExecutionBindings executionBindings;

            switch (item.Kind)
            {
                case ScheduledWorkKind.Prefill:
                {
                    var binding = bindings.ResolvePrefill(item.SequenceId, item.TokenGrant);
                    step = new PrefillExecutionStep(
                        item.SequenceId,
                        binding.ModelId,
                        item.TokenGrant);
                    executionBindings = new ExecutionBindings(
                        new Dictionary<SequenceId, ReadOnlyMemory<int>>
                        {
                            [item.SequenceId] = binding.Tokens
                        });
                    break;
                }

                case ScheduledWorkKind.Decode:
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

            prepared[index] = new PreparedItem(
                new CompiledExecutionPlan(
                    DerivePlanId(batch.ScheduleId, index),
                    item.Priority,
                    new[] { step }),
                executionBindings);
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
        CompiledExecutionPlan Plan,
        ExecutionBindings Bindings);
}
