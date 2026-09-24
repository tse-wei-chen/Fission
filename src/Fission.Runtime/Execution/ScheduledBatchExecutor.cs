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

    private PreparedItem[] Prepare(
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
        CompiledExecutionPlan Plan,
        ExecutionBindings Bindings);
}
