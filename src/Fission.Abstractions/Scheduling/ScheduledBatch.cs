namespace Fission.Abstractions.Scheduling;

public enum ScheduledWorkKind
{
    Prefill,
    Decode
}

/// <summary>
/// C#-friendly hot-path contract emitted by the F# scheduling policy.
/// TokenGrant, KvPageGrant, and KvByteGrant are the retained resources admitted
/// for this scheduling quantum. TransientKvByteGrant is the temporary immutable
/// successor-frontier reservation required while the prior state is still live.
/// CompletesPrefill marks the final chunk that may transition a sequence to decode.
/// </summary>
public readonly record struct ScheduledWorkItem(
    SequenceId SequenceId,
    ScheduledWorkKind Kind,
    int TokenGrant,
    int KvPageGrant,
    int Priority,
    bool CompletesPrefill,
    long KvByteGrant = 0,
    long TransientKvByteGrant = 0);

public sealed record ScheduledBatch(
    Guid ScheduleId,
    IReadOnlyList<ScheduledWorkItem> Items,
    int ConsumedTokens,
    int ConsumedKvPages,
    long ConsumedKvBytes = 0,
    long ConsumedTransientKvBytes = 0);

public readonly record struct RuntimeKvCapacity(
    int CapacityPages,
    int AllocatedPages,
    int AvailablePages,
    int TokensPerPage);
