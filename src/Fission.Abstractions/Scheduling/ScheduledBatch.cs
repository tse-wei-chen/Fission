namespace Fission.Abstractions.Scheduling;

public enum ScheduledWorkKind
{
    Prefill,
    Decode
}

/// <summary>
/// C#-friendly hot-path contract emitted by the F# scheduling policy.
/// TokenGrant and KvPageGrant are the resources admitted for this scheduling quantum.
/// </summary>
public readonly record struct ScheduledWorkItem(
    SequenceId SequenceId,
    ScheduledWorkKind Kind,
    int TokenGrant,
    int KvPageGrant,
    int Priority);

public sealed record ScheduledBatch(
    Guid ScheduleId,
    IReadOnlyList<ScheduledWorkItem> Items,
    int ConsumedTokens,
    int ConsumedKvPages);
