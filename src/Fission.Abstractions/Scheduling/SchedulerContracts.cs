namespace Fission.Abstractions.Scheduling;

public enum SchedulingPhase
{
    Waiting,
    Prefilling,
    Decoding,
    Suspended,
    Finished,
    Cancelled
}

public readonly record struct SchedulingCandidate(
    SequenceId SequenceId,
    SchedulingPhase Phase,
    DateTimeOffset? Deadline,
    DateTimeOffset EnqueuedAt,
    int TokenDemand,
    int Position,
    int TokensPerKvPage,
    int Priority);

public readonly record struct SchedulingBudget(
    int MaxBatchTokens,
    int AvailableKvPages,
    int MaxBatchSequences);

public readonly record struct SchedulingPolicyOptions(
    int DecodeTokenReserve,
    int MaxPrefillChunkTokens,
    TimeSpan DeadlineUrgencyWindow);

public enum SchedulingDeferralReason
{
    NotRunnable,
    TokenBudget,
    KvBudget,
    BatchSequenceBudget
}

public enum SchedulingRejectionReason
{
    InvalidTokenDemand,
    InvalidPosition,
    InvalidKvPageSize,
    InvalidDecodeQuantum,
    InvalidKvPageDemand,
    TokenDemandExceedsBatchCapacity,
    KvDemandExceedsCapacity
}

public readonly record struct SchedulingDeferral(
    SequenceId SequenceId,
    SchedulingDeferralReason Reason);

public readonly record struct SchedulingRejection(
    SequenceId SequenceId,
    SchedulingRejectionReason Reason);

public sealed record SchedulingKernelResult(
    ScheduledBatch Batch,
    IReadOnlyList<SchedulingDeferral> Deferred,
    IReadOnlyList<SchedulingRejection> Rejected);

public interface ISchedulingKernel
{
    SchedulingKernelResult Schedule(
        Guid scheduleId,
        DateTimeOffset now,
        SchedulingBudget budget,
        SchedulingPolicyOptions policy,
        IReadOnlyList<SchedulingCandidate> candidates);
}
