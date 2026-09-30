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
    int Priority,
    long KvBytesPerToken = 0,
    DeviceId? ExecutionDevice = null);

/// <summary>
/// Physical device-memory headroom available for additional transient allocations
/// on one execution device. Existing active and idle-resident allocations are
/// already charged before this value reaches the scheduling kernel.
/// </summary>
public readonly record struct SchedulingDeviceMemoryBudget(
    DeviceId Device,
    long AvailableBytes);

/// <summary>
/// Logical retained-KV byte slack remains independent from physical device-memory
/// headroom. The scheduler checks retained and immutable-successor KV constraints
/// against AvailableKvBytes, while DeviceMemory applies an additional per-device
/// transient allocation constraint when supplied.
/// </summary>
public readonly record struct SchedulingBudget(
    int MaxBatchTokens,
    int AvailableKvPages,
    int MaxBatchSequences,
    long AvailableKvBytes = long.MaxValue,
    IReadOnlyList<SchedulingDeviceMemoryBudget>? DeviceMemory = null);

public readonly record struct SchedulingPolicyOptions(
    int DecodeTokenReserve,
    int MaxPrefillChunkTokens,
    TimeSpan DeadlineUrgencyWindow);

public enum SchedulingDeferralReason
{
    NotRunnable,
    TokenBudget,
    KvBudget,
    KvByteBudget,
    TransientKvByteBudget,
    DeviceMemoryBudget,
    BatchSequenceBudget
}

public enum SchedulingRejectionReason
{
    InvalidTokenDemand,
    InvalidPosition,
    InvalidKvPageSize,
    InvalidDecodeQuantum,
    InvalidKvPageDemand,
    InvalidKvBytesPerToken,
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
