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
    DeviceId? ExecutionDevice = null,
    int KvPageWriteOverhead = 0);

/// <summary>
/// Physical device-memory headroom available for additional transient allocations
/// on one execution device. Existing active and idle-resident allocations are
/// already charged before this value reaches the scheduling kernel.
/// </summary>
public readonly record struct SchedulingDeviceMemoryBudget(
    DeviceId Device,
    long AvailableBytes);

/// <summary>
/// Maximum number of work items that may be selected for one physical execution
/// actor in the current atomic scheduler submission. This is independent from the
/// global MaxBatchSequences limit so heterogeneous device actors do not clamp one
/// another while each actor still receives a batch within its own capacity.
/// </summary>
public readonly record struct SchedulingDeviceSequenceBudget(
    DeviceId Device,
    int MaxSequences);

/// <summary>
/// Logical retained-KV byte slack remains independent from physical device-memory
/// headroom. The scheduler checks retained and immutable-successor KV constraints
/// against AvailableKvBytes, while DeviceMemory applies an additional per-device
/// transient allocation constraint when supplied. DeviceSequences independently
/// constrains the number of selected items sent to each physical execution actor.
/// </summary>
public readonly record struct SchedulingBudget(
    int MaxBatchTokens,
    int AvailableKvPages,
    int MaxBatchSequences,
    long AvailableKvBytes = long.MaxValue,
    IReadOnlyList<SchedulingDeviceMemoryBudget>? DeviceMemory = null,
    IReadOnlyList<SchedulingDeviceSequenceBudget>? DeviceSequences = null);

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
    BatchSequenceBudget,
    DeviceSequenceBudget
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
