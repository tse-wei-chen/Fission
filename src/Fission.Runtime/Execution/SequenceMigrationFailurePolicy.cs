namespace Fission.Runtime.Execution;

/// <summary>
/// Runtime-visible phase of a transport-aware sequence migration.
/// </summary>
public enum SequenceMigrationPhase
{
    EstimateBytes,
    SourceCapabilityDiscovery,
    TargetCapabilityDiscovery,
    Planning,
    Admission,
    Prepare,
    Attestation,
    Import,
    Commit,
    TargetRollback,
    SourceRollback
}

/// <summary>
/// Stable failure taxonomy for observability and later scheduler/health policy.
/// </summary>
public enum SequenceMigrationFailureClass
{
    CallerCanceled,
    TimedOut,
    AdmissionRejected,
    PlanningFailure,
    ProtocolViolation,
    BackendFailure
}

/// <summary>
/// Conservative device-health signal associated with one migration failure.
/// This is evidence for policy; it does not itself evict or restart a device.
/// </summary>
[Flags]
public enum SequenceMigrationHealthImpact
{
    None = 0,
    SourceSuspect = 1,
    TargetSuspect = 2,
    BothSuspect = SourceSuspect | TargetSuspect
}

public readonly record struct SequenceMigrationFailureClassification(
    SequenceMigrationPhase Phase,
    SequenceMigrationFailureClass FailureClass,
    SequenceMigrationHealthImpact HealthImpact,
    TimeSpan? Timeout = null);

/// <summary>
/// Cooperative phase deadlines. Infinite values preserve the pre-timeout runtime
/// behavior and are therefore the default. Backends must observe cancellation for
/// a phase deadline to complete; the runtime waits for that phase to unwind before
/// starting rollback so abort never races an in-flight import or commit.
/// </summary>
public sealed record SequenceMigrationTimeoutPolicy
{
    public static SequenceMigrationTimeoutPolicy Disabled { get; } = new();

    public TimeSpan EstimateBytesTimeout { get; init; } = Timeout.InfiniteTimeSpan;
    public TimeSpan CapabilityDiscoveryTimeout { get; init; } = Timeout.InfiniteTimeSpan;
    public TimeSpan AdmissionTimeout { get; init; } = Timeout.InfiniteTimeSpan;
    public TimeSpan PrepareTimeout { get; init; } = Timeout.InfiniteTimeSpan;
    public TimeSpan ImportTimeout { get; init; } = Timeout.InfiniteTimeSpan;
    public TimeSpan CommitTimeout { get; init; } = Timeout.InfiniteTimeSpan;
    public TimeSpan RollbackTimeout { get; init; } = Timeout.InfiniteTimeSpan;

    public TimeSpan GetTimeout(SequenceMigrationPhase phase) =>
        phase switch
        {
            SequenceMigrationPhase.EstimateBytes => EstimateBytesTimeout,
            SequenceMigrationPhase.SourceCapabilityDiscovery or
            SequenceMigrationPhase.TargetCapabilityDiscovery => CapabilityDiscoveryTimeout,
            SequenceMigrationPhase.Admission => AdmissionTimeout,
            SequenceMigrationPhase.Prepare => PrepareTimeout,
            SequenceMigrationPhase.Import => ImportTimeout,
            SequenceMigrationPhase.Commit => CommitTimeout,
            SequenceMigrationPhase.TargetRollback or
            SequenceMigrationPhase.SourceRollback => RollbackTimeout,
            SequenceMigrationPhase.Planning or
            SequenceMigrationPhase.Attestation => Timeout.InfiniteTimeSpan,
            _ => throw new ArgumentOutOfRangeException(nameof(phase), phase, null)
        };

    public void Validate()
    {
        ValidateTimeout(EstimateBytesTimeout, nameof(EstimateBytesTimeout));
        ValidateTimeout(CapabilityDiscoveryTimeout, nameof(CapabilityDiscoveryTimeout));
        ValidateTimeout(AdmissionTimeout, nameof(AdmissionTimeout));
        ValidateTimeout(PrepareTimeout, nameof(PrepareTimeout));
        ValidateTimeout(ImportTimeout, nameof(ImportTimeout));
        ValidateTimeout(CommitTimeout, nameof(CommitTimeout));
        ValidateTimeout(RollbackTimeout, nameof(RollbackTimeout));
    }

    private static void ValidateTimeout(TimeSpan timeout, string parameterName)
    {
        if (timeout != Timeout.InfiniteTimeSpan && timeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                timeout,
                "Migration timeout must be positive or Timeout.InfiniteTimeSpan.");
        }
    }
}

public sealed class SequenceMigrationTimeoutException : TimeoutException
{
    public SequenceMigrationTimeoutException(
        SequenceMigrationPhase phase,
        TimeSpan timeout,
        Exception? innerException = null)
        : base(
            $"Sequence migration phase {phase} exceeded its cooperative timeout of {timeout}.",
            innerException)
    {
        Phase = phase;
        Timeout = timeout;
    }

    public SequenceMigrationPhase Phase { get; }
    public TimeSpan Timeout { get; }
}

public static class SequenceMigrationFailureClassifier
{
    public static SequenceMigrationFailureClassification Classify(
        Exception failure,
        SequenceMigrationPhase phase,
        CancellationToken callerCancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(failure);

        if (failure is OperationCanceledException && callerCancellationToken.IsCancellationRequested)
        {
            return new SequenceMigrationFailureClassification(
                phase,
                SequenceMigrationFailureClass.CallerCanceled,
                SequenceMigrationHealthImpact.None);
        }

        if (failure is SequenceMigrationTimeoutException timeoutFailure)
        {
            return new SequenceMigrationFailureClassification(
                timeoutFailure.Phase,
                SequenceMigrationFailureClass.TimedOut,
                HealthImpactFor(timeoutFailure.Phase),
                timeoutFailure.Timeout);
        }

        var failureClass = phase switch
        {
            SequenceMigrationPhase.Admission =>
                SequenceMigrationFailureClass.AdmissionRejected,
            SequenceMigrationPhase.Planning =>
                SequenceMigrationFailureClass.PlanningFailure,
            SequenceMigrationPhase.Attestation =>
                SequenceMigrationFailureClass.ProtocolViolation,
            _ => SequenceMigrationFailureClass.BackendFailure
        };

        return new SequenceMigrationFailureClassification(
            phase,
            failureClass,
            failureClass is SequenceMigrationFailureClass.AdmissionRejected or
                SequenceMigrationFailureClass.PlanningFailure
                ? SequenceMigrationHealthImpact.None
                : HealthImpactFor(phase));
    }

    public static SequenceMigrationHealthImpact HealthImpactFor(
        SequenceMigrationPhase phase) =>
        phase switch
        {
            SequenceMigrationPhase.EstimateBytes or
            SequenceMigrationPhase.SourceCapabilityDiscovery or
            SequenceMigrationPhase.Prepare or
            SequenceMigrationPhase.Attestation or
            SequenceMigrationPhase.Commit or
            SequenceMigrationPhase.SourceRollback =>
                SequenceMigrationHealthImpact.SourceSuspect,

            SequenceMigrationPhase.TargetCapabilityDiscovery or
            SequenceMigrationPhase.Import or
            SequenceMigrationPhase.TargetRollback =>
                SequenceMigrationHealthImpact.TargetSuspect,

            SequenceMigrationPhase.Planning or
            SequenceMigrationPhase.Admission =>
                SequenceMigrationHealthImpact.None,

            _ => throw new ArgumentOutOfRangeException(nameof(phase), phase, null)
        };
}
