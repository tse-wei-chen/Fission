namespace Fission.Abstractions.Execution;

public abstract record ExecutionStep(SequenceId SequenceId);

public sealed record PrefillExecutionStep(
    SequenceId SequenceId,
    ModelId ModelId,
    int TokenCount,
    bool CompletesPrefill) : ExecutionStep(SequenceId);

public sealed record DecodeExecutionStep(
    SequenceId SequenceId,
    int MaxTokens) : ExecutionStep(SequenceId);

public sealed record ForkKvExecutionStep(
    SequenceId SequenceId,
    int Branches) : ExecutionStep(SequenceId);

public sealed record SnapshotKvExecutionStep(
    SequenceId SequenceId) : ExecutionStep(SequenceId);

public sealed record RestoreKvExecutionStep(
    SequenceId SequenceId,
    KvSnapshotId SnapshotId) : ExecutionStep(SequenceId);

public sealed record MigrateKvExecutionStep(
    SequenceId SequenceId,
    DeviceId TargetDevice) : ExecutionStep(SequenceId);

public sealed record CompiledExecutionPlan(
    Guid PlanId,
    int Priority,
    IReadOnlyList<ExecutionStep> Steps);
