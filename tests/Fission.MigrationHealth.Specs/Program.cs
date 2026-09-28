using System.Diagnostics.CodeAnalysis;
using Fission.Abstractions;
using Fission.Runtime.Execution;
using Fission.Runtime.Tracing;

static void Require(
    [DoesNotReturnIf(false)] bool condition,
    string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

var source = new DeviceId("gpu:health-source");
var target = new DeviceId("gpu:health-target");
var inner = new InMemoryExecutionTraceSink();
var health = new SequenceMigrationHealthTraceSink(inner);

health.Record(new ExecutionTraceEvent(
    Guid.NewGuid(),
    ExecutionTraceKind.PlanStarted,
    StepIndex: -1,
    Operation: "plan"));
Require(health.SuspectDeviceCount == 0, "Non-migration events must not create health evidence.");

health.Record(new ExecutionTraceEvent(
    Guid.NewGuid(),
    ExecutionTraceKind.MigrationRolledBack,
    StepIndex: 0,
    Operation: "MigrateKvExecutionStep",
    Device: source,
    TargetDevice: target,
    MigrationPhase: SequenceMigrationPhase.Import,
    MigrationFailureClass: SequenceMigrationFailureClass.TimedOut,
    MigrationHealthImpact: SequenceMigrationHealthImpact.TargetSuspect,
    MigrationTimeout: TimeSpan.FromMilliseconds(25)));

Require(health.TryGet(target, out var targetHealth), "Target health evidence must be queryable.");
Require(targetHealth.IsSuspect, "Any retained migration health evidence must mark the device suspect.");
Require(targetHealth.EvidenceCount == 1, "First target evidence must increment total evidence once.");
Require(targetHealth.SourceEvidenceCount == 0 && targetHealth.TargetEvidenceCount == 1, "Target-role evidence accounting must be preserved.");
Require(targetHealth.LastPhase == SequenceMigrationPhase.Import, "Latest health evidence must preserve migration phase.");
Require(targetHealth.LastFailureClass == SequenceMigrationFailureClass.TimedOut, "Latest health evidence must preserve failure class.");
Require(targetHealth.LastTimeout == TimeSpan.FromMilliseconds(25), "Latest timeout evidence must preserve deadline duration.");

health.Record(new ExecutionTraceEvent(
    Guid.NewGuid(),
    ExecutionTraceKind.MigrationFailed,
    StepIndex: 0,
    Operation: "MigrateKvExecutionStep",
    Device: source,
    TargetDevice: target,
    MigrationPhase: SequenceMigrationPhase.Prepare,
    MigrationFailureClass: SequenceMigrationFailureClass.BackendFailure,
    MigrationHealthImpact: SequenceMigrationHealthImpact.SourceSuspect));

Require(health.TryGet(source, out var sourceHealth), "Source health evidence must be queryable.");
Require(sourceHealth.EvidenceCount == 1, "First source evidence must increment total evidence once.");
Require(sourceHealth.SourceEvidenceCount == 1 && sourceHealth.TargetEvidenceCount == 0, "Source-role evidence accounting must be preserved.");

health.Record(new ExecutionTraceEvent(
    Guid.NewGuid(),
    ExecutionTraceKind.MigrationRollbackFailed,
    StepIndex: 0,
    Operation: "MigrateKvExecutionStep",
    Device: source,
    TargetDevice: target,
    MigrationPhase: SequenceMigrationPhase.TargetRollback,
    MigrationFailureClass: SequenceMigrationFailureClass.BackendFailure,
    MigrationHealthImpact: SequenceMigrationHealthImpact.BothSuspect));

Require(health.TryGet(source, out sourceHealth), "Source evidence must remain available after aggregation.");
Require(health.TryGet(target, out targetHealth), "Target evidence must remain available after aggregation.");
Require(sourceHealth.EvidenceCount == 2 && sourceHealth.SourceEvidenceCount == 2, "BothSuspect must add source-role evidence to the source device.");
Require(targetHealth.EvidenceCount == 2 && targetHealth.TargetEvidenceCount == 2, "BothSuspect must add target-role evidence to the target device.");
Require(sourceHealth.LastPhase == SequenceMigrationPhase.TargetRollback, "Latest source evidence must track the terminal rollback phase.");
Require(targetHealth.LastPhase == SequenceMigrationPhase.TargetRollback, "Latest target evidence must track the terminal rollback phase.");

health.Record(new ExecutionTraceEvent(
    Guid.NewGuid(),
    ExecutionTraceKind.MigrationFailed,
    StepIndex: 0,
    Operation: "MigrateKvExecutionStep",
    Device: source,
    TargetDevice: target,
    MigrationPhase: SequenceMigrationPhase.Prepare,
    MigrationFailureClass: SequenceMigrationFailureClass.CallerCanceled,
    MigrationHealthImpact: SequenceMigrationHealthImpact.None));

Require(health.TryGet(source, out var sourceAfterCancellation) && sourceAfterCancellation.EvidenceCount == 2, "Caller cancellation must not add health evidence.");
Require(health.TryGet(target, out var targetAfterCancellation) && targetAfterCancellation.EvidenceCount == 2, "Caller cancellation must not add target health evidence.");
Require(inner.Snapshot().Count == 5, "Health consumer must forward every trace event to its downstream sink.");

var snapshot = health.Snapshot();
Require(snapshot.Count == 2, "Snapshot must contain one aggregate per suspect device.");
Require(health.SuspectDeviceCount == 2, "Suspect-device count must match retained aggregates.");

Require(health.Reset(target), "Reset must remove existing target evidence.");
Require(!health.TryGet(target, out _), "Reset target must no longer be suspect.");
Require(health.SuspectDeviceCount == 1, "Resetting one device must preserve other evidence.");

health.ResetAll();
Require(health.SuspectDeviceCount == 0, "ResetAll must clear all retained health evidence.");
Require(!health.Reset(source), "Reset after ResetAll must report that no evidence remained.");

Console.WriteLine("Fission migration health trace-consumer specs passed.");
