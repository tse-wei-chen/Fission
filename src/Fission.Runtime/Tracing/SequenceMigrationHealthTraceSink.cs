using System.Collections.Concurrent;
using Fission.Abstractions;
using Fission.Runtime.Execution;

namespace Fission.Runtime.Tracing;

/// <summary>
/// Aggregated migration-failure evidence for one execution device.
/// This is intentionally evidence, not an automatic eviction decision.
/// Consumers such as schedulers or health managers may use <see cref="IsSuspect"/>
/// as one input and explicitly reset evidence after recovery/revalidation.
/// </summary>
public readonly record struct SequenceMigrationDeviceHealth(
    DeviceId Device,
    int EvidenceCount,
    int SourceEvidenceCount,
    int TargetEvidenceCount,
    SequenceMigrationPhase LastPhase,
    SequenceMigrationFailureClass LastFailureClass,
    TimeSpan? LastTimeout)
{
    public bool IsSuspect => EvidenceCount > 0;
}

/// <summary>
/// Consumes terminal migration trace events and turns their conservative
/// <see cref="SequenceMigrationHealthImpact"/> flags into queryable per-device
/// health evidence. An optional downstream sink can persist/stream the same event,
/// allowing this consumer to be inserted without replacing existing trace storage.
///
/// The sink never quarantines, restarts, or reroutes a device by itself. That keeps
/// migration transaction correctness separate from deployment-specific health policy.
/// </summary>
public sealed class SequenceMigrationHealthTraceSink : IExecutionTraceSink
{
    private readonly ConcurrentDictionary<DeviceId, SequenceMigrationDeviceHealth> _devices = new();
    private readonly IExecutionTraceSink? _inner;

    public SequenceMigrationHealthTraceSink(IExecutionTraceSink? inner = null)
    {
        _inner = inner;
    }

    public int SuspectDeviceCount => _devices.Count;

    public void Record(ExecutionTraceEvent traceEvent)
    {
        ArgumentNullException.ThrowIfNull(traceEvent);

        ConsumeHealthEvidence(traceEvent);
        _inner?.Record(traceEvent);
    }

    public bool TryGet(DeviceId device, out SequenceMigrationDeviceHealth health) =>
        _devices.TryGetValue(device, out health);

    public IReadOnlyList<SequenceMigrationDeviceHealth> Snapshot() =>
        _devices.Values.ToArray();

    public bool Reset(DeviceId device) =>
        _devices.TryRemove(device, out _);

    public void ResetAll() =>
        _devices.Clear();

    private void ConsumeHealthEvidence(ExecutionTraceEvent traceEvent)
    {
        if (traceEvent.Kind is not (
                ExecutionTraceKind.MigrationFailed or
                ExecutionTraceKind.MigrationRolledBack or
                ExecutionTraceKind.MigrationRollbackFailed) ||
            traceEvent.MigrationHealthImpact is not { } impact ||
            impact == SequenceMigrationHealthImpact.None ||
            traceEvent.MigrationPhase is not { } phase ||
            traceEvent.MigrationFailureClass is not { } failureClass)
        {
            return;
        }

        if ((impact & SequenceMigrationHealthImpact.SourceSuspect) != 0 &&
            traceEvent.Device is { } sourceDevice)
        {
            AddEvidence(
                sourceDevice,
                sourceEvidence: true,
                phase,
                failureClass,
                traceEvent.MigrationTimeout);
        }

        if ((impact & SequenceMigrationHealthImpact.TargetSuspect) != 0 &&
            traceEvent.TargetDevice is { } targetDevice)
        {
            AddEvidence(
                targetDevice,
                sourceEvidence: false,
                phase,
                failureClass,
                traceEvent.MigrationTimeout);
        }
    }

    private void AddEvidence(
        DeviceId device,
        bool sourceEvidence,
        SequenceMigrationPhase phase,
        SequenceMigrationFailureClass failureClass,
        TimeSpan? timeout)
    {
        _devices.AddOrUpdate(
            device,
            static (id, state) => new SequenceMigrationDeviceHealth(
                id,
                EvidenceCount: 1,
                SourceEvidenceCount: state.SourceEvidence ? 1 : 0,
                TargetEvidenceCount: state.SourceEvidence ? 0 : 1,
                state.Phase,
                state.FailureClass,
                state.Timeout),
            static (_, current, state) => current with
            {
                EvidenceCount = checked(current.EvidenceCount + 1),
                SourceEvidenceCount = checked(
                    current.SourceEvidenceCount + (state.SourceEvidence ? 1 : 0)),
                TargetEvidenceCount = checked(
                    current.TargetEvidenceCount + (state.SourceEvidence ? 0 : 1)),
                LastPhase = state.Phase,
                LastFailureClass = state.FailureClass,
                LastTimeout = state.Timeout
            },
            (SourceEvidence: sourceEvidence,
             Phase: phase,
             FailureClass: failureClass,
             Timeout: timeout));
    }
}
