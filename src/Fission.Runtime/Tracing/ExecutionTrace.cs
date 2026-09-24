using System.Collections.Concurrent;
using Fission.Abstractions;

namespace Fission.Runtime.Tracing;

public enum ExecutionTraceKind
{
    PlanStarted,
    StepStarted,
    StepCompleted,
    SequenceForked,
    SnapshotCreated,
    PlanCompleted
}

public sealed record ExecutionTraceEvent(
    Guid PlanId,
    ExecutionTraceKind Kind,
    int StepIndex,
    string Operation,
    SequenceId? SequenceId = null,
    SequenceId? RelatedSequenceId = null,
    KvSnapshotId? SnapshotId = null,
    int? Position = null,
    int? KvPageCount = null,
    DeviceId? Device = null);

public sealed record RecordedExecutionTraceEvent(
    long Ordinal,
    ExecutionTraceEvent Event);

public interface IExecutionTraceSink
{
    void Record(ExecutionTraceEvent traceEvent);
}

public sealed class InMemoryExecutionTraceSink : IExecutionTraceSink
{
    private readonly ConcurrentQueue<RecordedExecutionTraceEvent> _events = new();
    private long _ordinal;

    public void Record(ExecutionTraceEvent traceEvent)
    {
        ArgumentNullException.ThrowIfNull(traceEvent);
        var ordinal = Interlocked.Increment(ref _ordinal);
        _events.Enqueue(new RecordedExecutionTraceEvent(ordinal, traceEvent));
    }

    public IReadOnlyList<RecordedExecutionTraceEvent> Snapshot() =>
        _events.OrderBy(static item => item.Ordinal).ToArray();
}

public sealed record ReplaySequenceState(
    SequenceId SequenceId,
    SequenceId? ParentSequenceId,
    int Position,
    int KvPageCount,
    DeviceId Device);

public sealed record ExecutionReplayResult(
    IReadOnlyDictionary<SequenceId, ReplaySequenceState> Sequences,
    IReadOnlySet<KvSnapshotId> Snapshots);

/// <summary>
/// Reconstructs sequence metadata from trace events without invoking a model backend.
/// This is the seed of Fission's deterministic time-travel/scheduler simulation path.
/// </summary>
public static class ExecutionTraceReplay
{
    public static ExecutionReplayResult Replay(IEnumerable<RecordedExecutionTraceEvent> recordedEvents)
    {
        ArgumentNullException.ThrowIfNull(recordedEvents);

        var sequences = new Dictionary<SequenceId, ReplaySequenceState>();
        var snapshots = new HashSet<KvSnapshotId>();
        long previousOrdinal = 0;

        foreach (var recorded in recordedEvents.OrderBy(static item => item.Ordinal))
        {
            if (recorded.Ordinal <= previousOrdinal)
            {
                throw new InvalidOperationException("Trace ordinals must be strictly increasing.");
            }

            previousOrdinal = recorded.Ordinal;
            var traceEvent = recorded.Event;

            if (traceEvent.Kind == ExecutionTraceKind.SequenceForked)
            {
                if (traceEvent.SequenceId is not { } childId ||
                    traceEvent.RelatedSequenceId is not { } parentId ||
                    traceEvent.Position is not { } position ||
                    traceEvent.KvPageCount is not { } pageCount ||
                    traceEvent.Device is not { } device)
                {
                    throw new InvalidOperationException("Fork trace event is missing required state.");
                }

                sequences[childId] = new ReplaySequenceState(
                    childId,
                    parentId,
                    position,
                    pageCount,
                    device);
                continue;
            }

            if (traceEvent.Kind == ExecutionTraceKind.SnapshotCreated &&
                traceEvent.SnapshotId is { } snapshotId)
            {
                snapshots.Add(snapshotId);
            }

            if (traceEvent.Kind != ExecutionTraceKind.StepCompleted ||
                traceEvent.SequenceId is not { } sequenceId ||
                traceEvent.Position is not { } completedPosition ||
                traceEvent.KvPageCount is not { } completedPageCount ||
                traceEvent.Device is not { } completedDevice)
            {
                continue;
            }

            sequences.TryGetValue(sequenceId, out var previous);
            sequences[sequenceId] = new ReplaySequenceState(
                sequenceId,
                previous?.ParentSequenceId,
                completedPosition,
                completedPageCount,
                completedDevice);
        }

        return new ExecutionReplayResult(sequences, snapshots);
    }
}
