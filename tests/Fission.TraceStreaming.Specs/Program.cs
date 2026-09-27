using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Tracing;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static async Task WaitUntilAsync(
    Func<bool> predicate,
    TimeSpan timeout,
    string scenario)
{
    var deadline = DateTime.UtcNow + timeout;
    while (!predicate())
    {
        if (DateTime.UtcNow >= deadline)
        {
            throw new TimeoutException($"Timed out waiting for {scenario}.");
        }

        await Task.Delay(10);
    }
}

static async Task<IReadOnlyList<RecordedExecutionTraceEvent>> ReadSegmentsAsync(
    IEnumerable<string> segmentFiles)
{
    var result = new List<RecordedExecutionTraceEvent>();
    foreach (var path in segmentFiles)
    {
        result.AddRange(await ExecutionTraceJsonLines.ReadFileAsync(path));
    }

    return result;
}

static ExecutionTraceEvent[] CreateTraceEvents(
    out SequenceId sequenceId,
    out DeviceId targetDevice)
{
    var planId = Guid.NewGuid();
    sequenceId = SequenceId.New();
    var sourceDevice = new DeviceId("gpu:0");
    targetDevice = new DeviceId("gpu:1");

    return
    [
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.PlanStarted,
            -1,
            "Plan"),
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.StepCompleted,
            0,
            "PrefillExecutionStep",
            sequenceId,
            Position: 32,
            KvPageCount: 2,
            Device: sourceDevice),
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.MigrationStarted,
            1,
            "MigrateKvExecutionStep",
            sequenceId,
            Position: 32,
            KvPageCount: 2,
            Device: sourceDevice,
            TargetDevice: targetDevice),
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.MigrationPlanned,
            1,
            "MigrateKvExecutionStep",
            sequenceId,
            Position: 32,
            KvPageCount: 2,
            Device: sourceDevice,
            TargetDevice: targetDevice,
            TransportId: "host-staging-v1",
            TransportKind: SequenceMigrationTransportKind.HostStaging,
            TransferBytes: 131_072,
            EstimatedDuration: TimeSpan.FromTicks(4_000)),
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.MigrationCommitted,
            1,
            "MigrateKvExecutionStep",
            sequenceId,
            Position: 32,
            KvPageCount: 2,
            Device: sourceDevice,
            TargetDevice: targetDevice,
            TransactionId: Guid.NewGuid(),
            TransportId: "host-staging-v1",
            TransportKind: SequenceMigrationTransportKind.HostStaging,
            TransferBytes: 131_072,
            EstimatedDuration: TimeSpan.FromTicks(4_000),
            Elapsed: TimeSpan.FromTicks(7_500)),
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.StepCompleted,
            1,
            "MigrateKvExecutionStep",
            sequenceId,
            Position: 32,
            KvPageCount: 2,
            Device: targetDevice),
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.MigrationRollbackFailed,
            2,
            "MigrateKvExecutionStep",
            sequenceId,
            Position: 32,
            KvPageCount: 2,
            Device: targetDevice,
            TargetDevice: sourceDevice,
            TransactionId: Guid.NewGuid(),
            TransportId: "host-staging-v1",
            TransportKind: SequenceMigrationTransportKind.HostStaging,
            TransferBytes: 131_072,
            Elapsed: TimeSpan.FromTicks(9_000),
            FailureType: typeof(InvalidOperationException).FullName,
            RollbackFailureCount: 1),
        new ExecutionTraceEvent(
            planId,
            ExecutionTraceKind.PlanCompleted,
            3,
            "Plan")
    ];
}

static async Task RunRotationAndReplayAsync()
{
    var directory = Path.Combine(
        Path.GetTempPath(),
        $"fission-streaming-trace-{Guid.NewGuid():N}");

    try
    {
        var options = new StreamingExecutionTraceOptions
        {
            Capacity = 64,
            MaxEventsPerSegment = 3,
            FlushInterval = TimeSpan.FromHours(1),
            DurabilityPolicy = ExecutionTraceDurabilityPolicy.FlushToDisk
        };

        var sink = new StreamingExecutionTraceSink(directory, options);
        var traceEvents = CreateTraceEvents(out var sequenceId, out var targetDevice);
        foreach (var traceEvent in traceEvents)
        {
            sink.Record(traceEvent);
        }

        await sink.DisposeAsync();

        Require(sink.BackgroundFailure is null, "Successful streaming persistence must not report a background failure.");
        Require(sink.DroppedEventCount == 0, "Rotation scenario must not drop events.");
        Require(sink.PersistedEventCount == traceEvents.Length, "All trace events must be persisted.");
        Require(sink.LastPersistedOrdinal == traceEvents.Length, "Last persisted ordinal must match the event count.");
        Require(sink.SegmentFiles.Count == 3, "Eight events with a three-event segment limit must produce three files.");
        Require(sink.SegmentFiles.All(File.Exists), "Published segment paths must point at completed files.");
        Require(!Directory.EnumerateFiles(directory).Any(static path => path.Contains(".tmp-", StringComparison.Ordinal)), "Completed streaming persistence must not leave temporary files.");

        var recorded = await ReadSegmentsAsync(sink.SegmentFiles);
        Require(recorded.Count == traceEvents.Length, "Reading all segments must restore every trace event.");
        Require(recorded.Select(static item => item.Ordinal).SequenceEqual(Enumerable.Range(1, traceEvents.Length).Select(static value => (long)value)), "Segment ordinals must remain globally monotonic.");
        Require(recorded.Select(static item => item.Event).SequenceEqual(traceEvents), "Segment persistence must preserve complete event metadata.");

        var replay = ExecutionTraceReplay.Replay(recorded);
        Require(replay.Sequences[sequenceId].Device == targetDevice, "Streaming trace segments must replay the committed migration target.");
        Require(replay.Sequences[sequenceId].Position == 32, "Streaming trace replay must preserve sequence position.");
    }
    finally
    {
        Directory.Delete(directory, recursive: true);
    }
}

static async Task RunPeriodicFlushAsync()
{
    var directory = Path.Combine(
        Path.GetTempPath(),
        $"fission-periodic-trace-{Guid.NewGuid():N}");

    try
    {
        var sink = new StreamingExecutionTraceSink(
            directory,
            new StreamingExecutionTraceOptions
            {
                Capacity = 16,
                MaxEventsPerSegment = 128,
                FlushInterval = TimeSpan.FromMilliseconds(40)
            });

        sink.Record(new ExecutionTraceEvent(
            Guid.NewGuid(),
            ExecutionTraceKind.PlanStarted,
            -1,
            "Plan"));

        await WaitUntilAsync(
            () => sink.PersistedEventCount == 1,
            TimeSpan.FromSeconds(3),
            "periodic trace flush");

        Require(sink.SegmentFiles.Count == 1, "Low-volume traffic must become durable without filling a segment.");
        var persisted = await ExecutionTraceJsonLines.ReadFileAsync(sink.SegmentFiles[0]);
        Require(persisted.Count == 1, "Periodic flush segment must contain the queued event.");

        await sink.DisposeAsync();
    }
    finally
    {
        Directory.Delete(directory, recursive: true);
    }
}

static async Task RunConcurrentRecordAsync()
{
    var directory = Path.Combine(
        Path.GetTempPath(),
        $"fission-concurrent-trace-{Guid.NewGuid():N}");

    try
    {
        var sink = new StreamingExecutionTraceSink(
            directory,
            new StreamingExecutionTraceOptions
            {
                Capacity = 1_024,
                MaxEventsPerSegment = 97,
                FlushInterval = TimeSpan.FromHours(1)
            });

        var planId = Guid.NewGuid();
        Parallel.For(
            0,
            512,
            index => sink.Record(new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.StepStarted,
                index,
                $"ConcurrentStep:{index}")));

        await sink.DisposeAsync();

        Require(sink.DroppedEventCount == 0, "Adequately sized concurrent trace buffering must not drop events.");
        var recorded = await ReadSegmentsAsync(sink.SegmentFiles);
        Require(recorded.Count == 512, "Concurrent recording must persist every event.");
        Require(recorded.Select(static item => item.Ordinal).SequenceEqual(Enumerable.Range(1, 512).Select(static value => (long)value)), "Concurrent Record calls must still produce a globally ordered trace.");
        Require(recorded.Select(static item => item.Event.Operation).Distinct(StringComparer.Ordinal).Count() == 512, "Concurrent trace events must not be duplicated or lost.");
    }
    finally
    {
        Directory.Delete(directory, recursive: true);
    }
}

static async Task RunBackgroundFailureIsolationAsync()
{
    var directory = Path.Combine(
        Path.GetTempPath(),
        $"fission-failed-trace-{Guid.NewGuid():N}");
    var sink = new StreamingExecutionTraceSink(
        directory,
        new StreamingExecutionTraceOptions
        {
            Capacity = 8,
            MaxEventsPerSegment = 1,
            FlushInterval = TimeSpan.FromHours(1),
            BackgroundFailurePolicy = ExecutionTraceBackgroundFailurePolicy.Isolate
        });

    Directory.Delete(directory, recursive: true);
    File.WriteAllText(directory, "blocks trace directory recreation");

    try
    {
        sink.Record(new ExecutionTraceEvent(
            Guid.NewGuid(),
            ExecutionTraceKind.PlanStarted,
            -1,
            "Plan"));

        try
        {
            await sink.Completion;
            throw new InvalidOperationException("Expected the background writer to fail when the trace directory becomes a file.");
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }

        Require(sink.BackgroundFailure is not null, "Background persistence failure must be observable.");

        sink.Record(new ExecutionTraceEvent(
            Guid.NewGuid(),
            ExecutionTraceKind.PlanStarted,
            -1,
            "PlanAfterFailure"));
        Require(sink.DroppedEventCount >= 2, "Isolated background failure must account for abandoned and later trace events.");

        try
        {
            await sink.DisposeAsync();
            throw new InvalidOperationException("DisposeAsync must surface the background persistence failure.");
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
    finally
    {
        File.Delete(directory);
    }
}

await RunRotationAndReplayAsync();
await RunPeriodicFlushAsync();
await RunConcurrentRecordAsync();
await RunBackgroundFailureIsolationAsync();

Console.WriteLine("Fission streaming execution trace specs passed.");
