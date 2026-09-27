using System.Text;
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

static async Task RequireInvalidDataAsync(Func<Task> action, string scenario)
{
    try
    {
        await action();
    }
    catch (InvalidDataException)
    {
        return;
    }

    throw new InvalidOperationException($"Expected InvalidDataException for {scenario}.");
}

static IReadOnlyList<RecordedExecutionTraceEvent> CreateTrace()
{
    var planId = Guid.NewGuid();
    var sequenceId = SequenceId.New();
    var snapshotId = KvSnapshotId.New();
    var source = new DeviceId("gpu:0");
    var target = new DeviceId("gpu:1");
    var transactionId = Guid.NewGuid();

    return new RecordedExecutionTraceEvent[]
    {
        new(
            1,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.PlanStarted,
                -1,
                "Plan")),
        new(
            2,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.StepCompleted,
                0,
                "PrefillExecutionStep",
                sequenceId,
                Position: 64,
                KvPageCount: 4,
                Device: source)),
        new(
            3,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.SnapshotCreated,
                1,
                "SnapshotKvExecutionStep",
                sequenceId,
                SnapshotId: snapshotId,
                Position: 64,
                KvPageCount: 4,
                Device: source)),
        new(
            4,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.MigrationStarted,
                2,
                "MigrateKvExecutionStep",
                sequenceId,
                Position: 64,
                KvPageCount: 4,
                Device: source,
                TargetDevice: target)),
        new(
            5,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.MigrationPlanned,
                2,
                "MigrateKvExecutionStep",
                sequenceId,
                Position: 64,
                KvPageCount: 4,
                Device: source,
                TargetDevice: target,
                TransportId: "host-staging-v1",
                TransportKind: SequenceMigrationTransportKind.HostStaging,
                TransferBytes: 262_144,
                EstimatedDuration: TimeSpan.FromTicks(12_345))),
        new(
            6,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.MigrationCommitted,
                2,
                "MigrateKvExecutionStep",
                sequenceId,
                Position: 64,
                KvPageCount: 4,
                Device: source,
                TargetDevice: target,
                TransactionId: transactionId,
                TransportId: "host-staging-v1",
                TransportKind: SequenceMigrationTransportKind.HostStaging,
                TransferBytes: 262_144,
                EstimatedDuration: TimeSpan.FromTicks(12_345),
                Elapsed: TimeSpan.FromTicks(54_321))),
        new(
            7,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.StepCompleted,
                2,
                "MigrateKvExecutionStep",
                sequenceId,
                Position: 64,
                KvPageCount: 4,
                Device: target)),
        new(
            8,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.MigrationRollbackFailed,
                3,
                "MigrateKvExecutionStep",
                sequenceId,
                Position: 64,
                KvPageCount: 4,
                Device: target,
                TargetDevice: source,
                TransactionId: Guid.NewGuid(),
                TransportId: "host-staging-v1",
                TransportKind: SequenceMigrationTransportKind.HostStaging,
                TransferBytes: 262_144,
                EstimatedDuration: TimeSpan.FromTicks(11_111),
                Elapsed: TimeSpan.FromTicks(22_222),
                FailureType: typeof(InvalidOperationException).FullName,
                RollbackFailureCount: 2)),
        new(
            9,
            new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.PlanCompleted,
                4,
                "Plan"))
    };
}

static async Task RunMemoryRoundTripAsync()
{
    var expected = CreateTrace();
    await using var stream = new MemoryStream();

    await ExecutionTraceJsonLines.WriteAsync(stream, expected);
    Require(stream.CanWrite, "WriteAsync must leave caller-owned streams open.");

    var jsonl = Encoding.UTF8.GetString(stream.ToArray());
    var lines = jsonl.Split('\n', StringSplitOptions.RemoveEmptyEntries);
    Require(lines.Length == expected.Count + 1, "JSONL must contain one header plus one line per trace event.");
    Require(lines[0].Contains("\"schema\":\"fission.execution-trace\"", StringComparison.Ordinal), "Trace header must identify the schema.");
    Require(lines[0].Contains("\"version\":1", StringComparison.Ordinal), "Trace header must identify schema version 1.");
    Require(jsonl.Contains("\"transportId\":\"host-staging-v1\"", StringComparison.Ordinal), "Migration transport metadata must be persisted.");
    Require(jsonl.Contains("\"rollbackFailureCount\":2", StringComparison.Ordinal), "Rollback failure metadata must be persisted.");

    stream.Position = 0;
    var actual = await ExecutionTraceJsonLines.ReadAsync(stream);
    Require(stream.CanRead, "ReadAsync must leave caller-owned streams open.");
    Require(expected.SequenceEqual(actual), "Trace events must round-trip without losing runtime metadata.");

    var replay = ExecutionTraceReplay.Replay(actual);
    var sequenceId = expected[1].Event.SequenceId!.Value;
    var snapshotId = expected[2].Event.SnapshotId!.Value;
    Require(replay.Sequences[sequenceId].Device == new DeviceId("gpu:1"), "Persisted trace must replay the committed target placement.");
    Require(replay.Sequences[sequenceId].Position == 64, "Persisted trace must replay sequence position.");
    Require(replay.Snapshots.Contains(snapshotId), "Persisted trace must replay snapshot identity.");
}

static async Task RunFileRoundTripAsync()
{
    var expected = CreateTrace();
    var path = Path.Combine(
        Path.GetTempPath(),
        $"fission-{Guid.NewGuid():N}.trace");

    try
    {
        await ExecutionTraceJsonLines.WriteFileAsync(path, expected);
        Require(File.Exists(path), "WriteFileAsync must create the trace file.");
        Require(new FileInfo(path).Length > 0, "Persisted trace file must not be empty.");

        var actual = await ExecutionTraceJsonLines.ReadFileAsync(path);
        Require(expected.SequenceEqual(actual), "File trace must round-trip without metadata loss.");
    }
    finally
    {
        File.Delete(path);
    }
}

static async Task RunValidationAsync()
{
    const string badVersion =
        "{\"recordType\":\"header\",\"schema\":\"fission.execution-trace\",\"version\":99}\n";
    await RequireInvalidDataAsync(
        async () =>
        {
            await using var stream = new MemoryStream(Encoding.UTF8.GetBytes(badVersion));
            _ = await ExecutionTraceJsonLines.ReadAsync(stream);
        },
        "unsupported schema version");

    const string unknownKind =
        "{\"recordType\":\"header\",\"schema\":\"fission.execution-trace\",\"version\":1}\n" +
        "{\"recordType\":\"event\",\"ordinal\":1,\"planId\":\"00000000-0000-0000-0000-000000000001\",\"kind\":\"FutureUnknownKind\",\"stepIndex\":0,\"operation\":\"FutureStep\"}\n";
    await RequireInvalidDataAsync(
        async () =>
        {
            await using var stream = new MemoryStream(Encoding.UTF8.GetBytes(unknownKind));
            _ = await ExecutionTraceJsonLines.ReadAsync(stream);
        },
        "unknown event kind");

    var duplicatedOrdinal = CreateTrace().Take(2).ToArray();
    duplicatedOrdinal[1] = duplicatedOrdinal[1] with { Ordinal = duplicatedOrdinal[0].Ordinal };
    await RequireInvalidDataAsync(
        async () =>
        {
            await using var stream = new MemoryStream();
            await ExecutionTraceJsonLines.WriteAsync(stream, duplicatedOrdinal);
        },
        "non-increasing ordinals");
}

await RunMemoryRoundTripAsync();
await RunFileRoundTripAsync();
await RunValidationAsync();

Console.WriteLine("Fission trace persistence specs passed.");
