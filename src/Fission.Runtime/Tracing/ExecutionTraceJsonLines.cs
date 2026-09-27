using System.Text;
using System.Text.Json;
using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Tracing;

/// <summary>
/// Persists execution traces in a versioned, line-delimited JSON format that is
/// suitable for offline replay, diffing, and scheduler simulation.
/// </summary>
public static class ExecutionTraceJsonLines
{
    public const string SchemaName = "fission.execution-trace";
    public const int SchemaVersion = 1;

    private const string HeaderRecordType = "header";
    private const string EventRecordType = "event";
    private const int DefaultBufferSize = 64 * 1024;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public static async ValueTask WriteFileAsync(
        string path,
        IEnumerable<RecordedExecutionTraceEvent> recordedEvents,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(recordedEvents);

        await using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.Read,
            DefaultBufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);

        await WriteAsync(stream, recordedEvents, cancellationToken).ConfigureAwait(false);
    }

    public static async ValueTask<IReadOnlyList<RecordedExecutionTraceEvent>> ReadFileAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        await using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            DefaultBufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);

        return await ReadAsync(stream, cancellationToken).ConfigureAwait(false);
    }

    public static async ValueTask WriteAsync(
        Stream destination,
        IEnumerable<RecordedExecutionTraceEvent> recordedEvents,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(recordedEvents);

        if (!destination.CanWrite)
        {
            throw new ArgumentException("Trace destination stream must be writable.", nameof(destination));
        }

        using var writer = new StreamWriter(
            destination,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            DefaultBufferSize,
            leaveOpen: true);

        var header = new TraceHeaderRecord(
            HeaderRecordType,
            SchemaName,
            SchemaVersion);
        await WriteRecordAsync(writer, header, cancellationToken).ConfigureAwait(false);

        long previousOrdinal = 0;
        foreach (var recorded in recordedEvents)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(recorded);

            if (recorded.Ordinal <= previousOrdinal)
            {
                throw new InvalidDataException(
                    $"Trace ordinals must be strictly increasing; got {recorded.Ordinal} after {previousOrdinal}.");
            }

            previousOrdinal = recorded.Ordinal;
            var traceEvent = recorded.Event;
            ArgumentNullException.ThrowIfNull(traceEvent);

            var persisted = new TraceEventRecord(
                EventRecordType,
                recorded.Ordinal,
                traceEvent.PlanId,
                traceEvent.Kind.ToString(),
                traceEvent.StepIndex,
                traceEvent.Operation,
                traceEvent.SequenceId?.Value,
                traceEvent.RelatedSequenceId?.Value,
                traceEvent.SnapshotId?.Value,
                traceEvent.Position,
                traceEvent.KvPageCount,
                traceEvent.Device?.Value,
                traceEvent.TargetDevice?.Value,
                traceEvent.TransactionId,
                traceEvent.TransportId,
                traceEvent.TransportKind?.ToString(),
                traceEvent.TransferBytes,
                traceEvent.EstimatedDuration?.Ticks,
                traceEvent.Elapsed?.Ticks,
                traceEvent.FailureType,
                traceEvent.RollbackFailureCount);

            await WriteRecordAsync(writer, persisted, cancellationToken).ConfigureAwait(false);
        }

        await writer.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    public static async ValueTask<IReadOnlyList<RecordedExecutionTraceEvent>> ReadAsync(
        Stream source,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);

        if (!source.CanRead)
        {
            throw new ArgumentException("Trace source stream must be readable.", nameof(source));
        }

        using var reader = new StreamReader(
            source,
            Encoding.UTF8,
            detectEncodingFromByteOrderMarks: true,
            DefaultBufferSize,
            leaveOpen: true);

        var headerLine = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
        if (headerLine is null)
        {
            throw new InvalidDataException("Execution trace is empty; expected a schema header.");
        }

        TraceHeaderRecord header;
        try
        {
            header = JsonSerializer.Deserialize<TraceHeaderRecord>(headerLine, JsonOptions)
                ?? throw new InvalidDataException("Execution trace header is null.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("Execution trace header is not valid JSON.", exception);
        }

        ValidateHeader(header);

        var result = new List<RecordedExecutionTraceEvent>();
        long previousOrdinal = 0;
        var lineNumber = 1;

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
            if (line is null)
            {
                break;
            }

            lineNumber++;
            if (string.IsNullOrWhiteSpace(line))
            {
                throw new InvalidDataException(
                    $"Execution trace line {lineNumber} is empty.");
            }

            TraceEventRecord persisted;
            try
            {
                persisted = JsonSerializer.Deserialize<TraceEventRecord>(line, JsonOptions)
                    ?? throw new InvalidDataException(
                        $"Execution trace line {lineNumber} deserialized to null.");
            }
            catch (JsonException exception)
            {
                throw new InvalidDataException(
                    $"Execution trace line {lineNumber} is not valid JSON.",
                    exception);
            }

            if (!string.Equals(persisted.RecordType, EventRecordType, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Execution trace line {lineNumber} has record type '{persisted.RecordType}', expected '{EventRecordType}'.");
            }

            if (persisted.Ordinal <= previousOrdinal)
            {
                throw new InvalidDataException(
                    $"Trace ordinals must be strictly increasing; line {lineNumber} has {persisted.Ordinal} after {previousOrdinal}.");
            }

            previousOrdinal = persisted.Ordinal;
            result.Add(new RecordedExecutionTraceEvent(
                persisted.Ordinal,
                ToRuntimeEvent(persisted, lineNumber)));
        }

        return result;
    }

    private static async ValueTask WriteRecordAsync<T>(
        StreamWriter writer,
        T value,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(value, JsonOptions);
        await writer.WriteLineAsync(json.AsMemory(), cancellationToken).ConfigureAwait(false);
    }

    private static void ValidateHeader(TraceHeaderRecord header)
    {
        if (!string.Equals(header.RecordType, HeaderRecordType, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Execution trace first record has type '{header.RecordType}', expected '{HeaderRecordType}'.");
        }

        if (!string.Equals(header.Schema, SchemaName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Unsupported execution trace schema '{header.Schema}'.");
        }

        if (header.Version != SchemaVersion)
        {
            throw new InvalidDataException(
                $"Unsupported execution trace schema version {header.Version}; expected {SchemaVersion}.");
        }
    }

    private static ExecutionTraceEvent ToRuntimeEvent(
        TraceEventRecord persisted,
        int lineNumber)
    {
        if (!Enum.TryParse<ExecutionTraceKind>(
                persisted.Kind,
                ignoreCase: false,
                out var kind) ||
            !Enum.IsDefined(kind))
        {
            throw new InvalidDataException(
                $"Execution trace line {lineNumber} has unknown event kind '{persisted.Kind}'.");
        }

        SequenceMigrationTransportKind? transportKind = null;
        if (persisted.TransportKind is not null)
        {
            if (!Enum.TryParse<SequenceMigrationTransportKind>(
                    persisted.TransportKind,
                    ignoreCase: false,
                    out var parsedTransportKind) ||
                !Enum.IsDefined(parsedTransportKind))
            {
                throw new InvalidDataException(
                    $"Execution trace line {lineNumber} has unknown transport kind '{persisted.TransportKind}'.");
            }

            transportKind = parsedTransportKind;
        }

        if (string.IsNullOrWhiteSpace(persisted.Operation))
        {
            throw new InvalidDataException(
                $"Execution trace line {lineNumber} has an empty operation name.");
        }

        return new ExecutionTraceEvent(
            persisted.PlanId,
            kind,
            persisted.StepIndex,
            persisted.Operation,
            persisted.SequenceId is { } sequenceId ? new SequenceId(sequenceId) : null,
            persisted.RelatedSequenceId is { } relatedSequenceId
                ? new SequenceId(relatedSequenceId)
                : null,
            persisted.SnapshotId is { } snapshotId ? new KvSnapshotId(snapshotId) : null,
            persisted.Position,
            persisted.KvPageCount,
            persisted.Device is { } device ? new DeviceId(device) : null,
            persisted.TargetDevice is { } targetDevice ? new DeviceId(targetDevice) : null,
            persisted.TransactionId,
            persisted.TransportId,
            transportKind,
            persisted.TransferBytes,
            persisted.EstimatedDurationTicks is { } estimatedDurationTicks
                ? TimeSpan.FromTicks(estimatedDurationTicks)
                : null,
            persisted.ElapsedTicks is { } elapsedTicks
                ? TimeSpan.FromTicks(elapsedTicks)
                : null,
            persisted.FailureType,
            persisted.RollbackFailureCount);
    }

    private sealed record TraceHeaderRecord(
        string RecordType,
        string Schema,
        int Version);

    private sealed record TraceEventRecord(
        string RecordType,
        long Ordinal,
        Guid PlanId,
        string Kind,
        int StepIndex,
        string Operation,
        Guid? SequenceId,
        Guid? RelatedSequenceId,
        Guid? SnapshotId,
        int? Position,
        int? KvPageCount,
        string? Device,
        string? TargetDevice,
        Guid? TransactionId,
        string? TransportId,
        string? TransportKind,
        long? TransferBytes,
        long? EstimatedDurationTicks,
        long? ElapsedTicks,
        string? FailureType,
        int? RollbackFailureCount);
}
