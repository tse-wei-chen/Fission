using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;

namespace Fission.Runtime.Tracing;

public enum ExecutionTraceBufferOverflowPolicy
{
    DropNewest,
    Throw
}

public enum ExecutionTraceBackgroundFailurePolicy
{
    Isolate,
    ThrowOnRecord
}

public enum ExecutionTraceDurabilityPolicy
{
    SegmentFlush,
    FlushToDisk
}

public sealed record StreamingExecutionTraceOptions
{
    public int Capacity { get; init; } = 4_096;
    public int MaxEventsPerSegment { get; init; } = 1_024;
    public TimeSpan FlushInterval { get; init; } = TimeSpan.FromSeconds(1);
    public string FilePrefix { get; init; } = "fission-trace";
    public ExecutionTraceBufferOverflowPolicy OverflowPolicy { get; init; } =
        ExecutionTraceBufferOverflowPolicy.DropNewest;
    public ExecutionTraceBackgroundFailurePolicy BackgroundFailurePolicy { get; init; } =
        ExecutionTraceBackgroundFailurePolicy.Isolate;
    public ExecutionTraceDurabilityPolicy DurabilityPolicy { get; init; } =
        ExecutionTraceDurabilityPolicy.SegmentFlush;
}

/// <summary>
/// Streams execution-trace events to versioned JSONL segment files without
/// performing file I/O on the caller's Record path. Events are assigned a
/// process-local monotonic ordinal, offered to a bounded channel with TryWrite,
/// and persisted by one background writer.
/// </summary>
public sealed class StreamingExecutionTraceSink : IExecutionTraceSink, IAsyncDisposable
{
    private const int DurableFlushBufferSize = 4 * 1024;

    private readonly object _recordGate = new();
    private readonly string _directory;
    private readonly StreamingExecutionTraceOptions _options;
    private readonly Channel<RecordedExecutionTraceEvent> _channel;
    private readonly ConcurrentQueue<string> _segmentFiles = new();
    private readonly Task _pumpTask;
    private long _nextOrdinal;
    private long _droppedEventCount;
    private long _persistedEventCount;
    private long _lastPersistedOrdinal;
    private int _segmentIndex;
    private int _disposeRequested;
    private Exception? _backgroundFailure;

    public StreamingExecutionTraceSink(
        string directory,
        StreamingExecutionTraceOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        _options = options ?? new StreamingExecutionTraceOptions();
        ValidateOptions(_options);

        _directory = Path.GetFullPath(directory);
        Directory.CreateDirectory(_directory);

        SessionId = Guid.NewGuid();
        _channel = Channel.CreateBounded<RecordedExecutionTraceEvent>(
            new BoundedChannelOptions(_options.Capacity)
            {
                SingleReader = true,
                SingleWriter = false,
                FullMode = BoundedChannelFullMode.Wait,
                AllowSynchronousContinuations = false
            });

        _pumpTask = Task.Run(PumpAsync);
    }

    public Guid SessionId { get; }
    public long DroppedEventCount => Interlocked.Read(ref _droppedEventCount);
    public long PersistedEventCount => Interlocked.Read(ref _persistedEventCount);
    public long LastPersistedOrdinal => Interlocked.Read(ref _lastPersistedOrdinal);
    public Exception? BackgroundFailure => Volatile.Read(ref _backgroundFailure);
    public Task Completion => _pumpTask;

    public IReadOnlyList<string> SegmentFiles => _segmentFiles.ToArray();

    public void Record(ExecutionTraceEvent traceEvent)
    {
        ArgumentNullException.ThrowIfNull(traceEvent);

        lock (_recordGate)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposeRequested) != 0,
                this);

            var ordinal = checked(++_nextOrdinal);
            var backgroundFailure = Volatile.Read(ref _backgroundFailure);
            if (backgroundFailure is not null)
            {
                Interlocked.Increment(ref _droppedEventCount);
                if (_options.BackgroundFailurePolicy ==
                    ExecutionTraceBackgroundFailurePolicy.ThrowOnRecord)
                {
                    throw new InvalidOperationException(
                        "Execution trace background writer has failed.",
                        backgroundFailure);
                }

                return;
            }

            if (_channel.Writer.TryWrite(
                    new RecordedExecutionTraceEvent(ordinal, traceEvent)))
            {
                return;
            }

            Interlocked.Increment(ref _droppedEventCount);
            if (_options.OverflowPolicy == ExecutionTraceBufferOverflowPolicy.Throw)
            {
                throw new InvalidOperationException(
                    $"Execution trace buffer capacity {_options.Capacity} was exceeded.");
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        lock (_recordGate)
        {
            if (Interlocked.Exchange(ref _disposeRequested, 1) == 0)
            {
                _channel.Writer.TryComplete();
            }
        }

        await _pumpTask.ConfigureAwait(false);
    }

    private async Task PumpAsync()
    {
        var batch = new List<RecordedExecutionTraceEvent>(_options.MaxEventsPerSegment);
        long batchStartedAt = 0;

        try
        {
            while (true)
            {
                if (batch.Count == 0)
                {
                    if (!await _channel.Reader.WaitToReadAsync().ConfigureAwait(false))
                    {
                        break;
                    }

                    batchStartedAt = Stopwatch.GetTimestamp();
                }

                while (batch.Count < _options.MaxEventsPerSegment &&
                       _channel.Reader.TryRead(out var recorded))
                {
                    batch.Add(recorded);
                }

                if (batch.Count == 0)
                {
                    continue;
                }

                if (batch.Count >= _options.MaxEventsPerSegment)
                {
                    await PersistSegmentAsync(batch).ConfigureAwait(false);
                    batch.Clear();
                    continue;
                }

                if (_channel.Reader.Completion.IsCompleted)
                {
                    await PersistSegmentAsync(batch).ConfigureAwait(false);
                    batch.Clear();
                    break;
                }

                var elapsed = Stopwatch.GetElapsedTime(batchStartedAt);
                var remaining = _options.FlushInterval - elapsed;
                if (remaining <= TimeSpan.Zero)
                {
                    await PersistSegmentAsync(batch).ConfigureAwait(false);
                    batch.Clear();
                    continue;
                }

                var waitForData = _channel.Reader.WaitToReadAsync().AsTask();
                var flushDelay = Task.Delay(remaining);
                var completed = await Task.WhenAny(waitForData, flushDelay)
                    .ConfigureAwait(false);

                if (completed == flushDelay)
                {
                    await PersistSegmentAsync(batch).ConfigureAwait(false);
                    batch.Clear();
                    continue;
                }

                if (!await waitForData.ConfigureAwait(false))
                {
                    await PersistSegmentAsync(batch).ConfigureAwait(false);
                    batch.Clear();
                    break;
                }
            }
        }
        catch (Exception failure)
        {
            Volatile.Write(ref _backgroundFailure, failure);

            var abandoned = batch.Count;
            while (_channel.Reader.TryRead(out _))
            {
                abandoned++;
            }

            if (abandoned != 0)
            {
                Interlocked.Add(ref _droppedEventCount, abandoned);
            }

            _channel.Writer.TryComplete(failure);
            throw;
        }
    }

    private async ValueTask PersistSegmentAsync(
        IReadOnlyList<RecordedExecutionTraceEvent> batch)
    {
        if (batch.Count == 0)
        {
            return;
        }

        var firstOrdinal = batch[0].Ordinal;
        var lastOrdinal = batch[^1].Ordinal;
        var previousOrdinal = Interlocked.Read(ref _lastPersistedOrdinal);
        if (previousOrdinal != 0 && firstOrdinal <= previousOrdinal)
        {
            throw new InvalidDataException(
                $"Streaming trace segment begins at ordinal {firstOrdinal} after already persisting {previousOrdinal}.");
        }

        var nextSegmentIndex = checked(_segmentIndex + 1);
        var fileName =
            $"{_options.FilePrefix}-{SessionId:N}-{nextSegmentIndex:D6}-{firstOrdinal:D20}-{lastOrdinal:D20}.trace";
        var finalPath = Path.Combine(_directory, fileName);
        var tempPath = finalPath + $".tmp-{Guid.NewGuid():N}";

        try
        {
            await ExecutionTraceJsonLines.WriteFileAsync(tempPath, batch)
                .ConfigureAwait(false);

            if (_options.DurabilityPolicy == ExecutionTraceDurabilityPolicy.FlushToDisk)
            {
                using var stream = new FileStream(
                    tempPath,
                    FileMode.Open,
                    FileAccess.ReadWrite,
                    FileShare.Read,
                    DurableFlushBufferSize,
                    FileOptions.SequentialScan);
                stream.Flush(flushToDisk: true);
            }

            File.Move(tempPath, finalPath, overwrite: false);

            _segmentIndex = nextSegmentIndex;
            _segmentFiles.Enqueue(finalPath);
            Interlocked.Add(ref _persistedEventCount, batch.Count);
            Interlocked.Exchange(ref _lastPersistedOrdinal, lastOrdinal);
        }
        catch
        {
            try
            {
                File.Delete(tempPath);
            }
            catch
            {
                // Preserve the original persistence failure.
            }

            throw;
        }
    }

    private static void ValidateOptions(StreamingExecutionTraceOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxEventsPerSegment);

        if (options.FlushInterval <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options.FlushInterval),
                options.FlushInterval,
                "Flush interval must be positive.");
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(options.FilePrefix);
        if (options.FilePrefix.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 ||
            options.FilePrefix.Contains(Path.DirectorySeparatorChar) ||
            options.FilePrefix.Contains(Path.AltDirectorySeparatorChar))
        {
            throw new ArgumentException(
                "Trace file prefix must be a simple file-name component.",
                nameof(options.FilePrefix));
        }

        if (!Enum.IsDefined(options.OverflowPolicy))
        {
            throw new ArgumentOutOfRangeException(nameof(options.OverflowPolicy));
        }

        if (!Enum.IsDefined(options.BackgroundFailurePolicy))
        {
            throw new ArgumentOutOfRangeException(nameof(options.BackgroundFailurePolicy));
        }

        if (!Enum.IsDefined(options.DurabilityPolicy))
        {
            throw new ArgumentOutOfRangeException(nameof(options.DurabilityPolicy));
        }
    }
}
