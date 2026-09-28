namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Retention policy for GC-pinned managed host buffers used by decoder staging.
/// Buffers are exact-length arrays so they can be passed directly to existing
/// OrtValue array-backed tensor constructors without exposing unused capacity.
/// </summary>
public sealed record PinnedHostStagingPoolOptions
{
    public int MaxRetainedBuffersPerLength { get; init; } = 8;
    public long MaxRetainedBytes { get; init; } = 256L * 1024 * 1024;
    public bool ClearOnReturn { get; init; } = true;

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegative(MaxRetainedBuffersPerLength);
        ArgumentOutOfRangeException.ThrowIfNegative(MaxRetainedBytes);
    }
}

public readonly record struct PinnedHostStagingPoolStatistics(
    long AllocatedBuffers,
    long ReusedBuffers,
    long ReturnedBuffers,
    long DroppedBuffers,
    int RetainedBuffers,
    long RetainedBytes);

internal sealed class PinnedFloatBufferPool : IDisposable
{
    private readonly object _gate = new();
    private readonly Dictionary<int, Stack<float[]>> _buckets = new();
    private readonly PinnedHostStagingPoolOptions _options;
    private long _allocatedBuffers;
    private long _reusedBuffers;
    private long _returnedBuffers;
    private long _droppedBuffers;
    private int _retainedBuffers;
    private long _retainedBytes;
    private bool _disposed;

    public PinnedFloatBufferPool(PinnedHostStagingPoolOptions? options = null)
    {
        _options = options ?? new PinnedHostStagingPoolOptions();
        _options.Validate();
    }

    public PinnedFloatBufferLease Rent(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);

        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            if (_buckets.TryGetValue(length, out var bucket) && bucket.Count != 0)
            {
                var buffer = bucket.Pop();
                if (bucket.Count == 0)
                {
                    _buckets.Remove(length);
                }

                _reusedBuffers++;
                _retainedBuffers--;
                _retainedBytes = checked(_retainedBytes - GetByteLength(length));
                return new PinnedFloatBufferLease(this, buffer);
            }

            _allocatedBuffers++;
        }

        var allocated = GC.AllocateUninitializedArray<float>(length, pinned: true);
        return new PinnedFloatBufferLease(this, allocated);
    }

    public PinnedHostStagingPoolStatistics GetStatistics()
    {
        lock (_gate)
        {
            return new PinnedHostStagingPoolStatistics(
                _allocatedBuffers,
                _reusedBuffers,
                _returnedBuffers,
                _droppedBuffers,
                _retainedBuffers,
                _retainedBytes);
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _buckets.Clear();
            _retainedBuffers = 0;
            _retainedBytes = 0;
        }
    }

    private void Return(float[] buffer)
    {
        var bytes = GetByteLength(buffer.Length);
        if (_options.ClearOnReturn)
        {
            Array.Clear(buffer);
        }

        lock (_gate)
        {
            _returnedBuffers++;

            if (_disposed ||
                _options.MaxRetainedBuffersPerLength == 0 ||
                _options.MaxRetainedBytes < bytes ||
                _retainedBytes > _options.MaxRetainedBytes - bytes)
            {
                _droppedBuffers++;
                return;
            }

            if (!_buckets.TryGetValue(buffer.Length, out var bucket))
            {
                bucket = new Stack<float[]>();
                _buckets.Add(buffer.Length, bucket);
            }

            if (bucket.Count >= _options.MaxRetainedBuffersPerLength)
            {
                _droppedBuffers++;
                return;
            }

            bucket.Push(buffer);
            _retainedBuffers++;
            _retainedBytes = checked(_retainedBytes + bytes);
        }
    }

    private static long GetByteLength(int elementCount) =>
        checked(elementCount * (long)sizeof(float));

    internal sealed class PinnedFloatBufferLease : IDisposable
    {
        private PinnedFloatBufferPool? _owner;

        internal PinnedFloatBufferLease(
            PinnedFloatBufferPool owner,
            float[] buffer)
        {
            _owner = owner;
            Buffer = buffer;
        }

        public float[] Buffer { get; }
        public Span<float> Span => Buffer.AsSpan();

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?.Return(Buffer);
        }
    }
}
