namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Retention bounds for exact-size CUDA device allocation reuse.
/// </summary>
public sealed record CudaDeviceMemoryPoolOptions
{
    /// <summary>
    /// Maximum total bytes retained while idle. Zero disables retention.
    /// </summary>
    public required long MaxRetainedBytes { get; init; }

    /// <summary>
    /// Maximum retained allocations for one exact byte length.
    /// </summary>
    public int MaxRetainedBuffersPerSize { get; init; } = 8;

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegative(MaxRetainedBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(MaxRetainedBuffersPerSize);
    }
}

/// <summary>
/// Point-in-time allocation/reuse counters for a CUDA device-memory pool.
/// </summary>
public sealed record CudaDeviceMemoryPoolStatistics(
    long NativeAllocations,
    long Reuses,
    long Returns,
    long Drops,
    int RetainedBuffers,
    long RetainedBytes)
{
    /// <summary>
    /// Allocations currently leased to callers.
    /// </summary>
    public int ActiveBuffers { get; init; }

    /// <summary>
    /// CUDA bytes currently leased to callers.
    /// </summary>
    public long ActiveBytes { get; init; }

    /// <summary>
    /// Active plus idle-retained allocations still resident on the CUDA device.
    /// </summary>
    public int ReservedBuffers => checked(ActiveBuffers + RetainedBuffers);

    /// <summary>
    /// Active plus idle-retained bytes still resident on the CUDA device.
    /// </summary>
    public long ReservedBytes => checked(ActiveBytes + RetainedBytes);

    /// <summary>
    /// Maximum simultaneously active bytes observed by this pool.
    /// </summary>
    public long PeakActiveBytes { get; init; }

    /// <summary>
    /// Maximum active plus retained bytes observed by this pool.
    /// </summary>
    public long PeakReservedBytes { get; init; }

    /// <summary>
    /// Number of idle buffers released by explicit trim operations.
    /// </summary>
    public long TrimmedBuffers { get; init; }

    /// <summary>
    /// Number of idle bytes released by explicit trim operations.
    /// </summary>
    public long TrimmedBytes { get; init; }
}

/// <summary>
/// Result of one explicit idle-retention trim.
/// </summary>
public readonly record struct CudaDeviceMemoryPoolTrimResult(
    int ReleasedBuffers,
    long ReleasedBytes,
    int RetainedBuffers,
    long RetainedBytes);

/// <summary>
/// Bounded exact-size reuse layer over <see cref="CudaDeviceMemoryAllocator"/>.
///
/// Disposing an outstanding allocation returns its native handle to this pool when
/// both retention limits permit it. Otherwise the handle is released immediately.
/// The pool never rounds allocation sizes and never returns a differently-sized
/// CUDA allocation to a caller.
/// </summary>
public sealed class CudaPooledDeviceMemoryAllocator :
    CudaDeviceMemoryAllocator,
    IDisposable
{
    private readonly object _gate = new();
    private readonly CudaDeviceMemoryPoolOptions _poolOptions;
    private readonly Dictionary<long, Stack<CudaDeviceAllocationHandle>> _buckets = new();
    private long _nativeAllocations;
    private long _reuses;
    private long _returns;
    private long _drops;
    private long _retainedBytes;
    private int _retainedBuffers;
    private long _activeBytes;
    private int _activeBuffers;
    private long _peakActiveBytes;
    private long _peakReservedBytes;
    private long _trimmedBuffers;
    private long _trimmedBytes;
    private int _disposed;

    public CudaPooledDeviceMemoryAllocator(
        CudaDeviceMemoryPoolOptions poolOptions,
        CudaDeviceMemoryAllocatorOptions? allocatorOptions = null)
        : base(allocatorOptions)
    {
        ArgumentNullException.ThrowIfNull(poolOptions);
        poolOptions.Validate();
        _poolOptions = poolOptions;
    }

    internal CudaPooledDeviceMemoryAllocator(
        ICudaDeviceMemoryApi cuda,
        CudaDeviceMemoryPoolOptions poolOptions,
        CudaDeviceMemoryAllocatorOptions? allocatorOptions = null)
        : base(cuda, allocatorOptions)
    {
        ArgumentNullException.ThrowIfNull(poolOptions);
        poolOptions.Validate();
        _poolOptions = poolOptions;
    }

    public CudaDeviceMemoryPoolStatistics Statistics
    {
        get
        {
            lock (_gate)
            {
                return new CudaDeviceMemoryPoolStatistics(
                    NativeAllocations: _nativeAllocations,
                    Reuses: _reuses,
                    Returns: _returns,
                    Drops: _drops,
                    RetainedBuffers: _retainedBuffers,
                    RetainedBytes: _retainedBytes)
                {
                    ActiveBuffers = _activeBuffers,
                    ActiveBytes = _activeBytes,
                    PeakActiveBytes = _peakActiveBytes,
                    PeakReservedBytes = _peakReservedBytes,
                    TrimmedBuffers = _trimmedBuffers,
                    TrimmedBytes = _trimmedBytes
                };
            }
        }
    }

    public override CudaDeviceMemoryAllocation Allocate(long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        ThrowIfDisposed();

        lock (_gate)
        {
            ThrowIfDisposed();
            if (_buckets.TryGetValue(byteLength, out var bucket) && bucket.Count > 0)
            {
                var reused = bucket.Pop();
                _retainedBuffers--;
                _retainedBytes = checked(_retainedBytes - byteLength);
                if (bucket.Count == 0)
                {
                    _buckets.Remove(byteLength);
                }

                _reuses = checked(_reuses + 1);
                MarkLeaseIssued(byteLength);
                return CreateLease(reused, byteLength);
            }
        }

        var handle = AllocateHandle(byteLength);
        var rejectBecauseDisposed = false;
        lock (_gate)
        {
            _nativeAllocations = checked(_nativeAllocations + 1);
            if (Volatile.Read(ref _disposed) != 0)
            {
                rejectBecauseDisposed = true;
            }
            else
            {
                MarkLeaseIssued(byteLength);
            }
        }

        if (rejectBecauseDisposed)
        {
            handle.Dispose();
            throw new ObjectDisposedException(nameof(CudaPooledDeviceMemoryAllocator));
        }

        return CreateLease(handle, byteLength);
    }

    /// <summary>
    /// Releases currently idle allocations until retained bytes are at or below
    /// <paramref name="targetRetainedBytes"/>. Largest exact-size buckets are
    /// released first to reduce cudaFree submissions for a requested byte target.
    /// Active caller leases are never revoked. Leases returned after this snapshot
    /// may be retained again under the configured pool bounds.
    /// </summary>
    public CudaDeviceMemoryPoolTrimResult TrimRetained(
        long targetRetainedBytes = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetRetainedBytes);
        ThrowIfDisposed();

        List<CudaDeviceAllocationHandle>? released = null;
        long releasedBytes = 0;
        CudaDeviceMemoryPoolTrimResult result;

        lock (_gate)
        {
            ThrowIfDisposed();
            if (_retainedBytes > targetRetainedBytes)
            {
                var sizes = _buckets.Keys.ToArray();
                Array.Sort(sizes);
                released = new List<CudaDeviceAllocationHandle>();

                for (var sizeIndex = sizes.Length - 1;
                     sizeIndex >= 0 && _retainedBytes > targetRetainedBytes;
                     sizeIndex--)
                {
                    var byteLength = sizes[sizeIndex];
                    if (!_buckets.TryGetValue(byteLength, out var bucket))
                    {
                        continue;
                    }

                    while (bucket.Count > 0 && _retainedBytes > targetRetainedBytes)
                    {
                        released.Add(bucket.Pop());
                        _retainedBuffers--;
                        _retainedBytes = checked(_retainedBytes - byteLength);
                        releasedBytes = checked(releasedBytes + byteLength);
                    }

                    if (bucket.Count == 0)
                    {
                        _buckets.Remove(byteLength);
                    }
                }

                _trimmedBuffers = checked(_trimmedBuffers + released.Count);
                _trimmedBytes = checked(_trimmedBytes + releasedBytes);
            }

            result = new CudaDeviceMemoryPoolTrimResult(
                ReleasedBuffers: released?.Count ?? 0,
                ReleasedBytes: releasedBytes,
                RetainedBuffers: _retainedBuffers,
                RetainedBytes: _retainedBytes);
        }

        if (released is not null)
        {
            foreach (var handle in released)
            {
                handle.Dispose();
            }
        }

        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        List<CudaDeviceAllocationHandle> retained;
        lock (_gate)
        {
            retained = new List<CudaDeviceAllocationHandle>(_retainedBuffers);
            foreach (var bucket in _buckets.Values)
            {
                while (bucket.Count > 0)
                {
                    retained.Add(bucket.Pop());
                }
            }

            _buckets.Clear();
            _retainedBuffers = 0;
            _retainedBytes = 0;
        }

        foreach (var handle in retained)
        {
            handle.Dispose();
        }
    }

    private CudaDeviceMemoryAllocation CreateLease(
        CudaDeviceAllocationHandle handle,
        long byteLength) =>
        new(
            handle,
            byteLength,
            DeviceId,
            Return);

    private void MarkLeaseIssued(long byteLength)
    {
        _activeBuffers = checked(_activeBuffers + 1);
        _activeBytes = checked(_activeBytes + byteLength);
        _peakActiveBytes = Math.Max(_peakActiveBytes, _activeBytes);
        _peakReservedBytes = Math.Max(
            _peakReservedBytes,
            checked(_activeBytes + _retainedBytes));
    }

    private void Return(
        CudaDeviceAllocationHandle handle,
        long byteLength)
    {
        var retained = false;
        lock (_gate)
        {
            _returns = checked(_returns + 1);
            _activeBuffers--;
            _activeBytes = checked(_activeBytes - byteLength);

            if (Volatile.Read(ref _disposed) == 0 &&
                _poolOptions.MaxRetainedBytes > 0 &&
                _poolOptions.MaxRetainedBuffersPerSize > 0 &&
                byteLength <= _poolOptions.MaxRetainedBytes - _retainedBytes)
            {
                if (!_buckets.TryGetValue(byteLength, out var bucket))
                {
                    bucket = new Stack<CudaDeviceAllocationHandle>();
                    _buckets.Add(byteLength, bucket);
                }

                if (bucket.Count < _poolOptions.MaxRetainedBuffersPerSize)
                {
                    bucket.Push(handle);
                    _retainedBuffers++;
                    _retainedBytes = checked(_retainedBytes + byteLength);
                    retained = true;
                }
            }

            if (!retained)
            {
                _drops = checked(_drops + 1);
            }
        }

        if (!retained)
        {
            handle.Dispose();
        }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}
