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
                    NativeAllocations: Interlocked.Read(ref _nativeAllocations),
                    Reuses: Interlocked.Read(ref _reuses),
                    Returns: Interlocked.Read(ref _returns),
                    Drops: Interlocked.Read(ref _drops),
                    RetainedBuffers: _retainedBuffers,
                    RetainedBytes: _retainedBytes);
            }
        }
    }

    public override CudaDeviceMemoryAllocation Allocate(long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        ThrowIfDisposed();

        CudaDeviceAllocationHandle? handle = null;
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_buckets.TryGetValue(byteLength, out var bucket) && bucket.Count > 0)
            {
                handle = bucket.Pop();
                _retainedBuffers--;
                _retainedBytes = checked(_retainedBytes - byteLength);
                if (bucket.Count == 0)
                {
                    _buckets.Remove(byteLength);
                }
            }
        }

        if (handle is not null)
        {
            Interlocked.Increment(ref _reuses);
            return CreateLease(handle, byteLength);
        }

        handle = AllocateHandle(byteLength);
        Interlocked.Increment(ref _nativeAllocations);
        return CreateLease(handle, byteLength);
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

    private void Return(
        CudaDeviceAllocationHandle handle,
        long byteLength)
    {
        Interlocked.Increment(ref _returns);
        var retained = false;
        lock (_gate)
        {
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
        }

        if (retained)
        {
            return;
        }

        Interlocked.Increment(ref _drops);
        handle.Dispose();
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}
