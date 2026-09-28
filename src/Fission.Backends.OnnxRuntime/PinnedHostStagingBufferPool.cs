namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Allocation boundary for host-staging FP32 buffers. The returned Memory must
/// remain valid and stable for the lifetime of the buffer and must support pinning
/// so ONNX Runtime can create zero-copy OrtValue views over it.
///
/// CUDA-specific implementations can back Memory with a custom MemoryManager over
/// page-locked native memory without changing migration payload ownership or pooling.
/// </summary>
public interface IHostStagingFloatBufferAllocator
{
    IHostStagingFloatBuffer Allocate(int length);
}

/// <summary>
/// One exact-length host-staging allocation. Dispose releases the physical backing
/// allocation; pool leases may retain the buffer across migrations before doing so.
/// </summary>
public interface IHostStagingFloatBuffer : IDisposable
{
    Memory<float> Memory { get; }
}

/// <summary>
/// Optional physical capability for a host-staging buffer that is CUDA page-locked
/// and can therefore be used directly as a cudaMemcpyAsync host endpoint.
///
/// Implementations must keep <see cref="Pointer"/> stable and valid for the same
/// lifetime as <see cref="IHostStagingFloatBuffer.Memory"/>. Ordinary GC-pinned
/// buffers deliberately do not implement this capability: a stable managed address
/// is not equivalent to a CUDA page-locked host allocation.
/// </summary>
public interface ICudaPageLockedHostStagingFloatBuffer : IHostStagingFloatBuffer
{
    nint Pointer { get; }
}

/// <summary>
/// Default allocator used by the ONNX staging path. Buffers are exact-length arrays
/// allocated directly into the pinned object heap so their managed address is stable.
/// </summary>
public sealed class GcPinnedHostStagingFloatBufferAllocator : IHostStagingFloatBufferAllocator
{
    public static GcPinnedHostStagingFloatBufferAllocator Shared { get; } = new();

    public IHostStagingFloatBuffer Allocate(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        return new GcPinnedHostStagingFloatBuffer(
            GC.AllocateUninitializedArray<float>(length, pinned: true));
    }

    private sealed class GcPinnedHostStagingFloatBuffer(float[] buffer) : IHostStagingFloatBuffer
    {
        public Memory<float> Memory { get; } = buffer;

        public void Dispose()
        {
            // GC owns the pinned-object-heap allocation. Dropping the final managed
            // reference is the physical release operation.
        }
    }
}

/// <summary>
/// Retention policy for host buffers used by decoder staging. Buffers are exact
/// length so they can be exposed directly to ONNX Runtime without unused capacity.
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
    private readonly Dictionary<int, Stack<IHostStagingFloatBuffer>> _buckets = new();
    private readonly PinnedHostStagingPoolOptions _options;
    private readonly IHostStagingFloatBufferAllocator _allocator;
    private long _allocatedBuffers;
    private long _reusedBuffers;
    private long _returnedBuffers;
    private long _droppedBuffers;
    private int _retainedBuffers;
    private long _retainedBytes;
    private bool _disposed;

    public PinnedFloatBufferPool(
        PinnedHostStagingPoolOptions? options = null,
        IHostStagingFloatBufferAllocator? allocator = null)
    {
        _options = options ?? new PinnedHostStagingPoolOptions();
        _options.Validate();
        _allocator = allocator ?? GcPinnedHostStagingFloatBufferAllocator.Shared;
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
        }

        var allocated = _allocator.Allocate(length) ??
            throw new InvalidOperationException(
                $"Host-staging allocator {_allocator.GetType().Name} returned null.");

        if (allocated.Memory.Length != length)
        {
            var actualLength = allocated.Memory.Length;
            allocated.Dispose();
            throw new InvalidOperationException(
                $"Host-staging allocator {_allocator.GetType().Name} returned {actualLength} element(s); expected exact length {length}.");
        }

        lock (_gate)
        {
            if (_disposed)
            {
                allocated.Dispose();
                throw new ObjectDisposedException(nameof(PinnedFloatBufferPool));
            }

            _allocatedBuffers++;
        }

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
        IHostStagingFloatBuffer[] retained;
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            retained = _buckets.Values
                .SelectMany(static bucket => bucket)
                .ToArray();
            _buckets.Clear();
            _retainedBuffers = 0;
            _retainedBytes = 0;
        }

        foreach (var buffer in retained)
        {
            buffer.Dispose();
        }
    }

    private void Return(IHostStagingFloatBuffer buffer)
    {
        var length = buffer.Memory.Length;
        var bytes = GetByteLength(length);
        if (_options.ClearOnReturn)
        {
            buffer.Memory.Span.Clear();
        }

        var retain = false;
        lock (_gate)
        {
            _returnedBuffers++;

            if (!_disposed &&
                _options.MaxRetainedBuffersPerLength != 0 &&
                _options.MaxRetainedBytes >= bytes &&
                _retainedBytes <= _options.MaxRetainedBytes - bytes)
            {
                if (!_buckets.TryGetValue(length, out var bucket))
                {
                    bucket = new Stack<IHostStagingFloatBuffer>();
                    _buckets.Add(length, bucket);
                }

                if (bucket.Count < _options.MaxRetainedBuffersPerLength)
                {
                    bucket.Push(buffer);
                    _retainedBuffers++;
                    _retainedBytes = checked(_retainedBytes + bytes);
                    retain = true;
                }
            }

            if (!retain)
            {
                _droppedBuffers++;
            }
        }

        if (!retain)
        {
            buffer.Dispose();
        }
    }

    private static long GetByteLength(int elementCount) =>
        checked(elementCount * (long)sizeof(float));

    internal sealed class PinnedFloatBufferLease : IDisposable
    {
        private PinnedFloatBufferPool? _owner;
        private readonly IHostStagingFloatBuffer _buffer;

        internal PinnedFloatBufferLease(
            PinnedFloatBufferPool owner,
            IHostStagingFloatBuffer buffer)
        {
            _owner = owner;
            _buffer = buffer;
        }

        public Memory<float> Memory => _buffer.Memory;
        public Span<float> Span => _buffer.Memory.Span;

        internal nint GetCudaPageLockedPointer()
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _owner) is null,
                this);

            var cudaBuffer = _buffer as ICudaPageLockedHostStagingFloatBuffer ??
                throw new InvalidOperationException(
                    $"Host-staging buffer {_buffer.GetType().Name} is not CUDA page-locked and cannot be used for asynchronous CUDA DMA.");
            var pointer = cudaBuffer.Pointer;
            if (pointer == 0)
            {
                throw new InvalidOperationException(
                    "CUDA page-locked host-staging buffer exposed a null native pointer.");
            }

            return pointer;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?.Return(_buffer);
        }
    }
}
