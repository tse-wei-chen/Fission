using System.Buffers;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Short-lived exact-length view over an ArrayPool rental.
///
/// ArrayPool may return an array larger than requested. Consumers must use
/// Memory/Span so tensor shapes never accidentally include spare pool capacity.
/// Dispose returns the physical array exactly once; no Memory/Span may be used
/// after disposal.
/// </summary>
internal sealed class ArrayPoolLease<T> : IDisposable
{
    private readonly ArrayPool<T> _pool;
    private T[]? _buffer;

    public ArrayPoolLease(ArrayPool<T> pool, int length)
    {
        ArgumentNullException.ThrowIfNull(pool);
        if (length < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        _pool = pool;
        _buffer = pool.Rent(length);
        Length = length;
    }

    public int Length { get; }
    public bool IsDisposed => Volatile.Read(ref _buffer) is null;

    public Memory<T> Memory
    {
        get
        {
            var buffer = Volatile.Read(ref _buffer)
                ?? throw new ObjectDisposedException(nameof(ArrayPoolLease<T>));
            return buffer.AsMemory(0, Length);
        }
    }

    public Span<T> Span
    {
        get
        {
            var buffer = Volatile.Read(ref _buffer)
                ?? throw new ObjectDisposedException(nameof(ArrayPoolLease<T>));
            return buffer.AsSpan(0, Length);
        }
    }

    public void Dispose()
    {
        var buffer = Interlocked.Exchange(ref _buffer, null);
        if (buffer is null)
        {
            return;
        }

        _pool.Return(buffer, clearArray: false);
    }
}
