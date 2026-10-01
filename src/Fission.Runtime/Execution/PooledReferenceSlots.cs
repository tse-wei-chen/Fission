using System.Buffers;

namespace Fission.Runtime.Execution;

/// <summary>
/// Fixed logical-length reference storage backed by ArrayPool. Callers own the
/// lifetime boundary and must not access the slots after Return. The logical
/// range is cleared on rent because ArrayPool does not guarantee zeroed storage;
/// the full rented buffer is cleared again when returned so references do not
/// escape through the shared pool.
/// </summary>
internal sealed class PooledReferenceSlots<T>
    where T : class
{
    private T?[]? _buffer;

    internal PooledReferenceSlots(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
        var buffer = ArrayPool<T?>.Shared.Rent(length);
        Array.Clear(buffer, 0, length);
        _buffer = buffer;
    }

    internal int Length { get; }

    internal bool IsReturned => Volatile.Read(ref _buffer) is null;

    internal T? this[int index]
    {
        get => GetBuffer()[index];
        set => GetBuffer()[index] = value;
    }

    internal void Return()
    {
        var buffer = Interlocked.Exchange(ref _buffer, null);
        if (buffer is not null)
        {
            ArrayPool<T?>.Shared.Return(buffer, clearArray: true);
        }
    }

    private T?[] GetBuffer() =>
        Volatile.Read(ref _buffer) ??
        throw new ObjectDisposedException(
            nameof(PooledReferenceSlots<T>),
            "Pooled slot storage was already returned.");
}
