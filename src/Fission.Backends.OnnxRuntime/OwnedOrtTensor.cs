using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Caller-owned tensor buffer plus the OrtValue view pinned over that buffer.
/// The buffer can be reused across inference calls while ownership is active.
/// Ownership may be explicitly transferred with DetachValue; after transfer this
/// wrapper no longer disposes or exposes the OrtValue/buffer.
/// </summary>
public sealed class OwnedOrtTensor<T> : IDisposable
    where T : unmanaged
{
    private const int Active = 0;
    private const int Disposed = 1;
    private const int Transferred = 2;

    private readonly T[] _buffer;
    private readonly long[] _shape;
    private OrtValue? _value;
    private int _state;

    public OwnedOrtTensor(long[] shape)
        : this(new T[ComputeElementCount(shape)], shape)
    {
    }

    public OwnedOrtTensor(T[] buffer, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentNullException.ThrowIfNull(shape);

        var elementCount = ComputeElementCount(shape);
        if (buffer.Length != elementCount)
        {
            throw new ArgumentException(
                $"Tensor buffer contains {buffer.Length} elements but shape requires {elementCount}.",
                nameof(buffer));
        }

        _buffer = buffer;
        _shape = shape.ToArray();
        _value = OrtValue.CreateTensorValueFromMemory(_buffer, _shape);
    }

    public int ElementCount => _buffer.Length;
    public IReadOnlyList<long> Shape => _shape;
    public bool IsDisposed => Volatile.Read(ref _state) == Disposed;
    public bool IsTransferred => Volatile.Read(ref _state) == Transferred;
    public bool OwnsValue => Volatile.Read(ref _state) == Active;

    public OrtValue Value
    {
        get
        {
            ThrowIfNotActive();
            return _value ?? throw new InvalidOperationException("Owned OrtValue is missing.");
        }
    }

    public Span<T> Span
    {
        get
        {
            ThrowIfNotActive();
            return _buffer.AsSpan();
        }
    }

    public ReadOnlySpan<T> ReadOnlySpan
    {
        get
        {
            ThrowIfNotActive();
            return _buffer.AsSpan();
        }
    }

    /// <summary>
    /// Transfers the OrtValue to another owner without disposing it. This wrapper
    /// becomes permanently transferred and cannot be used for tensor/buffer access.
    /// Dispose becomes a no-op after transfer.
    /// </summary>
    public OrtValue DetachValue()
    {
        if (Interlocked.CompareExchange(ref _state, Transferred, Active) != Active)
        {
            ThrowIfNotActive();
        }

        return Interlocked.Exchange(ref _value, null)
            ?? throw new InvalidOperationException("Owned OrtValue is missing during transfer.");
    }

    public void Dispose()
    {
        var previous = Interlocked.CompareExchange(ref _state, Disposed, Active);
        if (previous != Active)
        {
            return;
        }

        Interlocked.Exchange(ref _value, null)?.Dispose();
    }

    private void ThrowIfNotActive()
    {
        var state = Volatile.Read(ref _state);
        if (state == Disposed)
        {
            throw new ObjectDisposedException(nameof(OwnedOrtTensor<T>));
        }

        if (state == Transferred)
        {
            throw new InvalidOperationException(
                "OwnedOrtTensor has transferred its OrtValue to another owner.");
        }
    }

    private static int ComputeElementCount(long[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Length == 0)
        {
            throw new ArgumentException(
                "Tensor shape must contain at least one dimension.",
                nameof(shape));
        }

        long count = 1;
        foreach (var dimension in shape)
        {
            if (dimension <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(shape),
                    "Preallocated tensor dimensions must be positive and fully known.");
            }

            count = checked(count * dimension);
        }

        if (count > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(shape),
                $"Tensor requires {count} elements, which exceeds managed array limits for this allocator.");
        }

        return checked((int)count);
    }
}
