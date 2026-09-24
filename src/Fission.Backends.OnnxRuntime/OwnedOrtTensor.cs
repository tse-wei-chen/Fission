using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Caller-owned tensor buffer plus the OrtValue view pinned over that buffer.
/// The buffer can be reused across inference calls while the tensor is alive.
/// Disposing this object releases the OrtValue and its pin; ONNX Runtime does not
/// take ownership when the value is supplied through the preallocated-output Run
/// overload.
/// </summary>
public sealed class OwnedOrtTensor<T> : IDisposable
    where T : unmanaged
{
    private readonly T[] _buffer;
    private readonly long[] _shape;
    private OrtValue? _value;
    private int _disposed;

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
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public OrtValue Value
    {
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _value ?? throw new ObjectDisposedException(nameof(OwnedOrtTensor<T>));
        }
    }

    public Span<T> Span
    {
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _buffer.AsSpan();
        }
    }

    public ReadOnlySpan<T> ReadOnlySpan
    {
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _buffer.AsSpan();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        Interlocked.Exchange(ref _value, null)?.Dispose();
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
