using System.Buffers;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Ref-counted managed backing storage for one dense decoder cohort frontier.
///
/// The arena owns one rented key/value buffer per layer. A builder reference keeps
/// the buffers alive while ORT output handles and row states are being created;
/// every DecoderOrtState row then retains the arena independently. Buffers return
/// to the binding-owned pool only after the builder and the final row state release
/// their references.
/// </summary>
internal sealed class DecoderOrtCohortArena
{
    private readonly ArrayPool<float> _bufferPool;
    private readonly float[][] _keyBuffers;
    private readonly float[][] _valueBuffers;
    private readonly int _logicalBufferLength;
    private int _referenceCount = 1;
    private int _returned;

    public DecoderOrtCohortArena(
        int position,
        int batchSize,
        int layerCount,
        int logicalBufferLength,
        ArrayPool<float> bufferPool)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (batchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        }

        if (layerCount < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(layerCount));
        }

        if (logicalBufferLength < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(logicalBufferLength));
        }

        ArgumentNullException.ThrowIfNull(bufferPool);

        Position = position;
        BatchSize = batchSize;
        _logicalBufferLength = logicalBufferLength;
        _bufferPool = bufferPool;
        _keyBuffers = new float[layerCount][];
        _valueBuffers = new float[layerCount][];

        var rentedKeys = 0;
        var rentedValues = 0;
        try
        {
            for (var layer = 0; layer < layerCount; layer++)
            {
                _keyBuffers[layer] = bufferPool.Rent(logicalBufferLength);
                rentedKeys++;
                _valueBuffers[layer] = bufferPool.Rent(logicalBufferLength);
                rentedValues++;
            }
        }
        catch
        {
            for (var layer = rentedValues - 1; layer >= 0; layer--)
            {
                bufferPool.Return(_valueBuffers[layer], clearArray: false);
            }

            for (var layer = rentedKeys - 1; layer >= 0; layer--)
            {
                bufferPool.Return(_keyBuffers[layer], clearArray: false);
            }

            throw;
        }
    }

    public int Position { get; }
    public int BatchSize { get; }
    public int LayerCount => _keyBuffers.Length;
    public int LogicalBufferLength => _logicalBufferLength;
    public bool IsReturned => Volatile.Read(ref _returned) != 0;

    public Memory<float> GetKeyMemory(int layer)
    {
        ThrowIfReturned();
        ValidateLayer(layer);
        return _keyBuffers[layer].AsMemory(0, _logicalBufferLength);
    }

    public Memory<float> GetValueMemory(int layer)
    {
        ThrowIfReturned();
        ValidateLayer(layer);
        return _valueBuffers[layer].AsMemory(0, _logicalBufferLength);
    }

    /// <summary>
    /// Adds an ownership reference. This is safe while the builder reference or
    /// another state reference is still alive; retaining an already-returned arena
    /// is rejected rather than risking use-after-return.
    /// </summary>
    public void Retain()
    {
        while (true)
        {
            var current = Volatile.Read(ref _referenceCount);
            if (current <= 0 || IsReturned)
            {
                throw new ObjectDisposedException(
                    nameof(DecoderOrtCohortArena),
                    "Cannot retain a cohort arena after its buffers were returned.");
            }

            if (Interlocked.CompareExchange(
                    ref _referenceCount,
                    checked(current + 1),
                    current) == current)
            {
                return;
            }
        }
    }

    /// <summary>
    /// Releases one ownership reference and returns every rented buffer exactly
    /// once when the final reference disappears.
    /// </summary>
    public void Release()
    {
        var remaining = Interlocked.Decrement(ref _referenceCount);
        if (remaining < 0)
        {
            throw new InvalidOperationException(
                "Decoder cohort arena was released more times than it was retained.");
        }

        if (remaining != 0)
        {
            return;
        }

        if (Interlocked.Exchange(ref _returned, 1) != 0)
        {
            throw new InvalidOperationException(
                "Decoder cohort arena buffers were already returned.");
        }

        for (var layer = _keyBuffers.Length - 1; layer >= 0; layer--)
        {
            _bufferPool.Return(_valueBuffers[layer], clearArray: false);
            _bufferPool.Return(_keyBuffers[layer], clearArray: false);
        }
    }

    private void ValidateLayer(int layer)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        if (layer >= _keyBuffers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }
    }

    private void ThrowIfReturned() =>
        ObjectDisposedException.ThrowIf(IsReturned, this);
}

internal readonly record struct DecoderOrtCohortSlice(
    DecoderOrtCohortArena Arena,
    int Row);
