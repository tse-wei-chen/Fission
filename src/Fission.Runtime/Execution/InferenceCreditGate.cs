namespace Fission.Runtime.Execution;

/// <summary>
/// Weighted asynchronous capacity gate for device inference work. Acquisitions
/// are serialized so a multi-item atomic envelope cannot partially consume the
/// available credits while another producer acquires the remainder. Credits are
/// returned only when the associated device work reaches a terminal state.
/// </summary>
internal sealed class InferenceCreditGate : IDisposable
{
    private readonly SemaphoreSlim _credits;
    private readonly SemaphoreSlim _acquireGate = new(1, 1);
    private int _disposed;

    public InferenceCreditGate(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        Capacity = capacity;
        _credits = new SemaphoreSlim(capacity, capacity);
    }

    public int Capacity { get; }

    public async ValueTask<Lease> AcquireAsync(
        int count,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);

        if (count > Capacity)
        {
            throw new InvalidOperationException(
                $"Inference work requires {count} device credit(s), but the configured capacity is {Capacity}.");
        }

        await _acquireGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        var acquired = 0;

        try
        {
            for (; acquired < count; acquired++)
            {
                await _credits.WaitAsync(cancellationToken).ConfigureAwait(false);
            }

            return new Lease(this, count);
        }
        catch
        {
            if (acquired != 0)
            {
                _credits.Release(acquired);
            }

            throw;
        }
        finally
        {
            _acquireGate.Release();
        }
    }

    private void Release(int count)
    {
        if (Volatile.Read(ref _disposed) == 0)
        {
            _credits.Release(count);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _acquireGate.Dispose();
        _credits.Dispose();
    }

    internal sealed class Lease : IDisposable
    {
        private InferenceCreditGate? _owner;
        private readonly int _count;

        internal Lease(InferenceCreditGate owner, int count)
        {
            _owner = owner;
            _count = count;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?.Release(_count);
        }
    }
}
