namespace Fission.Runtime.Kv;

public sealed class KvPageLease
{
    private readonly KvPagePool _pool;
    private int _referenceCount = 1;

    internal KvPageLease(long pageId, KvPagePool pool)
    {
        PageId = pageId;
        _pool = pool;
    }

    public long PageId { get; }
    public int ReferenceCount => Volatile.Read(ref _referenceCount);

    internal KvPageLease Acquire()
    {
        while (true)
        {
            var current = Volatile.Read(ref _referenceCount);
            if (current == 0)
            {
                throw new ObjectDisposedException(nameof(KvPageLease));
            }

            if (current == int.MaxValue)
            {
                throw new OverflowException("KV page lease reference count overflowed.");
            }

            if (Interlocked.CompareExchange(ref _referenceCount, current + 1, current) == current)
            {
                return this;
            }
        }
    }

    internal void Release()
    {
        var count = Interlocked.Decrement(ref _referenceCount);
        if (count < 0)
        {
            Interlocked.Increment(ref _referenceCount);
            throw new InvalidOperationException("KV page lease released more than it was acquired.");
        }

        if (count == 0)
        {
            _pool.Return();
        }
    }
}
