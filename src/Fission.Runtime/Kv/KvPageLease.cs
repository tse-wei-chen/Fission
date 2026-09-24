using System.Threading;

namespace Fission.Runtime.Kv;

public sealed class KvPageLease
{
    private int _referenceCount = 1;

    public KvPageLease(long pageId)
    {
        PageId = pageId;
    }

    public long PageId { get; }
    public int ReferenceCount => Volatile.Read(ref _referenceCount);

    internal KvPageLease Acquire()
    {
        if (Interlocked.Increment(ref _referenceCount) <= 1)
        {
            throw new ObjectDisposedException(nameof(KvPageLease));
        }

        return this;
    }

    internal void Release()
    {
        var count = Interlocked.Decrement(ref _referenceCount);
        if (count < 0)
        {
            throw new InvalidOperationException("KV page lease released more than it was acquired.");
        }
    }
}
