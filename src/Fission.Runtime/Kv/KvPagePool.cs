namespace Fission.Runtime.Kv;

public sealed class KvPageCapacityExceededException : InvalidOperationException
{
    public KvPageCapacityExceededException(int capacity)
        : base($"KV page capacity {capacity} is exhausted.")
    {
        Capacity = capacity;
    }

    public int Capacity { get; }
}

/// <summary>
/// Metadata-level KV page allocator. A rented page consumes capacity until the
/// final shared lease reference is released. Forks and snapshots therefore do
/// not consume additional physical-page capacity by themselves.
/// </summary>
public sealed class KvPagePool
{
    private readonly int _capacity;
    private int _allocatedPages;
    private long _nextPageId;

    public KvPagePool(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        _capacity = capacity;
    }

    public int Capacity => _capacity;
    public int AllocatedPages => Volatile.Read(ref _allocatedPages);
    public int AvailablePages => _capacity - AllocatedPages;

    internal KvPageLease Rent()
    {
        while (true)
        {
            var current = Volatile.Read(ref _allocatedPages);
            if (current >= _capacity)
            {
                throw new KvPageCapacityExceededException(_capacity);
            }

            if (Interlocked.CompareExchange(ref _allocatedPages, current + 1, current) != current)
            {
                continue;
            }

            var pageId = Interlocked.Increment(ref _nextPageId);
            return new KvPageLease(pageId, this);
        }
    }

    internal void Return()
    {
        var remaining = Interlocked.Decrement(ref _allocatedPages);
        if (remaining < 0)
        {
            Interlocked.Increment(ref _allocatedPages);
            throw new InvalidOperationException("KV page pool returned more pages than it rented.");
        }
    }
}
