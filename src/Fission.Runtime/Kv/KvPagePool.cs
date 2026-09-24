namespace Fission.Runtime.Kv;

public sealed class KvPageCapacityExceededException : InvalidOperationException
{
    public KvPageCapacityExceededException(int capacity, int requestedPages = 1)
        : base($"KV page capacity {capacity} cannot satisfy a request for {requestedPages} additional page(s).")
    {
        Capacity = capacity;
        RequestedPages = requestedPages;
    }

    public int Capacity { get; }
    public int RequestedPages { get; }
}

/// <summary>
/// Metadata-level KV page allocator. A rented page consumes capacity until the
/// final shared lease reference is released. Page demand is derived from token
/// position and a fixed token block size, matching paged KV allocation semantics.
/// </summary>
public sealed class KvPagePool
{
    private readonly int _capacity;
    private readonly int _tokensPerPage;
    private int _allocatedPages;
    private long _nextPageId;

    public KvPagePool(int capacity, int tokensPerPage = 16)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokensPerPage);
        _capacity = capacity;
        _tokensPerPage = tokensPerPage;
    }

    public int Capacity => _capacity;
    public int TokensPerPage => _tokensPerPage;
    public int AllocatedPages => Volatile.Read(ref _allocatedPages);
    public int AvailablePages => _capacity - AllocatedPages;

    public int IncrementalPagesFor(int position, int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegative(tokenCount);

        var endPosition = (long)position + tokenCount;
        var pageDelta = PagesForTokens(endPosition) - PagesForTokens(position);
        return checked((int)pageDelta);
    }

    public int TokensWritableWith(int position, int additionalPages)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegative(additionalPages);

        var currentPages = PagesForTokens(position);
        var capacityPages = currentPages + additionalPages;
        var capacityTokens = checked(capacityPages * _tokensPerPage);
        var writable = Math.Max(0L, capacityTokens - position);
        return writable > int.MaxValue ? int.MaxValue : (int)writable;
    }

    internal IReadOnlyList<KvPageLease> Rent(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (count == 0)
        {
            return Array.Empty<KvPageLease>();
        }

        while (true)
        {
            var current = Volatile.Read(ref _allocatedPages);
            if ((long)current + count > _capacity)
            {
                throw new KvPageCapacityExceededException(_capacity, count);
            }

            if (Interlocked.CompareExchange(ref _allocatedPages, current + count, current) != current)
            {
                continue;
            }

            var leases = new KvPageLease[count];
            for (var index = 0; index < count; index++)
            {
                var pageId = Interlocked.Increment(ref _nextPageId);
                leases[index] = new KvPageLease(pageId, this);
            }

            return leases;
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

    private long PagesForTokens(long tokenCount)
    {
        if (tokenCount <= 0)
        {
            return 0;
        }

        return ((tokenCount - 1) / _tokensPerPage) + 1;
    }
}
