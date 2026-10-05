using Fission.Abstractions;

namespace Fission.Runtime.Kv;

public sealed class KvPageTable : IDisposable
{
    private readonly object _gate = new();
    private readonly KvPagePool _pool;
    private List<KvPageLease> _pages;
    private bool _disposed;

    public KvPageTable()
        : this(new KvPagePool(int.MaxValue), [])
    {
    }

    internal KvPageTable(KvPagePool pool)
        : this(pool, [])
    {
    }

    private KvPageTable(KvPagePool pool, IEnumerable<KvPageLease> acquiredPages)
    {
        _pool = pool;
        _pages = [.. acquiredPages];
    }

    public int TokensPerPage => _pool.TokensPerPage;

    public int Count
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return _pages.Count;
            }
        }
    }

    public IReadOnlyList<long> PageIds
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return _pages.Select(static page => page.PageId).ToArray();
            }
        }
    }

    internal int WriteOverheadForTokenRange(int position, int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegative(tokenCount);

        lock (_gate)
        {
            ThrowIfDisposed();
            return RequiresTailCopyOnWrite(position, tokenCount) ? 1 : 0;
        }
    }

    internal int AdditionalPagesForTokenRange(int position, int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegative(tokenCount);

        lock (_gate)
        {
            ThrowIfDisposed();
            var appendedPages = _pool.IncrementalPagesFor(position, tokenCount);
            var copyOnWritePages = RequiresTailCopyOnWrite(position, tokenCount) ? 1 : 0;
            return checked(appendedPages + copyOnWritePages);
        }
    }

    internal int AppendForTokenRange(int position, int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokenCount);

        lock (_gate)
        {
            ThrowIfDisposed();

            var copyTail = RequiresTailCopyOnWrite(position, tokenCount);
            var appendedPages = _pool.IncrementalPagesFor(position, tokenCount);
            var requiredPages = checked(appendedPages + (copyTail ? 1 : 0));
            if (requiredPages == 0)
            {
                return 0;
            }

            // Ensure List<T> cannot allocate after pool capacity has been reserved.
            // Once Rent succeeds the mutation below is allocation-free and the old
            // shared tail is released only after its replacement is installed.
            _pages.EnsureCapacity(checked(_pages.Count + appendedPages));
            var rented = _pool.Rent(requiredPages);
            var rentedIndex = 0;

            if (copyTail)
            {
                var sharedTail = _pages[^1];
                _pages[^1] = rented[rentedIndex++];
                sharedTail.Release();
            }

            while (rentedIndex < rented.Count)
            {
                _pages.Add(rented[rentedIndex++]);
            }

            return requiredPages;
        }
    }

    public KvPageTable Fork()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return new KvPageTable(_pool, _pages.Select(static page => page.Acquire()));
        }
    }

    public KvSnapshot Snapshot(SequenceId sequenceId, long version, int position)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var acquiredPages = _pages.Select(static page => page.Acquire()).ToArray();
            return new KvSnapshot(sequenceId, version, position, _pool, acquiredPages);
        }
    }

    public static KvPageTable Restore(KvSnapshot snapshot) =>
        new(snapshot.Pool, snapshot.AcquirePages());

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            foreach (var page in _pages)
            {
                page.Release();
            }

            _pages = [];
        }
    }

    private bool RequiresTailCopyOnWrite(int position, int tokenCount)
    {
        if (tokenCount == 0 ||
            position == 0 ||
            position % _pool.TokensPerPage == 0 ||
            _pages.Count == 0)
        {
            return false;
        }

        return _pages[^1].ReferenceCount > 1;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
