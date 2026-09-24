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

    internal int AppendForTokenRange(int position, int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokenCount);

        lock (_gate)
        {
            ThrowIfDisposed();
            var requiredPages = _pool.IncrementalPagesFor(position, tokenCount);
            if (requiredPages == 0)
            {
                return 0;
            }

            var rented = _pool.Rent(requiredPages);
            _pages.AddRange(rented);
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

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
