using Fission.Abstractions;

namespace Fission.Runtime.Kv;

public sealed class KvPageTable : IDisposable
{
    private readonly object _gate = new();
    private List<KvPageLease> _pages;
    private bool _disposed;

    public KvPageTable()
        : this([])
    {
    }

    private KvPageTable(IEnumerable<KvPageLease> acquiredPages)
    {
        _pages = [.. acquiredPages];
    }

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

    public void Append(long pageId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            _pages.Add(new KvPageLease(pageId));
        }
    }

    public KvPageTable Fork()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return new KvPageTable(_pages.Select(static page => page.Acquire()));
        }
    }

    public KvSnapshot Snapshot(SequenceId sequenceId, long version, int position)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var acquiredPages = _pages.Select(static page => page.Acquire()).ToArray();
            return new KvSnapshot(sequenceId, version, position, acquiredPages);
        }
    }

    public static KvPageTable Restore(KvSnapshot snapshot) =>
        new(snapshot.AcquirePages());

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
