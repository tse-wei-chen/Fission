using Fission.Abstractions;

namespace Fission.Runtime.Kv;

public sealed class KvSnapshot : IDisposable
{
    private readonly IReadOnlyList<KvPageLease> _pages;
    private int _disposed;

    internal KvSnapshot(SequenceId sequenceId, long version, IReadOnlyList<KvPageLease> acquiredPages)
    {
        Id = KvSnapshotId.New();
        SequenceId = sequenceId;
        Version = version;
        _pages = acquiredPages;
    }

    public KvSnapshotId Id { get; }
    public SequenceId SequenceId { get; }
    public long Version { get; }
    public IReadOnlyList<long> PageIds => _pages.Select(static page => page.PageId).ToArray();

    internal IReadOnlyList<KvPageLease> AcquirePages()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _pages.Select(static page => page.Acquire()).ToArray();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var page in _pages)
        {
            page.Release();
        }
    }
}
