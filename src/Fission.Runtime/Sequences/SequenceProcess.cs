using Fission.Abstractions;
using Fission.Runtime.Kv;

namespace Fission.Runtime.Sequences;

public enum SequenceStatus
{
    Waiting,
    Prefilling,
    Decoding,
    Suspended,
    Finished,
    Cancelled
}

public sealed class SequenceProcess : IDisposable
{
    private readonly object _gate = new();
    private bool _disposed;

    public SequenceProcess(ModelId model, DeviceId device)
        : this(SequenceId.New(), model, device, new KvPageTable(), 0, 0)
    {
    }

    private SequenceProcess(
        SequenceId id,
        ModelId model,
        DeviceId device,
        KvPageTable kv,
        long version,
        int position)
    {
        Id = id;
        Model = model;
        Device = device;
        Kv = kv;
        Version = version;
        Position = position;
        Status = SequenceStatus.Waiting;
    }

    public SequenceId Id { get; }
    public ModelId Model { get; }
    public DeviceId Device { get; private set; }
    public SequenceStatus Status { get; private set; }
    public long Version { get; private set; }
    public int Position { get; private set; }
    public KvPageTable Kv { get; private set; }

    internal static SequenceProcess Create(
        SequenceId id,
        ModelId model,
        DeviceId device,
        KvPagePool pagePool) =>
        new(id, model, device, new KvPageTable(pagePool), 0, 0);

    public void TransitionTo(SequenceStatus next)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (!CanTransition(Status, next))
            {
                throw new InvalidOperationException($"Invalid sequence transition: {Status} -> {next}.");
            }

            Status = next;
            Version++;
        }
    }

    public KvSnapshot Snapshot()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return Kv.Snapshot(Id, Version, Position);
        }
    }

    public SequenceProcess Fork()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return new SequenceProcess(SequenceId.New(), Model, Device, Kv.Fork(), Version, Position)
            {
                Status = Status
            };
        }
    }

    internal void Restore(KvSnapshot snapshot)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var restored = KvPageTable.Restore(snapshot);
            Kv.Dispose();
            Kv = restored;
            Position = snapshot.Position;
            Version = Math.Max(Version, snapshot.Version) + 1;
        }
    }

    internal void RecordPrefill(int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokenCount);

        lock (_gate)
        {
            ThrowIfDisposed();
            Kv.Append();
            Position += tokenCount;
            Version++;
        }
    }

    internal void RecordDecode()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Kv.Append();
            Position++;
            Version++;
        }
    }

    public void MigrateTo(DeviceId target)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Device = target;
            Version++;
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            Kv.Dispose();
        }
    }

    private static bool CanTransition(SequenceStatus current, SequenceStatus next) =>
        (current, next) switch
        {
            (SequenceStatus.Waiting, SequenceStatus.Prefilling) => true,
            (SequenceStatus.Prefilling, SequenceStatus.Decoding) => true,
            (SequenceStatus.Prefilling, SequenceStatus.Suspended) => true,
            (SequenceStatus.Decoding, SequenceStatus.Suspended) => true,
            (SequenceStatus.Suspended, SequenceStatus.Prefilling) => true,
            (SequenceStatus.Suspended, SequenceStatus.Decoding) => true,
            (SequenceStatus.Decoding, SequenceStatus.Finished) => true,
            (_, SequenceStatus.Cancelled) when current is not SequenceStatus.Finished => true,
            _ => false
        };

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
