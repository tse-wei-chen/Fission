using Fission.Abstractions;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Reference-counted ownership graph for immutable decoder state versions.
///
/// Each sequence or snapshot owns one reference to a state version. Snapshot and
/// fork acquire the current reference rather than cloning the physical state.
/// Replacing a sequence installs a new immutable version and releases only that
/// sequence's reference to the previous version. The payload is disposed exactly
/// once when the final sequence/snapshot reference disappears.
///
/// The device actor already serializes backend transaction calls, but this store
/// also protects its maps so diagnostics and disposal remain deterministic.
/// </summary>
public sealed class DecoderStateStore<TState> : IDisposable
    where TState : class, IDisposable
{
    private readonly object _gate = new();
    private readonly Dictionary<SequenceId, SharedState> _sequences = new();
    private readonly Dictionary<KvSnapshotId, SharedState> _snapshots = new();
    private bool _disposed;

    public int SequenceCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return _sequences.Count;
            }
        }
    }

    public int SnapshotCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return _snapshots.Count;
            }
        }
    }

    /// <summary>
    /// Adds a sequence and transfers ownership of <paramref name="state"/> to
    /// the store only when the add succeeds. On failure the caller still owns it.
    /// </summary>
    public void AddSequence(SequenceId sequenceId, TState state)
    {
        ArgumentNullException.ThrowIfNull(state);

        lock (_gate)
        {
            ThrowIfDisposed();
            if (_sequences.ContainsKey(sequenceId))
            {
                throw new InvalidOperationException(
                    $"Decoder state already exists for sequence {sequenceId}.");
            }

            _sequences.Add(sequenceId, new SharedState(state));
        }
    }

    public bool TryGetSequence(SequenceId sequenceId, out TState? state)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_sequences.TryGetValue(sequenceId, out var shared))
            {
                state = shared.Value;
                return true;
            }

            state = null;
            return false;
        }
    }

    public TState GetSequence(SequenceId sequenceId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return GetSequenceShared(sequenceId).Value;
        }
    }

    /// <summary>
    /// Installs a new immutable state version for a sequence. Ownership of the
    /// new state transfers to the store only when the sequence exists.
    /// </summary>
    public void ReplaceSequence(SequenceId sequenceId, TState state)
    {
        ArgumentNullException.ThrowIfNull(state);

        SharedState previous;
        lock (_gate)
        {
            ThrowIfDisposed();
            if (!_sequences.TryGetValue(sequenceId, out previous!))
            {
                throw new KeyNotFoundException(
                    $"Decoder state does not exist for sequence {sequenceId}.");
            }

            _sequences[sequenceId] = new SharedState(state);
        }

        previous.Release();
    }

    public void SnapshotSequence(SequenceId sequenceId, KvSnapshotId snapshotId)
    {
        SharedState acquired;
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_snapshots.ContainsKey(snapshotId))
            {
                throw new InvalidOperationException(
                    $"Decoder snapshot state already exists for snapshot {snapshotId}.");
            }

            acquired = GetSequenceShared(sequenceId).Acquire();
            _snapshots.Add(snapshotId, acquired);
        }
    }

    /// <summary>
    /// Forks all branches onto the parent's current immutable state version.
    /// No physical state copy occurs. If registration fails, all references
    /// acquired by this call are rolled back.
    /// </summary>
    public void ForkSequence(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds)
    {
        ArgumentNullException.ThrowIfNull(branchSequenceIds);
        if (branchSequenceIds.Count == 0)
        {
            throw new ArgumentException(
                "At least one branch sequence id is required.",
                nameof(branchSequenceIds));
        }

        lock (_gate)
        {
            ThrowIfDisposed();
            var distinctBranches = new HashSet<SequenceId>();
            foreach (var branchId in branchSequenceIds)
            {
                if (branchId == parentSequenceId)
                {
                    throw new InvalidOperationException(
                        "A decoder branch id cannot equal its parent sequence id.");
                }

                if (!distinctBranches.Add(branchId))
                {
                    throw new InvalidOperationException(
                        $"Duplicate decoder branch id {branchId}.");
                }

                if (_sequences.ContainsKey(branchId))
                {
                    throw new InvalidOperationException(
                        $"Decoder state already exists for branch sequence {branchId}.");
                }
            }

            var parent = GetSequenceShared(parentSequenceId);
            var added = new List<SequenceId>(branchSequenceIds.Count);
            try
            {
                foreach (var branchId in branchSequenceIds)
                {
                    _sequences.Add(branchId, parent.Acquire());
                    added.Add(branchId);
                }
            }
            catch
            {
                foreach (var branchId in added)
                {
                    var shared = _sequences[branchId];
                    _sequences.Remove(branchId);
                    shared.Release();
                }

                throw;
            }
        }
    }

    public void RestoreSequence(SequenceId sequenceId, KvSnapshotId snapshotId)
    {
        SharedState previous;
        lock (_gate)
        {
            ThrowIfDisposed();
            if (!_sequences.TryGetValue(sequenceId, out previous!))
            {
                throw new KeyNotFoundException(
                    $"Decoder state does not exist for sequence {sequenceId}.");
            }

            if (!_snapshots.TryGetValue(snapshotId, out var snapshot))
            {
                throw new KeyNotFoundException(
                    $"Decoder snapshot state does not exist for snapshot {snapshotId}.");
            }

            _sequences[sequenceId] = snapshot.Acquire();
        }

        previous.Release();
    }

    public bool ReleaseSequence(SequenceId sequenceId)
    {
        SharedState? removed;
        lock (_gate)
        {
            ThrowIfDisposed();
            if (!_sequences.Remove(sequenceId, out removed))
            {
                return false;
            }
        }

        removed.Release();
        return true;
    }

    public bool ReleaseSnapshot(KvSnapshotId snapshotId)
    {
        SharedState? removed;
        lock (_gate)
        {
            ThrowIfDisposed();
            if (!_snapshots.Remove(snapshotId, out removed))
            {
                return false;
            }
        }

        removed.Release();
        return true;
    }

    public void Dispose()
    {
        SharedState[] owned;
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            owned = _sequences.Values.Concat(_snapshots.Values).ToArray();
            _sequences.Clear();
            _snapshots.Clear();
        }

        foreach (var shared in owned)
        {
            shared.Release();
        }
    }

    private SharedState GetSequenceShared(SequenceId sequenceId)
    {
        if (_sequences.TryGetValue(sequenceId, out var state))
        {
            return state;
        }

        throw new KeyNotFoundException(
            $"Decoder state does not exist for sequence {sequenceId}.");
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(_disposed, this);

    private sealed class SharedState
    {
        private int _referenceCount;
        private int _disposed;

        public SharedState(TState value)
        {
            Value = value;
            _referenceCount = 1;
        }

        public TState Value { get; }

        public SharedState Acquire()
        {
            while (true)
            {
                var current = Volatile.Read(ref _referenceCount);
                if (current <= 0)
                {
                    throw new ObjectDisposedException(nameof(SharedState));
                }

                if (Interlocked.CompareExchange(
                        ref _referenceCount,
                        checked(current + 1),
                        current) == current)
                {
                    return this;
                }
            }
        }

        public void Release()
        {
            var remaining = Interlocked.Decrement(ref _referenceCount);
            if (remaining < 0)
            {
                throw new InvalidOperationException(
                    "Decoder state reference count dropped below zero.");
            }

            if (remaining == 0 && Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                Value.Dispose();
            }
        }
    }
}
