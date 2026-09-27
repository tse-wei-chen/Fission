namespace Fission.Runtime.Execution;

/// <summary>
/// Bounds concurrent physical migration pressure independently from inference
/// queue capacity. A lease reserves both one transfer slot and its estimated
/// in-flight bytes until the full migration transaction (including rollback)
/// finishes.
/// </summary>
public sealed class SequenceMigrationAdmissionController : IDisposable
{
    private readonly object _gate = new();
    private readonly long _maxInflightBytes;
    private readonly SemaphoreSlim _transferSlots;
    private TaskCompletionSource<bool> _capacityChanged = CreateSignal();
    private long _inflightBytes;
    private int _activeTransfers;
    private int _disposed;

    public SequenceMigrationAdmissionController(
        long maxInflightBytes,
        int maxConcurrentTransfers)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxInflightBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxConcurrentTransfers);

        _maxInflightBytes = maxInflightBytes;
        _transferSlots = new SemaphoreSlim(maxConcurrentTransfers, maxConcurrentTransfers);
        MaxConcurrentTransfers = maxConcurrentTransfers;
    }

    public long MaxInflightBytes => _maxInflightBytes;
    public int MaxConcurrentTransfers { get; }

    public long InflightBytes
    {
        get
        {
            lock (_gate)
            {
                return _inflightBytes;
            }
        }
    }

    public int ActiveTransfers
    {
        get
        {
            lock (_gate)
            {
                return _activeTransfers;
            }
        }
    }

    public async ValueTask<Lease> AcquireAsync(
        long estimatedBytes,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(estimatedBytes);

        if (estimatedBytes > _maxInflightBytes)
        {
            throw new InvalidOperationException(
                $"Migration needs {estimatedBytes} byte(s), but admission allows at most {_maxInflightBytes} in-flight byte(s).");
        }

        await _transferSlots.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            while (true)
            {
                Task capacityChanged;
                lock (_gate)
                {
                    ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                    if (_inflightBytes <= _maxInflightBytes - estimatedBytes)
                    {
                        _inflightBytes += estimatedBytes;
                        _activeTransfers++;
                        return new Lease(this, estimatedBytes);
                    }

                    capacityChanged = _capacityChanged.Task;
                }

                await capacityChanged.WaitAsync(cancellationToken).ConfigureAwait(false);
            }
        }
        catch
        {
            _transferSlots.Release();
            throw;
        }
    }

    private void Release(long estimatedBytes)
    {
        TaskCompletionSource<bool> signal;
        lock (_gate)
        {
            _inflightBytes = checked(_inflightBytes - estimatedBytes);
            _activeTransfers = checked(_activeTransfers - 1);
            signal = _capacityChanged;
            _capacityChanged = CreateSignal();
        }

        _transferSlots.Release();
        signal.TrySetResult(true);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        TaskCompletionSource<bool> signal;
        lock (_gate)
        {
            signal = _capacityChanged;
            _capacityChanged = CreateSignal();
        }

        signal.TrySetResult(true);
        _transferSlots.Dispose();
    }

    private static TaskCompletionSource<bool> CreateSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    public sealed class Lease : IDisposable
    {
        private SequenceMigrationAdmissionController? _owner;
        private readonly long _estimatedBytes;

        internal Lease(
            SequenceMigrationAdmissionController owner,
            long estimatedBytes)
        {
            _owner = owner;
            _estimatedBytes = estimatedBytes;
        }

        public long EstimatedBytes => _estimatedBytes;

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?.Release(_estimatedBytes);
        }
    }
}
