namespace Fission.Runtime.Execution;

/// <summary>
/// Weighted asynchronous capacity gate for device inference work. Uncontended
/// acquisitions deduct their full weight in one critical section. Contended
/// acquisitions queue FIFO so a multi-item atomic envelope cannot partially
/// consume capacity or be bypassed by smaller later work. Credits are returned
/// only when the associated device work reaches a terminal state.
/// </summary>
internal sealed class InferenceCreditGate : IDisposable
{
    private readonly object _gate = new();
    private int _available;
    private Waiter? _head;
    private Waiter? _tail;
    private int _disposed;

    public InferenceCreditGate(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        Capacity = capacity;
        _available = capacity;
    }

    public int Capacity { get; }

    public ValueTask<Lease> AcquireAsync(
        int count,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);

        if (count > Capacity)
        {
            throw new InvalidOperationException(
                $"Inference work requires {count} device credit(s), but the configured capacity is {Capacity}.");
        }

        cancellationToken.ThrowIfCancellationRequested();

        Waiter waiter;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            if (_head is null && count <= _available)
            {
                _available -= count;
                return new ValueTask<Lease>(new Lease(this, count));
            }

            waiter = new Waiter(this, count, cancellationToken);
            EnqueueLocked(waiter);
        }

        waiter.AttachCancellation();
        return new ValueTask<Lease>(waiter.Completion.Task);
    }

    private void EnqueueLocked(Waiter waiter)
    {
        waiter.IsQueued = true;
        waiter.Previous = _tail;
        waiter.Next = null;

        if (_tail is null)
        {
            _head = waiter;
        }
        else
        {
            _tail.Next = waiter;
        }

        _tail = waiter;
    }

    private void RemoveLocked(Waiter waiter)
    {
        if (!waiter.IsQueued)
        {
            return;
        }

        if (waiter.Previous is null)
        {
            _head = waiter.Next;
        }
        else
        {
            waiter.Previous.Next = waiter.Next;
        }

        if (waiter.Next is null)
        {
            _tail = waiter.Previous;
        }
        else
        {
            waiter.Next.Previous = waiter.Previous;
        }

        waiter.Previous = null;
        waiter.Next = null;
        waiter.IsQueued = false;
    }

    private List<Waiter>? DrainWaitersLocked()
    {
        List<Waiter>? granted = null;

        while (_head is { } waiter && waiter.Count <= _available)
        {
            RemoveLocked(waiter);
            _available -= waiter.Count;
            (granted ??= new List<Waiter>()).Add(waiter);
        }

        return granted;
    }

    private void Cancel(Waiter waiter)
    {
        List<Waiter>? granted = null;
        var canceled = false;

        lock (_gate)
        {
            if (waiter.IsQueued)
            {
                RemoveLocked(waiter);
                canceled = true;

                if (Volatile.Read(ref _disposed) == 0)
                {
                    granted = DrainWaitersLocked();
                }
            }
        }

        if (!canceled)
        {
            return;
        }

        waiter.Completion.TrySetCanceled(waiter.CancellationToken);
        CompleteGranted(granted);
    }

    private void Release(int count)
    {
        List<Waiter>? granted;

        lock (_gate)
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                return;
            }

            var available = checked(_available + count);
            if (available > Capacity)
            {
                throw new InvalidOperationException(
                    $"Inference credit gate release would exceed configured capacity {Capacity}.");
            }

            _available = available;
            granted = DrainWaitersLocked();
        }

        CompleteGranted(granted);
    }

    private void CompleteGranted(List<Waiter>? granted)
    {
        if (granted is null)
        {
            return;
        }

        foreach (var waiter in granted)
        {
            waiter.DisposeCancellationRegistration();
            waiter.Completion.TrySetResult(new Lease(this, waiter.Count));
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        List<Waiter>? pending = null;
        lock (_gate)
        {
            while (_head is { } waiter)
            {
                RemoveLocked(waiter);
                (pending ??= new List<Waiter>()).Add(waiter);
            }
        }

        if (pending is null)
        {
            return;
        }

        foreach (var waiter in pending)
        {
            waiter.DisposeCancellationRegistration();
            waiter.Completion.TrySetException(
                new ObjectDisposedException(nameof(InferenceCreditGate)));
        }
    }

    private sealed class Waiter
    {
        private readonly InferenceCreditGate _owner;
        private CancellationTokenRegistration _cancellationRegistration;
        private int _cancellationAttached;

        public Waiter(
            InferenceCreditGate owner,
            int count,
            CancellationToken cancellationToken)
        {
            _owner = owner;
            Count = count;
            CancellationToken = cancellationToken;
            Completion = new TaskCompletionSource<Lease>(
                TaskCreationOptions.RunContinuationsAsynchronously);
        }

        public int Count { get; }
        public CancellationToken CancellationToken { get; }
        public TaskCompletionSource<Lease> Completion { get; }
        public Waiter? Previous { get; set; }
        public Waiter? Next { get; set; }
        public bool IsQueued { get; set; }

        public void AttachCancellation()
        {
            if (!CancellationToken.CanBeCanceled)
            {
                return;
            }

            var registration = CancellationToken.Register(
                static state =>
                {
                    var waiter = (Waiter)state!;
                    waiter._owner.Cancel(waiter);
                },
                this);

            _cancellationRegistration = registration;
            Volatile.Write(ref _cancellationAttached, 1);

            // Grant/disposal may have won the race before registration was
            // attached. In that case nobody else can dispose this late handle.
            if (Completion.Task.IsCompleted)
            {
                registration.Dispose();
            }
        }

        public void DisposeCancellationRegistration()
        {
            if (Volatile.Read(ref _cancellationAttached) != 0)
            {
                _cancellationRegistration.Dispose();
            }
        }
    }

    internal sealed class Lease : IDisposable
    {
        private InferenceCreditGate? _owner;
        private readonly int _count;

        internal Lease(InferenceCreditGate owner, int count)
        {
            _owner = owner;
            _count = count;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?.Release(_count);
        }
    }
}
