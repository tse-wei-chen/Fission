using System.Collections;
using System.Runtime.ExceptionServices;
using System.Threading.Channels;
using System.Threading.Tasks.Sources;
using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

/// <summary>
/// Single-device actor that owns backend execution. Producers enqueue work;
/// one consumer drains currently available work into micro-batches and invokes
/// the backend serially. Stateful control operations are queue-order barriers:
/// inference before a control is flushed first, the control executes, then later
/// inference may proceed. Mixed prefill/decode inference preserves queue order;
/// only contiguous work of the same kind is coalesced into one backend batch.
/// Scheduler-selected work may additionally use an atomic submission envelope so
/// one logical scheduling batch reaches the actor with deterministic membership.
/// Inference capacity is accounted by item credits rather than channel entries,
/// so an N-item envelope consumes the same bounded capacity as N scalar submits.
/// Caller cancellation may withdraw work only before queue acceptance; once a
/// device operation is accepted, its terminal result is observed so runtime and
/// backend state advance at the same transaction boundary. Transaction-control
/// failures are completed on that control without poisoning the actor; inference
/// execution failures remain actor-fatal because batch state may be ambiguous.
/// </summary>
public sealed partial class ContinuousBatchExecutor : IAsyncDisposable
{
    private static readonly AsyncLocal<AtomicSubmissionSlot?> AmbientAtomicSlot = new();

    private readonly IInferenceBackend _backend;
    private readonly Channel<PendingWork> _queue;
    private readonly InferenceCreditGate _inferenceCredits;
    private readonly int _maxBatchSize;
    private readonly Task _pump;
    private int _disposed;

    private ContinuousBatchExecutor(
        IInferenceBackend backend,
        int capacity,
        int maxBatchSize)
    {
        _backend = backend;
        _maxBatchSize = maxBatchSize;
        _inferenceCredits = new InferenceCreditGate(capacity);
        _queue = Channel.CreateBounded<PendingWork>(new BoundedChannelOptions(capacity)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.Wait
        });
        _pump = Task.Run(PumpAsync);
    }

    public DeviceId Device => _backend.Device;
    public string BackendName => _backend.Name;
    internal int InferenceCapacity => _inferenceCredits.Capacity;
    internal bool SupportsTransactionalMigration => _backend is ISequenceMigrationBackend;

    public static async ValueTask<ContinuousBatchExecutor> CreateAsync(
        IInferenceBackend backend,
        int capacity = 4096,
        int maxBatchSize = 64,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBatchSize);

        try
        {
            await backend.InitializeAsync(cancellationToken).ConfigureAwait(false);
            return new ContinuousBatchExecutor(backend, capacity, maxBatchSize);
        }
        catch
        {
            await backend.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    internal static AtomicSubmissionBatch BeginAtomicSubmission(int itemCount) =>
        new(itemCount);

    public ValueTask<BackendStepResult> SubmitPrefillAsync(
        PrefillItem item,
        CancellationToken cancellationToken = default) =>
        SubmitInferenceAsync(RentPrefill(item), cancellationToken);

    public ValueTask<BackendStepResult> SubmitDecodeAsync(
        DecodeItem item,
        CancellationToken cancellationToken = default) =>
        SubmitInferenceAsync(RentDecode(item), cancellationToken);

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(
            new PendingSnapshot(sequenceId, snapshotId),
            cancellationToken);

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(
            new PendingFork(parentSequenceId, branchSequenceIds.ToArray()),
            cancellationToken);

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(
            new PendingRestore(sequenceId, snapshotId),
            cancellationToken);

    public ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(
            new PendingMigration(sequenceId, targetDevice),
            cancellationToken);

    internal async ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        var work = new PendingPrepareMigration(sequenceId, targetDevice);
        await SubmitControlAsync(work, cancellationToken).ConfigureAwait(false);
        return work.Transfer ?? throw new InvalidOperationException(
            $"Backend {BackendName} completed migration prepare without a transfer token.");
    }

    internal ValueTask ImportSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(new PendingImportMigration(transfer), cancellationToken);

    internal ValueTask CommitSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(new PendingCommitMigration(transfer), cancellationToken);

    internal ValueTask AbortSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(new PendingAbortMigration(transfer), cancellationToken);

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(new PendingReleaseSnapshot(snapshotId), cancellationToken);

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default) =>
        SubmitControlAsync(new PendingReleaseSequence(sequenceId), cancellationToken);

    private async ValueTask<BackendStepResult> SubmitInferenceAsync<TWork>(
        TWork work,
        CancellationToken cancellationToken)
        where TWork : PendingInference
    {
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            if (AmbientAtomicSlot.Value is { } slot)
            {
                await slot.Batch.RegisterAsync(
                        slot.Index,
                        this,
                        work,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
            else
            {
                var credits = await _inferenceCredits.AcquireAsync(1, cancellationToken)
                    .ConfigureAwait(false);
                work.AttachCredits(credits);
                work.RetainActorOwnership();

                try
                {
                    await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
                }
                catch
                {
                    work.ReleaseActorOwnership();
                    work.ReleaseCredits();
                    throw;
                }
            }
        }
        catch
        {
            work.ReleaseCredits();
            work.AbandonSubmission();
            throw;
        }

        return await work.WaitAsync().ConfigureAwait(false);
    }

    private async ValueTask SubmitControlAsync(
        PendingControl work,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
        await work.Completion.Task.ConfigureAwait(false);
    }

    private async Task PumpAsync()
    {
        var reader = _queue.Reader;
        var batch = new List<PendingWork>(_maxBatchSize);
        var inferenceSegment = new List<PendingInference>(_maxBatchSize);

        try
        {
            while (await reader.WaitToReadAsync().ConfigureAwait(false))
            {
                batch.Clear();
                while (batch.Count < _maxBatchSize && reader.TryRead(out var work))
                {
                    batch.Add(work);
                }

                try
                {
                    await ExecuteBatchAsync(batch, inferenceSegment).ConfigureAwait(false);
                }
                catch (Exception exception)
                {
                    foreach (var work in batch)
                    {
                        work.Fail(exception);
                    }

                    throw;
                }
                finally
                {
                    foreach (var work in batch)
                    {
                        ReleaseActorOwnership(work);
                    }
                }
            }
        }
        catch (Exception exception)
        {
            _queue.Writer.TryComplete(exception);
            while (reader.TryRead(out var work))
            {
                try
                {
                    work.Fail(exception);
                }
                finally
                {
                    ReleaseActorOwnership(work);
                }
            }

            throw;
        }
    }

    private static void ReleaseActorOwnership(PendingWork work)
    {
        switch (work)
        {
            case PendingInference inference:
                inference.ReleaseActorOwnership();
                break;

            case AtomicSubmissionBatch submission:
                submission.ReleaseActorOwnership();
                break;
        }
    }

    private async Task ExecuteBatchAsync(
        IReadOnlyList<PendingWork> batch,
        List<PendingInference> inferenceSegment)
    {
        inferenceSegment.Clear();

        foreach (var work in batch)
        {
            if (work is PendingInference inference)
            {
                inferenceSegment.Add(inference);
                continue;
            }

            await ExecuteInferenceSegmentAsync(inferenceSegment).ConfigureAwait(false);
            inferenceSegment.Clear();

            switch (work)
            {
                case AtomicSubmissionBatch submission:
                    try
                    {
                        await ExecuteInferenceSegmentAsync(submission).ConfigureAwait(false);
                    }
                    finally
                    {
                        submission.ReleaseCredits();
                    }
                    break;

                case PendingControl control:
                    try
                    {
                        await control.ExecuteAsync(_backend).ConfigureAwait(false);
                        control.Completion.TrySetResult(true);
                    }
                    catch (Exception exception)
                    {
                        control.Fail(exception);
                    }
                    break;

                default:
                    throw new InvalidOperationException(
                        $"Unsupported device work type {work.GetType().Name}.");
            }
        }

        await ExecuteInferenceSegmentAsync(inferenceSegment).ConfigureAwait(false);
        inferenceSegment.Clear();
    }

    private async Task ExecuteInferenceSegmentAsync(
        IReadOnlyList<PendingInference> segment)
    {
        var index = 0;
        while (index < segment.Count)
        {
            switch (segment[index])
            {
                case PendingPrefill:
                {
                    var end = index + 1;
                    while (end < segment.Count &&
                           end - index < _maxBatchSize &&
                           segment[end] is PendingPrefill)
                    {
                        end++;
                    }

                    var count = end - index;
                    IReadOnlyList<BackendStepResult> results;
                    try
                    {
                        results = await _backend.PrefillAsync(
                                PreparePrefillBatch(segment, index, count))
                            .ConfigureAwait(false);
                    }
                    finally
                    {
                        ClearPrefillBatch();
                    }

                    Complete(segment, index, count, results);
                    index = end;
                    break;
                }

                case PendingDecode:
                {
                    var end = index + 1;
                    while (end < segment.Count &&
                           end - index < _maxBatchSize &&
                           segment[end] is PendingDecode)
                    {
                        end++;
                    }

                    var count = end - index;
                    IReadOnlyList<BackendStepResult> results;
                    try
                    {
                        results = await _backend.DecodeAsync(
                                PrepareDecodeBatch(segment, index, count))
                            .ConfigureAwait(false);
                    }
                    finally
                    {
                        ClearDecodeBatch();
                    }

                    Complete(segment, index, count, results);
                    index = end;
                    break;
                }

                default:
                    throw new InvalidOperationException(
                        $"Unsupported inference work type {segment[index].GetType().Name}.");
            }
        }
    }

    private static void Complete(
        IReadOnlyList<PendingInference> segment,
        int start,
        int count,
        IReadOnlyList<BackendStepResult> results)
    {
        if (count != results.Count)
        {
            throw new InvalidOperationException(
                $"Backend returned {results.Count} results for {count} work items.");
        }

        for (var i = 0; i < count; i++)
        {
            var work = segment[start + i];
            var expectedSequenceId = work switch
            {
                PendingPrefill prefill => prefill.Item.SequenceId,
                PendingDecode decode => decode.Item.SequenceId,
                _ => throw new InvalidOperationException(
                    $"Unsupported inference work type {work.GetType().Name}.")
            };

            if (results[i].SequenceId != expectedSequenceId)
            {
                throw new InvalidOperationException(
                    $"Backend result at index {i} belongs to sequence {results[i].SequenceId}, " +
                    $"but the corresponding work item belongs to {expectedSequenceId}.");
            }
        }

        for (var i = 0; i < count; i++)
        {
            segment[start + i].Complete(results[i]);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _queue.Writer.TryComplete();

        ExceptionDispatchInfo? pumpFailure = null;
        ExceptionDispatchInfo? backendDisposeFailure = null;

        try
        {
            await _pump.ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            pumpFailure = ExceptionDispatchInfo.Capture(exception);
        }

        try
        {
            await _backend.DisposeAsync().ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            backendDisposeFailure = ExceptionDispatchInfo.Capture(exception);
        }

        if (pumpFailure is not null && backendDisposeFailure is not null)
        {
            throw new AggregateException(
                "Device actor and backend disposal both failed.",
                new[]
                {
                    pumpFailure.SourceException,
                    backendDisposeFailure.SourceException
                });
        }

        if (pumpFailure is not null)
        {
            pumpFailure.Throw();
        }

        if (backendDisposeFailure is not null)
        {
            backendDisposeFailure.Throw();
        }
    }

    internal sealed class AtomicSubmissionBatch :
        PendingWork,
        IReadOnlyList<PendingInference>,
        IDisposable
    {
        private readonly object _gate = new();
        private readonly PooledReferenceSlots<PendingInference> _slots;
        private ContinuousBatchExecutor? _executor;
        private Exception? _failure;
        private int _registeredCount;
        private bool _enqueued;
        private bool _atomicOwnershipReleased;
        private int _actorOwned;
        private int _disposed;

        internal AtomicSubmissionBatch(int itemCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(itemCount);
            _slots = new PooledReferenceSlots<PendingInference>(itemCount);
        }

        public int Count => _slots.Length;
        internal bool SlotStorageReturned => _slots.IsReturned;

        public PendingInference this[int index] =>
            _slots[index] ?? throw new InvalidOperationException(
                $"Atomic inference envelope slot {index} was not registered.");

        internal IDisposable EnterSlot(int index)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            if (index >= _slots.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            var slot = new AtomicSubmissionSlot(this, index, AmbientAtomicSlot.Value);
            AmbientAtomicSlot.Value = slot;
            return slot;
        }

        internal async ValueTask RegisterAsync(
            int index,
            ContinuousBatchExecutor executor,
            PendingInference work,
            CancellationToken cancellationToken)
        {
            var shouldSubmit = false;

            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                if (_failure is not null)
                {
                    throw new InvalidOperationException(
                        "Atomic inference submission was already aborted.",
                        _failure);
                }

                if (_slots[index] is not null)
                {
                    throw new InvalidOperationException(
                        $"Atomic inference slot {index} was registered more than once.");
                }

                if (_executor is null)
                {
                    _executor = executor;
                }
                else if (!ReferenceEquals(_executor, executor))
                {
                    var mismatch = new InvalidOperationException(
                        "One atomic inference submission cannot target more than one device actor.");
                    Abort(mismatch);
                    throw mismatch;
                }

                work.RetainAtomicOwnership();
                _slots[index] = work;
                _registeredCount++;
                if (_registeredCount == _slots.Length)
                {
                    RetainActorOwnership();
                    shouldSubmit = true;
                }
            }

            if (!shouldSubmit)
            {
                return;
            }

            try
            {
                var credits = await executor._inferenceCredits
                    .AcquireAsync(_slots.Length, cancellationToken)
                    .ConfigureAwait(false);

                lock (_gate)
                {
                    if (_failure is not null)
                    {
                        credits.Dispose();
                        throw new InvalidOperationException(
                            "Atomic inference submission was aborted while waiting for device credits.",
                            _failure);
                    }
                }

                AttachCredits(credits);
                await executor._queue.Writer.WriteAsync(this, cancellationToken)
                    .ConfigureAwait(false);

                lock (_gate)
                {
                    _enqueued = true;
                }
            }
            catch (Exception exception)
            {
                ReleaseCredits();
                Abort(exception);
                ReleaseActorOwnership();
                throw;
            }
        }

        internal void Abort(Exception exception)
        {
            ArgumentNullException.ThrowIfNull(exception);

            lock (_gate)
            {
                if (_failure is not null)
                {
                    return;
                }

                _failure = exception;
                var releaseOwnership = !_atomicOwnershipReleased;
                _atomicOwnershipReleased = true;

                for (var index = 0; index < _slots.Length; index++)
                {
                    if (_slots[index] is not { } work)
                    {
                        continue;
                    }

                    work.Fail(exception);
                    if (releaseOwnership)
                    {
                        work.ReleaseAtomicOwnership();
                    }
                }
            }
        }

        internal void RetainActorOwnership()
        {
            var retained = 0;
            try
            {
                for (; retained < _slots.Length; retained++)
                {
                    this[retained].RetainActorOwnership();
                }

                Volatile.Write(ref _actorOwned, 1);
            }
            catch
            {
                for (var index = 0; index < retained; index++)
                {
                    this[index].ReleaseActorOwnership();
                }

                throw;
            }
        }

        internal void ReleaseActorOwnership()
        {
            for (var index = 0; index < _slots.Length; index++)
            {
                this[index].ReleaseActorOwnership();
            }

            Volatile.Write(ref _actorOwned, 0);
            TryReturnSlotStorage();
        }

        public IEnumerator<PendingInference> GetEnumerator()
        {
            for (var index = 0; index < _slots.Length; index++)
            {
                yield return this[index];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override void Fail(Exception exception)
        {
            Abort(exception);
            ReleaseCredits();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            lock (_gate)
            {
                Exception? incompleteFailure = null;
                if (!_enqueued && _failure is null)
                {
                    incompleteFailure = new InvalidOperationException(
                        "Atomic inference submission ended before every slot registered work.");
                    _failure = incompleteFailure;
                }

                var releaseOwnership = !_atomicOwnershipReleased;
                _atomicOwnershipReleased = true;

                if (incompleteFailure is not null)
                {
                    for (var index = 0; index < _slots.Length; index++)
                    {
                        _slots[index]?.Fail(incompleteFailure);
                    }
                }

                if (releaseOwnership)
                {
                    for (var index = 0; index < _slots.Length; index++)
                    {
                        _slots[index]?.ReleaseAtomicOwnership();
                    }
                }
            }

            TryReturnSlotStorage();
        }

        private void TryReturnSlotStorage()
        {
            if (Volatile.Read(ref _disposed) == 0 ||
                Volatile.Read(ref _actorOwned) != 0)
            {
                return;
            }

            lock (_gate)
            {
                if (Volatile.Read(ref _disposed) != 0 &&
                    Volatile.Read(ref _actorOwned) == 0)
                {
                    _slots.Return();
                }
            }
        }
    }

    private sealed class AtomicSubmissionSlot(
        AtomicSubmissionBatch batch,
        int index,
        AtomicSubmissionSlot? previous) : IDisposable
    {
        private int _disposed;

        internal AtomicSubmissionBatch Batch { get; } = batch;
        internal int Index { get; } = index;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                AmbientAtomicSlot.Value = previous;
            }
        }
    }

    internal abstract class PendingWork
    {
        private InferenceCreditGate.Lease? _credits;

        internal void AttachCredits(InferenceCreditGate.Lease credits)
        {
            ArgumentNullException.ThrowIfNull(credits);
            if (Interlocked.CompareExchange(ref _credits, credits, null) is not null)
            {
                credits.Dispose();
                throw new InvalidOperationException("Device work already owns inference credits.");
            }
        }

        internal void ReleaseCredits() =>
            Interlocked.Exchange(ref _credits, null)?.Dispose();

        public abstract void Fail(Exception exception);
    }

    internal abstract class PendingInference : PendingWork, IValueTaskSource<BackendStepResult>
    {
        private ManualResetValueTaskSourceCore<BackendStepResult> _completion;
        private ContinuousBatchExecutor? _poolOwner;
        private int _terminal;
        private int _consumerOwned;
        private int _actorOwned;
        private int _atomicOwned;
        private int _recycleReady;
        private int _returned;

        protected PendingInference()
        {
            _completion.RunContinuationsAsynchronously = true;
        }

        protected void InitializeForUse(ContinuousBatchExecutor owner)
        {
            ArgumentNullException.ThrowIfNull(owner);
            _completion.Reset();
            _poolOwner = owner;
            _terminal = 0;
            _consumerOwned = 1;
            _actorOwned = 0;
            _atomicOwned = 0;
            _recycleReady = 0;
            _returned = 0;
        }

        internal ValueTask<BackendStepResult> WaitAsync() =>
            new(this, _completion.Version);

        internal void RetainActorOwnership()
        {
            if (Interlocked.CompareExchange(ref _actorOwned, 1, 0) != 0)
            {
                throw new InvalidOperationException(
                    "Inference work already has actor ownership.");
            }
        }

        internal void ReleaseActorOwnership()
        {
            if (Interlocked.Exchange(ref _actorOwned, 0) != 0)
            {
                TryRecycle();
            }
        }

        internal void RetainAtomicOwnership()
        {
            if (Interlocked.CompareExchange(ref _atomicOwned, 1, 0) != 0)
            {
                throw new InvalidOperationException(
                    "Inference work already has atomic-submission ownership.");
            }
        }

        internal void ReleaseAtomicOwnership()
        {
            if (Interlocked.Exchange(ref _atomicOwned, 0) != 0)
            {
                TryRecycle();
            }
        }

        internal void AbandonSubmission()
        {
            Volatile.Write(ref _recycleReady, 1);
            ReleaseConsumerOwnership();
        }

        internal void Complete(BackendStepResult result)
        {
            if (Interlocked.CompareExchange(ref _terminal, 1, 0) == 0)
            {
                _completion.SetResult(result);
                Volatile.Write(ref _recycleReady, 1);
            }

            ReleaseCredits();
            TryRecycle();
        }

        public override void Fail(Exception exception)
        {
            ArgumentNullException.ThrowIfNull(exception);
            if (Interlocked.CompareExchange(ref _terminal, 1, 0) == 0)
            {
                _completion.SetException(exception);
                Volatile.Write(ref _recycleReady, 1);
            }

            ReleaseCredits();
            TryRecycle();
        }

        BackendStepResult IValueTaskSource<BackendStepResult>.GetResult(short token)
        {
            var ownsCurrentGeneration = token == _completion.Version;
            try
            {
                return _completion.GetResult(token);
            }
            finally
            {
                if (ownsCurrentGeneration)
                {
                    ReleaseConsumerOwnership();
                }
            }
        }

        ValueTaskSourceStatus IValueTaskSource<BackendStepResult>.GetStatus(short token) =>
            _completion.GetStatus(token);

        void IValueTaskSource<BackendStepResult>.OnCompleted(
            Action<object?> continuation,
            object? state,
            short token,
            ValueTaskSourceOnCompletedFlags flags) =>
            _completion.OnCompleted(continuation, state, token, flags);

        private void ReleaseConsumerOwnership()
        {
            if (Interlocked.Exchange(ref _consumerOwned, 0) != 0)
            {
                TryRecycle();
            }
        }

        private void TryRecycle()
        {
            if (Volatile.Read(ref _recycleReady) == 0 ||
                Volatile.Read(ref _consumerOwned) != 0 ||
                Volatile.Read(ref _actorOwned) != 0 ||
                Volatile.Read(ref _atomicOwned) != 0)
            {
                return;
            }

            if (Interlocked.CompareExchange(ref _returned, 1, 0) != 0)
            {
                return;
            }

            var owner = Interlocked.Exchange(ref _poolOwner, null);
            owner?.ReturnInferenceWorkToPool(this);
        }
    }

    private sealed class PendingPrefill : PendingInference
    {
        public PrefillItem Item { get; private set; }

        internal void Initialize(ContinuousBatchExecutor owner, PrefillItem item)
        {
            InitializeForUse(owner);
            Item = item;
        }

        internal void ClearItemForPool() => Item = default;
    }

    private sealed class PendingDecode : PendingInference
    {
        public DecodeItem Item { get; private set; }

        internal void Initialize(ContinuousBatchExecutor owner, DecodeItem item)
        {
            InitializeForUse(owner);
            Item = item;
        }

        internal void ClearItemForPool() => Item = default;
    }

    private abstract class PendingControl : PendingWork
    {
        protected PendingControl()
        {
            Completion = new TaskCompletionSource<bool>(
                TaskCreationOptions.RunContinuationsAsynchronously);
        }

        public TaskCompletionSource<bool> Completion { get; }
        public abstract ValueTask ExecuteAsync(IInferenceBackend backend);

        public override void Fail(Exception exception)
        {
            Completion.TrySetException(exception);
            ReleaseCredits();
        }
    }

    private sealed class PendingSnapshot(
        SequenceId sequenceId,
        KvSnapshotId snapshotId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.SnapshotSequenceAsync(sequenceId, snapshotId);
    }

    private sealed class PendingFork(
        SequenceId parentSequenceId,
        SequenceId[] branchSequenceIds) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.ForkSequenceAsync(parentSequenceId, branchSequenceIds);
    }

    private sealed class PendingRestore(
        SequenceId sequenceId,
        KvSnapshotId snapshotId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.RestoreSequenceAsync(sequenceId, snapshotId);
    }

    private sealed class PendingMigration(
        SequenceId sequenceId,
        DeviceId targetDevice) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.MigrateSequenceAsync(sequenceId, targetDevice);
    }

    private sealed class PendingPrepareMigration(
        SequenceId sequenceId,
        DeviceId targetDevice) : PendingControl
    {
        public SequenceMigrationTransfer? Transfer { get; private set; }

        public override async ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            var migration = RequireTransactionalMigration(backend);
            var transfer = await migration.PrepareSequenceMigrationAsync(
                    sequenceId,
                    targetDevice)
                .ConfigureAwait(false);

            if (transfer.SequenceId != sequenceId ||
                transfer.SourceDevice != backend.Device ||
                transfer.TargetDevice != targetDevice)
            {
                throw new InvalidOperationException(
                    $"Backend {backend.Name} returned migration transfer {transfer.TransactionId} " +
                    "with sequence or device identity that does not match the prepare request.");
            }

            Transfer = transfer;
        }
    }

    private sealed class PendingImportMigration(
        SequenceMigrationTransfer transfer) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            if (transfer.TargetDevice != backend.Device)
            {
                throw new InvalidOperationException(
                    $"Migration transfer {transfer.TransactionId} targets {transfer.TargetDevice}, " +
                    $"but import was submitted to actor {backend.Device}.");
            }

            return RequireTransactionalMigration(backend)
                .ImportSequenceMigrationAsync(transfer);
        }
    }

    private sealed class PendingCommitMigration(
        SequenceMigrationTransfer transfer) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            if (transfer.SourceDevice != backend.Device)
            {
                throw new InvalidOperationException(
                    $"Migration transfer {transfer.TransactionId} originates on {transfer.SourceDevice}, " +
                    $"but commit was submitted to actor {backend.Device}.");
            }

            return RequireTransactionalMigration(backend)
                .CommitSequenceMigrationAsync(transfer);
        }
    }

    private sealed class PendingAbortMigration(
        SequenceMigrationTransfer transfer) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            if (backend.Device != transfer.SourceDevice && backend.Device != transfer.TargetDevice)
            {
                throw new InvalidOperationException(
                    $"Migration transfer {transfer.TransactionId} belongs to " +
                    $"{transfer.SourceDevice}->{transfer.TargetDevice}, but abort was submitted to {backend.Device}.");
            }

            return RequireTransactionalMigration(backend)
                .AbortSequenceMigrationAsync(transfer);
        }
    }

    private sealed class PendingReleaseSnapshot(
        KvSnapshotId snapshotId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.ReleaseSnapshotAsync(snapshotId);
    }

    private sealed class PendingReleaseSequence(
        SequenceId sequenceId) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend) =>
            backend.ReleaseSequenceAsync(sequenceId);
    }

    private static ISequenceMigrationBackend RequireTransactionalMigration(
        IInferenceBackend backend) =>
        backend as ISequenceMigrationBackend ??
        throw new NotSupportedException(
            $"Backend '{backend.Name}' does not implement transactional sequence migration.");
}
