using System.Runtime.ExceptionServices;
using System.Threading.Channels;
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

        await backend.InitializeAsync(cancellationToken).ConfigureAwait(false);
        return new ContinuousBatchExecutor(backend, capacity, maxBatchSize);
    }

    internal static AtomicSubmissionBatch BeginAtomicSubmission(int itemCount) =>
        new(itemCount);

    public ValueTask<BackendStepResult> SubmitPrefillAsync(
        PrefillItem item,
        CancellationToken cancellationToken = default) =>
        SubmitInferenceAsync(new PendingPrefill(item), cancellationToken);

    public ValueTask<BackendStepResult> SubmitDecodeAsync(
        DecodeItem item,
        CancellationToken cancellationToken = default) =>
        SubmitInferenceAsync(new PendingDecode(item), cancellationToken);

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

            try
            {
                await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
            }
            catch
            {
                work.ReleaseCredits();
                throw;
            }
        }

        return await work.Completion.Task.ConfigureAwait(false);
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

        try
        {
            while (await reader.WaitToReadAsync().ConfigureAwait(false))
            {
                var batch = new List<PendingWork>(_maxBatchSize);
                while (batch.Count < _maxBatchSize && reader.TryRead(out var work))
                {
                    batch.Add(work);
                }

                try
                {
                    await ExecuteBatchAsync(batch).ConfigureAwait(false);
                }
                catch (Exception exception)
                {
                    foreach (var work in batch)
                    {
                        work.Fail(exception);
                    }

                    throw;
                }
            }
        }
        catch (Exception exception)
        {
            _queue.Writer.TryComplete(exception);
            while (reader.TryRead(out var work))
            {
                work.Fail(exception);
            }

            throw;
        }
    }

    private async Task ExecuteBatchAsync(IReadOnlyList<PendingWork> batch)
    {
        var inferenceSegment = new List<PendingInference>(_maxBatchSize);

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
                case PendingInferenceEnvelope envelope:
                    try
                    {
                        await ExecuteInferenceSegmentAsync(envelope.Items).ConfigureAwait(false);
                    }
                    finally
                    {
                        envelope.ReleaseCredits();
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
                    var work = new PendingPrefill[count];
                    var items = new PrefillItem[count];
                    for (var offset = 0; offset < count; offset++)
                    {
                        var pending = (PendingPrefill)segment[index + offset];
                        work[offset] = pending;
                        items[offset] = pending.Item;
                    }

                    var results = await _backend.PrefillAsync(
                        new PrefillBatch(items)).ConfigureAwait(false);
                    Complete(work, results);
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
                    var work = new PendingDecode[count];
                    var items = new DecodeItem[count];
                    for (var offset = 0; offset < count; offset++)
                    {
                        var pending = (PendingDecode)segment[index + offset];
                        work[offset] = pending;
                        items[offset] = pending.Item;
                    }

                    var results = await _backend.DecodeAsync(
                        new DecodeBatch(items)).ConfigureAwait(false);
                    Complete(work, results);
                    index = end;
                    break;
                }

                default:
                    throw new InvalidOperationException(
                        $"Unsupported inference work type {segment[index].GetType().Name}.");
            }
        }
    }

    private static void Complete<TWork>(
        IReadOnlyList<TWork> work,
        IReadOnlyList<BackendStepResult> results)
        where TWork : PendingInference
    {
        if (work.Count != results.Count)
        {
            throw new InvalidOperationException(
                $"Backend returned {results.Count} results for {work.Count} work items.");
        }

        for (var i = 0; i < work.Count; i++)
        {
            var expectedSequenceId = work[i] switch
            {
                PendingPrefill prefill => prefill.Item.SequenceId,
                PendingDecode decode => decode.Item.SequenceId,
                _ => throw new InvalidOperationException(
                    $"Unsupported inference work type {work[i].GetType().Name}.")
            };

            if (results[i].SequenceId != expectedSequenceId)
            {
                throw new InvalidOperationException(
                    $"Backend result at index {i} belongs to sequence {results[i].SequenceId}, " +
                    $"but the corresponding work item belongs to {expectedSequenceId}.");
            }
        }

        for (var i = 0; i < work.Count; i++)
        {
            work[i].Complete(results[i]);
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

    internal sealed class AtomicSubmissionBatch : IDisposable
    {
        private readonly object _gate = new();
        private readonly PendingInference?[] _slots;
        private ContinuousBatchExecutor? _executor;
        private Exception? _failure;
        private int _registeredCount;
        private bool _enqueued;
        private int _disposed;

        internal AtomicSubmissionBatch(int itemCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(itemCount);
            _slots = new PendingInference[itemCount];
        }

        internal IDisposable EnterSlot(int index)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            if (index >= _slots.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            var previous = AmbientAtomicSlot.Value;
            AmbientAtomicSlot.Value = new AtomicSubmissionSlot(this, index);
            return new AtomicSlotLease(previous);
        }

        internal async ValueTask RegisterAsync(
            int index,
            ContinuousBatchExecutor executor,
            PendingInference work,
            CancellationToken cancellationToken)
        {
            PendingInferenceEnvelope? envelope = null;

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

                _slots[index] = work;
                _registeredCount++;
                if (_registeredCount == _slots.Length)
                {
                    var items = new PendingInference[_slots.Length];
                    for (var slot = 0; slot < _slots.Length; slot++)
                    {
                        items[slot] = _slots[slot]!;
                    }

                    envelope = new PendingInferenceEnvelope(items);
                }
            }

            if (envelope is null)
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

                envelope.AttachCredits(credits);
                await executor._queue.Writer.WriteAsync(envelope, cancellationToken)
                    .ConfigureAwait(false);

                lock (_gate)
                {
                    _enqueued = true;
                }
            }
            catch (Exception exception)
            {
                envelope.ReleaseCredits();
                Abort(exception);
                throw;
            }
        }

        internal void Abort(Exception exception)
        {
            ArgumentNullException.ThrowIfNull(exception);
            PendingInference[] registered;

            lock (_gate)
            {
                if (_failure is not null)
                {
                    return;
                }

                _failure = exception;
                registered = _slots
                    .Where(static item => item is not null)
                    .Select(static item => item!)
                    .ToArray();
            }

            foreach (var work in registered)
            {
                work.Fail(exception);
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            if (!_enqueued && _failure is null)
            {
                Abort(new InvalidOperationException(
                    "Atomic inference submission ended before every slot registered work."));
            }
        }
    }

    private sealed class AtomicSlotLease(AtomicSubmissionSlot? previous) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                AmbientAtomicSlot.Value = previous;
            }
        }
    }

    private sealed record AtomicSubmissionSlot(
        AtomicSubmissionBatch Batch,
        int Index);

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

    internal abstract class PendingInference : PendingWork
    {
        protected PendingInference()
        {
            Completion = new TaskCompletionSource<BackendStepResult>(
                TaskCreationOptions.RunContinuationsAsynchronously);
        }

        public TaskCompletionSource<BackendStepResult> Completion { get; }

        internal void Complete(BackendStepResult result)
        {
            Completion.TrySetResult(result);
            ReleaseCredits();
        }

        public override void Fail(Exception exception)
        {
            Completion.TrySetException(exception);
            ReleaseCredits();
        }
    }

    private sealed class PendingPrefill(PrefillItem item) : PendingInference
    {
        public PrefillItem Item { get; } = item;
    }

    private sealed class PendingDecode(DecodeItem item) : PendingInference
    {
        public DecodeItem Item { get; } = item;
    }

    private sealed class PendingInferenceEnvelope(PendingInference[] items) : PendingWork
    {
        public IReadOnlyList<PendingInference> Items { get; } = items;

        public override void Fail(Exception exception)
        {
            foreach (var item in Items)
            {
                item.Fail(exception);
            }

            ReleaseCredits();
        }
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
