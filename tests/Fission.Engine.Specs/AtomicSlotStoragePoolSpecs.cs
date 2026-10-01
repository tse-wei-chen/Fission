using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class AtomicSlotStoragePoolSpecs
{
    private static Task? _runTask;

    [ModuleInitializer]
    internal static void Start()
    {
        _runTask = Task.Run(RunAsync);
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => CompleteBeforeExit();
    }

    private static void CompleteBeforeExit()
    {
        var runTask = _runTask;
        if (runTask is null)
        {
            return;
        }

        try
        {
            runTask.GetAwaiter().GetResult();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Atomic slot storage pool spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        VerifyIncompleteSubmissionReturnsStorage();
        await VerifyActorRetirementDelaysStorageReturnAsync();
        await VerifyDisposedCreditWaitKeepsStorageAliveAsync();
    }

    private static void VerifyIncompleteSubmissionReturnsStorage()
    {
        var submission = ContinuousBatchExecutor.BeginAtomicSubmission(2);
        Require(
            !submission.SlotStorageReturned,
            "Fresh atomic submissions must own their rented slot storage.");

        submission.Dispose();

        Require(
            submission.SlotStorageReturned,
            "An incomplete submission with no actor owner must return its slot storage on dispose.");
    }

    private static async Task VerifyActorRetirementDelaysStorageReturnAsync()
    {
        var backend = new BlockingSecondDecodeBackend(
            new DeviceId("cpu:atomic-slot-actor-retirement"));
        await using var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 2,
            maxBatchSize: 2);
        var model = new ModelId("atomic-slot-actor-retirement-model");
        var submission = ContinuousBatchExecutor.BeginAtomicSubmission(2);

        Task<BackendStepResult> first;
        using (submission.EnterSlot(0))
        {
            first = executor.SubmitDecodeAsync(
                    new DecodeItem(SequenceId.New(), model, Position: 1))
                .AsTask();
        }

        Task<BackendStepResult> second;
        using (submission.EnterSlot(1))
        {
            second = executor.SubmitDecodeAsync(
                    new DecodeItem(SequenceId.New(), model, Position: 2))
                .AsTask();
        }

        await backend.BatchStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        submission.Dispose();

        Require(
            !submission.SlotStorageReturned,
            "Caller retirement must not return slot storage while the actor still owns the atomic batch.");

        backend.AllowBatch();
        await Task.WhenAll(first, second).WaitAsync(TimeSpan.FromSeconds(5));
        await WaitForReturnedAsync(
            submission,
            "Slot storage must return after both caller and actor ownership retire.");
    }

    private static async Task VerifyDisposedCreditWaitKeepsStorageAliveAsync()
    {
        var backend = new BlockingFirstDecodeBackend(
            new DeviceId("cpu:atomic-slot-credit-wait"));
        await using var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 1,
            maxBatchSize: 1);
        var model = new ModelId("atomic-slot-credit-wait-model");

        var occupying = executor.SubmitDecodeAsync(
                new DecodeItem(SequenceId.New(), model, Position: 3))
            .AsTask();
        await backend.FirstDecodeStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));

        var submission = ContinuousBatchExecutor.BeginAtomicSubmission(1);
        Task<BackendStepResult> waiting;
        using (submission.EnterSlot(0))
        {
            waiting = executor.SubmitDecodeAsync(
                    new DecodeItem(SequenceId.New(), model, Position: 4))
                .AsTask();
        }

        submission.Dispose();
        Require(
            !submission.SlotStorageReturned,
            "A disposed final registration waiting for credits still owns actor-side cleanup state.");

        backend.AllowFirstDecode();
        await occupying.WaitAsync(TimeSpan.FromSeconds(5));

        var failed = false;
        try
        {
            await waiting.WaitAsync(TimeSpan.FromSeconds(5));
        }
        catch (InvalidOperationException exception) when (
            exception.Message.Contains("Atomic inference submission", StringComparison.Ordinal))
        {
            failed = true;
        }

        Require(
            failed,
            "The disposed atomic registration must observe the submission abort after credits become available.");
        await WaitForReturnedAsync(
            submission,
            "The aborted credit waiter must return slot storage only after its cleanup releases actor ownership.");
    }

    private static async Task WaitForReturnedAsync(
        ContinuousBatchExecutor.AtomicSubmissionBatch submission,
        string message)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (DateTime.UtcNow < deadline)
        {
            if (submission.SlotStorageReturned)
            {
                return;
            }

            await Task.Delay(1).ConfigureAwait(false);
        }

        throw new InvalidOperationException(message);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class BlockingSecondDecodeBackend(DeviceId device) : IInferenceBackend
    {
        private readonly TaskCompletionSource<bool> _batchStarted =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<bool> _allowBatch =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private bool _initialized;

        public string Name => "blocking-second-decode-slot-storage-spec";
        public DeviceId Device { get; } = device;
        internal TaskCompletionSource<bool> BatchStarted => _batchStarted;

        internal void AllowBatch() => _allowBatch.TrySetResult(true);

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _initialized = true;
            return ValueTask.CompletedTask;
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public async ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            _batchStarted.TrySetResult(true);
            await _allowBatch.Task.ConfigureAwait(false);
            return batch.Items.Select(static item =>
                new BackendStepResult(item.SequenceId, TokenId: 300 + item.Position)).ToArray();
        }

        public ValueTask DisposeAsync()
        {
            _allowBatch.TrySetResult(true);
            _initialized = false;
            return ValueTask.CompletedTask;
        }

        private void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException("Backend is not initialized.");
            }
        }
    }

    private sealed class BlockingFirstDecodeBackend(DeviceId device) : IInferenceBackend
    {
        private readonly TaskCompletionSource<bool> _firstDecodeStarted =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<bool> _allowFirstDecode =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _decodeCalls;
        private bool _initialized;

        public string Name => "blocking-first-decode-slot-storage-spec";
        public DeviceId Device { get; } = device;
        internal TaskCompletionSource<bool> FirstDecodeStarted => _firstDecodeStarted;

        internal void AllowFirstDecode() => _allowFirstDecode.TrySetResult(true);

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _initialized = true;
            return ValueTask.CompletedTask;
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public async ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            var call = Interlocked.Increment(ref _decodeCalls);
            if (call == 1)
            {
                _firstDecodeStarted.TrySetResult(true);
                await _allowFirstDecode.Task.ConfigureAwait(false);
            }

            return batch.Items.Select(static item =>
                new BackendStepResult(item.SequenceId, TokenId: 400 + item.Position)).ToArray();
        }

        public ValueTask DisposeAsync()
        {
            _allowFirstDecode.TrySetResult(true);
            _initialized = false;
            return ValueTask.CompletedTask;
        }

        private void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException("Backend is not initialized.");
            }
        }
    }
}
