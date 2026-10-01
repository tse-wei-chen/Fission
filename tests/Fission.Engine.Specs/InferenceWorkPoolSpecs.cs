using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class InferenceWorkPoolSpecs
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
            Console.Error.WriteLine($"Inference work pool spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        await VerifyCancelledAdmissionReturnsWorkAsync();
        await VerifyAtomicOwnershipDelaysReuseAsync();
        await VerifyActorOwnershipOutlivesAtomicOwnerAsync();
    }

    private static async Task VerifyCancelledAdmissionReturnsWorkAsync()
    {
        var backend = new ImmediateBackend(new DeviceId("cpu:pool-cancel"));
        await using var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 2,
            maxBatchSize: 2);
        var model = new ModelId("pool-cancel-model");

        using (var cancellation = new CancellationTokenSource())
        {
            cancellation.Cancel();
            var cancelled = false;
            try
            {
                await executor.SubmitDecodeAsync(
                    new DecodeItem(SequenceId.New(), model, Position: 0),
                    cancellation.Token);
            }
            catch (OperationCanceledException)
            {
                cancelled = true;
            }

            Require(cancelled, "Pre-admission cancellation must still cancel the submitter.");
        }

        await WaitForPoolAsync(
            executor,
            expectedPrefill: 0,
            expectedDecode: 1,
            "Cancelled decode work must return to the pool without reaching the actor.");
        Require(
            executor.InferenceWorkCreatedCount == 1,
            "The cancelled admission should create exactly one decode work item.");

        var sequenceId = SequenceId.New();
        var result = await executor.SubmitDecodeAsync(
            new DecodeItem(sequenceId, model, Position: 3));
        Require(
            result.SequenceId == sequenceId && result.TokenId == 203,
            "A work item reused after cancelled admission must carry only the new request state.");

        await WaitForPoolAsync(
            executor,
            expectedPrefill: 0,
            expectedDecode: 1,
            "The reused decode work item must return to the pool after completion.");
        Require(
            executor.InferenceWorkCreatedCount == 1,
            "The valid decode after cancellation must reuse the existing work item.");
    }

    private static async Task VerifyAtomicOwnershipDelaysReuseAsync()
    {
        var backend = new ImmediateBackend(new DeviceId("cpu:pool-atomic-owner"));
        await using var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 2,
            maxBatchSize: 2);
        var model = new ModelId("pool-atomic-owner-model");
        var sequenceId = SequenceId.New();
        var submission = ContinuousBatchExecutor.BeginAtomicSubmission(1);

        Task<BackendStepResult> pending;
        using (submission.EnterSlot(0))
        {
            pending = executor.SubmitPrefillAsync(
                    new PrefillItem(
                        sequenceId,
                        model,
                        new ReadOnlyMemory<int>(new[] { 9 })))
                .AsTask();
        }

        var result = await pending.WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            result.SequenceId == sequenceId && result.TokenId == 101,
            "Atomic pooled prefill must preserve its terminal result.");

        await Task.Yield();
        Require(
            executor.InferenceWorkPoolCounts.Prefill == 0,
            "Consumed atomic work must not return to the pool while its atomic submission can still Abort it.");

        submission.Dispose();
        await WaitForPoolAsync(
            executor,
            expectedPrefill: 1,
            expectedDecode: 0,
            "Disposing a completed atomic submission must release its final ownership hold.");
        Require(
            executor.InferenceWorkCreatedCount == 1,
            "The single atomic prefill should create one pooled work item.");

        var reusedSequenceId = SequenceId.New();
        var reused = await executor.SubmitPrefillAsync(
            new PrefillItem(
                reusedSequenceId,
                model,
                new ReadOnlyMemory<int>(new[] { 10 })));
        Require(
            reused.SequenceId == reusedSequenceId && reused.TokenId == 101,
            "A new generation must not observe stale atomic-submission state from the previous generation.");
        Require(
            executor.InferenceWorkCreatedCount == 1,
            "The scalar prefill after atomic disposal must reuse the pooled work item.");
    }

    private static async Task VerifyActorOwnershipOutlivesAtomicOwnerAsync()
    {
        var backend = new BlockingDecodeBackend(new DeviceId("cpu:pool-actor-owner"));
        await using var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 2,
            maxBatchSize: 2);
        var model = new ModelId("pool-actor-owner-model");
        var prefillSequence = SequenceId.New();
        var decodeSequence = SequenceId.New();
        var submission = ContinuousBatchExecutor.BeginAtomicSubmission(2);

        Task<BackendStepResult> prefill;
        using (submission.EnterSlot(0))
        {
            prefill = executor.SubmitPrefillAsync(
                    new PrefillItem(
                        prefillSequence,
                        model,
                        new ReadOnlyMemory<int>(new[] { 1 })))
                .AsTask();
        }

        Task<BackendStepResult> decode;
        using (submission.EnterSlot(1))
        {
            decode = executor.SubmitDecodeAsync(
                    new DecodeItem(decodeSequence, model, Position: 4))
                .AsTask();
        }

        await backend.DecodeStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        submission.Dispose();

        var prefillResult = await prefill.WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            prefillResult.SequenceId == prefillSequence && prefillResult.TokenId == 101,
            "The first envelope segment must complete before the blocked decode segment.");
        Require(
            executor.InferenceWorkPoolCounts.Prefill == 0,
            "Releasing atomic ownership must not recycle an item while the actor still owns the envelope.");

        backend.AllowDecode();
        var decodeResult = await decode.WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            decodeResult.SequenceId == decodeSequence && decodeResult.TokenId == 204,
            "The blocked decode must complete with its own generation state.");

        await WaitForPoolAsync(
            executor,
            expectedPrefill: 1,
            expectedDecode: 1,
            "Both envelope items must recycle only after actor execution has fully left the envelope.");
        Require(
            executor.InferenceWorkCreatedCount == 2,
            "The two-item envelope should need exactly one prefill and one decode work object.");

        var nextPrefill = SequenceId.New();
        var nextDecode = SequenceId.New();
        var nextPrefillResult = await executor.SubmitPrefillAsync(
            new PrefillItem(
                nextPrefill,
                model,
                new ReadOnlyMemory<int>(new[] { 2 })));
        var nextDecodeResult = await executor.SubmitDecodeAsync(
            new DecodeItem(nextDecode, model, Position: 8));

        Require(
            nextPrefillResult.SequenceId == nextPrefill &&
            nextDecodeResult.SequenceId == nextDecode,
            "Reused work objects must carry only their next-generation sequence identities.");
        Require(
            executor.InferenceWorkCreatedCount == 2,
            "Steady-state scalar requests should reuse both work objects after the atomic envelope retires.");
    }

    private static async Task WaitForPoolAsync(
        ContinuousBatchExecutor executor,
        int expectedPrefill,
        int expectedDecode,
        string message)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (DateTime.UtcNow < deadline)
        {
            var counts = executor.InferenceWorkPoolCounts;
            if (counts.Prefill == expectedPrefill && counts.Decode == expectedDecode)
            {
                return;
            }

            await Task.Delay(1).ConfigureAwait(false);
        }

        var final = executor.InferenceWorkPoolCounts;
        throw new InvalidOperationException(
            $"{message} Observed pool counts prefill={final.Prefill}, decode={final.Decode}.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class ImmediateBackend(DeviceId device) : IInferenceBackend
    {
        private bool _initialized;

        public string Name => "immediate-pool-spec";
        public DeviceId Device { get; } = device;

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _initialized = true;
            return ValueTask.CompletedTask;
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items.Select(static item =>
                    new BackendStepResult(item.SequenceId, TokenId: 101)).ToArray());
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items.Select(static item =>
                    new BackendStepResult(item.SequenceId, TokenId: 200 + item.Position)).ToArray());
        }

        public ValueTask DisposeAsync()
        {
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

    private sealed class BlockingDecodeBackend(DeviceId device) : IInferenceBackend
    {
        private readonly TaskCompletionSource<bool> _decodeStarted =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<bool> _allowDecode =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private bool _initialized;

        public string Name => "blocking-decode-pool-spec";
        public DeviceId Device { get; } = device;
        public TaskCompletionSource<bool> DecodeStarted => _decodeStarted;

        public void AllowDecode() => _allowDecode.TrySetResult(true);

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _initialized = true;
            return ValueTask.CompletedTask;
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items.Select(static item =>
                    new BackendStepResult(item.SequenceId, TokenId: 101)).ToArray());
        }

        public async ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            _decodeStarted.TrySetResult(true);
            await _allowDecode.Task.ConfigureAwait(false);
            return batch.Items.Select(static item =>
                new BackendStepResult(item.SequenceId, TokenId: 200 + item.Position)).ToArray();
        }

        public ValueTask DisposeAsync()
        {
            _allowDecode.TrySetResult(true);
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
