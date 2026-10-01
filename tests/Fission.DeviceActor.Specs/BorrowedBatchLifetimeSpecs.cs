using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class BorrowedBatchLifetimeSpecs
{
    [ModuleInitializer]
    internal static void Run() => RunAsync().GetAwaiter().GetResult();

    private static async Task RunAsync()
    {
        await VerifyPrefillBatchLivesUntilBackendCompletionAsync();
        await VerifyDecodeBatchLivesUntilBackendCompletionAsync();
    }

    private static async Task VerifyPrefillBatchLivesUntilBackendCompletionAsync()
    {
        var backend = new DelayedBatchReadBackend(new DeviceId("cpu:borrowed-prefill"));
        await using var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 8,
            maxBatchSize: 8);

        var model = new ModelId("borrowed-batch-model");
        var first = SequenceId.New();
        var second = SequenceId.New();
        var third = SequenceId.New();

        var firstWork = executor.SubmitPrefillAsync(
            new PrefillItem(first, model, new ReadOnlyMemory<int>(new[] { 1 }))).AsTask();
        await backend.FirstPrefillEntered.WaitAsync(TimeSpan.FromSeconds(5));

        var secondWork = executor.SubmitPrefillAsync(
            new PrefillItem(second, model, new ReadOnlyMemory<int>(new[] { 2 }))).AsTask();
        var thirdWork = executor.SubmitPrefillAsync(
            new PrefillItem(third, model, new ReadOnlyMemory<int>(new[] { 3 }))).AsTask();
        await Task.Yield();

        backend.AllowFirstPrefillRead();
        await Task.WhenAll(firstWork, secondWork, thirdWork)
            .WaitAsync(TimeSpan.FromSeconds(5));

        Require(
            backend.PrefillCalls.Count == 2,
            "Queued prefills should execute as one delayed scalar call followed by one two-item micro-batch.");
        Require(
            backend.PrefillCalls[0].SequenceEqual(new[] { first }),
            "The delayed backend must still observe the original first prefill after awaiting inside the backend call.");
        Require(
            backend.PrefillCalls[1].SequenceEqual(new[] { second, third }),
            "Reusable prefill storage must resize to the next micro-batch without leaking the prior item.");
    }

    private static async Task VerifyDecodeBatchLivesUntilBackendCompletionAsync()
    {
        var backend = new DelayedBatchReadBackend(new DeviceId("cpu:borrowed-decode"));
        await using var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 8,
            maxBatchSize: 8);

        var model = new ModelId("borrowed-batch-model");
        var first = SequenceId.New();
        var second = SequenceId.New();
        var third = SequenceId.New();

        var firstWork = executor.SubmitDecodeAsync(
            new DecodeItem(first, model, Position: 1)).AsTask();
        await backend.FirstDecodeEntered.WaitAsync(TimeSpan.FromSeconds(5));

        var secondWork = executor.SubmitDecodeAsync(
            new DecodeItem(second, model, Position: 2)).AsTask();
        var thirdWork = executor.SubmitDecodeAsync(
            new DecodeItem(third, model, Position: 3)).AsTask();
        await Task.Yield();

        backend.AllowFirstDecodeRead();
        await Task.WhenAll(firstWork, secondWork, thirdWork)
            .WaitAsync(TimeSpan.FromSeconds(5));

        Require(
            backend.DecodeCalls.Count == 2,
            "Queued decodes should execute as one delayed scalar call followed by one two-item micro-batch.");
        Require(
            backend.DecodeCalls[0].SequenceEqual(new[] { first }),
            "The delayed backend must still observe the original first decode after awaiting inside the backend call.");
        Require(
            backend.DecodeCalls[1].SequenceEqual(new[] { second, third }),
            "Reusable decode storage must resize to the next micro-batch without leaking the prior item.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class DelayedBatchReadBackend(DeviceId device) : IInferenceBackend
    {
        private readonly TaskCompletionSource<bool> _firstPrefillEntered =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<bool> _allowFirstPrefillRead =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<bool> _firstDecodeEntered =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<bool> _allowFirstDecodeRead =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _prefillCallCount;
        private int _decodeCallCount;

        public string Name => "delayed-batch-read";
        public DeviceId Device { get; } = device;
        public Task FirstPrefillEntered => _firstPrefillEntered.Task;
        public Task FirstDecodeEntered => _firstDecodeEntered.Task;
        public List<SequenceId[]> PrefillCalls { get; } = new();
        public List<SequenceId[]> DecodeCalls { get; } = new();

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.CompletedTask;
        }

        public async ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            var call = Interlocked.Increment(ref _prefillCallCount);
            if (call == 1)
            {
                _firstPrefillEntered.TrySetResult(true);
                await _allowFirstPrefillRead.Task.WaitAsync(cancellationToken)
                    .ConfigureAwait(false);
            }

            var sequences = new SequenceId[batch.Items.Count];
            var results = new BackendStepResult[batch.Items.Count];
            for (var index = 0; index < batch.Items.Count; index++)
            {
                sequences[index] = batch.Items[index].SequenceId;
                results[index] = new BackendStepResult(sequences[index], TokenId: index);
            }

            PrefillCalls.Add(sequences);
            return results;
        }

        public async ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            var call = Interlocked.Increment(ref _decodeCallCount);
            if (call == 1)
            {
                _firstDecodeEntered.TrySetResult(true);
                await _allowFirstDecodeRead.Task.WaitAsync(cancellationToken)
                    .ConfigureAwait(false);
            }

            var sequences = new SequenceId[batch.Items.Count];
            var results = new BackendStepResult[batch.Items.Count];
            for (var index = 0; index < batch.Items.Count; index++)
            {
                sequences[index] = batch.Items[index].SequenceId;
                results[index] = new BackendStepResult(sequences[index], TokenId: index);
            }

            DecodeCalls.Add(sequences);
            return results;
        }

        public void AllowFirstPrefillRead() => _allowFirstPrefillRead.TrySetResult(true);

        public void AllowFirstDecodeRead() => _allowFirstDecodeRead.TrySetResult(true);

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
