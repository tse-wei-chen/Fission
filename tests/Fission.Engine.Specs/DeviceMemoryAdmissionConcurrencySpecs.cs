using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;

internal static class DeviceMemoryAdmissionConcurrencySpecs
{
    [ModuleInitializer]
    internal static void Run() =>
        RunAsync().GetAwaiter().GetResult();

    private static async Task RunAsync()
    {
        var deviceId = new DeviceId("cpu:shared-memory-admission");
        var modelId = new ModelId("shared-memory-admission-model");
        var backend = new BlockingPressureBackend(deviceId, kvBytesPerToken: 128);
        await using var device = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 8,
            maxBatchSize: 4);
        using var runtime = new ExecutionPlanExecutor(
            device,
            kvPagePool: new KvPagePool(capacity: 8, tokensPerPage: 4));

        var options = new InferenceEngineOptions(
            MaxBatchTokens: 1,
            MaxBatchSequences: 1,
            Scheduling: new SchedulingPolicyOptions(
                DecodeTokenReserve: 0,
                MaxPrefillChunkTokens: 1,
                DeadlineUrgencyWindow: TimeSpan.Zero),
            MaxDeviceBytes: 128);
        using var firstEngine = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            options,
            kvMemoryProfile: backend);
        using var secondEngine = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            options,
            kvMemoryProfile: backend);

        var now = new DateTimeOffset(2026, 9, 30, 12, 0, 0, TimeSpan.Zero);
        var firstSequence = firstEngine.Submit(
            modelId,
            new[] { 1 },
            maxNewTokens: 4,
            enqueuedAt: now);
        var secondSequence = secondEngine.Submit(
            modelId,
            new[] { 2 },
            maxNewTokens: 4,
            enqueuedAt: now.AddMilliseconds(1));

        var firstCycleTask = firstEngine.RunCycleAsync(now.AddMilliseconds(2)).AsTask();
        await backend.FirstPrefillStarted.Task
            .WaitAsync(TimeSpan.FromSeconds(5));

        var blockedSecondCycle = await secondEngine
            .RunCycleAsync(now.AddMilliseconds(3))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));

        Require(
            blockedSecondCycle.Batch.Items.Count == 0,
            "A second engine sharing the runtime must not admit the same physical headroom while the first batch holds a transient reservation.");
        Require(
            blockedSecondCycle.Deferred.Count == 1 &&
            blockedSecondCycle.Deferred[0].SequenceId == secondSequence &&
            blockedSecondCycle.Deferred[0].Reason == SchedulingDeferralReason.DeviceMemoryBudget,
            "Shared-runtime reservation pressure must surface as DeviceMemoryBudget deferral for the competing engine.");
        Require(
            backend.PrefillCalls == 1,
            "The competing engine must be deferred before a second backend prefill reaches the device actor.");

        backend.ReleaseFirstPrefill();
        var firstCycle = await firstCycleTask.WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            firstCycle.Batch.Items.Count == 1 &&
            firstCycle.Batch.Items[0].SequenceId == firstSequence,
            "The first engine must retain and execute the batch that owns the reservation.");

        var admittedSecondCycle = await secondEngine
            .RunCycleAsync(now.AddMilliseconds(4))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            admittedSecondCycle.Batch.Items.Count == 1 &&
            admittedSecondCycle.Batch.Items[0].SequenceId == secondSequence,
            "Releasing the first batch reservation must make the physical headroom reusable by the second engine.");
        Require(
            backend.PrefillCalls == 2,
            "The second backend prefill should execute only after the first reservation is released.");

        await firstEngine.CancelAsync(firstSequence);
        await secondEngine.CancelAsync(secondSequence);
        Require(
            runtime.SequenceCount == 0,
            "Shared-runtime admission spec must release both runtime sequences after cancellation.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class BlockingPressureBackend :
        IInferenceBackend,
        IInferenceKvMemoryProfile,
        IInferenceDeviceMemoryPressureSource
    {
        private readonly long _kvBytesPerToken;
        private readonly TaskCompletionSource<bool> _releaseFirstPrefill =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private bool _initialized;
        private int _prefillCalls;

        public BlockingPressureBackend(DeviceId device, long kvBytesPerToken)
        {
            Device = device;
            _kvBytesPerToken = kvBytesPerToken;
        }

        public string Name => "blocking-shared-admission";
        public DeviceId Device { get; }
        public int PrefillCalls => Volatile.Read(ref _prefillCalls);
        public TaskCompletionSource<bool> FirstPrefillStarted { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public long GetKvBytesPerToken(ModelId modelId) => _kvBytesPerToken;

        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            pressure = new InferenceDeviceMemoryPressure(
                ActiveBytes: 0,
                ReclaimableBytes: 0,
                ReservedBytes: 0,
                PeakReservedBytes: 0);
            return true;
        }

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _initialized = true;
            return ValueTask.CompletedTask;
        }

        public async ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();

            var call = Interlocked.Increment(ref _prefillCalls);
            if (call == 1)
            {
                FirstPrefillStarted.TrySetResult(true);
                await _releaseFirstPrefill.Task
                    .WaitAsync(cancellationToken)
                    .ConfigureAwait(false);
            }

            return batch.Items
                .Select(static item => new BackendStepResult(
                    item.SequenceId,
                    TokenId: 7))
                .ToArray();
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items
                    .Select(static item => new BackendStepResult(
                        item.SequenceId,
                        TokenId: 8))
                    .ToArray());
        }

        public ValueTask ReleaseSequenceAsync(
            SequenceId sequenceId,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.CompletedTask;
        }

        public void ReleaseFirstPrefill() =>
            _releaseFirstPrefill.TrySetResult(true);

        public ValueTask DisposeAsync()
        {
            _releaseFirstPrefill.TrySetResult(true);
            _initialized = false;
            return ValueTask.CompletedTask;
        }

        private void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException(
                    "Blocking shared-admission backend is not initialized.");
            }
        }
    }
}
