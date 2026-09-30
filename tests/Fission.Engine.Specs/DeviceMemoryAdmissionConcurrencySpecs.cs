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
    private static Task? _runTask;

    [ModuleInitializer]
    internal static void Start()
    {
        // A genuinely async concurrency scenario must not synchronously block the
        // module initializer. Let the executable start normally, then verify the
        // task before process exit.
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
            Console.Error.WriteLine(
                $"Device-memory admission concurrency spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        await RunAdmissionSerializationAsync();
        await RunUntilCompleteBackpressureAsync();
        await RunWorkerBackpressureAsync();
    }

    private static async Task RunAdmissionSerializationAsync()
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

        var options = CreateOptions();
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

        Task<InferenceCycleResult>? firstCycleTask = null;
        Task<InferenceCycleResult>? secondCycleTask = null;
        try
        {
            firstCycleTask = firstEngine.RunCycleAsync(now.AddMilliseconds(2)).AsTask();
            await backend.FirstPrefillStarted.Task
                .WaitAsync(TimeSpan.FromSeconds(5));

            secondCycleTask = secondEngine
                .RunCycleAsync(now.AddMilliseconds(3))
                .AsTask();
            var blockedSecondCycle = await secondCycleTask
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
        finally
        {
            await ReleaseAndDrainAsync(backend, firstCycleTask, secondCycleTask);
        }
    }

    private static async Task RunUntilCompleteBackpressureAsync()
    {
        var deviceId = new DeviceId("cpu:run-until-memory-backpressure");
        var modelId = new ModelId("run-until-memory-backpressure-model");
        var backend = new BlockingPressureBackend(deviceId, kvBytesPerToken: 128);
        await using var device = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 8,
            maxBatchSize: 4);
        using var runtime = new ExecutionPlanExecutor(
            device,
            kvPagePool: new KvPagePool(capacity: 8, tokensPerPage: 4));
        using var blocker = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            CreateOptions(),
            kvMemoryProfile: backend);
        using var waiter = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            CreateOptions(),
            kvMemoryProfile: backend);

        var now = new DateTimeOffset(2026, 9, 30, 12, 10, 0, TimeSpan.Zero);
        var blockingSequence = blocker.Submit(
            modelId,
            new[] { 11 },
            maxNewTokens: 4,
            enqueuedAt: now);
        var waitingSequence = waiter.Submit(
            modelId,
            new[] { 12 },
            maxNewTokens: 1,
            enqueuedAt: now.AddMilliseconds(1));

        Task<InferenceCycleResult>? blockerCycle = null;
        try
        {
            blockerCycle = blocker.RunCycleAsync(now.AddMilliseconds(2)).AsTask();
            await backend.FirstPrefillStarted.Task
                .WaitAsync(TimeSpan.FromSeconds(5));

            var runUntilComplete = waiter.RunUntilCompleteAsync(maxCycles: 8).AsTask();
            var premature = await Task.WhenAny(
                runUntilComplete,
                Task.Delay(TimeSpan.FromMilliseconds(150)));
            Require(
                !ReferenceEquals(premature, runUntilComplete),
                "RunUntilCompleteAsync must wait instead of failing or completing while another engine owns the only device-memory quantum.");
            Require(
                backend.PrefillCalls == 1,
                "RunUntilCompleteAsync must remain behind the outstanding reservation before relief is signaled.");

            backend.ReleaseFirstPrefill();
            await blockerCycle.WaitAsync(TimeSpan.FromSeconds(5));
            var cycles = await runUntilComplete.WaitAsync(TimeSpan.FromSeconds(5));
            Require(
                cycles.Count >= 2 && cycles.Any(static cycle => cycle.Batch.Items.Count == 0),
                "RunUntilCompleteAsync should preserve the blocked empty cycle and retry after reservation release.");
            Require(
                waiter.GetSnapshot(waitingSequence).IsCompleted,
                "RunUntilCompleteAsync must resume and complete the waiting request after device-memory relief.");

            await blocker.CancelAsync(blockingSequence);
            Require(runtime.SequenceCount == 0,
                "RunUntilComplete backpressure spec must release all runtime sequences.");
        }
        finally
        {
            await ReleaseAndDrainAsync(backend, blockerCycle);
        }
    }

    private static async Task RunWorkerBackpressureAsync()
    {
        var deviceId = new DeviceId("cpu:worker-memory-backpressure");
        var modelId = new ModelId("worker-memory-backpressure-model");
        var backend = new BlockingPressureBackend(deviceId, kvBytesPerToken: 128);
        await using var device = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 8,
            maxBatchSize: 4);
        using var runtime = new ExecutionPlanExecutor(
            device,
            kvPagePool: new KvPagePool(capacity: 8, tokensPerPage: 4));
        using var blocker = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            CreateOptions(),
            kvMemoryProfile: backend);
        using var workerEngine = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            CreateOptions(),
            kvMemoryProfile: backend);
        await using var worker = new InferenceWorker(
            workerEngine,
            new InferenceWorkerOptions(AdmissionCapacity: 4));

        var now = new DateTimeOffset(2026, 9, 30, 12, 20, 0, TimeSpan.Zero);
        var blockingSequence = blocker.Submit(
            modelId,
            new[] { 21 },
            maxNewTokens: 4,
            enqueuedAt: now);

        Task<InferenceCycleResult>? blockerCycle = null;
        try
        {
            blockerCycle = blocker.RunCycleAsync(now.AddMilliseconds(1)).AsTask();
            await backend.FirstPrefillStarted.Task
                .WaitAsync(TimeSpan.FromSeconds(5));

            var stream = await worker.SubmitAsync(
                modelId,
                new[] { 22 },
                maxNewTokens: 1,
                enqueuedAt: now.AddMilliseconds(2));
            var premature = await Task.WhenAny(
                stream.Completion,
                Task.Delay(TimeSpan.FromMilliseconds(150)));
            Require(
                !ReferenceEquals(premature, stream.Completion),
                "InferenceWorker must remain alive and wait while shared-runtime memory is transiently reserved elsewhere.");
            Require(
                backend.PrefillCalls == 1,
                "Worker backpressure must prevent a second prefill from reaching the device actor before reservation release.");

            backend.ReleaseFirstPrefill();
            await blockerCycle.WaitAsync(TimeSpan.FromSeconds(5));
            var completed = await stream.Completion.WaitAsync(TimeSpan.FromSeconds(5));
            Require(
                completed.IsCompleted &&
                completed.FinishReason == InferenceFinishReason.Stop,
                "InferenceWorker must resume scheduling and finish the waiting request after reservation release.");
            Require(
                backend.PrefillCalls == 2,
                "Worker request should reach backend prefill exactly once after the blocking reservation is released.");

            await blocker.CancelAsync(blockingSequence);
            Require(runtime.SequenceCount == 0,
                "Worker backpressure spec must release all runtime sequences.");
        }
        finally
        {
            await ReleaseAndDrainAsync(backend, blockerCycle);
        }
    }

    private static InferenceEngineOptions CreateOptions() =>
        new(
            MaxBatchTokens: 1,
            MaxBatchSequences: 1,
            Scheduling: new SchedulingPolicyOptions(
                DecodeTokenReserve: 0,
                MaxPrefillChunkTokens: 1,
                DeadlineUrgencyWindow: TimeSpan.Zero),
            MaxDeviceBytes: 128);

    private static async Task ReleaseAndDrainAsync(
        BlockingPressureBackend backend,
        params Task<InferenceCycleResult>?[] cycles)
    {
        // A failed assertion or timeout must not leave the fake backend blocked,
        // otherwise actor disposal would hide the real regression by hanging CI.
        backend.ReleaseFirstPrefill();

        foreach (var cycle in cycles)
        {
            if (cycle is null)
            {
                continue;
            }

            try
            {
                await cycle.WaitAsync(TimeSpan.FromSeconds(5));
            }
            catch
            {
                // Preserve the original failure while making cleanup progress.
            }
        }
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
                .Select(item => new BackendStepResult(
                    item.SequenceId,
                    TokenId: 7,
                    IsFinished: call > 1))
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
