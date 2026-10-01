using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;

internal static class DeviceInferenceAdmissionConcurrencySpecs
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
            Console.Error.WriteLine(
                $"Device inference admission concurrency spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var deviceId = new DeviceId("cpu:shared-inference-credit");
        var backend = new BlockingFirstPrefillBackend(deviceId);
        await using var device = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 1,
            maxBatchSize: 8);
        using var runtime = new ExecutionPlanExecutor(
            device,
            kvPagePool: new KvPagePool(capacity: 8, tokensPerPage: 4));
        using var firstEngine = CreateEngine(runtime);
        using var secondEngine = CreateEngine(runtime);

        var model = new ModelId("shared-inference-credit-model");
        var now = new DateTimeOffset(2026, 10, 1, 3, 0, 0, TimeSpan.Zero);
        var firstSequence = firstEngine.Submit(
            model,
            new[] { 1 },
            maxNewTokens: 2,
            enqueuedAt: now);
        var secondSequence = secondEngine.Submit(
            model,
            new[] { 2 },
            maxNewTokens: 1,
            enqueuedAt: now.AddMilliseconds(1));

        Task<InferenceCycleResult>? firstCycleTask = null;
        try
        {
            firstCycleTask = firstEngine
                .RunCycleAsync(now.AddMilliseconds(2))
                .AsTask();
            await backend.FirstPrefillStarted.Task
                .WaitAsync(TimeSpan.FromSeconds(5));

            var reserved = runtime.GetDeviceInferenceReservationState([deviceId]);
            Require(
                reserved.Reservations.Count == 1 &&
                reserved.Reservations[0].Items == 1,
                "The first engine must reserve the actor's only inference item credit before backend execution completes.");

            var blockedCycle = await secondEngine
                .RunCycleAsync(now.AddMilliseconds(3))
                .AsTask()
                .WaitAsync(TimeSpan.FromSeconds(5));
            Require(
                blockedCycle.Batch.Items.Count == 0,
                "A second engine sharing a capacity=1 actor must defer instead of blocking inside actor credit acquisition.");
            Require(
                blockedCycle.Deferred.Count == 1 &&
                blockedCycle.Deferred[0].SequenceId == secondSequence &&
                blockedCycle.Deferred[0].Reason == SchedulingDeferralReason.DeviceSequenceBudget,
                "Shared item-credit contention must surface as DeviceSequenceBudget.");
            Require(
                backend.PrefillCalls == 1,
                "A scheduler-deferred second engine must not submit another backend prefill while the first credit is reserved.");
            Require(
                blockedCycle.DeviceInferenceBackpressureReservations is { Count: 1 } observed &&
                observed[0].Device == deviceId,
                "An empty DeviceSequenceBudget cycle caused by a shared reservation must carry a device-scoped release token.");

            var runUntilComplete = secondEngine
                .RunUntilCompleteAsync(maxCycles: 8)
                .AsTask();
            await RequirePendingAsync(
                runUntilComplete,
                "RunUntilCompleteAsync must wait for inference-credit relief instead of failing or spinning.");
            Require(
                backend.PrefillCalls == 1,
                "Inference backpressure wait must not hot-poll work into the actor.");

            backend.ReleaseFirstPrefill();
            var firstCycle = await firstCycleTask
                .WaitAsync(TimeSpan.FromSeconds(5));
            Require(
                firstCycle.Batch.Items.Count == 1 &&
                firstCycle.Batch.Items[0].SequenceId == firstSequence,
                "The first engine must complete its originally admitted quantum after the backend gate is released.");

            var secondCycles = await runUntilComplete
                .WaitAsync(TimeSpan.FromSeconds(5));
            Require(
                secondCycles.Count >= 2,
                "RunUntilCompleteAsync should contain at least the blocked cycle and the resumed execution cycle.");
            Require(
                secondEngine.GetSnapshot(secondSequence).IsCompleted,
                "The second engine must resume and complete after the first engine releases its inference reservation.");
            Require(
                backend.PrefillCalls == 2,
                "Exactly one resumed backend prefill is expected after inference-credit relief.");
            Require(
                runtime.GetDeviceInferenceReservationState([deviceId])
                    .Reservations.Count == 0,
                "Inference item reservations must be released when the device group reaches a terminal state.");

            await firstEngine.CancelAsync(firstSequence);
            Require(
                runtime.SequenceCount == 0,
                "Shared inference admission regression cleanup must leave no runtime sequences.");
        }
        finally
        {
            backend.ReleaseFirstPrefill();
            if (firstCycleTask is not null)
            {
                try
                {
                    await firstCycleTask.WaitAsync(TimeSpan.FromSeconds(5));
                }
                catch
                {
                    // Preserve the original assertion while making actor disposal
                    // independent from the intentionally blocked first backend call.
                }
            }
        }
    }

    private static InferenceEngine CreateEngine(ExecutionPlanExecutor runtime) =>
        new(
            runtime,
            new SchedulingKernel(),
            new InferenceEngineOptions(
                MaxBatchTokens: 1,
                MaxBatchSequences: 1,
                Scheduling: new SchedulingPolicyOptions(
                    DecodeTokenReserve: 0,
                    MaxPrefillChunkTokens: 1,
                    DeadlineUrgencyWindow: TimeSpan.Zero)));

    private static async Task RequirePendingAsync(Task task, string message)
    {
        var completed = await Task.WhenAny(
            task,
            Task.Delay(TimeSpan.FromMilliseconds(150)));
        Require(!ReferenceEquals(completed, task), message);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class BlockingFirstPrefillBackend : IInferenceBackend
    {
        private readonly TaskCompletionSource<bool> _releaseFirstPrefill =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private bool _initialized;
        private int _prefillCalls;

        public BlockingFirstPrefillBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "shared-inference-credit-probe";
        public DeviceId Device { get; }
        public int PrefillCalls => Volatile.Read(ref _prefillCalls);
        public TaskCompletionSource<bool> FirstPrefillStarted { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

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
                    TokenId: call,
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
                        TokenId: 9,
                        IsFinished: true))
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

        public ValueTask DisposeAsync()
        {
            _releaseFirstPrefill.TrySetResult(true);
            _initialized = false;
            return ValueTask.CompletedTask;
        }

        public void ReleaseFirstPrefill() =>
            _releaseFirstPrefill.TrySetResult(true);

        private void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException(
                    "Shared inference credit probe is not initialized.");
            }
        }
    }
}
