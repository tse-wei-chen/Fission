using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Plan;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;

internal static class PerDeviceReservationCompletionSpecs
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
                $"Per-device reservation completion spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var fastDeviceId = new DeviceId("cpu:reservation-completion-fast");
        var slowDeviceId = new DeviceId("cpu:reservation-completion-slow");
        var modelId = new ModelId("reservation-completion-model");
        const long kvBytesPerToken = 128;

        var fastBackend = new FastPressureBackend(fastDeviceId);
        var slowBackend = new BlockingDecodePressureBackend(slowDeviceId);
        var memoryProfile = new ConstantKvMemoryProfile(kvBytesPerToken);

        await using var fastDevice = await ContinuousBatchExecutor.CreateAsync(
            fastBackend,
            capacity: 4,
            maxBatchSize: 4);
        await using var slowDevice = await ContinuousBatchExecutor.CreateAsync(
            slowBackend,
            capacity: 4,
            maxBatchSize: 4);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(fastDevice, slowDevice),
            kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));

        var options = new InferenceEngineOptions(
            MaxBatchTokens: 2,
            MaxBatchSequences: 2,
            Scheduling: new SchedulingPolicyOptions(
                DecodeTokenReserve: 0,
                MaxPrefillChunkTokens: 1,
                DeadlineUrgencyWindow: TimeSpan.Zero),
            MaxDeviceBytes: 256);

        using var mixedEngine = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            options,
            memoryProfile);
        using var competingEngine = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            options,
            memoryProfile);

        var now = new DateTimeOffset(2026, 10, 1, 2, 0, 0, TimeSpan.Zero);
        var fastSequence = mixedEngine.Submit(
            modelId,
            new[] { 1 },
            maxNewTokens: 4,
            enqueuedAt: now);
        var slowSequence = mixedEngine.Submit(
            modelId,
            new[] { 2 },
            maxNewTokens: 4,
            enqueuedAt: now.AddMilliseconds(1));

        // Materialize both engine-owned sequences on the default/fast actor first.
        var prefillCycle = await mixedEngine
            .RunCycleAsync(now.AddMilliseconds(2))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            prefillCycle.Batch.Items.Count == 2,
            "Setup prefill must materialize both mixed-engine sequences in one cycle.");

        await runtime.ExecuteAsync(
            new CompiledExecutionPlan(
                Guid.NewGuid(),
                0,
                new ExecutionStep[]
                {
                    new MigrateKvExecutionStep(slowSequence, slowDeviceId)
                }),
            new ExecutionBindings(
                new Dictionary<SequenceId, ReadOnlyMemory<int>>()))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            runtime.ResolveExecutionDevice(slowSequence) == slowDeviceId,
            "Setup migration must move the blocked sequence to the slow actor.");

        var competingSequence = competingEngine.Submit(
            modelId,
            new[] { 3 },
            maxNewTokens: 1,
            enqueuedAt: now.AddMilliseconds(3));

        Task<InferenceCycleResult>? mixedCycleTask = null;
        try
        {
            mixedCycleTask = mixedEngine
                .RunCycleAsync(now.AddMilliseconds(4))
                .AsTask();
            await slowBackend.DecodeStarted.Task
                .WaitAsync(TimeSpan.FromSeconds(5));

            // Both decode quanta need a 2-token successor frontier (256 bytes), one
            // on each physical actor. The slow actor remains blocked, but the fast
            // actor should release its transient reservation as soon as its own
            // scheduled group reaches a terminal state.
            await WaitUntilAsync(
                () => runtime.GetDeviceMemoryReservations([fastDeviceId]).Count == 0,
                "Fast-device transient reservation was not released while another device in the same cycle remained blocked.");

            var slowReservations = runtime.GetDeviceMemoryReservations([slowDeviceId]);
            Require(
                slowReservations.Count == 1 && slowReservations[0].Bytes == 256,
                "Slow-device reservation must remain charged until its blocked decode completes.");
            Require(
                !mixedCycleTask.IsCompleted,
                "The mixed cycle should still be waiting for the intentionally blocked slow actor.");

            var competingCycle = await competingEngine
                .RunCycleAsync(now.AddMilliseconds(5))
                .AsTask()
                .WaitAsync(TimeSpan.FromSeconds(5));
            Require(
                competingCycle.Batch.Items.Count == 1 &&
                competingCycle.Batch.Items[0].SequenceId == competingSequence,
                "A competing engine must reuse fast-device headroom before the unrelated slow-device group completes.");
            Require(
                competingEngine.GetSnapshot(competingSequence).IsCompleted,
                "Competing fast-device request must complete while the slow mixed-device group is still blocked.");
            Require(
                !mixedCycleTask.IsCompleted,
                "Fast-device reuse must not require completing the slow-device group first.");

            slowBackend.ReleaseDecode();
            var mixedCycle = await mixedCycleTask.WaitAsync(TimeSpan.FromSeconds(5));
            Require(
                mixedCycle.Batch.Items.Count == 2 &&
                mixedCycle.Batch.Items.Any(item => item.SequenceId == fastSequence) &&
                mixedCycle.Batch.Items.Any(item => item.SequenceId == slowSequence),
                "Mixed cycle must preserve both per-device decode selections after the slow group resumes.");
            Require(
                runtime.GetDeviceMemoryReservations([fastDeviceId, slowDeviceId]).Count == 0,
                "All transient reservations must be released after the mixed cycle completes.");

            await mixedEngine.CancelAsync(fastSequence);
            await mixedEngine.CancelAsync(slowSequence);
            Require(
                runtime.SequenceCount == 0,
                "Per-device reservation completion regression must release every runtime sequence.");
        }
        finally
        {
            slowBackend.ReleaseDecode();
            if (mixedCycleTask is not null)
            {
                try
                {
                    await mixedCycleTask.WaitAsync(TimeSpan.FromSeconds(5));
                }
                catch
                {
                    // Preserve the original regression while ensuring actor cleanup
                    // is not hidden behind the deliberately blocked backend.
                }
            }
        }
    }

    private static async Task WaitUntilAsync(
        Func<bool> condition,
        string timeoutMessage)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (!condition())
        {
            if (DateTime.UtcNow >= deadline)
            {
                throw new InvalidOperationException(timeoutMessage);
            }

            await Task.Delay(TimeSpan.FromMilliseconds(10));
        }
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class ConstantKvMemoryProfile : IInferenceKvMemoryProfile
    {
        private readonly long _bytesPerToken;

        public ConstantKvMemoryProfile(long bytesPerToken)
        {
            _bytesPerToken = bytesPerToken;
        }

        public long GetKvBytesPerToken(ModelId modelId) => _bytesPerToken;
    }

    private abstract class PressureBackendBase :
        IInferenceBackend,
        IInferenceDeviceMemoryPressureSource
    {
        private bool _initialized;

        protected PressureBackendBase(DeviceId device)
        {
            Device = device;
        }

        public abstract string Name { get; }
        public DeviceId Device { get; }

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

        public abstract ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default);

        public abstract ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default);

        public ValueTask MigrateSequenceAsync(
            SequenceId sequenceId,
            DeviceId targetDevice,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.CompletedTask;
        }

        public ValueTask ReleaseSequenceAsync(
            SequenceId sequenceId,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.CompletedTask;
        }

        public virtual ValueTask DisposeAsync()
        {
            _initialized = false;
            return ValueTask.CompletedTask;
        }

        protected void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException(
                    $"Backend {Name} is not initialized.");
            }
        }
    }

    private sealed class FastPressureBackend : PressureBackendBase
    {
        private int _prefillItems;

        public FastPressureBackend(DeviceId device)
            : base(device)
        {
        }

        public override string Name => "reservation-completion-fast";

        public override ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();

            var previous = Interlocked.Add(ref _prefillItems, batch.Items.Count) -
                batch.Items.Count;
            var finish = previous >= 2;
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items
                    .Select(item => new BackendStepResult(
                        item.SequenceId,
                        TokenId: 10,
                        IsFinished: finish))
                    .ToArray());
        }

        public override ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items
                    .Select(static item => new BackendStepResult(
                        item.SequenceId,
                        TokenId: 11,
                        IsFinished: false))
                    .ToArray());
        }
    }

    private sealed class BlockingDecodePressureBackend : PressureBackendBase
    {
        private readonly TaskCompletionSource<bool> _releaseDecode =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public BlockingDecodePressureBackend(DeviceId device)
            : base(device)
        {
        }

        public override string Name => "reservation-completion-slow";
        public TaskCompletionSource<bool> DecodeStarted { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public override ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items
                    .Select(static item => new BackendStepResult(
                        item.SequenceId,
                        TokenId: 20,
                        IsFinished: false))
                    .ToArray());
        }

        public override async ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            DecodeStarted.TrySetResult(true);
            await _releaseDecode.Task
                .WaitAsync(cancellationToken)
                .ConfigureAwait(false);
            return batch.Items
                .Select(static item => new BackendStepResult(
                    item.SequenceId,
                    TokenId: 21,
                    IsFinished: false))
                .ToArray();
        }

        public void ReleaseDecode() => _releaseDecode.TrySetResult(true);

        public override ValueTask DisposeAsync()
        {
            _releaseDecode.TrySetResult(true);
            return base.DisposeAsync();
        }
    }
}
