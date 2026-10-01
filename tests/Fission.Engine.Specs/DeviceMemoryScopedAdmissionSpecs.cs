using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;

internal static class DeviceMemoryScopedAdmissionSpecs
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
                $"Device-memory scoped admission spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var primaryId = new DeviceId("cpu:engine-scoped-pressure-0");
        var unrelatedId = new DeviceId("cpu:engine-scoped-pressure-1");
        var modelId = new ModelId("engine-scoped-pressure-model");
        var primaryBackend = new CountingAdmissionBackend(primaryId);
        var unrelatedBackend = new CountingAdmissionBackend(unrelatedId);

        await using var primaryDevice = await ContinuousBatchExecutor.CreateAsync(
            primaryBackend,
            capacity: 4,
            maxBatchSize: 2);
        await using var unrelatedDevice = await ContinuousBatchExecutor.CreateAsync(
            unrelatedBackend,
            capacity: 4,
            maxBatchSize: 2);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(primaryDevice, unrelatedDevice),
            kvPagePool: new KvPagePool(capacity: 8, tokensPerPage: 4));
        using var engine = new InferenceEngine(
            runtime,
            new SchedulingKernel(),
            new InferenceEngineOptions(
                MaxBatchTokens: 1,
                MaxBatchSequences: 1,
                Scheduling: new SchedulingPolicyOptions(
                    DecodeTokenReserve: 0,
                    MaxPrefillChunkTokens: 1,
                    DeadlineUrgencyWindow: TimeSpan.Zero),
                MaxDeviceBytes: 1024),
            kvMemoryProfile: primaryBackend);

        var sequenceId = engine.Submit(
            modelId,
            new[] { 1 },
            maxNewTokens: 1,
            enqueuedAt: new DateTimeOffset(2026, 10, 1, 0, 0, 0, TimeSpan.Zero));

        var cycle = await engine.RunCycleAsync(
                new DateTimeOffset(2026, 10, 1, 0, 0, 1, TimeSpan.Zero))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));

        Require(
            cycle.Batch.Items.Count == 1 &&
            cycle.Batch.Items[0].SequenceId == sequenceId,
            "The scoped-admission engine spec must admit its default-device request.");
        Require(
            primaryBackend.PressureQueries > 0,
            "Engine device-memory admission must observe pressure from the request's physical execution device.");
        Require(
            unrelatedBackend.PressureQueries == 0,
            "Engine device-memory admission must not query unrelated registered device actors.");
        Require(
            unrelatedBackend.PrefillCalls == 0,
            "The unrelated physical actor must not receive inference work from the default-device request.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class CountingAdmissionBackend :
        IInferenceBackend,
        IInferenceKvMemoryProfile,
        IInferenceDeviceMemoryPressureSource
    {
        private bool _initialized;
        private int _pressureQueries;
        private int _prefillCalls;

        public CountingAdmissionBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "counting-engine-admission";
        public DeviceId Device { get; }
        public int PressureQueries => Volatile.Read(ref _pressureQueries);
        public int PrefillCalls => Volatile.Read(ref _prefillCalls);

        public long GetKvBytesPerToken(ModelId modelId) => 128;

        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            EnsureInitialized();
            Interlocked.Increment(ref _pressureQueries);
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

        public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref _prefillCalls);
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items
                    .Select(static item => new BackendStepResult(
                        item.SequenceId,
                        TokenId: 7,
                        IsFinished: true))
                    .ToArray());
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
                        TokenId: 8,
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
            _initialized = false;
            return ValueTask.CompletedTask;
        }

        private void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException(
                    "Counting admission backend is not initialized.");
            }
        }
    }
}
