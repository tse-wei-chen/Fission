using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;

internal static class DeviceSequenceCapacitySpecs
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
                $"Device sequence capacity spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        await VerifyUnrelatedSmallActorDoesNotClampAsync();
        await VerifyTargetActorCapacityStillAppliesAsync();
    }

    private static async Task VerifyUnrelatedSmallActorDoesNotClampAsync()
    {
        var primaryBackend = new CapacityProbeBackend(
            new DeviceId("cpu:sequence-capacity-primary-large"));
        var unrelatedBackend = new CapacityProbeBackend(
            new DeviceId("cpu:sequence-capacity-unrelated-small"));

        // Device sequence admission is bounded by inference item credits, not by
        // the backend micro-batch size. The primary actor can admit four items but
        // deliberately executes them one at a time; the unrelated actor has only
        // one admission credit and must not clamp work targeted at the primary.
        await using var primary = await ContinuousBatchExecutor.CreateAsync(
            primaryBackend,
            capacity: 4,
            maxBatchSize: 1);
        await using var unrelated = await ContinuousBatchExecutor.CreateAsync(
            unrelatedBackend,
            capacity: 1,
            maxBatchSize: 4);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(primary, unrelated),
            kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));
        using var engine = CreateEngine(runtime, maxBatchSequences: 4);

        var model = new ModelId("heterogeneous-sequence-capacity-model");
        var sequences = Enumerable.Range(0, 3)
            .Select(index => engine.Submit(
                model,
                new[] { index + 1 },
                maxNewTokens: 1,
                enqueuedAt: new DateTimeOffset(2026, 10, 1, 1, 0, index, TimeSpan.Zero)))
            .ToArray();

        var cycle = await engine.RunCycleAsync(
                new DateTimeOffset(2026, 10, 1, 1, 1, 0, TimeSpan.Zero))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));

        Require(
            cycle.Batch.Items.Count == 3,
            "An unrelated capacity=1 actor must not clamp a capacity=4 default actor to one selected item.");
        Require(
            cycle.CompletedSequences.Count == 3 &&
            sequences.All(sequence => cycle.CompletedSequences.Contains(sequence)),
            "All three requests admitted to the larger default actor must complete in the same cycle.");
        Require(
            primaryBackend.PrefillItems == 3 &&
            primaryBackend.MaxObservedPrefillBatch == 1,
            "The primary actor should admit three selected items while honoring its independent maxBatchSize=1 backend chunking limit.");
        Require(
            unrelatedBackend.PrefillItems == 0,
            "The unrelated smaller actor must not receive default-device inference work.");
    }

    private static async Task VerifyTargetActorCapacityStillAppliesAsync()
    {
        var primaryBackend = new CapacityProbeBackend(
            new DeviceId("cpu:sequence-capacity-primary-small"));
        var unrelatedBackend = new CapacityProbeBackend(
            new DeviceId("cpu:sequence-capacity-unrelated-large"));

        // The target actor can execute a four-item backend micro-batch, but only
        // two inference items may be admitted atomically because its credit
        // capacity is two. The scheduler must therefore defer the third item.
        await using var primary = await ContinuousBatchExecutor.CreateAsync(
            primaryBackend,
            capacity: 2,
            maxBatchSize: 4);
        await using var unrelated = await ContinuousBatchExecutor.CreateAsync(
            unrelatedBackend,
            capacity: 4,
            maxBatchSize: 1);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(primary, unrelated),
            kvPagePool: new KvPagePool(capacity: 16, tokensPerPage: 4));
        using var engine = CreateEngine(runtime, maxBatchSequences: 4);

        var model = new ModelId("target-sequence-capacity-model");
        var sequences = Enumerable.Range(0, 3)
            .Select(index => engine.Submit(
                model,
                new[] { index + 11 },
                maxNewTokens: 1,
                enqueuedAt: new DateTimeOffset(2026, 10, 1, 2, 0, index, TimeSpan.Zero)))
            .ToArray();

        var cycle = await engine.RunCycleAsync(
                new DateTimeOffset(2026, 10, 1, 2, 1, 0, TimeSpan.Zero))
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));

        Require(
            cycle.Batch.Items.Count == 2,
            "A capacity=2 target actor must never receive more than two selected items even when global MaxBatchSequences is four.");
        Require(
            cycle.Deferred.Count == 1 &&
            cycle.Deferred[0].Reason == SchedulingDeferralReason.DeviceSequenceBudget,
            "The third request targeting a saturated actor must defer with DeviceSequenceBudget.");
        Require(
            primaryBackend.PrefillItems == 2 &&
            primaryBackend.MaxObservedPrefillBatch == 2,
            "The target actor must receive exactly its two-item admission capacity in the first cycle.");
        Require(
            unrelatedBackend.PrefillItems == 0,
            "A larger unrelated actor must not absorb work without explicit placement.");

        var deferredSequence = sequences.Single(sequence =>
            cycle.Deferred.Any(deferred => deferred.SequenceId == sequence));
        await engine.CancelAsync(deferredSequence);
        Require(
            engine.ActiveRequestCount == 0,
            "Capacity regression cleanup must leave no active engine request.");
    }

    private static InferenceEngine CreateEngine(
        ExecutionPlanExecutor runtime,
        int maxBatchSequences) =>
        new(
            runtime,
            new SchedulingKernel(),
            new InferenceEngineOptions(
                MaxBatchTokens: maxBatchSequences,
                MaxBatchSequences: maxBatchSequences,
                Scheduling: new SchedulingPolicyOptions(
                    DecodeTokenReserve: 0,
                    MaxPrefillChunkTokens: 1,
                    DeadlineUrgencyWindow: TimeSpan.Zero)));

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class CapacityProbeBackend : IInferenceBackend
    {
        private bool _initialized;
        private int _prefillItems;
        private int _maxObservedPrefillBatch;

        public CapacityProbeBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "device-sequence-capacity-probe";
        public DeviceId Device { get; }
        public int PrefillItems => Volatile.Read(ref _prefillItems);
        public int MaxObservedPrefillBatch => Volatile.Read(ref _maxObservedPrefillBatch);

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

            Interlocked.Add(ref _prefillItems, batch.Items.Count);
            UpdateMaxBatch(batch.Items.Count);
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

        private void UpdateMaxBatch(int candidate)
        {
            while (true)
            {
                var current = Volatile.Read(ref _maxObservedPrefillBatch);
                if (candidate <= current ||
                    Interlocked.CompareExchange(
                        ref _maxObservedPrefillBatch,
                        candidate,
                        current) == current)
                {
                    return;
                }
            }
        }

        private void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException(
                    "Device sequence capacity probe backend is not initialized.");
            }
        }
    }
}
