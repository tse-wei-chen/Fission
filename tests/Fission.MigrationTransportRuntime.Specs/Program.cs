using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static ExecutionBindings EmptyBindings() =>
    new(new Dictionary<SequenceId, ReadOnlyMemory<int>>());

static SequenceProcess GetSequence(ExecutionPlanExecutor runtime, SequenceId sequenceId)
{
    if (!runtime.TryGetSequence(sequenceId, out var sequence) || sequence is null)
    {
        throw new InvalidOperationException($"Sequence {sequenceId} is missing.");
    }

    return sequence;
}

static async Task PrefillAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId,
    ModelId model)
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new PrefillExecutionStep(sequenceId, model, 4, CompletesPrefill: true)
            }),
        new ExecutionBindings(
            new Dictionary<SequenceId, ReadOnlyMemory<int>>
            {
                [sequenceId] = new ReadOnlyMemory<int>(new[] { 1, 2, 3, 4 })
            }));
}

static Task MigrateAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId,
    DeviceId targetDevice,
    CancellationToken cancellationToken = default) =>
    runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new MigrateKvExecutionStep(sequenceId, targetDevice)
            }),
        EmptyBindings(),
        cancellationToken).AsTask();

static Task DecodeOneAsync(
    ExecutionPlanExecutor runtime,
    SequenceId sequenceId) =>
    runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new DecodeExecutionStep(sequenceId, 1)
            }),
        EmptyBindings()).AsTask();

static async Task RunSuccessAsync()
{
    var sourceId = new DeviceId("gpu:source");
    var targetId = new DeviceId("gpu:target");
    const long estimatedBytes = 256;
    using var admission = new SequenceMigrationAdmissionController(
        maxInflightBytes: 1024,
        maxConcurrentTransfers: 1);
    var fabric = new TransportFabric(admission, estimatedBytes, sourceId, targetId);
    var sourceBackend = new TransportBackend(sourceId, fabric, estimatedBytes);
    var targetBackend = new TransportBackend(targetId, fabric, estimatedBytes);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        kvPagePool: new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    var model = new ModelId("transport-runtime-success");
    await PrefillAsync(runtime, sequenceId, model);
    var versionBefore = GetSequence(runtime, sequenceId).Version;

    await MigrateAsync(runtime, sequenceId, targetId);

    var sequence = GetSequence(runtime, sequenceId);
    Require(sequence.Device == targetId, "Transport-aware success must publish target placement.");
    Require(sequence.Version == versionBefore + 1, "Transport-aware success must advance version exactly once.");
    Require(sourceBackend.EstimateCalls == 1, "Source backend must estimate physical migration bytes exactly once.");
    Require(sourceBackend.CapabilityCalls == 1, "Source capabilities must be queried through the source actor.");
    Require(targetBackend.CapabilityCalls == 1, "Target capabilities must be queried through the target actor.");
    Require(sourceBackend.LegacyPrepareCalls == 0, "Transport-aware migration must not call legacy prepare.");
    Require(sourceBackend.PlannedPrepareCalls == 1, "Transport-aware migration must use planned prepare.");
    Require(fabric.LastPlan?.TransportId == "cuda-p2p", "Planner must select the fastest mutually supported transport.");
    Require(admission.ActiveTransfers == 0 && admission.InflightBytes == 0, "Admission lease must release after commit.");
    Require(!fabric.Contains(sourceId, sequenceId), "Successful commit must remove source physical state.");
    Require(fabric.GetPosition(targetId, sequenceId) == 4, "Successful import must create runnable target state.");

    await DecodeOneAsync(runtime, sequenceId);
    Require(fabric.GetPosition(targetId, sequenceId) == 5, "Post-migration decode must execute on target state.");
    Require(
        fabric.Events.SequenceEqual(new[]
        {
            "prefill:4@gpu:source",
            "prepare:cuda-p2p:4",
            "import:cuda-p2p:4",
            "commit:cuda-p2p:4",
            "decode:5@gpu:target"
        }),
        $"Transport-aware success order is incorrect: [{string.Join(',', fabric.Events)}].");
}

static async Task RunPlanMismatchAsync()
{
    var sourceId = new DeviceId("gpu:source");
    var targetId = new DeviceId("gpu:target");
    const long estimatedBytes = 256;
    using var admission = new SequenceMigrationAdmissionController(1024, 1);
    var fabric = new TransportFabric(admission, estimatedBytes, sourceId, targetId);
    var sourceBackend = new TransportBackend(sourceId, fabric, estimatedBytes)
    {
        AttestDifferentPlan = true
    };
    var targetBackend = new TransportBackend(targetId, fabric, estimatedBytes);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        kvPagePool: new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("transport-runtime-mismatch"));
    var versionBefore = GetSequence(runtime, sequenceId).Version;

    var failed = false;
    try
    {
        await MigrateAsync(runtime, sequenceId, targetId);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("runtime selected", StringComparison.Ordinal))
    {
        failed = true;
    }

    Require(failed, "A backend-attested transport plan mismatch must fail the migration.");
    var sequence = GetSequence(runtime, sequenceId);
    Require(sequence.Device == sourceId, "Plan mismatch must not publish target placement.");
    Require(sequence.Version == versionBefore, "Plan mismatch must not advance sequence version.");
    Require(fabric.Contains(sourceId, sequenceId), "Source abort must leave source state runnable after plan mismatch.");
    Require(!fabric.Contains(targetId, sequenceId), "Plan mismatch must be rejected before target import.");
    Require(fabric.SourceAbortCalls == 1, "Plan mismatch must abort source transfer resources.");
    Require(fabric.TargetAbortCalls == 0, "Plan mismatch before import must not abort target state.");
    Require(admission.ActiveTransfers == 0 && admission.InflightBytes == 0, "Mismatch rollback must release admission after abort.");

    await DecodeOneAsync(runtime, sequenceId);
    Require(fabric.GetPosition(sourceId, sequenceId) == 5, "Source must remain decodable after plan mismatch rollback.");
}

static async Task RunImportFailureAsync()
{
    var sourceId = new DeviceId("gpu:source");
    var targetId = new DeviceId("gpu:target");
    const long estimatedBytes = 256;
    using var admission = new SequenceMigrationAdmissionController(1024, 1);
    var fabric = new TransportFabric(admission, estimatedBytes, sourceId, targetId);
    var sourceBackend = new TransportBackend(sourceId, fabric, estimatedBytes);
    var targetBackend = new TransportBackend(targetId, fabric, estimatedBytes)
    {
        FailImportAfterStaging = true
    };

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        kvPagePool: new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("transport-runtime-import-failure"));
    var versionBefore = GetSequence(runtime, sequenceId).Version;

    var failed = false;
    try
    {
        await MigrateAsync(runtime, sequenceId, targetId);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("transport-import-failed", StringComparison.Ordinal))
    {
        failed = true;
    }

    Require(failed, "Target import failure must propagate through transport-aware migration.");
    var sequence = GetSequence(runtime, sequenceId);
    Require(sequence.Device == sourceId, "Import failure must keep source placement.");
    Require(sequence.Version == versionBefore, "Import failure must not advance sequence version.");
    Require(fabric.GetPosition(sourceId, sequenceId) == 4, "Source state must remain runnable after rollback.");
    Require(!fabric.Contains(targetId, sequenceId), "Target abort must remove staged target state.");
    Require(fabric.TargetAbortCalls == 1 && fabric.SourceAbortCalls == 1, "Import failure must abort target then source.");
    Require(admission.ActiveTransfers == 0 && admission.InflightBytes == 0, "Admission must release only after rollback finishes.");

    await DecodeOneAsync(runtime, sequenceId);
    Require(fabric.GetPosition(sourceId, sequenceId) == 5, "Source actor must remain usable after transport rollback.");
}

static async Task RunAdmissionCancellationAsync()
{
    var sourceId = new DeviceId("gpu:source");
    var targetId = new DeviceId("gpu:target");
    const long estimatedBytes = 256;
    using var admission = new SequenceMigrationAdmissionController(
        maxInflightBytes: 1024,
        maxConcurrentTransfers: 2);
    var fabric = new TransportFabric(admission, estimatedBytes, sourceId, targetId);
    var sourceBackend = new TransportBackend(sourceId, fabric, estimatedBytes);
    var targetBackend = new TransportBackend(targetId, fabric, estimatedBytes);

    await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(sourceBackend, 8, 8);
    await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(targetBackend, 8, 8);
    using var runtime = new ExecutionPlanExecutor(
        new ExecutionDeviceRegistry(sourceDevice, targetDevice),
        new SequenceMigrationTransportPlanner(),
        admission,
        kvPagePool: new KvPagePool(16, 4));

    var sequenceId = SequenceId.New();
    await PrefillAsync(runtime, sequenceId, new ModelId("transport-runtime-admission-cancel"));

    using var blocker = await admission.AcquireAsync(900);
    using var cancellation = new CancellationTokenSource();
    var migration = MigrateAsync(runtime, sequenceId, targetId, cancellation.Token);
    await Task.Yield();
    await Task.Yield();
    Require(sourceBackend.PlannedPrepareCalls == 0, "Migration must not prepare while admission bytes are unavailable.");

    cancellation.Cancel();
    var cancelled = false;
    try
    {
        await migration;
    }
    catch (OperationCanceledException)
    {
        cancelled = true;
    }

    Require(cancelled, "Migration waiting for admission must observe caller cancellation.");
    Require(GetSequence(runtime, sequenceId).Device == sourceId, "Cancelled admission wait must leave source placement unchanged.");
    Require(sourceBackend.PlannedPrepareCalls == 0, "Cancelled admission wait must not allocate transfer state.");

    blocker.Dispose();
    Require(admission.ActiveTransfers == 0 && admission.InflightBytes == 0, "Cancelled admission waiter must not leak slot or bytes.");

    await MigrateAsync(runtime, sequenceId, targetId);
    Require(GetSequence(runtime, sequenceId).Device == targetId, "Migration must be retryable after admission cancellation.");
    Require(sourceBackend.PlannedPrepareCalls == 1, "Retry must reach planned prepare exactly once.");
}

await RunSuccessAsync();
await RunPlanMismatchAsync();
await RunImportFailureAsync();
await RunAdmissionCancellationAsync();

Console.WriteLine("Fission transport-aware migration runtime specs passed.");

sealed class TransportTransfer : SequenceMigrationTransfer
{
    public TransportTransfer(
        Guid transactionId,
        SequenceId sequenceId,
        DeviceId sourceDevice,
        DeviceId targetDevice,
        int position,
        SequenceMigrationTransportPlan transportPlan)
        : base(transactionId, sequenceId, sourceDevice, targetDevice, transportPlan)
    {
        Position = position;
    }

    public int Position { get; }
}

sealed class TransportFabric
{
    private readonly object _gate = new();
    private readonly Dictionary<DeviceId, Dictionary<SequenceId, int>> _states;
    private readonly List<string> _events = new();
    private readonly SequenceMigrationAdmissionController _admission;
    private readonly long _expectedAdmissionBytes;

    public TransportFabric(
        SequenceMigrationAdmissionController admission,
        long expectedAdmissionBytes,
        params DeviceId[] devices)
    {
        _admission = admission;
        _expectedAdmissionBytes = expectedAdmissionBytes;
        _states = devices.ToDictionary(
            static device => device,
            static _ => new Dictionary<SequenceId, int>());
    }

    public SequenceMigrationTransportPlan? LastPlan { get; set; }
    public int SourceAbortCalls { get; private set; }
    public int TargetAbortCalls { get; private set; }

    public IReadOnlyList<string> Events
    {
        get
        {
            lock (_gate)
            {
                return _events.ToArray();
            }
        }
    }

    public bool Contains(DeviceId device, SequenceId sequenceId)
    {
        lock (_gate)
        {
            return _states[device].ContainsKey(sequenceId);
        }
    }

    public int GetPosition(DeviceId device, SequenceId sequenceId)
    {
        lock (_gate)
        {
            return _states[device].TryGetValue(sequenceId, out var position)
                ? position
                : throw new InvalidOperationException($"No state for {sequenceId} on {device}.");
        }
    }

    public void SetPosition(DeviceId device, SequenceId sequenceId, int position)
    {
        lock (_gate)
        {
            _states[device][sequenceId] = position;
        }
    }

    public void Remove(DeviceId device, SequenceId sequenceId)
    {
        lock (_gate)
        {
            _states[device].Remove(sequenceId);
        }
    }

    public void Record(string message)
    {
        lock (_gate)
        {
            _events.Add(message);
        }
    }

    public void AssertAdmissionHeld(string phase)
    {
        if (_admission.ActiveTransfers != 1 ||
            _admission.InflightBytes != _expectedAdmissionBytes)
        {
            throw new InvalidOperationException(
                $"Admission lease was not held during {phase}: " +
                $"active={_admission.ActiveTransfers}, bytes={_admission.InflightBytes}.");
        }
    }

    public void RecordSourceAbort() => SourceAbortCalls++;
    public void RecordTargetAbort() => TargetAbortCalls++;
}

sealed class TransportBackend : IInferenceBackend, ISequenceMigrationTransportBackend
{
    private readonly TransportFabric _fabric;
    private readonly long _estimatedBytes;
    private bool _initialized;

    public TransportBackend(
        DeviceId device,
        TransportFabric fabric,
        long estimatedBytes)
    {
        Device = device;
        _fabric = fabric;
        _estimatedBytes = estimatedBytes;
    }

    public string Name => $"transport-runtime:{Device}";
    public DeviceId Device { get; }
    public bool AttestDifferentPlan { get; init; }
    public bool FailImportAfterStaging { get; init; }
    public int EstimateCalls { get; private set; }
    public int CapabilityCalls { get; private set; }
    public int LegacyPrepareCalls { get; private set; }
    public int PlannedPrepareCalls { get; private set; }

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
        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            var position = item.Tokens.Length;
            _fabric.SetPosition(Device, item.SequenceId, position);
            _fabric.Record($"prefill:{position}@{Device}");
            results[index] = new BackendStepResult(item.SequenceId, position);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            var current = _fabric.GetPosition(Device, item.SequenceId);
            if (current != item.Position)
            {
                throw new InvalidOperationException(
                    $"Decode position {item.Position} does not match backend position {current} on {Device}.");
            }

            var next = checked(current + 1);
            _fabric.SetPosition(Device, item.SequenceId, next);
            _fabric.Record($"decode:{next}@{Device}");
            results[index] = new BackendStepResult(item.SequenceId, next);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public IReadOnlyList<SequenceMigrationTransportCapability> GetSequenceMigrationTransportCapabilities(
        DeviceId peerDevice)
    {
        EnsureInitialized();
        CapabilityCalls++;
        return new[]
        {
            new SequenceMigrationTransportCapability(
                "cuda-p2p",
                SequenceMigrationTransportKind.DirectDevice,
                512,
                Device.Value.Contains("source", StringComparison.Ordinal)
                    ? 50_000_000_000
                    : 40_000_000_000,
                TimeSpan.FromTicks(80),
                Preference: 10),
            new SequenceMigrationTransportCapability(
                "pinned-host",
                SequenceMigrationTransportKind.HostStaging,
                0,
                10_000_000_000,
                TimeSpan.FromTicks(400),
                Preference: 5)
        };
    }

    public ValueTask<long> EstimateSequenceMigrationBytesAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _ = _fabric.GetPosition(Device, sequenceId);
        EstimateCalls++;
        return ValueTask.FromResult(_estimatedBytes);
    }

    public ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default) =>
        throw new InvalidOperationException("legacy-migrate-used-by-transport-aware-backend");

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        LegacyPrepareCalls++;
        throw new InvalidOperationException("legacy-prepare-used-by-transport-aware-backend");
    }

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _fabric.AssertAdmissionHeld("prepare");
        PlannedPrepareCalls++;
        _fabric.LastPlan = transportPlan;
        var position = _fabric.GetPosition(Device, sequenceId);
        var attestedPlan = AttestDifferentPlan
            ? transportPlan with { TransportId = "backend-substituted" }
            : transportPlan;
        var transfer = new TransportTransfer(
            Guid.NewGuid(),
            sequenceId,
            Device,
            targetDevice,
            position,
            attestedPlan);
        _fabric.Record($"prepare:{transportPlan.TransportId}:{position}");
        return ValueTask.FromResult<SequenceMigrationTransfer>(transfer);
    }

    public ValueTask ImportSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _fabric.AssertAdmissionHeld("import");
        var physical = RequireTransfer(transfer);
        _fabric.SetPosition(Device, physical.SequenceId, physical.Position);
        _fabric.Record($"import:{physical.TransportPlan!.TransportId}:{physical.Position}");
        if (FailImportAfterStaging)
        {
            throw new InvalidOperationException("transport-import-failed-after-staging");
        }

        return ValueTask.CompletedTask;
    }

    public ValueTask CommitSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _fabric.AssertAdmissionHeld("commit");
        var physical = RequireTransfer(transfer);
        _fabric.Remove(Device, physical.SequenceId);
        _fabric.Record($"commit:{physical.TransportPlan!.TransportId}:{physical.Position}");
        return ValueTask.CompletedTask;
    }

    public ValueTask AbortSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _fabric.AssertAdmissionHeld("abort");
        var physical = RequireTransfer(transfer);

        if (Device == physical.TargetDevice)
        {
            _fabric.Remove(Device, physical.SequenceId);
            _fabric.RecordTargetAbort();
            return ValueTask.CompletedTask;
        }

        if (Device == physical.SourceDevice)
        {
            _fabric.SetPosition(Device, physical.SequenceId, physical.Position);
            _fabric.RecordSourceAbort();
            return ValueTask.CompletedTask;
        }

        throw new InvalidOperationException(
            $"Backend {Device} does not participate in migration {physical.TransactionId}.");
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        _fabric.Remove(Device, sequenceId);
        return ValueTask.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private static TransportTransfer RequireTransfer(SequenceMigrationTransfer transfer) =>
        transfer as TransportTransfer ??
        throw new InvalidOperationException(
            $"Unsupported migration transfer type {transfer.GetType().Name}.");

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException($"Backend {Device} is not initialized.");
        }
    }
}
