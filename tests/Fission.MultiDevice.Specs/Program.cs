using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
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

static SequenceProcess GetSequence(ExecutionPlanExecutor runtime, SequenceId sequenceId)
{
    if (!runtime.TryGetSequence(sequenceId, out var sequence) || sequence is null)
    {
        throw new InvalidOperationException($"Sequence {sequenceId} is missing.");
    }

    return sequence;
}

static ExecutionBindings PrefillBindings(SequenceId sequenceId, params int[] tokens) =>
    new(new Dictionary<SequenceId, ReadOnlyMemory<int>>
    {
        [sequenceId] = new ReadOnlyMemory<int>(tokens)
    });

var EmptyBindings = new ExecutionBindings(
    new Dictionary<SequenceId, ReadOnlyMemory<int>>());

var model = new ModelId("multi-device-model");
var sourceId = new DeviceId("gpu:0");
var targetId = new DeviceId("gpu:1");
var missingId = new DeviceId("gpu:missing");
var fabric = new SharedInferenceFabric(sourceId, targetId);
var sourceBackend = new FabricBackend(sourceId, fabric);
var targetBackend = new FabricBackend(targetId, fabric);

await using var sourceDevice = await ContinuousBatchExecutor.CreateAsync(
    sourceBackend,
    capacity: 1,
    maxBatchSize: 8);
await using var targetDevice = await ContinuousBatchExecutor.CreateAsync(
    targetBackend,
    capacity: 1,
    maxBatchSize: 8);

var devices = new ExecutionDeviceRegistry(sourceDevice, targetDevice);
Require(devices.Count == 2, "Multi-device registry must expose both actors.");
Require(devices.DefaultDevice == sourceId, "The first registered actor must be the default placement target.");

using var runtime = new ExecutionPlanExecutor(
    devices,
    kvPagePool: new KvPagePool(capacity: 32, tokensPerPage: 4));

// Two independent sequences start on the default device.
var sourceSequenceId = SequenceId.New();
var migratedSequenceId = SequenceId.New();

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[]
        {
            new PrefillExecutionStep(sourceSequenceId, model, 4, CompletesPrefill: true)
        }),
    PrefillBindings(sourceSequenceId, 1, 2, 3, 4));

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[]
        {
            new PrefillExecutionStep(migratedSequenceId, model, 4, CompletesPrefill: true)
        }),
    PrefillBindings(migratedSequenceId, 5, 6, 7, 8));

Require(sourceBackend.PrefillCalls == 2, "Initial prefills must route to the default device actor.");
Require(targetBackend.PrefillCalls == 0, "Target device must not receive initial work before migration.");

// Migration is issued on the source actor. The backend/fabric contract guarantees
// that when it returns, the target actor can continue the sequence.
await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[]
        {
            new MigrateKvExecutionStep(migratedSequenceId, targetId)
        }),
    EmptyBindings);

Require(
    GetSequence(runtime, migratedSequenceId).Device == targetId,
    "Runtime placement must commit the target device after migration.");
Require(sourceBackend.MigrationCalls == 1, "Migration must execute through the current/source actor.");
Require(
    fabric.GetPlacement(migratedSequenceId) == targetId,
    "Backend fabric placement must converge before runtime routes new inference to the target.");

// A two-item scheduled batch may span two actors even when each actor capacity is
// only one item. Each actor receives its own deterministic atomic envelope.
var schedulerBridge = new ScheduledBatchExecutor(runtime);
var scheduled = new ScheduledBatch(
    Guid.NewGuid(),
    new ScheduledWorkItem[]
    {
        new(sourceSequenceId, ScheduledWorkKind.Decode, 1, 1, 0, false),
        new(migratedSequenceId, ScheduledWorkKind.Decode, 1, 1, 0, false)
    },
    ConsumedTokens: 2,
    ConsumedKvPages: 2);

await schedulerBridge.ExecuteAsync(
    scheduled,
    new ScheduledExecutionBindings(
        new Dictionary<SequenceId, ScheduledPrefillBinding>()));

Require(sourceBackend.DecodeCalls == 1, "Source-resident decode must execute on source actor.");
Require(targetBackend.DecodeCalls == 1, "Migrated decode must execute on target actor.");
Require(GetSequence(runtime, sourceSequenceId).Position == 5, "Source sequence must advance exactly once.");
Require(GetSequence(runtime, migratedSequenceId).Position == 5, "Migrated sequence must advance exactly once.");

// Unknown targets are rejected before backend side effects or metadata changes.
var unknownTargetSequenceId = SequenceId.New();
await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[]
        {
            new PrefillExecutionStep(unknownTargetSequenceId, model, 4, CompletesPrefill: true)
        }),
    PrefillBindings(unknownTargetSequenceId, 9, 10, 11, 12));

var migrationsBeforeUnknownTarget = sourceBackend.MigrationCalls;
var unknownBefore = GetSequence(runtime, unknownTargetSequenceId);
var unknownVersionBefore = unknownBefore.Version;
var unknownRejected = false;
try
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[]
            {
                new MigrateKvExecutionStep(unknownTargetSequenceId, missingId)
            }),
        EmptyBindings);
}
catch (KeyNotFoundException exception)
    when (exception.Message.Contains("not registered", StringComparison.Ordinal))
{
    unknownRejected = true;
}

Require(unknownRejected, "Migration to an unregistered device must be rejected.");
Require(
    sourceBackend.MigrationCalls == migrationsBeforeUnknownTarget,
    "Unknown target rejection must occur before the source backend migration hook.");
Require(
    GetSequence(runtime, unknownTargetSequenceId).Device == sourceId &&
    GetSequence(runtime, unknownTargetSequenceId).Version == unknownVersionBefore,
    "Unknown target rejection must leave runtime placement/version unchanged.");

// Snapshot state remains owned by the device where it was created. Migrating the
// sequence does not silently make an old backend snapshot restorable elsewhere.
var snapshotSequenceId = SequenceId.New();
await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[]
        {
            new PrefillExecutionStep(snapshotSequenceId, model, 4, CompletesPrefill: true)
        }),
    PrefillBindings(snapshotSequenceId, 20, 21, 22, 23));

var snapshotResult = await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new SnapshotKvExecutionStep(snapshotSequenceId) }),
    EmptyBindings);
var snapshotId = snapshotResult.Snapshots.Single();

await runtime.ExecuteAsync(
    new CompiledExecutionPlan(
        Guid.NewGuid(),
        0,
        new ExecutionStep[] { new MigrateKvExecutionStep(snapshotSequenceId, targetId) }),
    EmptyBindings);

var crossDeviceRestoreRejected = false;
try
{
    await runtime.ExecuteAsync(
        new CompiledExecutionPlan(
            Guid.NewGuid(),
            0,
            new ExecutionStep[] { new RestoreKvExecutionStep(snapshotSequenceId, snapshotId) }),
        EmptyBindings);
}
catch (InvalidOperationException exception)
    when (exception.Message.Contains("snapshot state belongs to device", StringComparison.Ordinal))
{
    crossDeviceRestoreRejected = true;
}

Require(crossDeviceRestoreRejected, "Cross-device restore of a source-owned snapshot must be rejected.");
Require(targetBackend.RestoreCalls == 0, "Snapshot locality rejection must happen before target backend restore work.");
Require(await runtime.ReleaseSnapshotAsync(snapshotId), "Snapshot release must succeed after sequence migration.");
Require(sourceBackend.ReleaseSnapshotCalls == 1, "Snapshot release must route back to its owning source actor.");

// Sequence cleanup follows current placement, including migrated sequences.
foreach (var sequenceId in new[]
{
    sourceSequenceId,
    migratedSequenceId,
    unknownTargetSequenceId,
    snapshotSequenceId
})
{
    GetSequence(runtime, sequenceId).TransitionTo(SequenceStatus.Cancelled);
    Require(await runtime.ReleaseSequenceAsync(sequenceId), $"Sequence {sequenceId} must release successfully.");
}

Require(sourceBackend.ReleaseSequenceCalls == 2, "Source actor must release the two source-resident sequences.");
Require(targetBackend.ReleaseSequenceCalls == 2, "Target actor must release both migrated sequences.");
Require(runtime.SequenceCount == 0, "All runtime sequence metadata must be reclaimed.");
Require(runtime.SnapshotCount == 0, "All runtime snapshot metadata must be reclaimed.");
Require(runtime.KvPages.AllocatedPages == 0, "All metadata KV pages must return to the pool.");
Require(fabric.SequenceCount == 0, "Shared backend fabric must not retain sequence state after cleanup.");
Require(fabric.SnapshotCount == 0, "Shared backend fabric must not retain snapshot state after cleanup.");

Console.WriteLine(
    $"Fission multi-device runtime specs passed: sourceDecode={sourceBackend.DecodeCalls}, " +
    $"targetDecode={targetBackend.DecodeCalls}, migrations={sourceBackend.MigrationCalls}.");

sealed class SharedInferenceFabric
{
    private readonly object _gate = new();
    private readonly HashSet<DeviceId> _devices;
    private readonly Dictionary<SequenceId, int> _sequenceStates = new();
    private readonly Dictionary<SequenceId, DeviceId> _placements = new();
    private readonly Dictionary<KvSnapshotId, SnapshotState> _snapshots = new();

    public SharedInferenceFabric(params DeviceId[] devices)
    {
        _devices = devices.ToHashSet();
    }

    public int SequenceCount
    {
        get
        {
            lock (_gate)
            {
                return _sequenceStates.Count;
            }
        }
    }

    public int SnapshotCount
    {
        get
        {
            lock (_gate)
            {
                return _snapshots.Count;
            }
        }
    }

    public BackendStepResult Prefill(DeviceId device, PrefillItem item)
    {
        lock (_gate)
        {
            if (_placements.TryGetValue(item.SequenceId, out var placement) && placement != device)
            {
                throw new InvalidOperationException(
                    $"Prefill for {item.SequenceId} reached {device}, but placement is {placement}.");
            }

            var next = _sequenceStates.TryGetValue(item.SequenceId, out var current)
                ? checked(current + item.Tokens.Length)
                : item.Tokens.Length;
            _sequenceStates[item.SequenceId] = next;
            _placements[item.SequenceId] = device;
            return new BackendStepResult(item.SequenceId, next);
        }
    }

    public BackendStepResult Decode(DeviceId device, DecodeItem item)
    {
        lock (_gate)
        {
            RequirePlacement(device, item.SequenceId);
            if (!_sequenceStates.TryGetValue(item.SequenceId, out var current))
            {
                throw new InvalidOperationException($"Missing state for {item.SequenceId}.");
            }

            if (current != item.Position)
            {
                throw new InvalidOperationException(
                    $"Decode position {item.Position} does not match fabric state {current} for {item.SequenceId}.");
            }

            var next = checked(current + 1);
            _sequenceStates[item.SequenceId] = next;
            return new BackendStepResult(item.SequenceId, next);
        }
    }

    public void Migrate(DeviceId source, SequenceId sequenceId, DeviceId target)
    {
        lock (_gate)
        {
            RequirePlacement(source, sequenceId);
            if (!_devices.Contains(target))
            {
                throw new InvalidOperationException($"Fabric target {target} is unavailable.");
            }

            _placements[sequenceId] = target;
        }
    }

    public DeviceId GetPlacement(SequenceId sequenceId)
    {
        lock (_gate)
        {
            return _placements[sequenceId];
        }
    }

    public void Snapshot(DeviceId device, SequenceId sequenceId, KvSnapshotId snapshotId)
    {
        lock (_gate)
        {
            RequirePlacement(device, sequenceId);
            _snapshots.Add(
                snapshotId,
                new SnapshotState(_sequenceStates[sequenceId], device));
        }
    }

    public void Restore(DeviceId device, SequenceId sequenceId, KvSnapshotId snapshotId)
    {
        lock (_gate)
        {
            RequirePlacement(device, sequenceId);
            if (!_snapshots.TryGetValue(snapshotId, out var snapshot))
            {
                throw new InvalidOperationException($"Missing snapshot {snapshotId}.");
            }

            if (snapshot.Device != device)
            {
                throw new InvalidOperationException(
                    $"Snapshot {snapshotId} belongs to {snapshot.Device}, not {device}.");
            }

            _sequenceStates[sequenceId] = snapshot.Position;
        }
    }

    public void ReleaseSnapshot(DeviceId device, KvSnapshotId snapshotId)
    {
        lock (_gate)
        {
            if (!_snapshots.TryGetValue(snapshotId, out var snapshot) || snapshot.Device != device)
            {
                throw new InvalidOperationException(
                    $"Snapshot {snapshotId} is not owned by device {device}.");
            }

            _snapshots.Remove(snapshotId);
        }
    }

    public void ReleaseSequence(DeviceId device, SequenceId sequenceId)
    {
        lock (_gate)
        {
            RequirePlacement(device, sequenceId);
            _sequenceStates.Remove(sequenceId);
            _placements.Remove(sequenceId);
        }
    }

    private void RequirePlacement(DeviceId device, SequenceId sequenceId)
    {
        if (!_placements.TryGetValue(sequenceId, out var placement))
        {
            throw new InvalidOperationException($"Missing placement for {sequenceId}.");
        }

        if (placement != device)
        {
            throw new InvalidOperationException(
                $"Sequence {sequenceId} is placed on {placement}, not {device}.");
        }
    }

    private readonly record struct SnapshotState(int Position, DeviceId Device);
}

sealed class FabricBackend : IInferenceBackend
{
    private readonly SharedInferenceFabric _fabric;
    private bool _initialized;

    public FabricBackend(DeviceId device, SharedInferenceFabric fabric)
    {
        Device = device;
        _fabric = fabric;
    }

    public string Name => $"fabric-spec/{Device}";
    public DeviceId Device { get; }
    public int PrefillCalls { get; private set; }
    public int DecodeCalls { get; private set; }
    public int MigrationCalls { get; private set; }
    public int RestoreCalls { get; private set; }
    public int ReleaseSnapshotCalls { get; private set; }
    public int ReleaseSequenceCalls { get; private set; }

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
            cancellationToken.ThrowIfCancellationRequested();
            PrefillCalls++;
            results[index] = _fabric.Prefill(Device, batch.Items[index]);
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
            cancellationToken.ThrowIfCancellationRequested();
            DecodeCalls++;
            results[index] = _fabric.Decode(Device, batch.Items[index]);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        MigrationCalls++;
        _fabric.Migrate(Device, sequenceId, targetDevice);
        return ValueTask.CompletedTask;
    }

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _fabric.Snapshot(Device, sequenceId, snapshotId);
        return ValueTask.CompletedTask;
    }

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        RestoreCalls++;
        _fabric.Restore(Device, sequenceId, snapshotId);
        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        ReleaseSnapshotCalls++;
        _fabric.ReleaseSnapshot(Device, snapshotId);
        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        ReleaseSequenceCalls++;
        _fabric.ReleaseSequence(Device, sequenceId);
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
            throw new InvalidOperationException($"Backend {Device} is not initialized.");
        }
    }
}
