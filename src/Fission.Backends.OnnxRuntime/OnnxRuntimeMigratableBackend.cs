using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Opt-in ONNX Runtime backend host for adapters that own a concrete physical
/// sequence-migration codec. The ordinary OnnxRuntimeBackend remains deliberately
/// non-transactional so adapters without migratable state keep their legacy
/// behavior and are never misclassified by runtime interface checks.
/// </summary>
public sealed class OnnxRuntimeMigratableBackend :
    IInferenceBackend,
    ISequenceMigrationTransportBackend
{
    private readonly OnnxRuntimeBackend _inner;
    private readonly IOnnxRuntimeSequenceMigrationAdapter _migration;

    public OnnxRuntimeMigratableBackend(
        OnnxRuntimeBackendOptions options,
        IOnnxRuntimeExecutionAdapter adapter,
        Func<SessionOptions>? sessionOptionsFactory = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(adapter);

        _migration = adapter as IOnnxRuntimeSequenceMigrationAdapter ??
            throw new ArgumentException(
                $"ONNX adapter '{adapter.Name}' does not expose a physical sequence-migration implementation.",
                nameof(adapter));
        if (!_migration.SupportsSequenceMigration)
        {
            throw new ArgumentException(
                $"ONNX adapter '{adapter.Name}' is not configured with a migratable physical-state codec.",
                nameof(adapter));
        }

        _inner = new OnnxRuntimeBackend(options, adapter, sessionOptionsFactory);
    }

    public string Name => _inner.Name;
    public DeviceId Device => _inner.Device;
    public ModelId ModelId => _inner.ModelId;
    public bool IsInitialized => _inner.IsInitialized;

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default) =>
        _inner.InitializeAsync(cancellationToken);

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default) =>
        _inner.PrefillAsync(batch, cancellationToken);

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default) =>
        _inner.DecodeAsync(batch, cancellationToken);

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        _inner.SnapshotSequenceAsync(sequenceId, snapshotId, cancellationToken);

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default) =>
        _inner.ForkSequenceAsync(parentSequenceId, branchSequenceIds, cancellationToken);

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        _inner.RestoreSequenceAsync(sequenceId, snapshotId, cancellationToken);

    public ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default) =>
        _inner.MigrateSequenceAsync(sequenceId, targetDevice, cancellationToken);

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        _inner.ReleaseSnapshotAsync(snapshotId, cancellationToken);

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default) =>
        _inner.ReleaseSequenceAsync(sequenceId, cancellationToken);

    public IReadOnlyList<SequenceMigrationTransportCapability>
        GetSequenceMigrationTransportCapabilities(DeviceId peerDevice)
    {
        EnsureInitialized();
        return _migration.GetSequenceMigrationTransportCapabilities(
            ModelId,
            Device,
            peerDevice);
    }

    public ValueTask<long> EstimateSequenceMigrationBytesAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        return _migration.EstimateSequenceMigrationBytesAsync(
            ModelId,
            Device,
            sequenceId,
            targetDevice,
            cancellationToken);
    }

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        return _migration.PrepareSequenceMigrationAsync(
            ModelId,
            Device,
            sequenceId,
            targetDevice,
            transportPlan: null,
            cancellationToken);
    }

    public ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        ArgumentNullException.ThrowIfNull(transportPlan);
        return _migration.PrepareSequenceMigrationAsync(
            ModelId,
            Device,
            sequenceId,
            targetDevice,
            transportPlan,
            cancellationToken);
    }

    public ValueTask ImportSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        return _migration.ImportSequenceMigrationAsync(
            ModelId,
            Device,
            transfer,
            cancellationToken);
    }

    public ValueTask CommitSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        return _migration.CommitSequenceMigrationAsync(
            ModelId,
            Device,
            transfer,
            cancellationToken);
    }

    public ValueTask AbortSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        return _migration.AbortSequenceMigrationAsync(
            ModelId,
            Device,
            transfer,
            cancellationToken);
    }

    private void EnsureInitialized()
    {
        if (!IsInitialized)
        {
            throw new InvalidOperationException(
                "ONNX Runtime migratable backend has not been initialized.");
        }
    }

    public ValueTask DisposeAsync() => _inner.DisposeAsync();
}
