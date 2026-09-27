using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Internal bridge between a migration-capable ONNX backend host and a
/// model-specific execution adapter. Device/model identity remains owned by the
/// backend; physical sequence-state encoding remains owned by the adapter/binding.
/// </summary>
internal interface IOnnxRuntimeSequenceMigrationAdapter
{
    IReadOnlyList<SequenceMigrationTransportCapability> GetSequenceMigrationTransportCapabilities(
        ModelId modelId,
        DeviceId localDevice,
        DeviceId peerDevice);

    ValueTask<long> EstimateSequenceMigrationBytesAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default);

    ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        ModelId modelId,
        DeviceId sourceDevice,
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan? transportPlan,
        CancellationToken cancellationToken = default);

    ValueTask ImportSequenceMigrationAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default);

    ValueTask CommitSequenceMigrationAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default);

    ValueTask AbortSequenceMigrationAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default);
}
