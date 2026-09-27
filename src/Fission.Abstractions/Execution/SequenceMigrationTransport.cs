namespace Fission.Abstractions.Execution;

/// <summary>
/// Broad transport families used for migration planning and observability.
/// TransportId remains the compatibility key; Kind is descriptive metadata.
/// </summary>
public enum SequenceMigrationTransportKind
{
    DirectDevice,
    SharedMemory,
    HostStaging,
    RemoteMemory
}

/// <summary>
/// One transport path a backend can use with a specific peer device.
/// MaxTransferBytes == 0 means the backend does not impose a per-transfer size cap.
/// EstimatedBandwidthBytesPerSecond must be positive and EstimatedFixedLatency
/// must be non-negative for planner use.
/// </summary>
public sealed record SequenceMigrationTransportCapability(
    string TransportId,
    SequenceMigrationTransportKind Kind,
    long MaxTransferBytes,
    long EstimatedBandwidthBytesPerSecond,
    TimeSpan EstimatedFixedLatency,
    int Preference = 0);

/// <summary>
/// Concrete transport decision shared by the source and target backends.
/// Effective limits are the conservative intersection of both peers.
/// </summary>
public sealed record SequenceMigrationTransportPlan(
    string TransportId,
    SequenceMigrationTransportKind Kind,
    long EstimatedBytes,
    long EffectiveMaxTransferBytes,
    long EffectiveBandwidthBytesPerSecond,
    TimeSpan EffectiveFixedLatency,
    TimeSpan EstimatedDuration,
    int Preference);

/// <summary>
/// Optional extension implemented by backends/fabrics that expose physical
/// migration transport information to a topology-aware runtime.
///
/// Capabilities are peer-specific: both source and target must advertise the same
/// TransportId and Kind before the planner may select that path. The source owns
/// byte estimation because only it necessarily has the authoritative live state.
///
/// A transport-aware prepare receives the exact runtime-selected plan. Its returned
/// transfer must expose an equal TransportPlan so import/commit can verify that the
/// backend did not silently substitute a different physical route.
/// </summary>
public interface ISequenceMigrationTransportBackend : ISequenceMigrationBackend
{
    IReadOnlyList<SequenceMigrationTransportCapability> GetSequenceMigrationTransportCapabilities(
        DeviceId peerDevice);

    ValueTask<long> EstimateSequenceMigrationBytesAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default);

    ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan transportPlan,
        CancellationToken cancellationToken = default);
}
