namespace Fission.Abstractions.Execution;

/// <summary>
/// Result of one best-effort physical device-memory reclaim operation.
/// Released bytes are bytes returned to the device during this operation;
/// reclaimable and reserved bytes describe the source after reclaim completes.
/// </summary>
public readonly record struct InferenceDeviceMemoryReclaimResult(
    long ReleasedBytes,
    long ReclaimableBytes,
    long ReservedBytes);

/// <summary>
/// Optional backend capability for safely releasing reclaimable physical device
/// memory without revoking live inference state. Calls are serialized by the
/// device actor so a backend may treat reclaim as a queue-order control barrier.
/// </summary>
public interface IInferenceDeviceMemoryReclaimer
{
    ValueTask<InferenceDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
        long targetReclaimableBytes,
        CancellationToken cancellationToken = default);
}
