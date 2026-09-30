using Fission.Abstractions;

namespace Fission.Runtime.Execution;

/// <summary>
/// Result of one device-actor memory reclaim barrier.
/// </summary>
public readonly record struct RuntimeDeviceMemoryReclaimResult(
    DeviceId Device,
    long ReleasedBytes,
    long ReclaimableBytes,
    long ReservedBytes);

public sealed partial class ExecutionPlanExecutor
{
    internal async ValueTask<RuntimeDeviceMemoryReclaimResult>
        ReclaimDeviceMemoryCoreAsync(
            DeviceId device,
            long targetReclaimableBytes,
            CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegative(targetReclaimableBytes);

        var actor = _devices.ResolveRegistered(device);
        var result = await actor.ReclaimDeviceMemoryAsync(
                targetReclaimableBytes,
                cancellationToken)
            .ConfigureAwait(false);

        return new RuntimeDeviceMemoryReclaimResult(
            device,
            result.ReleasedBytes,
            result.ReclaimableBytes,
            result.ReservedBytes);
    }
}

public static class ExecutionPlanExecutorMemoryReclaimExtensions
{
    /// <summary>
    /// Runs a queue-order control barrier on one registered execution actor and
    /// asks its backend to reduce reclaimable physical memory toward the target.
    /// Live inference state must not be revoked by this operation.
    /// </summary>
    public static ValueTask<RuntimeDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
        this ExecutionPlanExecutor executor,
        DeviceId device,
        long targetReclaimableBytes = 0,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(executor);
        return executor.ReclaimDeviceMemoryCoreAsync(
            device,
            targetReclaimableBytes,
            cancellationToken);
    }
}
