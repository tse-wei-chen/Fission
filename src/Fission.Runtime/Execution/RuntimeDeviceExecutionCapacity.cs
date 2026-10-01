using Fission.Abstractions;

namespace Fission.Runtime.Execution;

/// <summary>
/// Physical device-actor execution limits exposed without leaking the actor itself.
/// InferenceCreditCapacity bounds the number of inference items that may be owned
/// by the actor's weighted credit gate, including one atomic scheduler envelope.
/// MaxBackendBatchSize independently bounds one contiguous same-kind backend call;
/// a larger admitted envelope is split into multiple backend micro-batches.
/// </summary>
public readonly record struct RuntimeDeviceExecutionCapacity(
    DeviceId Device,
    int InferenceCreditCapacity,
    int MaxBackendBatchSize);

public static class ExecutionPlanExecutorDeviceCapacityExtensions
{
    /// <summary>
    /// Returns execution-capacity dimensions for every registered physical actor,
    /// ordered by physical device id.
    /// </summary>
    public static IReadOnlyList<RuntimeDeviceExecutionCapacity>
        GetDeviceExecutionCapacities(this ExecutionPlanExecutor executor)
    {
        ArgumentNullException.ThrowIfNull(executor);
        return executor.GetDeviceExecutionCapacitiesCore(
            executor.RegisteredDevices.ToArray());
    }

    /// <summary>
    /// Returns execution-capacity dimensions only for the requested physical
    /// actors. Duplicate ids are normalized and output is ordered by device id.
    /// </summary>
    public static IReadOnlyList<RuntimeDeviceExecutionCapacity>
        GetDeviceExecutionCapacities(
            this ExecutionPlanExecutor executor,
            IReadOnlyList<DeviceId> devices)
    {
        ArgumentNullException.ThrowIfNull(executor);
        ArgumentNullException.ThrowIfNull(devices);
        return executor.GetDeviceExecutionCapacitiesCore(devices);
    }

    /// <summary>
    /// Returns execution-capacity dimensions for one registered physical actor.
    /// </summary>
    public static RuntimeDeviceExecutionCapacity GetDeviceExecutionCapacity(
        this ExecutionPlanExecutor executor,
        DeviceId device)
    {
        ArgumentNullException.ThrowIfNull(executor);
        return executor.GetDeviceExecutionCapacityCore(device);
    }
}

public sealed partial class ExecutionPlanExecutor
{
    internal IReadOnlyCollection<DeviceId> RegisteredDevices => _devices.Devices;

    internal IReadOnlyList<RuntimeDeviceExecutionCapacity>
        GetDeviceExecutionCapacitiesCore(IReadOnlyList<DeviceId> devices)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(devices);

        var normalized = devices
            .Distinct()
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .ToArray();
        var result = new RuntimeDeviceExecutionCapacity[normalized.Length];

        for (var index = 0; index < normalized.Length; index++)
        {
            result[index] = GetDeviceExecutionCapacityCore(normalized[index]);
        }

        return result;
    }

    internal RuntimeDeviceExecutionCapacity GetDeviceExecutionCapacityCore(
        DeviceId device)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var actor = _devices.ResolveRegistered(device);
        return new RuntimeDeviceExecutionCapacity(
            actor.Device,
            actor.InferenceCapacity,
            actor.MaxBackendBatchSize);
    }
}

public sealed partial class ContinuousBatchExecutor
{
    internal int MaxBackendBatchSize => _maxBatchSize;
}
