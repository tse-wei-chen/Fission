using Fission.Abstractions;

namespace Fission.Runtime.Execution;

/// <summary>
/// Physical device-memory residency exposed by one registered execution actor.
/// This feedback is intentionally independent from logical KV-page/KV-byte budget
/// accounting so callers can build a separate device-pressure policy without
/// double-counting sequence state.
/// </summary>
public readonly record struct RuntimeDeviceMemoryPressure(
    DeviceId Device,
    long ActiveBytes,
    long ReclaimableBytes,
    long ReservedBytes,
    long PeakReservedBytes);

public sealed partial class ExecutionPlanExecutor
{
    internal IReadOnlyList<RuntimeDeviceMemoryPressure>
        GetDeviceMemoryPressureCore() =>
        _devices.GetDeviceMemoryPressure();
}

public static class ExecutionPlanExecutorMemoryPressureExtensions
{
    public static IReadOnlyList<RuntimeDeviceMemoryPressure> GetDeviceMemoryPressure(
        this ExecutionPlanExecutor executor)
    {
        ArgumentNullException.ThrowIfNull(executor);
        return executor.GetDeviceMemoryPressureCore();
    }
}
