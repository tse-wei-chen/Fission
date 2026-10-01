namespace Fission.Runtime.Execution;

/// <summary>
/// Backward-compatible conservative runtime view of inference item-credit
/// capacity. In a multi-device runtime this is the minimum credit capacity across
/// all registered physical actors; it does not describe backend micro-batch width.
/// Prefer GetDeviceExecutionCapacities when device-local limits matter.
/// </summary>
public readonly record struct RuntimeExecutionCapacity(int MaxInferenceItems);

public static class ExecutionPlanExecutorCapacityExtensions
{
    public static RuntimeExecutionCapacity GetExecutionCapacity(
        this ExecutionPlanExecutor executor)
    {
        ArgumentNullException.ThrowIfNull(executor);
        return new RuntimeExecutionCapacity(executor.DeviceInferenceCapacity);
    }
}
