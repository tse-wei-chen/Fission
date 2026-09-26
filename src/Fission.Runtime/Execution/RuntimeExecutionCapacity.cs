namespace Fission.Runtime.Execution;

/// <summary>
/// Stable runtime feedback describing the maximum number of inference items that
/// one atomic scheduler submission may contain for the current execution target.
/// This intentionally exposes capacity rather than the device actor itself.
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
