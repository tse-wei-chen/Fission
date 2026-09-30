using Fission.Abstractions;

namespace Fission.Runtime.Execution;

public static class ExecutionPlanExecutorDeviceExtensions
{
    /// <summary>
    /// Resolves the physical device actor that would execute the next inference
    /// step for a sequence. Unknown sequences resolve to the runtime default device.
    /// </summary>
    public static DeviceId GetExecutionDevice(
        this ExecutionPlanExecutor executor,
        SequenceId sequenceId)
    {
        ArgumentNullException.ThrowIfNull(executor);
        return executor.ResolveExecutionDevice(sequenceId);
    }
}
