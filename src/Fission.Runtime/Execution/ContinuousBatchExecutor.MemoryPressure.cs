using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    internal bool TryGetDeviceMemoryPressure(
        out InferenceDeviceMemoryPressure pressure)
    {
        if (_backend is not IInferenceDeviceMemoryPressureSource source ||
            !source.TryGetDeviceMemoryPressure(out pressure))
        {
            pressure = default;
            return false;
        }

        ValidateDeviceMemoryPressure(pressure);
        return true;
    }

    private static void ValidateDeviceMemoryPressure(
        InferenceDeviceMemoryPressure pressure)
    {
        if (pressure.ActiveBytes < 0 ||
            pressure.ReclaimableBytes < 0 ||
            pressure.ReservedBytes < 0 ||
            pressure.PeakReservedBytes < 0)
        {
            throw new InvalidOperationException(
                "Backend device-memory pressure cannot contain negative byte counts.");
        }

        if (checked(pressure.ActiveBytes + pressure.ReclaimableBytes) !=
            pressure.ReservedBytes)
        {
            throw new InvalidOperationException(
                "Backend device-memory pressure must satisfy reserved = active + reclaimable bytes.");
        }

        if (pressure.PeakReservedBytes < pressure.ReservedBytes)
        {
            throw new InvalidOperationException(
                "Backend peak reserved device memory cannot be smaller than current reserved bytes.");
        }
    }
}
