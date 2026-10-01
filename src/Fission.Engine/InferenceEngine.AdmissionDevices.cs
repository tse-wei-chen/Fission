using Fission.Abstractions;
using Fission.Abstractions.Scheduling;

namespace Fission.Engine;

public sealed partial class InferenceEngine
{
    private static DeviceId[] BuildAdmissionDevices(
        IReadOnlyList<SchedulingCandidate> candidates)
    {
        DeviceId singleDevice = default;
        var hasSingleDevice = false;
        List<DeviceId>? multipleDevices = null;

        for (var index = 0; index < candidates.Count; index++)
        {
            if (candidates[index].ExecutionDevice is not { } device)
            {
                continue;
            }

            if (multipleDevices is not null)
            {
                if (!ContainsDevice(multipleDevices, device))
                {
                    multipleDevices.Add(device);
                }

                continue;
            }

            if (!hasSingleDevice)
            {
                singleDevice = device;
                hasSingleDevice = true;
                continue;
            }

            if (singleDevice.Equals(device))
            {
                continue;
            }

            multipleDevices = new List<DeviceId>(2)
            {
                singleDevice,
                device
            };
        }

        if (multipleDevices is null)
        {
            return hasSingleDevice
                ? new[] { singleDevice }
                : Array.Empty<DeviceId>();
        }

        multipleDevices.Sort(
            static (left, right) =>
                StringComparer.Ordinal.Compare(left.Value, right.Value));
        return multipleDevices.ToArray();
    }

    private static bool ContainsDevice(
        IReadOnlyList<DeviceId> devices,
        DeviceId device)
    {
        for (var index = 0; index < devices.Count; index++)
        {
            if (devices[index].Equals(device))
            {
                return true;
            }
        }

        return false;
    }
}
