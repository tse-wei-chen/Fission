using Fission.Abstractions;

namespace Fission.Runtime.Execution;

/// <summary>
/// Maps logical device ids to the single-device actors that own backend execution.
/// The registry does not own executor lifetimes; callers remain responsible for
/// disposing every registered ContinuousBatchExecutor.
/// </summary>
public sealed class ExecutionDeviceRegistry
{
    private readonly IReadOnlyDictionary<DeviceId, ContinuousBatchExecutor> _devices;

    public ExecutionDeviceRegistry(
        ContinuousBatchExecutor defaultDevice,
        params ContinuousBatchExecutor[] additionalDevices)
    {
        ArgumentNullException.ThrowIfNull(defaultDevice);
        ArgumentNullException.ThrowIfNull(additionalDevices);

        var devices = new Dictionary<DeviceId, ContinuousBatchExecutor>
        {
            [defaultDevice.Device] = defaultDevice
        };

        foreach (var device in additionalDevices)
        {
            ArgumentNullException.ThrowIfNull(device);
            if (!devices.TryAdd(device.Device, device))
            {
                throw new InvalidOperationException(
                    $"Execution device {device.Device} is registered more than once.");
            }
        }

        _devices = devices;
        DefaultDevice = defaultDevice.Device;
    }

    public DeviceId DefaultDevice { get; }
    public int Count => _devices.Count;
    public IReadOnlyCollection<DeviceId> Devices => _devices.Keys.ToArray();

    public bool Contains(DeviceId device) => _devices.ContainsKey(device);

    internal ContinuousBatchExecutor Resolve(DeviceId device)
    {
        if (_devices.TryGetValue(device, out var executor))
        {
            return executor;
        }

        throw new KeyNotFoundException(
            $"Execution device {device} is not registered in this runtime.");
    }

    internal int MinimumInferenceCapacity =>
        _devices.Values.Min(static device => device.InferenceCapacity);

    internal int GetInferenceCapacity(DeviceId device) =>
        Resolve(device).InferenceCapacity;
}
