using Fission.Abstractions;

namespace Fission.Runtime.Execution;

/// <summary>
/// Maps logical device ids to the single-device actors that own backend execution.
/// The registry does not own executor lifetimes; callers remain responsible for
/// disposing every registered ContinuousBatchExecutor.
///
/// A one-actor registry preserves the existing backend-internal migration model:
/// logical placement may change to another DeviceId while all work remains routed
/// through the same actor. With multiple actors, logical placement must resolve to
/// an explicitly registered actor.
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

    internal ContinuousBatchExecutor ResolveRegistered(DeviceId device)
    {
        if (_devices.TryGetValue(device, out var executor))
        {
            return executor;
        }

        throw new KeyNotFoundException(
            $"Execution device {device} is not registered in this runtime.");
    }

    internal ContinuousBatchExecutor ResolvePlacement(DeviceId placement)
    {
        if (_devices.TryGetValue(placement, out var executor))
        {
            return executor;
        }

        if (_devices.Count == 1)
        {
            return _devices[DefaultDevice];
        }

        throw new KeyNotFoundException(
            $"Execution device {placement} is not registered in this runtime.");
    }

    internal void ValidateMigrationTarget(DeviceId targetDevice)
    {
        if (_devices.Count > 1)
        {
            _ = ResolveRegistered(targetDevice);
        }
    }

    internal int MinimumInferenceCapacity =>
        _devices.Values.Min(static device => device.InferenceCapacity);

    internal int GetInferenceCapacity(DeviceId actorDevice) =>
        ResolveRegistered(actorDevice).InferenceCapacity;

    internal IReadOnlyList<RuntimeDeviceMemoryPressure> GetDeviceMemoryPressure() =>
        GetDeviceMemoryPressureNormalized(
            _devices.Keys
                .OrderBy(static device => device.Value, StringComparer.Ordinal)
                .ToArray());

    internal IReadOnlyList<RuntimeDeviceMemoryPressure> GetDeviceMemoryPressure(
        IReadOnlyList<DeviceId> devices)
    {
        ArgumentNullException.ThrowIfNull(devices);

        var normalized = devices
            .Distinct()
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .ToArray();
        return GetDeviceMemoryPressureNormalized(normalized);
    }

    internal IReadOnlyList<RuntimeDeviceMemoryPressure> GetDeviceMemoryPressureNormalized(
        IReadOnlyList<DeviceId> devices)
    {
        var pressure = new List<RuntimeDeviceMemoryPressure>(devices.Count);

        foreach (var deviceId in devices)
        {
            var executor = ResolveRegistered(deviceId);
            if (!executor.TryGetDeviceMemoryPressure(out var snapshot))
            {
                continue;
            }

            pressure.Add(new RuntimeDeviceMemoryPressure(
                deviceId,
                snapshot.ActiveBytes,
                snapshot.ReclaimableBytes,
                snapshot.ReservedBytes,
                snapshot.PeakReservedBytes));
        }

        return pressure;
    }
}
