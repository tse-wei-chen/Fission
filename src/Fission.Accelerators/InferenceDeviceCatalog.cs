using Fission.Abstractions;

namespace Fission.Accelerators;

/// <summary>
/// Startup-oriented registry of discovered accelerator descriptors. Registration
/// is deterministic and duplicate DeviceIds are rejected rather than silently
/// replacing another provider's device.
/// </summary>
public sealed class InferenceDeviceCatalog
{
    private readonly Dictionary<DeviceId, InferenceDeviceDescriptor> _devices = new();

    public int Count => _devices.Count;

    public void Register(InferenceDeviceDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);

        if (!_devices.TryAdd(descriptor.Device, descriptor))
        {
            var existing = _devices[descriptor.Device];
            throw new InvalidOperationException(
                $"Device '{descriptor.Device}' is already registered by provider " +
                $"'{existing.Provider}'.");
        }
    }

    public bool TryGet(
        DeviceId device,
        out InferenceDeviceDescriptor? descriptor) =>
        _devices.TryGetValue(device, out descriptor);

    public IReadOnlyList<InferenceDeviceDescriptor> Snapshot() =>
        _devices.Values
            .OrderBy(static descriptor => descriptor.Device.Value, StringComparer.Ordinal)
            .ToArray();

    public IReadOnlyList<InferenceDeviceDescriptor> FindByKind(
        AcceleratorKind kind) =>
        _devices.Values
            .Where(descriptor => descriptor.Kind == kind)
            .OrderBy(static descriptor => descriptor.Device.Value, StringComparer.Ordinal)
            .ToArray();

    public IReadOnlyList<InferenceDeviceDescriptor> FindByProvider(
        ExecutionProviderId provider) =>
        _devices.Values
            .Where(descriptor => descriptor.Provider == provider)
            .OrderBy(static descriptor => descriptor.Device.Value, StringComparer.Ordinal)
            .ToArray();

    public static async ValueTask<InferenceDeviceCatalog> DiscoverAsync(
        IEnumerable<IInferenceDeviceDiscovery> discoveryProviders,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(discoveryProviders);

        var catalog = new InferenceDeviceCatalog();
        foreach (var discovery in discoveryProviders)
        {
            ArgumentNullException.ThrowIfNull(discovery);
            cancellationToken.ThrowIfCancellationRequested();

            var discovered = await discovery.DiscoverAsync(cancellationToken)
                .ConfigureAwait(false)
                ?? throw new InvalidOperationException(
                    $"Device discovery provider '{discovery.Name}' returned null.");

            foreach (var descriptor in discovered)
            {
                catalog.Register(descriptor);
            }
        }

        return catalog;
    }
}
