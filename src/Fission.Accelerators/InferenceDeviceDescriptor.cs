using Fission.Abstractions;

namespace Fission.Accelerators;

/// <summary>
/// Provider-neutral metadata for one executable inference device.
///
/// DeviceId remains the runtime identity and is deliberately opaque. Kind,
/// provider and capabilities are metadata; consumers must not infer them by
/// parsing DeviceId prefixes.
/// </summary>
public sealed record InferenceDeviceDescriptor
{
    public InferenceDeviceDescriptor(
        DeviceId device,
        AcceleratorKind kind,
        ExecutionProviderId provider,
        InferenceDeviceCapabilities capabilities = InferenceDeviceCapabilities.None,
        DeviceMemoryTopology memoryTopology = DeviceMemoryTopology.Unknown,
        int? ordinal = null,
        long? totalMemoryBytes = null,
        string? vendor = null,
        string? architecture = null)
    {
        if (string.IsNullOrWhiteSpace(device.Value))
        {
            throw new ArgumentException("Device id cannot be empty.", nameof(device));
        }

        if (string.IsNullOrWhiteSpace(provider.Value))
        {
            throw new ArgumentException(
                "Execution provider id cannot be empty.",
                nameof(provider));
        }

        if (ordinal is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal), "Device ordinal cannot be negative.");
        }

        if (totalMemoryBytes is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(totalMemoryBytes),
                "Total memory must be positive when specified.");
        }

        Device = device;
        Kind = kind;
        Provider = provider;
        Capabilities = capabilities;
        MemoryTopology = memoryTopology;
        Ordinal = ordinal;
        TotalMemoryBytes = totalMemoryBytes;
        Vendor = NormalizeOptional(vendor);
        Architecture = NormalizeOptional(architecture);
    }

    public DeviceId Device { get; }
    public AcceleratorKind Kind { get; }
    public ExecutionProviderId Provider { get; }
    public InferenceDeviceCapabilities Capabilities { get; }
    public DeviceMemoryTopology MemoryTopology { get; }
    public int? Ordinal { get; }
    public long? TotalMemoryBytes { get; }
    public string? Vendor { get; }
    public string? Architecture { get; }

    public bool Supports(InferenceDeviceCapabilities capability) =>
        (Capabilities & capability) == capability;

    private static string? NormalizeOptional(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : value.Trim();
}
