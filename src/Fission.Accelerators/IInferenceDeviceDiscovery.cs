namespace Fission.Accelerators;

/// <summary>
/// Startup-time discovery boundary for provider-specific device probes.
/// Implementations may inspect CUDA, Metal/MPS, OpenVINO, QNN, CoreML, PJRT or
/// another vendor runtime without coupling those dependencies to Fission core.
/// </summary>
public interface IInferenceDeviceDiscovery
{
    string Name { get; }

    ValueTask<IReadOnlyList<InferenceDeviceDescriptor>> DiscoverAsync(
        CancellationToken cancellationToken = default);
}
