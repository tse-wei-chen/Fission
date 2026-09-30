namespace Fission.Abstractions.Execution;

/// <summary>
/// Physical device-memory residency reported by an inference backend.
/// This is deliberately separate from scheduler KV-byte accounting: active bytes
/// describe currently leased native memory, reclaimable bytes describe idle cache
/// that may be released without revoking live inference state, and reserved bytes
/// are the physical sum still resident on the device.
/// </summary>
public readonly record struct InferenceDeviceMemoryPressure(
    long ActiveBytes,
    long ReclaimableBytes,
    long ReservedBytes,
    long PeakReservedBytes);

/// <summary>
/// Optional backend capability for observing physical device-memory residency.
/// Implementations must be safe to query concurrently with serialized inference
/// execution. Returning false means the backend does not expose trustworthy
/// physical-memory accounting for its current configuration.
/// </summary>
public interface IInferenceDeviceMemoryPressureSource
{
    bool TryGetDeviceMemoryPressure(
        out InferenceDeviceMemoryPressure pressure);
}
