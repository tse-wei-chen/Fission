namespace Fission.Abstractions.Execution;

/// <summary>
/// Physical device-memory residency reported by an inference backend.
/// This is deliberately separate from scheduler KV-byte accounting:
/// non-reclaimable bytes cannot currently be returned without disturbing live
/// backend state, reclaimable bytes are idle cache that may be released safely,
/// and reserved bytes are their physical sum still resident on the device.
/// </summary>
public readonly record struct InferenceDeviceMemoryPressure(
    long NonReclaimableBytes,
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
