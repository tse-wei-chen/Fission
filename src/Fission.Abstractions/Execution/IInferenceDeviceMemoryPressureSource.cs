namespace Fission.Abstractions.Execution;

/// <summary>
/// Physical device-memory residency reported by an inference backend.
/// This is deliberately separate from scheduler KV-byte accounting. Active bytes
/// are currently non-reclaimable resident bytes for the reporting source,
/// reclaimable bytes are idle cache that may be released safely, and reserved
/// bytes are their physical sum still resident on the device.
/// </summary>
public readonly record struct InferenceDeviceMemoryPressure(
    long ActiveBytes,
    long ReclaimableBytes,
    long ReservedBytes,
    long PeakReservedBytes)
{
    /// <summary>
    /// Policy-oriented alias for bytes that cannot currently be reclaimed without
    /// disturbing live backend state. A future aggregate source may include model
    /// weights or graph/workspace residency here in addition to active leases.
    /// </summary>
    public long NonReclaimableBytes => ActiveBytes;
}

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
