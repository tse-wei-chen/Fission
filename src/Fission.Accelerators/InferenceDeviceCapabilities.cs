namespace Fission.Accelerators;

/// <summary>
/// Optional accelerator capabilities discovered from a concrete provider/device
/// pairing. Capabilities are intentionally descriptive rather than prescriptive:
/// runtime and scheduler policy may consume them later without assuming a device
/// kind implies a feature.
/// </summary>
[Flags]
public enum InferenceDeviceCapabilities : ulong
{
    None = 0,
    BatchedPrefill = 1UL << 0,
    BatchedDecode = 1UL << 1,
    ResidentKv = 1UL << 2,
    Snapshot = 1UL << 3,
    Fork = 1UL << 4,
    Migration = 1UL << 5,
    DeviceMemoryAccounting = 1UL << 6,
    DeviceMemoryReclaim = 1UL << 7,
    AsyncTransfer = 1UL << 8,
    DynamicShapes = 1UL << 9,
    QuantizedExecution = 1UL << 10,
    UnifiedMemory = 1UL << 11
}
