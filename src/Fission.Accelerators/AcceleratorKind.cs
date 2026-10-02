namespace Fission.Accelerators;

/// <summary>
/// Stable physical accelerator categories. This describes what a device is, not
/// which software stack executes work on it.
/// </summary>
public enum AcceleratorKind
{
    Cpu,
    Gpu,
    Npu,
    Tpu,
    Fpga,
    Custom
}

/// <summary>
/// High-level memory topology exposed by a device discovery provider.
/// </summary>
public enum DeviceMemoryTopology
{
    Unknown,
    Host,
    Dedicated,
    Unified
}
