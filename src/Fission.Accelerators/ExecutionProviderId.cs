namespace Fission.Accelerators;

/// <summary>
/// Open provider identity. Known providers are exposed by
/// <see cref="KnownExecutionProviders"/>, while custom/vendor providers may use
/// any stable non-empty identifier without changing Fission core enums.
/// </summary>
public readonly record struct ExecutionProviderId
{
    public ExecutionProviderId(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        Value = value.Trim();
    }

    public string Value { get; }

    public override string ToString() => Value;
}

public static class KnownExecutionProviders
{
    public static readonly ExecutionProviderId Cpu = new("cpu");
    public static readonly ExecutionProviderId Cuda = new("cuda");
    public static readonly ExecutionProviderId TensorRt = new("tensorrt");
    public static readonly ExecutionProviderId DirectMl = new("directml");
    public static readonly ExecutionProviderId OpenVino = new("openvino");
    public static readonly ExecutionProviderId Qnn = new("qnn");
    public static readonly ExecutionProviderId VitisAi = new("vitis-ai");
    public static readonly ExecutionProviderId CoreMl = new("coreml");
    public static readonly ExecutionProviderId Mps = new("mps");
    public static readonly ExecutionProviderId Pjrt = new("pjrt");
}
