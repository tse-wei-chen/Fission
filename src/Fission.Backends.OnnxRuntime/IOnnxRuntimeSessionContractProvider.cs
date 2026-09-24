namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Optional adapter capability for declaring the ONNX graph signature it expects.
/// The generic backend validates this contract against the live InferenceSession
/// before adapter initialization allocates or registers any model-owned state.
/// </summary>
public interface IOnnxRuntimeSessionContractProvider
{
    OnnxSessionContract SessionContract { get; }
}
