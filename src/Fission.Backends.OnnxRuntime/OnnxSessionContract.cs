using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Declarative ONNX tensor requirement used to validate a live InferenceSession
/// before an execution adapter begins owning model state.
/// </summary>
public sealed record OnnxTensorContract(
    string LogicalName,
    string TensorName,
    int? Rank = null,
    TensorElementType? ElementType = null);

/// <summary>
/// Model/export-specific session signature. Extra tensors in the ONNX graph are
/// allowed; every tensor declared here must exist and satisfy the requested shape
/// metadata.
/// </summary>
public sealed record OnnxSessionContract(
    IReadOnlyList<OnnxTensorContract> Inputs,
    IReadOnlyList<OnnxTensorContract> Outputs)
{
    public void Validate(InferenceSession session)
    {
        ArgumentNullException.ThrowIfNull(session);
        ValidateContracts(Inputs, session.InputMetadata, "input");
        ValidateContracts(Outputs, session.OutputMetadata, "output");
    }

    private static void ValidateContracts(
        IReadOnlyList<OnnxTensorContract> contracts,
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string direction)
    {
        var logicalNames = new HashSet<string>(StringComparer.Ordinal);
        var tensorNames = new HashSet<string>(StringComparer.Ordinal);

        foreach (var contract in contracts)
        {
            if (string.IsNullOrWhiteSpace(contract.LogicalName))
            {
                throw new InvalidOperationException(
                    $"ONNX {direction} contract contains an empty logical name.");
            }

            if (string.IsNullOrWhiteSpace(contract.TensorName))
            {
                throw new InvalidOperationException(
                    $"ONNX {direction} contract '{contract.LogicalName}' contains an empty tensor name.");
            }

            if (!logicalNames.Add(contract.LogicalName))
            {
                throw new InvalidOperationException(
                    $"ONNX {direction} contract contains duplicate logical name '{contract.LogicalName}'.");
            }

            if (!tensorNames.Add(contract.TensorName))
            {
                throw new InvalidOperationException(
                    $"ONNX {direction} contract maps more than one logical value to tensor '{contract.TensorName}'.");
            }

            if (!metadata.TryGetValue(contract.TensorName, out var node))
            {
                throw new InvalidOperationException(
                    $"ONNX session is missing required {direction} '{contract.TensorName}' " +
                    $"for logical value '{contract.LogicalName}'.");
            }

            if (!node.IsTensor)
            {
                throw new InvalidOperationException(
                    $"ONNX {direction} '{contract.TensorName}' for '{contract.LogicalName}' is not a tensor.");
            }

            if (contract.Rank is { } rank && node.Dimensions.Length != rank)
            {
                throw new InvalidOperationException(
                    $"ONNX {direction} '{contract.TensorName}' for '{contract.LogicalName}' has rank " +
                    $"{node.Dimensions.Length}; expected {rank}.");
            }

            if (contract.ElementType is { } elementType && node.ElementDataType != elementType)
            {
                throw new InvalidOperationException(
                    $"ONNX {direction} '{contract.TensorName}' for '{contract.LogicalName}' has element type " +
                    $"{node.ElementDataType}; expected {elementType}.");
            }
        }
    }
}
