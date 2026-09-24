using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Executes an ONNX Runtime session into caller-owned OrtValue output buffers.
///
/// Unlike Run overloads that allocate and return a disposable output collection,
/// this path never transfers output ownership to ONNX Runtime. The caller remains
/// responsible for every output OrtValue and may transfer those values into a
/// longer-lived owner such as DecoderOrtState after the run completes.
/// </summary>
public static class CallerOwnedOrtRun
{
    public static void Execute(
        InferenceSession session,
        RunOptions runOptions,
        IReadOnlyCollection<string> inputNames,
        IReadOnlyCollection<OrtValue> inputValues,
        IReadOnlyCollection<string> outputNames,
        IReadOnlyCollection<OrtValue> outputValues)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(runOptions);
        ArgumentNullException.ThrowIfNull(inputNames);
        ArgumentNullException.ThrowIfNull(inputValues);
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(outputValues);

        if (inputNames.Count != inputValues.Count)
        {
            throw new ArgumentException(
                "Input name and OrtValue counts must match.",
                nameof(inputValues));
        }

        if (outputNames.Count != outputValues.Count)
        {
            throw new ArgumentException(
                "Output name and OrtValue counts must match.",
                nameof(outputValues));
        }

        if (outputValues.Count == 0)
        {
            throw new ArgumentException(
                "At least one caller-owned output OrtValue is required.",
                nameof(outputValues));
        }

        var ownedOutputs = new HashSet<OrtValue>(ReferenceEqualityComparer.Instance);
        foreach (var output in outputValues)
        {
            ArgumentNullException.ThrowIfNull(output);
            if (!output.IsTensor)
            {
                throw new ArgumentException(
                    "Caller-owned ORT outputs must be tensors.",
                    nameof(outputValues));
            }

            if (!ownedOutputs.Add(output))
            {
                throw new ArgumentException(
                    "The same OrtValue cannot back more than one output slot.",
                    nameof(outputValues));
            }
        }

        session.Run(
            runOptions,
            inputNames,
            inputValues,
            outputNames,
            outputValues);
    }
}
