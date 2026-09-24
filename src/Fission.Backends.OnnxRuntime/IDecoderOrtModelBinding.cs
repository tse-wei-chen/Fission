using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Result of one decoder model step. Ownership of State transfers to the caller
/// when the method returns successfully.
/// </summary>
public sealed record DecoderOrtStepResult(
    int TokenId,
    DecoderOrtState State,
    bool IsFinished = false);

/// <summary>
/// Export/model-specific decoder binding.
///
/// The binding owns tensor packing, output allocation, ONNX Runtime execution,
/// and logits-to-token policy. It must treat priorState as immutable and return a
/// newly owned DecoderOrtState for every successful model step.
/// </summary>
public interface IDecoderOrtModelBinding : IDisposable
{
    string Name { get; }
    OnnxSessionContract SessionContract { get; }

    ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }

    DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default);

    DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default);
}
