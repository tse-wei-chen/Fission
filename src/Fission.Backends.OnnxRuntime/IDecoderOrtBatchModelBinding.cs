using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Optional decoder capability for executing an entire decode batch as one
/// model-binding operation. Results must preserve input ordering and each
/// successful result must own a distinct immutable decoder state instance.
///
/// The generic adapter retains the scalar IDecoderOrtModelBinding path for
/// bindings that do not implement this interface.
/// </summary>
public interface IDecoderOrtBatchModelBinding : IDecoderOrtModelBinding
{
    IReadOnlyList<DecoderOrtStepResult> ExecuteDecodeBatch(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default);
}
