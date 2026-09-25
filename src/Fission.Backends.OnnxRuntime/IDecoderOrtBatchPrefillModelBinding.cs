using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Optional decoder capability for executing an entire normalized prefill batch
/// as one model-binding operation.
///
/// Each item has a concrete Position. A null prior state denotes an initial
/// prefill starting at position 0; a non-null prior state denotes continuation
/// from that immutable decoder state. Results must preserve input ordering and
/// own distinct immutable state instances.
/// </summary>
public interface IDecoderOrtBatchPrefillModelBinding : IDecoderOrtModelBinding
{
    IReadOnlyList<DecoderOrtStepResult> ExecutePrefillBatch(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default);
}
