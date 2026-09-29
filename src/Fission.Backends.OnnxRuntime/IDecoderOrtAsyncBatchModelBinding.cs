using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Optional decoder capability for batch execution that must await asynchronous
/// preparation or native completion before model results are safe to consume.
///
/// The adapter prefers this capability over the synchronous batch interface when
/// both are implemented. Results preserve input ordering and transfer ownership
/// of one distinct immutable state per successful item to the adapter.
/// </summary>
public interface IDecoderOrtAsyncBatchModelBinding : IDecoderOrtModelBinding
{
    ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecuteDecodeBatchAsync(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Optional asynchronous counterpart of <see cref="IDecoderOrtBatchPrefillModelBinding"/>.
///
/// Each item is normalized to a concrete Position before invocation. A null prior
/// state denotes initial prefill; a non-null prior state denotes continuation from
/// that immutable decoder state. The adapter awaits this capability before it
/// validates or commits any returned state.
/// </summary>
public interface IDecoderOrtAsyncBatchPrefillModelBinding : IDecoderOrtModelBinding
{
    ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecutePrefillBatchAsync(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default);
}
