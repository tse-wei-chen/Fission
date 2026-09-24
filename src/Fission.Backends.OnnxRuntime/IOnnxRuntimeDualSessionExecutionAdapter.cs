using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Optional adapter capability for backends that host distinct prefill and decode
/// ONNX Runtime sessions.
/// </summary>
public interface IOnnxRuntimeDualSessionExecutionAdapter
{
    ValueTask InitializeAsync(
        InferenceSession prefillSession,
        InferenceSession decodeSession,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Optional adapter capability for declaring distinct live-graph contracts for
/// prefill and with-past decode sessions.
/// </summary>
public interface IOnnxRuntimeDualSessionContractProvider
{
    OnnxSessionContract PrefillSessionContract { get; }
    OnnxSessionContract DecodeSessionContract { get; }
}

/// <summary>
/// Model binding variant for export families that use a distinct no-past prefill
/// graph and with-past decode graph.
/// </summary>
public interface IDecoderOrtDualSessionModelBinding : IDecoderOrtModelBinding
{
    OnnxSessionContract PrefillSessionContract { get; }

    ValueTask InitializeAsync(
        InferenceSession prefillSession,
        InferenceSession decodeSession,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }
}
