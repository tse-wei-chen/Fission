using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Bridges a concrete decoder geometry into the generic retained-KV scheduling
/// contract without exposing ONNX Runtime tensor types to the engine.
///
/// The profile reports steady-state retained KV bytes per token. It intentionally
/// does not include transient model-step workspace or the temporary overlap of an
/// immutable prior state and its newly produced successor state.
/// </summary>
public sealed class DecoderOrtGeometryKvMemoryProfile : IInferenceKvMemoryProfile
{
    private readonly ModelId _modelId;
    private readonly DecoderOrtGeometry _geometry;

    public DecoderOrtGeometryKvMemoryProfile(
        ModelId modelId,
        DecoderOrtGeometry geometry)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        _modelId = modelId;
        _geometry = geometry;
    }

    public long GetKvBytesPerToken(ModelId modelId)
    {
        if (modelId != _modelId)
        {
            throw new InvalidOperationException(
                $"KV memory profile is bound to model {_modelId}, not {modelId}.");
        }

        return _geometry.GetKvBytesPerSequence(sequenceLength: 1);
    }
}
