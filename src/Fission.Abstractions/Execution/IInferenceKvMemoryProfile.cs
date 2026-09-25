namespace Fission.Abstractions.Execution;

/// <summary>
/// Optional backend/model capability that exposes the incremental physical KV
/// memory cost of one additional token. A value of zero means the backend does
/// not provide physical-byte accounting and higher layers should rely on their
/// other resource budgets.
/// </summary>
public interface IInferenceKvMemoryProfile
{
    long GetKvBytesPerToken(ModelId modelId);
}
