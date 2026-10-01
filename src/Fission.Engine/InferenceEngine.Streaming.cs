using Fission.Abstractions;

namespace Fission.Engine;

public sealed partial class InferenceEngine
{
    /// <summary>
    /// Reads one generated token without materializing a request snapshot or
    /// copying the complete generated-token history. The returned count and token
    /// are captured under the same request-state lock.
    /// </summary>
    internal bool TryReadGeneratedToken(
        SequenceId sequenceId,
        int index,
        out int tokenId,
        out int generatedTokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        lock (_gate)
        {
            if (!_requests.TryGetValue(sequenceId, out var request))
            {
                throw new KeyNotFoundException($"Request {sequenceId} does not exist.");
            }

            generatedTokenCount = request.GeneratedTokens.Count;
            if ((uint)index >= (uint)generatedTokenCount)
            {
                tokenId = default;
                return false;
            }

            tokenId = request.GeneratedTokens[index];
            return true;
        }
    }
}
