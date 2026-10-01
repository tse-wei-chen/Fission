using Fission.Abstractions;

namespace Fission.Engine;

public sealed partial class InferenceEngine
{
    /// <summary>
    /// Transfers ownership of an already-isolated prompt buffer into engine request
    /// state without copying it again. Internal callers must not retain or mutate
    /// <paramref name="promptTokens"/> after this method is called.
    /// </summary>
    internal SequenceId SubmitOwned(
        ModelId modelId,
        int[] promptTokens,
        int maxNewTokens,
        int priority = 0,
        DateTimeOffset? deadline = null,
        DateTimeOffset? enqueuedAt = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(promptTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxNewTokens);

        if (promptTokens.Length == 0)
        {
            throw new ArgumentException("Prompt must contain at least one token.", nameof(promptTokens));
        }

        var sequenceId = SequenceId.New();
        var state = new RequestState(
            sequenceId,
            modelId,
            promptTokens,
            maxNewTokens,
            priority,
            deadline,
            enqueuedAt ?? DateTimeOffset.UtcNow);

        lock (_gate)
        {
            _requests.Add(sequenceId, state);
        }

        return sequenceId;
    }
}
