namespace Fission.Abstractions.Execution;

/// <summary>
/// Describes a physical KV write that must first detach a shared partial tail.
/// The runtime intentionally exposes only physical write geometry here: backend
/// implementations do not receive runtime page ids or reference counts.
/// </summary>
public readonly record struct InferenceKvWriteIntent
{
    public InferenceKvWriteIntent(
        int copyOnWritePages,
        int tokensPerPage,
        int tailTokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(copyOnWritePages);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokensPerPage);
        if (tailTokenCount <= 0 || tailTokenCount >= tokensPerPage)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tailTokenCount),
                "A copy-on-write tail must be partially occupied.");
        }

        CopyOnWritePages = copyOnWritePages;
        TokensPerPage = tokensPerPage;
        TailTokenCount = tailTokenCount;
    }

    public int CopyOnWritePages { get; }
    public int TokensPerPage { get; }
    public int TailTokenCount { get; }
    public bool RequiresMaterialization => CopyOnWritePages > 0;
}

/// <summary>
/// Optional capability for stateful inference backends whose physical KV backing
/// must be detached before a divergent write into a shared partial tail.
///
/// The device actor invokes these hooks immediately before the matching backend
/// PrefillAsync/DecodeAsync call, and only when at least one work item carries a
/// materialization intent. Implementations may replace physical backing with an
/// equivalent private copy, but must not advance the sequence causal frontier.
/// If materialization fails, the previously committed state must remain readable.
/// Batch objects and their item collections are borrowed for the duration of the
/// ValueTask and must not be retained.
/// </summary>
public interface IInferenceKvWriteMaterializer
{
    ValueTask MaterializePrefillKvWritesAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default);

    ValueTask MaterializeDecodeKvWritesAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default);
}
