namespace Fission.Abstractions.Execution;

/// <summary>
/// Describes a physical KV write that must first detach a shared partial tail.
/// The runtime intentionally exposes only physical write geometry here: backend
/// implementations do not receive runtime page ids or reference counts.
///
/// A default value means no special physical materialization is required. When
/// RequiresMaterialization is true, a stateful backend that mutates paged KV in
/// place must make the affected tail privately writable before applying the
/// corresponding PrefillItem or DecodeItem. Backends that already produce a new
/// immutable state version for every inference step satisfy this contract without
/// an extra copy hook.
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
