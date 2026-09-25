namespace Fission.Abstractions.Execution;

/// <summary>
/// One prefill chunk. Position is the number of tokens already committed for the
/// sequence before Tokens begin. Position=0 therefore denotes the first chunk;
/// later chunks must continue from backend state at the same position.
/// </summary>
public readonly record struct PrefillItem(
    SequenceId SequenceId,
    ModelId ModelId,
    ReadOnlyMemory<int> Tokens,
    int Position = 0);

public readonly record struct DecodeItem(
    SequenceId SequenceId,
    ModelId ModelId,
    int Position);

public sealed record PrefillBatch(IReadOnlyList<PrefillItem> Items);
public sealed record DecodeBatch(IReadOnlyList<DecodeItem> Items);

public readonly record struct BackendStepResult(
    SequenceId SequenceId,
    int TokenId,
    bool IsFinished = false);

/// <summary>
/// Device-facing execution boundary. Implementations must preserve input ordering
/// in their returned result list so the runtime can complete individual tickets
/// without sequence-id lookups on the hot path.
///
/// Stateful backends may retain logits, decoder state, physical KV, and snapshot
/// state between calls. Lifecycle and transaction hooks are serialized by the
/// device actor. Their default implementations are no-ops for stateless backends.
/// </summary>
public interface IInferenceBackend : IAsyncDisposable
{
    string Name { get; }
    DeviceId Device { get; }

    ValueTask InitializeAsync(CancellationToken cancellationToken = default);

    ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default);

    ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default);

    ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }

    ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }

    ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }

    ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }

    ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }
}
