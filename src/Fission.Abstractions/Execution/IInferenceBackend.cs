namespace Fission.Abstractions.Execution;

/// <summary>
/// One prefill chunk. Position, when supplied, is the number of tokens already
/// committed for the sequence before Tokens begin. A null Position lets a
/// stateful backend infer the continuation point from its current immutable state;
/// stateless backends may ignore it.
/// </summary>
public readonly record struct PrefillItem(
    SequenceId SequenceId,
    ModelId ModelId,
    ReadOnlyMemory<int> Tokens,
    int? Position = null);

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
/// Prefill/decode batch objects and their Items collections are borrowed inputs.
/// Their storage may be reused as soon as the returned ValueTask completes, so an
/// implementation must not retain a batch, its Items collection, or lazily depend
/// on either through the returned result collection after completion. Any data
/// needed beyond the call must be copied into backend-owned state before the
/// ValueTask completes.
///
/// Stateful backends may retain logits, decoder state, physical KV, and snapshot
/// state between calls. Lifecycle and transaction hooks are serialized by the
/// device actor. Their default implementations are no-ops for stateless backends.
/// A backend that supports migration is responsible for moving or rerouting the
/// sequence's physical state before MigrateSequenceAsync completes.
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

    ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
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
