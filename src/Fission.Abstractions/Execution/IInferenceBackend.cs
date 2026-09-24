namespace Fission.Abstractions.Execution;

public readonly record struct PrefillItem(
    SequenceId SequenceId,
    ModelId ModelId,
    ReadOnlyMemory<int> Tokens);

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
}
