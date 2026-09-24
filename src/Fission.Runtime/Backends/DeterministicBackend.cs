using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Backends;

/// <summary>
/// A zero-model backend for validating scheduling/execution semantics before
/// introducing GPU-specific behavior. It deliberately returns deterministic
/// pseudo token ids for a given sequence and position.
/// </summary>
public sealed class DeterministicBackend : IInferenceBackend
{
    private bool _initialized;

    public DeterministicBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => "deterministic";
    public DeviceId Device { get; }

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _initialized = true;
        return ValueTask.CompletedTask;
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();

        var results = new BackendStepResult[batch.Items.Count];
        for (var i = 0; i < batch.Items.Count; i++)
        {
            var item = batch.Items[i];
            results[i] = new BackendStepResult(
                item.SequenceId,
                TokenFor(item.SequenceId, item.Tokens.Length));
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();

        var results = new BackendStepResult[batch.Items.Count];
        for (var i = 0; i < batch.Items.Count; i++)
        {
            var item = batch.Items[i];
            results[i] = new BackendStepResult(
                item.SequenceId,
                TokenFor(item.SequenceId, item.Position));
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask DisposeAsync()
    {
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private static int TokenFor(SequenceId sequenceId, int position)
    {
        Span<byte> bytes = stackalloc byte[16];
        sequenceId.Value.TryWriteBytes(bytes);

        uint hash = 2166136261;
        foreach (var value in bytes)
        {
            hash ^= value;
            hash *= 16777619;
        }

        hash ^= unchecked((uint)position);
        hash *= 16777619;
        return (int)(hash % 32_000);
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Backend has not been initialized.");
        }
    }
}
