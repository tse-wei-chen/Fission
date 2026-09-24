using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Generic stateful decoder adapter.
///
/// Model/export-specific tensor work is delegated to IDecoderOrtModelBinding.
/// This class owns sequence/snapshot state transactions and commits each batch
/// only after every model step has produced a valid immutable next-state version.
/// </summary>
public sealed class DecoderOnlyOnnxExecutionAdapter :
    IOnnxRuntimeExecutionAdapter,
    IOnnxRuntimeSessionContractProvider
{
    private readonly IDecoderOrtModelBinding _binding;
    private readonly DecoderStateStore<DecoderOrtState> _states = new();
    private int _initialized;
    private int _disposed;

    public DecoderOnlyOnnxExecutionAdapter(IDecoderOrtModelBinding binding)
    {
        ArgumentNullException.ThrowIfNull(binding);
        _binding = binding;
    }

    public string Name => $"decoder/{_binding.Name}";
    public OnnxSessionContract SessionContract => _binding.SessionContract;

    public async ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(session);
        cancellationToken.ThrowIfCancellationRequested();

        if (Volatile.Read(ref _initialized) != 0)
        {
            throw new InvalidOperationException(
                "Decoder ONNX adapter is already initialized.");
        }

        await _binding.InitializeAsync(session, cancellationToken)
            .ConfigureAwait(false);
        Volatile.Write(ref _initialized, 1);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        InferenceSession session,
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(batch);
        cancellationToken.ThrowIfCancellationRequested();

        if (batch.Items.Count == 0)
        {
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                Array.Empty<BackendStepResult>());
        }

        var seen = new HashSet<SequenceId>();
        foreach (var item in batch.Items)
        {
            if (!seen.Add(item.SequenceId))
            {
                throw new InvalidOperationException(
                    $"Prefill batch contains duplicate sequence {item.SequenceId}.");
            }

            if (_states.TryGetSequence(item.SequenceId, out _))
            {
                throw new InvalidOperationException(
                    $"Decoder state already exists for prefill sequence {item.SequenceId}.");
            }
        }

        var pending = new DecoderOrtStepResult[batch.Items.Count];
        var produced = 0;
        try
        {
            for (var index = 0; index < batch.Items.Count; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var item = batch.Items[index];
                var step = _binding.ExecutePrefill(
                    session,
                    item,
                    cancellationToken);
                ValidateStepResult(
                    step,
                    expectedPosition: item.Tokens.Length,
                    priorState: null);
                pending[index] = step;
                produced++;
            }
        }
        catch
        {
            DisposeProducedStates(pending, produced);
            throw;
        }

        var committed = 0;
        try
        {
            for (; committed < batch.Items.Count; committed++)
            {
                _states.AddSequence(
                    batch.Items[committed].SequenceId,
                    pending[committed].State);
            }
        }
        catch
        {
            for (var index = 0; index < committed; index++)
            {
                _states.ReleaseSequence(batch.Items[index].SequenceId);
            }

            for (var index = committed; index < pending.Length; index++)
            {
                pending[index].State.Dispose();
            }

            throw;
        }

        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < results.Length; index++)
        {
            results[index] = new BackendStepResult(
                batch.Items[index].SequenceId,
                pending[index].TokenId,
                pending[index].IsFinished);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        InferenceSession session,
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(batch);
        cancellationToken.ThrowIfCancellationRequested();

        if (batch.Items.Count == 0)
        {
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                Array.Empty<BackendStepResult>());
        }

        var priorStates = new DecoderOrtState[batch.Items.Count];
        var seen = new HashSet<SequenceId>();
        for (var index = 0; index < batch.Items.Count; index++)
        {
            var item = batch.Items[index];
            if (!seen.Add(item.SequenceId))
            {
                throw new InvalidOperationException(
                    $"Decode batch contains duplicate sequence {item.SequenceId}.");
            }

            var prior = _states.GetSequence(item.SequenceId);
            if (prior.Position != item.Position)
            {
                throw new InvalidOperationException(
                    $"Decode position mismatch for sequence {item.SequenceId}: " +
                    $"runtime requested {item.Position}, backend state is {prior.Position}.");
            }

            if (prior.NextTokenId is null)
            {
                throw new InvalidOperationException(
                    $"Decode state for sequence {item.SequenceId} is missing its next-token frontier.");
            }

            priorStates[index] = prior;
        }

        var pending = new DecoderOrtStepResult[batch.Items.Count];
        var produced = 0;
        try
        {
            for (var index = 0; index < batch.Items.Count; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var item = batch.Items[index];
                var step = _binding.ExecuteDecode(
                    session,
                    item,
                    priorStates[index],
                    cancellationToken);
                ValidateStepResult(
                    step,
                    expectedPosition: checked(item.Position + 1),
                    priorStates[index]);
                pending[index] = step;
                produced++;
            }
        }
        catch
        {
            DisposeProducedStates(pending, produced);
            throw;
        }

        // All failure-prone model work has completed. Calls are serialized by the
        // device actor, so the prevalidated sequence set cannot change before this
        // commit phase.
        for (var index = 0; index < batch.Items.Count; index++)
        {
            _states.ReplaceSequence(
                batch.Items[index].SequenceId,
                pending[index].State);
        }

        var results = new BackendStepResult[batch.Items.Count];
        for (var index = 0; index < results.Length; index++)
        {
            results[index] = new BackendStepResult(
                batch.Items[index].SequenceId,
                pending[index].TokenId,
                pending[index].IsFinished);
        }

        return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(results);
    }

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _states.SnapshotSequence(sequenceId, snapshotId);
        return ValueTask.CompletedTask;
    }

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _states.ForkSequence(parentSequenceId, branchSequenceIds);
        return ValueTask.CompletedTask;
    }

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        _states.RestoreSequence(sequenceId, snapshotId);
        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (!_states.ReleaseSnapshot(snapshotId))
        {
            throw new KeyNotFoundException(
                $"Decoder snapshot state does not exist for snapshot {snapshotId}.");
        }

        return ValueTask.CompletedTask;
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        if (!_states.ReleaseSequence(sequenceId))
        {
            throw new KeyNotFoundException(
                $"Decoder state does not exist for sequence {sequenceId}.");
        }

        return ValueTask.CompletedTask;
    }

    private static void ValidateStepResult(
        DecoderOrtStepResult step,
        int expectedPosition,
        DecoderOrtState? priorState)
    {
        ArgumentNullException.ThrowIfNull(step);
        ArgumentNullException.ThrowIfNull(step.State);

        if (step.TokenId < 0)
        {
            throw new InvalidOperationException(
                $"Decoder binding returned invalid token id {step.TokenId}.");
        }

        if (step.State.IsDisposed)
        {
            throw new InvalidOperationException(
                "Decoder binding returned an already-disposed state.");
        }

        if (step.State.Position != expectedPosition)
        {
            throw new InvalidOperationException(
                $"Decoder binding returned state position {step.State.Position}; " +
                $"expected {expectedPosition}.");
        }

        if (step.State.NextTokenId != step.TokenId)
        {
            throw new InvalidOperationException(
                $"Decoder binding returned token {step.TokenId}, but physical state frontier is " +
                $"{step.State.NextTokenId?.ToString() ?? "missing"}.");
        }

        if (priorState is not null && ReferenceEquals(step.State, priorState))
        {
            throw new InvalidOperationException(
                "Decoder binding must return a new immutable state version for decode.");
        }
    }

    private static void DisposeProducedStates(
        DecoderOrtStepResult[] steps,
        int count)
    {
        for (var index = count - 1; index >= 0; index--)
        {
            steps[index].State.Dispose();
        }
    }

    private void EnsureInitialized()
    {
        ThrowIfDisposed();
        if (Volatile.Read(ref _initialized) == 0)
        {
            throw new InvalidOperationException(
                "Decoder ONNX adapter has not been initialized.");
        }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _states.Dispose();
        _binding.Dispose();
        Volatile.Write(ref _initialized, 0);
    }
}
