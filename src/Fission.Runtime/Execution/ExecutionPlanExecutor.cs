using System.Collections.Concurrent;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Kv;
using Fission.Runtime.Sequences;

namespace Fission.Runtime.Execution;

public sealed class ExecutionBindings
{
    private readonly IReadOnlyDictionary<SequenceId, ReadOnlyMemory<int>> _prefillTokens;

    public ExecutionBindings(IReadOnlyDictionary<SequenceId, ReadOnlyMemory<int>> prefillTokens)
    {
        _prefillTokens = prefillTokens;
    }

    public ReadOnlyMemory<int> ResolvePrefill(SequenceId sequenceId, int expectedTokenCount)
    {
        if (!_prefillTokens.TryGetValue(sequenceId, out var tokens))
        {
            throw new KeyNotFoundException($"No prefill token binding exists for sequence {sequenceId}.");
        }

        if (tokens.Length != expectedTokenCount)
        {
            throw new InvalidOperationException(
                $"Plan expects {expectedTokenCount} prefill tokens for {sequenceId}, but binding contains {tokens.Length}.");
        }

        return tokens;
    }
}

public sealed record ForkExecutionResult(
    SequenceId Parent,
    IReadOnlyList<SequenceId> Branches);

public sealed record ExecutionPlanResult(
    Guid PlanId,
    IReadOnlyList<BackendStepResult> BackendResults,
    IReadOnlyList<KvSnapshotId> Snapshots,
    IReadOnlyList<ForkExecutionResult> Forks);

/// <summary>
/// Interprets a compiled inference plan against the stateful runtime.
/// Multiple plans may execute concurrently and converge on the single-device
/// continuous batch executor for backend work.
/// </summary>
public sealed class ExecutionPlanExecutor : IDisposable
{
    private readonly ContinuousBatchExecutor _device;
    private readonly ConcurrentDictionary<SequenceId, SequenceProcess> _sequences = new();
    private readonly ConcurrentDictionary<KvSnapshotId, KvSnapshot> _snapshots = new();
    private int _disposed;

    public ExecutionPlanExecutor(ContinuousBatchExecutor device)
    {
        _device = device;
    }

    public int SequenceCount => _sequences.Count;
    public int SnapshotCount => _snapshots.Count;

    public bool TryGetSequence(SequenceId sequenceId, out SequenceProcess? sequence) =>
        _sequences.TryGetValue(sequenceId, out sequence);

    public async ValueTask<ExecutionPlanResult> ExecuteAsync(
        CompiledExecutionPlan plan,
        ExecutionBindings bindings,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        var backendResults = new List<BackendStepResult>();
        var snapshotIds = new List<KvSnapshotId>();
        var forks = new List<ForkExecutionResult>();

        foreach (var step in plan.Steps)
        {
            cancellationToken.ThrowIfCancellationRequested();

            switch (step)
            {
                case PrefillExecutionStep prefill:
                    await ExecutePrefillAsync(prefill, bindings, backendResults, cancellationToken)
                        .ConfigureAwait(false);
                    break;

                case DecodeExecutionStep decode:
                    await ExecuteDecodeAsync(decode, backendResults, cancellationToken)
                        .ConfigureAwait(false);
                    break;

                case SnapshotKvExecutionStep snapshot:
                {
                    var sequence = GetSequence(snapshot.SequenceId);
                    var state = sequence.Snapshot();
                    if (!_snapshots.TryAdd(state.Id, state))
                    {
                        state.Dispose();
                        throw new InvalidOperationException($"Duplicate snapshot id {state.Id}.");
                    }

                    snapshotIds.Add(state.Id);
                    break;
                }

                case ForkKvExecutionStep fork:
                {
                    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fork.Branches);
                    var parent = GetSequence(fork.SequenceId);
                    var branchIds = new SequenceId[fork.Branches];

                    for (var index = 0; index < fork.Branches; index++)
                    {
                        var branch = parent.Fork();
                        if (!_sequences.TryAdd(branch.Id, branch))
                        {
                            branch.Dispose();
                            throw new InvalidOperationException($"Duplicate forked sequence id {branch.Id}.");
                        }

                        branchIds[index] = branch.Id;
                    }

                    forks.Add(new ForkExecutionResult(parent.Id, branchIds));
                    break;
                }

                case RestoreKvExecutionStep restore:
                {
                    var sequence = GetSequence(restore.SequenceId);
                    if (!_snapshots.TryGetValue(restore.SnapshotId, out var snapshot))
                    {
                        throw new KeyNotFoundException($"Snapshot {restore.SnapshotId} is not owned by this executor.");
                    }

                    sequence.Restore(snapshot);
                    break;
                }

                case MigrateKvExecutionStep migrate:
                {
                    var sequence = GetSequence(migrate.SequenceId);
                    sequence.MigrateTo(migrate.TargetDevice);
                    break;
                }

                default:
                    throw new NotSupportedException($"Unsupported execution step: {step.GetType().Name}.");
            }
        }

        return new ExecutionPlanResult(plan.PlanId, backendResults, snapshotIds, forks);
    }

    private async ValueTask ExecutePrefillAsync(
        PrefillExecutionStep step,
        ExecutionBindings bindings,
        List<BackendStepResult> results,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(step.TokenCount);

        var sequence = _sequences.GetOrAdd(
            step.SequenceId,
            id => SequenceProcess.Create(id, step.ModelId, _device.Device));

        if (sequence.Model != step.ModelId)
        {
            throw new InvalidOperationException(
                $"Sequence {step.SequenceId} is already bound to model {sequence.Model}, not {step.ModelId}.");
        }

        if (sequence.Status == SequenceStatus.Waiting || sequence.Status == SequenceStatus.Suspended)
        {
            sequence.TransitionTo(SequenceStatus.Prefilling);
        }
        else if (sequence.Status != SequenceStatus.Prefilling)
        {
            throw new InvalidOperationException(
                $"Cannot prefill sequence {step.SequenceId} while it is {sequence.Status}.");
        }

        var tokens = bindings.ResolvePrefill(step.SequenceId, step.TokenCount);
        var result = await _device.SubmitPrefillAsync(
            new PrefillItem(step.SequenceId, step.ModelId, tokens),
            cancellationToken).ConfigureAwait(false);

        sequence.RecordPrefill(step.TokenCount);
        sequence.TransitionTo(SequenceStatus.Decoding);
        results.Add(result);
    }

    private async ValueTask ExecuteDecodeAsync(
        DecodeExecutionStep step,
        List<BackendStepResult> results,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(step.MaxTokens);
        var sequence = GetSequence(step.SequenceId);

        if (sequence.Status == SequenceStatus.Suspended)
        {
            sequence.TransitionTo(SequenceStatus.Decoding);
        }

        if (sequence.Status != SequenceStatus.Decoding)
        {
            throw new InvalidOperationException(
                $"Cannot decode sequence {step.SequenceId} while it is {sequence.Status}.");
        }

        for (var index = 0; index < step.MaxTokens; index++)
        {
            var result = await _device.SubmitDecodeAsync(
                new DecodeItem(sequence.Id, sequence.Model, sequence.Position),
                cancellationToken).ConfigureAwait(false);

            sequence.RecordDecode();
            results.Add(result);

            if (result.IsFinished)
            {
                sequence.TransitionTo(SequenceStatus.Finished);
                break;
            }
        }
    }

    private SequenceProcess GetSequence(SequenceId sequenceId)
    {
        if (_sequences.TryGetValue(sequenceId, out var sequence))
        {
            return sequence;
        }

        throw new KeyNotFoundException($"Sequence {sequenceId} does not exist in this executor.");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var snapshot in _snapshots.Values)
        {
            snapshot.Dispose();
        }

        _snapshots.Clear();

        foreach (var sequence in _sequences.Values)
        {
            sequence.Dispose();
        }

        _sequences.Clear();
    }
}
