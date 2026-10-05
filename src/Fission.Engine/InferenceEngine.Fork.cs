using System.Runtime.ExceptionServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Sequences;

namespace Fission.Engine;

public sealed record InferenceForkResult(
    SequenceId Parent,
    IReadOnlyList<SequenceId> Branches);

public sealed partial class InferenceEngine
{
    private static readonly ExecutionBindings EmptyForkBindings =
        new(new Dictionary<SequenceId, ReadOnlyMemory<int>>());

    /// <summary>
    /// Forks one live request into independent runtime sequences that immediately
    /// become normal scheduler candidates. Branches inherit the parent's prompt,
    /// generated-token history, remaining max-new-token budget, priority,
    /// deadline, and enqueue age.
    ///
    /// The parent must already have materialized runtime state (at least one
    /// prefill quantum). Backend fork support remains authoritative: a backend
    /// that cannot clone its state fails the operation before any branch request
    /// is registered with the engine.
    /// </summary>
    public async ValueTask<InferenceForkResult> ForkAsync(
        SequenceId parentSequenceId,
        int branches,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(branches);

        await _cycleGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            RequestState parentRequest;
            lock (_gate)
            {
                if (!_requests.TryGetValue(parentSequenceId, out parentRequest!))
                {
                    throw new KeyNotFoundException(
                        $"Request {parentSequenceId} does not exist.");
                }

                if (parentRequest.IsCompleted)
                {
                    throw new InvalidOperationException(
                        $"Cannot fork completed request {parentSequenceId}.");
                }
            }

            if (!_runtime.TryGetSequence(parentSequenceId, out var parentSequence) ||
                parentSequence is null)
            {
                throw new InvalidOperationException(
                    $"Cannot fork request {parentSequenceId} before its runtime state is materialized.");
            }

            if (parentSequence.Status is SequenceStatus.Finished or SequenceStatus.Cancelled)
            {
                throw new InvalidOperationException(
                    $"Cannot fork terminal runtime sequence {parentSequenceId} while it is {parentSequence.Status}.");
            }

            var execution = await _runtime.ExecuteAsync(
                    new CompiledExecutionPlan(
                        Guid.NewGuid(),
                        Priority: 0,
                        Steps: new ExecutionStep[]
                        {
                            new ForkKvExecutionStep(parentSequenceId, branches)
                        }),
                    EmptyForkBindings,
                    cancellationToken)
                .ConfigureAwait(false);

            if (execution.Forks.Count != 1 ||
                execution.Forks[0].Parent != parentSequenceId ||
                execution.Forks[0].Branches.Count != branches)
            {
                throw new InvalidOperationException(
                    $"Runtime fork result for {parentSequenceId} did not match the requested branch count {branches}.");
            }

            var branchIds = execution.Forks[0].Branches;
            try
            {
                RegisterForkRequests(parentRequest, branchIds);
            }
            catch (Exception registrationFailure)
            {
                var rollbackFailures = await RollbackForkBranchesAsync(branchIds)
                    .ConfigureAwait(false);
                if (rollbackFailures.Count == 0)
                {
                    ExceptionDispatchInfo.Capture(registrationFailure).Throw();
                }

                throw new AggregateException(
                    $"Forked runtime branches for {parentSequenceId} but failed to register them with the engine, and rollback also encountered errors.",
                    new[] { registrationFailure }.Concat(rollbackFailures));
            }

            return new InferenceForkResult(parentSequenceId, branchIds);
        }
        finally
        {
            _cycleGate.Release();
        }
    }

    private void RegisterForkRequests(
        RequestState parent,
        IReadOnlyList<SequenceId> branchIds)
    {
        var branchRequests = new RequestState[branchIds.Count];
        for (var index = 0; index < branchIds.Count; index++)
        {
            var branch = new RequestState(
                branchIds[index],
                parent.ModelId,
                parent.PromptTokens,
                parent.MaxNewTokens,
                parent.Priority,
                parent.Deadline,
                parent.EnqueuedAt);
            branch.GeneratedTokens.AddRange(parent.GeneratedTokens);
            branchRequests[index] = branch;
        }

        lock (_gate)
        {
            var nextActiveCount = checked(_activeRequestCount + branchIds.Count);
            _requests.EnsureCapacity(checked(_requests.Count + branchIds.Count));

            for (var index = 0; index < branchIds.Count; index++)
            {
                if (_requests.ContainsKey(branchIds[index]))
                {
                    throw new InvalidOperationException(
                        $"Forked request id {branchIds[index]} is already registered with the engine.");
                }
            }

            var added = 0;
            try
            {
                for (; added < branchRequests.Length; added++)
                {
                    var branch = branchRequests[added];
                    _requests.Add(branch.SequenceId, branch);
                }
            }
            catch
            {
                for (var index = 0; index < added; index++)
                {
                    _requests.Remove(branchRequests[index].SequenceId);
                }

                throw;
            }

            _activeRequestCount = nextActiveCount;
        }
    }

    private async ValueTask<List<Exception>> RollbackForkBranchesAsync(
        IReadOnlyList<SequenceId> branchIds)
    {
        var failures = new List<Exception>();

        for (var index = 0; index < branchIds.Count; index++)
        {
            var branchId = branchIds[index];
            try
            {
                if (!_runtime.TryGetSequence(branchId, out var sequence) || sequence is null)
                {
                    continue;
                }

                if (sequence.Status is not SequenceStatus.Finished and not SequenceStatus.Cancelled)
                {
                    sequence.TransitionTo(SequenceStatus.Cancelled);
                }

                await _runtime.ReleaseSequenceAsync(branchId, CancellationToken.None)
                    .ConfigureAwait(false);
            }
            catch (Exception rollbackFailure)
            {
                failures.Add(rollbackFailure);
            }
        }

        return failures;
    }
}
