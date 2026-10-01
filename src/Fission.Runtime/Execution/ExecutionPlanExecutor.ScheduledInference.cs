using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Sequences;
using Fission.Runtime.Tracing;

namespace Fission.Runtime.Execution;

public sealed partial class ExecutionPlanExecutor
{
    /// <summary>
    /// Executes the scheduler's validated one-step prefill/decode plan without
    /// materializing a generic CompiledExecutionPlan and its reservation/result
    /// bookkeeping. Atomic actor registration is passed explicitly so the
    /// scheduled hot path does not need an AsyncLocal slot scope.
    /// </summary>
    internal async ValueTask<ExecutionPlanResult> ExecuteScheduledInferenceAsync(
        Guid planId,
        ExecutionStep step,
        ExecutionBindings bindings,
        ContinuousBatchExecutor.AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(step);
        ArgumentNullException.ThrowIfNull(bindings);
        ArgumentNullException.ThrowIfNull(submission);

        if (step is not PrefillExecutionStep and not DecodeExecutionStep)
        {
            throw new NotSupportedException(
                $"Scheduled single-inference execution does not support {step.GetType().Name}.");
        }

        ReserveSequence(step.SequenceId, $"plan {planId}");
        try
        {
            Record(new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.PlanStarted,
                -1,
                "Plan"));

            cancellationToken.ThrowIfCancellationRequested();

            var operation = step.GetType().Name;
            RecordSequenceState(
                planId,
                ExecutionTraceKind.StepStarted,
                0,
                operation,
                step.SequenceId);

            BackendStepResult result = step switch
            {
                PrefillExecutionStep prefill => await ExecuteScheduledPrefillAsync(
                        prefill,
                        bindings,
                        submission,
                        slot,
                        cancellationToken)
                    .ConfigureAwait(false),
                DecodeExecutionStep decode => await ExecuteScheduledDecodeAsync(
                        decode,
                        submission,
                        slot,
                        cancellationToken)
                    .ConfigureAwait(false),
                _ => throw new InvalidOperationException(
                    $"Unsupported scheduled inference step {step.GetType().Name}.")
            };

            RecordSequenceState(
                planId,
                ExecutionTraceKind.StepCompleted,
                0,
                operation,
                step.SequenceId);

            Record(new ExecutionTraceEvent(
                planId,
                ExecutionTraceKind.PlanCompleted,
                1,
                "Plan"));

            return new ExecutionPlanResult(
                planId,
                new[] { result },
                Array.Empty<KvSnapshotId>(),
                Array.Empty<ForkExecutionResult>());
        }
        finally
        {
            _sequenceReservations.TryRemove(step.SequenceId, out _);
        }
    }

    private async ValueTask<BackendStepResult> ExecuteScheduledPrefillAsync(
        PrefillExecutionStep step,
        ExecutionBindings bindings,
        ContinuousBatchExecutor.AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(step.TokenCount);

        var sequence = _sequences.GetOrAdd(
            step.SequenceId,
            id => SequenceProcess.Create(id, step.ModelId, _devices.DefaultDevice, _kvPagePool));

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
        var device = _devices.ResolvePlacement(sequence.Device);
        var result = await device.SubmitPrefillAsync(
                new PrefillItem(step.SequenceId, step.ModelId, tokens),
                submission,
                slot,
                cancellationToken)
            .ConfigureAwait(false);

        sequence.RecordPrefill(step.TokenCount);
        if (step.CompletesPrefill)
        {
            sequence.TransitionTo(SequenceStatus.Decoding);
        }

        return result;
    }

    private async ValueTask<BackendStepResult> ExecuteScheduledDecodeAsync(
        DecodeExecutionStep step,
        ContinuousBatchExecutor.AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken)
    {
        if (step.MaxTokens != 1)
        {
            throw new InvalidOperationException(
                "Scheduled decode execution must contain exactly one token step.");
        }

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

        var device = _devices.ResolvePlacement(sequence.Device);
        var result = await device.SubmitDecodeAsync(
                new DecodeItem(sequence.Id, sequence.Model, sequence.Position),
                submission,
                slot,
                cancellationToken)
            .ConfigureAwait(false);

        sequence.RecordDecode();
        if (result.IsFinished)
        {
            sequence.TransitionTo(SequenceStatus.Finished);
        }

        return result;
    }
}
