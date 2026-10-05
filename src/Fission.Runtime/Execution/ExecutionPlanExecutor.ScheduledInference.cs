using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Sequences;
using Fission.Runtime.Tracing;

namespace Fission.Runtime.Execution;

internal enum ScheduledInferenceKind
{
    Prefill,
    Decode
}

internal readonly record struct ScheduledInferenceStep(
    ScheduledInferenceKind Kind,
    SequenceId SequenceId,
    ModelId ModelId,
    int TokenCount,
    bool CompletesPrefill)
{
    internal static ScheduledInferenceStep Prefill(
        SequenceId sequenceId,
        ModelId modelId,
        int tokenCount,
        bool completesPrefill) =>
        new(
            ScheduledInferenceKind.Prefill,
            sequenceId,
            modelId,
            tokenCount,
            completesPrefill);

    internal static ScheduledInferenceStep Decode(SequenceId sequenceId) =>
        new(
            ScheduledInferenceKind.Decode,
            sequenceId,
            default,
            TokenCount: 1,
            CompletesPrefill: false);

    internal string Operation => Kind switch
    {
        ScheduledInferenceKind.Prefill => nameof(PrefillExecutionStep),
        ScheduledInferenceKind.Decode => nameof(DecodeExecutionStep),
        _ => throw new NotSupportedException(
            $"Unsupported scheduled inference kind {Kind}.")
    };
}

public sealed partial class ExecutionPlanExecutor
{
    /// <summary>
    /// Executes the scheduler's validated one-step prefill/decode work without
    /// materializing a generic ExecutionStep or CompiledExecutionPlan and their
    /// reservation/result bookkeeping. Atomic actor registration is passed
    /// explicitly so the scheduled hot path does not need an AsyncLocal slot scope.
    /// </summary>
    internal async ValueTask<BackendStepResult> ExecuteScheduledInferenceAsync(
        Guid planId,
        ScheduledInferenceStep step,
        ReadOnlyMemory<int> prefillTokens,
        ScheduledDeviceGroup deviceGroup,
        int slot,
        ScheduledDeviceGroupTable groups,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(deviceGroup);
        ArgumentNullException.ThrowIfNull(groups);

        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            var submission = deviceGroup.Submission;

            var operation = step.Operation;
            ReserveSequence(step.SequenceId, $"plan {planId}");
            try
            {
                Record(new ExecutionTraceEvent(
                    planId,
                    ExecutionTraceKind.PlanStarted,
                    -1,
                    "Plan"));

                cancellationToken.ThrowIfCancellationRequested();

                RecordSequenceState(
                    planId,
                    ExecutionTraceKind.StepStarted,
                    0,
                    operation,
                    step.SequenceId);

                BackendStepResult result = step.Kind switch
                {
                    ScheduledInferenceKind.Prefill => await ExecuteScheduledPrefillAsync(
                            step,
                            prefillTokens,
                            submission,
                            slot,
                            cancellationToken)
                        .ConfigureAwait(false),
                    ScheduledInferenceKind.Decode => await ExecuteScheduledDecodeAsync(
                            step,
                            submission,
                            slot,
                            cancellationToken)
                        .ConfigureAwait(false),
                    _ => throw new NotSupportedException(
                        $"Unsupported scheduled inference kind {step.Kind}.")
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

                return result;
            }
            finally
            {
                _sequenceReservations.TryRemove(step.SequenceId, out _);
            }
        }
        catch (Exception exception)
        {
            groups.Abort(exception);
            throw;
        }
        finally
        {
            deviceGroup.CompleteItem();
        }
    }

    private async ValueTask<BackendStepResult> ExecuteScheduledPrefillAsync(
        ScheduledInferenceStep step,
        ReadOnlyMemory<int> tokens,
        ContinuousBatchExecutor.AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(step.TokenCount);
        if (tokens.Length != step.TokenCount)
        {
            throw new InvalidOperationException(
                $"Plan expects {step.TokenCount} prefill tokens for {step.SequenceId}, " +
                $"but binding contains {tokens.Length}.");
        }

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

        var position = sequence.Position;
        var kvWrite = sequence.KvWriteIntentFor(step.TokenCount);
        var device = _devices.ResolvePlacement(sequence.Device);
        var result = await device.SubmitPrefillAsync(
                new PrefillItem(
                    step.SequenceId,
                    step.ModelId,
                    tokens,
                    Position: position,
                    KvWrite: kvWrite),
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
        ScheduledInferenceStep step,
        ContinuousBatchExecutor.AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken)
    {
        if (step.TokenCount != 1)
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

        var position = sequence.Position;
        var kvWrite = sequence.KvWriteIntentFor(1);
        var device = _devices.ResolvePlacement(sequence.Device);
        var result = await device.SubmitDecodeAsync(
                new DecodeItem(
                    sequence.Id,
                    sequence.Model,
                    position,
                    KvWrite: kvWrite),
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
