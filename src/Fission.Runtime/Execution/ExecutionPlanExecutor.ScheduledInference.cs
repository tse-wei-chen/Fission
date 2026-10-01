using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Tracing;

namespace Fission.Runtime.Execution;

public sealed partial class ExecutionPlanExecutor
{
    /// <summary>
    /// Executes the scheduler's validated one-step prefill/decode plan without
    /// materializing a generic CompiledExecutionPlan and its reservation/result
    /// bookkeeping. The sequence reservation and trace boundary intentionally
    /// match ExecuteAsync for a one-step inference-only plan.
    /// </summary>
    internal async ValueTask<ExecutionPlanResult> ExecuteScheduledInferenceAsync(
        Guid planId,
        ExecutionStep step,
        ExecutionBindings bindings,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(step);
        ArgumentNullException.ThrowIfNull(bindings);

        if (step is not PrefillExecutionStep and not DecodeExecutionStep)
        {
            throw new NotSupportedException(
                $"Scheduled single-inference execution does not support {step.GetType().Name}.");
        }

        ReserveSequence(step.SequenceId, $"plan {planId}");
        try
        {
            var backendResults = new List<BackendStepResult>(1);

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

            switch (step)
            {
                case PrefillExecutionStep prefill:
                    await ExecutePrefillAsync(
                            prefill,
                            bindings,
                            backendResults,
                            cancellationToken)
                        .ConfigureAwait(false);
                    break;

                case DecodeExecutionStep decode:
                    await ExecuteDecodeAsync(
                            decode,
                            backendResults,
                            cancellationToken)
                        .ConfigureAwait(false);
                    break;
            }

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
                backendResults,
                Array.Empty<KvSnapshotId>(),
                Array.Empty<ForkExecutionResult>());
        }
        finally
        {
            _sequenceReservations.TryRemove(step.SequenceId, out _);
        }
    }
}
