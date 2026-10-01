using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    internal ValueTask<BackendStepResult> SubmitPrefillAsync(
        PrefillItem item,
        AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken = default) =>
        SubmitScheduledAtomicInferenceAsync(
            RentPrefill(item),
            submission,
            slot,
            cancellationToken);

    internal ValueTask<BackendStepResult> SubmitDecodeAsync(
        DecodeItem item,
        AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken = default) =>
        SubmitScheduledAtomicInferenceAsync(
            RentDecode(item),
            submission,
            slot,
            cancellationToken);

    private ValueTask<BackendStepResult> SubmitScheduledAtomicInferenceAsync<TWork>(
        TWork work,
        AtomicSubmissionBatch submission,
        int slot,
        CancellationToken cancellationToken)
        where TWork : PendingInference
    {
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            ArgumentNullException.ThrowIfNull(submission);

            var registration = submission.RegisterAsync(
                slot,
                this,
                work,
                cancellationToken);
            return registration.IsCompletedSuccessfully
                ? work.WaitAsync()
                : AwaitRegistrationAndCompletionAsync(registration, work);
        }
        catch (Exception exception)
        {
            work.ReleaseCredits();
            work.AbandonSubmission();
            return ValueTask.FromException<BackendStepResult>(exception);
        }
    }
}
