namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    // Exact overloads keep the generic async implementation as a safe fallback for
    // future inference work types while routing the two production hot-path work
    // items through a non-async common path.
    private ValueTask<Fission.Abstractions.Execution.BackendStepResult> SubmitInferenceAsync(
        PendingPrefill work,
        CancellationToken cancellationToken) =>
        SubmitInferenceFastPath(work, cancellationToken);

    private ValueTask<Fission.Abstractions.Execution.BackendStepResult> SubmitInferenceAsync(
        PendingDecode work,
        CancellationToken cancellationToken) =>
        SubmitInferenceFastPath(work, cancellationToken);

    private ValueTask<Fission.Abstractions.Execution.BackendStepResult> SubmitInferenceFastPath<TWork>(
        TWork work,
        CancellationToken cancellationToken)
        where TWork : PendingInference
    {
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            if (AmbientAtomicSlot.Value is { } slot)
            {
                var registration = slot.Batch.RegisterAsync(
                    slot.Index,
                    this,
                    work,
                    cancellationToken);
                return registration.IsCompletedSuccessfully
                    ? work.WaitAsync()
                    : AwaitRegistrationAndCompletionAsync(registration, work);
            }

            var acquisition = _inferenceCredits.AcquireAsync(1, cancellationToken);
            if (!acquisition.IsCompletedSuccessfully)
            {
                return AwaitAdmissionQueueAndCompletionAsync(
                    acquisition,
                    work,
                    cancellationToken);
            }

            work.AttachCredits(acquisition.Result);
            work.RetainActorOwnership();
            ValueTask write;
            try
            {
                write = _queue.Writer.WriteAsync(work, cancellationToken);
            }
            catch
            {
                work.ReleaseActorOwnership();
                work.ReleaseCredits();
                throw;
            }

            return write.IsCompletedSuccessfully
                ? work.WaitAsync()
                : AwaitQueueAndCompletionAsync(write, work);
        }
        catch (Exception exception)
        {
            // The previous async method captured synchronous admission failures in
            // its returned ValueTask. Preserve that API behavior on the non-async
            // fast path instead of throwing from SubmitPrefill/DecodeAsync itself.
            work.ReleaseCredits();
            work.AbandonSubmission();
            return ValueTask.FromException<Fission.Abstractions.Execution.BackendStepResult>(
                exception);
        }
    }

    private static async ValueTask<Fission.Abstractions.Execution.BackendStepResult>
        AwaitRegistrationAndCompletionAsync<TWork>(
            ValueTask registration,
            TWork work)
        where TWork : PendingInference
    {
        try
        {
            await registration.ConfigureAwait(false);
        }
        catch
        {
            work.AbandonSubmission();
            throw;
        }

        return await work.WaitAsync().ConfigureAwait(false);
    }

    private async ValueTask<Fission.Abstractions.Execution.BackendStepResult>
        AwaitAdmissionQueueAndCompletionAsync<TWork>(
            ValueTask<InferenceCreditGate.Lease> acquisition,
            TWork work,
            CancellationToken cancellationToken)
        where TWork : PendingInference
    {
        var actorOwned = false;
        try
        {
            var credits = await acquisition.ConfigureAwait(false);
            work.AttachCredits(credits);
            work.RetainActorOwnership();
            actorOwned = true;
            await _queue.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            if (actorOwned)
            {
                work.ReleaseActorOwnership();
            }

            work.ReleaseCredits();
            work.AbandonSubmission();
            throw;
        }

        return await work.WaitAsync().ConfigureAwait(false);
    }

    private static async ValueTask<Fission.Abstractions.Execution.BackendStepResult>
        AwaitQueueAndCompletionAsync<TWork>(
            ValueTask write,
            TWork work)
        where TWork : PendingInference
    {
        try
        {
            await write.ConfigureAwait(false);
        }
        catch
        {
            work.ReleaseActorOwnership();
            work.ReleaseCredits();
            work.AbandonSubmission();
            throw;
        }

        return await work.WaitAsync().ConfigureAwait(false);
    }
}
