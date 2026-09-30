using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    internal async ValueTask<InferenceDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
        long targetReclaimableBytes,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetReclaimableBytes);
        var work = new PendingMemoryReclaim(targetReclaimableBytes);
        await SubmitControlAsync(work, cancellationToken).ConfigureAwait(false);
        return work.Result ?? throw new InvalidOperationException(
            $"Backend {BackendName} completed device-memory reclaim without a result.");
    }

    private sealed class PendingMemoryReclaim(
        long targetReclaimableBytes) : PendingControl
    {
        public InferenceDeviceMemoryReclaimResult? Result { get; private set; }

        public override async ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            var reclaimer = backend as IInferenceDeviceMemoryReclaimer ??
                throw new NotSupportedException(
                    $"Backend '{backend.Name}' does not implement device-memory reclaim.");

            var result = await reclaimer.ReclaimDeviceMemoryAsync(targetReclaimableBytes)
                .ConfigureAwait(false);
            Validate(result);
            Result = result;
        }

        private static void Validate(InferenceDeviceMemoryReclaimResult result)
        {
            if (result.ReleasedBytes < 0 ||
                result.ReclaimableBytes < 0 ||
                result.ReservedBytes < 0)
            {
                throw new InvalidOperationException(
                    "Backend device-memory reclaim result cannot contain negative byte counts.");
            }

            if (result.ReclaimableBytes > result.ReservedBytes)
            {
                throw new InvalidOperationException(
                    "Backend reclaimable device memory cannot exceed current reserved bytes.");
            }
        }
    }
}
