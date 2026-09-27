using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    /// <summary>
    /// Transactional migration controls are allowed to observe cooperative
    /// cancellation after queue acceptance because the migration protocol owns
    /// rollback semantics for partial physical state. General control/inference
    /// work keeps the actor's original acceptance-boundary cancellation rules.
    /// </summary>
    internal ValueTask ImportSequenceMigrationCancellableAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken) =>
        SubmitControlAsync(
            new PendingCancellableImportMigration(transfer, cancellationToken),
            cancellationToken);

    internal ValueTask CommitSequenceMigrationCancellableAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken) =>
        SubmitControlAsync(
            new PendingCancellableCommitMigration(transfer, cancellationToken),
            cancellationToken);

    internal ValueTask AbortSequenceMigrationCancellableAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken) =>
        SubmitControlAsync(
            new PendingCancellableAbortMigration(transfer, cancellationToken),
            cancellationToken);

    private sealed class PendingCancellableImportMigration(
        SequenceMigrationTransfer transfer,
        CancellationToken operationCancellationToken) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            if (transfer.TargetDevice != backend.Device)
            {
                throw new InvalidOperationException(
                    $"Migration transfer {transfer.TransactionId} targets {transfer.TargetDevice}, " +
                    $"but import was submitted to actor {backend.Device}.");
            }

            return RequireTransactionalMigration(backend)
                .ImportSequenceMigrationAsync(transfer, operationCancellationToken);
        }
    }

    private sealed class PendingCancellableCommitMigration(
        SequenceMigrationTransfer transfer,
        CancellationToken operationCancellationToken) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            if (transfer.SourceDevice != backend.Device)
            {
                throw new InvalidOperationException(
                    $"Migration transfer {transfer.TransactionId} originates on {transfer.SourceDevice}, " +
                    $"but commit was submitted to actor {backend.Device}.");
            }

            return RequireTransactionalMigration(backend)
                .CommitSequenceMigrationAsync(transfer, operationCancellationToken);
        }
    }

    private sealed class PendingCancellableAbortMigration(
        SequenceMigrationTransfer transfer,
        CancellationToken operationCancellationToken) : PendingControl
    {
        public override ValueTask ExecuteAsync(IInferenceBackend backend)
        {
            if (backend.Device != transfer.SourceDevice && backend.Device != transfer.TargetDevice)
            {
                throw new InvalidOperationException(
                    $"Migration transfer {transfer.TransactionId} belongs to " +
                    $"{transfer.SourceDevice}->{transfer.TargetDevice}, but abort was submitted to {backend.Device}.");
            }

            return RequireTransactionalMigration(backend)
                .AbortSequenceMigrationAsync(transfer, operationCancellationToken);
        }
    }
}
