namespace Fission.Abstractions.Execution;

/// <summary>
/// Opaque backend-owned transfer object carried by the runtime between source and
/// target device actors. Implementations may wrap host buffers, CUDA IPC handles,
/// NIXL/RDMA descriptors, shared immutable state, or another backend-specific
/// transfer representation. The runtime never inspects the payload.
/// </summary>
public abstract class SequenceMigrationTransfer
{
    protected SequenceMigrationTransfer(
        Guid transactionId,
        SequenceId sequenceId,
        DeviceId sourceDevice,
        DeviceId targetDevice,
        SequenceMigrationTransportPlan? transportPlan = null)
    {
        if (transactionId == Guid.Empty)
        {
            throw new ArgumentException("Migration transaction id cannot be empty.", nameof(transactionId));
        }

        TransactionId = transactionId;
        SequenceId = sequenceId;
        SourceDevice = sourceDevice;
        TargetDevice = targetDevice;
        TransportPlan = transportPlan;
    }

    public Guid TransactionId { get; }
    public SequenceId SequenceId { get; }
    public DeviceId SourceDevice { get; }
    public DeviceId TargetDevice { get; }

    /// <summary>
    /// Physical transport decision attested by a transport-aware backend.
    /// Legacy migration backends leave this null.
    /// </summary>
    public SequenceMigrationTransportPlan? TransportPlan { get; }
}

/// <summary>
/// Optional two-actor migration protocol for stateful backends.
///
/// The runtime executes the protocol in this order:
/// 1. prepare on the source actor,
/// 2. import on the target actor,
/// 3. commit on the source actor,
/// 4. publish runtime placement.
///
/// If import or commit fails, AbortSequenceMigrationAsync is invoked best-effort
/// on target and source with caller cancellation suppressed. Implementations must
/// therefore retain enough source state during prepare/commit to restore the
/// source when abort follows a failed commit. Import must likewise be reversible.
/// Prepare failures must leave the source usable because no transfer token exists
/// for the runtime to abort.
///
/// Commit is the source-side destructive/finalizing phase. A successful commit
/// means the target already owns runnable state. Runtime metadata is changed only
/// after commit returns successfully. Commit and abort own cleanup of any transport
/// resources referenced by the transfer token.
/// </summary>
public interface ISequenceMigrationBackend
{
    /// <summary>
    /// Allows a backend type that conditionally exposes migration (for example a
    /// generic session host whose adapter may or may not own migratable state) to
    /// participate without making every instance appear migration-capable.
    /// Existing dedicated migration backends inherit the default true value.
    /// </summary>
    bool SupportsSequenceMigration => true;

    ValueTask<SequenceMigrationTransfer> PrepareSequenceMigrationAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default);

    ValueTask ImportSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default);

    ValueTask CommitSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default);

    ValueTask AbortSequenceMigrationAsync(
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken = default);
}
