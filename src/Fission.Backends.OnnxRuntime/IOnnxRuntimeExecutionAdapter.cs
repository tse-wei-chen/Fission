using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Model-specific ONNX execution policy. The backend owns the InferenceSession;
/// the adapter owns the model's tensor schema, batching rules, sampling policy,
/// and per-sequence/snapshot state such as logits or KV OrtValues.
///
/// Calls are serialized by ContinuousBatchExecutor, so implementations do not
/// need to make their mutable state concurrently writable.
/// </summary>
public interface IOnnxRuntimeExecutionAdapter : IDisposable
{
    string Name { get; }

    ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }

    ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        InferenceSession session,
        PrefillBatch batch,
        CancellationToken cancellationToken = default);

    ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        InferenceSession session,
        DecodeBatch batch,
        CancellationToken cancellationToken = default);

    ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException(
            $"ONNX adapter '{Name}' does not implement backend snapshot state.");

    ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException(
            $"ONNX adapter '{Name}' does not implement backend fork state.");

    ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException(
            $"ONNX adapter '{Name}' does not implement backend restore state.");

    ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException(
            $"ONNX adapter '{Name}' does not implement backend snapshot release.");

    ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default);
}
