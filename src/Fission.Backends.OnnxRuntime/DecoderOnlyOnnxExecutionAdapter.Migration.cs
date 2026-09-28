using Fission.Abstractions;
using Fission.Abstractions.Execution;

namespace Fission.Backends.OnnxRuntime;

internal sealed class DecoderOrtHostStagingTransfer : SequenceMigrationTransfer
{
    private int _payloadOwnerReleased;

    public DecoderOrtHostStagingTransfer(
        Guid transactionId,
        SequenceId sequenceId,
        DeviceId sourceDevice,
        DeviceId targetDevice,
        ModelId modelId,
        DecoderOrtHostStagingPayload payload,
        SequenceMigrationTransportPlan? transportPlan)
        : base(
            transactionId,
            sequenceId,
            sourceDevice,
            targetDevice,
            transportPlan)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ModelId = modelId;
        Payload = payload;
    }

    public ModelId ModelId { get; }
    public DecoderOrtHostStagingPayload Payload { get; }

    public void ReleasePayloadOwner()
    {
        if (Interlocked.Exchange(ref _payloadOwnerReleased, 1) == 0)
        {
            Payload.Dispose();
        }
    }
}

public sealed partial class DecoderOnlyOnnxExecutionAdapter :
    IOnnxRuntimeSequenceMigrationAdapter
{
    private const long HostStagingBandwidthBytesPerSecond = 8L * 1024 * 1024 * 1024;
    private static readonly TimeSpan HostStagingFixedLatency = TimeSpan.FromTicks(500);

    bool IOnnxRuntimeSequenceMigrationAdapter.SupportsSequenceMigration =>
        _binding is IDecoderOrtHostStagingBinding;

    IReadOnlyList<SequenceMigrationTransportCapability>
        IOnnxRuntimeSequenceMigrationAdapter.GetSequenceMigrationTransportCapabilities(
            ModelId modelId,
            DeviceId localDevice,
            DeviceId peerDevice)
    {
        EnsureInitialized();
        var binding = RequireHostStagingBinding();
        return new[]
        {
            new SequenceMigrationTransportCapability(
                BuildHostStagingTransportId(modelId, binding.HostStagingFormatId),
                SequenceMigrationTransportKind.HostStaging,
                MaxTransferBytes: 0,
                EstimatedBandwidthBytesPerSecond: HostStagingBandwidthBytesPerSecond,
                EstimatedFixedLatency: HostStagingFixedLatency,
                Preference: 0)
        };
    }

    ValueTask<long> IOnnxRuntimeSequenceMigrationAdapter.EstimateSequenceMigrationBytesAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var binding = RequireHostStagingBinding();
        var state = _states.GetSequence(sequenceId);
        var bytes = binding.EstimateHostStagingBytes(state);
        if (bytes <= 0)
        {
            throw new InvalidOperationException(
                $"Decoder host-staging binding '{binding.Name}' estimated {bytes} byte(s) for sequence {sequenceId}.");
        }

        return ValueTask.FromResult(bytes);
    }

    async ValueTask<SequenceMigrationTransfer>
        IOnnxRuntimeSequenceMigrationAdapter.PrepareSequenceMigrationAsync(
            ModelId modelId,
            DeviceId sourceDevice,
            SequenceId sequenceId,
            DeviceId targetDevice,
            SequenceMigrationTransportPlan? transportPlan,
            CancellationToken cancellationToken)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var binding = RequireHostStagingBinding();
        var state = _states.GetSequence(sequenceId);
        var expectedBytes = binding.EstimateHostStagingBytes(state);
        if (expectedBytes <= 0)
        {
            throw new InvalidOperationException(
                $"Decoder host-staging binding '{binding.Name}' estimated {expectedBytes} byte(s) for sequence {sequenceId}.");
        }

        if (transportPlan is not null)
        {
            ValidateTransportPlan(modelId, binding, expectedBytes, transportPlan);
        }

        var payload = binding is IDecoderOrtAsyncHostStagingBinding asyncBinding
            ? await asyncBinding.ExportHostStagingStateAsync(
                state,
                cancellationToken).ConfigureAwait(false)
            : binding.ExportHostStagingState(state, cancellationToken);
        ArgumentNullException.ThrowIfNull(payload);
        try
        {
            if (!StringComparer.Ordinal.Equals(payload.FormatId, binding.HostStagingFormatId))
            {
                throw new InvalidOperationException(
                    $"Decoder host-staging binding '{binding.Name}' exported format '{payload.FormatId}', " +
                    $"but advertises '{binding.HostStagingFormatId}'.");
            }

            if (payload.Position != state.Position || payload.NextTokenId != state.NextTokenId)
            {
                throw new InvalidOperationException(
                    "Decoder host-staging payload causal frontier does not match the source state.");
            }

            if (payload.ByteLength != expectedBytes)
            {
                throw new InvalidOperationException(
                    $"Decoder host-staging payload contains {payload.ByteLength} byte(s), but prepare estimated {expectedBytes}.");
            }

            return new DecoderOrtHostStagingTransfer(
                Guid.NewGuid(),
                sequenceId,
                sourceDevice,
                targetDevice,
                modelId,
                payload,
                transportPlan);
        }
        catch
        {
            payload.Dispose();
            throw;
        }
    }

    async ValueTask IOnnxRuntimeSequenceMigrationAdapter.ImportSequenceMigrationAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var binding = RequireHostStagingBinding();
        var staged = RequireTransfer(modelId, transfer);
        if (staged.TargetDevice != localDevice)
        {
            throw new InvalidOperationException(
                $"Decoder migration {staged.TransactionId} targets {staged.TargetDevice}, not local device {localDevice}.");
        }

        ValidatePayloadFormat(binding, staged.Payload);
        var imported = await ImportHostStagingStateAsync(
            binding,
            staged.Payload,
            cancellationToken).ConfigureAwait(false);
        ArgumentNullException.ThrowIfNull(imported);
        try
        {
            ValidateImportedState(staged.Payload, imported);
            _states.AddSequence(staged.SequenceId, imported);
        }
        catch
        {
            imported.Dispose();
            throw;
        }
    }

    ValueTask IOnnxRuntimeSequenceMigrationAdapter.CommitSequenceMigrationAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var staged = RequireTransfer(modelId, transfer);
        if (staged.SourceDevice != localDevice)
        {
            throw new InvalidOperationException(
                $"Decoder migration {staged.TransactionId} originates on {staged.SourceDevice}, not local device {localDevice}.");
        }

        if (!_states.ReleaseSequence(staged.SequenceId))
        {
            throw new KeyNotFoundException(
                $"Decoder source state does not exist for migration sequence {staged.SequenceId}.");
        }

        staged.ReleasePayloadOwner();
        return ValueTask.CompletedTask;
    }

    async ValueTask IOnnxRuntimeSequenceMigrationAdapter.AbortSequenceMigrationAsync(
        ModelId modelId,
        DeviceId localDevice,
        SequenceMigrationTransfer transfer,
        CancellationToken cancellationToken)
    {
        EnsureInitialized();
        cancellationToken.ThrowIfCancellationRequested();
        var binding = RequireHostStagingBinding();
        var staged = RequireTransfer(modelId, transfer);

        if (localDevice == staged.TargetDevice)
        {
            _states.ReleaseSequence(staged.SequenceId);
            return;
        }

        if (localDevice != staged.SourceDevice)
        {
            throw new InvalidOperationException(
                $"Device {localDevice} does not participate in decoder migration {staged.TransactionId}.");
        }

        if (_states.TryGetSequence(staged.SequenceId, out _))
        {
            staged.ReleasePayloadOwner();
            return;
        }

        ValidatePayloadFormat(binding, staged.Payload);
        var restored = await ImportHostStagingStateAsync(
            binding,
            staged.Payload,
            cancellationToken).ConfigureAwait(false);
        ArgumentNullException.ThrowIfNull(restored);
        try
        {
            ValidateImportedState(staged.Payload, restored);
            _states.AddSequence(staged.SequenceId, restored);
            staged.ReleasePayloadOwner();
        }
        catch
        {
            restored.Dispose();
            throw;
        }
    }

    private IDecoderOrtHostStagingBinding RequireHostStagingBinding() =>
        _binding as IDecoderOrtHostStagingBinding ??
        throw new NotSupportedException(
            $"Decoder binding '{_binding.Name}' does not implement host-staging migration.");

    private static string BuildHostStagingTransportId(
        ModelId modelId,
        string formatId) =>
        $"onnx-host-staging:{modelId.Value}:{formatId}";

    private static void ValidateTransportPlan(
        ModelId modelId,
        IDecoderOrtHostStagingBinding binding,
        long expectedBytes,
        SequenceMigrationTransportPlan transportPlan)
    {
        var expectedTransportId = BuildHostStagingTransportId(
            modelId,
            binding.HostStagingFormatId);
        if (transportPlan.Kind != SequenceMigrationTransportKind.HostStaging ||
            !StringComparer.Ordinal.Equals(transportPlan.TransportId, expectedTransportId))
        {
            throw new InvalidOperationException(
                $"Decoder binding '{binding.Name}' cannot materialize migration transport " +
                $"'{transportPlan.TransportId}' ({transportPlan.Kind}); expected '{expectedTransportId}'.");
        }

        if (transportPlan.EstimatedBytes != expectedBytes)
        {
            throw new InvalidOperationException(
                $"Decoder migration plan estimates {transportPlan.EstimatedBytes} byte(s), " +
                $"but the source state requires {expectedBytes}.");
        }
    }

    private static DecoderOrtHostStagingTransfer RequireTransfer(
        ModelId modelId,
        SequenceMigrationTransfer transfer)
    {
        ArgumentNullException.ThrowIfNull(transfer);
        var staged = transfer as DecoderOrtHostStagingTransfer ??
            throw new InvalidOperationException(
                $"Unsupported ONNX decoder migration transfer type {transfer.GetType().Name}.");
        if (staged.ModelId != modelId)
        {
            throw new InvalidOperationException(
                $"Decoder migration {staged.TransactionId} belongs to model {staged.ModelId}, not {modelId}.");
        }

        return staged;
    }

    private static void ValidatePayloadFormat(
        IDecoderOrtHostStagingBinding binding,
        DecoderOrtHostStagingPayload payload)
    {
        if (!StringComparer.Ordinal.Equals(payload.FormatId, binding.HostStagingFormatId))
        {
            throw new InvalidOperationException(
                $"Decoder host-staging payload format '{payload.FormatId}' is incompatible with " +
                $"binding format '{binding.HostStagingFormatId}'.");
        }
    }

    private static async ValueTask<DecoderOrtState> ImportHostStagingStateAsync(
        IDecoderOrtHostStagingBinding binding,
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken)
    {
        if (binding is IDecoderOrtAsyncHostStagingImportBinding asyncBinding)
        {
            return await asyncBinding.ImportHostStagingStateAsync(
                payload,
                cancellationToken).ConfigureAwait(false);
        }

        return binding.ImportHostStagingState(payload, cancellationToken);
    }

    private static void ValidateImportedState(
        DecoderOrtHostStagingPayload payload,
        DecoderOrtState state)
    {
        if (state.IsDisposed)
        {
            throw new InvalidOperationException(
                "Decoder host-staging import returned an already-disposed state.");
        }

        if (state.Position != payload.Position || state.NextTokenId != payload.NextTokenId)
        {
            throw new InvalidOperationException(
                "Decoder host-staging import changed the causal frontier.");
        }
    }
}
