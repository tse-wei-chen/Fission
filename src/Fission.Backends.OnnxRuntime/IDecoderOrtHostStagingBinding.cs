using Fission.Abstractions.Execution;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Immutable managed-memory representation of one decoder state during host-staged
/// migration. Concrete bindings own the payload format; the generic adapter only
/// carries it between source and target actors.
/// </summary>
public abstract class DecoderOrtHostStagingPayload
{
    protected DecoderOrtHostStagingPayload(
        string formatId,
        int position,
        int? nextTokenId,
        long byteLength)
    {
        if (string.IsNullOrWhiteSpace(formatId))
        {
            throw new ArgumentException("Host-staging format id cannot be empty.", nameof(formatId));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (nextTokenId is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(nextTokenId));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);

        FormatId = formatId;
        Position = position;
        NextTokenId = nextTokenId;
        ByteLength = byteLength;
    }

    public string FormatId { get; }
    public int Position { get; }
    public int? NextTokenId { get; }
    public long ByteLength { get; }
}

/// <summary>
/// Optional decoder-binding codec for a real host-staging migration data path.
/// Export must deep-copy the immutable live state into managed host memory.
/// Import must create a newly owned DecoderOrtState whose OrtValues do not alias
/// the source decoder state's OrtValue instances.
/// </summary>
public interface IDecoderOrtHostStagingBinding : IDecoderOrtModelBinding
{
    /// <summary>
    /// Versioned payload compatibility key. The ONNX backend combines this with
    /// ModelId to build the runtime transport id, so peers only negotiate when
    /// both the model and payload codec match.
    /// </summary>
    string HostStagingFormatId { get; }

    long EstimateHostStagingBytes(DecoderOrtState state);

    DecoderOrtHostStagingPayload ExportHostStagingState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default);

    DecoderOrtState ImportHostStagingState(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default);
}
