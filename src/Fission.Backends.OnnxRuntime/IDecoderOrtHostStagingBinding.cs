using Fission.Abstractions.Execution;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Host-memory representation of one decoder state during staged migration.
/// Concrete codecs own the payload format and may own pooled/pinned buffers.
///
/// One owner reference belongs to the migration transfer. Codecs whose imported
/// OrtValues are views over payload memory must call <see cref="Retain"/> and pass
/// the returned lease as the DecoderOrtState lifetime anchor. The source-side
/// terminal commit/abort releases the transfer owner; buffers are reclaimed only
/// after every retained state lease is also released.
/// </summary>
public abstract class DecoderOrtHostStagingPayload : IDisposable
{
    private int _referenceCount = 1;
    private int _ownerReleased;
    private int _resourcesDisposed;

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
    public bool IsDisposed => Volatile.Read(ref _resourcesDisposed) != 0;

    /// <summary>
    /// Retains payload-owned host memory for a state whose OrtValues alias that
    /// memory. Dispose the returned lease only after all such OrtValues are gone.
    /// DecoderOrtState recognizes this lease as an owned lifetime anchor and does
    /// that automatically when the lease is passed to its lifetime-anchor constructor.
    /// </summary>
    public IDisposable Retain()
    {
        while (true)
        {
            if (Volatile.Read(ref _ownerReleased) != 0)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            var current = Volatile.Read(ref _referenceCount);
            if (current <= 0)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            if (current == int.MaxValue)
            {
                throw new InvalidOperationException("Host-staging payload reference count overflowed.");
            }

            if (Interlocked.CompareExchange(
                    ref _referenceCount,
                    current + 1,
                    current) == current)
            {
                return new PayloadLifetimeLease(this);
            }
        }
    }

    /// <summary>
    /// Releases the migration transfer's owner reference. Derived payloads can
    /// override <see cref="DisposeCore"/> to return pooled/pinned buffers when the
    /// final retained state lease is also gone.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _ownerReleased, 1) != 0)
        {
            return;
        }

        ReleaseReference();
        GC.SuppressFinalize(this);
    }

    protected virtual void DisposeCore()
    {
    }

    private void ReleaseReference()
    {
        var remaining = Interlocked.Decrement(ref _referenceCount);
        if (remaining < 0)
        {
            throw new InvalidOperationException("Host-staging payload reference count underflowed.");
        }

        if (remaining != 0)
        {
            return;
        }

        if (Interlocked.Exchange(ref _resourcesDisposed, 1) == 0)
        {
            DisposeCore();
        }
    }

    private sealed class PayloadLifetimeLease :
        IDisposable,
        IDecoderOrtOwnedLifetimeAnchor
    {
        private DecoderOrtHostStagingPayload? _payload;

        public PayloadLifetimeLease(DecoderOrtHostStagingPayload payload)
        {
            _payload = payload;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _payload, null)?.ReleaseReference();
        }

        void IDecoderOrtOwnedLifetimeAnchor.Release() => Dispose();
    }
}

/// <summary>
/// Physical decoder-state codec used by the migration protocol. The codec is
/// deliberately independent from model execution capabilities so a high-throughput
/// batch/chunked model binding can keep its exact runtime type while migration is
/// supplied by a separate component.
///
/// Export must deep-copy the live state into payload-owned host memory. Import
/// must create a newly owned DecoderOrtState whose OrtValues do not alias source
/// decoder OrtValues. If imported OrtValues alias payload memory, retain the
/// payload and pass that lease as the DecoderOrtState lifetime anchor.
/// </summary>
public interface IDecoderOrtHostStagingCodec
{
    string Name { get; }

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

/// <summary>
/// Backward-compatible combined capability for bindings that own both model
/// execution and their migration codec. Existing bindings can keep implementing
/// this interface; the adapter treats the binding itself as its default codec.
/// </summary>
public interface IDecoderOrtHostStagingBinding :
    IDecoderOrtModelBinding,
    IDecoderOrtHostStagingCodec
{
}

/// <summary>
/// Optional codec extension whose source export has a real asynchronous completion
/// boundary, such as CUDA device-to-host DMA.
/// </summary>
public interface IDecoderOrtAsyncHostStagingCodec : IDecoderOrtHostStagingCodec
{
    ValueTask<DecoderOrtHostStagingPayload> ExportHostStagingStateAsync(
        DecoderOrtState state,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Backward-compatible combined binding form of
/// <see cref="IDecoderOrtAsyncHostStagingCodec"/>.
/// </summary>
public interface IDecoderOrtAsyncHostStagingBinding :
    IDecoderOrtHostStagingBinding,
    IDecoderOrtAsyncHostStagingCodec
{
}

/// <summary>
/// Optional codec extension whose target import has a real asynchronous completion
/// boundary, such as CUDA host-to-device DMA.
/// </summary>
public interface IDecoderOrtAsyncHostStagingImportCodec : IDecoderOrtHostStagingCodec
{
    ValueTask<DecoderOrtState> ImportHostStagingStateAsync(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Backward-compatible combined binding form of
/// <see cref="IDecoderOrtAsyncHostStagingImportCodec"/>.
/// </summary>
public interface IDecoderOrtAsyncHostStagingImportBinding :
    IDecoderOrtHostStagingBinding,
    IDecoderOrtAsyncHostStagingImportCodec
{
}
