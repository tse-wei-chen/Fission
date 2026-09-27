using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Opt-in decorator that adds a managed-host migration codec to the existing
/// Optimum legacy FP32 decoder binding without changing its inference behavior.
/// Source export deep-copies every K/V tensor into exact-length GC-pinned pooled
/// arrays. Target import creates new OrtValue views over those staged arrays and
/// retains the payload until the imported immutable state is disposed.
/// </summary>
public sealed class OptimumLegacyFloatHostStagingBinding :
    IDecoderOrtHostStagingBinding,
    IDecoderOrtBatchPrefillModelBinding,
    IDecoderOrtBatchModelBinding,
    IDecoderOrtChunkedPrefillModelBinding
{
    private const long MetadataBytes = sizeof(int) * 2L;
    private readonly OptimumLegacyFloatDecoderBinding _inner;
    private readonly PinnedFloatBufferPool _stagingPool;
    private int _disposed;

    public OptimumLegacyFloatHostStagingBinding(
        OptimumLegacyFloatDecoderBinding inner)
        : this(inner, stagingPoolOptions: null)
    {
    }

    public OptimumLegacyFloatHostStagingBinding(
        OptimumLegacyFloatDecoderBinding inner,
        PinnedHostStagingPoolOptions? stagingPoolOptions)
    {
        ArgumentNullException.ThrowIfNull(inner);
        _inner = inner;
        _stagingPool = new PinnedFloatBufferPool(stagingPoolOptions);
    }

    public string Name => _inner.Name;
    public OnnxSessionContract SessionContract => _inner.SessionContract;
    public DecoderOrtGeometry Geometry => _inner.Geometry;
    public OptimumLegacyFloatDecoderBinding Inner => _inner;
    public PinnedHostStagingPoolStatistics HostStagingPoolStatistics =>
        _stagingPool.GetStatistics();

    public string HostStagingFormatId =>
        $"optimum-legacy-fp32-kv-v1:l{Geometry.NumHiddenLayers}:h{Geometry.NumKvHeads}:d{Geometry.HeadDim}";

    public ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.InitializeAsync(session, cancellationToken);
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecutePrefill(session, item, cancellationToken);
    }

    public DecoderOrtStepResult ExecutePrefillChunk(
        InferenceSession session,
        PrefillItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecutePrefillChunk(
            session,
            item,
            priorState,
            cancellationToken);
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecutePrefillBatch(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecutePrefillBatch(
            session,
            items,
            priorStates,
            cancellationToken);
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecuteDecode(session, item, priorState, cancellationToken);
    }

    public IReadOnlyList<DecoderOrtStepResult> ExecuteDecodeBatch(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecuteDecodeBatch(
            session,
            items,
            priorStates,
            cancellationToken);
    }

    public long EstimateHostStagingBytes(DecoderOrtState state)
    {
        ThrowIfDisposed();
        ValidateState(state);
        return checked(Geometry.GetKvBytesPerSequence(state.Position) + MetadataBytes);
    }

    public DecoderOrtHostStagingPayload ExportHostStagingState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        ValidateState(state);

        var expectedElementsPerTensor = checked(
            Geometry.NumKvHeads * state.Position * Geometry.HeadDim);
        var keys = new PinnedFloatBufferPool.PinnedFloatBufferLease[Geometry.NumHiddenLayers];
        var values = new PinnedFloatBufferPool.PinnedFloatBufferLease[Geometry.NumHiddenLayers];
        var acquired = new List<PinnedFloatBufferPool.PinnedFloatBufferLease>(
            Geometry.NumHiddenLayers * 2);

        try
        {
            for (var layer = 0; layer < Geometry.NumHiddenLayers; layer++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var source = state.GetLayer(layer);
                var sourceKey = source.Key.GetTensorDataAsSpan<float>();
                var sourceValue = source.Value.GetTensorDataAsSpan<float>();
                if (sourceKey.Length != expectedElementsPerTensor ||
                    sourceValue.Length != expectedElementsPerTensor)
                {
                    throw new InvalidOperationException(
                        $"Decoder layer {layer} KV payload does not match position {state.Position} and configured geometry.");
                }

                var keyBuffer = _stagingPool.Rent(expectedElementsPerTensor);
                acquired.Add(keyBuffer);
                sourceKey.CopyTo(keyBuffer.Span);
                keys[layer] = keyBuffer;

                var valueBuffer = _stagingPool.Rent(expectedElementsPerTensor);
                acquired.Add(valueBuffer);
                sourceValue.CopyTo(valueBuffer.Span);
                values[layer] = valueBuffer;
            }

            var payload = new OptimumLegacyFloatHostStagingPayload(
                HostStagingFormatId,
                state.Position,
                state.NextTokenId,
                EstimateHostStagingBytes(state),
                keys,
                values);
            acquired.Clear();
            return payload;
        }
        catch
        {
            for (var index = acquired.Count - 1; index >= 0; index--)
            {
                acquired[index].Dispose();
            }

            throw;
        }
    }

    public DecoderOrtState ImportHostStagingState(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        var staged = payload as OptimumLegacyFloatHostStagingPayload ??
            throw new InvalidOperationException(
                $"Unsupported Optimum host-staging payload type {payload.GetType().Name}.");
        if (!StringComparer.Ordinal.Equals(staged.FormatId, HostStagingFormatId))
        {
            throw new InvalidOperationException(
                $"Optimum host-staging payload format '{staged.FormatId}' does not match '{HostStagingFormatId}'.");
        }

        var expectedBytes = checked(
            Geometry.GetKvBytesPerSequence(staged.Position) + MetadataBytes);
        if (staged.ByteLength != expectedBytes)
        {
            throw new InvalidOperationException(
                $"Optimum host-staging payload reports {staged.ByteLength} byte(s); expected {expectedBytes}.");
        }

        if (staged.Keys.Count != Geometry.NumHiddenLayers ||
            staged.Values.Count != Geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                "Optimum host-staging payload layer count does not match decoder geometry.");
        }

        var shape = Geometry.GetPastKvShape(batchSize: 1, staged.Position);
        var expectedElementsPerTensor = checked(
            Geometry.NumKvHeads * staged.Position * Geometry.HeadDim);
        var layers = new DecoderOrtLayerState[Geometry.NumHiddenLayers];
        var owned = new List<OrtValue>(Geometry.NumHiddenLayers * 2);
        IDisposable? lifetimeLease = null;

        try
        {
            lifetimeLease = staged.Retain();

            for (var layer = 0; layer < Geometry.NumHiddenLayers; layer++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var keyBuffer = staged.Keys[layer].Buffer;
                var valueBuffer = staged.Values[layer].Buffer;
                if (keyBuffer.Length != expectedElementsPerTensor ||
                    valueBuffer.Length != expectedElementsPerTensor)
                {
                    throw new InvalidOperationException(
                        $"Optimum host-staging layer {layer} payload length does not match decoder geometry.");
                }

                var key = OrtValue.CreateTensorValueFromMemory(keyBuffer, shape);
                owned.Add(key);
                var value = OrtValue.CreateTensorValueFromMemory(valueBuffer, shape);
                owned.Add(value);
                layers[layer] = new DecoderOrtLayerState(key, value);
            }

            var state = new DecoderOrtState(
                staged.Position,
                layers,
                staged.NextTokenId,
                lifetimeAnchor: lifetimeLease);
            lifetimeLease = null;
            owned.Clear();
            return state;
        }
        catch
        {
            for (var index = owned.Count - 1; index >= 0; index--)
            {
                owned[index].Dispose();
            }

            lifetimeLease?.Dispose();
            throw;
        }
    }

    private void ValidateState(DecoderOrtState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.IsDisposed)
        {
            throw new ObjectDisposedException(nameof(state));
        }

        if (state.LayerCount != Geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                $"Decoder state contains {state.LayerCount} layer(s); geometry requires {Geometry.NumHiddenLayers}.");
        }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _stagingPool.Dispose();
        _inner.Dispose();
    }

    private sealed class OptimumLegacyFloatHostStagingPayload :
        DecoderOrtHostStagingPayload
    {
        public OptimumLegacyFloatHostStagingPayload(
            string formatId,
            int position,
            int? nextTokenId,
            long byteLength,
            IReadOnlyList<PinnedFloatBufferPool.PinnedFloatBufferLease> keys,
            IReadOnlyList<PinnedFloatBufferPool.PinnedFloatBufferLease> values)
            : base(formatId, position, nextTokenId, byteLength)
        {
            ArgumentNullException.ThrowIfNull(keys);
            ArgumentNullException.ThrowIfNull(values);
            Keys = keys.ToArray();
            Values = values.ToArray();
        }

        public IReadOnlyList<PinnedFloatBufferPool.PinnedFloatBufferLease> Keys { get; }
        public IReadOnlyList<PinnedFloatBufferPool.PinnedFloatBufferLease> Values { get; }

        protected override void DisposeCore()
        {
            for (var index = Values.Count - 1; index >= 0; index--)
            {
                Values[index].Dispose();
            }

            for (var index = Keys.Count - 1; index >= 0; index--)
            {
                Keys[index].Dispose();
            }
        }
    }
}
