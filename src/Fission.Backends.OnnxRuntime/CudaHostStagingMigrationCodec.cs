namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Standalone CUDA host-staging migration codec that composes a CUDA-resident
/// decoder binding with the reusable asynchronous D2H exporter and H2D importer.
///
/// The codec does not own or dispose any supplied collaborator. In particular,
/// the execution binding, exporter, importer, copy engines, allocators and staging
/// pools remain caller-owned and must outlive every migration that can use them.
/// </summary>
public sealed class CudaHostStagingMigrationCodec :
    IDecoderOrtAsyncHostStagingCodec,
    IDecoderOrtAsyncHostStagingImportCodec
{
    private readonly IDecoderOrtCudaResidentStateBinding _residentBinding;
    private readonly CudaDeviceToHostStagingExporter _exporter;
    private readonly CudaHostToDeviceStateImporter _importer;
    private readonly string _cudaFormatId;

    public CudaHostStagingMigrationCodec(
        IDecoderOrtCudaResidentStateBinding residentBinding,
        CudaDeviceToHostStagingExporter exporter,
        CudaHostToDeviceStateImporter importer)
    {
        ArgumentNullException.ThrowIfNull(residentBinding);
        ArgumentNullException.ThrowIfNull(exporter);
        ArgumentNullException.ThrowIfNull(importer);
        if (string.IsNullOrWhiteSpace(residentBinding.CudaResidentStateFormatId))
        {
            throw new ArgumentException(
                "CUDA-resident binding format id cannot be empty.",
                nameof(residentBinding));
        }

        _residentBinding = residentBinding;
        _exporter = exporter;
        _importer = importer;
        _cudaFormatId = residentBinding.CudaResidentStateFormatId;
        HostStagingFormatId =
            CudaDeviceToHostStagingExporter.BuildHostStagingFormatId(
                _cudaFormatId);
    }

    public string Name => $"cuda-host-staging/{_residentBinding.Name}";
    public string HostStagingFormatId { get; }
    public int TargetDeviceId => _importer.DeviceId;
    public string CudaResidentStateFormatId => _cudaFormatId;

    public long EstimateHostStagingBytes(DecoderOrtState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(state.IsDisposed, state);

        using var residentState = _residentBinding.AcquireCudaResidentState(state);
        ArgumentNullException.ThrowIfNull(residentState);
        ValidateResidentFormat(residentState);
        return CudaDeviceToHostStagingExporter.EstimateHostStagingBytes(
            residentState);
    }

    public DecoderOrtHostStagingPayload ExportHostStagingState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException(
            "CUDA host-staging export is asynchronous. Use ExportHostStagingStateAsync.");

    public DecoderOrtState ImportHostStagingState(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException(
            "CUDA host-staging import is asynchronous. Use ImportHostStagingStateAsync.");

    public async ValueTask<DecoderOrtHostStagingPayload> ExportHostStagingStateAsync(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        cancellationToken.ThrowIfCancellationRequested();

        var payload = await _exporter.ExportAsync(
            _residentBinding,
            state,
            cancellationToken).ConfigureAwait(false);
        if (!StringComparer.Ordinal.Equals(payload.SourceCudaFormatId, _cudaFormatId) ||
            !StringComparer.Ordinal.Equals(payload.FormatId, HostStagingFormatId))
        {
            payload.Dispose();
            throw new InvalidOperationException(
                $"CUDA exporter returned source/host format '{payload.SourceCudaFormatId}'/'{payload.FormatId}', " +
                $"but codec expects '{_cudaFormatId}'/'{HostStagingFormatId}'.");
        }

        return payload;
    }

    public async ValueTask<DecoderOrtState> ImportHostStagingStateAsync(
        DecoderOrtHostStagingPayload payload,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(payload);
        cancellationToken.ThrowIfCancellationRequested();
        var cudaPayload = payload as DecoderOrtCudaHostStagingPayload ??
            throw new InvalidOperationException(
                $"CUDA host-staging codec cannot import payload type {payload.GetType().Name}.");

        return await _importer.ImportAsync(
            cudaPayload,
            _cudaFormatId,
            cancellationToken).ConfigureAwait(false);
    }

    private void ValidateResidentFormat(
        DecoderOrtCudaResidentStateLease residentState)
    {
        if (!StringComparer.Ordinal.Equals(
                residentState.FormatId,
                _cudaFormatId))
        {
            throw new InvalidOperationException(
                $"CUDA-resident binding '{_residentBinding.Name}' acquired format " +
                $"'{residentState.FormatId}', but codec expects '{_cudaFormatId}'.");
        }
    }
}
