using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

public sealed record DualSessionOnnxRuntimeBackendOptions(
    ModelId ModelId,
    DeviceId Device,
    OnnxRuntimeModelSource PrefillModel,
    OnnxRuntimeModelSource DecodeModel,
    int? IntraOpNumThreads = null,
    int? InterOpNumThreads = null,
    GraphOptimizationLevel GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
    OnnxSessionContract? PrefillSessionContract = null,
    OnnxSessionContract? DecodeSessionContract = null);

/// <summary>
/// ONNX Runtime host for export families that use one graph for no-past prefill
/// and another graph for with-past decode.
///
/// Both sessions feed the same execution adapter, so sequence/snapshot state stays
/// in one transactional state graph while model execution is routed by phase.
/// </summary>
public sealed class DualSessionOnnxRuntimeBackend : IInferenceBackend
{
    private readonly DualSessionOnnxRuntimeBackendOptions _options;
    private readonly IOnnxRuntimeExecutionAdapter _adapter;
    private readonly Func<SessionOptions>? _sessionOptionsFactory;
    private InferenceSession? _prefillSession;
    private InferenceSession? _decodeSession;
    private int _disposed;

    public DualSessionOnnxRuntimeBackend(
        DualSessionOnnxRuntimeBackendOptions options,
        IOnnxRuntimeExecutionAdapter adapter,
        Func<SessionOptions>? sessionOptionsFactory = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.PrefillModel);
        ArgumentNullException.ThrowIfNull(options.DecodeModel);
        ArgumentNullException.ThrowIfNull(adapter);

        if (options.IntraOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "IntraOpNumThreads must be positive when specified.");
        }

        if (options.InterOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "InterOpNumThreads must be positive when specified.");
        }

        _options = options;
        _adapter = adapter;
        _sessionOptionsFactory = sessionOptionsFactory;
    }

    public string Name => $"onnxruntime-dual/{_adapter.Name}";
    public DeviceId Device => _options.Device;
    public ModelId ModelId => _options.ModelId;
    public bool IsInitialized =>
        _prefillSession is not null && _decodeSession is not null;

    public async ValueTask InitializeAsync(
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        if (_prefillSession is not null || _decodeSession is not null)
        {
            throw new InvalidOperationException(
                "Dual-session ONNX Runtime backend is already initialized.");
        }

        var prefillSession = CreateSession(_options.PrefillModel);
        InferenceSession? decodeSession = null;
        try
        {
            decodeSession = CreateSession(_options.DecodeModel);

            var dualContracts = _adapter as IOnnxRuntimeDualSessionContractProvider;
            var singleContract = _adapter as IOnnxRuntimeSessionContractProvider;

            var prefillContract = _options.PrefillSessionContract ??
                dualContracts?.PrefillSessionContract ??
                singleContract?.SessionContract;
            var decodeContract = _options.DecodeSessionContract ??
                dualContracts?.DecodeSessionContract ??
                singleContract?.SessionContract;

            prefillContract?.Validate(prefillSession);
            decodeContract?.Validate(decodeSession);

            if (_adapter is IOnnxRuntimeDualSessionExecutionAdapter dualAdapter)
            {
                await dualAdapter.InitializeAsync(
                        prefillSession,
                        decodeSession,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
            else
            {
                // Existing single-session adapters are initialized against the
                // decode graph. Their PrefillAsync still receives the distinct
                // prefill session for actual execution.
                await _adapter.InitializeAsync(decodeSession, cancellationToken)
                    .ConfigureAwait(false);
            }

            _prefillSession = prefillSession;
            _decodeSession = decodeSession;
            decodeSession = null;
        }
        catch
        {
            decodeSession?.Dispose();
            prefillSession.Dispose();
            throw;
        }
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        var session = GetPrefillSession();
        ValidateModels(batch.Items.Select(static item => item.ModelId));
        return _adapter.PrefillAsync(session, batch, cancellationToken);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        var session = GetDecodeSession();
        ValidateModels(batch.Items.Select(static item => item.ModelId));
        return _adapter.DecodeAsync(session, batch, cancellationToken);
    }

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        _ = GetDecodeSession();
        return _adapter.SnapshotSequenceAsync(
            sequenceId,
            snapshotId,
            cancellationToken);
    }

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default)
    {
        _ = GetDecodeSession();
        return _adapter.ForkSequenceAsync(
            parentSequenceId,
            branchSequenceIds,
            cancellationToken);
    }

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        _ = GetDecodeSession();
        return _adapter.RestoreSequenceAsync(
            sequenceId,
            snapshotId,
            cancellationToken);
    }

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        _ = GetDecodeSession();
        return _adapter.ReleaseSnapshotAsync(snapshotId, cancellationToken);
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        _ = GetDecodeSession();
        return _adapter.ReleaseSequenceAsync(sequenceId, cancellationToken);
    }

    private InferenceSession CreateSession(OnnxRuntimeModelSource source)
    {
        using var sessionOptions =
            _sessionOptionsFactory?.Invoke() ?? new SessionOptions();
        sessionOptions.GraphOptimizationLevel = _options.GraphOptimizationLevel;

        if (_options.IntraOpNumThreads is { } intraOpNumThreads)
        {
            sessionOptions.IntraOpNumThreads = intraOpNumThreads;
        }

        if (_options.InterOpNumThreads is { } interOpNumThreads)
        {
            sessionOptions.InterOpNumThreads = interOpNumThreads;
        }

        return source.CreateSession(sessionOptions);
    }

    private void ValidateModels(IEnumerable<ModelId> modelIds)
    {
        foreach (var modelId in modelIds)
        {
            if (modelId != _options.ModelId)
            {
                throw new InvalidOperationException(
                    $"Dual-session ONNX Runtime backend is bound to model {_options.ModelId}, not {modelId}.");
            }
        }
    }

    private InferenceSession GetPrefillSession()
    {
        ThrowIfDisposed();
        return _prefillSession ?? throw new InvalidOperationException(
            "Dual-session ONNX Runtime backend has not been initialized.");
    }

    private InferenceSession GetDecodeSession()
    {
        ThrowIfDisposed();
        return _decodeSession ?? throw new InvalidOperationException(
            "Dual-session ONNX Runtime backend has not been initialized.");
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return ValueTask.CompletedTask;
        }

        _adapter.Dispose();
        Interlocked.Exchange(ref _decodeSession, null)?.Dispose();
        Interlocked.Exchange(ref _prefillSession, null)?.Dispose();
        return ValueTask.CompletedTask;
    }
}
