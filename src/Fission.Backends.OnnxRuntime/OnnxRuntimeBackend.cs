using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

public sealed record OnnxRuntimeBackendOptions(
    ModelId ModelId,
    DeviceId Device,
    OnnxRuntimeModelSource Model,
    int? IntraOpNumThreads = null,
    int? InterOpNumThreads = null,
    GraphOptimizationLevel GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
    OnnxSessionContract? SessionContract = null,
    IInferenceDeviceMemoryPressureSource? DeviceMemoryPressureSource = null,
    IInferenceDeviceMemoryReclaimer? DeviceMemoryReclaimer = null);

/// <summary>
/// ONNX Runtime session host. Model-specific tensor names, shapes, KV schemas,
/// sampling, batching, and state transactions live in
/// IOnnxRuntimeExecutionAdapter rather than in the generic runtime backend.
/// A configured session contract, or a contract supplied by the adapter, validates
/// the live graph signature before the adapter initializes model-owned state.
/// </summary>
public sealed class OnnxRuntimeBackend :
    IInferenceBackend,
    IInferenceDeviceMemoryPressureSource,
    IInferenceDeviceMemoryReclaimer
{
    private readonly OnnxRuntimeBackendOptions _options;
    private readonly IOnnxRuntimeExecutionAdapter _adapter;
    private readonly Func<SessionOptions>? _sessionOptionsFactory;
    private readonly IDisposable? _ownedResource;
    private readonly IInferenceDeviceMemoryPressureSource? _deviceMemoryPressureSource;
    private readonly IInferenceDeviceMemoryReclaimer? _deviceMemoryReclaimer;
    private InferenceSession? _session;
    private int _disposed;

    public OnnxRuntimeBackend(
        OnnxRuntimeBackendOptions options,
        IOnnxRuntimeExecutionAdapter adapter,
        Func<SessionOptions>? sessionOptionsFactory = null,
        IDisposable? ownedResource = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.Model);
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
        _ownedResource = ownedResource;
        _deviceMemoryPressureSource =
            options.DeviceMemoryPressureSource ??
            adapter as IInferenceDeviceMemoryPressureSource;
        _deviceMemoryReclaimer =
            options.DeviceMemoryReclaimer ??
            options.DeviceMemoryPressureSource as IInferenceDeviceMemoryReclaimer ??
            adapter as IInferenceDeviceMemoryReclaimer;
    }

    public string Name => $"onnxruntime/{_adapter.Name}";
    public DeviceId Device => _options.Device;
    public ModelId ModelId => _options.ModelId;
    public bool IsInitialized => _session is not null;

    public bool TryGetDeviceMemoryPressure(
        out InferenceDeviceMemoryPressure pressure)
    {
        if (Volatile.Read(ref _disposed) != 0 ||
            _deviceMemoryPressureSource is null)
        {
            pressure = default;
            return false;
        }

        return _deviceMemoryPressureSource.TryGetDeviceMemoryPressure(out pressure);
    }

    public ValueTask<InferenceDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
        long targetReclaimableBytes,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetReclaimableBytes);
        _ = GetSession();
        return _deviceMemoryReclaimer?.ReclaimDeviceMemoryAsync(
                targetReclaimableBytes,
                cancellationToken)
            ?? ValueTask.FromException<InferenceDeviceMemoryReclaimResult>(
                new NotSupportedException(
                    $"ONNX Runtime backend '{Name}' does not expose device-memory reclaim."));
    }

    public async ValueTask InitializeAsync(
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        if (_session is not null)
        {
            throw new InvalidOperationException("ONNX Runtime backend is already initialized.");
        }

        using var sessionOptions = _sessionOptionsFactory?.Invoke() ?? new SessionOptions();
        sessionOptions.GraphOptimizationLevel = _options.GraphOptimizationLevel;

        if (_options.IntraOpNumThreads is { } intraOpNumThreads)
        {
            sessionOptions.IntraOpNumThreads = intraOpNumThreads;
        }

        if (_options.InterOpNumThreads is { } interOpNumThreads)
        {
            sessionOptions.InterOpNumThreads = interOpNumThreads;
        }

        var session = _options.Model.CreateSession(sessionOptions);
        try
        {
            var sessionContract = _options.SessionContract ??
                (_adapter as IOnnxRuntimeSessionContractProvider)?.SessionContract;
            sessionContract?.Validate(session);

            await _adapter.InitializeAsync(session, cancellationToken)
                .ConfigureAwait(false);
            _session = session;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        var session = GetSession();
        ValidateModels(batch.Items);
        return _adapter.PrefillAsync(session, batch, cancellationToken);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        var session = GetSession();
        ValidateModels(batch.Items);
        return _adapter.DecodeAsync(session, batch, cancellationToken);
    }

    public ValueTask SnapshotSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        _ = GetSession();
        return _adapter.SnapshotSequenceAsync(sequenceId, snapshotId, cancellationToken);
    }

    public ValueTask ForkSequenceAsync(
        SequenceId parentSequenceId,
        IReadOnlyList<SequenceId> branchSequenceIds,
        CancellationToken cancellationToken = default)
    {
        _ = GetSession();
        return _adapter.ForkSequenceAsync(parentSequenceId, branchSequenceIds, cancellationToken);
    }

    public ValueTask RestoreSequenceAsync(
        SequenceId sequenceId,
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        _ = GetSession();
        return _adapter.RestoreSequenceAsync(sequenceId, snapshotId, cancellationToken);
    }

    public ValueTask MigrateSequenceAsync(
        SequenceId sequenceId,
        DeviceId targetDevice,
        CancellationToken cancellationToken = default)
    {
        _ = GetSession();
        return _adapter.MigrateSequenceAsync(sequenceId, targetDevice, cancellationToken);
    }

    public ValueTask ReleaseSnapshotAsync(
        KvSnapshotId snapshotId,
        CancellationToken cancellationToken = default)
    {
        _ = GetSession();
        return _adapter.ReleaseSnapshotAsync(snapshotId, cancellationToken);
    }

    public ValueTask ReleaseSequenceAsync(
        SequenceId sequenceId,
        CancellationToken cancellationToken = default)
    {
        _ = GetSession();
        return _adapter.ReleaseSequenceAsync(sequenceId, cancellationToken);
    }

    private void ValidateModels(IReadOnlyList<PrefillItem> items)
    {
        for (var index = 0; index < items.Count; index++)
        {
            ValidateModel(items[index].ModelId);
        }
    }

    private void ValidateModels(IReadOnlyList<DecodeItem> items)
    {
        for (var index = 0; index < items.Count; index++)
        {
            ValidateModel(items[index].ModelId);
        }
    }

    private void ValidateModel(ModelId modelId)
    {
        if (modelId != _options.ModelId)
        {
            throw new InvalidOperationException(
                $"ONNX Runtime backend is bound to model {_options.ModelId}, not {modelId}.");
        }
    }

    private InferenceSession GetSession()
    {
        ThrowIfDisposed();
        return _session ?? throw new InvalidOperationException(
            "ONNX Runtime backend has not been initialized.");
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return ValueTask.CompletedTask;
        }

        List<Exception>? failures = null;
        TryDispose(_adapter, ref failures);
        TryDispose(Interlocked.Exchange(ref _session, null), ref failures);
        TryDispose(_ownedResource, ref failures);

        if (failures is { Count: 1 })
        {
            throw failures[0];
        }

        if (failures is { Count: > 1 })
        {
            throw new AggregateException(
                "One or more ONNX Runtime backend resources failed to dispose.",
                failures);
        }

        return ValueTask.CompletedTask;
    }

    private static void TryDispose(
        IDisposable? resource,
        ref List<Exception>? failures)
    {
        if (resource is null)
        {
            return;
        }

        try
        {
            resource.Dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }
}
