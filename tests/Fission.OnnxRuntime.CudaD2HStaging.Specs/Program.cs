using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require(
    [DoesNotReturnIf(false)] bool condition,
    string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

await RunAsyncExportAndPoolReuseAsync();
await RunCancellationKeepsNativeLifetimesAsync();
await RunNonPageLockedDestinationRejectedAsync();

Console.WriteLine("Fission ONNX Runtime CUDA D2H staging specs passed.");

static async Task RunAsyncExportAndPoolReuseAsync()
{
    var cuda = new FakeCudaAsyncCopyApi();
    await using var copyEngine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 2,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    var allocator = new FakeCudaPageLockedAllocator();
    using var exporter = new CudaDeviceToHostStagingExporter(
        copyEngine,
        allocator,
        new PinnedHostStagingPoolOptions
        {
            MaxRetainedBuffersPerLength = 4,
            MaxRetainedBytes = 1024,
            ClearOnReturn = true
        });
    using var source = SyntheticCudaState.Create(
        new[] { 1f, 2f, 3f, 4f },
        new[] { 5f, 6f, 7f, 8f },
        position: 2,
        nextTokenId: 19);
    var lifetime = new LifetimeProbe();
    using var binding = new FakeCudaResidentBinding(source, lifetime);

    var firstExport = exporter.ExportAsync(binding, source.State).AsTask();
    await WaitUntilAsync(
        () => cuda.EventRecordCalls == 2,
        "D2H export must submit key and value copies before completion is released.");

    Require(!firstExport.IsCompleted, "Host-staging payload must not publish before CUDA completion events.");
    Require(lifetime.ActiveLeases == 1, "Source CUDA allocation retain must stay alive while D2H DMA is in flight.");
    Require(cuda.MemcpyCalls.Count == 2, "One decoder layer must submit exactly two CUDA copies.");
    Require(cuda.MemcpyCalls.All(static copy => copy.Kind == CudaMemcpyKind.DeviceToHost), "Every staging copy must use cudaMemcpyDeviceToHost.");
    Require(exporter.HostStagingPoolStatistics.ReturnedBuffers == 0, "In-flight destination buffers must not return to the pool.");

    cuda.CompleteAllRecordedEvents();
    var firstPayload = await firstExport;
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Source CUDA retain must release only after all D2H copies complete.");
    Require(firstPayload.Position == 2 && firstPayload.NextTokenId == 19, "D2H staging must preserve the decoder causal frontier.");
    Require(firstPayload.SourceCudaFormatId == binding.CudaResidentStateFormatId, "D2H payload must attest its source CUDA layout format.");
    Require(firstPayload.FormatId == CudaDeviceToHostStagingExporter.BuildHostStagingFormatId(binding.CudaResidentStateFormatId), "D2H payload format must derive deterministically from the CUDA layout format.");
    Require(firstPayload.ByteLength == 40, "D2H payload accounting must include two 16-byte tensors plus causal metadata.");
    Require(firstPayload.GetKeyMemory(0).Span.SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "D2H key payload must preserve source bytes.");
    Require(firstPayload.GetValueMemory(0).Span.SequenceEqual(new[] { 5f, 6f, 7f, 8f }), "D2H value payload must preserve source bytes.");
    Require(firstPayload.GetKeyShape(0).SequenceEqual(source.Shape), "D2H key shape must preserve source tensor geometry.");
    Require(firstPayload.GetValueShape(0).SequenceEqual(source.Shape), "D2H value shape must preserve source tensor geometry.");

    firstPayload.Dispose();
    var returned = exporter.HostStagingPoolStatistics;
    Require(returned.ReturnedBuffers == 2 && returned.RetainedBuffers == 2, "Payload disposal must return both page-locked buffers to the staging pool.");

    var secondExport = exporter.ExportAsync(binding, source.State).AsTask();
    await WaitUntilAsync(
        () => cuda.EventRecordCalls == 4,
        "Second D2H export must submit another key/value pair.");
    cuda.CompleteAllRecordedEvents();
    var secondPayload = await secondExport;
    var reused = exporter.HostStagingPoolStatistics;
    Require(reused.AllocatedBuffers == 2 && reused.ReusedBuffers == 2, "Repeated same-shape D2H export must reuse page-locked staging allocations.");
    secondPayload.Dispose();
}

static async Task RunCancellationKeepsNativeLifetimesAsync()
{
    var cuda = new FakeCudaAsyncCopyApi();
    await using var copyEngine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 2,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    var allocator = new FakeCudaPageLockedAllocator();
    using var exporter = new CudaDeviceToHostStagingExporter(copyEngine, allocator);
    using var source = SyntheticCudaState.Create(
        new[] { 11f, 12f, 13f, 14f },
        new[] { 21f, 22f, 23f, 24f },
        position: 2,
        nextTokenId: 23);
    var lifetime = new LifetimeProbe();
    using var binding = new FakeCudaResidentBinding(source, lifetime);
    using var cancellation = new CancellationTokenSource();

    var export = exporter.ExportAsync(
        binding,
        source.State,
        cancellation.Token).AsTask();
    await WaitUntilAsync(
        () => cuda.EventRecordCalls == 2,
        "Cancellation test requires both D2H copies to reach native submission.");

    cancellation.Cancel();
    await Task.Delay(20);
    Require(!export.IsCompleted, "Cancellation after cudaMemcpyAsync submission must wait for native completion.");
    Require(lifetime.ActiveLeases == 1, "Canceled D2H export must retain source CUDA allocations while native work is incomplete.");
    Require(exporter.HostStagingPoolStatistics.ReturnedBuffers == 0, "Canceled in-flight D2H destinations must not return to the pool early.");

    cuda.CompleteAllRecordedEvents();
    var canceled = false;
    try
    {
        _ = await export;
    }
    catch (OperationCanceledException)
    {
        canceled = true;
    }

    Require(canceled, "Caller cancellation must surface after submitted D2H copies complete.");
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Canceled D2H export must release source CUDA retain after native completion.");
    var statistics = exporter.HostStagingPoolStatistics;
    Require(statistics.ReturnedBuffers == 2 && statistics.RetainedBuffers == 2, "Canceled D2H export must reclaim both page-locked destinations after DMA drains.");
}

static async Task RunNonPageLockedDestinationRejectedAsync()
{
    var cuda = new FakeCudaAsyncCopyApi();
    await using var copyEngine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 1,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    using var exporter = new CudaDeviceToHostStagingExporter(
        copyEngine,
        new FakeStableButNotCudaPageLockedAllocator());
    using var source = SyntheticCudaState.Create(
        new[] { 31f, 32f, 33f, 34f },
        new[] { 41f, 42f, 43f, 44f },
        position: 2,
        nextTokenId: 29);
    var lifetime = new LifetimeProbe();
    using var binding = new FakeCudaResidentBinding(source, lifetime);

    Exception? observed = null;
    try
    {
        _ = await exporter.ExportAsync(binding, source.State);
    }
    catch (Exception exception)
    {
        observed = exception;
    }

    Require(observed is InvalidOperationException, "D2H exporter must reject a stable host pointer that is not explicitly CUDA page-locked.");
    Require(cuda.MemcpyAsyncCalls == 0, "No CUDA DMA may be submitted when the destination lacks the page-locked capability.");
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Pre-submission destination validation failure must release the source CUDA retain.");
}

static async Task WaitUntilAsync(Func<bool> predicate, string failureMessage)
{
    var timeout = Stopwatch.StartNew();
    while (!predicate())
    {
        if (timeout.Elapsed >= TimeSpan.FromSeconds(3))
        {
            throw new InvalidOperationException(failureMessage);
        }

        await Task.Delay(1);
    }
}

sealed class FakeCudaResidentBinding : IDecoderOrtCudaResidentStateBinding
{
    private readonly SyntheticCudaState _source;
    private readonly LifetimeProbe _lifetime;

    public FakeCudaResidentBinding(
        SyntheticCudaState source,
        LifetimeProbe lifetime)
    {
        _source = source;
        _lifetime = lifetime;
    }

    public string Name => "fake-cuda-resident";
    public OnnxSessionContract SessionContract => throw new NotSupportedException();
    public string CudaResidentStateFormatId => "fake-cuda-fp32-kv-v1";

    public DecoderOrtCudaResidentStateLease AcquireCudaResidentState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!ReferenceEquals(state, _source.State))
        {
            throw new InvalidOperationException("Unexpected decoder state instance.");
        }

        return DecoderOrtCudaResidentStateLease.Create(
            CudaResidentStateFormatId,
            state,
            deviceId: 0,
            _source.CreateLayerViews(),
            _lifetime.Retain());
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public void Dispose()
    {
    }
}

sealed class SyntheticCudaState : IDisposable
{
    private readonly OrtMemoryInfo _memoryInfo;
    private int _disposed;

    private SyntheticCudaState(
        OrtMemoryInfo memoryInfo,
        nint keyPointer,
        nint valuePointer,
        long byteLength,
        long[] shape,
        DecoderOrtState state)
    {
        _memoryInfo = memoryInfo;
        KeyPointer = keyPointer;
        ValuePointer = valuePointer;
        ByteLength = byteLength;
        Shape = shape;
        State = state;
    }

    public nint KeyPointer { get; }
    public nint ValuePointer { get; }
    public long ByteLength { get; }
    public long[] Shape { get; }
    public DecoderOrtState State { get; }

    public static SyntheticCudaState Create(
        float[] keyValues,
        float[] valueValues,
        int position,
        int? nextTokenId)
    {
        ArgumentNullException.ThrowIfNull(keyValues);
        ArgumentNullException.ThrowIfNull(valueValues);
        if (keyValues.Length != valueValues.Length || keyValues.Length == 0)
        {
            throw new ArgumentException("Synthetic key/value buffers must have the same non-zero length.");
        }

        var shape = new long[] { 1, 1, 1, keyValues.Length };
        var byteLength = checked(keyValues.Length * (long)sizeof(float));
        var keyPointer = Marshal.AllocHGlobal(checked((int)byteLength));
        var valuePointer = Marshal.AllocHGlobal(checked((int)byteLength));
        Marshal.Copy(keyValues, 0, keyPointer, keyValues.Length);
        Marshal.Copy(valueValues, 0, valuePointer, valueValues.Length);
        var memoryInfo = new OrtMemoryInfo(
            "Cuda",
            OrtAllocatorType.DeviceAllocator,
            0,
            OrtMemType.Default);
        OrtValue? key = null;
        OrtValue? value = null;
        try
        {
            key = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                shape,
                keyPointer,
                byteLength);
            value = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                shape,
                valuePointer,
                byteLength);
            var state = new DecoderOrtState(
                position,
                new[] { new DecoderOrtLayerState(key, value) },
                nextTokenId);
            key = null;
            value = null;
            return new SyntheticCudaState(
                memoryInfo,
                keyPointer,
                valuePointer,
                byteLength,
                shape,
                state);
        }
        catch
        {
            value?.Dispose();
            key?.Dispose();
            memoryInfo.Dispose();
            Marshal.FreeHGlobal(valuePointer);
            Marshal.FreeHGlobal(keyPointer);
            throw;
        }
    }

    public DecoderOrtCudaLayerView[] CreateLayerViews() =>
        new[]
        {
            new DecoderOrtCudaLayerView(
                new CudaDeviceTensorView(
                    KeyPointer,
                    ByteLength,
                    TensorElementType.Float,
                    Shape,
                    0),
                new CudaDeviceTensorView(
                    ValuePointer,
                    ByteLength,
                    TensorElementType.Float,
                    Shape,
                    0))
        };

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        State.Dispose();
        _memoryInfo.Dispose();
        Marshal.FreeHGlobal(ValuePointer);
        Marshal.FreeHGlobal(KeyPointer);
    }
}

sealed class LifetimeProbe
{
    private int _activeLeases;
    private int _releaseCount;

    public int ActiveLeases => Volatile.Read(ref _activeLeases);
    public int ReleaseCount => Volatile.Read(ref _releaseCount);

    public IDisposable Retain()
    {
        Interlocked.Increment(ref _activeLeases);
        return new Lease(this);
    }

    private void Release()
    {
        Interlocked.Decrement(ref _activeLeases);
        Interlocked.Increment(ref _releaseCount);
    }

    private sealed class Lease : IDisposable
    {
        private LifetimeProbe? _owner;

        public Lease(LifetimeProbe owner)
        {
            _owner = owner;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?.Release();
        }
    }
}

sealed class FakeCudaPageLockedAllocator : IHostStagingFloatBufferAllocator
{
    public IHostStagingFloatBuffer Allocate(int length) =>
        new Buffer(length);

    private sealed class Buffer : ICudaPageLockedHostStagingFloatBuffer
    {
        private readonly float[] _buffer;
        private GCHandle _handle;
        private int _disposed;

        public Buffer(int length)
        {
            _buffer = new float[length];
            _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
        }

        public Memory<float> Memory
        {
            get
            {
                ThrowIfDisposed();
                return _buffer;
            }
        }

        public nint Pointer
        {
            get
            {
                ThrowIfDisposed();
                return _handle.AddrOfPinnedObject();
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            if (_handle.IsAllocated)
            {
                _handle.Free();
            }
        }

        private void ThrowIfDisposed() =>
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}

sealed class FakeStableButNotCudaPageLockedAllocator : IHostStagingFloatBufferAllocator
{
    public IHostStagingFloatBuffer Allocate(int length) =>
        new Buffer(length);

    private sealed class Buffer : IHostStagingFloatBuffer
    {
        private readonly float[] _buffer;
        private GCHandle _handle;
        private int _disposed;

        public Buffer(int length)
        {
            _buffer = new float[length];
            _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
        }

        public Memory<float> Memory
        {
            get
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                return _buffer;
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && _handle.IsAllocated)
            {
                _handle.Free();
            }
        }
    }
}

sealed class FakeCudaAsyncCopyApi : ICudaAsyncCopyApi
{
    private readonly object _gate = new();
    private readonly HashSet<nint> _streams = new();
    private readonly Dictionary<nint, EventState> _events = new();
    private readonly List<MemcpyCall> _memcpyCalls = new();
    private long _nextHandle = 100;
    private int _memcpyAsyncCalls;
    private int _eventRecordCalls;

    public int MemcpyAsyncCalls => Volatile.Read(ref _memcpyAsyncCalls);
    public int EventRecordCalls => Volatile.Read(ref _eventRecordCalls);

    public IReadOnlyList<MemcpyCall> MemcpyCalls
    {
        get
        {
            lock (_gate)
            {
                return _memcpyCalls.ToArray();
            }
        }
    }

    public int StreamCreateWithFlags(out nint stream, uint flags)
    {
        lock (_gate)
        {
            stream = (nint)++_nextHandle;
            _streams.Add(stream);
        }

        return 0;
    }

    public int StreamDestroy(nint stream)
    {
        lock (_gate)
        {
            return _streams.Remove(stream) ? 0 : 17;
        }
    }

    public int StreamSynchronize(nint stream)
    {
        lock (_gate)
        {
            foreach (var state in _events.Values.Where(item => item.Stream == stream))
            {
                state.Completed = true;
            }
        }

        return 0;
    }

    public int MemcpyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream)
    {
        Interlocked.Increment(ref _memcpyAsyncCalls);
        lock (_gate)
        {
            if (!_streams.Contains(stream))
            {
                return 17;
            }

            _memcpyCalls.Add(new MemcpyCall(destination, source, byteLength, kind, stream));
        }

        var count = checked((int)byteLength);
        var bytes = new byte[count];
        Marshal.Copy(source, bytes, 0, count);
        Marshal.Copy(bytes, 0, destination, count);
        return 0;
    }

    public int EventCreateWithFlags(out nint completionEvent, uint flags)
    {
        lock (_gate)
        {
            completionEvent = (nint)++_nextHandle;
            _events.Add(completionEvent, new EventState());
        }

        return 0;
    }

    public int EventRecord(nint completionEvent, nint stream)
    {
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state) ||
                !_streams.Contains(stream))
            {
                return 17;
            }

            state.Stream = stream;
            state.Recorded = true;
        }

        Interlocked.Increment(ref _eventRecordCalls);
        return 0;
    }

    public int EventQuery(nint completionEvent)
    {
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state) || !state.Recorded)
            {
                return 17;
            }

            return state.Completed
                ? 0
                : CudaAsyncCopyEngine.CudaErrorNotReady;
        }
    }

    public int EventDestroy(nint completionEvent)
    {
        lock (_gate)
        {
            return _events.Remove(completionEvent) ? 0 : 17;
        }
    }

    public string? GetErrorString(int errorCode) => $"fake-cuda-error-{errorCode}";

    public void CompleteAllRecordedEvents()
    {
        lock (_gate)
        {
            foreach (var state in _events.Values.Where(static state => state.Recorded))
            {
                state.Completed = true;
            }
        }
    }

    public readonly record struct MemcpyCall(
        nint Destination,
        nint Source,
        nuint ByteLength,
        CudaMemcpyKind Kind,
        nint Stream);

    private sealed class EventState
    {
        public nint Stream { get; set; }
        public bool Recorded { get; set; }
        public bool Completed { get; set; }
    }
}
