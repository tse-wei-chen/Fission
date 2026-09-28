using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require([DoesNotReturnIf(false)] bool condition, string message)
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
    await using var copyEngine = new CudaAsyncCopyEngine(cuda, new CudaAsyncCopyEngineOptions
    {
        StreamCount = 2,
        CompletionPollInterval = TimeSpan.FromMilliseconds(1)
    });
    using var exporter = new CudaDeviceToHostStagingExporter(
        copyEngine,
        new FakeCudaPageLockedAllocator(),
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
    await WaitUntilAsync(() => cuda.EventRecordCalls == 2, "D2H export did not submit both KV copies.");
    Require(!firstExport.IsCompleted, "Payload must not publish before CUDA completion.");
    Require(lifetime.ActiveLeases == 1, "Source CUDA retain must stay alive while DMA is in flight.");
    Require(cuda.MemcpyCalls.Count == 2, "One layer must submit exactly key/value copies.");
    Require(cuda.MemcpyCalls.All(static call => call.Kind == CudaMemcpyKind.DeviceToHost), "Every staging copy must be DeviceToHost.");
    Require(exporter.HostStagingPoolStatistics.ReturnedBuffers == 0, "In-flight destinations must not return to the pool.");

    cuda.CompleteAllRecordedEvents();
    var firstPayload = await firstExport;
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Source retain must release after D2H completion.");
    Require(firstPayload.Position == 2 && firstPayload.NextTokenId == 19, "D2H staging must preserve causal frontier.");
    Require(firstPayload.SourceCudaFormatId == binding.CudaResidentStateFormatId, "Payload must preserve source CUDA format id.");
    Require(firstPayload.FormatId == CudaDeviceToHostStagingExporter.BuildHostStagingFormatId(binding.CudaResidentStateFormatId), "Host format id must be deterministic.");
    Require(firstPayload.ByteLength == 40, "Payload bytes must include two 16-byte tensors plus metadata.");
    Require(firstPayload.GetKeyMemory(0).Span.SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "Key bytes must survive D2H staging.");
    Require(firstPayload.GetValueMemory(0).Span.SequenceEqual(new[] { 5f, 6f, 7f, 8f }), "Value bytes must survive D2H staging.");
    Require(firstPayload.GetKeyShape(0).SequenceEqual(source.Shape), "Key shape must survive staging.");
    firstPayload.Dispose();

    var returned = exporter.HostStagingPoolStatistics;
    Require(returned.ReturnedBuffers == 2 && returned.RetainedBuffers == 2, "Payload disposal must return both staging buffers.");

    var secondExport = exporter.ExportAsync(binding, source.State).AsTask();
    await WaitUntilAsync(() => cuda.EventRecordCalls == 4, "Second export did not submit both copies.");
    cuda.CompleteAllRecordedEvents();
    var secondPayload = await secondExport;
    var reused = exporter.HostStagingPoolStatistics;
    Require(reused.AllocatedBuffers == 2 && reused.ReusedBuffers == 2, "Same-shape export must reuse page-locked buffers.");
    secondPayload.Dispose();
}

static async Task RunCancellationKeepsNativeLifetimesAsync()
{
    var cuda = new FakeCudaAsyncCopyApi();
    await using var copyEngine = new CudaAsyncCopyEngine(cuda, new CudaAsyncCopyEngineOptions
    {
        StreamCount = 2,
        CompletionPollInterval = TimeSpan.FromMilliseconds(1)
    });
    using var exporter = new CudaDeviceToHostStagingExporter(copyEngine, new FakeCudaPageLockedAllocator());
    using var source = SyntheticCudaState.Create(
        new[] { 11f, 12f, 13f, 14f },
        new[] { 21f, 22f, 23f, 24f },
        position: 2,
        nextTokenId: 23);
    var lifetime = new LifetimeProbe();
    using var binding = new FakeCudaResidentBinding(source, lifetime);
    using var cancellation = new CancellationTokenSource();

    var export = exporter.ExportAsync(binding, source.State, cancellation.Token).AsTask();
    await WaitUntilAsync(() => cuda.EventRecordCalls == 2, "Cancellation test requires submitted DMA.");
    cancellation.Cancel();
    await Task.Delay(20);
    Require(!export.IsCompleted, "Post-submit cancellation must wait for native completion.");
    Require(lifetime.ActiveLeases == 1, "Canceled in-flight export must retain CUDA source allocations.");
    Require(exporter.HostStagingPoolStatistics.ReturnedBuffers == 0, "Canceled in-flight destinations must stay leased.");

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

    Require(canceled, "Cancellation must surface after native completion.");
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Canceled export must release source retain after drain.");
    var stats = exporter.HostStagingPoolStatistics;
    Require(stats.ReturnedBuffers == 2 && stats.RetainedBuffers == 2, "Canceled export must reclaim both destinations after drain.");
}

static async Task RunNonPageLockedDestinationRejectedAsync()
{
    var cuda = new FakeCudaAsyncCopyApi();
    await using var copyEngine = new CudaAsyncCopyEngine(cuda, new CudaAsyncCopyEngineOptions
    {
        StreamCount = 1,
        CompletionPollInterval = TimeSpan.FromMilliseconds(1)
    });
    using var exporter = new CudaDeviceToHostStagingExporter(copyEngine, new FakeStableHostAllocator());
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

    Require(observed is InvalidOperationException, "Stable but non-CUDA-page-locked destination must be rejected.");
    Require(cuda.MemcpyAsyncCalls == 0, "No CUDA DMA may be submitted to a non-page-locked destination.");
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Pre-submit validation failure must release source retain.");
}

static async Task WaitUntilAsync(Func<bool> predicate, string message)
{
    var timeout = Stopwatch.StartNew();
    while (!predicate())
    {
        if (timeout.Elapsed >= TimeSpan.FromSeconds(3))
        {
            throw new InvalidOperationException(message);
        }

        await Task.Delay(1);
    }
}

sealed class FakeCudaResidentBinding : IDecoderOrtCudaResidentStateBinding
{
    private readonly SyntheticCudaState _source;
    private readonly LifetimeProbe _lifetime;

    public FakeCudaResidentBinding(SyntheticCudaState source, LifetimeProbe lifetime)
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
            0,
            _source.CreateLayerViews(),
            _lifetime.Retain());
    }

    public DecoderOrtStepResult ExecutePrefill(InferenceSession session, PrefillItem item, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public DecoderOrtStepResult ExecuteDecode(InferenceSession session, DecodeItem item, DecoderOrtState priorState, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public void Dispose() { }
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

    public static SyntheticCudaState Create(float[] keys, float[] values, int position, int? nextTokenId)
    {
        if (keys.Length == 0 || keys.Length != values.Length)
        {
            throw new ArgumentException("Synthetic key/value buffers must have equal non-zero length.");
        }

        var shape = new long[] { 1, 1, 1, keys.Length };
        var bytes = checked(keys.Length * (long)sizeof(float));
        var keyPointer = Marshal.AllocHGlobal(checked((int)bytes));
        var valuePointer = Marshal.AllocHGlobal(checked((int)bytes));
        Marshal.Copy(keys, 0, keyPointer, keys.Length);
        Marshal.Copy(values, 0, valuePointer, values.Length);
        var memoryInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        OrtValue? key = null;
        OrtValue? value = null;
        try
        {
            key = OrtValue.CreateTensorValueWithData(memoryInfo, TensorElementType.Float, shape, keyPointer, bytes);
            value = OrtValue.CreateTensorValueWithData(memoryInfo, TensorElementType.Float, shape, valuePointer, bytes);
            var state = new DecoderOrtState(position, new[] { new DecoderOrtLayerState(key, value) }, nextTokenId);
            key = null;
            value = null;
            return new SyntheticCudaState(memoryInfo, keyPointer, valuePointer, bytes, shape, state);
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
    [
        new DecoderOrtCudaLayerView(
            new CudaDeviceTensorView(KeyPointer, ByteLength, TensorElementType.Float, Shape, 0),
            new CudaDeviceTensorView(ValuePointer, ByteLength, TensorElementType.Float, Shape, 0))
    ];

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
    private int _active;
    private int _released;
    public int ActiveLeases => Volatile.Read(ref _active);
    public int ReleaseCount => Volatile.Read(ref _released);

    public IDisposable Retain()
    {
        Interlocked.Increment(ref _active);
        return new Lease(this);
    }

    private void Release()
    {
        Interlocked.Decrement(ref _active);
        Interlocked.Increment(ref _released);
    }

    private sealed class Lease(LifetimeProbe owner) : IDisposable
    {
        private LifetimeProbe? _owner = owner;
        public void Dispose() => Interlocked.Exchange(ref _owner, null)?.Release();
    }
}

sealed class FakeCudaPageLockedAllocator : IHostStagingFloatBufferAllocator
{
    public IHostStagingFloatBuffer Allocate(int length) => new Buffer(length);

    private sealed class Buffer : ICudaPageLockedHostStagingFloatBuffer
    {
        private readonly float[] _buffer = new float[length];
        private GCHandle _handle = GCHandle.Alloc(new float[0]);
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
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && _handle.IsAllocated)
            {
                _handle.Free();
            }
        }

        private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}

sealed class FakeStableHostAllocator : IHostStagingFloatBufferAllocator
{
    public IHostStagingFloatBuffer Allocate(int length) => new Buffer(length);

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
    private readonly List<MemcpyCall> _copies = new();
    private long _nextHandle = 100;
    private int _memcpyCalls;
    private int _recordCalls;

    public int MemcpyAsyncCalls => Volatile.Read(ref _memcpyCalls);
    public int EventRecordCalls => Volatile.Read(ref _recordCalls);
    public IReadOnlyList<MemcpyCall> MemcpyCalls
    {
        get { lock (_gate) { return _copies.ToArray(); } }
    }

    public int StreamCreateWithFlags(out nint stream, uint flags)
    {
        lock (_gate)
        {
            stream = (nint)(++_nextHandle);
            _streams.Add(stream);
        }
        return 0;
    }

    public int StreamDestroy(nint stream)
    {
        lock (_gate) { return _streams.Remove(stream) ? 0 : 17; }
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

    public int MemcpyAsync(nint destination, nint source, nuint byteLength, CudaMemcpyKind kind, nint stream)
    {
        Interlocked.Increment(ref _memcpyCalls);
        lock (_gate)
        {
            if (!_streams.Contains(stream)) return 17;
            _copies.Add(new MemcpyCall(destination, source, byteLength, kind, stream));
        }

        var bytes = new byte[checked((int)byteLength)];
        Marshal.Copy(source, bytes, 0, bytes.Length);
        Marshal.Copy(bytes, 0, destination, bytes.Length);
        return 0;
    }

    public int EventCreateWithFlags(out nint completionEvent, uint flags)
    {
        lock (_gate)
        {
            completionEvent = (nint)(++_nextHandle);
            _events.Add(completionEvent, new EventState());
        }
        return 0;
    }

    public int EventRecord(nint completionEvent, nint stream)
    {
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state) || !_streams.Contains(stream)) return 17;
            state.Stream = stream;
            state.Recorded = true;
        }
        Interlocked.Increment(ref _recordCalls);
        return 0;
    }

    public int EventQuery(nint completionEvent)
    {
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state) || !state.Recorded) return 17;
            return state.Completed ? 0 : CudaAsyncCopyEngine.CudaErrorNotReady;
        }
    }

    public int EventDestroy(nint completionEvent)
    {
        lock (_gate) { return _events.Remove(completionEvent) ? 0 : 17; }
    }

    public string? GetErrorString(int errorCode) => $"fake-cuda-error-{errorCode}";

    public void CompleteAllRecordedEvents()
    {
        lock (_gate)
        {
            foreach (var state in _events.Values.Where(static item => item.Recorded)) state.Completed = true;
        }
    }

    public readonly record struct MemcpyCall(nint Destination, nint Source, nuint ByteLength, CudaMemcpyKind Kind, nint Stream);
    private sealed class EventState
    {
        public nint Stream { get; set; }
        public bool Recorded { get; set; }
        public bool Completed { get; set; }
    }
}
