using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require([DoesNotReturnIf(false)] bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

await RunExportAndReuseAsync();
await RunCancellationAsync();
await RunNonPageLockedRejectedAsync();
Console.WriteLine("Fission ONNX Runtime CUDA D2H staging specs passed.");

static async Task RunExportAndReuseAsync()
{
    var cuda = new FakeCudaApi();
    await using var engine = new CudaAsyncCopyEngine(cuda, new CudaAsyncCopyEngineOptions
    {
        StreamCount = 2,
        CompletionPollInterval = TimeSpan.FromMilliseconds(1)
    });
    using var exporter = new CudaDeviceToHostStagingExporter(
        engine,
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
    using var binding = new FakeCudaBinding(source, lifetime);

    var export = exporter.ExportAsync(binding, source.State).AsTask();
    await WaitUntilAsync(() => cuda.EventRecordCalls == 2, "Both KV copies must be submitted.");
    Require(!export.IsCompleted, "Payload must not publish before CUDA events complete.");
    Require(lifetime.Active == 1, "Source CUDA retain must survive in-flight DMA.");
    Require(cuda.Copies.Count == 2, "One layer must submit exactly two copies.");
    Require(cuda.Copies.All(static copy => copy.Kind == CudaMemcpyKind.DeviceToHost), "Copies must be DeviceToHost.");
    Require(exporter.HostStagingPoolStatistics.ReturnedBuffers == 0, "In-flight destinations must stay leased.");

    cuda.CompleteAll();
    var payload = await export;
    Require(lifetime.Active == 0 && lifetime.Released == 1, "Source retain must release after all DMA completes.");
    Require(payload.Position == 2 && payload.NextTokenId == 19, "Causal frontier must be preserved.");
    Require(payload.SourceCudaFormatId == binding.CudaResidentStateFormatId, "Source CUDA format must be preserved.");
    Require(payload.FormatId == CudaDeviceToHostStagingExporter.BuildHostStagingFormatId(binding.CudaResidentStateFormatId), "Host format id must be deterministic.");
    Require(payload.ByteLength == 40, "Payload bytes must include KV and causal metadata.");
    Require(payload.GetKeyMemory(0).Span.SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "Key bytes must survive D2H.");
    Require(payload.GetValueMemory(0).Span.SequenceEqual(new[] { 5f, 6f, 7f, 8f }), "Value bytes must survive D2H.");
    Require(payload.GetKeyShape(0).SequenceEqual(source.Shape), "Tensor shape must survive staging.");
    payload.Dispose();

    var returned = exporter.HostStagingPoolStatistics;
    Require(returned.ReturnedBuffers == 2 && returned.RetainedBuffers == 2, "Payload disposal must return both buffers.");

    var second = exporter.ExportAsync(binding, source.State).AsTask();
    await WaitUntilAsync(() => cuda.EventRecordCalls == 4, "Second export must submit both copies.");
    cuda.CompleteAll();
    var secondPayload = await second;
    var reused = exporter.HostStagingPoolStatistics;
    Require(reused.AllocatedBuffers == 2 && reused.ReusedBuffers == 2, "Repeated same-shape export must reuse page-locked buffers.");
    secondPayload.Dispose();
}

static async Task RunCancellationAsync()
{
    var cuda = new FakeCudaApi();
    await using var engine = new CudaAsyncCopyEngine(cuda, new CudaAsyncCopyEngineOptions
    {
        StreamCount = 2,
        CompletionPollInterval = TimeSpan.FromMilliseconds(1)
    });
    using var exporter = new CudaDeviceToHostStagingExporter(engine, new FakeCudaPageLockedAllocator());
    using var source = SyntheticCudaState.Create(
        new[] { 11f, 12f, 13f, 14f },
        new[] { 21f, 22f, 23f, 24f },
        position: 2,
        nextTokenId: 23);
    var lifetime = new LifetimeProbe();
    using var binding = new FakeCudaBinding(source, lifetime);
    using var cancellation = new CancellationTokenSource();

    var export = exporter.ExportAsync(binding, source.State, cancellation.Token).AsTask();
    await WaitUntilAsync(() => cuda.EventRecordCalls == 2, "Cancellation test requires submitted DMA.");
    cancellation.Cancel();
    await Task.Delay(20);
    Require(!export.IsCompleted, "Post-submit cancellation must wait for native completion.");
    Require(lifetime.Active == 1, "Canceled in-flight export must retain source CUDA allocation.");
    Require(exporter.HostStagingPoolStatistics.ReturnedBuffers == 0, "Canceled in-flight destinations must stay leased.");

    cuda.CompleteAll();
    var canceled = false;
    try { _ = await export; }
    catch (OperationCanceledException) { canceled = true; }

    Require(canceled, "Cancellation must surface after native completion.");
    Require(lifetime.Active == 0 && lifetime.Released == 1, "Canceled export must release source retain after drain.");
    var stats = exporter.HostStagingPoolStatistics;
    Require(stats.ReturnedBuffers == 2 && stats.RetainedBuffers == 2, "Canceled export must reclaim both destinations after drain.");
}

static async Task RunNonPageLockedRejectedAsync()
{
    var cuda = new FakeCudaApi();
    await using var engine = new CudaAsyncCopyEngine(cuda, new CudaAsyncCopyEngineOptions
    {
        StreamCount = 1,
        CompletionPollInterval = TimeSpan.FromMilliseconds(1)
    });
    using var exporter = new CudaDeviceToHostStagingExporter(engine, new FakeStableHostAllocator());
    using var source = SyntheticCudaState.Create(
        new[] { 31f, 32f, 33f, 34f },
        new[] { 41f, 42f, 43f, 44f },
        position: 2,
        nextTokenId: 29);
    var lifetime = new LifetimeProbe();
    using var binding = new FakeCudaBinding(source, lifetime);

    Exception? observed = null;
    try { _ = await exporter.ExportAsync(binding, source.State); }
    catch (Exception exception) { observed = exception; }

    Require(observed is InvalidOperationException, "Stable but non-CUDA-page-locked host memory must be rejected.");
    Require(cuda.MemcpyCalls == 0, "No DMA may be submitted to a non-page-locked destination.");
    Require(lifetime.Active == 0 && lifetime.Released == 1, "Pre-submit failure must release the source retain.");
}

static async Task WaitUntilAsync(Func<bool> predicate, string message)
{
    var timeout = Stopwatch.StartNew();
    while (!predicate())
    {
        if (timeout.Elapsed >= TimeSpan.FromSeconds(3)) throw new InvalidOperationException(message);
        await Task.Delay(1);
    }
}

sealed class FakeCudaBinding : IDecoderOrtCudaResidentStateBinding
{
    private readonly SyntheticCudaState _source;
    private readonly LifetimeProbe _lifetime;

    public FakeCudaBinding(SyntheticCudaState source, LifetimeProbe lifetime)
    {
        _source = source;
        _lifetime = lifetime;
    }

    public string Name => "fake-cuda-resident";
    public OnnxSessionContract SessionContract => throw new NotSupportedException();
    public string CudaResidentStateFormatId => "fake-cuda-fp32-kv-v1";

    public DecoderOrtCudaResidentStateLease AcquireCudaResidentState(DecoderOrtState state, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!ReferenceEquals(state, _source.State)) throw new InvalidOperationException("Unexpected state.");
        return DecoderOrtCudaResidentStateLease.Create(
            CudaResidentStateFormatId,
            state,
            0,
            _source.CreateViews(),
            _lifetime.Retain());
    }

    public DecoderOrtStepResult ExecutePrefill(InferenceSession session, PrefillItem item, CancellationToken cancellationToken = default) => throw new NotSupportedException();
    public DecoderOrtStepResult ExecuteDecode(InferenceSession session, DecodeItem item, DecoderOrtState priorState, CancellationToken cancellationToken = default) => throw new NotSupportedException();
    public void Dispose() { }
}

sealed class SyntheticCudaState : IDisposable
{
    private readonly OrtMemoryInfo _memoryInfo;
    private int _disposed;

    private SyntheticCudaState(OrtMemoryInfo memoryInfo, nint key, nint value, long bytes, long[] shape, DecoderOrtState state)
    {
        _memoryInfo = memoryInfo;
        KeyPointer = key;
        ValuePointer = value;
        ByteLength = bytes;
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
        if (keys.Length == 0 || keys.Length != values.Length) throw new ArgumentException("Invalid synthetic KV length.");
        var shape = new long[] { 1, 1, 1, keys.Length };
        var bytes = checked(keys.Length * (long)sizeof(float));
        var keyPtr = Marshal.AllocHGlobal(checked((int)bytes));
        var valuePtr = Marshal.AllocHGlobal(checked((int)bytes));
        Marshal.Copy(keys, 0, keyPtr, keys.Length);
        Marshal.Copy(values, 0, valuePtr, values.Length);
        var memoryInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        OrtValue? key = null;
        OrtValue? value = null;
        try
        {
            key = OrtValue.CreateTensorValueWithData(memoryInfo, TensorElementType.Float, shape, keyPtr, bytes);
            value = OrtValue.CreateTensorValueWithData(memoryInfo, TensorElementType.Float, shape, valuePtr, bytes);
            var state = new DecoderOrtState(position, new[] { new DecoderOrtLayerState(key, value) }, nextTokenId);
            key = null;
            value = null;
            return new SyntheticCudaState(memoryInfo, keyPtr, valuePtr, bytes, shape, state);
        }
        catch
        {
            value?.Dispose();
            key?.Dispose();
            memoryInfo.Dispose();
            Marshal.FreeHGlobal(valuePtr);
            Marshal.FreeHGlobal(keyPtr);
            throw;
        }
    }

    public DecoderOrtCudaLayerView[] CreateViews() =>
    [
        new DecoderOrtCudaLayerView(
            new CudaDeviceTensorView(KeyPointer, ByteLength, TensorElementType.Float, Shape, 0),
            new CudaDeviceTensorView(ValuePointer, ByteLength, TensorElementType.Float, Shape, 0))
    ];

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
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
    public int Active => Volatile.Read(ref _active);
    public int Released => Volatile.Read(ref _released);

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
            get { ThrowIfDisposed(); return _buffer; }
        }

        public nint Pointer
        {
            get { ThrowIfDisposed(); return _handle.AddrOfPinnedObject(); }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && _handle.IsAllocated) _handle.Free();
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
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && _handle.IsAllocated) _handle.Free();
        }
    }
}

sealed class FakeCudaApi : ICudaAsyncCopyApi
{
    private readonly object _gate = new();
    private readonly HashSet<nint> _streams = new();
    private readonly Dictionary<nint, EventState> _events = new();
    private readonly List<MemcpyCall> _copies = new();
    private long _next = 100;
    private int _memcpyCalls;
    private int _recordCalls;

    public int MemcpyCalls => Volatile.Read(ref _memcpyCalls);
    public int EventRecordCalls => Volatile.Read(ref _recordCalls);
    public IReadOnlyList<MemcpyCall> Copies { get { lock (_gate) return _copies.ToArray(); } }

    public int StreamCreateWithFlags(out nint stream, uint flags)
    {
        lock (_gate)
        {
            stream = (nint)(++_next);
            _streams.Add(stream);
        }
        return 0;
    }

    public int StreamDestroy(nint stream) { lock (_gate) return _streams.Remove(stream) ? 0 : 17; }

    public int StreamSynchronize(nint stream)
    {
        lock (_gate)
        {
            foreach (var state in _events.Values.Where(item => item.Stream == stream)) state.Completed = true;
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
            completionEvent = (nint)(++_next);
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

    public int EventDestroy(nint completionEvent) { lock (_gate) return _events.Remove(completionEvent) ? 0 : 17; }
    public string? GetErrorString(int errorCode) => $"fake-cuda-error-{errorCode}";

    public void CompleteAll()
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
