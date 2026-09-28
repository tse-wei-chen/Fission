using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

static void Require(
    [DoesNotReturnIf(false)] bool condition,
    string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

await RunRoundTripAndDeviceScopeAsync();
await RunCancellationDrainsBeforeFreeAsync();
await RunAllocationFailureBeforeSubmissionAsync();
await RunFormatAndDeviceMismatchAsync();

Console.WriteLine("Fission ONNX Runtime CUDA H2D state specs passed.");

static async Task RunRoundTripAndDeviceScopeAsync()
{
    const string cudaFormat = "cuda-kv:test-v1";
    using var fixture = PayloadFixture.Create(cudaFormat);
    var cuda = new FakeCudaRuntime(initialDevice: 7) { AutoComplete = true };
    await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
        deviceId: 2,
        cuda,
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 2,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = 2 });
    var importer = new CudaHostToDeviceStateImporter(allocator, copyEngine);

    var state = await importer.ImportAsync(fixture.Payload, cudaFormat);

    Require(state.Position == 4 && state.NextTokenId == 19, "H2D import must preserve the decoder causal frontier.");
    Require(state.LayerCount == 1, "H2D import must preserve layer count.");
    Require(cuda.MallocCalls == 2, "One key and one value tensor must allocate two CUDA device buffers.");
    Require(cuda.MemcpyAsyncCalls == 2, "One key and one value tensor must submit two H2D copies.");
    Require(cuda.MemcpyCalls.All(static call => call.Kind == CudaMemcpyKind.HostToDevice), "Target import must use HostToDevice memcpy direction.");
    Require(cuda.OperationDevices.Count != 0 && cuda.OperationDevices.All(static device => device == 2), "Every device-scoped CUDA operation must execute on the configured target ordinal.");
    Require(cuda.CurrentDevice == 7, "Device-scoped CUDA operations must restore the caller thread's ambient device.");

    var layer = state.GetLayer(0);
    using (var keyInfo = layer.Key.GetTensorMemoryInfo())
    using (var valueInfo = layer.Value.GetTensorMemoryInfo())
    {
        Require(keyInfo.Name == "Cuda" && valueInfo.Name == "Cuda", "Imported OrtValues must advertise CUDA device memory.");
        Require(keyInfo.Id == 2 && valueInfo.Id == 2, "Imported OrtValues must advertise the target CUDA ordinal.");
        Require(keyInfo.GetMemoryType() == OrtMemType.Default && valueInfo.GetMemoryType() == OrtMemType.Default, "Imported OrtValues must use device-default ORT memory.");
    }

    Require(layer.Key.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 1, 1, 4, 1 }), "Imported key shape must match staged metadata.");
    Require(layer.Value.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 1, 1, 4, 1 }), "Imported value shape must match staged metadata.");
    Require(layer.Key.GetTensorSizeInBytes() == 16 && layer.Value.GetTensorSizeInBytes() == 16, "Imported OrtValue byte geometry must match staged FP32 payloads.");

    var allocations = cuda.ActiveAllocationPointers;
    Require(allocations.Count == 2, "Target allocations must remain alive while DecoderOrtState is alive.");
    Require(cuda.ReadFloats(allocations[0], 4).SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "H2D key bytes must equal the staged host payload.");
    Require(cuda.ReadFloats(allocations[1], 4).SequenceEqual(new[] { 11f, 12f, 13f, 14f }), "H2D value bytes must equal the staged host payload.");

    fixture.Payload.Dispose();
    Require(cuda.FreeCalls == 0, "Releasing the host payload after H2D completion must not release target CUDA allocations.");
    Require(cuda.ReadFloats(allocations[0], 4).SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "Target CUDA state must be independent from released host staging memory.");

    state.Dispose();
    Require(cuda.FreeCalls == 2 && cuda.ActiveAllocationCount == 0, "DecoderOrtState disposal must release every target CUDA allocation exactly once.");
}

static async Task RunCancellationDrainsBeforeFreeAsync()
{
    const string cudaFormat = "cuda-kv:cancel-v1";
    using var fixture = PayloadFixture.Create(cudaFormat);
    var cuda = new FakeCudaRuntime(initialDevice: 5) { AutoComplete = false };
    await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
        deviceId: 1,
        cuda,
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 2,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = 1 });
    var importer = new CudaHostToDeviceStateImporter(allocator, copyEngine);
    using var cancellation = new CancellationTokenSource();

    var import = importer.ImportAsync(
        fixture.Payload,
        cudaFormat,
        cancellation.Token).AsTask();

    await WaitUntilAsync(
        () => cuda.EventRecordCalls == 2,
        "Cancellation spec requires both H2D copies to reach native submission.");

    cancellation.Cancel();
    await Task.Delay(20);
    Require(!import.IsCompleted, "Cancellation after H2D submission must remain pending until CUDA events complete.");
    Require(cuda.FreeCalls == 0 && cuda.ActiveAllocationCount == 2, "Target CUDA allocations must stay alive while canceled DMA is still in flight.");

    cuda.CompleteAllRecordedEvents();
    var canceled = false;
    try
    {
        _ = await import;
    }
    catch (OperationCanceledException)
    {
        canceled = true;
    }

    Require(canceled, "Caller cancellation must surface after submitted H2D work reaches terminal completion.");
    Require(cuda.FreeCalls == 2 && cuda.ActiveAllocationCount == 0, "Canceled import must free target CUDA allocations only after DMA drain.");
    Require(cuda.CurrentDevice == 5, "Cancellation cleanup must restore the caller thread's ambient CUDA device.");
}

static async Task RunAllocationFailureBeforeSubmissionAsync()
{
    const string cudaFormat = "cuda-kv:alloc-fail-v1";
    using var fixture = PayloadFixture.Create(cudaFormat);
    var cuda = new FakeCudaRuntime(initialDevice: 3)
    {
        AutoComplete = true,
        FailMallocOnCall = 2,
        MallocFailureCode = 77
    };
    await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
        deviceId: 0,
        cuda,
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 2,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = 0 });
    var importer = new CudaHostToDeviceStateImporter(allocator, copyEngine);

    CudaRuntimeException? observed = null;
    try
    {
        _ = await importer.ImportAsync(fixture.Payload, cudaFormat);
    }
    catch (CudaRuntimeException exception)
    {
        observed = exception;
    }

    Require(observed is { Operation: "cudaMalloc", ErrorCode: 77 }, "cudaMalloc failure must retain native operation and error code.");
    Require(cuda.MemcpyAsyncCalls == 0, "Target allocation failure must occur before any H2D copy is submitted.");
    Require(cuda.FreeCalls == 1 && cuda.ActiveAllocationCount == 0, "Allocation failure must release already-created target CUDA buffers.");
    Require(cuda.CurrentDevice == 3, "Allocation failure must restore the ambient CUDA device.");
}

static async Task RunFormatAndDeviceMismatchAsync()
{
    const string cudaFormat = "cuda-kv:mismatch-v1";
    using var fixture = PayloadFixture.Create(cudaFormat);
    var cuda = new FakeCudaRuntime(initialDevice: 4) { AutoComplete = true };
    await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
        deviceId: 2,
        cuda,
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 1,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = 2 });
    var importer = new CudaHostToDeviceStateImporter(allocator, copyEngine);

    var formatRejected = false;
    try
    {
        _ = await importer.ImportAsync(fixture.Payload, "cuda-kv:other-v1");
    }
    catch (InvalidOperationException)
    {
        formatRejected = true;
    }

    Require(formatRejected, "Target importer must reject an incompatible CUDA physical-layout format before allocation.");
    Require(cuda.MallocCalls == 0 && cuda.MemcpyAsyncCalls == 0, "Format mismatch must fail before target allocation or H2D submission.");

    await using var otherEngine = new CudaDeviceBoundAsyncCopyEngine(
        deviceId: 1,
        cuda,
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 1,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });
    var deviceRejected = false;
    try
    {
        _ = new CudaHostToDeviceStateImporter(allocator, otherEngine);
    }
    catch (ArgumentException)
    {
        deviceRejected = true;
    }

    Require(deviceRejected, "Importer must reject allocator/copy-engine device ordinal mismatch.");
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

sealed class PayloadFixture : IDisposable
{
    private readonly PinnedFloatBufferPool _pool;
    private int _disposed;

    private PayloadFixture(
        PinnedFloatBufferPool pool,
        DecoderOrtCudaHostStagingPayload payload)
    {
        _pool = pool;
        Payload = payload;
    }

    public DecoderOrtCudaHostStagingPayload Payload { get; }

    public static PayloadFixture Create(string cudaFormat)
    {
        var pool = new PinnedFloatBufferPool(
            new PinnedHostStagingPoolOptions
            {
                MaxRetainedBuffersPerLength = 2,
                MaxRetainedBytes = 1024,
                ClearOnReturn = true
            },
            new FakeCudaPageLockedAllocator());
        PinnedFloatBufferPool.PinnedFloatBufferLease? key = null;
        PinnedFloatBufferPool.PinnedFloatBufferLease? value = null;
        try
        {
            key = pool.Rent(4);
            value = pool.Rent(4);
            new[] { 1f, 2f, 3f, 4f }.CopyTo(key.Span);
            new[] { 11f, 12f, 13f, 14f }.CopyTo(value.Span);

            var payload = new DecoderOrtCudaHostStagingPayload(
                CudaDeviceToHostStagingExporter.BuildHostStagingFormatId(cudaFormat),
                cudaFormat,
                position: 4,
                nextTokenId: 19,
                byteLength: 40,
                new[] { key },
                new[] { value },
                new IReadOnlyList<long>[] { new long[] { 1, 1, 4, 1 } },
                new IReadOnlyList<long>[] { new long[] { 1, 1, 4, 1 } });
            key = null;
            value = null;
            return new PayloadFixture(pool, payload);
        }
        catch
        {
            value?.Dispose();
            key?.Dispose();
            pool.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        Payload.Dispose();
        _pool.Dispose();
    }
}

sealed class FakeCudaPageLockedAllocator : IHostStagingFloatBufferAllocator
{
    public IHostStagingFloatBuffer Allocate(int length) => new Buffer(length);

    private sealed class Buffer : ICudaPageLockedHostStagingFloatBuffer
    {
        private readonly float[] _values;
        private GCHandle _handle;
        private int _disposed;

        public Buffer(int length)
        {
            _values = new float[length];
            _handle = GCHandle.Alloc(_values, GCHandleType.Pinned);
        }

        public Memory<float> Memory
        {
            get
            {
                ThrowIfDisposed();
                return _values;
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

        private void ThrowIfDisposed() =>
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}

sealed class FakeCudaRuntime : ICudaAsyncCopyApi, ICudaDeviceMemoryApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _currentDevice;
    private readonly HashSet<nint> _streams = new();
    private readonly Dictionary<nint, EventState> _events = new();
    private readonly List<MemcpyCall> _memcpyCalls = new();
    private readonly List<nint> _allocationOrder = new();
    private readonly HashSet<nint> _allocations = new();
    private readonly List<int> _operationDevices = new();
    private long _nextHandle = 100;
    private int _mallocCalls;
    private int _freeCalls;
    private int _memcpyAsyncCalls;
    private int _eventRecordCalls;

    public FakeCudaRuntime(int initialDevice)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
    }

    public bool AutoComplete { get; set; }
    public int FailMallocOnCall { get; set; }
    public int MallocFailureCode { get; set; } = 55;

    public int CurrentDevice => _currentDevice.Value;
    public int MallocCalls => Volatile.Read(ref _mallocCalls);
    public int FreeCalls => Volatile.Read(ref _freeCalls);
    public int MemcpyAsyncCalls => Volatile.Read(ref _memcpyAsyncCalls);
    public int EventRecordCalls => Volatile.Read(ref _eventRecordCalls);

    public int ActiveAllocationCount
    {
        get
        {
            lock (_gate)
            {
                return _allocations.Count;
            }
        }
    }

    public IReadOnlyList<int> OperationDevices
    {
        get
        {
            lock (_gate)
            {
                return _operationDevices.ToArray();
            }
        }
    }

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

    public IReadOnlyList<nint> ActiveAllocationPointers
    {
        get
        {
            lock (_gate)
            {
                return _allocationOrder.Where(_allocations.Contains).ToArray();
            }
        }
    }

    public float[] ReadFloats(nint pointer, int count)
    {
        lock (_gate)
        {
            if (!_allocations.Contains(pointer))
            {
                throw new ObjectDisposedException("fake CUDA allocation");
            }
        }

        var result = new float[count];
        Marshal.Copy(pointer, result, 0, count);
        return result;
    }

    public int GetDevice(out int deviceId)
    {
        deviceId = _currentDevice.Value;
        return 0;
    }

    public int SetDevice(int deviceId)
    {
        _currentDevice.Value = deviceId;
        return 0;
    }

    public int Malloc(out nint pointer, nuint byteLength)
    {
        var call = Interlocked.Increment(ref _mallocCalls);
        RecordOperationDevice();
        if (FailMallocOnCall != 0 && call == FailMallocOnCall)
        {
            pointer = 0;
            return MallocFailureCode;
        }

        pointer = Marshal.AllocHGlobal(checked((int)byteLength));
        lock (_gate)
        {
            _allocations.Add(pointer);
            _allocationOrder.Add(pointer);
        }

        return 0;
    }

    public int Free(nint pointer)
    {
        RecordOperationDevice();
        var removed = false;
        lock (_gate)
        {
            removed = _allocations.Remove(pointer);
        }

        if (!removed)
        {
            return 17;
        }

        Marshal.FreeHGlobal(pointer);
        Interlocked.Increment(ref _freeCalls);
        return 0;
    }

    public int StreamCreateWithFlags(out nint stream, uint flags)
    {
        RecordOperationDevice();
        lock (_gate)
        {
            stream = (nint)(++_nextHandle);
            _streams.Add(stream);
        }

        return 0;
    }

    public int StreamDestroy(nint stream)
    {
        RecordOperationDevice();
        lock (_gate)
        {
            return _streams.Remove(stream) ? 0 : 17;
        }
    }

    public int StreamSynchronize(nint stream)
    {
        RecordOperationDevice();
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
        RecordOperationDevice();
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
        RecordOperationDevice();
        lock (_gate)
        {
            completionEvent = (nint)(++_nextHandle);
            _events.Add(completionEvent, new EventState());
        }

        return 0;
    }

    public int EventRecord(nint completionEvent, nint stream)
    {
        Interlocked.Increment(ref _eventRecordCalls);
        RecordOperationDevice();
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state) || !_streams.Contains(stream))
            {
                return 17;
            }

            state.Stream = stream;
            state.Recorded = true;
            state.Completed = AutoComplete;
        }

        return 0;
    }

    public int EventQuery(nint completionEvent)
    {
        RecordOperationDevice();
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state) || !state.Recorded)
            {
                return 17;
            }

            return state.Completed ? 0 : CudaAsyncCopyEngine.CudaErrorNotReady;
        }
    }

    public int EventDestroy(nint completionEvent)
    {
        RecordOperationDevice();
        lock (_gate)
        {
            return _events.Remove(completionEvent) ? 0 : 17;
        }
    }

    public void CompleteAllRecordedEvents()
    {
        lock (_gate)
        {
            foreach (var state in _events.Values)
            {
                if (state.Recorded)
                {
                    state.Completed = true;
                }
            }
        }
    }

    public string? GetErrorString(int errorCode) => $"fake cuda error {errorCode}";

    private void RecordOperationDevice()
    {
        lock (_gate)
        {
            _operationDevices.Add(_currentDevice.Value);
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
