using System.Diagnostics;
using System.Runtime.InteropServices;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

await RunMixedReorderedDuplicateGatherAsync();
await RunCancellationLifetimeAsync();

Console.WriteLine("Fission Optimum CUDA D2D gather specs passed.");

static OptimumLegacyDecoderProfile CreateProfile() =>
    OptimumLegacyDecoderProfile.CreateLlamaLike(
        numHiddenLayers: 1,
        numKvHeads: 1,
        headDim: 1,
        vocabularySize: 8);

static async Task RunMixedReorderedDuplicateGatherAsync()
{
    const int deviceId = 2;
    const string formatId = "gather-spec:fp32:v1";
    var profile = CreateProfile();
    var cuda = new FakeCudaRuntime(initialDevice: 9) { AutoCompleteEvents = true };
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
    await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
        deviceId,
        cuda,
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 4,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    var shape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 2);
    var firstArena = new CudaDecoderOrtCohortArena(
        position: 2,
        batchSize: 2,
        layerCount: 1,
        perSequenceShape: shape,
        allocator);
    var firstRow0 = firstArena.CreateRowState(0, nextTokenId: 1);
    var firstRow1 = firstArena.CreateRowState(1, nextTokenId: 2);
    firstArena.Release();

    var secondArena = new CudaDecoderOrtCohortArena(
        position: 2,
        batchSize: 1,
        layerCount: 1,
        perSequenceShape: shape,
        allocator);
    var secondRow0 = secondArena.CreateRowState(0, nextTokenId: 3);
    secondArena.Release();

    var sourcePointers = cuda.ActiveAllocationPointers.ToArray();
    Require(sourcePointers.Length == 4,
        "Two source arenas with one layer must own four CUDA allocations.");
    cuda.WriteFloats(sourcePointers[0], new[] { 10f, 11f, 20f, 21f });
    cuda.WriteFloats(sourcePointers[1], new[] { 110f, 111f, 120f, 121f });
    cuda.WriteFloats(sourcePointers[2], new[] { 30f, 31f });
    cuda.WriteFloats(sourcePointers[3], new[] { 130f, 131f });

    using var gathered = await OptimumLegacyCudaGatheredPastKvBatch.CreateAsync(
        profile,
        new[] { firstRow1, secondRow0, firstRow1 },
        formatId,
        allocator,
        copyEngine);

    Require(gathered.PastSequenceLength == 2,
        "Gathered past-KV must preserve the common causal position.");
    Require(gathered.BatchSize == 3 && gathered.DeviceId == deviceId,
        "Gathered past-KV must expose the requested dense batch on the target device.");
    Require(gathered.CopiedBytes == 48,
        "Three rows x one layer x key/value x two FP32 elements must copy 48 bytes.");
    Require(gathered.InputNames.SequenceEqual(new[]
    {
        "past_key_values.0.key",
        "past_key_values.0.value"
    }), "Gathered past-KV must preserve exact Optimum input names.");
    Require(gathered.InputValues.Count == 2,
        "One layer must expose one gathered key/value input pair.");
    foreach (var value in gathered.InputValues)
    {
        using var memory = value.GetTensorMemoryInfo();
        Require(memory.Name == "Cuda" && memory.Id == deviceId,
            "Gathered past-KV inputs must remain CUDA-resident.");
        Require(value.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 3, 1, 2, 1 }),
            "Gathered past-KV must expose a dense [B,H,S,D] batch.");
        Require(value.GetTensorSizeInBytes() == 24,
            "Three gathered rows must expose 24 bytes per key/value tensor.");
    }

    Require(cuda.MallocCalls == 6,
        "Gather must allocate exactly one destination key/value buffer for one layer.");
    Require(cuda.MemcpyCalls.Count == 6 &&
            cuda.MemcpyCalls.All(static call =>
                call.Kind == CudaMemcpyKind.DeviceToDevice && call.ByteLength == 8),
        "Mixed/reordered/duplicate gather must retain six independent 8-byte D2D row copies.");

    var destinationPointers = cuda.ActiveAllocationPointers.Skip(4).ToArray();
    Require(destinationPointers.Length == 2,
        "Gather destination must own exactly key/value allocations.");
    Require(cuda.ReadFloats(destinationPointers[0], 6).SequenceEqual(
        new[] { 20f, 21f, 30f, 31f, 20f, 21f }),
        "Gathered key rows must preserve requested reorder, mixed arena, and duplicate semantics.");
    Require(cuda.ReadFloats(destinationPointers[1], 6).SequenceEqual(
        new[] { 120f, 121f, 130f, 131f, 120f, 121f }),
        "Gathered value rows must preserve requested reorder, mixed arena, and duplicate semantics.");

    firstRow0.Dispose();
    firstRow1.Dispose();
    secondRow0.Dispose();
    Require(cuda.FreeCalls == 4,
        "Source allocations must become releasable after gather copy completion.");
    Require(cuda.ActiveAllocationPointers.Count == 2,
        "Gathered dense destination must remain alive after every source state is released.");

    gathered.Dispose();
    Require(cuda.FreeCalls == 6 && cuda.ActiveAllocationPointers.Count == 0,
        "Gather transaction disposal must release its destination allocations exactly once.");
    Require(cuda.MemcpyDevices.All(static device => device == deviceId),
        "Every D2D gather copy must execute under the configured CUDA ordinal.");
    Require(cuda.CurrentDevice == 9,
        "Device-scoped gather copies and allocation lifetime must restore the ambient CUDA device.");
}

static async Task RunCancellationLifetimeAsync()
{
    const int deviceId = 5;
    const string formatId = "gather-cancel:fp32:v1";
    var profile = CreateProfile();
    var cuda = new FakeCudaRuntime(initialDevice: 11) { AutoCompleteEvents = false };
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
    await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
        deviceId,
        cuda,
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 4,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    var shape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 2);
    var sourceArena = new CudaDecoderOrtCohortArena(
        position: 2,
        batchSize: 2,
        layerCount: 1,
        perSequenceShape: shape,
        allocator);
    var row0 = sourceArena.CreateRowState(0, nextTokenId: 4);
    var row1 = sourceArena.CreateRowState(1, nextTokenId: 5);
    sourceArena.Release();

    using var cancellation = new CancellationTokenSource();
    var gatherTask = OptimumLegacyCudaGatheredPastKvBatch.CreateAsync(
            profile,
            new[] { row0, row1 },
            formatId,
            allocator,
            copyEngine,
            cancellation.Token)
        .AsTask();

    await WaitUntilAsync(
        () => cuda.MemcpyCalls.Count == 2,
        "Cancellation lifetime test requires both coalesced D2D copies to be submitted.");
    Require(cuda.MemcpyCalls.All(static call =>
            call.Kind == CudaMemcpyKind.DeviceToDevice && call.ByteLength == 16),
        "Two contiguous source rows must coalesce into one 16-byte key copy and one 16-byte value copy.");
    Require(cuda.MallocCalls == 4 && cuda.FreeCalls == 0,
        "Source and destination allocations must all be alive while D2D copies are pending.");

    row0.Dispose();
    row1.Dispose();
    Require(cuda.FreeCalls == 0,
        "Gather source leases must retain source allocations after caller states are released.");

    cancellation.Cancel();
    await Task.Delay(20);
    Require(!gatherTask.IsCompleted,
        "Post-submission cancellation must wait for CUDA completion events before gather cleanup.");
    Require(cuda.FreeCalls == 0,
        "Cancellation must not free source or destination pointers while native copies are pending.");

    cuda.CompleteAllEvents();
    var canceled = false;
    try
    {
        using var unexpected = await gatherTask;
    }
    catch (OperationCanceledException)
    {
        canceled = true;
    }

    Require(canceled,
        "Gather must surface caller cancellation after all submitted D2D copies become terminal.");
    Require(cuda.FreeCalls == 4 && cuda.ActiveAllocationPointers.Count == 0,
        "Canceled gather must release source and destination allocations only after native completion.");
    Require(cuda.CurrentDevice == 11,
        "Canceled gather cleanup must restore the ambient CUDA device.");
}

static async Task WaitUntilAsync(Func<bool> predicate, string failureMessage)
{
    var stopwatch = Stopwatch.StartNew();
    while (!predicate())
    {
        if (stopwatch.Elapsed >= TimeSpan.FromSeconds(3))
        {
            throw new InvalidOperationException(failureMessage);
        }

        await Task.Delay(1);
    }
}

sealed class FakeCudaRuntime : ICudaDeviceMemoryApi, ICudaAsyncCopyApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _currentDevice;
    private readonly List<nint> _allocationOrder = new();
    private readonly HashSet<nint> _activeAllocations = new();
    private readonly Dictionary<nint, FakeEvent> _events = new();
    private readonly List<MemcpyCall> _memcpyCalls = new();
    private readonly List<int> _memcpyDevices = new();
    private long _nextHandle = 1000;
    private int _mallocCalls;
    private int _freeCalls;

    public FakeCudaRuntime(int initialDevice)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
    }

    public bool AutoCompleteEvents { get; set; }
    public int CurrentDevice => _currentDevice.Value;
    public int MallocCalls => Volatile.Read(ref _mallocCalls);
    public int FreeCalls => Volatile.Read(ref _freeCalls);

    public IReadOnlyList<nint> ActiveAllocationPointers
    {
        get
        {
            lock (_gate)
            {
                return _allocationOrder.Where(_activeAllocations.Contains).ToArray();
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

    public IReadOnlyList<int> MemcpyDevices
    {
        get
        {
            lock (_gate)
            {
                return _memcpyDevices.ToArray();
            }
        }
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
        pointer = Marshal.AllocHGlobal(checked((int)byteLength));
        Interlocked.Increment(ref _mallocCalls);
        lock (_gate)
        {
            _activeAllocations.Add(pointer);
            _allocationOrder.Add(pointer);
        }

        return 0;
    }

    public int Free(nint pointer)
    {
        lock (_gate)
        {
            if (!_activeAllocations.Remove(pointer))
            {
                return 17;
            }
        }

        Marshal.FreeHGlobal(pointer);
        Interlocked.Increment(ref _freeCalls);
        return 0;
    }

    public int StreamCreateWithFlags(out nint stream, uint flags)
    {
        _ = flags;
        stream = (nint)Interlocked.Increment(ref _nextHandle);
        return 0;
    }

    public int StreamDestroy(nint stream)
    {
        _ = stream;
        return 0;
    }

    public int StreamSynchronize(nint stream)
    {
        lock (_gate)
        {
            foreach (var state in _events.Values.Where(state => state.Stream == stream))
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
        var count = checked((int)byteLength);
        var bytes = new byte[count];
        Marshal.Copy(source, bytes, 0, count);
        Marshal.Copy(bytes, 0, destination, count);
        lock (_gate)
        {
            _memcpyCalls.Add(new MemcpyCall(destination, source, byteLength, kind, stream));
            _memcpyDevices.Add(_currentDevice.Value);
        }

        return 0;
    }

    public int EventCreateWithFlags(out nint completionEvent, uint flags)
    {
        _ = flags;
        completionEvent = (nint)Interlocked.Increment(ref _nextHandle);
        lock (_gate)
        {
            _events.Add(completionEvent, new FakeEvent());
        }

        return 0;
    }

    public int EventRecord(nint completionEvent, nint stream)
    {
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state))
            {
                return 17;
            }

            state.Stream = stream;
            state.Completed = AutoCompleteEvents;
        }

        return 0;
    }

    public int EventQuery(nint completionEvent)
    {
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state))
            {
                return 17;
            }

            return state.Completed ? 0 : CudaAsyncCopyEngine.CudaErrorNotReady;
        }
    }

    public int EventDestroy(nint completionEvent)
    {
        lock (_gate)
        {
            return _events.Remove(completionEvent) ? 0 : 17;
        }
    }

    public string? GetErrorString(int errorCode) => $"fake CUDA error {errorCode}";

    public void CompleteAllEvents()
    {
        lock (_gate)
        {
            foreach (var state in _events.Values)
            {
                state.Completed = true;
            }
        }
    }

    public void WriteFloats(nint pointer, float[] values) =>
        Marshal.Copy(values, 0, pointer, values.Length);

    public float[] ReadFloats(nint pointer, int count)
    {
        var values = new float[count];
        Marshal.Copy(pointer, values, 0, count);
        return values;
    }

    private sealed class FakeEvent
    {
        public nint Stream { get; set; }
        public bool Completed { get; set; }
    }

    public readonly record struct MemcpyCall(
        nint Destination,
        nint Source,
        nuint ByteLength,
        CudaMemcpyKind Kind,
        nint Stream);
}
