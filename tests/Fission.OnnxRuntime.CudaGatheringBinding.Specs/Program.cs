using System.Buffers;
using System.Runtime.InteropServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

const string MulModelBase64 =
    "CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0" +
    "KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4K" +
    "DAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==";

const int deviceId = 2;
var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 1,
    numKvHeads: 1,
    headDim: 1,
    vocabularySize: 4);
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
var logitsPool = new ScriptedFloatPool(new[]
{
    new[] { 0f, 10f, 0f, 0f, 0f, 0f, 10f, 0f },
    new[] { 0f, 0f, 0f, 10f, 10f, 0f, 0f, 0f }
});

var runCount = 0;
void FakeRun(
    InferenceSession session,
    RunOptions runOptions,
    IReadOnlyCollection<string> inputNames,
    IReadOnlyCollection<OrtValue> inputValues,
    IReadOnlyCollection<string> outputNames,
    IReadOnlyCollection<OrtValue> outputValues)
{
    _ = session;
    _ = runOptions;
    runCount++;

    var names = inputNames.ToArray();
    var inputs = inputValues.ToArray();
    var outputs = outputValues.ToArray();
    Require(names.SequenceEqual(new[]
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values.0.key",
        "past_key_values.0.value"
    }), "Gather-aware binding must preserve exact Optimum input ordering.");
    Require(outputNames.SequenceEqual(new[]
    {
        "logits",
        "present.0.key",
        "present.0.value"
    }), "Gather-aware binding must preserve exact Optimum output ordering.");

    using (var pastMemory = inputs[3].GetTensorMemoryInfo())
    {
        Require(pastMemory.Name == "Cuda" && pastMemory.Id == deviceId,
            "Gather-aware decode must keep past KV CUDA-resident.");
    }
    using (var keyMemory = outputs[1].GetTensorMemoryInfo())
    using (var valueMemory = outputs[2].GetTensorMemoryInfo())
    {
        Require(keyMemory.Name == "Cuda" && keyMemory.Id == deviceId &&
                valueMemory.Name == "Cuda" && valueMemory.Id == deviceId,
            "Gather-aware decode must bind present KV directly on CUDA.");
    }

    var inputIds = inputs[0].GetTensorDataAsSpan<long>().ToArray();
    var pastShape = inputs[3].GetTensorTypeAndShape().Shape;
    var presentShape = outputs[1].GetTensorTypeAndShape().Shape;
    Require(pastShape.SequenceEqual(new long[] { 2, 1, 2, 1 }),
        "Both gathered and direct dense runs must expose [2,1,2,1] past KV.");
    Require(presentShape.SequenceEqual(new long[] { 2, 1, 3, 1 }),
        "Both gathered and direct dense runs must advance to [2,1,3,1] present KV.");

    if (runCount == 1)
    {
        Require(inputIds.SequenceEqual(new long[] { 3, 3 }),
            "Duplicate/fork gather must preserve both logical next-token frontiers.");
        Require(cuda.MemcpyCalls.Count == 4,
            "One-layer two-row duplicate gather must submit exactly four D2D copies before ORT.");
        Require(cuda.MemcpyCalls.All(static call =>
            call.Kind == CudaMemcpyKind.DeviceToDevice && call.ByteLength == 8),
            "Duplicate gather must copy one two-element FP32 row for each key/value slot.");
        Require(cuda.ReadFloats(cuda.MemcpyCalls[0].Destination, 2).SequenceEqual(new[] { 10f, 11f }) &&
                cuda.ReadFloats(cuda.MemcpyCalls[1].Destination, 2).SequenceEqual(new[] { 110f, 111f }) &&
                cuda.ReadFloats(cuda.MemcpyCalls[2].Destination, 2).SequenceEqual(new[] { 10f, 11f }) &&
                cuda.ReadFloats(cuda.MemcpyCalls[3].Destination, 2).SequenceEqual(new[] { 110f, 111f }),
            "Gathered CUDA rows must duplicate source key/value bytes without host staging.");
    }
    else if (runCount == 2)
    {
        Require(inputIds.SequenceEqual(new long[] { 1, 2 }),
            "Complete dense cohort must preserve row next-token order.");
        Require(cuda.MemcpyCalls.Count == 4,
            "Complete dense cohort must bypass D2D gather entirely.");
    }
    else
    {
        throw new InvalidOperationException($"Unexpected fake ORT run {runCount}.");
    }
}

using var session = new InferenceSession(Convert.FromBase64String(MulModelBase64));
using var inner = new OptimumLegacyCudaFloatDecoderBinding(
    profile,
    allocator,
    FakeRun,
    scratchFloatPool: logitsPool);
using var binding = new OptimumLegacyCudaGatheringBinding(
    profile,
    allocator,
    copyEngine,
    inner);
var modelId = new ModelId("cuda-gathering-binding-spec");

var singleShape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 2);
var forkArena = new CudaDecoderOrtCohortArena(
    position: 2,
    batchSize: 1,
    layerCount: 1,
    perSequenceShape: singleShape,
    allocator);
var forkSource = forkArena.CreateRowState(0, nextTokenId: 3);
forkArena.Release();
var forkPointers = cuda.ActiveAllocationPointers.ToArray();
Require(forkPointers.Length == 2,
    "Single source row must own exactly one key/value allocation pair.");
cuda.WriteFloats(forkPointers[0], new[] { 10f, 11f });
cuda.WriteFloats(forkPointers[1], new[] { 110f, 111f });

var forkResults = await binding.ExecuteDecodeBatchAsync(
    session,
    new[]
    {
        new DecodeItem(SequenceId.New(), modelId, Position: 2),
        new DecodeItem(SequenceId.New(), modelId, Position: 2)
    },
    new[] { forkSource, forkSource });

Require(forkResults.Count == 2 && forkResults[0].TokenId == 1 && forkResults[1].TokenId == 2,
    "Duplicate/fork gather must produce one independently sampled result per logical row.");
Require(forkResults.All(static result => result.State.Position == 3),
    "Gathered decode must advance every row frontier exactly once.");
Require(binding.GatheredBatchCount == 1 && binding.GatheredBytes == 32,
    "Duplicate/fork batch must record exactly one 32-byte D2D gather.");
Require(binding.SingletonFallbackRunCount == 0,
    "Duplicate/fork layout must not degrade to singleton ORT launches once gather is available.");
Require(binding.OrtRunCount == 1 && runCount == 1,
    "Duplicate/fork layout must execute exactly one batched ORT run.");
Require(binding.CudaPastReuseCount == 1,
    "Gathered dense arena must reuse CUDA past KV through the existing binding.");
Require(cuda.MallocCalls == 6 && cuda.FreeCalls == 2,
    "Gathered run must allocate source, temporary gathered past, and present pairs, then release only temporary gathered past before returning.");

forkSource.Dispose();
foreach (var result in forkResults)
{
    result.State.Dispose();
}
Require(cuda.MallocCalls == 6 && cuda.FreeCalls == 6 && cuda.ActiveAllocationPointers.Count == 0,
    "Duplicate/fork source and result disposal must release every CUDA allocation exactly once.");

var denseShape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 2);
var denseArena = new CudaDecoderOrtCohortArena(
    position: 2,
    batchSize: 2,
    layerCount: 1,
    perSequenceShape: denseShape,
    allocator);
var dense0 = denseArena.CreateRowState(0, nextTokenId: 1);
var dense1 = denseArena.CreateRowState(1, nextTokenId: 2);
denseArena.Release();
var copiesBeforeDense = cuda.MemcpyCalls.Count;

var denseResults = await binding.ExecuteDecodeBatchAsync(
    session,
    new[]
    {
        new DecodeItem(SequenceId.New(), modelId, Position: 2),
        new DecodeItem(SequenceId.New(), modelId, Position: 2)
    },
    new[] { dense0, dense1 });

Require(denseResults.Count == 2 && denseResults[0].TokenId == 3 && denseResults[1].TokenId == 0,
    "Direct dense CUDA batch must preserve independent greedy sampling.");
Require(binding.GatheredBatchCount == 1 && binding.GatheredBytes == 32,
    "Direct dense cohort must not increment gather counters.");
Require(cuda.MemcpyCalls.Count == copiesBeforeDense,
    "Direct dense cohort must issue no D2D gather copies.");
Require(binding.SingletonFallbackRunCount == 0,
    "Direct dense cohort must remain a single batched execution.");
Require(binding.OrtRunCount == 2 && runCount == 2,
    "Duplicate gather and dense reuse must each require exactly one ORT run.");
Require(binding.CudaPastReuseCount == 2,
    "Both gathered and direct dense runs must use the CUDA past-reuse path.");

dense0.Dispose();
dense1.Dispose();
foreach (var result in denseResults)
{
    result.State.Dispose();
}
Require(cuda.MallocCalls == 10 && cuda.FreeCalls == 10 && cuda.ActiveAllocationPointers.Count == 0,
    "Dense bypass teardown must release all source and present CUDA allocations.");
Require(cuda.CurrentDevice == 9,
    "Gather-aware allocation/copy/free operations must restore the ambient CUDA device.");
Require(logitsPool.RentCount == 2 && logitsPool.ReturnCount == 2,
    "Each batched ORT run must return its host logits scratch lease.");

Console.WriteLine(
    $"Fission CUDA gathering binding specs passed: runs={binding.OrtRunCount}, " +
    $"gathers={binding.GatheredBatchCount}, bytes={binding.GatheredBytes}, copies={cuda.MemcpyCalls.Count}.");

sealed class ScriptedFloatPool : ArrayPool<float>
{
    private readonly Queue<float[]> _scripts;
    private int _rentCount;
    private int _returnCount;

    public ScriptedFloatPool(IEnumerable<float[]> scripts)
    {
        _scripts = new Queue<float[]>(scripts.Select(static values => values.ToArray()));
    }

    public int RentCount => Volatile.Read(ref _rentCount);
    public int ReturnCount => Volatile.Read(ref _returnCount);

    public override float[] Rent(int minimumLength)
    {
        Interlocked.Increment(ref _rentCount);
        if (_scripts.Count == 0)
        {
            throw new InvalidOperationException("No scripted logits remain.");
        }

        var values = _scripts.Dequeue();
        if (values.Length < minimumLength)
        {
            throw new InvalidOperationException(
                $"Scripted logits length {values.Length} is smaller than requested {minimumLength}.");
        }

        return values;
    }

    public override void Return(float[] array, bool clearArray = false)
    {
        _ = array;
        _ = clearArray;
        Interlocked.Increment(ref _returnCount);
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
