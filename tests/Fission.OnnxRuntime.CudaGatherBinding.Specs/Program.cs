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

const int deviceId = 4;
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
    new[] { 10f, 0f, 0f, 0f, 0f, 0f, 0f, 10f },
    new[] { 0f, 10f, 0f, 0f, 0f, 0f, 10f, 0f }
});

var fakeRunCount = 0;
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
    fakeRunCount++;

    var names = inputNames.ToArray();
    var inputs = inputValues.ToArray();
    var outputNameArray = outputNames.ToArray();
    var outputs = outputValues.ToArray();
    var pastIndex = Array.IndexOf(names, "past_key_values.0.key");
    var presentIndex = Array.IndexOf(outputNameArray, "present.0.key");
    Require(pastIndex >= 0 && presentIndex >= 0,
        "Gather-aware runs must bind Optimum past/present KV names.");

    using (var pastInfo = inputs[pastIndex].GetTensorMemoryInfo())
    using (var presentInfo = outputs[presentIndex].GetTensorMemoryInfo())
    {
        Require(pastInfo.Name == "Cuda" && pastInfo.Id == deviceId,
            "Direct and gathered past KV must remain CUDA-resident.");
        Require(presentInfo.Name == "Cuda" && presentInfo.Id == deviceId,
            "Present KV must be written directly to CUDA memory.");
    }

    var pastShape = inputs[pastIndex].GetTensorTypeAndShape().Shape;
    var presentShape = outputs[presentIndex].GetTensorTypeAndShape().Shape;
    var inputIds = inputs[0].GetTensorDataAsSpan<long>().ToArray();
    if (fakeRunCount == 1)
    {
        Require(pastShape.SequenceEqual(new long[] { 2, 1, 2, 1 }),
            "Complete dense cohort must reuse its two-token past frontier directly.");
        Require(presentShape.SequenceEqual(new long[] { 2, 1, 3, 1 }),
            "Dense decode must advance to a three-token present frontier.");
        Require(inputIds.SequenceEqual(new long[] { 1, 2 }),
            "Reversed logical dense requests must restore physical CUDA row order before ORT.");
        Require(cuda.MemcpyCalls.Count == 0,
            "Complete dense cohort must not submit D2D gather copies.");
    }
    else if (fakeRunCount == 2)
    {
        Require(pastShape.SequenceEqual(new long[] { 2, 1, 3, 1 }),
            "Duplicate-row fork must enter ORT as one dense gathered past tensor.");
        Require(presentShape.SequenceEqual(new long[] { 2, 1, 4, 1 }),
            "Gathered fork decode must advance to a four-token present frontier.");
        Require(inputIds.SequenceEqual(new long[] { 3, 3 }),
            "Gathered duplicate rows must preserve the shared next-token frontier.");
        Require(cuda.MemcpyCalls.Count == 4,
            "One-layer two-row fork gather must submit exactly four D2D copies before ORT.");
    }
    else
    {
        throw new InvalidOperationException(
            $"Gather binding must execute exactly two ORT runs, observed {fakeRunCount}.");
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

var sourceArena = new CudaDecoderOrtCohortArena(
    position: 2,
    batchSize: 2,
    layerCount: 1,
    perSequenceShape: new long[] { 1, 1, 2, 1 },
    allocator);
var row0 = sourceArena.CreateRowState(0, nextTokenId: 1);
var row1 = sourceArena.CreateRowState(1, nextTokenId: 2);
sourceArena.Release();
Require(cuda.MallocCalls == 2,
    "One source layer must allocate exactly one CUDA key/value pair.");

var modelId = new ModelId("cuda-gather-binding");
var firstId = SequenceId.New();
var secondId = SequenceId.New();
var dense = await binding.ExecuteDecodeBatchAsync(
    session,
    new[]
    {
        new DecodeItem(secondId, modelId, Position: 2),
        new DecodeItem(firstId, modelId, Position: 2)
    },
    new[] { row1, row0 });

Require(dense.Count == 2 && dense[0].TokenId == 3 && dense[1].TokenId == 0,
    "Dense physical-row logits must map back to reversed logical request order.");
Require(binding.GatheredBatchCount == 0 && binding.GatheredBytes == 0,
    "Complete dense cohorts must not pay D2D gather cost.");
Require(binding.SingletonFallbackRunCount == 0,
    "Complete dense cohorts must not pay singleton fallback cost.");
Require(binding.OrtRunCount == 1 && fakeRunCount == 1,
    "Complete dense cohort must execute exactly one ORT run.");
Require(binding.CudaPastReuseCount == 1 && cuda.MemcpyCalls.Count == 0,
    "Dense path must reuse CUDA past without any memcpy.");
Require(cuda.MallocCalls == 4,
    "Dense reuse must allocate only the next present key/value pair.");

var forked = await binding.ExecuteDecodeBatchAsync(
    session,
    new[]
    {
        new DecodeItem(SequenceId.New(), modelId, Position: 3),
        new DecodeItem(SequenceId.New(), modelId, Position: 3)
    },
    new[] { dense[0].State, dense[0].State });

Require(forked.Count == 2 && forked[0].TokenId == 1 && forked[1].TokenId == 2,
    "Gathered fork batch must preserve logical row ordering and independent logits.");
Require(!ReferenceEquals(forked[0].State, forked[1].State),
    "Gathered fork execution must return independently owned CUDA states.");
Require(forked.All(static result => result.State.Position == 4),
    "Gathered fork rows must advance their causal frontier exactly once.");
Require(binding.GatheredBatchCount == 1 && binding.GatheredBytes == 48,
    "Two rows x one layer x key/value x three FP32 elements must gather exactly 48 bytes.");
Require(binding.SingletonFallbackRunCount == 0,
    "D2D-capable fork layout must not fall back to singleton ORT launches.");
Require(binding.OrtRunCount == 2 && fakeRunCount == 2 && binding.CudaPastReuseCount == 2,
    "Dense execution and gathered fork must each use one batched ORT CUDA-past run.");
Require(cuda.MallocCalls == 8 && cuda.FreeCalls == 2,
    "Fork gather must allocate one temporary key/value pair plus one next-state pair and release only the temporary pair before returning.");
Require(cuda.MemcpyCalls.Count == 4 &&
        cuda.MemcpyCalls.All(static call =>
            call.Kind == CudaMemcpyKind.DeviceToDevice && call.ByteLength == 12),
    "Two duplicate rows must submit exactly four 12-byte D2D key/value copies.");
Require(cuda.MemcpyCalls[0].Source == cuda.MemcpyCalls[2].Source &&
        cuda.MemcpyCalls[1].Source == cuda.MemcpyCalls[3].Source,
    "Duplicate/fork gather must intentionally read the same source key/value row twice.");
Require(cuda.MemcpyDevices.All(static current => current == deviceId),
    "Every gathered D2D copy must execute under the configured CUDA ordinal.");
Require(logitsPool.RentCount == 2 && logitsPool.ReturnCount == 2,
    "Dense and gathered ORT runs must both return host logits scratch buffers.");
Require(cuda.CurrentDevice == 9,
    "CUDA gather allocation/copy/free scopes must restore the ambient device.");

row0.Dispose();
row1.Dispose();
Require(cuda.FreeCalls == 4,
    "Source arena must free after both original row owners are released.");
dense[0].State.Dispose();
dense[1].State.Dispose();
Require(cuda.FreeCalls == 6,
    "Dense successor arena must free after both logical row states are released.");
forked[0].State.Dispose();
forked[1].State.Dispose();
Require(cuda.FreeCalls == 8 && cuda.ActiveAllocationPointers.Count == 0,
    "All gather-aware CUDA allocations must be released exactly once.");

Console.WriteLine(
    $"Fission CUDA gather binding specs passed: runs={binding.OrtRunCount}, " +
    $"gatheredBatches={binding.GatheredBatchCount}, gatheredBytes={binding.GatheredBytes}, " +
    $"d2dCopies={cuda.MemcpyCalls.Count}, mallocs={cuda.MallocCalls}, frees={cuda.FreeCalls}.");

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
