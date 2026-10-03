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
    "CAMSBmNoZW50YTpwChUKAVgKAVcSAVlaBW11bF8xIgNNdWwSCG11bCB0ZXN0" +
    "KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4K" +
    "DAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==";

const int deviceId = 3;
var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 1,
    numKvHeads: 1,
    headDim: 1,
    vocabularySize: 4);
var cuda = new FakeCudaRuntime(initialDevice: 8) { AutoCompleteEvents = true };
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
    new[] { 0f, 0f, 0f, 0f, 0f, 10f, 0f, 0f },
    new[] { 0f, 0f, 0f, 0f, 0f, 0f, 10f, 0f },
    new[] { 0f, 0f, 0f, 10f, 10f, 0f, 0f, 0f },
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
        "GQA-safe runs must bind Optimum past/present KV names.");

    using (var pastInfo = inputs[pastIndex].GetTensorMemoryInfo())
    using (var presentInfo = outputs[presentIndex].GetTensorMemoryInfo())
    {
        Require(pastInfo.Name == "Cuda" && pastInfo.Id == deviceId,
            "GQA-safe continuation past KV must remain CUDA-resident.");
        Require(presentInfo.Name == "Cuda" && presentInfo.Id == deviceId,
            "GQA-safe continuation present KV must remain CUDA-resident.");
    }

    var pastShape = inputs[pastIndex].GetTensorTypeAndShape().Shape;
    var presentShape = outputs[presentIndex].GetTensorTypeAndShape().Shape;
    var inputIds = inputs[0].GetTensorDataAsSpan<long>().ToArray();
    switch (fakeRunCount)
    {
        case 1:
            Require(pastShape.SequenceEqual(new long[] { 1, 1, 2, 1 }),
                "First continuation row must enter ORT with batch size one and two-token past.");
            Require(presentShape.SequenceEqual(new long[] { 1, 1, 4, 1 }),
                "First continuation row must advance to a four-token present frontier.");
            Require(inputIds.SequenceEqual(new long[] { 10, 11 }),
                "First continuation row must preserve its two-token prompt chunk.");
            Require(cuda.MemcpyCalls.Count == 0,
                "Singleton continuation prefill must not gather dense source rows.");
            break;
        case 2:
            Require(pastShape.SequenceEqual(new long[] { 1, 1, 2, 1 }),
                "Second continuation row must also enter ORT with batch size one.");
            Require(presentShape.SequenceEqual(new long[] { 1, 1, 4, 1 }),
                "Second continuation row must advance independently to position four.");
            Require(inputIds.SequenceEqual(new long[] { 20, 21 }),
                "Second continuation row must preserve its two-token prompt chunk.");
            Require(cuda.MemcpyCalls.Count == 0,
                "GQA continuation safety split must not introduce D2D copies during prefill.");
            break;
        case 3:
            Require(pastShape.SequenceEqual(new long[] { 2, 1, 4, 1 }),
                "First decode must rebuild the singleton frontiers into one dense batch.");
            Require(presentShape.SequenceEqual(new long[] { 2, 1, 5, 1 }),
                "First decode must advance the dense frontier to position five.");
            Require(inputIds.SequenceEqual(new long[] { 1, 2 }),
                "Gathered decode must preserve singleton next-token frontiers by logical row.");
            Require(cuda.MemcpyCalls.Count == 4,
                "One-layer two-row decode recovery must submit exactly four D2D copies.");
            break;
        case 4:
            Require(pastShape.SequenceEqual(new long[] { 2, 1, 5, 1 }),
                "Second decode must reuse the dense successor frontier directly.");
            Require(presentShape.SequenceEqual(new long[] { 2, 1, 6, 1 }),
                "Second decode must remain batched while advancing to position six.");
            Require(inputIds.SequenceEqual(new long[] { 3, 0 }),
                "Second decode must consume the first dense decode's sampled frontiers.");
            Require(cuda.MemcpyCalls.Count == 4,
                "Once decode rebuilds a dense arena, later decode must not gather again.");
            break;
        default:
            throw new InvalidOperationException(
                $"GQA safety regression expected four ORT runs, observed {fakeRunCount}.");
    }
}

using var session = new InferenceSession(Convert.FromBase64String(MulModelBase64));
var inner = new OptimumLegacyCudaFloatDecoderBinding(
    profile,
    allocator,
    FakeRun,
    scratchFloatPool: logitsPool);
var gathering = new OptimumLegacyCudaGatheringBinding(
    profile,
    allocator,
    copyEngine,
    inner);
using var binding = new OptimumLegacyCudaGqaSafeBinding(gathering);

var sourceArena = new CudaDecoderOrtCohortArena(
    position: 2,
    batchSize: 2,
    layerCount: 1,
    perSequenceShape: new long[] { 1, 1, 2, 1 },
    allocator);
var row0 = sourceArena.CreateRowState(0, nextTokenId: 0);
var row1 = sourceArena.CreateRowState(1, nextTokenId: 0);
sourceArena.Release();

var modelId = new ModelId("cuda-gqa-safe");
var sequence0 = SequenceId.New();
var sequence1 = SequenceId.New();
var continuation = await binding.ExecutePrefillBatchAsync(
    session,
    new[]
    {
        new PrefillItem(sequence0, modelId, new[] { 10, 11 }, Position: 2),
        new PrefillItem(sequence1, modelId, new[] { 20, 21 }, Position: 2)
    },
    new DecoderOrtState?[] { row0, row1 });

Require(continuation.Count == 2 &&
        continuation[0].TokenId == 1 &&
        continuation[1].TokenId == 2,
    "GQA-safe continuation must preserve logical result order across singleton runs.");
Require(continuation.All(static result => result.State.Position == 4),
    "Both singleton continuation frontiers must advance to position four.");
Require(binding.GqaContinuationSingletonRunCount == 2,
    "Two-row multi-token continuation must execute exactly two GQA-safe singleton runs.");
Require(binding.SingletonFallbackRunCount == 0,
    "Explicit GQA safety splitting must remain distinct from unsupported-layout fallback.");
Require(binding.GatheredBatchCount == 0 && cuda.MemcpyCalls.Count == 0,
    "Continuation prefill must stay singleton and CUDA-resident without D2D gather.");
Require(binding.OrtRunCount == 2 && fakeRunCount == 2,
    "Continuation prefill must issue one ORT run per row.");

var firstDecode = await binding.ExecuteDecodeBatchAsync(
    session,
    new[]
    {
        new DecodeItem(sequence0, modelId, Position: 4),
        new DecodeItem(sequence1, modelId, Position: 4)
    },
    new[] { continuation[0].State, continuation[1].State });

Require(firstDecode.Count == 2 &&
        firstDecode[0].TokenId == 3 &&
        firstDecode[1].TokenId == 0,
    "First decode after GQA-safe prefill must recover batched logical results.");
Require(binding.GatheredBatchCount == 1 && binding.GatheredBytes == 64,
    "First decode must gather two rows x key/value x four FP32 past elements exactly once.");
Require(cuda.MemcpyCalls.Count == 4 &&
        cuda.MemcpyCalls.All(static call =>
            call.Kind == CudaMemcpyKind.DeviceToDevice && call.ByteLength == 16),
    "Decode recovery must use four 16-byte D2D row copies and no host KV path.");
Require(binding.OrtRunCount == 3 && fakeRunCount == 3,
    "First decode must recover to one batched ORT run.");

var secondDecode = await binding.ExecuteDecodeBatchAsync(
    session,
    new[]
    {
        new DecodeItem(sequence0, modelId, Position: 5),
        new DecodeItem(sequence1, modelId, Position: 5)
    },
    new[] { firstDecode[0].State, firstDecode[1].State });

Require(secondDecode.Count == 2 &&
        secondDecode[0].TokenId == 1 &&
        secondDecode[1].TokenId == 2,
    "Dense successor decode must preserve batched sampling order.");
Require(binding.GatheredBatchCount == 1 && cuda.MemcpyCalls.Count == 4,
    "Steady-state decode must reuse the dense arena without another gather.");
Require(binding.OrtRunCount == 4 && fakeRunCount == 4,
    "Second decode must remain one batched ORT run.");
Require(binding.CudaPastReuseCount == 4,
    "Two singleton prefills plus two batched decodes must all reuse CUDA-resident past state.");
Require(logitsPool.RentCount == 4 && logitsPool.ReturnCount == 4,
    "Every GQA-safe CUDA run must return its host logits scratch lease.");
Require(cuda.CurrentDevice == 8,
    "CUDA allocation and gather scopes must restore the ambient device ordinal.");

row0.Dispose();
row1.Dispose();
continuation[0].State.Dispose();
continuation[1].State.Dispose();
firstDecode[0].State.Dispose();
firstDecode[1].State.Dispose();
secondDecode[0].State.Dispose();
secondDecode[1].State.Dispose();
Require(cuda.ActiveAllocationPointers.Count == 0,
    "All GQA-safe continuation, gather, and dense decode CUDA allocations must be released.");

Console.WriteLine(
    $"Fission CUDA GQA safety specs passed: runs={binding.OrtRunCount}, " +
    $"gqaSingletonRuns={binding.GqaContinuationSingletonRunCount}, " +
    $"gatheredBatches={binding.GatheredBatchCount}, gatheredBytes={binding.GatheredBytes}, " +
    $"d2dCopies={cuda.MemcpyCalls.Count}.");

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

    public FakeCudaRuntime(int initialDevice)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
    }

    public bool AutoCompleteEvents { get; set; }
    public int CurrentDevice => _currentDevice.Value;

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
            _memcpyCalls.Add(new MemcpyCall(
                destination,
                source,
                byteLength,
                kind,
                stream));
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
