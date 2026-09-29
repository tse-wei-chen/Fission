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
var cuda = new FakeCudaMemoryApi(initialDevice: 9);
var allocator = new CudaDeviceMemoryAllocator(
    cuda,
    new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
var logitsPool = new ScriptedFloatPool(new[]
{
    new[] { 10f, 0f, 0f, 0f, 0f, 0f, 0f, 10f },
    new[] { 0f, 10f, 0f, 0f },
    new[] { 0f, 0f, 10f, 0f }
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
    var outputs = outputValues.ToArray();
    Require(names.Contains("past_key_values.0.key"),
        "Every fallback-policy run must bind the Optimum past-key input.");
    Require(outputNames.Contains("present.0.key"),
        "Every fallback-policy run must bind the Optimum present-key output.");

    var pastIndex = Array.IndexOf(names, "past_key_values.0.key");
    using var pastInfo = inputs[pastIndex].GetTensorMemoryInfo();
    Require(pastInfo.Name == "Cuda" && pastInfo.Id == deviceId,
        "Dense and singleton fallback past KV must remain CUDA-resident.");
    var pastShape = inputs[pastIndex].GetTensorTypeAndShape().Shape;

    using var presentInfo = outputs[1].GetTensorMemoryInfo();
    Require(presentInfo.Name == "Cuda" && presentInfo.Id == deviceId,
        "Fallback-policy present KV must be written directly to CUDA memory.");

    if (fakeRunCount == 1)
    {
        Require(pastShape.SequenceEqual(new long[] { 2, 1, 2, 1 }),
            "A complete two-row cohort must remain one dense CUDA batch.");
        Require(inputs[0].GetTensorDataAsSpan<long>().SequenceEqual(new long[] { 1, 2 }),
            "Dense reuse must restore physical CUDA row order before execution.");
    }
    else
    {
        Require(pastShape.SequenceEqual(new long[] { 1, 1, 3, 1 }),
            "Unsupported fork layout must be split into singleton CUDA past inputs.");
        Require(inputs[0].GetTensorDataAsSpan<long>().SequenceEqual(new long[] { 3 }),
            "Each fork fallback run must feed the shared branch token frontier.");
    }
}

using var session = new InferenceSession(Convert.FromBase64String(MulModelBase64));
using var inner = new OptimumLegacyCudaFloatDecoderBinding(
    profile,
    allocator,
    FakeRun,
    scratchFloatPool: logitsPool);
using var binding = new OptimumLegacyCudaSplitFallbackBinding(inner);

// Seed a complete two-row CUDA frontier at position 2. The builder reference is
// released immediately; the two row states become the independent owners.
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

var modelId = new ModelId("cuda-split-fallback");
var firstId = SequenceId.New();
var secondId = SequenceId.New();
var dense = binding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(secondId, modelId, Position: 2),
        new DecodeItem(firstId, modelId, Position: 2)
    },
    new[] { row1, row0 });

Require(dense.Count == 2,
    "Dense CUDA cohort must return one result per logical request.");
Require(dense[0].TokenId == 3 && dense[1].TokenId == 0,
    "Dense physical-row logits must map back to reversed logical request order.");
Require(binding.SingletonFallbackRunCount == 0,
    "Complete dense CUDA cohorts must not pay singleton fallback cost.");
Require(binding.OrtRunCount == 1 && fakeRunCount == 1,
    "Complete dense CUDA cohort must execute exactly one ORT run.");
Require(binding.CudaPastReuseCount == 1,
    "Dense path must reuse the source CUDA past once.");
Require(cuda.MallocCalls == 4,
    "Dense reuse must allocate only the next present key/value pair.");

// Simulate a fork: two logical requests consume the same immutable physical row.
// The strict inner binding would reject this multi-row layout. The policy wrapper
// must split it into two singleton CUDA runs instead of host-packing KV.
var forkA = SequenceId.New();
var forkB = SequenceId.New();
var forked = binding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(forkA, modelId, Position: 3),
        new DecodeItem(forkB, modelId, Position: 3)
    },
    new[] { dense[0].State, dense[0].State });

Require(forked.Count == 2 && forked[0].TokenId == 1 && forked[1].TokenId == 2,
    "Singleton fallback runs must preserve logical request ordering and independent logits.");
Require(!ReferenceEquals(forked[0].State, forked[1].State),
    "Fork fallback must produce independently owned CUDA states.");
Require(forked.All(static result => result.State.Position == 4),
    "Both split fork runs must advance their causal frontier.");
Require(binding.SingletonFallbackRunCount == 2,
    "A two-branch unsupported layout must account for two singleton fallback runs.");
Require(binding.OrtRunCount == 3 && fakeRunCount == 3,
    "Dense execution plus two singleton fallbacks must total three ORT runs.");
Require(binding.CudaPastReuseCount == 3,
    "Dense plus two singleton executions must all reuse CUDA-resident past KV.");
Require(cuda.MallocCalls == 8,
    "Fallback must allocate only one next key/value pair per singleton run; past KV must not be copied.");
Require(logitsPool.RentCount == 3 && logitsPool.ReturnCount == 3,
    "Each dense/singleton run must return its host logits scratch buffer.");

row0.Dispose();
row1.Dispose();
Require(cuda.FreeCalls == 2,
    "Source arena must free after its row owners are released.");
dense[0].State.Dispose();
dense[1].State.Dispose();
Require(cuda.FreeCalls == 4,
    "Dense successor arena must free after both logical rows are released.");
forked[0].State.Dispose();
Require(cuda.FreeCalls == 6,
    "Each singleton fallback state must independently own one CUDA key/value pair.");
forked[1].State.Dispose();
Require(cuda.FreeCalls == 8 && cuda.ActiveAllocationPointers.Count == 0,
    "All split-fallback CUDA allocations must be released exactly once.");
Require(cuda.OperationDevices.All(static current => current == deviceId),
    "All fallback-policy cudaMalloc/cudaFree calls must run on the configured device.");
Require(cuda.CurrentDevice == 9,
    "CUDA operations must restore the ambient device after every allocation/free.");

Console.WriteLine(
    $"Fission CUDA split fallback specs passed: runs={binding.OrtRunCount}, " +
    $"fallbackRuns={binding.SingletonFallbackRunCount}, cudaReuse={binding.CudaPastReuseCount}, " +
    $"mallocs={cuda.MallocCalls}, frees={cuda.FreeCalls}.");

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

sealed class FakeCudaMemoryApi : ICudaDeviceMemoryApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _currentDevice;
    private readonly List<nint> _allocationOrder = new();
    private readonly HashSet<nint> _active = new();
    private readonly List<int> _operationDevices = new();
    private int _mallocCalls;
    private int _freeCalls;

    public FakeCudaMemoryApi(int initialDevice)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
    }

    public int CurrentDevice => _currentDevice.Value;
    public int MallocCalls => Volatile.Read(ref _mallocCalls);
    public int FreeCalls => Volatile.Read(ref _freeCalls);

    public IReadOnlyList<nint> ActiveAllocationPointers
    {
        get
        {
            lock (_gate)
            {
                return _allocationOrder.Where(_active.Contains).ToArray();
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

    public int GetDevice(out int currentDevice)
    {
        currentDevice = _currentDevice.Value;
        return 0;
    }

    public int SetDevice(int deviceId)
    {
        _currentDevice.Value = deviceId;
        return 0;
    }

    public int Malloc(out nint pointer, nuint byteLength)
    {
        RecordDevice();
        Interlocked.Increment(ref _mallocCalls);
        pointer = Marshal.AllocHGlobal(checked((int)byteLength));
        lock (_gate)
        {
            _active.Add(pointer);
            _allocationOrder.Add(pointer);
        }
        return 0;
    }

    public int Free(nint pointer)
    {
        RecordDevice();
        lock (_gate)
        {
            if (!_active.Remove(pointer))
            {
                return 17;
            }
        }

        Marshal.FreeHGlobal(pointer);
        Interlocked.Increment(ref _freeCalls);
        return 0;
    }

    public string? GetErrorString(int errorCode) => $"fake CUDA error {errorCode}";

    private void RecordDevice()
    {
        lock (_gate)
        {
            _operationDevices.Add(_currentDevice.Value);
        }
    }
}
