using System.Buffers;
using System.Runtime.InteropServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

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
var cuda = new FakeCudaMemoryApi(initialDevice: 7);
var allocator = new CudaDeviceMemoryAllocator(
    cuda,
    new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
var logitsPool = new ScriptedFloatPool(
    new[]
    {
        new[] { 0f, 10f, 0f, 0f, 0f, 0f, 10f, 0f }
    });
var decodeLogitsAllocator = new ScriptedHostStagingAllocator(
    new[]
    {
        new[] { 0f, 0f, 10f, 0f, 0f, 0f, 0f, 10f },
        new[] { 10f, 0f, 0f, 0f }
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
    var outNames = outputNames.ToArray();
    var outputs = outputValues.ToArray();

    Require(names.SequenceEqual(new[]
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values.0.key",
        "past_key_values.0.value"
    }), "CUDA binding must preserve exact Optimum input ordering.");
    Require(outNames.SequenceEqual(new[]
    {
        "logits",
        "present.0.key",
        "present.0.value"
    }), "CUDA binding must preserve exact Optimum output ordering.");

    using (var logitsMemory = outputs[0].GetTensorMemoryInfo())
    {
        Require(logitsMemory.Name != "Cuda", "Logits must remain host-backed for greedy sampling.");
    }

    foreach (var output in outputs.Skip(1))
    {
        using var memory = output.GetTensorMemoryInfo();
        Require(memory.Name == "Cuda" && memory.Id == deviceId,
            "Present KV outputs must be bound directly to the configured CUDA device.");
    }

    var inputIds = inputs[0].GetTensorDataAsSpan<long>().ToArray();
    var pastShape = inputs[3].GetTensorTypeAndShape().Shape;
    var presentShape = outputs[1].GetTensorTypeAndShape().Shape;

    switch (runCount)
    {
        case 1:
            Require(inputIds.SequenceEqual(new long[] { 3, 0 }),
                "Initial prefill input rows must preserve request order.");
            Require(pastShape.SequenceEqual(new long[] { 2, 1, 0, 1 }),
                "Initial prefill must bind an empty batched past frontier.");
            Require(presentShape.SequenceEqual(new long[] { 2, 1, 1, 1 }),
                "Initial CUDA present frontier must have one token per row.");
            using (var memory = inputs[3].GetTensorMemoryInfo())
            {
                Require(memory.Name != "Cuda", "Empty initial past may remain host-backed.");
            }
            break;

        case 2:
            Require(inputIds.SequenceEqual(new long[] { 1, 2 }),
                "Reversed logical decode requests must be restored to physical CUDA row order.");
            Require(pastShape.SequenceEqual(new long[] { 2, 1, 1, 1 }),
                "Dense CUDA reuse must expose the complete prior batched frontier.");
            Require(presentShape.SequenceEqual(new long[] { 2, 1, 2, 1 }),
                "Dense decode must advance the present frontier to two tokens.");
            using (var memory = inputs[3].GetTensorMemoryInfo())
            {
                Require(memory.Name == "Cuda" && memory.Id == deviceId,
                    "Dense past KV must remain CUDA-resident with no host pack.");
            }
            break;

        case 3:
            Require(inputIds.SequenceEqual(new long[] { 3 }),
                "Singleton decode must feed the retained next-token frontier.");
            Require(pastShape.SequenceEqual(new long[] { 1, 1, 2, 1 }),
                "Singleton CUDA past must rebind exactly one resident row.");
            Require(presentShape.SequenceEqual(new long[] { 1, 1, 3, 1 }),
                "Singleton decode must advance the frontier to three tokens.");
            using (var memory = inputs[3].GetTensorMemoryInfo())
            {
                Require(memory.Name == "Cuda" && memory.Id == deviceId,
                    "Singleton past KV must remain CUDA-resident.");
            }
            break;

        default:
            throw new InvalidOperationException($"Unexpected fake ORT run {runCount}.");
    }
}

using var session = new InferenceSession(Convert.FromBase64String(MulModelBase64));
using var binding = new OptimumLegacyCudaFloatDecoderBinding(
    profile,
    allocator,
    FakeRun,
    scratchFloatPool: logitsPool,
    decodeLogitsHostAllocator: decodeLogitsAllocator);

Require(binding.DeviceId == deviceId, "CUDA binding must expose its allocator device ordinal.");
Require(binding.CudaResidentStateFormatId.Contains("l1:h1:d1", StringComparison.Ordinal),
    "CUDA resident-state format must encode physical KV geometry.");
Require(binding.PageLockedDecodeLogitsEnabled,
    "Configured decode logits host staging must enable the page-locked decode path.");

var modelId = new ModelId("cuda-binding-spec");
var firstId = SequenceId.New();
var secondId = SequenceId.New();
var initial = binding.ExecutePrefillBatch(
    session,
    new[]
    {
        new PrefillItem(firstId, modelId, new ReadOnlyMemory<int>(new[] { 3 }), Position: 0),
        new PrefillItem(secondId, modelId, new ReadOnlyMemory<int>(new[] { 0 }), Position: 0)
    },
    new DecoderOrtState?[] { null, null });

Require(initial.Count == 2 && initial[0].TokenId == 1 && initial[1].TokenId == 2,
    "Initial CUDA prefill must sample independent row logits.");
Require(initial.All(static result => result.State.Position == 1),
    "Initial CUDA prefill must advance both physical frontiers to one.");
Require(cuda.MallocCalls == 2,
    "One-layer two-row CUDA prefill must allocate only present key/value buffers.");

var decoded = binding.ExecuteDecodeBatch(
    session,
    new[]
    {
        new DecodeItem(secondId, modelId, Position: 1),
        new DecodeItem(firstId, modelId, Position: 1)
    },
    new[] { initial[1].State, initial[0].State });

Require(decoded[0].TokenId == 3 && decoded[1].TokenId == 2,
    "CUDA dense reuse must map physical-row logits back to reversed request order.");
Require(decoded.All(static result => result.State.Position == 2),
    "Dense CUDA decode must advance both frontiers to two.");
Require(binding.CudaPastReuseCount == 1,
    "Dense decode must count one CUDA past reuse.");
Require(cuda.MallocCalls == 4,
    "Dense past reuse must allocate only the next present key/value buffers.");

var runCountBeforeRejectedFork = runCount;
var mallocsBeforeRejectedFork = cuda.MallocCalls;
var duplicateRejected = false;
try
{
    _ = binding.ExecuteDecodeBatch(
        session,
        new[]
        {
            new DecodeItem(SequenceId.New(), modelId, Position: 2),
            new DecodeItem(SequenceId.New(), modelId, Position: 2)
        },
        new[] { decoded[0].State, decoded[0].State });
}
catch (NotSupportedException)
{
    duplicateRejected = true;
}
Require(duplicateRejected,
    "Duplicate multi-row CUDA states must fail instead of silently host-packing or gathering.");
Require(runCount == runCountBeforeRejectedFork && cuda.MallocCalls == mallocsBeforeRejectedFork,
    "Rejected CUDA gather layouts must fail before ORT execution or new device allocation.");

var singleton = binding.ExecuteDecode(
    session,
    new DecodeItem(secondId, modelId, Position: 2),
    decoded[0].State);
Require(singleton.TokenId == 0 && singleton.State.Position == 3,
    "Singleton CUDA row decode must run directly from one resident row.");
Require(binding.CudaPastReuseCount == 2,
    "Singleton decode must count as a retained CUDA past reuse.");
Require(cuda.MallocCalls == 6,
    "Singleton past reuse must allocate only the next present key/value buffers.");
Require(binding.OrtRunCount == 3 && runCount == 3,
    "Prefill, dense decode and singleton decode must each execute exactly one ORT run.");

using var resident = binding.AcquireCudaResidentState(singleton.State);
Require(resident.DeviceId == deviceId && resident.Position == 3 && resident.NextTokenId == 0,
    "Binding resident-state capability must preserve device and causal frontier.");
Require(resident.ByteLength == 24,
    "One FP32 layer with three key/value elements must expose 24 resident bytes.");

initial[0].State.Dispose();
initial[1].State.Dispose();
Require(cuda.FreeCalls == 2,
    "Releasing the first frontier must free exactly its key/value allocations.");
decoded[0].State.Dispose();
decoded[1].State.Dispose();
Require(cuda.FreeCalls == 4,
    "Releasing the second frontier must free its key/value allocations after reuse ends.");
singleton.State.Dispose();
Require(cuda.FreeCalls == 4,
    "Outstanding resident borrow must retain the singleton frontier after state disposal.");
resident.Dispose();
Require(cuda.FreeCalls == 6 && cuda.ActiveAllocationPointers.Count == 0,
    "Final resident borrow release must free every CUDA binding allocation.");
Require(cuda.OperationDevices.All(static current => current == deviceId),
    "CUDA binding allocations and frees must execute under the configured device ordinal.");
Require(cuda.CurrentDevice == 7,
    "CUDA binding allocation/free operations must restore the ambient device.");
Require(logitsPool.RentCount == 1 && logitsPool.ReturnCount == 1,
    "Prefill must keep using ordinary host logits scratch.");
Require(binding.PageLockedDecodeLogitsRentCount == 2,
    "Dense and singleton decode must each rent page-locked host logits staging.");
Require(decodeLogitsAllocator.AllocateCount == 2,
    "Distinct dense and singleton decode shapes must allocate exact-length host staging buffers.");

binding.Dispose();
Require(decodeLogitsAllocator.DisposeCount == 2,
    "Disposing the CUDA binding must release retained decode logits staging buffers.");

Console.WriteLine(
    $"Fission CUDA Optimum binding specs passed: runs={binding.OrtRunCount}, " +
    $"cudaPastReuse={binding.CudaPastReuseCount}, mallocs={cuda.MallocCalls}, frees={cuda.FreeCalls}.");

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

sealed class ScriptedHostStagingAllocator : IHostStagingFloatBufferAllocator
{
    private readonly Queue<float[]> _scripts;
    private int _allocateCount;
    private int _disposeCount;

    public ScriptedHostStagingAllocator(IEnumerable<float[]> scripts)
    {
        _scripts = new Queue<float[]>(scripts.Select(static values => values.ToArray()));
    }

    public int AllocateCount => Volatile.Read(ref _allocateCount);
    public int DisposeCount => Volatile.Read(ref _disposeCount);

    public IHostStagingFloatBuffer Allocate(int length)
    {
        Interlocked.Increment(ref _allocateCount);
        if (_scripts.Count == 0)
        {
            throw new InvalidOperationException("No scripted decode logits remain.");
        }

        var values = _scripts.Dequeue();
        if (values.Length != length)
        {
            throw new InvalidOperationException(
                $"Scripted decode logits length {values.Length} does not match requested {length}.");
        }

        return new ScriptedHostStagingBuffer(values, this);
    }

    private sealed class ScriptedHostStagingBuffer(
        float[] values,
        ScriptedHostStagingAllocator owner) : IHostStagingFloatBuffer
    {
        private float[]? _values = values;

        public Memory<float> Memory =>
            _values ??
            throw new ObjectDisposedException(nameof(ScriptedHostStagingBuffer));

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _values, null) is not null)
            {
                Interlocked.Increment(ref owner._disposeCount);
            }
        }
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
