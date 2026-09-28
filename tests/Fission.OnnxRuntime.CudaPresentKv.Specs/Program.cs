using System.Runtime.InteropServices;
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

const int deviceId = 3;
const string formatId = "optimum-legacy:fp32:cuda-present-v1";
var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 2,
    numKvHeads: 1,
    headDim: 2,
    vocabularySize: 16);
var cuda = new FakeCudaMemoryApi(initialDevice: 9);
var allocator = new CudaDeviceMemoryAllocator(
    cuda,
    new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
var present = new OptimumLegacyCudaPresentKvBatch(
    profile,
    batchSize: 2,
    pastSequenceLength: 3,
    sequenceLength: 1,
    allocator);

Require(present.Position == 4, "Present-KV transaction must advance the causal position by sequence length.");
Require(present.BatchSize == 2, "Present-KV transaction must preserve cohort batch size.");
Require(present.DeviceId == deviceId, "Present-KV transaction must expose its CUDA ordinal.");
Require(
    present.OutputNames.SequenceEqual(new[]
    {
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value"
    }),
    "Present-KV outputs must use exact Optimum legacy layer names and key/value ordering.");
Require(present.OutputValues.Count == 4, "Two Optimum layers require four caller-owned present-KV outputs.");
Require(cuda.MallocCalls == 4, "Present-KV output transaction must allocate one key/value CUDA buffer per layer.");
Require(cuda.ActiveAllocationPointers.Count == 4, "All present-KV CUDA buffers must remain live while the transaction is open.");

foreach (var value in present.OutputValues)
{
    using var info = value.GetTensorMemoryInfo();
    Require(info.Name == "Cuda" && info.Id == deviceId, "Present-KV outputs must be CUDA OrtValues on the target device.");
    Require(
        value.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 2, 1, 4, 2 }),
        "Present-KV output must expose the full batched [B,H,S,D] shape.");
    Require(value.GetTensorSizeInBytes() == 64, "Two-row present-KV output must expose 64 FP32 bytes.");
}

var appendedNames = new List<string>();
var appendedValues = new List<OrtValue>();
present.AppendOutputs(appendedNames, appendedValues);
Require(appendedNames.SequenceEqual(present.OutputNames), "AppendOutputs must preserve exact output-name order.");
Require(appendedValues.SequenceEqual(present.OutputValues), "AppendOutputs must append the same caller-owned OrtValue instances.");

var allocations = cuda.ActiveAllocationPointers.ToArray();
for (var index = 0; index < allocations.Length; index++)
{
    cuda.WriteFloats(
        allocations[index],
        Enumerable.Range(0, 16)
            .Select(value => (float)(index * 100 + value))
            .ToArray());
}

var row0 = present.CreateRowState(0, nextTokenId: 7);
var row1 = present.CreateRowState(1, nextTokenId: 8);
present.Dispose();
Require(cuda.FreeCalls == 0, "Disposing batched output wrappers must not free CUDA KV while row states retain the arena.");

var disposedValuesRejected = false;
try
{
    _ = present.OutputValues;
}
catch (ObjectDisposedException)
{
    disposedValuesRejected = true;
}
Require(disposedValuesRejected, "Disposed present-KV transaction must not expose dead batched OrtValues.");

var borrow = CudaDecoderOrtCohortArena.AcquireResidentState(formatId, row1);
Require(borrow.Position == 4 && borrow.NextTokenId == 8, "Resident borrow must preserve the row frontier after the output transaction is gone.");
Require(borrow.DeviceId == deviceId, "Resident borrow must preserve the output CUDA ordinal.");
Require(borrow.ByteLength == 128, "Two layers of key/value row tensors must expose 128 resident bytes.");
var layer0 = borrow.GetLayer(0);
var layer1 = borrow.GetLayer(1);
Require(layer0.Key.DevicePointer == allocations[0] + 32, "Row 1 layer 0 key must point one row into the batched output allocation.");
Require(layer0.Value.DevicePointer == allocations[1] + 32, "Row 1 layer 0 value must point one row into the batched output allocation.");
Require(layer1.Key.DevicePointer == allocations[2] + 32, "Row 1 layer 1 key must point one row into the batched output allocation.");
Require(layer1.Value.DevicePointer == allocations[3] + 32, "Row 1 layer 1 value must point one row into the batched output allocation.");
Require(
    cuda.ReadFloats(layer1.Value.DevicePointer, 8).SequenceEqual(
        Enumerable.Range(8, 8).Select(value => (float)(300 + value))),
    "Resident row must observe the exact bytes ORT would have written into its batched present-KV output.");

row1.Dispose();
row0.Dispose();
Require(cuda.FreeCalls == 0, "Resident borrow must independently retain CUDA present-KV after all row states are disposed.");
borrow.Dispose();
Require(cuda.FreeCalls == 4 && cuda.ActiveAllocationPointers.Count == 0, "Final resident borrow release must free all present-KV CUDA allocations.");
Require(cuda.OperationDevices.All(static current => current == deviceId), "Present-KV cudaMalloc/cudaFree operations must run under the configured device ordinal.");
Require(cuda.CurrentDevice == 9, "Present-KV allocation and release must restore the ambient CUDA device.");

var nonFloatProfile = OptimumLegacyDecoderProfile.CreateLlamaLike(
    numHiddenLayers: 1,
    numKvHeads: 1,
    headDim: 2,
    vocabularySize: 16,
    kvElementType: TensorElementType.Double);
var nonFloatRejected = false;
try
{
    _ = new OptimumLegacyCudaPresentKvBatch(
        nonFloatProfile,
        batchSize: 1,
        pastSequenceLength: 0,
        sequenceLength: 1,
        allocator);
}
catch (ArgumentException)
{
    nonFloatRejected = true;
}
Require(nonFloatRejected, "CUDA present-KV transaction must reject non-FP32 KV profiles before allocation.");
Require(cuda.MallocCalls == 4, "Rejected non-FP32 profile must not allocate CUDA memory.");

Console.WriteLine(
    $"Fission Optimum CUDA present-KV specs passed: outputs={appendedNames.Count}, " +
    $"allocations={cuda.MallocCalls}, frees={cuda.FreeCalls}.");

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

    public void WriteFloats(nint pointer, float[] values) =>
        Marshal.Copy(values, 0, pointer, values.Length);

    public float[] ReadFloats(nint pointer, int count)
    {
        var values = new float[count];
        Marshal.Copy(pointer, values, 0, count);
        return values;
    }

    private void RecordDevice()
    {
        lock (_gate)
        {
            _operationDevices.Add(_currentDevice.Value);
        }
    }
}
