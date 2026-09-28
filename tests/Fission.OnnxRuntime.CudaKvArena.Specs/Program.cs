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

RunArenaLifetimeAndSlices();
RunAllocationFailureRollback();

Console.WriteLine("Fission ONNX Runtime CUDA KV arena specs passed.");

static void RunArenaLifetimeAndSlices()
{
    const int targetDevice = 2;
    const string formatId = "cuda-kv:arena-fp32-v1";
    var cuda = new FakeCudaDeviceMemoryApi(initialDevice: 7);
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = targetDevice });
    var sourceShape = new long[] { 1, 1, 3, 2 };
    var arena = new CudaDecoderOrtCohortArena(
        position: 5,
        batchSize: 2,
        layerCount: 2,
        sourceShape,
        allocator);

    sourceShape[2] = 99;
    Require(
        arena.PerSequenceShape.SequenceEqual(new long[] { 1, 1, 3, 2 }),
        "CUDA arena must defensively copy per-sequence shape metadata.");
    Require(
        arena.BatchedShape.SequenceEqual(new long[] { 2, 1, 3, 2 }),
        "CUDA arena must derive batched shape without exposing mutable metadata.");
    Require(arena.PerSequenceByteLength == 24, "One row must contain six FP32 elements = 24 bytes per tensor.");
    Require(arena.BatchedByteLength == 48, "Two rows must contain 48 bytes per batched tensor.");
    Require(cuda.MallocCalls == 4, "Two layers of key/value CUDA KV require four allocations.");
    Require(cuda.ActiveAllocationPointers.Count == 4, "All CUDA cohort allocations must remain live after construction.");
    Require(cuda.OperationDevices.All(static device => device == targetDevice), "Every cudaMalloc must execute on the arena device ordinal.");
    Require(cuda.CurrentDevice == 7, "Arena allocation must restore the caller thread's ambient CUDA device.");

    var pointers = cuda.ActiveAllocationPointers.ToArray();
    cuda.WriteFloats(pointers[0], new[]
    {
        1f, 2f, 3f, 4f, 5f, 6f,
        101f, 102f, 103f, 104f, 105f, 106f
    });
    cuda.WriteFloats(pointers[1], new[]
    {
        11f, 12f, 13f, 14f, 15f, 16f,
        111f, 112f, 113f, 114f, 115f, 116f
    });
    cuda.WriteFloats(pointers[2], new[]
    {
        21f, 22f, 23f, 24f, 25f, 26f,
        121f, 122f, 123f, 124f, 125f, 126f
    });
    cuda.WriteFloats(pointers[3], new[]
    {
        31f, 32f, 33f, 34f, 35f, 36f,
        131f, 132f, 133f, 134f, 135f, 136f
    });

    var batchedLayer = arena.CreateBatchedLayer(0);
    try
    {
        using var keyInfo = batchedLayer.Key.GetTensorMemoryInfo();
        using var valueInfo = batchedLayer.Value.GetTensorMemoryInfo();
        Require(keyInfo.Name == "Cuda" && valueInfo.Name == "Cuda", "Batched arena views must advertise CUDA memory.");
        Require(keyInfo.Id == targetDevice && valueInfo.Id == targetDevice, "Batched arena views must advertise the target CUDA ordinal.");
        Require(
            batchedLayer.Key.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 2, 1, 3, 2 }),
            "Batched key view must expose the full cohort shape.");
        Require(
            batchedLayer.Value.GetTensorSizeInBytes() == 48,
            "Batched value view must expose the full allocation byte length.");
    }
    finally
    {
        batchedLayer.Value.Dispose();
        batchedLayer.Key.Dispose();
    }

    var row0 = arena.CreateRowState(row: 0, nextTokenId: 41);
    var row1 = arena.CreateRowState(row: 1, nextTokenId: 42);
    Require(row0.Position == 5 && row1.Position == 5, "Row states must preserve the cohort causal position.");
    Require(row0.NextTokenId == 41 && row1.NextTokenId == 42, "Row states must preserve their independent next-token frontiers.");

    for (var layer = 0; layer < row0.LayerCount; layer++)
    {
        var rowLayer = row1.GetLayer(layer);
        using var keyInfo = rowLayer.Key.GetTensorMemoryInfo();
        using var valueInfo = rowLayer.Value.GetTensorMemoryInfo();
        Require(keyInfo.Name == "Cuda" && valueInfo.Name == "Cuda", "Row views must remain CUDA OrtValues.");
        Require(keyInfo.Id == targetDevice && valueInfo.Id == targetDevice, "Row views must retain target device metadata.");
        Require(
            rowLayer.Key.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 1, 1, 3, 2 }),
            "Row key view must expose one per-sequence shape.");
        Require(rowLayer.Key.GetTensorSizeInBytes() == 24, "Row key view must expose one row byte length.");
    }

    arena.Release();
    Require(cuda.FreeCalls == 0, "Dropping the builder reference must not free CUDA KV while row states remain alive.");

    var wrongRowRejected = false;
    try
    {
        _ = arena.AcquireResidentState(formatId, row0, row: 1);
    }
    catch (InvalidOperationException)
    {
        wrongRowRejected = true;
    }
    Require(wrongRowRejected, "Resident borrow must reject a state that does not own the requested arena row.");

    var borrow = arena.AcquireResidentState(formatId, row1, row: 1);
    Require(borrow.DeviceId == targetDevice, "Resident borrow must preserve CUDA device ordinal.");
    Require(borrow.Position == 5 && borrow.NextTokenId == 42, "Resident borrow must preserve row causal frontier.");
    Require(borrow.ByteLength == 96, "Two layers of key/value row tensors must expose 96 resident bytes.");

    var layer0 = borrow.GetLayer(0);
    var layer1 = borrow.GetLayer(1);
    Require(layer0.Key.DevicePointer == pointers[0] + 24, "Row 1 layer 0 key pointer must be one row offset into the batched allocation.");
    Require(layer0.Value.DevicePointer == pointers[1] + 24, "Row 1 layer 0 value pointer must be one row offset into the batched allocation.");
    Require(layer1.Key.DevicePointer == pointers[2] + 24, "Row 1 layer 1 key pointer must be one row offset into the batched allocation.");
    Require(layer1.Value.DevicePointer == pointers[3] + 24, "Row 1 layer 1 value pointer must be one row offset into the batched allocation.");
    Require(
        cuda.ReadFloats(layer0.Key.DevicePointer, 6).SequenceEqual(new[] { 101f, 102f, 103f, 104f, 105f, 106f }),
        "Resident row pointer must address row 1 key bytes without repacking.");
    Require(
        cuda.ReadFloats(layer1.Value.DevicePointer, 6).SequenceEqual(new[] { 131f, 132f, 133f, 134f, 135f, 136f }),
        "Resident row pointer must address row 1 value bytes without repacking.");

    row1.Dispose();
    Require(cuda.FreeCalls == 0, "Disposing the borrowed row state must not free CUDA KV while another row and the borrow remain alive.");
    row0.Dispose();
    Require(cuda.FreeCalls == 0, "Disposing the final row state must not free CUDA KV while the migration borrow remains alive.");

    borrow.Dispose();
    Require(arena.IsReleased, "Final resident-borrow release must retire the CUDA arena.");
    Require(cuda.FreeCalls == 4, "Final arena release must cudaFree every layer key/value allocation exactly once.");
    Require(cuda.ActiveAllocationPointers.Count == 0, "Final arena release must leave no active CUDA allocations.");
    Require(cuda.OperationDevices.All(static device => device == targetDevice), "cudaFree must execute under the arena's original CUDA device ordinal.");
    Require(cuda.CurrentDevice == 7, "Final CUDA arena release must restore the ambient CUDA device.");
}

static void RunAllocationFailureRollback()
{
    var cuda = new FakeCudaDeviceMemoryApi(initialDevice: 4)
    {
        FailMallocOnCall = 3,
        MallocFailureCode = 77
    };
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = 1 });

    CudaRuntimeException? observed = null;
    try
    {
        _ = new CudaDecoderOrtCohortArena(
            position: 1,
            batchSize: 2,
            layerCount: 2,
            perSequenceShape: new long[] { 1, 1, 2, 2 },
            allocator);
    }
    catch (CudaRuntimeException exception)
    {
        observed = exception;
    }

    Require(observed is { Operation: "cudaMalloc", ErrorCode: 77 }, "Arena allocation failure must preserve cudaMalloc error details.");
    Require(cuda.MallocCalls == 3, "Failure spec must fail on the configured third cudaMalloc call.");
    Require(cuda.FreeCalls == 2, "Arena construction failure must release every earlier successful CUDA allocation.");
    Require(cuda.ActiveAllocationPointers.Count == 0, "Arena construction failure must leave no CUDA allocation leak.");
    Require(cuda.OperationDevices.All(static device => device == 1), "Allocation rollback must run under the configured CUDA device.");
    Require(cuda.CurrentDevice == 4, "Allocation rollback must restore the caller thread's ambient CUDA device.");
}

sealed class FakeCudaDeviceMemoryApi : ICudaDeviceMemoryApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _currentDevice;
    private readonly List<nint> _allocationOrder = new();
    private readonly HashSet<nint> _activeAllocations = new();
    private readonly List<int> _operationDevices = new();
    private int _mallocCalls;
    private int _freeCalls;

    public FakeCudaDeviceMemoryApi(int initialDevice)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
    }

    public int FailMallocOnCall { get; set; }
    public int MallocFailureCode { get; set; } = 55;
    public int CurrentDevice => _currentDevice.Value;
    public int MallocCalls => Volatile.Read(ref _mallocCalls);
    public int FreeCalls => Volatile.Read(ref _freeCalls);

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
        RecordOperationDevice();
        var call = Interlocked.Increment(ref _mallocCalls);
        if (FailMallocOnCall != 0 && call == FailMallocOnCall)
        {
            pointer = 0;
            return MallocFailureCode;
        }

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
        RecordOperationDevice();
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

    public string? GetErrorString(int errorCode) => $"fake CUDA error {errorCode}";

    public void WriteFloats(nint pointer, float[] values) =>
        Marshal.Copy(values, 0, pointer, values.Length);

    public float[] ReadFloats(nint pointer, int count)
    {
        var values = new float[count];
        Marshal.Copy(pointer, values, 0, count);
        return values;
    }

    private void RecordOperationDevice()
    {
        lock (_gate)
        {
            _operationDevices.Add(_currentDevice.Value);
        }
    }
}
