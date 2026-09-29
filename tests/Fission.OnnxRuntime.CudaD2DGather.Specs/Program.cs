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

const int deviceId = 5;
const int ambientDevice = 11;
const int position = 2;
const string formatId = "spec:fp32:l2:h1:d1:cuda-v1";
var geometry = new DecoderOrtGeometry(
    numHiddenLayers: 2,
    numKvHeads: 1,
    headDim: 1,
    vocabularySize: 16,
    kvElementType: TensorElementType.Float);
var cuda = new FakeCudaRuntime(ambientDevice);
var allocator = new CudaDeviceMemoryAllocator(
    cuda,
    new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
    deviceId,
    cuda,
    cuda,
    new CudaAsyncCopyEngineOptions
    {
        StreamCount = 3,
        CompletionPollInterval = TimeSpan.FromMilliseconds(1)
    });
var gatherer = new CudaDecoderOrtStateGatherer(
    geometry,
    formatId,
    allocator,
    copyEngine);

var sourceArena = new CudaDecoderOrtCohortArena(
    position,
    batchSize: 3,
    layerCount: geometry.NumHiddenLayers,
    perSequenceShape: geometry.GetPastKvShape(1, position),
    allocator);
var source = new[]
{
    sourceArena.CreateRowState(0, nextTokenId: 10),
    sourceArena.CreateRowState(1, nextTokenId: 11),
    sourceArena.CreateRowState(2, nextTokenId: 12)
};
sourceArena.Release();
Require(cuda.MallocCalls == 4,
    "Two-layer source cohort must allocate four CUDA key/value buffers.");

for (var row = 0; row < source.Length; row++)
{
    using var lease = source[row].AcquireCudaResidentState(formatId);
    for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
    {
        var resident = lease.GetLayer(layer);
        cuda.WriteFloats(
            resident.Key.DevicePointer,
            ExpectedRow(row, layer, valueSlot: false));
        cuda.WriteFloats(
            resident.Value.DevicePointer,
            ExpectedRow(row, layer, valueSlot: true));
    }
}

// Validation must complete before target allocation. A CPU state has the right
// causal geometry but no CUDA resident-state source, so the gather must fail with
// zero new cudaMalloc/cudaMemcpy activity.
using (var cpuState = CreateCpuState(geometry, position, nextTokenId: 7))
{
    var rejected = false;
    try
    {
        _ = await gatherer.GatherAsync(new[] { source[0], cpuState });
    }
    catch (InvalidOperationException)
    {
        rejected = true;
    }

    Require(rejected,
        "Non-CUDA source rows must be rejected before target allocation.");
    Require(cuda.MallocCalls == 4 && cuda.MemcpyCalls.Count == 0,
        "Invalid source validation must not allocate or submit D2D copies.");
}

// Reorder and duplicate/fork in one gather: target rows become [source2, source0,
// source2]. This is exactly the layout the split fallback would otherwise execute
// as multiple singleton ORT runs.
using var gathered = await gatherer.GatherAsync(
    new[] { source[2], source[0], source[2] });
Require(gathered.Position == position && gathered.DeviceId == deviceId,
    "Gathered batch must preserve causal position and CUDA device.");
Require(gathered.BatchSize == 3,
    "Gathered batch must preserve requested logical row count including duplicates.");
Require(gathered.States.Select(static state => state.NextTokenId)
    .SequenceEqual(new int?[] { 12, 10, 12 }),
    "Gather must preserve each requested row's next-token frontier.");
Require(cuda.MallocCalls == 8,
    "Gather target must allocate one batched key/value pair per layer, independent of row count.");
Require(cuda.MemcpyCalls.Count == 12,
    "Three rows × two layers × key/value requires twelve D2D copies.");
Require(cuda.MemcpyCalls.All(static call =>
    call.Kind == CudaMemcpyKind.DeviceToDevice && call.ByteLength == 8),
    "Gather must use only eight-byte FP32 device-to-device row copies.");

CudaDecoderOrtCohortArena? gatheredArena = null;
for (var row = 0; row < gathered.States.Count; row++)
{
    Require(gathered.States[row].TryGetCudaCohortSlice(out var slice),
        "Every gathered state must be a row of one dense CUDA cohort.");
    Require(slice.Row == row,
        "Gathered target rows must be materialized in requested logical order.");
    if (gatheredArena is null)
    {
        gatheredArena = slice.Arena;
    }
    else
    {
        Require(ReferenceEquals(gatheredArena, slice.Arena),
            "Every gathered row must share the same dense target arena.");
    }
}

// Source ownership is no longer needed after GatherAsync completion. Releasing all
// source rows must free only the four source allocations; target bytes remain live.
foreach (var state in source)
{
    state.Dispose();
}
Require(cuda.FreeCalls == 4,
    "Completed gather must release every source retain so source arena can free independently.");

var sourceRowByTarget = new[] { 2, 0, 2 };
for (var row = 0; row < gathered.States.Count; row++)
{
    using var lease = gathered.States[row].AcquireCudaResidentState(formatId);
    Require(lease.DeviceId == deviceId && lease.Position == position,
        "Gathered row resident lease must preserve device and position.");
    for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
    {
        var resident = lease.GetLayer(layer);
        Require(cuda.ReadFloats(resident.Key.DevicePointer, 2)
            .SequenceEqual(ExpectedRow(sourceRowByTarget[row], layer, valueSlot: false)),
            $"Gathered row {row} layer {layer} key bytes must match selected source row.");
        Require(cuda.ReadFloats(resident.Value.DevicePointer, 2)
            .SequenceEqual(ExpectedRow(sourceRowByTarget[row], layer, valueSlot: true)),
            $"Gathered row {row} layer {layer} value bytes must match selected source row.");
    }
}

gathered.Dispose();
Require(cuda.FreeCalls == 8 && cuda.ActiveAllocationCount == 0,
    "Disposing gathered batch must free every target CUDA allocation exactly once.");
Require(cuda.OperationDevices.All(static current => current == deviceId),
    "All cudaMalloc/cudaFree/cudaMemcpyAsync operations must run on the configured device.");
Require(cuda.CurrentDevice == ambientDevice,
    "Device-scoped gather operations must restore the ambient CUDA device.");
Require(copyEngine.ActiveCopies == 0 && copyEngine.PendingCompletions == 0,
    "Gather completion must leave no in-flight copy or completion-event state.");

Console.WriteLine(
    $"Fission CUDA D2D gather specs passed: copies={cuda.MemcpyCalls.Count}, " +
    $"mallocs={cuda.MallocCalls}, frees={cuda.FreeCalls}.");

static float[] ExpectedRow(int row, int layer, bool valueSlot)
{
    var baseValue = (row * 1000f) + (layer * 100f) + (valueSlot ? 50f : 0f);
    return new[] { baseValue + 1f, baseValue + 2f };
}

static DecoderOrtState CreateCpuState(
    DecoderOrtGeometry geometry,
    int position,
    int nextTokenId)
{
    var shape = geometry.GetPastKvShape(1, position);
    var length = checked((int)shape.Aggregate(1L, static (count, dimension) => count * dimension));
    var layers = new DecoderOrtLayerState[geometry.NumHiddenLayers];
    var owned = new List<OrtValue>(geometry.NumHiddenLayers * 2);
    try
    {
        for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
        {
            var key = OrtValue.CreateTensorValueFromMemory(new float[length], shape);
            var value = OrtValue.CreateTensorValueFromMemory(new float[length], shape);
            owned.Add(key);
            owned.Add(value);
            layers[layer] = new DecoderOrtLayerState(key, value);
        }

        var state = new DecoderOrtState(position, layers, nextTokenId);
        owned.Clear();
        return state;
    }
    catch
    {
        for (var index = owned.Count - 1; index >= 0; index--)
        {
            owned[index].Dispose();
        }
        throw;
    }
}

sealed class FakeCudaRuntime :
    ICudaDeviceMemoryApi,
    ICudaAsyncCopyApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _currentDevice;
    private readonly HashSet<nint> _activeAllocations = new();
    private readonly List<int> _operationDevices = new();
    private readonly List<MemcpyCall> _memcpyCalls = new();
    private long _nextHandle = 100;
    private int _mallocCalls;
    private int _freeCalls;

    public FakeCudaRuntime(int initialDevice)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
    }

    public int CurrentDevice => _currentDevice.Value;
    public int MallocCalls => Volatile.Read(ref _mallocCalls);
    public int FreeCalls => Volatile.Read(ref _freeCalls);

    public int ActiveAllocationCount
    {
        get
        {
            lock (_gate)
            {
                return _activeAllocations.Count;
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
        Interlocked.Increment(ref _mallocCalls);
        pointer = Marshal.AllocHGlobal(checked((int)byteLength));
        lock (_gate)
        {
            _activeAllocations.Add(pointer);
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
        _ = stream;
        return 0;
    }

    public int MemcpyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream)
    {
        _ = stream;
        RecordOperationDevice();
        var length = checked((int)byteLength);
        var bytes = new byte[length];
        Marshal.Copy(source, bytes, 0, length);
        Marshal.Copy(bytes, 0, destination, length);
        lock (_gate)
        {
            _memcpyCalls.Add(new MemcpyCall(kind, byteLength));
        }
        return 0;
    }

    public int EventCreateWithFlags(out nint completionEvent, uint flags)
    {
        _ = flags;
        completionEvent = (nint)Interlocked.Increment(ref _nextHandle);
        return 0;
    }

    public int EventRecord(nint completionEvent, nint stream)
    {
        _ = completionEvent;
        _ = stream;
        return 0;
    }

    public int EventQuery(nint completionEvent)
    {
        _ = completionEvent;
        return 0;
    }

    public int EventDestroy(nint completionEvent)
    {
        _ = completionEvent;
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

    public readonly record struct MemcpyCall(
        CudaMemcpyKind Kind,
        nuint ByteLength);
}
