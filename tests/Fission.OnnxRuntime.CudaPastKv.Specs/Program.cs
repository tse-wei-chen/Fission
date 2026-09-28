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

RunZeroCopyDenseReuse();
RunUnsupportedCohortShapes();

Console.WriteLine("Fission Optimum CUDA past-KV specs passed.");

static OptimumLegacyDecoderProfile CreateProfile() =>
    OptimumLegacyDecoderProfile.CreateLlamaLike(
        numHiddenLayers: 2,
        numKvHeads: 1,
        headDim: 2,
        vocabularySize: 16);

static void RunZeroCopyDenseReuse()
{
    const int deviceId = 4;
    var profile = CreateProfile();
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
    var row0 = present.CreateRowState(0, nextTokenId: 17);
    var row1 = present.CreateRowState(1, nextTokenId: 18);
    present.Dispose();

    Require(cuda.MallocCalls == 4, "Two-layer present cohort must own four CUDA allocations before past reuse.");
    Require(cuda.FreeCalls == 0, "Row states must retain present CUDA allocations after output transaction disposal.");
    var allocationsBeforePast = cuda.ActiveAllocationPointers.ToArray();

    var past = new OptimumLegacyCudaPastKvBatch(
        profile,
        new[] { row0, row1 });

    Require(past.PastSequenceLength == 4, "Past-KV transaction must preserve the row causal position.");
    Require(past.BatchSize == 2, "Past-KV transaction must preserve the complete cohort batch size.");
    Require(past.DeviceId == deviceId, "Past-KV transaction must preserve CUDA device ordinal.");
    Require(
        past.InputNames.SequenceEqual(new[]
        {
            "past_key_values.0.key",
            "past_key_values.0.value",
            "past_key_values.1.key",
            "past_key_values.1.value"
        }),
        "Past-KV inputs must use exact Optimum legacy layer names and key/value ordering.");
    Require(past.InputValues.Count == 4, "Two layers require four caller-owned past-KV input OrtValues.");
    Require(cuda.MallocCalls == 4, "Zero-copy past-KV transaction must not allocate any additional CUDA memory.");
    Require(
        cuda.ActiveAllocationPointers.SequenceEqual(allocationsBeforePast),
        "Zero-copy past-KV transaction must reuse the exact existing CUDA allocations.");

    foreach (var value in past.InputValues)
    {
        using var info = value.GetTensorMemoryInfo();
        Require(info.Name == "Cuda" && info.Id == deviceId, "Past-KV inputs must remain CUDA OrtValues on the cohort device.");
        Require(
            value.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 2, 1, 4, 2 }),
            "Past-KV input must expose the full dense batched [B,H,S,D] shape.");
        Require(value.GetTensorSizeInBytes() == 64, "Past-KV input must expose the existing 64-byte batched allocation.");
    }

    var appendedNames = new List<string>();
    var appendedValues = new List<OrtValue>();
    past.AppendInputs(appendedNames, appendedValues);
    Require(appendedNames.SequenceEqual(past.InputNames), "AppendInputs must preserve exact past-KV name ordering.");
    Require(appendedValues.SequenceEqual(past.InputValues), "AppendInputs must reuse the same caller-owned CUDA OrtValues.");

    row1.Dispose();
    row0.Dispose();
    Require(cuda.FreeCalls == 0, "Past-KV transaction must retain the arena after all row states are disposed.");

    past.Dispose();
    Require(cuda.FreeCalls == 4, "Past-KV transaction final release must free the reused cohort allocations exactly once.");
    Require(cuda.ActiveAllocationPointers.Count == 0, "Past-KV transaction final release must leave no active CUDA allocation.");
    Require(cuda.MallocCalls == 4, "Dense CUDA past reuse must remain allocation-free for its entire lifetime.");
    Require(cuda.OperationDevices.All(static current => current == deviceId), "CUDA allocation/release must stay scoped to the cohort device.");
    Require(cuda.CurrentDevice == 9, "CUDA past-KV lifetime must restore the ambient CUDA device.");
}

static void RunUnsupportedCohortShapes()
{
    var profile = CreateProfile();
    var cuda = new FakeCudaMemoryApi(initialDevice: 6);
    var allocator = new CudaDeviceMemoryAllocator(
        cuda,
        new CudaDeviceMemoryAllocatorOptions { DeviceId = 1 });

    var firstPresent = new OptimumLegacyCudaPresentKvBatch(
        profile,
        batchSize: 2,
        pastSequenceLength: 1,
        sequenceLength: 1,
        allocator);
    var first0 = firstPresent.CreateRowState(0, nextTokenId: 3);
    var first1 = firstPresent.CreateRowState(1, nextTokenId: 4);
    firstPresent.Dispose();

    var subsetRejected = false;
    try
    {
        _ = new OptimumLegacyCudaPastKvBatch(profile, new[] { first0 });
    }
    catch (InvalidOperationException)
    {
        subsetRejected = true;
    }
    Require(subsetRejected, "Zero-copy CUDA past-KV must reject a subset of a dense cohort.");

    var reorderRejected = false;
    try
    {
        _ = new OptimumLegacyCudaPastKvBatch(profile, new[] { first1, first0 });
    }
    catch (InvalidOperationException)
    {
        reorderRejected = true;
    }
    Require(reorderRejected, "Zero-copy CUDA past-KV must reject row reordering instead of silently repacking through host memory.");

    var secondPresent = new OptimumLegacyCudaPresentKvBatch(
        profile,
        batchSize: 2,
        pastSequenceLength: 1,
        sequenceLength: 1,
        allocator);
    var second0 = secondPresent.CreateRowState(0, nextTokenId: 5);
    var second1 = secondPresent.CreateRowState(1, nextTokenId: 6);
    secondPresent.Dispose();

    var mixedRejected = false;
    try
    {
        _ = new OptimumLegacyCudaPastKvBatch(profile, new[] { first0, second1 });
    }
    catch (InvalidOperationException)
    {
        mixedRejected = true;
    }
    Require(mixedRejected, "Zero-copy CUDA past-KV must reject states from different CUDA cohort arenas.");

    var wrongGeometry = OptimumLegacyDecoderProfile.CreateLlamaLike(
        numHiddenLayers: 2,
        numKvHeads: 1,
        headDim: 3,
        vocabularySize: 16);
    var geometryRejected = false;
    try
    {
        _ = new OptimumLegacyCudaPastKvBatch(
            wrongGeometry,
            new[] { first0, first1 });
    }
    catch (InvalidOperationException)
    {
        geometryRejected = true;
    }
    Require(geometryRejected, "Zero-copy CUDA past-KV must reject a profile whose tensor geometry differs from the arena.");

    Require(cuda.MallocCalls == 8, "Rejected CUDA past-KV layouts must not allocate any extra device memory.");

    second1.Dispose();
    second0.Dispose();
    first1.Dispose();
    first0.Dispose();
    Require(cuda.FreeCalls == 8, "Rejected zero-copy attempts must not leak arena references.");
    Require(cuda.ActiveAllocationPointers.Count == 0, "All rejected-layout test arenas must release cleanly.");
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
