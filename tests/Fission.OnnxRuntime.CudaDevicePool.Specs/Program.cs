using System.Runtime.InteropServices;
using Fission.Backends.OnnxRuntime;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

const int deviceId = 3;
var cuda = new FakeCudaDeviceMemoryApi(initialDevice: 9);
using var pool = new CudaPooledDeviceMemoryAllocator(
    cuda,
    new CudaDeviceMemoryPoolOptions
    {
        MaxRetainedBytes = 32,
        MaxRetainedBuffersPerSize = 2
    },
    new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });

var shape = new long[] { 1, 1, 2, 1 };
var firstArena = new CudaDecoderOrtCohortArena(
    position: 2,
    batchSize: 2,
    layerCount: 1,
    perSequenceShape: shape,
    pool);
Require(cuda.MallocCalls == 2,
    "First one-layer CUDA cohort must perform exactly two native allocations.");
Require(pool.Statistics.NativeAllocations == 2 && pool.Statistics.Reuses == 0,
    "First cohort allocations must be recorded as fresh native allocations.");
firstArena.Release();

var afterFirstReturn = pool.Statistics;
Require(cuda.FreeCalls == 0,
    "Returning a cohort to an in-budget pool must not call cudaFree.");
Require(afterFirstReturn.Returns == 2 && afterFirstReturn.Drops == 0 &&
        afterFirstReturn.RetainedBuffers == 2 && afterFirstReturn.RetainedBytes == 32,
    "First cohort must retain its exact two 16-byte layer allocations.");

var secondArena = new CudaDecoderOrtCohortArena(
    position: 2,
    batchSize: 2,
    layerCount: 1,
    perSequenceShape: shape,
    pool);
Require(cuda.MallocCalls == 2,
    "Second same-shape cohort must reuse both retained CUDA allocations without cudaMalloc.");
Require(pool.Statistics.Reuses == 2 && pool.Statistics.RetainedBuffers == 0,
    "Second cohort must rent both retained exact-size buffers.");
secondArena.Release();
Require(pool.Statistics.RetainedBuffers == 2 && pool.Statistics.RetainedBytes == 32,
    "Second cohort release must return both buffers to the pool again.");

using (var differentSize = pool.Allocate(8))
{
    Require(cuda.MallocCalls == 3,
        "A different byte length must never reuse a retained 16-byte allocation.");
    Require(differentSize.ByteLength == 8 && differentSize.DeviceId == deviceId,
        "Different-size allocation must preserve exact geometry and device identity.");
}
Require(cuda.FreeCalls == 1,
    "Different-size return must be dropped when the retained-byte budget is already full.");
Require(pool.Statistics.Drops == 1 && pool.Statistics.RetainedBytes == 32,
    "Budget pressure must drop rather than over-retain CUDA memory.");

var firstReuse = pool.Allocate(16);
var secondReuse = pool.Allocate(16);
var overflow = pool.Allocate(16);
Require(cuda.MallocCalls == 4 && pool.Statistics.Reuses == 4,
    "Two exact buffers must be reused and a third simultaneous request must allocate natively.");
firstReuse.Dispose();
secondReuse.Dispose();
overflow.Dispose();
var afterOverflow = pool.Statistics;
Require(afterOverflow.Returns == 8 && afterOverflow.Drops == 2 &&
        afterOverflow.RetainedBuffers == 2 && afterOverflow.RetainedBytes == 32,
    "Per-size/budget bounds must retain exactly two 16-byte allocations and drop the overflow.");
Require(cuda.FreeCalls == 2,
    "Only the different-size and third-overflow allocations should have been freed so far.");

pool.Dispose();
Require(cuda.FreeCalls == 4 && cuda.ActivePointers.Count == 0,
    "Pool disposal must synchronously cudaFree every retained handle exactly once.");
Require(pool.Statistics.RetainedBuffers == 0 && pool.Statistics.RetainedBytes == 0,
    "Disposed pool must report no retained device memory.");

var allocateAfterDisposeFailed = false;
try
{
    using var unexpected = pool.Allocate(16);
}
catch (ObjectDisposedException)
{
    allocateAfterDisposeFailed = true;
}
Require(allocateAfterDisposeFailed,
    "Disposed CUDA pool must reject new allocations.");

using var secondPool = new CudaPooledDeviceMemoryAllocator(
    cuda,
    new CudaDeviceMemoryPoolOptions
    {
        MaxRetainedBytes = 16,
        MaxRetainedBuffersPerSize = 1
    },
    new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
var outstanding = secondPool.Allocate(16);
Require(cuda.MallocCalls == 5,
    "A separate pool must own an independent native allocation.");
secondPool.Dispose();
Require(cuda.FreeCalls == 4,
    "Disposing a pool must not free an allocation still leased to a caller.");
outstanding.Dispose();
Require(cuda.FreeCalls == 5 && cuda.ActivePointers.Count == 0,
    "An outstanding lease returned after pool disposal must cudaFree instead of being retained.");
Require(secondPool.Statistics.Returns == 1 && secondPool.Statistics.Drops == 1,
    "Post-disposal return must be observable as a dropped buffer.");

Require(cuda.MallocDevices.All(static current => current == deviceId) &&
        cuda.FreeDevices.All(static current => current == deviceId),
    "Every pooled cudaMalloc/cudaFree must execute under the configured device ordinal.");
Require(cuda.CurrentDevice == 9,
    "Pooled allocation and release must restore the caller's ambient CUDA device.");

Console.WriteLine(
    $"Fission CUDA device pool specs passed: native={pool.Statistics.NativeAllocations}, " +
    $"reuse={pool.Statistics.Reuses}, returns={pool.Statistics.Returns}, " +
    $"drops={pool.Statistics.Drops}, mallocs={cuda.MallocCalls}, frees={cuda.FreeCalls}.");

sealed class FakeCudaDeviceMemoryApi : ICudaDeviceMemoryApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _currentDevice;
    private readonly HashSet<nint> _activePointers = new();
    private readonly List<int> _mallocDevices = new();
    private readonly List<int> _freeDevices = new();
    private int _mallocCalls;
    private int _freeCalls;

    public FakeCudaDeviceMemoryApi(int initialDevice)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
    }

    public int CurrentDevice => _currentDevice.Value;
    public int MallocCalls => Volatile.Read(ref _mallocCalls);
    public int FreeCalls => Volatile.Read(ref _freeCalls);

    public IReadOnlyList<nint> ActivePointers
    {
        get
        {
            lock (_gate)
            {
                return _activePointers.ToArray();
            }
        }
    }

    public IReadOnlyList<int> MallocDevices
    {
        get
        {
            lock (_gate)
            {
                return _mallocDevices.ToArray();
            }
        }
    }

    public IReadOnlyList<int> FreeDevices
    {
        get
        {
            lock (_gate)
            {
                return _freeDevices.ToArray();
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
            _activePointers.Add(pointer);
            _mallocDevices.Add(_currentDevice.Value);
        }

        return 0;
    }

    public int Free(nint pointer)
    {
        lock (_gate)
        {
            if (!_activePointers.Remove(pointer))
            {
                return 17;
            }

            _freeDevices.Add(_currentDevice.Value);
        }

        Marshal.FreeHGlobal(pointer);
        Interlocked.Increment(ref _freeCalls);
        return 0;
    }

    public string? GetErrorString(int errorCode) => $"fake CUDA error {errorCode}";
}
