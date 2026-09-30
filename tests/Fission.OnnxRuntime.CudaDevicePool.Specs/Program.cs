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
var firstActive = pool.Statistics;
Require(firstActive.NativeAllocations == 2 && firstActive.Reuses == 0,
    "First cohort allocations must be recorded as fresh native allocations.");
Require(firstActive.ActiveBuffers == 2 && firstActive.ActiveBytes == 32 &&
        firstActive.RetainedBuffers == 0 && firstActive.ReservedBytes == 32,
    "Fresh cohort allocations must be visible as active CUDA residency.");
Require(firstActive.PeakActiveBytes == 32 && firstActive.PeakReservedBytes == 32,
    "First cohort must establish the initial active/reserved high-water marks.");
firstArena.Release();

var afterFirstReturn = pool.Statistics;
Require(cuda.FreeCalls == 0,
    "Returning a cohort to an in-budget pool must not call cudaFree.");
Require(afterFirstReturn.Returns == 2 && afterFirstReturn.Drops == 0 &&
        afterFirstReturn.RetainedBuffers == 2 && afterFirstReturn.RetainedBytes == 32,
    "First cohort must retain its exact two 16-byte layer allocations.");
Require(afterFirstReturn.ActiveBuffers == 0 && afterFirstReturn.ActiveBytes == 0 &&
        afterFirstReturn.ReservedBytes == 32,
    "Returned buffers must move from active to retained without changing reserved CUDA bytes.");

var secondArena = new CudaDecoderOrtCohortArena(
    position: 2,
    batchSize: 2,
    layerCount: 1,
    perSequenceShape: shape,
    pool);
Require(cuda.MallocCalls == 2,
    "Second same-shape cohort must reuse both retained CUDA allocations without cudaMalloc.");
var duringReuse = pool.Statistics;
Require(duringReuse.Reuses == 2 && duringReuse.RetainedBuffers == 0,
    "Second cohort must rent both retained exact-size buffers.");
Require(duringReuse.ActiveBuffers == 2 && duringReuse.ActiveBytes == 32 &&
        duringReuse.ReservedBytes == 32,
    "Reused allocations must become active without changing total reserved bytes.");
secondArena.Release();
Require(pool.Statistics.RetainedBuffers == 2 && pool.Statistics.RetainedBytes == 32,
    "Second cohort release must return both buffers to the pool again.");

using (var differentSize = pool.Allocate(8))
{
    Require(cuda.MallocCalls == 3,
        "A different byte length must never reuse a retained 16-byte allocation.");
    Require(differentSize.ByteLength == 8 && differentSize.DeviceId == deviceId,
        "Different-size allocation must preserve exact geometry and device identity.");
    var mixedResidency = pool.Statistics;
    Require(mixedResidency.ActiveBytes == 8 && mixedResidency.RetainedBytes == 32 &&
            mixedResidency.ReservedBytes == 40 && mixedResidency.PeakReservedBytes == 40,
        "Pool statistics must include simultaneously active and retained CUDA residency.");
}
Require(cuda.FreeCalls == 1,
    "Different-size return must be dropped when the retained-byte budget is already full.");
Require(pool.Statistics.Drops == 1 && pool.Statistics.RetainedBytes == 32,
    "Budget pressure must drop rather than over-retain CUDA memory.");

var firstReuse = pool.Allocate(16);
var secondReuse = pool.Allocate(16);
var overflow = pool.Allocate(16);
var peakConcurrency = pool.Statistics;
Require(cuda.MallocCalls == 4 && peakConcurrency.Reuses == 4,
    "Two exact buffers must be reused and a third simultaneous request must allocate natively.");
Require(peakConcurrency.ActiveBuffers == 3 && peakConcurrency.ActiveBytes == 48 &&
        peakConcurrency.ReservedBytes == 48 &&
        peakConcurrency.PeakActiveBytes == 48 && peakConcurrency.PeakReservedBytes == 48,
    "Pool high-water statistics must capture simultaneous active CUDA allocations.");
firstReuse.Dispose();
secondReuse.Dispose();
overflow.Dispose();
var afterOverflow = pool.Statistics;
Require(afterOverflow.Returns == 8 && afterOverflow.Drops == 2 &&
        afterOverflow.RetainedBuffers == 2 && afterOverflow.RetainedBytes == 32,
    "Per-size/budget bounds must retain exactly two 16-byte allocations and drop the overflow.");
Require(afterOverflow.ActiveBuffers == 0 && afterOverflow.ActiveBytes == 0 &&
        afterOverflow.ReservedBytes == 32,
    "All returned leases must leave only idle-retained residency.");
Require(cuda.FreeCalls == 2,
    "Only the different-size and third-overflow allocations should have been freed so far.");

var partialTrim = pool.TrimRetained(targetRetainedBytes: 16);
Require(partialTrim.ReleasedBuffers == 1 && partialTrim.ReleasedBytes == 16 &&
        partialTrim.RetainedBuffers == 1 && partialTrim.RetainedBytes == 16,
    "Targeted trim must synchronously release enough largest idle buffers to meet the target.");
Require(cuda.FreeCalls == 3,
    "Targeted trim must cudaFree the released idle allocation.");
var afterPartialTrim = pool.Statistics;
Require(afterPartialTrim.TrimmedBuffers == 1 && afterPartialTrim.TrimmedBytes == 16 &&
        afterPartialTrim.ReservedBytes == 16,
    "Trim accounting must track cumulative explicitly reclaimed CUDA residency.");

var noOpTrim = pool.TrimRetained(targetRetainedBytes: 16);
Require(noOpTrim.ReleasedBuffers == 0 && noOpTrim.ReleasedBytes == 0 &&
        noOpTrim.RetainedBytes == 16 && cuda.FreeCalls == 3,
    "Trimming to the current retained size must be a no-op.");

var fullTrim = pool.TrimRetained();
Require(fullTrim.ReleasedBuffers == 1 && fullTrim.ReleasedBytes == 16 &&
        fullTrim.RetainedBuffers == 0 && fullTrim.RetainedBytes == 0,
    "Default trim must release all currently idle retained allocations.");
Require(cuda.FreeCalls == 4 && pool.Statistics.ReservedBytes == 0,
    "Full trim must return idle CUDA residency to the device immediately.");
Require(pool.Statistics.TrimmedBuffers == 2 && pool.Statistics.TrimmedBytes == 32,
    "Cumulative trim counters must include partial and full trims.");

pool.Dispose();
Require(cuda.FreeCalls == 4 && cuda.ActivePointers.Count == 0,
    "Pool disposal after a full trim must not double-free released handles.");
Require(pool.Statistics.RetainedBuffers == 0 && pool.Statistics.RetainedBytes == 0 &&
        pool.Statistics.ActiveBuffers == 0 && pool.Statistics.ReservedBytes == 0,
    "Disposed pool must report no active or retained device memory.");

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
var activeSecondPool = secondPool.Statistics;
Require(activeSecondPool.ActiveBuffers == 1 && activeSecondPool.ActiveBytes == 16 &&
        activeSecondPool.RetainedBytes == 0 && activeSecondPool.ReservedBytes == 16,
    "Outstanding caller leases must be visible as active reserved CUDA memory.");
var activeTrim = secondPool.TrimRetained();
Require(activeTrim.ReleasedBuffers == 0 && activeTrim.ReleasedBytes == 0 &&
        cuda.FreeCalls == 4,
    "Trim must never revoke or free an active caller lease.");
secondPool.Dispose();
Require(cuda.FreeCalls == 4 && secondPool.Statistics.ReservedBytes == 16,
    "Disposing a pool must not free or hide an allocation still leased to a caller.");
outstanding.Dispose();
Require(cuda.FreeCalls == 5 && cuda.ActivePointers.Count == 0,
    "An outstanding lease returned after pool disposal must cudaFree instead of being retained.");
Require(secondPool.Statistics.Returns == 1 && secondPool.Statistics.Drops == 1 &&
        secondPool.Statistics.ActiveBuffers == 0 && secondPool.Statistics.ReservedBytes == 0,
    "Post-disposal return must clear active residency and be observable as a dropped buffer.");

using var mallocEntered = new ManualResetEventSlim(initialState: false);
using var continueMalloc = new ManualResetEventSlim(initialState: false);
var racingCuda = new FakeCudaDeviceMemoryApi(
    initialDevice: 9,
    mallocEntered,
    continueMalloc);
using var racingPool = new CudaPooledDeviceMemoryAllocator(
    racingCuda,
    new CudaDeviceMemoryPoolOptions
    {
        MaxRetainedBytes = 64,
        MaxRetainedBuffersPerSize = 2
    },
    new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
var raceTask = Task.Run(() =>
{
    try
    {
        using var unexpected = racingPool.Allocate(24);
        return false;
    }
    catch (ObjectDisposedException)
    {
        return true;
    }
});
var enteredMalloc = mallocEntered.Wait(TimeSpan.FromSeconds(5));
if (!enteredMalloc)
{
    continueMalloc.Set();
}
Require(enteredMalloc,
    "Allocate/dispose race spec must observe the native allocation in flight.");
racingPool.Dispose();
continueMalloc.Set();
Require(raceTask.GetAwaiter().GetResult(),
    "A native allocation completing after pool disposal must not escape as a caller lease.");
Require(racingCuda.MallocCalls == 1 && racingCuda.FreeCalls == 1 &&
        racingCuda.ActivePointers.Count == 0,
    "A post-disposal native allocation must be synchronously released exactly once.");
var afterRace = racingPool.Statistics;
Require(afterRace.NativeAllocations == 1 && afterRace.ActiveBuffers == 0 &&
        afterRace.RetainedBuffers == 0 && afterRace.ReservedBytes == 0,
    "Rejected post-disposal allocation must not appear as active or retained residency.");

Require(cuda.MallocDevices.All(static current => current == deviceId) &&
        cuda.FreeDevices.All(static current => current == deviceId),
    "Every pooled cudaMalloc/cudaFree must execute under the configured device ordinal.");
Require(cuda.CurrentDevice == 9,
    "Pooled allocation and release must restore the caller's ambient CUDA device.");
Require(racingCuda.MallocDevices.All(static current => current == deviceId) &&
        racingCuda.FreeDevices.All(static current => current == deviceId),
    "Allocate/dispose race cleanup must preserve explicit CUDA device scoping.");

Console.WriteLine(
    $"Fission CUDA device pool specs passed: native={pool.Statistics.NativeAllocations}, " +
    $"reuse={pool.Statistics.Reuses}, returns={pool.Statistics.Returns}, " +
    $"drops={pool.Statistics.Drops}, trimmed={pool.Statistics.TrimmedBytes}B, " +
    $"peak-reserved={pool.Statistics.PeakReservedBytes}B, " +
    $"mallocs={cuda.MallocCalls}, frees={cuda.FreeCalls}. Race cleanup verified.");

sealed class FakeCudaDeviceMemoryApi : ICudaDeviceMemoryApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _currentDevice;
    private readonly HashSet<nint> _activePointers = new();
    private readonly List<int> _mallocDevices = new();
    private readonly List<int> _freeDevices = new();
    private readonly ManualResetEventSlim? _mallocEntered;
    private readonly ManualResetEventSlim? _continueMalloc;
    private int _mallocCalls;
    private int _freeCalls;

    public FakeCudaDeviceMemoryApi(
        int initialDevice,
        ManualResetEventSlim? mallocEntered = null,
        ManualResetEventSlim? continueMalloc = null)
    {
        _currentDevice = new ThreadLocal<int>(() => initialDevice);
        _mallocEntered = mallocEntered;
        _continueMalloc = continueMalloc;
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
        _mallocEntered?.Set();
        _continueMalloc?.Wait();

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
