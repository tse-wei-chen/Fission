using System.Runtime.InteropServices;
using Fission.Abstractions.Execution;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Configuration for device-wide CUDA memory pressure observation.
/// </summary>
public sealed record CudaDeviceMemoryPressureMonitorOptions
{
    public int DeviceId { get; init; }
    public string? RuntimeLibraryPath { get; init; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegative(DeviceId);
        if (RuntimeLibraryPath is not null && string.IsNullOrWhiteSpace(RuntimeLibraryPath))
        {
            throw new ArgumentException(
                "CUDA Runtime library path must be non-empty when specified.",
                nameof(RuntimeLibraryPath));
        }
    }
}

/// <summary>
/// One cudaMemGetInfo sample for a physical CUDA device.
/// </summary>
public readonly record struct CudaDeviceMemoryInfo(
    long FreeBytes,
    long TotalBytes)
{
    public long UsedBytes => checked(TotalBytes - FreeBytes);
}

/// <summary>
/// Device-wide physical-memory pressure backed by cudaMemGetInfo.
///
/// The device-wide used byte count already includes allocator pools, model weights,
/// ONNX Runtime workspaces, CUDA Graph captures, and allocations made by other
/// contexts/processes. An optional known source is therefore used only to classify
/// part of those already-counted bytes as reclaimable; its reserved/active bytes are
/// never added again. This avoids double-counting a CUDA pool while still allowing
/// idle pool residency to participate in runtime reclaim.
/// </summary>
public sealed class CudaDeviceMemoryPressureMonitor :
    IInferenceDeviceMemoryPressureSource,
    IInferenceDeviceMemoryReclaimer
{
    private readonly ICudaDeviceMemoryInfoApi _cuda;
    private readonly IInferenceDeviceMemoryPressureSource? _knownReclaimableSource;
    private readonly IInferenceDeviceMemoryReclaimer? _reclaimer;
    private long _peakReservedBytes;

    public CudaDeviceMemoryPressureMonitor(
        CudaDeviceMemoryPressureMonitorOptions? options = null,
        IInferenceDeviceMemoryPressureSource? knownReclaimableSource = null)
        : this(
            NativeCudaDeviceMemoryInfoApi.Load(options?.RuntimeLibraryPath),
            options ?? new CudaDeviceMemoryPressureMonitorOptions(),
            knownReclaimableSource)
    {
    }

    internal CudaDeviceMemoryPressureMonitor(
        ICudaDeviceMemoryInfoApi cuda,
        CudaDeviceMemoryPressureMonitorOptions? options = null,
        IInferenceDeviceMemoryPressureSource? knownReclaimableSource = null)
    {
        ArgumentNullException.ThrowIfNull(cuda);
        options ??= new CudaDeviceMemoryPressureMonitorOptions();
        options.Validate();

        _cuda = cuda;
        _knownReclaimableSource = knownReclaimableSource;
        _reclaimer = knownReclaimableSource as IInferenceDeviceMemoryReclaimer;
        DeviceId = options.DeviceId;
    }

    public int DeviceId { get; }

    public static bool TryCreate(
        out CudaDeviceMemoryPressureMonitor? monitor,
        CudaDeviceMemoryPressureMonitorOptions? options = null,
        IInferenceDeviceMemoryPressureSource? knownReclaimableSource = null)
    {
        try
        {
            monitor = new CudaDeviceMemoryPressureMonitor(
                options,
                knownReclaimableSource);
            return true;
        }
        catch (Exception exception) when (exception is
            DllNotFoundException or
            EntryPointNotFoundException or
            BadImageFormatException)
        {
            monitor = null;
            return false;
        }
    }

    public CudaDeviceMemoryInfo GetMemoryInfo()
    {
        nuint freeBytes = 0;
        nuint totalBytes = 0;
        var result = CudaDeviceScope.Execute(
            _cuda,
            DeviceId,
            () => _cuda.MemGetInfo(out freeBytes, out totalBytes));
        ThrowIfCudaFailure("cudaMemGetInfo", result);

        var free = checked((long)freeBytes);
        var total = checked((long)totalBytes);
        if (free > total)
        {
            throw new InvalidOperationException(
                $"cudaMemGetInfo reported free bytes ({free}) larger than total bytes ({total}) on device {DeviceId}.");
        }

        return new CudaDeviceMemoryInfo(free, total);
    }

    public bool TryGetDeviceMemoryPressure(
        out InferenceDeviceMemoryPressure pressure)
    {
        var info = GetMemoryInfo();
        var usedBytes = info.UsedBytes;
        var reclaimableBytes = GetKnownReclaimableBytes(usedBytes);
        var activeBytes = checked(usedBytes - reclaimableBytes);
        var peakReservedBytes = ObservePeak(usedBytes);

        pressure = new InferenceDeviceMemoryPressure(
            ActiveBytes: activeBytes,
            ReclaimableBytes: reclaimableBytes,
            ReservedBytes: usedBytes,
            PeakReservedBytes: peakReservedBytes);
        return true;
    }

    public async ValueTask<InferenceDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
        long targetReclaimableBytes,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetReclaimableBytes);
        cancellationToken.ThrowIfCancellationRequested();

        var reclaimer = _reclaimer ?? throw new NotSupportedException(
            "CUDA device-wide pressure monitor has no reclaimable source configured.");
        var result = await reclaimer.ReclaimDeviceMemoryAsync(
                targetReclaimableBytes,
                cancellationToken)
            .ConfigureAwait(false);
        ValidateReclaimResult(result);

        if (!TryGetDeviceMemoryPressure(out var after))
        {
            throw new InvalidOperationException(
                "CUDA device-wide memory pressure became unavailable after reclaim.");
        }

        return new InferenceDeviceMemoryReclaimResult(
            ReleasedBytes: result.ReleasedBytes,
            ReclaimableBytes: after.ReclaimableBytes,
            ReservedBytes: after.ReservedBytes);
    }

    private long GetKnownReclaimableBytes(long deviceUsedBytes)
    {
        if (_knownReclaimableSource is null ||
            !_knownReclaimableSource.TryGetDeviceMemoryPressure(out var known))
        {
            // Device-wide used bytes remain trustworthy even when the optional
            // classifier is unavailable. Treating all usage as non-reclaimable is
            // conservative and prevents an unavailable cache tracker from hiding VRAM.
            return 0;
        }

        ValidatePressure(known);
        return Math.Min(deviceUsedBytes, known.ReclaimableBytes);
    }

    private long ObservePeak(long reservedBytes)
    {
        while (true)
        {
            var current = Volatile.Read(ref _peakReservedBytes);
            if (current >= reservedBytes)
            {
                return current;
            }

            if (Interlocked.CompareExchange(
                    ref _peakReservedBytes,
                    reservedBytes,
                    current) == current)
            {
                return reservedBytes;
            }
        }
    }

    private static void ValidatePressure(InferenceDeviceMemoryPressure pressure)
    {
        if (pressure.ActiveBytes < 0 ||
            pressure.ReclaimableBytes < 0 ||
            pressure.ReservedBytes < 0 ||
            pressure.PeakReservedBytes < 0)
        {
            throw new InvalidOperationException(
                "Known CUDA reclaimable source reported negative byte counts.");
        }

        if (checked(pressure.ActiveBytes + pressure.ReclaimableBytes) !=
            pressure.ReservedBytes)
        {
            throw new InvalidOperationException(
                "Known CUDA reclaimable source must satisfy reserved = active + reclaimable bytes.");
        }

        if (pressure.PeakReservedBytes < pressure.ReservedBytes)
        {
            throw new InvalidOperationException(
                "Known CUDA reclaimable source peak cannot be smaller than current reserved bytes.");
        }
    }

    private static void ValidateReclaimResult(
        InferenceDeviceMemoryReclaimResult result)
    {
        if (result.ReleasedBytes < 0 ||
            result.ReclaimableBytes < 0 ||
            result.ReservedBytes < 0 ||
            result.ReclaimableBytes > result.ReservedBytes)
        {
            throw new InvalidOperationException(
                "Known CUDA reclaimer returned invalid device-memory accounting.");
        }
    }

    private void ThrowIfCudaFailure(string operation, int result)
    {
        if (result != 0)
        {
            throw new CudaRuntimeException(
                operation,
                result,
                _cuda.GetErrorString(result));
        }
    }
}

internal interface ICudaDeviceMemoryInfoApi : ICudaDeviceSelectionApi
{
    int MemGetInfo(out nuint freeBytes, out nuint totalBytes);
}

internal sealed class NativeCudaDeviceMemoryInfoApi : ICudaDeviceMemoryInfoApi
{
    private readonly CudaMemGetInfoDelegate _memGetInfo;
    private readonly CudaGetDeviceDelegate _getDevice;
    private readonly CudaSetDeviceDelegate _setDevice;
    private readonly CudaGetErrorStringDelegate _getErrorString;

    private NativeCudaDeviceMemoryInfoApi(nint libraryHandle)
    {
        _memGetInfo = GetDelegate<CudaMemGetInfoDelegate>(libraryHandle, "cudaMemGetInfo");
        _getDevice = GetDelegate<CudaGetDeviceDelegate>(libraryHandle, "cudaGetDevice");
        _setDevice = GetDelegate<CudaSetDeviceDelegate>(libraryHandle, "cudaSetDevice");
        _getErrorString = GetDelegate<CudaGetErrorStringDelegate>(libraryHandle, "cudaGetErrorString");

        // cudart intentionally remains loaded for process lifetime, matching the
        // other CUDA Runtime adapters in this backend.
    }

    public static NativeCudaDeviceMemoryInfoApi Load(
        string? explicitLibraryPath = null)
    {
        if (explicitLibraryPath is not null)
        {
            if (NativeLibrary.TryLoad(explicitLibraryPath, out var explicitHandle))
            {
                return new NativeCudaDeviceMemoryInfoApi(explicitHandle);
            }

            throw new DllNotFoundException(
                $"Unable to load CUDA Runtime library '{explicitLibraryPath}'.");
        }

        var candidates = GetRuntimeLibraryCandidates();
        foreach (var candidate in candidates)
        {
            if (NativeLibrary.TryLoad(candidate, out var handle))
            {
                return new NativeCudaDeviceMemoryInfoApi(handle);
            }
        }

        throw new DllNotFoundException(
            $"Unable to load the CUDA Runtime library. Tried: {string.Join(", ", candidates)}.");
    }

    public int MemGetInfo(out nuint freeBytes, out nuint totalBytes) =>
        _memGetInfo(out freeBytes, out totalBytes);

    public int GetDevice(out int deviceId) => _getDevice(out deviceId);
    public int SetDevice(int deviceId) => _setDevice(deviceId);

    public string? GetErrorString(int errorCode)
    {
        var pointer = _getErrorString(errorCode);
        return pointer == 0 ? null : Marshal.PtrToStringAnsi(pointer);
    }

    private static TDelegate GetDelegate<TDelegate>(
        nint libraryHandle,
        string name)
        where TDelegate : Delegate
    {
        if (!NativeLibrary.TryGetExport(libraryHandle, name, out var address))
        {
            throw new EntryPointNotFoundException(
                $"CUDA Runtime library does not export required symbol '{name}'.");
        }

        return Marshal.GetDelegateForFunctionPointer<TDelegate>(address);
    }

    private static string[] GetRuntimeLibraryCandidates()
    {
        if (OperatingSystem.IsWindows())
        {
            return
            [
                "cudart64_13.dll",
                "cudart64_12.dll",
                "cudart64_11.dll",
                "cudart64.dll",
                "cudart.dll"
            ];
        }

        if (OperatingSystem.IsLinux())
        {
            return
            [
                "libcudart.so",
                "libcudart.so.13",
                "libcudart.so.12",
                "libcudart.so.11.0"
            ];
        }

        return ["libcudart.dylib"];
    }

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaMemGetInfoDelegate(
        out nuint freeBytes,
        out nuint totalBytes);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaGetDeviceDelegate(out int deviceId);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaSetDeviceDelegate(int deviceId);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint CudaGetErrorStringDelegate(int errorCode);
}
