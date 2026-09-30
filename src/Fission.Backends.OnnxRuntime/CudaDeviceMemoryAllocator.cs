using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Options for explicit CUDA device-memory allocation.
/// </summary>
public sealed record CudaDeviceMemoryAllocatorOptions
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
/// Exact CUDA device allocation whose pointer remains valid until disposal.
/// </summary>
public sealed class CudaDeviceMemoryAllocation : IDisposable
{
    private CudaDeviceAllocationHandle? _handle;
    private readonly Action<CudaDeviceAllocationHandle, long>? _release;

    internal CudaDeviceMemoryAllocation(
        CudaDeviceAllocationHandle handle,
        long byteLength,
        int deviceId,
        Action<CudaDeviceAllocationHandle, long>? release = null)
    {
        _handle = handle;
        _release = release;
        ByteLength = byteLength;
        DeviceId = deviceId;
    }

    public long ByteLength { get; }
    public int DeviceId { get; }
    public bool IsDisposed => Volatile.Read(ref _handle) is null;

    public nint Pointer =>
        Volatile.Read(ref _handle)?.DangerousGetHandle() ??
        throw new ObjectDisposedException(nameof(CudaDeviceMemoryAllocation));

    public void Dispose()
    {
        var handle = Interlocked.Exchange(ref _handle, null);
        if (handle is null)
        {
            return;
        }

        if (_release is null)
        {
            handle.Dispose();
            return;
        }

        _release(handle, ByteLength);
    }
}

/// <summary>
/// Device-ordinal-bound cudaMalloc/cudaFree allocator. Allocation and release both
/// enter the configured CUDA device and restore the calling thread's prior device.
/// </summary>
public class CudaDeviceMemoryAllocator
{
    private readonly ICudaDeviceMemoryApi _cuda;

    public CudaDeviceMemoryAllocator(
        CudaDeviceMemoryAllocatorOptions? options = null)
        : this(
            NativeCudaDeviceMemoryApi.Load(options?.RuntimeLibraryPath),
            options ?? new CudaDeviceMemoryAllocatorOptions())
    {
    }

    protected internal CudaDeviceMemoryAllocator(
        ICudaDeviceMemoryApi cuda,
        CudaDeviceMemoryAllocatorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(cuda);
        options ??= new CudaDeviceMemoryAllocatorOptions();
        options.Validate();
        _cuda = cuda;
        DeviceId = options.DeviceId;
    }

    public int DeviceId { get; }

    public static bool TryCreate(
        out CudaDeviceMemoryAllocator? allocator,
        CudaDeviceMemoryAllocatorOptions? options = null)
    {
        try
        {
            allocator = new CudaDeviceMemoryAllocator(options);
            return true;
        }
        catch (Exception exception) when (exception is
            DllNotFoundException or
            EntryPointNotFoundException or
            BadImageFormatException)
        {
            allocator = null;
            return false;
        }
    }

    public virtual CudaDeviceMemoryAllocation Allocate(long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        return new CudaDeviceMemoryAllocation(
            AllocateHandle(byteLength),
            byteLength,
            DeviceId);
    }

    private protected CudaDeviceAllocationHandle AllocateHandle(long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        var nativeBytes = checked((nuint)byteLength);
        nint pointer = 0;
        var result = ExecuteOnDevice(
            () => _cuda.Malloc(out pointer, nativeBytes));
        ThrowIfCudaFailure("cudaMalloc", result);
        if (pointer == 0)
        {
            throw new InvalidOperationException(
                "cudaMalloc reported success but returned a null device pointer.");
        }

        return new CudaDeviceAllocationHandle(
            _cuda,
            pointer,
            DeviceId);
    }

    private int ExecuteOnDevice(Func<int> operation) =>
        CudaDeviceScope.Execute(
            _cuda,
            DeviceId,
            operation);

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

internal interface ICudaDeviceMemoryApi : ICudaDeviceSelectionApi
{
    int Malloc(out nint pointer, nuint byteLength);
    int Free(nint pointer);
}

internal static class CudaDeviceScope
{
    public static int Execute(
        ICudaDeviceSelectionApi deviceSelection,
        int deviceId,
        Func<int> operation)
    {
        ArgumentNullException.ThrowIfNull(deviceSelection);
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);

        var getResult = deviceSelection.GetDevice(out var previousDevice);
        if (getResult != 0)
        {
            throw CreateFailure(
                deviceSelection,
                "cudaGetDevice",
                getResult);
        }

        var changed = previousDevice != deviceId;
        if (changed)
        {
            var setResult = deviceSelection.SetDevice(deviceId);
            if (setResult != 0)
            {
                throw CreateFailure(
                    deviceSelection,
                    "cudaSetDevice",
                    setResult);
            }
        }

        Exception? operationFailure = null;
        var result = 0;
        try
        {
            result = operation();
        }
        catch (Exception exception)
        {
            operationFailure = exception;
        }

        Exception? restoreFailure = null;
        if (changed)
        {
            var restoreResult = deviceSelection.SetDevice(previousDevice);
            if (restoreResult != 0)
            {
                restoreFailure = CreateFailure(
                    deviceSelection,
                    "cudaSetDevice(restore)",
                    restoreResult);
            }
        }

        if (operationFailure is not null && restoreFailure is not null)
        {
            throw new AggregateException(operationFailure, restoreFailure);
        }

        if (operationFailure is not null)
        {
            ExceptionDispatchInfo.Capture(operationFailure).Throw();
        }

        if (restoreFailure is not null)
        {
            throw restoreFailure;
        }

        return result;
    }

    private static CudaRuntimeException CreateFailure(
        ICudaDeviceSelectionApi deviceSelection,
        string operation,
        int errorCode) =>
        new(
            operation,
            errorCode,
            deviceSelection.GetErrorString(errorCode));
}

internal sealed class CudaDeviceAllocationHandle : SafeHandle
{
    private readonly ICudaDeviceMemoryApi _cuda;
    private readonly int _deviceId;

    public CudaDeviceAllocationHandle(
        ICudaDeviceMemoryApi cuda,
        nint pointer,
        int deviceId)
        : base(IntPtr.Zero, ownsHandle: true)
    {
        _cuda = cuda;
        _deviceId = deviceId;
        SetHandle(pointer);
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        try
        {
            return CudaDeviceScope.Execute(
                    _cuda,
                    _deviceId,
                    () => _cuda.Free(handle)) == 0;
        }
        catch
        {
            return false;
        }
    }
}

internal sealed class NativeCudaDeviceMemoryApi : ICudaDeviceMemoryApi
{
    private readonly CudaMallocDelegate _malloc;
    private readonly CudaFreeDelegate _free;
    private readonly CudaGetDeviceDelegate _getDevice;
    private readonly CudaSetDeviceDelegate _setDevice;
    private readonly CudaGetErrorStringDelegate _getErrorString;

    private NativeCudaDeviceMemoryApi(nint libraryHandle)
    {
        _malloc = GetDelegate<CudaMallocDelegate>(libraryHandle, "cudaMalloc");
        _free = GetDelegate<CudaFreeDelegate>(libraryHandle, "cudaFree");
        _getDevice = GetDelegate<CudaGetDeviceDelegate>(libraryHandle, "cudaGetDevice");
        _setDevice = GetDelegate<CudaSetDeviceDelegate>(libraryHandle, "cudaSetDevice");
        _getErrorString = GetDelegate<CudaGetErrorStringDelegate>(libraryHandle, "cudaGetErrorString");

        // Keep cudart loaded for process lifetime because SafeHandle finalization can
        // invoke cudaFree after the allocator object itself is gone.
    }

    public static NativeCudaDeviceMemoryApi Load(
        string? explicitLibraryPath = null)
    {
        if (explicitLibraryPath is not null)
        {
            if (NativeLibrary.TryLoad(explicitLibraryPath, out var explicitHandle))
            {
                return new NativeCudaDeviceMemoryApi(explicitHandle);
            }

            throw new DllNotFoundException(
                $"Unable to load CUDA Runtime library '{explicitLibraryPath}'.");
        }

        var candidates = GetRuntimeLibraryCandidates();
        foreach (var candidate in candidates)
        {
            if (NativeLibrary.TryLoad(candidate, out var handle))
            {
                return new NativeCudaDeviceMemoryApi(handle);
            }
        }

        throw new DllNotFoundException(
            $"Unable to load the CUDA Runtime library. Tried: {string.Join(", ", candidates)}.");
    }

    public int Malloc(out nint pointer, nuint byteLength) =>
        _malloc(out pointer, byteLength);

    public int Free(nint pointer) => _free(pointer);
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
    private delegate int CudaMallocDelegate(out nint pointer, nuint byteLength);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaFreeDelegate(nint pointer);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaGetDeviceDelegate(out int deviceId);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaSetDeviceDelegate(int deviceId);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint CudaGetErrorStringDelegate(int errorCode);
}
