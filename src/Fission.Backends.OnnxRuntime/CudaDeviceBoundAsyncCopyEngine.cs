using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Device-ordinal-bound façade over <see cref="CudaAsyncCopyEngine"/>.
///
/// CUDA Runtime current-device state is host-thread local, while .NET async
/// continuations and the copy engine completion pump can execute on different
/// thread-pool threads. This wrapper scopes every native stream/event/memcpy call
/// to one explicit CUDA device and restores the thread's previous device before
/// returning. That prevents multi-GPU copies from depending on ambient thread
/// state left by unrelated work.
/// </summary>
public sealed class CudaDeviceBoundAsyncCopyEngine : IAsyncDisposable
{
    private readonly CudaAsyncCopyEngine _inner;

    public CudaDeviceBoundAsyncCopyEngine(
        int deviceId,
        CudaAsyncCopyEngineOptions? options = null)
        : this(
            deviceId,
            NativeCudaAsyncCopyApi.Load(options?.RuntimeLibraryPath),
            NativeCudaDeviceSelectionApi.Load(options?.RuntimeLibraryPath),
            options)
    {
    }

    internal CudaDeviceBoundAsyncCopyEngine(
        int deviceId,
        ICudaAsyncCopyApi cuda,
        ICudaDeviceSelectionApi deviceSelection,
        CudaAsyncCopyEngineOptions? options = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        ArgumentNullException.ThrowIfNull(cuda);
        ArgumentNullException.ThrowIfNull(deviceSelection);

        DeviceId = deviceId;
        var scoped = new DeviceScopedCudaAsyncCopyApi(
            cuda,
            deviceSelection,
            deviceId);
        _inner = new CudaAsyncCopyEngine(scoped, options);
    }

    public int DeviceId { get; }
    public int StreamCount => _inner.StreamCount;
    public int ActiveCopies => _inner.ActiveCopies;
    public int PendingCompletions => _inner.PendingCompletions;

    internal CudaAsyncCopyEngine InnerEngine => _inner;

    public static bool TryCreate(
        int deviceId,
        out CudaDeviceBoundAsyncCopyEngine? engine,
        CudaAsyncCopyEngineOptions? options = null)
    {
        try
        {
            engine = new CudaDeviceBoundAsyncCopyEngine(deviceId, options);
            return true;
        }
        catch (Exception exception) when (exception is
            DllNotFoundException or
            EntryPointNotFoundException or
            BadImageFormatException)
        {
            engine = null;
            return false;
        }
    }

    public ValueTask CopyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        CancellationToken cancellationToken = default) =>
        _inner.CopyAsync(
            destination,
            source,
            byteLength,
            kind,
            cancellationToken);

    public ValueTask DisposeAsync() => _inner.DisposeAsync();
}

internal interface ICudaDeviceSelectionApi
{
    int GetDevice(out int deviceId);
    int SetDevice(int deviceId);
    string? GetErrorString(int errorCode);
}

internal sealed class DeviceScopedCudaAsyncCopyApi : ICudaAsyncCopyApi
{
    private readonly ICudaAsyncCopyApi _inner;
    private readonly ICudaDeviceSelectionApi _deviceSelection;
    private readonly int _deviceId;

    public DeviceScopedCudaAsyncCopyApi(
        ICudaAsyncCopyApi inner,
        ICudaDeviceSelectionApi deviceSelection,
        int deviceId)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(deviceSelection);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        _inner = inner;
        _deviceSelection = deviceSelection;
        _deviceId = deviceId;
    }

    public int StreamCreateWithFlags(out nint stream, uint flags)
    {
        nint created = 0;
        var result = Execute(() => _inner.StreamCreateWithFlags(out created, flags));
        stream = created;
        return result;
    }

    public int StreamDestroy(nint stream) =>
        Execute(() => _inner.StreamDestroy(stream));

    public int StreamSynchronize(nint stream) =>
        Execute(() => _inner.StreamSynchronize(stream));

    public int MemcpyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream) =>
        Execute(() => _inner.MemcpyAsync(
            destination,
            source,
            byteLength,
            kind,
            stream));

    public int EventCreateWithFlags(out nint completionEvent, uint flags)
    {
        nint created = 0;
        var result = Execute(() => _inner.EventCreateWithFlags(out created, flags));
        completionEvent = created;
        return result;
    }

    public int EventRecord(nint completionEvent, nint stream) =>
        Execute(() => _inner.EventRecord(completionEvent, stream));

    public int EventQuery(nint completionEvent) =>
        Execute(() => _inner.EventQuery(completionEvent));

    public int EventDestroy(nint completionEvent) =>
        Execute(() => _inner.EventDestroy(completionEvent));

    public string? GetErrorString(int errorCode) => _inner.GetErrorString(errorCode);

    private int Execute(Func<int> operation)
    {
        var getResult = _deviceSelection.GetDevice(out var previousDevice);
        if (getResult != 0)
        {
            throw CreateFailure("cudaGetDevice", getResult);
        }

        var changed = previousDevice != _deviceId;
        if (changed)
        {
            var setResult = _deviceSelection.SetDevice(_deviceId);
            if (setResult != 0)
            {
                throw CreateFailure("cudaSetDevice", setResult);
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
            var restoreResult = _deviceSelection.SetDevice(previousDevice);
            if (restoreResult != 0)
            {
                restoreFailure = CreateFailure(
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

    private CudaRuntimeException CreateFailure(
        string operation,
        int errorCode) =>
        new(
            operation,
            errorCode,
            _deviceSelection.GetErrorString(errorCode));
}

internal sealed class NativeCudaDeviceSelectionApi : ICudaDeviceSelectionApi
{
    private readonly CudaGetDeviceDelegate _getDevice;
    private readonly CudaSetDeviceDelegate _setDevice;
    private readonly CudaGetErrorStringDelegate _getErrorString;

    private NativeCudaDeviceSelectionApi(nint libraryHandle)
    {
        _getDevice = GetDelegate<CudaGetDeviceDelegate>(
            libraryHandle,
            "cudaGetDevice");
        _setDevice = GetDelegate<CudaSetDeviceDelegate>(
            libraryHandle,
            "cudaSetDevice");
        _getErrorString = GetDelegate<CudaGetErrorStringDelegate>(
            libraryHandle,
            "cudaGetErrorString");

        // Keep cudart loaded for process lifetime. Device-scoped native handles can
        // still be disposed after the public façade itself becomes unreachable.
    }

    public static NativeCudaDeviceSelectionApi Load(
        string? explicitLibraryPath = null)
    {
        if (explicitLibraryPath is not null)
        {
            if (NativeLibrary.TryLoad(explicitLibraryPath, out var explicitHandle))
            {
                return new NativeCudaDeviceSelectionApi(explicitHandle);
            }

            throw new DllNotFoundException(
                $"Unable to load CUDA Runtime library '{explicitLibraryPath}'.");
        }

        var candidates = GetRuntimeLibraryCandidates();
        foreach (var candidate in candidates)
        {
            if (NativeLibrary.TryLoad(candidate, out var handle))
            {
                return new NativeCudaDeviceSelectionApi(handle);
            }
        }

        throw new DllNotFoundException(
            $"Unable to load the CUDA Runtime library. Tried: {string.Join(", ", candidates)}.");
    }

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
    private delegate int CudaGetDeviceDelegate(out int deviceId);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaSetDeviceDelegate(int deviceId);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint CudaGetErrorStringDelegate(int errorCode);
}
