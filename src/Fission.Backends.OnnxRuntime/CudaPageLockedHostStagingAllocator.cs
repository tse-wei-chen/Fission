using System.Buffers;
using System.Runtime.InteropServices;

namespace Fission.Backends.OnnxRuntime;

[Flags]
public enum CudaHostAllocationFlags : uint
{
    Default = 0,
    Portable = 0x01,
    Mapped = 0x02,
    WriteCombined = 0x04
}

/// <summary>
/// Configuration for CUDA Runtime page-locked host allocation. Portable is the
/// default because migration staging may be consumed from a CUDA context other
/// than the context that first allocated the buffer.
/// </summary>
public sealed record CudaPageLockedHostStagingAllocatorOptions
{
    public CudaHostAllocationFlags Flags { get; init; } = CudaHostAllocationFlags.Portable;

    /// <summary>
    /// Optional exact CUDA Runtime library path/name. When omitted, Fission probes
    /// conventional cudart names for the current operating system.
    /// </summary>
    public string? RuntimeLibraryPath { get; init; }

    internal void Validate()
    {
        const CudaHostAllocationFlags known =
            CudaHostAllocationFlags.Portable |
            CudaHostAllocationFlags.Mapped |
            CudaHostAllocationFlags.WriteCombined;

        if ((Flags & ~known) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Flags),
                Flags,
                "CUDA host allocation flags contain unsupported bits.");
        }

        if (RuntimeLibraryPath is not null && string.IsNullOrWhiteSpace(RuntimeLibraryPath))
        {
            throw new ArgumentException(
                "CUDA Runtime library path must be non-empty when specified.",
                nameof(RuntimeLibraryPath));
        }
    }
}

public sealed class CudaRuntimeException : InvalidOperationException
{
    public CudaRuntimeException(
        string operation,
        int errorCode,
        string? errorDescription = null)
        : base(CreateMessage(operation, errorCode, errorDescription))
    {
        Operation = operation;
        ErrorCode = errorCode;
        ErrorDescription = errorDescription;
    }

    public string Operation { get; }
    public int ErrorCode { get; }
    public string? ErrorDescription { get; }

    private static string CreateMessage(
        string operation,
        int errorCode,
        string? errorDescription) =>
        string.IsNullOrWhiteSpace(errorDescription)
            ? $"CUDA Runtime operation {operation} failed with error code {errorCode}."
            : $"CUDA Runtime operation {operation} failed with error code {errorCode}: {errorDescription}";
}

/// <summary>
/// Allocates exact-length FP32 staging buffers with cudaHostAlloc and releases them
/// with cudaFreeHost. The native allocation is exposed through Memory&lt;float&gt; via
/// a custom MemoryManager, so the existing ONNX host-staging payload can retain it
/// without copying into a managed array.
///
/// Loading cudart does not require a CUDA allocation. Actual Allocate calls may
/// still fail when the driver/device/runtime is unavailable; such failures surface
/// as CudaRuntimeException with the native CUDA error code.
/// </summary>
public sealed class CudaPageLockedHostStagingFloatBufferAllocator :
    IHostStagingFloatBufferAllocator
{
    private readonly ICudaHostMemoryApi _cuda;
    private readonly uint _flags;

    public CudaPageLockedHostStagingFloatBufferAllocator(
        CudaPageLockedHostStagingAllocatorOptions? options = null)
        : this(
            NativeCudaHostMemoryApi.Load(options?.RuntimeLibraryPath),
            options ?? new CudaPageLockedHostStagingAllocatorOptions())
    {
    }

    internal CudaPageLockedHostStagingFloatBufferAllocator(
        ICudaHostMemoryApi cuda,
        CudaPageLockedHostStagingAllocatorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(cuda);
        options ??= new CudaPageLockedHostStagingAllocatorOptions();
        options.Validate();

        _cuda = cuda;
        _flags = (uint)options.Flags;
    }

    public static bool TryCreate(
        out CudaPageLockedHostStagingFloatBufferAllocator? allocator,
        CudaPageLockedHostStagingAllocatorOptions? options = null)
    {
        try
        {
            allocator = new CudaPageLockedHostStagingFloatBufferAllocator(options);
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

    public IHostStagingFloatBuffer Allocate(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);

        var byteLength = checked((nuint)length * (nuint)sizeof(float));
        var result = _cuda.HostAlloc(out var pointer, byteLength, _flags);
        if (result != 0)
        {
            throw new CudaRuntimeException(
                "cudaHostAlloc",
                result,
                _cuda.GetErrorString(result));
        }

        if (pointer == 0)
        {
            throw new InvalidOperationException(
                "cudaHostAlloc reported success but returned a null host pointer.");
        }

        var handle = new CudaHostAllocationHandle(_cuda, pointer);
        return new CudaPageLockedHostStagingFloatBuffer(
            new CudaPageLockedFloatMemoryManager(handle, length));
    }

    private sealed class CudaPageLockedHostStagingFloatBuffer :
        ICudaPageLockedHostStagingFloatBuffer
    {
        private CudaPageLockedFloatMemoryManager? _memoryManager;

        public CudaPageLockedHostStagingFloatBuffer(
            CudaPageLockedFloatMemoryManager memoryManager)
        {
            _memoryManager = memoryManager;
        }

        public Memory<float> Memory =>
            _memoryManager?.Memory ??
            throw new ObjectDisposedException(nameof(CudaPageLockedHostStagingFloatBuffer));

        public nint Pointer =>
            _memoryManager?.Pointer ??
            throw new ObjectDisposedException(nameof(CudaPageLockedHostStagingFloatBuffer));

        public void Dispose()
        {
            var memoryManager = Interlocked.Exchange(ref _memoryManager, null);
            if (memoryManager is not null)
            {
                ((IDisposable)memoryManager).Dispose();
            }
        }
    }

    private sealed unsafe class CudaPageLockedFloatMemoryManager : MemoryManager<float>
    {
        private readonly CudaHostAllocationHandle _handle;
        private readonly int _length;
        private int _disposed;

        public CudaPageLockedFloatMemoryManager(
            CudaHostAllocationHandle handle,
            int length)
        {
            _handle = handle;
            _length = length;
        }

        public nint Pointer
        {
            get
            {
                ThrowIfDisposed();
                return _handle.DangerousGetHandle();
            }
        }

        public override Span<float> GetSpan()
        {
            ThrowIfDisposed();
            return new Span<float>((void*)_handle.DangerousGetHandle(), _length);
        }

        public override MemoryHandle Pin(int elementIndex = 0)
        {
            ThrowIfDisposed();
            if ((uint)elementIndex > (uint)_length)
            {
                throw new ArgumentOutOfRangeException(nameof(elementIndex));
            }

            return new MemoryHandle(
                ((float*)_handle.DangerousGetHandle()) + elementIndex);
        }

        public override void Unpin()
        {
            // cudaHostAlloc memory is already physically page-locked. MemoryHandle
            // pin/unpin only brackets the consumer's logical pointer access.
        }

        protected override void Dispose(bool disposing)
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            _handle.Dispose();
        }

        private void ThrowIfDisposed() =>
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    private sealed class CudaHostAllocationHandle : SafeHandle
    {
        private readonly ICudaHostMemoryApi _cuda;

        public CudaHostAllocationHandle(ICudaHostMemoryApi cuda, nint pointer)
            : base(IntPtr.Zero, ownsHandle: true)
        {
            _cuda = cuda;
            SetHandle(pointer);
        }

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle() => _cuda.FreeHost(handle) == 0;
    }
}

internal interface ICudaHostMemoryApi
{
    int HostAlloc(out nint pointer, nuint byteLength, uint flags);
    int FreeHost(nint pointer);
    string? GetErrorString(int errorCode);
}

internal sealed class NativeCudaHostMemoryApi : ICudaHostMemoryApi
{
    private readonly CudaHostAllocDelegate _hostAlloc;
    private readonly CudaFreeHostDelegate _freeHost;
    private readonly CudaGetErrorStringDelegate _getErrorString;

    private NativeCudaHostMemoryApi(nint libraryHandle)
    {
        _hostAlloc = GetDelegate<CudaHostAllocDelegate>(libraryHandle, "cudaHostAlloc");
        _freeHost = GetDelegate<CudaFreeHostDelegate>(libraryHandle, "cudaFreeHost");
        _getErrorString = GetDelegate<CudaGetErrorStringDelegate>(libraryHandle, "cudaGetErrorString");

        // The library intentionally remains loaded for process lifetime. Individual
        // host buffers may outlive the allocator object that created them and still
        // need cudaFreeHost during payload/state teardown.
    }

    public static NativeCudaHostMemoryApi Load(string? explicitLibraryPath = null)
    {
        if (explicitLibraryPath is not null)
        {
            if (NativeLibrary.TryLoad(explicitLibraryPath, out var explicitHandle))
            {
                return new NativeCudaHostMemoryApi(explicitHandle);
            }

            throw new DllNotFoundException(
                $"Unable to load CUDA Runtime library '{explicitLibraryPath}'.");
        }

        var candidates = GetRuntimeLibraryCandidates();
        foreach (var candidate in candidates)
        {
            if (NativeLibrary.TryLoad(candidate, out var handle))
            {
                return new NativeCudaHostMemoryApi(handle);
            }
        }

        throw new DllNotFoundException(
            $"Unable to load the CUDA Runtime library. Tried: {string.Join(", ", candidates)}.");
    }

    public int HostAlloc(out nint pointer, nuint byteLength, uint flags) =>
        _hostAlloc(out pointer, byteLength, flags);

    public int FreeHost(nint pointer) => _freeHost(pointer);

    public string? GetErrorString(int errorCode)
    {
        var pointer = _getErrorString(errorCode);
        return pointer == 0 ? null : Marshal.PtrToStringAnsi(pointer);
    }

    private static TDelegate GetDelegate<TDelegate>(nint libraryHandle, string name)
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
    private delegate int CudaHostAllocDelegate(
        out nint pointer,
        nuint byteLength,
        uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaFreeHostDelegate(nint pointer);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint CudaGetErrorStringDelegate(int errorCode);
}
