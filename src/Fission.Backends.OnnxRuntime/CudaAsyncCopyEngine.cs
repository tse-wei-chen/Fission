using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace Fission.Backends.OnnxRuntime;

public enum CudaMemcpyKind
{
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4
}

/// <summary>
/// Controls the reusable CUDA stream pool and event-completion polling used by
/// <see cref="CudaAsyncCopyEngine"/>.
/// </summary>
public sealed record CudaAsyncCopyEngineOptions
{
    public int StreamCount { get; init; } = 4;
    public TimeSpan CompletionPollInterval { get; init; } = TimeSpan.FromMilliseconds(1);

    /// <summary>
    /// Optional exact CUDA Runtime library path/name. When omitted, Fission probes
    /// conventional cudart names for the current operating system.
    /// </summary>
    public string? RuntimeLibraryPath { get; init; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(StreamCount);

        if (CompletionPollInterval <= TimeSpan.Zero ||
            CompletionPollInterval == Timeout.InfiniteTimeSpan)
        {
            throw new ArgumentOutOfRangeException(
                nameof(CompletionPollInterval),
                CompletionPollInterval,
                "CUDA completion polling interval must be positive and finite.");
        }

        if (RuntimeLibraryPath is not null && string.IsNullOrWhiteSpace(RuntimeLibraryPath))
        {
            throw new ArgumentException(
                "CUDA Runtime library path must be non-empty when specified.",
                nameof(RuntimeLibraryPath));
        }
    }
}

/// <summary>
/// Bounded, reusable CUDA asynchronous copy primitive. Copies are submitted onto
/// a fixed pool of non-blocking streams and completed by querying disable-timing
/// CUDA events from one completion pump.
///
/// Cancellation is cooperative with native lifetime safety: cancellation before
/// submission prevents a copy; cancellation after cudaMemcpyAsync submission is
/// observed only after the completion event reports that the native work has
/// finished. Callers must keep source and destination allocations alive until the
/// returned operation completes, including when that completion is cancellation.
/// </summary>
public sealed class CudaAsyncCopyEngine : IAsyncDisposable
{
    internal const int CudaErrorNotReady = 600;
    internal const uint CudaStreamNonBlocking = 0x01;
    internal const uint CudaEventDisableTiming = 0x02;

    private readonly ICudaAsyncCopyApi _cuda;
    private readonly CudaAsyncCopyEngineOptions _options;
    private readonly ConcurrentQueue<nint> _streams = new();
    private readonly SemaphoreSlim _streamSlots;
    private readonly ConcurrentDictionary<long, PendingCopy> _pending = new();
    private readonly SemaphoreSlim _pendingSignal = new(0);
    private readonly CancellationTokenSource _disposeCancellation = new();
    private readonly CancellationTokenSource _pumpCancellation = new();
    private readonly Task _completionPump;
    private long _nextPendingId;
    private int _disposed;

    public CudaAsyncCopyEngine(CudaAsyncCopyEngineOptions? options = null)
        : this(
            NativeCudaAsyncCopyApi.Load(options?.RuntimeLibraryPath),
            options ?? new CudaAsyncCopyEngineOptions())
    {
    }

    internal CudaAsyncCopyEngine(
        ICudaAsyncCopyApi cuda,
        CudaAsyncCopyEngineOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(cuda);
        options ??= new CudaAsyncCopyEngineOptions();
        options.Validate();

        _cuda = cuda;
        _options = options;
        _streamSlots = new SemaphoreSlim(options.StreamCount, options.StreamCount);

        var createdStreams = new List<nint>(options.StreamCount);
        try
        {
            for (var index = 0; index < options.StreamCount; index++)
            {
                var result = _cuda.StreamCreateWithFlags(
                    out var stream,
                    CudaStreamNonBlocking);
                ThrowIfCudaFailure("cudaStreamCreateWithFlags", result);
                if (stream == 0)
                {
                    throw new InvalidOperationException(
                        "cudaStreamCreateWithFlags reported success but returned a null stream.");
                }

                createdStreams.Add(stream);
                _streams.Enqueue(stream);
            }
        }
        catch
        {
            foreach (var stream in createdStreams)
            {
                _ = _cuda.StreamDestroy(stream);
            }

            _streamSlots.Dispose();
            _disposeCancellation.Dispose();
            _pumpCancellation.Dispose();
            _pendingSignal.Dispose();
            throw;
        }

        _completionPump = Task.Run(CompletionPumpAsync);
    }

    public int StreamCount => _options.StreamCount;
    public int ActiveCopies => _options.StreamCount - _streamSlots.CurrentCount;
    public int PendingCompletions => _pending.Count;

    public static bool TryCreate(
        out CudaAsyncCopyEngine? engine,
        CudaAsyncCopyEngineOptions? options = null)
    {
        try
        {
            engine = new CudaAsyncCopyEngine(options);
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

    public async ValueTask CopyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfEqual(destination, 0);
        ArgumentOutOfRangeException.ThrowIfEqual(source, 0);
        ArgumentOutOfRangeException.ThrowIfZero(byteLength);
        ValidateMemcpyKind(kind);
        cancellationToken.ThrowIfCancellationRequested();

        using var acquireCancellation = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            _disposeCancellation.Token);

        try
        {
            await _streamSlots.WaitAsync(acquireCancellation.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (
            _disposeCancellation.IsCancellationRequested &&
            !cancellationToken.IsCancellationRequested)
        {
            throw new ObjectDisposedException(nameof(CudaAsyncCopyEngine));
        }

        nint stream = 0;
        nint completionEvent = 0;
        var copySubmitted = false;
        var pendingPublished = false;

        try
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                throw new ObjectDisposedException(nameof(CudaAsyncCopyEngine));
            }

            if (!_streams.TryDequeue(out stream) || stream == 0)
            {
                throw new InvalidOperationException(
                    "CUDA stream admission was granted without an available stream.");
            }

            var result = _cuda.EventCreateWithFlags(
                out completionEvent,
                CudaEventDisableTiming);
            ThrowIfCudaFailure("cudaEventCreateWithFlags", result);
            if (completionEvent == 0)
            {
                throw new InvalidOperationException(
                    "cudaEventCreateWithFlags reported success but returned a null event.");
            }

            result = _cuda.MemcpyAsync(
                destination,
                source,
                byteLength,
                kind,
                stream);
            ThrowIfCudaFailure("cudaMemcpyAsync", result);
            copySubmitted = true;

            result = _cuda.EventRecord(completionEvent, stream);
            if (result != 0)
            {
                var recordFailure = CreateCudaFailure("cudaEventRecord", result);
                var synchronizeFailure = SynchronizeAfterSubmissionFailure(stream);
                if (synchronizeFailure is not null)
                {
                    throw new AggregateException(recordFailure, synchronizeFailure);
                }

                throw recordFailure;
            }

            var pending = new PendingCopy(
                Interlocked.Increment(ref _nextPendingId),
                stream,
                completionEvent,
                cancellationToken);

            if (!_pending.TryAdd(pending.Id, pending))
            {
                var publishFailure = new InvalidOperationException(
                    $"CUDA pending-copy id {pending.Id} was published more than once.");
                var synchronizeFailure = SynchronizeAfterSubmissionFailure(stream);
                if (synchronizeFailure is not null)
                {
                    throw new AggregateException(publishFailure, synchronizeFailure);
                }

                throw publishFailure;
            }

            pendingPublished = true;
            _pendingSignal.Release();
            await pending.Completion.Task.ConfigureAwait(false);
        }
        finally
        {
            if (!pendingPublished)
            {
                if (copySubmitted && stream != 0)
                {
                    // A submission can outlive the caller-visible failure path (for
                    // example when event recording fails). Synchronize before any
                    // stream reuse or pointer-lifetime release.
                    _ = _cuda.StreamSynchronize(stream);
                }

                if (completionEvent != 0)
                {
                    _ = _cuda.EventDestroy(completionEvent);
                }

                if (stream != 0)
                {
                    _streams.Enqueue(stream);
                }

                _streamSlots.Release();
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _disposeCancellation.Cancel();

        // Own every stream slot before stopping the completion pump. In-flight
        // copies retain one slot until their event reaches a terminal state, so
        // this also guarantees there is no native DMA when streams are destroyed.
        for (var index = 0; index < _options.StreamCount; index++)
        {
            await _streamSlots.WaitAsync().ConfigureAwait(false);
        }

        _pumpCancellation.Cancel();
        try
        {
            await _completionPump.ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (_pumpCancellation.IsCancellationRequested)
        {
        }

        List<Exception>? failures = null;
        while (_streams.TryDequeue(out var stream))
        {
            var result = _cuda.StreamDestroy(stream);
            if (result != 0)
            {
                (failures ??= new List<Exception>()).Add(
                    CreateCudaFailure("cudaStreamDestroy", result));
            }
        }

        _streamSlots.Dispose();
        _pendingSignal.Dispose();
        _disposeCancellation.Dispose();
        _pumpCancellation.Dispose();

        if (failures is { Count: > 0 })
        {
            throw new AggregateException(
                "One or more CUDA streams could not be destroyed cleanly.",
                failures);
        }
    }

    private async Task CompletionPumpAsync()
    {
        while (true)
        {
            _pumpCancellation.Token.ThrowIfCancellationRequested();

            if (_pending.IsEmpty)
            {
                await _pendingSignal.WaitAsync(_pumpCancellation.Token)
                    .ConfigureAwait(false);
            }
            else
            {
                await Task.Delay(
                        _options.CompletionPollInterval,
                        _pumpCancellation.Token)
                    .ConfigureAwait(false);
            }

            while (_pendingSignal.Wait(0))
            {
            }

            foreach (var pair in _pending.ToArray())
            {
                PollPendingCopy(pair.Key, pair.Value);
            }
        }
    }

    private void PollPendingCopy(long id, PendingCopy pending)
    {
        var queryResult = _cuda.EventQuery(pending.CompletionEvent);
        if (queryResult == CudaErrorNotReady)
        {
            return;
        }

        if (!_pending.TryRemove(id, out _))
        {
            return;
        }

        Exception? failure = null;
        if (queryResult != 0)
        {
            failure = CreateCudaFailure("cudaEventQuery", queryResult);
            var synchronizeFailure = SynchronizeAfterSubmissionFailure(pending.Stream);
            if (synchronizeFailure is not null)
            {
                failure = new AggregateException(failure, synchronizeFailure);
            }
        }

        var destroyResult = _cuda.EventDestroy(pending.CompletionEvent);
        if (destroyResult != 0)
        {
            var destroyFailure = CreateCudaFailure("cudaEventDestroy", destroyResult);
            failure = failure is null
                ? destroyFailure
                : new AggregateException(failure, destroyFailure);
        }

        _streams.Enqueue(pending.Stream);
        _streamSlots.Release();

        if (failure is not null)
        {
            pending.Completion.TrySetException(failure);
            return;
        }

        if (pending.CallerCancellation.IsCancellationRequested)
        {
            pending.Completion.TrySetCanceled(pending.CallerCancellation);
            return;
        }

        pending.Completion.TrySetResult();
    }

    private Exception? SynchronizeAfterSubmissionFailure(nint stream)
    {
        var result = _cuda.StreamSynchronize(stream);
        return result == 0
            ? null
            : CreateCudaFailure("cudaStreamSynchronize", result);
    }

    private void ThrowIfCudaFailure(string operation, int errorCode)
    {
        if (errorCode != 0)
        {
            throw CreateCudaFailure(operation, errorCode);
        }
    }

    private CudaRuntimeException CreateCudaFailure(string operation, int errorCode) =>
        new(operation, errorCode, _cuda.GetErrorString(errorCode));

    private static void ValidateMemcpyKind(CudaMemcpyKind kind)
    {
        if (kind is < CudaMemcpyKind.HostToHost or > CudaMemcpyKind.Default)
        {
            throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
        }
    }

    private sealed class PendingCopy
    {
        public PendingCopy(
            long id,
            nint stream,
            nint completionEvent,
            CancellationToken callerCancellation)
        {
            Id = id;
            Stream = stream;
            CompletionEvent = completionEvent;
            CallerCancellation = callerCancellation;
            Completion = new TaskCompletionSource(
                TaskCreationOptions.RunContinuationsAsynchronously);
        }

        public long Id { get; }
        public nint Stream { get; }
        public nint CompletionEvent { get; }
        public CancellationToken CallerCancellation { get; }
        public TaskCompletionSource Completion { get; }
    }
}

internal interface ICudaAsyncCopyApi
{
    int StreamCreateWithFlags(out nint stream, uint flags);
    int StreamDestroy(nint stream);
    int StreamSynchronize(nint stream);
    int MemcpyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream);
    int EventCreateWithFlags(out nint completionEvent, uint flags);
    int EventRecord(nint completionEvent, nint stream);
    int EventQuery(nint completionEvent);
    int EventDestroy(nint completionEvent);
    string? GetErrorString(int errorCode);
}

internal sealed class NativeCudaAsyncCopyApi : ICudaAsyncCopyApi
{
    private readonly CudaStreamCreateWithFlagsDelegate _streamCreateWithFlags;
    private readonly CudaStreamDestroyDelegate _streamDestroy;
    private readonly CudaStreamSynchronizeDelegate _streamSynchronize;
    private readonly CudaMemcpyAsyncDelegate _memcpyAsync;
    private readonly CudaEventCreateWithFlagsDelegate _eventCreateWithFlags;
    private readonly CudaEventRecordDelegate _eventRecord;
    private readonly CudaEventQueryDelegate _eventQuery;
    private readonly CudaEventDestroyDelegate _eventDestroy;
    private readonly CudaGetErrorStringDelegate _getErrorString;

    private NativeCudaAsyncCopyApi(nint libraryHandle)
    {
        _streamCreateWithFlags = GetDelegate<CudaStreamCreateWithFlagsDelegate>(
            libraryHandle,
            "cudaStreamCreateWithFlags");
        _streamDestroy = GetDelegate<CudaStreamDestroyDelegate>(
            libraryHandle,
            "cudaStreamDestroy");
        _streamSynchronize = GetDelegate<CudaStreamSynchronizeDelegate>(
            libraryHandle,
            "cudaStreamSynchronize");
        _memcpyAsync = GetDelegate<CudaMemcpyAsyncDelegate>(
            libraryHandle,
            "cudaMemcpyAsync");
        _eventCreateWithFlags = GetDelegate<CudaEventCreateWithFlagsDelegate>(
            libraryHandle,
            "cudaEventCreateWithFlags");
        _eventRecord = GetDelegate<CudaEventRecordDelegate>(
            libraryHandle,
            "cudaEventRecord");
        _eventQuery = GetDelegate<CudaEventQueryDelegate>(
            libraryHandle,
            "cudaEventQuery");
        _eventDestroy = GetDelegate<CudaEventDestroyDelegate>(
            libraryHandle,
            "cudaEventDestroy");
        _getErrorString = GetDelegate<CudaGetErrorStringDelegate>(
            libraryHandle,
            "cudaGetErrorString");

        // Keep cudart loaded for process lifetime because streams/events may outlive
        // the public engine reference while DisposeAsync drains native work.
    }

    public static NativeCudaAsyncCopyApi Load(string? explicitLibraryPath = null)
    {
        if (explicitLibraryPath is not null)
        {
            if (NativeLibrary.TryLoad(explicitLibraryPath, out var explicitHandle))
            {
                return new NativeCudaAsyncCopyApi(explicitHandle);
            }

            throw new DllNotFoundException(
                $"Unable to load CUDA Runtime library '{explicitLibraryPath}'.");
        }

        var candidates = GetRuntimeLibraryCandidates();
        foreach (var candidate in candidates)
        {
            if (NativeLibrary.TryLoad(candidate, out var handle))
            {
                return new NativeCudaAsyncCopyApi(handle);
            }
        }

        throw new DllNotFoundException(
            $"Unable to load the CUDA Runtime library. Tried: {string.Join(", ", candidates)}.");
    }

    public int StreamCreateWithFlags(out nint stream, uint flags) =>
        _streamCreateWithFlags(out stream, flags);

    public int StreamDestroy(nint stream) => _streamDestroy(stream);

    public int StreamSynchronize(nint stream) => _streamSynchronize(stream);

    public int MemcpyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream) =>
        _memcpyAsync(destination, source, byteLength, kind, stream);

    public int EventCreateWithFlags(out nint completionEvent, uint flags) =>
        _eventCreateWithFlags(out completionEvent, flags);

    public int EventRecord(nint completionEvent, nint stream) =>
        _eventRecord(completionEvent, stream);

    public int EventQuery(nint completionEvent) => _eventQuery(completionEvent);

    public int EventDestroy(nint completionEvent) => _eventDestroy(completionEvent);

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
    private delegate int CudaStreamCreateWithFlagsDelegate(
        out nint stream,
        uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaStreamDestroyDelegate(nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaStreamSynchronizeDelegate(nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaMemcpyAsyncDelegate(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaEventCreateWithFlagsDelegate(
        out nint completionEvent,
        uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaEventRecordDelegate(
        nint completionEvent,
        nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaEventQueryDelegate(nint completionEvent);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int CudaEventDestroyDelegate(nint completionEvent);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint CudaGetErrorStringDelegate(int errorCode);
}
