using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Fission.Backends.OnnxRuntime;

static void Require(
    [DoesNotReturnIf(false)] bool condition,
    string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

await RunHostCopyAndFlagsAsync();
await RunBoundedConcurrencyAsync();
await RunCancellationAfterSubmissionAsync();
await RunMemcpyFailureAndReuseAsync();
await RunQueryFailureCleanupAsync();
await RunDisposeDrainAsync();
RunRuntimeProbeFailure();

Console.WriteLine("Fission ONNX Runtime CUDA async-copy specs passed.");

static async Task RunHostCopyAndFlagsAsync()
{
    var cuda = new FakeCudaAsyncCopyApi
    {
        AutoComplete = true,
        QueriesBeforeAutoComplete = 1
    };
    await using var engine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 2,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    using var source = NativeFloatBuffer.Create(new[] { 1f, 2f, 3f, 4f });
    using var destination = NativeFloatBuffer.Create(4);

    await engine.CopyAsync(
        destination.Pointer,
        source.Pointer,
        16,
        CudaMemcpyKind.HostToHost);

    Require(destination.Read().SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "Fake cudaMemcpyAsync must copy the requested bytes.");
    Require(cuda.StreamCreateCalls == 2, "Engine must create the configured number of reusable CUDA streams.");
    Require(cuda.StreamCreateFlags.All(static flags => flags == CudaAsyncCopyEngine.CudaStreamNonBlocking), "Every reusable CUDA stream must be non-blocking.");
    Require(cuda.EventCreateFlags.Single() == CudaAsyncCopyEngine.CudaEventDisableTiming, "Copy completion events must disable timing overhead.");
    Require(cuda.MemcpyCalls.Single().Kind == CudaMemcpyKind.HostToHost, "Memcpy kind must flow to the CUDA Runtime boundary.");
    Require(cuda.MemcpyCalls.Single().ByteLength == 16, "Memcpy byte count must flow unchanged to the CUDA Runtime boundary.");
    Require(cuda.EventQueryCalls >= 2, "Completion pump must tolerate cudaErrorNotReady before observing completion.");
    Require(engine.ActiveCopies == 0 && engine.PendingCompletions == 0, "Completed copies must release stream admission and pending-event state.");
}

static async Task RunBoundedConcurrencyAsync()
{
    var cuda = new FakeCudaAsyncCopyApi { AutoComplete = false };
    await using var engine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 2,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    using var source = NativeFloatBuffer.Create(new[] { 9f });
    using var destination1 = NativeFloatBuffer.Create(1);
    using var destination2 = NativeFloatBuffer.Create(1);
    using var destination3 = NativeFloatBuffer.Create(1);

    var first = engine.CopyAsync(destination1.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost).AsTask();
    var second = engine.CopyAsync(destination2.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost).AsTask();
    var third = engine.CopyAsync(destination3.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost).AsTask();

    await WaitUntilAsync(
        () => cuda.MemcpyAsyncCalls == 2,
        "Two streams must admit exactly two copies before native completion.");
    await Task.Delay(20);
    Require(cuda.MemcpyAsyncCalls == 2, "A third copy must wait while both CUDA streams are in flight.");
    Require(engine.ActiveCopies == 2, "Stream admission must expose the bounded in-flight copy count.");

    Require(cuda.CompleteOneRecordedEvent(), "At least one recorded CUDA event must be available for completion.");
    await WaitUntilAsync(
        () => cuda.EventRecordCalls == 3,
        "Completing one event must release exactly one stream slot and fully record the queued copy.");

    cuda.CompleteAllRecordedEvents();
    await Task.WhenAll(first, second, third);
    Require(destination1.Read()[0] == 9f && destination2.Read()[0] == 9f && destination3.Read()[0] == 9f, "All bounded copies must preserve payload bytes.");
}

static async Task RunCancellationAfterSubmissionAsync()
{
    var cuda = new FakeCudaAsyncCopyApi { AutoComplete = false };
    await using var engine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 1,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    using var source = NativeFloatBuffer.Create(new[] { 5f, 6f });
    using var destination = NativeFloatBuffer.Create(2);
    using var cancellation = new CancellationTokenSource();

    var copy = engine.CopyAsync(
        destination.Pointer,
        source.Pointer,
        8,
        CudaMemcpyKind.HostToHost,
        cancellation.Token).AsTask();

    await WaitUntilAsync(
        () => cuda.EventRecordCalls == 1,
        "CUDA copy must reach native submission before cancellation is tested.");

    cancellation.Cancel();
    await Task.Delay(20);
    Require(!copy.IsCompleted, "Cancellation after cudaMemcpyAsync submission must not release native lifetime before the completion event.");
    Require(cuda.EventDestroyCalls == 0, "Completion event must remain alive while submitted native work is incomplete.");

    cuda.CompleteAllRecordedEvents();
    var canceled = false;
    try
    {
        await copy;
    }
    catch (OperationCanceledException)
    {
        canceled = true;
    }

    Require(canceled, "Caller cancellation must surface after the submitted copy reaches native completion.");
    Require(cuda.EventDestroyCalls == 1, "Completed canceled copy must reclaim its CUDA event.");
    Require(engine.ActiveCopies == 0, "Canceled completed copy must return its stream slot.");
}

static async Task RunMemcpyFailureAndReuseAsync()
{
    var cuda = new FakeCudaAsyncCopyApi
    {
        AutoComplete = true,
        NextMemcpyResult = 77
    };
    await using var engine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 1,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    using var source = NativeFloatBuffer.Create(new[] { 12f });
    using var destination = NativeFloatBuffer.Create(1);

    CudaRuntimeException? observed = null;
    try
    {
        await engine.CopyAsync(destination.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost);
    }
    catch (CudaRuntimeException exception)
    {
        observed = exception;
    }

    Require(observed is not null, "cudaMemcpyAsync failure must surface through CudaRuntimeException.");
    Require(observed.Operation == "cudaMemcpyAsync" && observed.ErrorCode == 77, "Memcpy failure must retain native operation and error code.");
    Require(cuda.EventDestroyCalls == 1, "Failed pre-submission copy must reclaim its created completion event.");
    Require(cuda.StreamSynchronizeCalls == 0, "A cudaMemcpyAsync call that fails synchronously must not require stream synchronization.");
    Require(engine.ActiveCopies == 0, "Synchronous memcpy failure must return the stream slot.");

    await engine.CopyAsync(destination.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost);
    Require(destination.Read()[0] == 12f, "Stream must remain reusable after a synchronous memcpy submission failure.");
}

static async Task RunQueryFailureCleanupAsync()
{
    var cuda = new FakeCudaAsyncCopyApi
    {
        AutoComplete = false,
        NextEventQueryResult = 88
    };
    await using var engine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 1,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    using var source = NativeFloatBuffer.Create(new[] { 3f });
    using var destination = NativeFloatBuffer.Create(1);

    Exception? observed = null;
    try
    {
        await engine.CopyAsync(destination.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost);
    }
    catch (Exception exception)
    {
        observed = exception;
    }

    Require(observed is CudaRuntimeException { Operation: "cudaEventQuery", ErrorCode: 88 }, "Terminal cudaEventQuery failure must preserve native classification.");
    Require(cuda.StreamSynchronizeCalls >= 1, "Event-query failure must synchronize the submitted stream before reuse.");
    Require(cuda.EventDestroyCalls == 1, "Event-query failure must reclaim its completion event.");
    Require(engine.ActiveCopies == 0 && engine.PendingCompletions == 0, "Event-query failure must release pending state and stream admission.");

    cuda.AutoComplete = true;
    await engine.CopyAsync(destination.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost);
    Require(destination.Read()[0] == 3f, "Stream must remain reusable after query-error cleanup.");
}

static async Task RunDisposeDrainAsync()
{
    var cuda = new FakeCudaAsyncCopyApi { AutoComplete = false };
    var engine = new CudaAsyncCopyEngine(
        cuda,
        new CudaAsyncCopyEngineOptions
        {
            StreamCount = 1,
            CompletionPollInterval = TimeSpan.FromMilliseconds(1)
        });

    using var source = NativeFloatBuffer.Create(new[] { 42f });
    using var destination = NativeFloatBuffer.Create(1);

    var copy = engine.CopyAsync(destination.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost).AsTask();
    await WaitUntilAsync(
        () => cuda.EventRecordCalls == 1,
        "Dispose drain test requires one submitted CUDA copy.");

    var dispose = engine.DisposeAsync().AsTask();
    await Task.Delay(20);
    Require(!dispose.IsCompleted, "DisposeAsync must wait for in-flight native copy completion before destroying streams.");
    Require(cuda.StreamDestroyCalls == 0, "In-flight CUDA stream must not be destroyed before its completion event.");

    cuda.CompleteAllRecordedEvents();
    await copy;
    await dispose;
    Require(cuda.StreamDestroyCalls == 1, "DisposeAsync must destroy each reusable CUDA stream after draining native work.");

    var disposed = false;
    try
    {
        await engine.CopyAsync(destination.Pointer, source.Pointer, 4, CudaMemcpyKind.HostToHost);
    }
    catch (ObjectDisposedException)
    {
        disposed = true;
    }

    Require(disposed, "Disposed CUDA copy engine must reject new submissions.");
}

static void RunRuntimeProbeFailure()
{
    var created = CudaAsyncCopyEngine.TryCreate(
        out var engine,
        new CudaAsyncCopyEngineOptions
        {
            RuntimeLibraryPath = "__fission_missing_cuda_runtime_library__"
        });

    Require(!created && engine is null, "TryCreate must provide a non-throwing CUDA Runtime availability probe.");
}

static async Task WaitUntilAsync(Func<bool> predicate, string failureMessage)
{
    var timeout = Stopwatch.StartNew();
    while (!predicate())
    {
        if (timeout.Elapsed >= TimeSpan.FromSeconds(3))
        {
            throw new InvalidOperationException(failureMessage);
        }

        await Task.Delay(1);
    }
}

sealed class NativeFloatBuffer : IDisposable
{
    private nint _pointer;

    private NativeFloatBuffer(nint pointer, int length)
    {
        _pointer = pointer;
        Length = length;
    }

    public nint Pointer => _pointer != 0
        ? _pointer
        : throw new ObjectDisposedException(nameof(NativeFloatBuffer));
    public int Length { get; }

    public static NativeFloatBuffer Create(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        return new NativeFloatBuffer(
            Marshal.AllocHGlobal(checked(length * sizeof(float))),
            length);
    }

    public static NativeFloatBuffer Create(float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var buffer = Create(values.Length);
        Marshal.Copy(values, 0, buffer.Pointer, values.Length);
        return buffer;
    }

    public float[] Read()
    {
        var values = new float[Length];
        Marshal.Copy(Pointer, values, 0, Length);
        return values;
    }

    public void Dispose()
    {
        var pointer = Interlocked.Exchange(ref _pointer, 0);
        if (pointer != 0)
        {
            Marshal.FreeHGlobal(pointer);
        }
    }
}

sealed class FakeCudaAsyncCopyApi : ICudaAsyncCopyApi
{
    private readonly object _gate = new();
    private readonly HashSet<nint> _streams = new();
    private readonly Dictionary<nint, FakeEventState> _events = new();
    private readonly List<uint> _streamCreateFlags = new();
    private readonly List<uint> _eventCreateFlags = new();
    private readonly List<MemcpyCall> _memcpyCalls = new();
    private long _nextHandle = 100;
    private int _streamCreateCalls;
    private int _streamDestroyCalls;
    private int _streamSynchronizeCalls;
    private int _memcpyAsyncCalls;
    private int _eventRecordCalls;
    private int _eventQueryCalls;
    private int _eventDestroyCalls;

    public bool AutoComplete { get; set; }
    public int QueriesBeforeAutoComplete { get; set; }
    public int NextMemcpyResult { get; set; }
    public int NextEventQueryResult { get; set; }

    public int StreamCreateCalls => Volatile.Read(ref _streamCreateCalls);
    public int StreamDestroyCalls => Volatile.Read(ref _streamDestroyCalls);
    public int StreamSynchronizeCalls => Volatile.Read(ref _streamSynchronizeCalls);
    public int MemcpyAsyncCalls => Volatile.Read(ref _memcpyAsyncCalls);
    public int EventRecordCalls => Volatile.Read(ref _eventRecordCalls);
    public int EventQueryCalls => Volatile.Read(ref _eventQueryCalls);
    public int EventDestroyCalls => Volatile.Read(ref _eventDestroyCalls);

    public IReadOnlyList<uint> StreamCreateFlags
    {
        get
        {
            lock (_gate)
            {
                return _streamCreateFlags.ToArray();
            }
        }
    }

    public IReadOnlyList<uint> EventCreateFlags
    {
        get
        {
            lock (_gate)
            {
                return _eventCreateFlags.ToArray();
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

    public int StreamCreateWithFlags(out nint stream, uint flags)
    {
        Interlocked.Increment(ref _streamCreateCalls);
        lock (_gate)
        {
            stream = (nint)(++_nextHandle);
            _streams.Add(stream);
            _streamCreateFlags.Add(flags);
        }

        return 0;
    }

    public int StreamDestroy(nint stream)
    {
        Interlocked.Increment(ref _streamDestroyCalls);
        lock (_gate)
        {
            return _streams.Remove(stream) ? 0 : 17;
        }
    }

    public int StreamSynchronize(nint stream)
    {
        Interlocked.Increment(ref _streamSynchronizeCalls);
        lock (_gate)
        {
            foreach (var state in _events.Values.Where(item => item.Stream == stream))
            {
                state.Completed = true;
            }
        }

        return 0;
    }

    public int MemcpyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream)
    {
        Interlocked.Increment(ref _memcpyAsyncCalls);

        lock (_gate)
        {
            if (NextMemcpyResult != 0)
            {
                var result = NextMemcpyResult;
                NextMemcpyResult = 0;
                return result;
            }

            _memcpyCalls.Add(new MemcpyCall(destination, source, byteLength, kind, stream));
        }

        var count = checked((int)byteLength);
        var payload = new byte[count];
        Marshal.Copy(source, payload, 0, count);
        Marshal.Copy(payload, 0, destination, count);
        return 0;
    }

    public int EventCreateWithFlags(out nint completionEvent, uint flags)
    {
        lock (_gate)
        {
            completionEvent = (nint)(++_nextHandle);
            _events.Add(completionEvent, new FakeEventState());
            _eventCreateFlags.Add(flags);
        }

        return 0;
    }

    public int EventRecord(nint completionEvent, nint stream)
    {
        Interlocked.Increment(ref _eventRecordCalls);
        lock (_gate)
        {
            if (!_events.TryGetValue(completionEvent, out var state) ||
                !_streams.Contains(stream))
            {
                return 17;
            }

            state.Stream = stream;
            state.Recorded = true;
        }

        return 0;
    }

    public int EventQuery(nint completionEvent)
    {
        Interlocked.Increment(ref _eventQueryCalls);
        lock (_gate)
        {
            if (NextEventQueryResult != 0)
            {
                var result = NextEventQueryResult;
                NextEventQueryResult = 0;
                return result;
            }

            if (!_events.TryGetValue(completionEvent, out var state) || !state.Recorded)
            {
                return 17;
            }

            state.QueryCount++;
            if (state.Completed)
            {
                return 0;
            }

            if (AutoComplete && state.QueryCount > QueriesBeforeAutoComplete)
            {
                state.Completed = true;
                return 0;
            }

            return CudaAsyncCopyEngine.CudaErrorNotReady;
        }
    }

    public int EventDestroy(nint completionEvent)
    {
        Interlocked.Increment(ref _eventDestroyCalls);
        lock (_gate)
        {
            return _events.Remove(completionEvent) ? 0 : 17;
        }
    }

    public string? GetErrorString(int errorCode) => $"fake-cuda-error-{errorCode}";

    public bool CompleteOneRecordedEvent()
    {
        lock (_gate)
        {
            var state = _events.Values.FirstOrDefault(static item => item.Recorded && !item.Completed);
            if (state is null)
            {
                return false;
            }

            state.Completed = true;
            return true;
        }
    }

    public void CompleteAllRecordedEvents()
    {
        lock (_gate)
        {
            foreach (var state in _events.Values.Where(static item => item.Recorded))
            {
                state.Completed = true;
            }
        }
    }

    public readonly record struct MemcpyCall(
        nint Destination,
        nint Source,
        nuint ByteLength,
        CudaMemcpyKind Kind,
        nint Stream);

    private sealed class FakeEventState
    {
        public nint Stream { get; set; }
        public bool Recorded { get; set; }
        public bool Completed { get; set; }
        public int QueryCount { get; set; }
    }
}
