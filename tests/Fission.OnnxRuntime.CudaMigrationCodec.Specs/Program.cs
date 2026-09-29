using System.Runtime.InteropServices;
using Fission.Abstractions.Execution;
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

const string cudaFormat = "cuda-kv:codec-roundtrip-v1";
const int sourceDevice = 0;
const int targetDevice = 2;
var options = new CudaAsyncCopyEngineOptions
{
    StreamCount = 2,
    CompletionPollInterval = TimeSpan.FromMilliseconds(1)
};

using var sourceState = SourceCudaState.Create(
    sourceDevice,
    position: 4,
    nextTokenId: 19,
    new[] { 1f, 2f, 3f, 4f },
    new[] { 11f, 12f, 13f, 14f });
using var binding = new FakeCudaResidentBinding(
    cudaFormat,
    sourceDevice,
    sourceState.KeyPointer,
    sourceState.ValuePointer);
var sourceCuda = new FakeCudaRuntime(initialDevice: 6);
await using var sourceCopy = new CudaAsyncCopyEngine(sourceCuda, options);
using var exporter = new CudaDeviceToHostStagingExporter(
    sourceCopy,
    new FakeCudaPageLockedAllocator(),
    new PinnedHostStagingPoolOptions
    {
        MaxRetainedBuffersPerLength = 4,
        MaxRetainedBytes = 4096,
        ClearOnReturn = true
    });

var targetCuda = new FakeCudaRuntime(initialDevice: 8);
await using var targetCopy = new CudaDeviceBoundAsyncCopyEngine(
    targetDevice,
    targetCuda,
    targetCuda,
    options);
var targetAllocator = new CudaDeviceMemoryAllocator(
    targetCuda,
    new CudaDeviceMemoryAllocatorOptions { DeviceId = targetDevice });
var importer = new CudaHostToDeviceStateImporter(
    targetAllocator,
    targetCopy);
var codec = new CudaHostStagingMigrationCodec(
    binding,
    exporter,
    importer);

Require(
    codec.Name.Contains(binding.Name, StringComparison.Ordinal),
    "CUDA migration codec name must identify its resident binding.");
Require(
    codec.CudaResidentStateFormatId == cudaFormat,
    "CUDA migration codec must preserve the resident physical-layout format id.");
Require(
    codec.HostStagingFormatId == CudaDeviceToHostStagingExporter.BuildHostStagingFormatId(cudaFormat),
    "CUDA migration codec must derive the negotiated host format from the resident CUDA format.");
Require(codec.TargetDeviceId == targetDevice, "CUDA migration codec must expose the H2D target ordinal.");

var estimated = codec.EstimateHostStagingBytes(sourceState.State);
Require(estimated == 40, $"One 16-byte key + 16-byte value + 8-byte frontier must estimate 40 bytes, got {estimated}.");
Require(binding.AcquireCount == 1, "Byte estimation must acquire and release one validated resident-state lease.");

var syncExportRejected = false;
try
{
    _ = codec.ExportHostStagingState(sourceState.State);
}
catch (NotSupportedException)
{
    syncExportRejected = true;
}
Require(syncExportRejected, "CUDA codec must reject synchronous export instead of blocking on native completion.");

var payload = await codec.ExportHostStagingStateAsync(sourceState.State);
Require(payload is DecoderOrtCudaHostStagingPayload, "CUDA codec export must produce the CUDA page-locked payload type.");
Require(payload.ByteLength == estimated, "CUDA codec export byte length must match its estimate.");
Require(payload.Position == sourceState.State.Position && payload.NextTokenId == sourceState.State.NextTokenId, "CUDA codec export must preserve causal frontier.");
Require(binding.AcquireCount == 2, "Async export must acquire a fresh validated resident-state lease.");
Require(sourceCuda.MemcpyKinds.Count == 2 && sourceCuda.MemcpyKinds.All(static kind => kind == CudaMemcpyKind.DeviceToHost), "CUDA codec export must submit key/value D2H copies.");

var syncImportRejected = false;
try
{
    _ = codec.ImportHostStagingState(payload);
}
catch (NotSupportedException)
{
    syncImportRejected = true;
}
Require(syncImportRejected, "CUDA codec must reject synchronous import instead of blocking on native completion.");

var imported = await codec.ImportHostStagingStateAsync(payload);
Require(imported.Position == 4 && imported.NextTokenId == 19, "CUDA codec import must preserve causal frontier.");
Require(imported.LayerCount == 1, "CUDA codec import must preserve decoder layer count.");
Require(targetCuda.MemcpyKinds.Count == 2 && targetCuda.MemcpyKinds.All(static kind => kind == CudaMemcpyKind.HostToDevice), "CUDA codec import must submit key/value H2D copies.");
Require(targetCuda.OperationDevices.Count != 0 && targetCuda.OperationDevices.All(static device => device == targetDevice), "Target CUDA allocation/copy operations must execute on the configured target ordinal.");
Require(targetCuda.CurrentDevice == 8, "Target device-scoped operations must restore the ambient CUDA device.");

var importedLayer = imported.GetLayer(0);
using (var keyInfo = importedLayer.Key.GetTensorMemoryInfo())
using (var valueInfo = importedLayer.Value.GetTensorMemoryInfo())
{
    Require(keyInfo.Name == "Cuda" && valueInfo.Name == "Cuda", "Imported codec state must expose CUDA OrtValues.");
    Require(keyInfo.Id == targetDevice && valueInfo.Id == targetDevice, "Imported codec state must expose the target CUDA ordinal.");
}

var targetPointers = targetCuda.ActiveAllocationPointers;
Require(targetPointers.Count == 2, "Imported codec state must own one target key and one target value allocation.");
Require(targetCuda.ReadFloats(targetPointers[0], 4).SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "Round-trip key bytes must match the source CUDA state.");
Require(targetCuda.ReadFloats(targetPointers[1], 4).SequenceEqual(new[] { 11f, 12f, 13f, 14f }), "Round-trip value bytes must match the source CUDA state.");

var wrongFormatRejected = false;
try
{
    _ = imported.AcquireCudaResidentState("cuda-kv:wrong-format");
}
catch (InvalidOperationException)
{
    wrongFormatRejected = true;
}
Require(wrongFormatRejected, "Imported CUDA state must reject a resident borrow with a different physical-layout format id.");
Require(targetCuda.FreeCalls == 0, "Rejected resident borrow must not release target CUDA allocations.");

var importedBorrow = imported.AcquireCudaResidentState(cudaFormat);
Require(importedBorrow.FormatId == cudaFormat, "Imported resident borrow must preserve CUDA physical-layout format id.");
Require(importedBorrow.DeviceId == targetDevice, "Imported resident borrow must preserve target CUDA ordinal.");
Require(importedBorrow.Position == 4 && importedBorrow.NextTokenId == 19, "Imported resident borrow must preserve causal frontier.");
Require(importedBorrow.ByteLength == 32, "Imported resident borrow must expose exact key/value device bytes.");
var borrowedLayer = importedBorrow.GetLayer(0);
Require(borrowedLayer.Key.DevicePointer == targetPointers[0], "Imported resident borrow must expose the target key allocation pointer.");
Require(borrowedLayer.Value.DevicePointer == targetPointers[1], "Imported resident borrow must expose the target value allocation pointer.");

payload.Dispose();
Require(targetCuda.FreeCalls == 0, "Releasing host staging after import must not release target CUDA state.");
imported.Dispose();
Require(targetCuda.FreeCalls == 0 && targetCuda.ActiveAllocationPointers.Count == 2, "Imported state disposal must retain target CUDA allocations while an independent resident borrow is alive.");
Require(targetCuda.ReadFloats(borrowedLayer.Key.DevicePointer, 4).SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "Resident borrow key pointer must remain valid after imported state disposal.");
Require(targetCuda.ReadFloats(borrowedLayer.Value.DevicePointer, 4).SequenceEqual(new[] { 11f, 12f, 13f, 14f }), "Resident borrow value pointer must remain valid after imported state disposal.");

importedBorrow.Dispose();
Require(targetCuda.FreeCalls == 2 && targetCuda.ActiveAllocationPointers.Count == 0, "Final imported resident-borrow release must release target CUDA allocations exactly once.");
Require(binding.DisposeCount == 0, "Using the standalone codec must not dispose its execution/resident binding.");

Console.WriteLine(
    $"Fission CUDA migration codec specs passed: estimate={estimated}, " +
    $"sourceCopies={sourceCuda.MemcpyKinds.Count}, targetCopies={targetCuda.MemcpyKinds.Count}, " +
    $"acquires={binding.AcquireCount}.");

sealed class SourceCudaState : IDisposable
{
    private readonly OrtMemoryInfo _memoryInfo;
    private int _disposed;

    private SourceCudaState(
        nint keyPointer,
        nint valuePointer,
        OrtMemoryInfo memoryInfo,
        DecoderOrtState state)
    {
        KeyPointer = keyPointer;
        ValuePointer = valuePointer;
        _memoryInfo = memoryInfo;
        State = state;
    }

    public nint KeyPointer { get; }
    public nint ValuePointer { get; }
    public DecoderOrtState State { get; }

    public static SourceCudaState Create(
        int deviceId,
        int position,
        int nextTokenId,
        float[] keyValues,
        float[] valueValues)
    {
        var keyPointer = Marshal.AllocHGlobal(checked(keyValues.Length * sizeof(float)));
        var valuePointer = Marshal.AllocHGlobal(checked(valueValues.Length * sizeof(float)));
        OrtMemoryInfo? memoryInfo = null;
        OrtValue? key = null;
        OrtValue? value = null;
        try
        {
            Marshal.Copy(keyValues, 0, keyPointer, keyValues.Length);
            Marshal.Copy(valueValues, 0, valuePointer, valueValues.Length);
            memoryInfo = new OrtMemoryInfo(
                "Cuda",
                OrtAllocatorType.DeviceAllocator,
                deviceId,
                OrtMemType.Default);
            var shape = new long[] { 1, 1, keyValues.Length, 1 };
            key = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                shape,
                keyPointer,
                checked((long)keyValues.Length * sizeof(float)));
            value = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                shape,
                valuePointer,
                checked((long)valueValues.Length * sizeof(float)));
            var state = new DecoderOrtState(
                position,
                new[] { new DecoderOrtLayerState(key, value) },
                nextTokenId);
            key = null;
            value = null;
            return new SourceCudaState(
                keyPointer,
                valuePointer,
                memoryInfo,
                state);
        }
        catch
        {
            value?.Dispose();
            key?.Dispose();
            memoryInfo?.Dispose();
            Marshal.FreeHGlobal(valuePointer);
            Marshal.FreeHGlobal(keyPointer);
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        State.Dispose();
        _memoryInfo.Dispose();
        Marshal.FreeHGlobal(ValuePointer);
        Marshal.FreeHGlobal(KeyPointer);
    }
}

sealed class FakeCudaResidentBinding : IDecoderOrtCudaResidentStateBinding
{
    private readonly int _deviceId;
    private readonly nint _keyPointer;
    private readonly nint _valuePointer;
    private int _acquireCount;
    private int _disposeCount;

    public FakeCudaResidentBinding(
        string formatId,
        int deviceId,
        nint keyPointer,
        nint valuePointer)
    {
        CudaResidentStateFormatId = formatId;
        _deviceId = deviceId;
        _keyPointer = keyPointer;
        _valuePointer = valuePointer;
    }

    public string Name => "fake-cuda-resident";
    public string CudaResidentStateFormatId { get; }
    public int AcquireCount => Volatile.Read(ref _acquireCount);
    public int DisposeCount => Volatile.Read(ref _disposeCount);
    public OnnxSessionContract SessionContract { get; } = new(
        Array.Empty<OnnxTensorContract>(),
        Array.Empty<OnnxTensorContract>());

    public DecoderOrtCudaResidentStateLease AcquireCudaResidentState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _acquireCount);
        var shape = new long[] { 1, 1, 4, 1 };
        return DecoderOrtCudaResidentStateLease.Create(
            CudaResidentStateFormatId,
            state,
            _deviceId,
            new[]
            {
                new DecoderOrtCudaLayerView(
                    new CudaDeviceTensorView(
                        _keyPointer,
                        16,
                        TensorElementType.Float,
                        shape,
                        _deviceId),
                    new CudaDeviceTensorView(
                        _valuePointer,
                        16,
                        TensorElementType.Float,
                        shape,
                        _deviceId))
            },
            new NoopLease());
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public void Dispose() => Interlocked.Increment(ref _disposeCount);

    private sealed class NoopLease : IDisposable
    {
        public void Dispose()
        {
        }
    }
}

sealed class FakeCudaPageLockedAllocator : IHostStagingFloatBufferAllocator
{
    public IHostStagingFloatBuffer Allocate(int length) => new Buffer(length);

    private sealed class Buffer : ICudaPageLockedHostStagingFloatBuffer
    {
        private readonly float[] _values;
        private GCHandle _pin;
        private int _disposed;

        public Buffer(int length)
        {
            _values = new float[length];
            _pin = GCHandle.Alloc(_values, GCHandleType.Pinned);
        }

        public Memory<float> Memory
        {
            get
            {
                ThrowIfDisposed();
                return _values;
            }
        }

        public nint Pointer
        {
            get
            {
                ThrowIfDisposed();
                return _pin.AddrOfPinnedObject();
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && _pin.IsAllocated)
            {
                _pin.Free();
            }
        }

        private void ThrowIfDisposed() =>
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}

sealed class FakeCudaRuntime : ICudaAsyncCopyApi, ICudaDeviceMemoryApi
{
    private readonly object _gate = new();
    private readonly ThreadLocal<int> _device;
    private readonly HashSet<nint> _streams = new();
    private readonly HashSet<nint> _events = new();
    private readonly List<CudaMemcpyKind> _memcpyKinds = new();
    private readonly List<int> _operationDevices = new();
    private readonly List<nint> _allocationOrder = new();
    private readonly HashSet<nint> _allocations = new();
    private long _nextHandle = 100;
    private int _freeCalls;

    public FakeCudaRuntime(int initialDevice)
    {
        _device = new ThreadLocal<int>(() => initialDevice);
    }

    public int CurrentDevice => _device.Value;
    public int FreeCalls => Volatile.Read(ref _freeCalls);

    public IReadOnlyList<CudaMemcpyKind> MemcpyKinds
    {
        get
        {
            lock (_gate)
            {
                return _memcpyKinds.ToArray();
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

    public IReadOnlyList<nint> ActiveAllocationPointers
    {
        get
        {
            lock (_gate)
            {
                return _allocationOrder.Where(_allocations.Contains).ToArray();
            }
        }
    }

    public float[] ReadFloats(nint pointer, int count)
    {
        lock (_gate)
        {
            if (!_allocations.Contains(pointer))
            {
                throw new ObjectDisposedException("fake CUDA allocation");
            }
        }

        var values = new float[count];
        Marshal.Copy(pointer, values, 0, count);
        return values;
    }

    public int GetDevice(out int deviceId)
    {
        deviceId = _device.Value;
        return 0;
    }

    public int SetDevice(int deviceId)
    {
        _device.Value = deviceId;
        return 0;
    }

    public int Malloc(out nint pointer, nuint byteLength)
    {
        RecordDevice();
        pointer = Marshal.AllocHGlobal(checked((int)byteLength));
        lock (_gate)
        {
            _allocations.Add(pointer);
            _allocationOrder.Add(pointer);
        }
        return 0;
    }

    public int Free(nint pointer)
    {
        RecordDevice();
        lock (_gate)
        {
            if (!_allocations.Remove(pointer))
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
        RecordDevice();
        stream = NewHandle();
        lock (_gate)
        {
            _streams.Add(stream);
        }
        return 0;
    }

    public int StreamDestroy(nint stream)
    {
        RecordDevice();
        lock (_gate)
        {
            _streams.Remove(stream);
        }
        return 0;
    }

    public int StreamSynchronize(nint stream)
    {
        RecordDevice();
        return 0;
    }

    public unsafe int MemcpyAsync(
        nint destination,
        nint source,
        nuint byteLength,
        CudaMemcpyKind kind,
        nint stream)
    {
        RecordDevice();
        Buffer.MemoryCopy(
            (void*)source,
            (void*)destination,
            checked((long)byteLength),
            checked((long)byteLength));
        lock (_gate)
        {
            _memcpyKinds.Add(kind);
        }
        return 0;
    }

    public int EventCreateWithFlags(out nint completionEvent, uint flags)
    {
        RecordDevice();
        completionEvent = NewHandle();
        lock (_gate)
        {
            _events.Add(completionEvent);
        }
        return 0;
    }

    public int EventRecord(nint completionEvent, nint stream)
    {
        RecordDevice();
        return 0;
    }

    public int EventQuery(nint completionEvent)
    {
        RecordDevice();
        lock (_gate)
        {
            return _events.Contains(completionEvent) ? 0 : 9;
        }
    }

    public int EventDestroy(nint completionEvent)
    {
        RecordDevice();
        lock (_gate)
        {
            _events.Remove(completionEvent);
        }
        return 0;
    }

    public string? GetErrorString(int errorCode) => $"fake CUDA error {errorCode}";

    private nint NewHandle() => checked((nint)Interlocked.Increment(ref _nextHandle));

    private void RecordDevice()
    {
        lock (_gate)
        {
            _operationDevices.Add(_device.Value);
        }
    }
}
