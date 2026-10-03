using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

internal static class Fp16GatherSpecs
{
    [ModuleInitializer]
    internal static void Initialize() =>
        RunAsync().GetAwaiter().GetResult();

    private static async Task RunAsync()
    {
        const int deviceId = 7;
        const string formatId = "gather-spec:fp16:v1";
        var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
            numHiddenLayers: 1,
            numKvHeads: 1,
            headDim: 1,
            vocabularySize: 8,
            kvElementType: TensorElementType.Float16,
            logitsElementType: TensorElementType.Float16);
        var cuda = new Fp16FakeCudaRuntime(initialDevice: 13);
        var allocator = new CudaDeviceMemoryAllocator(
            cuda,
            new CudaDeviceMemoryAllocatorOptions { DeviceId = deviceId });
        await using var copyEngine = new CudaDeviceBoundAsyncCopyEngine(
            deviceId,
            cuda,
            cuda,
            new CudaAsyncCopyEngineOptions
            {
                StreamCount = 2,
                CompletionPollInterval = TimeSpan.FromMilliseconds(1)
            });

        var shape = profile.Geometry.GetPastKvShape(
            batchSize: 1,
            pastSequenceLength: 2);
        var sourceArena = new CudaDecoderOrtCohortArena(
            position: 2,
            batchSize: 2,
            layerCount: 1,
            perSequenceShape: shape,
            allocator,
            profile.Geometry.KvElementType);
        var row0 = sourceArena.CreateRowState(0, nextTokenId: 1);
        var row1 = sourceArena.CreateRowState(1, nextTokenId: 2);
        sourceArena.Release();

        var sourcePointers = cuda.ActiveAllocationPointers.ToArray();
        Require(sourcePointers.Length == 2,
            "One FP16 source arena with one layer must own key/value allocations.");
        cuda.WriteBytes(sourcePointers[0],
            [0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14]);
        cuda.WriteBytes(sourcePointers[1],
            [0x21, 0x22, 0x23, 0x24, 0x31, 0x32, 0x33, 0x34]);

        using var gathered = await OptimumLegacyCudaGatheredPastKvBatch.CreateAsync(
            profile,
            new[] { row1, row0 },
            formatId,
            allocator,
            copyEngine);

        Require(gathered.CopiedBytes == 16,
            "Two rows x one layer x key/value x two FP16 elements must copy 16 bytes.");
        Require(gathered.InputValues.All(static value => value.GetTensorSizeInBytes() == 8),
            "Two gathered FP16 rows must expose eight bytes per key/value tensor.");
        Require(cuda.MemcpyCalls.Count == 4 &&
                cuda.MemcpyCalls.All(static call =>
                    call.Kind == CudaMemcpyKind.DeviceToDevice && call.ByteLength == 4),
            "Reverse-order FP16 gather must issue four independent four-byte D2D copies.");

        var destinationPointers = cuda.ActiveAllocationPointers.Skip(2).ToArray();
        Require(destinationPointers.Length == 2,
            "FP16 gather destination must own one key/value allocation pair.");
        Require(cuda.ReadBytes(destinationPointers[0], 8).SequenceEqual(
            new byte[] { 0x11, 0x12, 0x13, 0x14, 0x01, 0x02, 0x03, 0x04 }),
            "FP16 gathered key rows must preserve requested reverse order.");
        Require(cuda.ReadBytes(destinationPointers[1], 8).SequenceEqual(
            new byte[] { 0x31, 0x32, 0x33, 0x34, 0x21, 0x22, 0x23, 0x24 }),
            "FP16 gathered value rows must preserve requested reverse order.");

        using (var gatheredRow = gathered.CreateRowState(0, nextTokenId: null))
        using (var lease = gatheredRow.AcquireCudaResidentState(formatId))
        {
            var layer = lease.GetLayer(0);
            Require(layer.Key.ElementType == TensorElementType.Float16 &&
                    layer.Value.ElementType == TensorElementType.Float16,
                "Gathered resident rows must retain FP16 KV element type.");
            Require(layer.Key.ByteLength == 4 && layer.Value.ByteLength == 4,
                "Gathered resident FP16 row views must use two-byte element stride.");
        }

        row0.Dispose();
        row1.Dispose();
        Require(cuda.ActiveAllocationPointers.Count == 2,
            "Gather destination must remain alive after FP16 source rows are released.");

        gathered.Dispose();
        Require(cuda.ActiveAllocationPointers.Count == 0,
            "FP16 gather disposal must release every CUDA allocation.");
        Require(cuda.CurrentDevice == 13,
            "FP16 gather must restore the ambient CUDA device.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class Fp16FakeCudaRuntime : ICudaDeviceMemoryApi, ICudaAsyncCopyApi
    {
        private readonly object _gate = new();
        private readonly ThreadLocal<int> _currentDevice;
        private readonly List<nint> _allocationOrder = new();
        private readonly HashSet<nint> _activeAllocations = new();
        private readonly Dictionary<nint, FakeEvent> _events = new();
        private readonly List<MemcpyCall> _memcpyCalls = new();
        private long _nextHandle = 2000;

        public Fp16FakeCudaRuntime(int initialDevice)
        {
            _currentDevice = new ThreadLocal<int>(() => initialDevice);
        }

        public int CurrentDevice => _currentDevice.Value;

        public IReadOnlyList<nint> ActiveAllocationPointers
        {
            get
            {
                lock (_gate)
                {
                    return _allocationOrder.Where(_activeAllocations.Contains).ToArray();
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
            lock (_gate)
            {
                _activeAllocations.Add(pointer);
                _allocationOrder.Add(pointer);
            }

            return 0;
        }

        public int Free(nint pointer)
        {
            lock (_gate)
            {
                if (!_activeAllocations.Remove(pointer))
                {
                    return 17;
                }
            }

            Marshal.FreeHGlobal(pointer);
            return 0;
        }

        public int StreamCreateWithFlags(out nint stream, uint flags)
        {
            _ = flags;
            stream = (nint)Interlocked.Increment(ref _nextHandle);
            return 0;
        }

        public int StreamDestroy(nint stream)
        {
            _ = stream;
            return 0;
        }

        public int StreamSynchronize(nint stream)
        {
            lock (_gate)
            {
                foreach (var state in _events.Values.Where(state => state.Stream == stream))
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
            var count = checked((int)byteLength);
            var bytes = new byte[count];
            Marshal.Copy(source, bytes, 0, count);
            Marshal.Copy(bytes, 0, destination, count);
            lock (_gate)
            {
                _memcpyCalls.Add(new MemcpyCall(destination, source, byteLength, kind, stream));
            }

            return 0;
        }

        public int EventCreateWithFlags(out nint completionEvent, uint flags)
        {
            _ = flags;
            completionEvent = (nint)Interlocked.Increment(ref _nextHandle);
            lock (_gate)
            {
                _events.Add(completionEvent, new FakeEvent());
            }

            return 0;
        }

        public int EventRecord(nint completionEvent, nint stream)
        {
            lock (_gate)
            {
                if (!_events.TryGetValue(completionEvent, out var state))
                {
                    return 17;
                }

                state.Stream = stream;
                state.Completed = true;
            }

            return 0;
        }

        public int EventQuery(nint completionEvent)
        {
            lock (_gate)
            {
                if (!_events.TryGetValue(completionEvent, out var state))
                {
                    return 17;
                }

                return state.Completed ? 0 : CudaAsyncCopyEngine.CudaErrorNotReady;
            }
        }

        public int EventDestroy(nint completionEvent)
        {
            lock (_gate)
            {
                return _events.Remove(completionEvent) ? 0 : 17;
            }
        }

        public string? GetErrorString(int errorCode) => $"fake CUDA error {errorCode}";

        public void WriteBytes(nint pointer, byte[] values) =>
            Marshal.Copy(values, 0, pointer, values.Length);

        public byte[] ReadBytes(nint pointer, int count)
        {
            var values = new byte[count];
            Marshal.Copy(pointer, values, 0, count);
            return values;
        }

        private sealed class FakeEvent
        {
            public nint Stream { get; set; }
            public bool Completed { get; set; }
        }

        public readonly record struct MemcpyCall(
            nint Destination,
            nint Source,
            nuint ByteLength,
            CudaMemcpyKind Kind,
            nint Stream);
    }
}
