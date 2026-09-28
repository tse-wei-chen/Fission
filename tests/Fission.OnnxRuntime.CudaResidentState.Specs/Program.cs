using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static void Require(
    [DoesNotReturnIf(false)] bool condition,
    string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

RunValidCudaResidentLease();
RunPinnedHostMemoryRejected();
RunDeviceMismatchRejected();
RunShapeByteAndTypeMismatchRejected();
RunDisposedStateRejected();
RunTensorViewCopiesShape();

Console.WriteLine("Fission ONNX Runtime CUDA-resident state specs passed.");

static void RunValidCudaResidentLease()
{
    using var synthetic = SyntheticDecoderState.Create(
        allocatorName: "Cuda",
        deviceId: 2,
        memoryType: OrtMemType.Default,
        position: 3,
        nextTokenId: 17,
        shape: new long[] { 1, 2, 3, 4 });
    var lifetime = new LifetimeProbe();

    var lease = DecoderOrtCudaResidentStateLease.Create(
        "cuda-kv:test-v1",
        synthetic.State,
        deviceId: 2,
        synthetic.CreateLayerViews(deviceId: 2),
        lifetime.Retain());

    Require(lifetime.ActiveLeases == 1, "Acquiring CUDA-resident state must retain the binding-owned device allocation lifetime.");
    Require(lease.FormatId == "cuda-kv:test-v1", "CUDA-resident lease must preserve the physical-layout format id.");
    Require(lease.Position == 3 && lease.NextTokenId == 17, "CUDA-resident lease must preserve the decoder causal frontier.");
    Require(lease.DeviceId == 2, "CUDA-resident lease must preserve the CUDA device ordinal.");
    Require(lease.LayerCount == 1, "CUDA-resident lease must preserve the decoder layer count.");
    Require(lease.ByteLength == synthetic.ByteLength * 2, "CUDA-resident lease must account for key and value bytes exactly.");

    var layer = lease.GetLayer(0);
    Require(layer.Key.DevicePointer == synthetic.KeyPointer, "Key device pointer must flow unchanged through the validated lease.");
    Require(layer.Value.DevicePointer == synthetic.ValuePointer, "Value device pointer must flow unchanged through the validated lease.");
    Require(layer.Key.Shape.SequenceEqual(synthetic.Shape), "Validated key shape must match the ORT tensor shape.");

    lease.Dispose();
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Disposing CUDA-resident state must release the native allocation retain exactly once.");

    lease.Dispose();
    Require(lifetime.ReleaseCount == 1, "CUDA-resident state disposal must be idempotent.");

    var disposedGuarded = false;
    try
    {
        _ = lease.GetLayer(0);
    }
    catch (ObjectDisposedException)
    {
        disposedGuarded = true;
    }

    Require(disposedGuarded, "Device pointers must not be obtainable from a disposed CUDA-resident lease.");
}

static void RunPinnedHostMemoryRejected()
{
    using var synthetic = SyntheticDecoderState.Create(
        allocatorName: "CudaPinned",
        deviceId: 0,
        memoryType: OrtMemType.CpuOutput,
        position: 1,
        nextTokenId: 5,
        shape: new long[] { 1, 1, 1, 4 });
    var lifetime = new LifetimeProbe();

    Exception? observed = null;
    try
    {
        _ = DecoderOrtCudaResidentStateLease.Create(
            "cuda-kv:test-v1",
            synthetic.State,
            deviceId: 0,
            synthetic.CreateLayerViews(deviceId: 0),
            lifetime.Retain());
    }
    catch (Exception exception)
    {
        observed = exception;
    }

    Require(observed is InvalidOperationException, "CUDA-pinned host memory must not be accepted as CUDA device-resident KV memory.");
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Validation failure must release the transferred native lifetime lease.");
}

static void RunDeviceMismatchRejected()
{
    using var synthetic = SyntheticDecoderState.Create(
        allocatorName: "Cuda",
        deviceId: 1,
        memoryType: OrtMemType.Default,
        position: 2,
        nextTokenId: 9,
        shape: new long[] { 1, 1, 2, 4 });
    var lifetime = new LifetimeProbe();

    Exception? observed = null;
    try
    {
        _ = DecoderOrtCudaResidentStateLease.Create(
            "cuda-kv:test-v1",
            synthetic.State,
            deviceId: 0,
            synthetic.CreateLayerViews(deviceId: 1),
            lifetime.Retain());
    }
    catch (Exception exception)
    {
        observed = exception;
    }

    Require(observed is InvalidOperationException, "A CUDA-resident state lease must reject pointers from a different device ordinal.");
    Require(lifetime.ReleaseCount == 1, "Device mismatch failure must release the transferred native lifetime lease.");
}

static void RunShapeByteAndTypeMismatchRejected()
{
    using var synthetic = SyntheticDecoderState.Create(
        allocatorName: "Cuda",
        deviceId: 0,
        memoryType: OrtMemType.Default,
        position: 2,
        nextTokenId: 11,
        shape: new long[] { 1, 1, 2, 4 });

    var shapeLifetime = new LifetimeProbe();
    var shapeViews = synthetic.CreateLayerViews(deviceId: 0);
    shapeViews[0] = shapeViews[0] with
    {
        Key = new CudaDeviceTensorView(
            synthetic.KeyPointer,
            synthetic.ByteLength,
            TensorElementType.Float,
            new long[] { 1, 1, 1, 8 },
            0)
    };
    RequireValidationFailure(
        synthetic.State,
        shapeViews,
        shapeLifetime,
        "Shape mismatch must be rejected before a CUDA transport can consume the pointer.");

    var byteLifetime = new LifetimeProbe();
    var byteViews = synthetic.CreateLayerViews(deviceId: 0);
    byteViews[0] = byteViews[0] with
    {
        Value = new CudaDeviceTensorView(
            synthetic.ValuePointer,
            synthetic.ByteLength + sizeof(float),
            TensorElementType.Float,
            synthetic.Shape,
            0)
    };
    RequireValidationFailure(
        synthetic.State,
        byteViews,
        byteLifetime,
        "Byte-length mismatch must be rejected before a CUDA transport can consume the pointer.");

    var typeLifetime = new LifetimeProbe();
    var typeViews = synthetic.CreateLayerViews(deviceId: 0);
    typeViews[0] = typeViews[0] with
    {
        Key = new CudaDeviceTensorView(
            synthetic.KeyPointer,
            synthetic.ByteLength,
            TensorElementType.Int32,
            synthetic.Shape,
            0)
    };
    RequireValidationFailure(
        synthetic.State,
        typeViews,
        typeLifetime,
        "Element-type mismatch must be rejected before a CUDA transport can consume the pointer.");
}

static void RunDisposedStateRejected()
{
    var synthetic = SyntheticDecoderState.Create(
        allocatorName: "Cuda",
        deviceId: 0,
        memoryType: OrtMemType.Default,
        position: 1,
        nextTokenId: 3,
        shape: new long[] { 1, 1, 1, 4 });
    var views = synthetic.CreateLayerViews(deviceId: 0);
    synthetic.State.Dispose();

    var lifetime = new LifetimeProbe();
    Exception? observed = null;
    try
    {
        _ = DecoderOrtCudaResidentStateLease.Create(
            "cuda-kv:test-v1",
            synthetic.State,
            deviceId: 0,
            views,
            lifetime.Retain());
    }
    catch (Exception exception)
    {
        observed = exception;
    }
    finally
    {
        synthetic.Dispose();
    }

    Require(observed is ObjectDisposedException, "Disposed decoder state must not publish CUDA-resident device pointers.");
    Require(lifetime.ReleaseCount == 1, "Disposed-state rejection must release the transferred native lifetime lease.");
}

static void RunTensorViewCopiesShape()
{
    var shape = new long[] { 1, 2, 3, 4 };
    var view = new CudaDeviceTensorView(
        (nint)123,
        96,
        TensorElementType.Float,
        shape,
        0);

    shape[2] = 99;
    Require(view.Shape.SequenceEqual(new long[] { 1, 2, 3, 4 }), "CUDA tensor view must snapshot shape metadata rather than alias caller-owned arrays.");
}

static void RequireValidationFailure(
    DecoderOrtState state,
    IReadOnlyList<DecoderOrtCudaLayerView> layers,
    LifetimeProbe lifetime,
    string message)
{
    Exception? observed = null;
    try
    {
        _ = DecoderOrtCudaResidentStateLease.Create(
            "cuda-kv:test-v1",
            state,
            deviceId: 0,
            layers,
            lifetime.Retain());
    }
    catch (Exception exception)
    {
        observed = exception;
    }

    Require(observed is InvalidOperationException, message);
    Require(lifetime.ActiveLeases == 0 && lifetime.ReleaseCount == 1, "Validation failure must deterministically release the transferred lifetime lease.");
}

sealed class SyntheticDecoderState : IDisposable
{
    private readonly OrtMemoryInfo _memoryInfo;
    private int _disposed;

    private SyntheticDecoderState(
        OrtMemoryInfo memoryInfo,
        nint keyPointer,
        nint valuePointer,
        long byteLength,
        long[] shape,
        int deviceId,
        DecoderOrtState state)
    {
        _memoryInfo = memoryInfo;
        KeyPointer = keyPointer;
        ValuePointer = valuePointer;
        ByteLength = byteLength;
        Shape = shape;
        DeviceId = deviceId;
        State = state;
    }

    public nint KeyPointer { get; }
    public nint ValuePointer { get; }
    public long ByteLength { get; }
    public long[] Shape { get; }
    public int DeviceId { get; }
    public DecoderOrtState State { get; }

    public static SyntheticDecoderState Create(
        string allocatorName,
        int deviceId,
        OrtMemType memoryType,
        int position,
        int? nextTokenId,
        long[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var elementCount = 1L;
        foreach (var dimension in shape)
        {
            elementCount = checked(elementCount * dimension);
        }

        var byteLength = checked(elementCount * sizeof(float));
        var allocationLength = checked((int)byteLength);
        var keyPointer = Marshal.AllocHGlobal(allocationLength);
        var valuePointer = Marshal.AllocHGlobal(allocationLength);
        var memoryInfo = new OrtMemoryInfo(
            allocatorName,
            OrtAllocatorType.DeviceAllocator,
            deviceId,
            memoryType);
        OrtValue? key = null;
        OrtValue? value = null;
        try
        {
            key = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                shape,
                keyPointer,
                byteLength);
            value = OrtValue.CreateTensorValueWithData(
                memoryInfo,
                TensorElementType.Float,
                shape,
                valuePointer,
                byteLength);
            var state = new DecoderOrtState(
                position,
                new[] { new DecoderOrtLayerState(key, value) },
                nextTokenId);
            key = null;
            value = null;
            return new SyntheticDecoderState(
                memoryInfo,
                keyPointer,
                valuePointer,
                byteLength,
                (long[])shape.Clone(),
                deviceId,
                state);
        }
        catch
        {
            value?.Dispose();
            key?.Dispose();
            memoryInfo.Dispose();
            Marshal.FreeHGlobal(valuePointer);
            Marshal.FreeHGlobal(keyPointer);
            throw;
        }
    }

    public DecoderOrtCudaLayerView[] CreateLayerViews(int deviceId) =>
        new[]
        {
            new DecoderOrtCudaLayerView(
                new CudaDeviceTensorView(
                    KeyPointer,
                    ByteLength,
                    TensorElementType.Float,
                    Shape,
                    deviceId),
                new CudaDeviceTensorView(
                    ValuePointer,
                    ByteLength,
                    TensorElementType.Float,
                    Shape,
                    deviceId))
        };

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

sealed class LifetimeProbe
{
    private int _activeLeases;
    private int _releaseCount;

    public int ActiveLeases => Volatile.Read(ref _activeLeases);
    public int ReleaseCount => Volatile.Read(ref _releaseCount);

    public IDisposable Retain()
    {
        Interlocked.Increment(ref _activeLeases);
        return new ProbeLease(this);
    }

    private void Release()
    {
        Interlocked.Decrement(ref _activeLeases);
        Interlocked.Increment(ref _releaseCount);
    }

    private sealed class ProbeLease : IDisposable
    {
        private LifetimeProbe? _owner;

        public ProbeLease(LifetimeProbe owner)
        {
            _owner = owner;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?.Release();
        }
    }
}
