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

RunPayloadLifetimeLease();
RunPinnedPoolReuse();
RunPinnedPoolRetentionLimit();
RunAllocatorBoundary();

Console.WriteLine("Fission ONNX Runtime pinned host-staging specs passed.");

static void RunPayloadLifetimeLease()
{
    var payload = new CountingPayload();
    var lease = payload.Retain();
    var key = OrtValue.CreateTensorValueFromMemory(payload.Key, new long[] { 1, 2 });
    var value = OrtValue.CreateTensorValueFromMemory(payload.Value, new long[] { 1, 2 });
    var state = new DecoderOrtState(
        position: 1,
        new[] { new DecoderOrtLayerState(key, value) },
        nextTokenId: 3,
        lifetimeAnchor: lease);

    payload.Dispose();
    Require(!payload.IsDisposed, "Releasing the transfer owner must not reclaim payload memory while a state lease is alive.");
    Require(payload.DisposeCoreCalls == 0, "Payload resources must remain alive through the imported state's OrtValues.");

    Require(
        state.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(new[] { 1f, 2f }),
        "Retained payload memory must remain readable after transfer ownership is released.");

    state.Dispose();
    Require(payload.IsDisposed, "Disposing the final state lease must reclaim payload resources.");
    Require(payload.DisposeCoreCalls == 1, "Payload resources must be reclaimed exactly once.");

    state.Dispose();
    payload.Dispose();
    Require(payload.DisposeCoreCalls == 1, "State/payload disposal must remain idempotent.");
}

static void RunPinnedPoolReuse()
{
    var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
        numHiddenLayers: 1,
        numKvHeads: 1,
        headDim: 2,
        vocabularySize: 16,
        kvElementType: TensorElementType.Float);
    using var binding = new OptimumLegacyFloatHostStagingBinding(
        new OptimumLegacyFloatDecoderBinding(profile),
        new PinnedHostStagingPoolOptions
        {
            MaxRetainedBuffersPerLength = 2,
            MaxRetainedBytes = 48,
            ClearOnReturn = true
        });

    var sourceKeyData = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
    var sourceValueData = new[] { 11f, 12f, 13f, 14f, 15f, 16f };
    var shape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 3);
    using var source = new DecoderOrtState(
        position: 3,
        new[]
        {
            new DecoderOrtLayerState(
                OrtValue.CreateTensorValueFromMemory(sourceKeyData, shape),
                OrtValue.CreateTensorValueFromMemory(sourceValueData, shape))
        },
        nextTokenId: 7);

    var payload1 = binding.ExportHostStagingState(source);
    var afterFirstExport = binding.HostStagingPoolStatistics;
    Require(afterFirstExport.AllocatedBuffers == 2, "First one-layer export must allocate one pinned key and one pinned value buffer.");
    Require(afterFirstExport.ReusedBuffers == 0, "First export cannot reuse staging buffers.");
    Require(afterFirstExport.RetainedBuffers == 0, "Rented staging buffers are active, not retained in the pool.");

    var imported1 = binding.ImportHostStagingState(payload1);
    payload1.Dispose();
    var whileImportedAlive = binding.HostStagingPoolStatistics;
    Require(whileImportedAlive.RetainedBuffers == 0, "Transfer-owner release must not return buffers while imported OrtValues still alias them.");
    Require(
        imported1.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(sourceKeyData),
        "Imported pinned key payload must preserve source bytes.");
    Require(
        imported1.GetLayer(0).Value.GetTensorDataAsSpan<float>().SequenceEqual(sourceValueData),
        "Imported pinned value payload must preserve source bytes.");

    imported1.Dispose();
    var afterImportedDispose = binding.HostStagingPoolStatistics;
    Require(afterImportedDispose.ReturnedBuffers == 2, "Final imported-state disposal must return both staging buffers.");
    Require(afterImportedDispose.RetainedBuffers == 2, "Both exact-length buffers should be retained for reuse.");
    Require(afterImportedDispose.RetainedBytes == 48, "Two six-element FP32 buffers must retain 48 bytes.");

    var payload2 = binding.ExportHostStagingState(source);
    var afterSecondExport = binding.HostStagingPoolStatistics;
    Require(afterSecondExport.AllocatedBuffers == 2, "Second same-shape export must avoid new pinned allocations.");
    Require(afterSecondExport.ReusedBuffers == 2, "Second same-shape export must reuse both returned staging buffers.");
    Require(afterSecondExport.RetainedBuffers == 0, "Reused buffers must leave the retained pool while active.");

    var imported2 = binding.ImportHostStagingState(payload2);
    payload2.Dispose();
    try
    {
        Require(
            imported2.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(sourceKeyData),
            "Reused pinned key buffer must be fully overwritten with the current export.");
        Require(
            imported2.GetLayer(0).Value.GetTensorDataAsSpan<float>().SequenceEqual(sourceValueData),
            "Reused pinned value buffer must be fully overwritten with the current export.");
    }
    finally
    {
        imported2.Dispose();
    }

    var finalStats = binding.HostStagingPoolStatistics;
    Require(finalStats.ReturnedBuffers == 4, "Two export/import lifetimes must return four buffers total.");
    Require(finalStats.RetainedBuffers == 2, "Pool retention must remain bounded after reuse.");
    Require(finalStats.DroppedBuffers == 0, "Configured retention capacity should hold both buffers.");
}

static void RunPinnedPoolRetentionLimit()
{
    var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
        numHiddenLayers: 1,
        numKvHeads: 1,
        headDim: 2,
        vocabularySize: 16,
        kvElementType: TensorElementType.Float);
    using var binding = new OptimumLegacyFloatHostStagingBinding(
        new OptimumLegacyFloatDecoderBinding(profile),
        new PinnedHostStagingPoolOptions
        {
            MaxRetainedBuffersPerLength = 1,
            MaxRetainedBytes = 24
        });

    var shape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 3);
    using var source = new DecoderOrtState(
        position: 3,
        new[]
        {
            new DecoderOrtLayerState(
                OrtValue.CreateTensorValueFromMemory(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, shape),
                OrtValue.CreateTensorValueFromMemory(new[] { 7f, 8f, 9f, 10f, 11f, 12f }, shape))
        });

    var payload = binding.ExportHostStagingState(source);
    payload.Dispose();

    var stats = binding.HostStagingPoolStatistics;
    Require(stats.ReturnedBuffers == 2, "Disposing an unimported payload must return both rented buffers.");
    Require(stats.RetainedBuffers == 1, "Per-length and byte limits must retain only one six-element FP32 buffer.");
    Require(stats.RetainedBytes == 24, "Retention byte accounting must match the single kept buffer.");
    Require(stats.DroppedBuffers == 1, "The second returned buffer must be dropped when retention is full.");
}

static void RunAllocatorBoundary()
{
    var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
        numHiddenLayers: 1,
        numKvHeads: 1,
        headDim: 2,
        vocabularySize: 16,
        kvElementType: TensorElementType.Float);
    var allocator = new TrackingHostStagingAllocator();
    using var binding = new OptimumLegacyFloatHostStagingBinding(
        new OptimumLegacyFloatDecoderBinding(profile),
        new PinnedHostStagingPoolOptions
        {
            MaxRetainedBuffersPerLength = 0,
            MaxRetainedBytes = 0,
            ClearOnReturn = false
        },
        allocator);

    var keyData = new[] { 2f, 4f, 6f, 8f };
    var valueData = new[] { 1f, 3f, 5f, 7f };
    var shape = profile.Geometry.GetPastKvShape(batchSize: 1, pastSequenceLength: 2);
    using var source = new DecoderOrtState(
        position: 2,
        new[]
        {
            new DecoderOrtLayerState(
                OrtValue.CreateTensorValueFromMemory(keyData, shape),
                OrtValue.CreateTensorValueFromMemory(valueData, shape))
        },
        nextTokenId: 9);

    var payload = binding.ExportHostStagingState(source);
    Require(allocator.AllocationCalls == 2, "Custom allocator must own fresh key/value host allocations.");
    Require(allocator.RequestedLengths.SequenceEqual(new[] { 4, 4 }), "Allocator must receive exact tensor element lengths.");

    var imported = binding.ImportHostStagingState(payload);
    payload.Dispose();
    Require(allocator.DisposeCalls == 0, "Transfer-owner release must not dispose custom buffers while imported OrtValues retain them.");
    Require(
        imported.GetLayer(0).Key.GetTensorDataAsSpan<float>().SequenceEqual(keyData),
        "Memory-backed custom allocator must preserve staged key data through OrtValue import.");
    Require(
        imported.GetLayer(0).Value.GetTensorDataAsSpan<float>().SequenceEqual(valueData),
        "Memory-backed custom allocator must preserve staged value data through OrtValue import.");

    imported.Dispose();
    Require(allocator.DisposeCalls == 2, "Final imported-state release must return and dispose both custom buffers when retention is disabled.");

    var stats = binding.HostStagingPoolStatistics;
    Require(stats.AllocatedBuffers == 2, "Pool statistics must count custom physical allocations.");
    Require(stats.ReturnedBuffers == 2 && stats.DroppedBuffers == 2, "Disabled retention must return then physically drop custom buffers.");
}

sealed class CountingPayload : DecoderOrtHostStagingPayload
{
    public CountingPayload()
        : base("counting-v1", position: 1, nextTokenId: 3, byteLength: 16)
    {
        Key = new[] { 1f, 2f };
        Value = new[] { 3f, 4f };
    }

    public float[] Key { get; }
    public float[] Value { get; }
    public int DisposeCoreCalls { get; private set; }

    protected override void DisposeCore()
    {
        DisposeCoreCalls++;
        Array.Clear(Key);
        Array.Clear(Value);
    }
}

sealed class TrackingHostStagingAllocator : IHostStagingFloatBufferAllocator
{
    private readonly List<int> _requestedLengths = new();

    public int AllocationCalls { get; private set; }
    public int DisposeCalls { get; private set; }
    public IReadOnlyList<int> RequestedLengths => _requestedLengths;

    public IHostStagingFloatBuffer Allocate(int length)
    {
        AllocationCalls++;
        _requestedLengths.Add(length);
        return new TrackingBuffer(length, this);
    }

    private sealed class TrackingBuffer : IHostStagingFloatBuffer
    {
        private readonly TrackingHostStagingAllocator _owner;
        private int _disposed;

        public TrackingBuffer(int length, TrackingHostStagingAllocator owner)
        {
            _owner = owner;
            Memory = new float[length];
        }

        public Memory<float> Memory { get; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                _owner.DisposeCalls++;
            }
        }
    }
}
