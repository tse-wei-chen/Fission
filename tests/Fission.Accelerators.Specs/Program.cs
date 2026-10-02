using Fission.Abstractions;
using Fission.Accelerators;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

var mps = new InferenceDeviceDescriptor(
    new DeviceId("mps:0"),
    AcceleratorKind.Gpu,
    KnownExecutionProviders.Mps,
    InferenceDeviceCapabilities.BatchedPrefill |
    InferenceDeviceCapabilities.BatchedDecode |
    InferenceDeviceCapabilities.ResidentKv |
    InferenceDeviceCapabilities.UnifiedMemory,
    DeviceMemoryTopology.Unified,
    ordinal: 0,
    vendor: "Apple");

Require(mps.Supports(InferenceDeviceCapabilities.BatchedDecode),
    "MPS descriptor must preserve declared decode capability.");
Require(mps.MemoryTopology == DeviceMemoryTopology.Unified,
    "MPS descriptor must preserve discovered memory topology.");
Require(KnownExecutionProviders.Mps != KnownExecutionProviders.CoreMl,
    "MPS and CoreML must remain separate execution-provider identities.");

var intelNpu = new InferenceDeviceDescriptor(
    new DeviceId("openvino-npu:0"),
    AcceleratorKind.Npu,
    KnownExecutionProviders.OpenVino,
    InferenceDeviceCapabilities.BatchedPrefill |
    InferenceDeviceCapabilities.BatchedDecode |
    InferenceDeviceCapabilities.DynamicShapes,
    ordinal: 0,
    vendor: "Intel");

var customProvider = new ExecutionProviderId("vendor-asic");
var custom = new InferenceDeviceDescriptor(
    new DeviceId("vendor-asic:7"),
    AcceleratorKind.Custom,
    customProvider,
    InferenceDeviceCapabilities.QuantizedExecution,
    ordinal: 7);

var catalog = new InferenceDeviceCatalog();
catalog.Register(mps);
catalog.Register(intelNpu);
catalog.Register(custom);

Require(catalog.Count == 3, "Catalog must contain every registered accelerator.");
Require(catalog.TryGet(mps.Device, out var resolvedMps) && ReferenceEquals(resolvedMps, mps),
    "Catalog lookup must preserve the registered descriptor.");
Require(catalog.FindByKind(AcceleratorKind.Npu).Single() == intelNpu,
    "Catalog kind lookup must find the NPU descriptor.");
Require(catalog.FindByProvider(KnownExecutionProviders.Mps).Single() == mps,
    "Catalog provider lookup must find the MPS descriptor.");
Require(catalog.FindByProvider(customProvider).Single() == custom,
    "Catalog must support provider ids that were not compiled into Fission.");

var ordered = catalog.Snapshot();
Require(
    ordered.Select(static descriptor => descriptor.Device.Value)
        .SequenceEqual(new[] { "mps:0", "openvino-npu:0", "vendor-asic:7" }),
    "Catalog snapshots must be deterministic by DeviceId.");

var duplicateRejected = false;
try
{
    catalog.Register(new InferenceDeviceDescriptor(
        new DeviceId("mps:0"),
        AcceleratorKind.Gpu,
        KnownExecutionProviders.CoreMl));
}
catch (InvalidOperationException)
{
    duplicateRejected = true;
}
Require(duplicateRejected, "Duplicate DeviceIds must be rejected across providers.");

var invalidOrdinalRejected = false;
try
{
    _ = new InferenceDeviceDescriptor(
        new DeviceId("cuda:-1"),
        AcceleratorKind.Gpu,
        KnownExecutionProviders.Cuda,
        ordinal: -1);
}
catch (ArgumentOutOfRangeException)
{
    invalidOrdinalRejected = true;
}
Require(invalidOrdinalRejected, "Negative ordinals must be rejected.");

var discovered = await InferenceDeviceCatalog.DiscoverAsync(new IInferenceDeviceDiscovery[]
{
    new StaticDiscovery(
        "apple",
        new InferenceDeviceDescriptor(
            new DeviceId("coreml:0"),
            AcceleratorKind.Npu,
            KnownExecutionProviders.CoreMl,
            InferenceDeviceCapabilities.QuantizedExecution)),
    new StaticDiscovery(
        "qualcomm",
        new InferenceDeviceDescriptor(
            new DeviceId("qnn-htp:0"),
            AcceleratorKind.Npu,
            KnownExecutionProviders.Qnn,
            InferenceDeviceCapabilities.QuantizedExecution |
            InferenceDeviceCapabilities.DeviceMemoryAccounting)),
    new StaticDiscovery(
        "pjrt",
        new InferenceDeviceDescriptor(
            new DeviceId("pjrt-tpu:0"),
            AcceleratorKind.Tpu,
            KnownExecutionProviders.Pjrt,
            InferenceDeviceCapabilities.BatchedPrefill |
            InferenceDeviceCapabilities.BatchedDecode))
});

Require(discovered.Count == 3, "Discovery must merge descriptors from multiple probes.");
Require(discovered.FindByKind(AcceleratorKind.Tpu).Single().Provider == KnownExecutionProviders.Pjrt,
    "TPU discovery must remain provider-neutral above PJRT.");
Require(discovered.FindByProvider(KnownExecutionProviders.Qnn).Single().Kind == AcceleratorKind.Npu,
    "QNN discovery must preserve the physical accelerator kind supplied by its probe.");

Console.WriteLine(
    $"Fission accelerator catalog specs passed: registered={catalog.Count}, discovered={discovered.Count}.");

sealed class StaticDiscovery(
    string name,
    params InferenceDeviceDescriptor[] devices) : IInferenceDeviceDiscovery
{
    public string Name { get; } = name;

    public ValueTask<IReadOnlyList<InferenceDeviceDescriptor>> DiscoverAsync(
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult<IReadOnlyList<InferenceDeviceDescriptor>>(devices);
    }
}
