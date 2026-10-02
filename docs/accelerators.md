# Accelerator and execution-provider model

Fission separates **device identity**, **physical accelerator kind**, **execution provider**, and **capabilities**.

That separation is intentional. A scheduler or runtime component must not infer hardware behavior by parsing strings such as `cuda:0`, `mps:0`, or `qnn-htp:0`.

## Layers

```text
DeviceId
  opaque runtime identity
        |
        v
InferenceDeviceDescriptor
  kind + provider + capabilities + memory topology
        |
        v
IInferenceDeviceDiscovery
  vendor/provider-specific startup probe
        |
        v
InferenceDeviceCatalog
  deterministic registry used by composition/configuration code
```

The existing `IInferenceBackend` remains the execution boundary. The accelerator catalog does not replace it and does not change scheduler behavior.

## Physical kinds

The initial stable physical categories are:

- `Cpu`
- `Gpu`
- `Npu`
- `Tpu`
- `Fpga`
- `Custom`

These categories describe what the device is. They do not imply any specific feature.

For example, an NPU is not assumed to support dynamic shapes, migration, resident KV, or device-memory accounting. A discovery provider must advertise those capabilities explicitly.

## Execution providers

`ExecutionProviderId` is deliberately open rather than an enum. Fission ships known ids for common paths, while a vendor plugin may register a new stable id without modifying Fission core.

Current known ids:

| Provider id | Intended integration lane |
| --- | --- |
| `cpu` | host execution |
| `cuda` | NVIDIA CUDA |
| `tensorrt` | NVIDIA TensorRT |
| `directml` | DirectML |
| `openvino` | Intel/OpenVINO CPU, GPU, or NPU |
| `qnn` | Qualcomm QNN/HTP-class accelerators |
| `vitis-ai` | AMD/Xilinx Vitis AI devices |
| `coreml` | Apple Core ML execution |
| `mps` | Apple Metal Performance Shaders / MPS Graph integration |
| `pjrt` | PJRT-class runtimes such as TPU integrations |

This table is an identity vocabulary, not a claim that Fission already has a production backend for every provider.

## MPS and Core ML are separate

Fission treats `mps` and `coreml` as different provider identities.

Metal Performance Shaders is a Metal-oriented compute stack. A future MPS integration may expose Apple GPU execution directly or through an MPS Graph bridge.

Core ML is a separate higher-level Apple execution path and may select different compute units. Keeping the provider ids separate prevents the runtime from conflating a direct Metal/MPS path with a Core ML path.

The physical `AcceleratorKind` is supplied by the discovery provider. It is not hard-coded from the provider name.

## Capabilities

`InferenceDeviceCapabilities` currently describes optional properties that matter to an inference runtime:

- batched prefill/decode,
- resident KV,
- snapshot/fork,
- migration,
- device-memory accounting/reclaim,
- asynchronous transfer,
- dynamic shapes,
- quantized execution,
- unified memory.

Capabilities are descriptive metadata. No scheduler policy consumes them yet.

This is important for heterogeneous machines: two devices using the same provider can expose different capabilities, memory topology, or model support.

## Discovery

Provider-specific packages should implement `IInferenceDeviceDiscovery`.

A discovery provider owns vendor APIs and translates their result into `InferenceDeviceDescriptor` instances. The generic catalog owns no CUDA, Metal, OpenVINO, QNN, Core ML, or PJRT native dependency.

Example topology:

```text
CudaDiscovery -----------+
OpenVinoDiscovery -------+
QnnDiscovery ------------+--> InferenceDeviceCatalog
MpsDiscovery ------------+
CoreMlDiscovery ---------+
PjrtDiscovery -----------+
CustomVendorDiscovery ---+
```

Duplicate `DeviceId` values are rejected. This prevents two provider integrations from silently claiming the same runtime identity.

## Memory topology

Memory topology is independent of accelerator kind:

- `Host`
- `Dedicated`
- `Unified`
- `Unknown`

A future device probe can therefore distinguish, for example, a discrete GPU from a unified-memory accelerator without changing the scheduler contract.

## Integration lanes

The intended first implementations are provider adapters outside scheduler/runtime policy:

```text
IInferenceBackend
    |
    +-- OnnxRuntimeBackend
    |      +-- CUDA / TensorRT
    |      +-- OpenVINO
    |      +-- QNN
    |      +-- Vitis AI
    |      +-- Core ML
    |
    +-- native MPS / Metal backend
    +-- PJRT backend
    +-- custom accelerator backend
```

Not every ONNX Runtime execution provider is necessarily available from the same NuGet package or on every operating system. Provider-specific packages and native dependencies belong in optional integration projects rather than `Fission.Abstractions` or `Fission.Runtime`.

## Current scope

This foundation provides metadata, discovery, and catalog semantics only.

It does **not** yet:

- enumerate real CUDA/MPS/NPU/TPU devices,
- create provider-specific ONNX Runtime `SessionOptions`,
- allocate MPS/Metal buffers,
- execute an LLM on an NPU,
- route scheduler work based on capabilities,
- migrate KV state between unlike accelerator families.

Those are subsequent integration steps. The goal of this layer is to let them arrive independently without introducing vendor checks into the scheduler hot path.
