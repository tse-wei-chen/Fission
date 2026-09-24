# ONNX Runtime backend

`Fission.Backends.OnnxRuntime` hosts a real ONNX Runtime `InferenceSession` behind the existing `IInferenceBackend` contract.

The generic backend deliberately does not encode a Qwen, Llama, GPT, or Optimum export schema. ONNX exports differ in tensor names, KV layouts, cache ownership, logits outputs, and batching conventions. Those details belong to `IOnnxRuntimeExecutionAdapter`.

## Ownership

```text
OnnxRuntimeBackend
  owns InferenceSession
  owns model binding (one Fission ModelId)
  validates model identity
  delegates model execution

IOnnxRuntimeExecutionAdapter
  owns model-specific tensor schema
  owns batching/tensorization policy
  owns sampling policy for the current backend contract
  owns per-sequence ORT state / KV OrtValues
  releases sequence state through ReleaseSequenceAsync
```

The adapter is disposed before the `InferenceSession`. An adapter may hold `OrtValue` instances or other native resources that must be relinquished while the session is still alive.

## Model sources

`OnnxRuntimeModelSource` supports model files and in-memory ONNX bytes. The byte source is useful for tests, embedded models, or a higher-level model registry that already owns artifact loading.

## Session options

The backend creates CPU `SessionOptions` by default and exposes a `SessionOptions` factory. A future CUDA-specific package can supply provider options without making the generic backend depend on GPU native libraries.

Thread counts and graph optimization level are configured through `OnnxRuntimeBackendOptions`.

## OrtValue lifecycle

New model adapters should use the `OrtValue` API rather than the deprecated `NamedOnnxValue` path. Every native output/input wrapper that owns or pins resources must be deterministically disposed. Fixed-shape hot paths should move toward preallocated/reused `OrtValue` buffers and I/O binding after correctness is established.

## Current specification

`Fission.OnnxRuntime.Specs` embeds ONNX Runtime's small `mul_1.onnx` test model as bytes. The spec creates a real `InferenceSession`, executes it through an adapter using `OrtValue`, verifies the numeric output for prefill and decode, and then verifies the device-actor sequence release hook clears adapter-owned state.

This proves the session/adapter/runtime integration. It is not an LLM adapter; causal-LM tensor/KV semantics are the next layer.
