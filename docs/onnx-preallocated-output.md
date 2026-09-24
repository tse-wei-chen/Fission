# Preallocated ONNX outputs

Persistent decoder KV needs output ownership that remains with the model adapter after an ONNX Runtime call returns.

`OwnedOrtTensor<T>` packages one caller-owned managed buffer with the `OrtValue` pinned over that buffer. It is the CPU correctness primitive for proving that ownership model before introducing pooled native/GPU allocations.

## Ownership

The caller creates and owns `OwnedOrtTensor<T>`. ONNX Runtime receives only its `OrtValue` handle through the preallocated-output `InferenceSession.Run` overload.

```text
adapter
  owns buffer
  owns OrtValue
      |
      v
InferenceSession.Run(..., outputValues)
      |
      +-- writes result into caller-owned allocation
```

`Run` does not return an output collection in this path and does not take ownership of the supplied output values. The same allocation may therefore be reused for later model steps as long as its shape and element type remain compatible.

Disposing `OwnedOrtTensor<T>` disposes its `OrtValue`, which releases the pin on the managed buffer.

## Shape validation

`OwnedOrtTensor<T>` currently requires a fully known positive shape because preallocated output memory must have a fixed element count before execution.

The constructor verifies that the product of all dimensions matches the supplied buffer length. Dynamic or zero dimensions are rejected at this layer.

For dynamic decoder exports, the adapter must resolve the concrete shape for the current step before allocating a tensor or use an IO-binding/provider path that supports dynamic allocation.

## Why this differs from normal Run outputs

The standard `Run` overload that allocates outputs returns a disposable collection that owns those output `OrtValue`s. Disposing the collection disposes its children, so individual values cannot be retained as persistent decoder state after the result collection is disposed.

The preallocated-output overload accepts caller-created `OrtValue`s instead. This gives Fission an unambiguous lifetime boundary suitable for `DecoderOrtState`.

## Decoder integration path

For a cached decoder step, the eventual adapter should:

1. retain the sequence's current `DecoderOrtState` as past-KV inputs;
2. resolve concrete present-KV and logits shapes;
3. allocate or rent caller-owned output tensors;
4. invoke `InferenceSession.Run` with preallocated output values;
5. sample/read logits;
6. transfer the successful present-KV output tensors into a new `DecoderOrtState`;
7. replace only the sequence's current state in `DecoderStateStore`;
8. return unused/failed-step allocations to the allocator.

Snapshot/fork references to the old immutable state remain valid until their final owners are released.

## Current allocator scope

`OwnedOrtTensor<T>` uses managed arrays and pinning. It is intentionally a correctness primitive, not the final high-performance allocator.

Later implementations can preserve the same caller-ownership contract while replacing the backing storage with:

- pooled pinned host memory;
- ORT allocator-owned native buffers;
- CUDA/device memory;
- `OrtIoBinding`;
- a fixed-shape tensor arena;
- paged physical KV buffers.

The key invariant is unchanged: the adapter, not a temporary ORT result collection, owns persistent output storage.
