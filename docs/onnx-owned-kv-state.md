# Owned ONNX KV state

`DecoderOrtState` is the concrete physical payload for a decoder state version backed by ONNX Runtime `OrtValue` tensors.

It is designed to live inside `DecoderStateStore<DecoderOrtState>`, so Fission snapshots and branches can share one immutable KV version while retaining deterministic native-resource ownership.

## Layer ownership

A state contains one key/value pair per decoder layer:

```text
DecoderOrtState
  Position
  Layer[0]
    Key   -> owned OrtValue
    Value -> owned OrtValue
  Layer[1]
    Key   -> owned OrtValue
    Value -> owned OrtValue
  ...
```

Construction validates the complete payload before ownership transfers to the state:

- at least one layer is required;
- every key/value must be a tensor `OrtValue`;
- the same `OrtValue` instance cannot occupy more than one owned slot.

If validation fails, the caller still owns every supplied `OrtValue` and remains responsible for disposing or reusing them.

After successful construction, `DecoderOrtState` owns all supplied values. Disposal is idempotent and releases value/key pairs in reverse layer order.

## Sharing through DecoderStateStore

`DecoderOrtState` itself does not implement reference counting. `DecoderStateStore` owns that graph.

A sequence, snapshot, or branch may reference the same immutable `DecoderOrtState` instance. Releasing one owner does not dispose the underlying `OrtValue`s while another owner remains. Only the final reference release calls `DecoderOrtState.Dispose()`.

A divergent decode should create a new `DecoderOrtState` from the model's new present-KV tensors and replace only that sequence's current state version.

## ONNX Runtime output ownership

Do not treat values returned by a normal `InferenceSession.Run(...)` output collection as independently transferable children.

ONNX Runtime returns a disposable output collection that owns its contained output values; disposing that collection disposes each contained `OrtValue`. Keeping one child after disposing the parent result collection would therefore retain an invalid/disposed value, while not disposing the collection would leak ownership.

Persistent decoder KV should instead be produced into adapter-owned output values, for example through:

- caller-preallocated output `OrtValue`s passed to a `Run` overload that accepts output values;
- `OrtIoBinding` with adapter-owned bound buffers;
- a provider-specific arena that creates and owns the output tensors.

Those caller-owned values can then be transferred into a new `DecoderOrtState` after a successful model step.

## Proven caller-owned Run path

Fission now exposes `CallerOwnedOrtRun.Execute(...)` as the explicit execution path for preallocated outputs.

The primitive validates the input/output collection counts and requires unique tensor `OrtValue` instances for each output slot, then calls the ONNX Runtime overload that writes directly into caller-owned output values.

The executable state spec proves the complete ownership path with a real ONNX graph:

1. create caller-owned output buffers and `OrtValue`s before the run;
2. execute `mul_1.onnx` into those outputs;
3. dispose the input values, `RunOptions`, and `InferenceSession`;
4. confirm the output `OrtValue`s are still valid and contain the expected inference results;
5. transfer the output values into `DecoderOrtState`;
6. fork the state through `DecoderStateStore`;
7. release the parent while the branch keeps the outputs alive;
8. release the final branch and let `DecoderOrtState` dispose the output values.

This is the supported ownership bridge for a future decoder adapter. It avoids extracting children from an ORT-owned result collection and remains compatible with later output pooling and GPU IO binding.

## Current scope

The current payload owns key/value tensors and records logical decoder position. It intentionally does not yet define:

- KV tensor rank/layout;
- number of KV heads or head dimension;
- element precision;
- CPU versus CUDA residency;
- paged versus contiguous physical storage;
- auxiliary model state such as cache position or sequence length;
- logits ownership.

Those constraints belong to the concrete decoder adapter/export geometry and output allocator.

## Next layer

The next layer is a real decoder-only execution adapter that combines the pieces already established:

- `DecoderOnlyOnnxContract` for export-specific tensor names;
- `CallerOwnedOrtRun` for persistent present-KV outputs;
- `DecoderOrtState` for physical KV ownership;
- `DecoderStateStore` for snapshot/fork/restore lifetime;
- a logits reader/sampler for producing the next token.

The first implementation should target one known export geometry rather than pretending every Qwen/Llama ONNX export has identical KV shapes. Model-specific manifests should supply the tensor names and geometry while the generic adapter owns the execution/state protocol.
