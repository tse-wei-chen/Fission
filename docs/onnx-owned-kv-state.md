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

The next execution primitive should prove caller-owned ORT outputs end to end:

1. allocate an output buffer and `OrtValue` before `Run`;
2. pass it to the ONNX Runtime overload that accepts output values;
3. execute the graph;
4. retain ownership after the call returns;
5. transfer successful KV outputs into `DecoderOrtState`;
6. let `DecoderStateStore` control snapshot/fork/restore lifetime.

That path avoids ambiguous ownership transfer and is also compatible with later buffer pooling and GPU IO binding.
