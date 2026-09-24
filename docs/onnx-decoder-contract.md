# ONNX decoder contract

Fission does not assume that every decoder-only ONNX export uses the same tensor names.

The ONNX Runtime backend owns the `InferenceSession`. A model adapter owns execution behavior and physical model state. Between those layers, a declarative contract validates that the live ONNX graph has the signature the adapter expects before the adapter allocates sequence state.

## Generic session contract

`OnnxSessionContract` contains required input and output tensors. Each `OnnxTensorContract` may constrain:

- a stable logical name used by Fission/model-adapter code;
- the actual tensor name in the ONNX export;
- tensor rank;
- tensor element type.

The validator allows extra inputs and outputs in the graph. It only requires every declared contract tensor to exist and satisfy its declared metadata.

Validation runs in `OnnxRuntimeBackend.InitializeAsync` after the `InferenceSession` is created and before `IOnnxRuntimeExecutionAdapter.InitializeAsync`. A signature mismatch therefore fails before the adapter creates model-owned sequence state.

## Decoder-only manifest

`DecoderOnlyOnnxContract` maps common causal-LM concepts onto export-specific names:

- `input_ids`;
- optional attention mask;
- optional position IDs;
- logits;
- per-layer past key/value inputs;
- per-layer present key/value outputs.

These are logical roles, not fixed ONNX names. For example, one export may use:

```text
past_key_values.0.key
past_key_values.0.value
```

while another may use:

```text
present.0.key
present.0.value
```

The manifest expands layer-indexed cache name patterns with either `%d` or `{0}` placeholders. Past/present key/value patterns are all-or-none: if any cache pattern is configured, all four patterns are required.

## Export-specific tensors

Some exports require additional state such as cache position, current sequence length, past sequence length, beam indices, or model-specific masks. `AdditionalInputs` and `AdditionalOutputs` allow those tensors to be declared without changing the generic backend or the stable decoder roles.

A concrete model/export adapter should own the semantics of those extra tensors.

## What this contract does not do

The contract is validation, not execution. It does not:

- allocate or retain KV `OrtValue` objects;
- decide cache tensor layout;
- concatenate or page past/present KV;
- choose logits sampling behavior;
- define EOS/stop-token policy;
- implement tensor batching across sequences;
- provide model tokenizer or chat-template behavior.

Those responsibilities remain in the execution adapter and its per-sequence state implementation.

## Next layer

A generic decoder adapter can now be built against stable logical roles rather than hard-coded ONNX tensor names. Its state layer should own past/present KV tensors and implement Fission's existing backend transaction protocol for snapshot, fork, restore, and release.

This separation lets Qwen-, Llama-, GPT-, or exporter-specific manifests reuse the same state/execution machinery when their tensor geometry is compatible, while still failing fast when an ONNX graph does not match the declared contract.
