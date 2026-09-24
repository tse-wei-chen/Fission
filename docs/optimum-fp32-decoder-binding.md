# Optimum legacy FP32 decoder binding

`OptimumLegacyFloatDecoderBinding` is Fission's first concrete causal-LM ONNX execution binding.

It plugs into `DecoderOnlyOnnxExecutionAdapter` and targets the separate Optimum-style decoder-with-past schema described by `OptimumLegacyDecoderProfile`.

## Supported execution path

The binding packs a real decoder step as:

```text
input_ids          Int64 [1, S]
attention_mask     Int64 [1, Past + S]
position_ids       Int64 [1, S]
past_key_values.*  Float [1, KVHeads, Past, HeadDim]
        |
        v
      ONNX Runtime
        |
        v
logits              Float [1, S, Vocabulary]
present.*           Float [1, KVHeads, Past + S, HeadDim]
```

All outputs are caller-owned/preallocated `OrtValue`s. Present KV tensors transfer into a new immutable `DecoderOrtState`; logits are step-local and are released after sampling.

## Token frontier

A decoder state now represents more than KV memory:

```text
DecoderOrtState
  Position
  NextTokenId
  KV layers...
```

`NextTokenId` is the sampled token that must be supplied as the next one-token decode input.

This is required for transactional inference. Snapshot, fork, and restore must preserve the complete causal frontier, not merely the KV tensors. Restoring a state therefore rewinds both:

- the physical KV cache;
- the token that the next decode step will consume.

`DecoderOnlyOnnxExecutionAdapter` enforces that every successful binding result has:

```text
BackendStepResult.TokenId == DecoderOrtState.NextTokenId
```

A decode state without a token frontier is rejected before the model binding runs.

## Prefill

The current binding uses the same decoder-with-past session for prefill by supplying zero-length past KV tensors:

```text
past KV [1, KVHeads, 0, HeadDim]
```

The graph must accept an empty sequence on the past-KV axis.

This works for the executable Fission decoder fixture and can work for compatible exported models, but it is not assumed to be universal. Optimum exports that require a distinct no-past decoder graph need a future dual-session binding.

## Decode

For one-token decode:

```text
input_ids = [priorState.NextTokenId]
position_ids = [priorState.Position]
attention_mask length = priorState.Position + 1
past KV = priorState KV
```

The binding preallocates present KV with sequence length `Past + 1`, executes ONNX Runtime through `CallerOwnedOrtRun`, samples from the last logits position, and returns a new immutable state at `Position + 1`.

The prior state is borrowed and never mutated or disposed by the binding.

## Sampling

The first implementation intentionally uses greedy sampling only.

`GreedySampleLastPosition` treats logits as `[1, S, Vocabulary]` and performs argmax only over the final sequence position. This is correct for causal generation after a multi-token prefill and for one-token decode.

EOS token ids can be supplied to the binding. Sampling an EOS id sets `DecoderOrtStepResult.IsFinished`, which flows through `BackendStepResult.IsFinished`.

A pluggable sampler is follow-up work; it should not change the decoder state ownership protocol.

## Executable causal fixture

`Fission.OnnxRuntime.Optimum.Specs` embeds a purpose-built one-layer ONNX decoder graph rather than reusing an unrelated arithmetic model.

Its graph exposes the same shape of interface used by the concrete binding:

```text
input_ids
attention_mask
position_ids
past_key_values.0.key
past_key_values.0.value
        ->
logits
present.0.key
present.0.value
```

The fixture appends the current token to key/value state and emits a deterministic token transition:

```text
0 -> 1 -> 2 -> 3 -> 0
```

The executable spec proves:

1. prompt token `3` prefills to sampled token `0`;
2. first decode consumes state frontier `0` and samples `1`;
3. second decode consumes frontier `1` and samples `2`;
4. restoring the branch to the post-prefill snapshot rewinds the frontier to `0`;
5. decoding again reproduces `0 -> 1`;
6. continuing reaches token `3` and propagates EOS;
7. present KV grows with every successful step;
8. all state operations run through the same `ContinuousBatchExecutor`, backend, adapter, and ORT ownership path used by production code.

This is the first end-to-end test in Fission where a decoder-shaped ONNX graph drives token generation and transactional state.

## Current limitations

The binding deliberately starts narrow:

- batch size is currently one per binding invocation;
- KV and logits are CPU FP32;
- output buffers are new managed arrays per model step;
- sampling is greedy;
- prefill depends on zero-length past support;
- there is no CUDA IO binding or buffer pool yet;
- it targets the separate legacy decoder-with-past schema, not Optimum's merged `use_cache_branch` decoder;
- auxiliary model-specific inputs beyond the profile are rejected.

These are implementation boundaries, not reasons to weaken the generic transactional runtime. New export/provider variants should add specialized bindings or allocators while preserving the same immutable state protocol.

## Next practical steps

The next production-oriented work should focus on execution mechanics rather than adding another abstraction layer:

1. support a distinct no-past prefill session plus with-past decode session;
2. move output allocation behind reusable pools / IO binding;
3. add FP16/BF16 and CUDA-resident KV;
4. replace greedy-only policy with a sampler interface;
5. expose geometry-derived KV byte cost to scheduler admission and memory budgets;
6. validate against an actual small Optimum Llama/Qwen2 export in an optional integration test.
