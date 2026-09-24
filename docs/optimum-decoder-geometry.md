# Optimum legacy decoder geometry

`OptimumLegacyDecoderProfile` is Fission's first concrete causal-LM export profile.

It targets the separate Hugging Face Optimum **decoder-with-past** graph used by Llama-like decoder exports, including Qwen2-style grouped-query attention geometry. It intentionally does not target Optimum's merged decoder graph with a `use_cache_branch` selector.

## Why a concrete profile exists

The generic `DecoderOnlyOnnxExecutionAdapter` owns state transactions, not model geometry. ONNX export configuration remains architecture-specific, so Fission does not infer KV shapes from model names or guess tensor layouts at runtime.

A profile supplies two related pieces:

- `DecoderOnlyOnnxContract`: graph tensor names, rank, and element-type requirements;
- `DecoderOrtGeometry`: the physical dimensions and byte accounting needed to allocate decoder tensors and budget KV memory.

## Tensor naming

The default Llama-like profile declares:

```text
inputs
  input_ids
  attention_mask
  position_ids
  past_key_values.0.key
  past_key_values.0.value
  ...

outputs
  logits
  present.0.key
  present.0.value
  ...
```

Every name remains overrideable because exporter-specific graphs can rename tensors.

## Tensor geometry

The initial physical KV layout is:

```text
[batch, kv_heads, sequence, head_dim]
```

The shape planner exposes:

```text
input_ids      [B, S]
position_ids   [B, S]
attention_mask [B, Past + S]
past KV        [B, KVHeads, Past, HeadDim]
present KV     [B, KVHeads, Past + S, HeadDim]
logits          [B, S, Vocabulary]
```

`NumKvHeads` is intentionally independent from the model's full attention-head count so grouped-query attention does not over-allocate KV state.

## Element types

KV storage currently accepts:

- `Float`;
- `Float16`;
- `BFloat16`.

The KV element type and logits element type are independent. The profile also requires `input_ids`, `attention_mask`, and `position_ids` to be Int64 for this Optimum export family.

`DecoderOnlyOnnxContract` now carries optional rank/type requirements for KV and logits so graph validation can fail before any decoder state is allocated.

## KV memory accounting

For one sequence:

```text
KV elements =
  layers * 2(key+value) * kv_heads * sequence_length * head_dim

KV bytes = KV elements * bytes_per_element
```

This is deliberately exposed on `DecoderOrtGeometry` so the scheduler can reason about memory without depending on ONNX Runtime objects.

Example used by the executable spec:

```text
layers      = 2
kv_heads    = 8
head_dim    = 128
seq         = 10
dtype       = fp16 (2 bytes)

KV elements = 40,960
KV bytes    = 81,920
```

## Current boundary

This profile defines names, shapes, dtypes, and memory cost. It does not yet allocate or pack a production Llama/Qwen2 step.

The next concrete binding should use this profile to build:

- Int64 `input_ids`, `attention_mask`, and `position_ids`;
- past-KV inputs from `DecoderOrtState`;
- caller-owned/preallocated logits and present-KV outputs;
- greedy sampling first, then pluggable sampling;
- a new immutable `DecoderOrtState` after every successful step.

The first binding should remain tied to this known Optimum legacy-cache geometry. Merged decoders, alternate cache layouts, and provider-specific packed/paged KV should be separate profiles or bindings.
