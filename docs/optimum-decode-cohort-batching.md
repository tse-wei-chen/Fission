# Optimum decode cohort batching

`OptimumLegacyFloatDecoderBinding` implements `IDecoderOrtBatchModelBinding` so a runtime `DecodeBatch` can become real model batching instead of a loop of scalar `InferenceSession.Run` calls.

## Cohort rule

The legacy dense KV layout is `[batch, kv_heads, sequence, head_dim]`. A single dense tensor therefore requires every row in a model invocation to have the same past sequence length. The binding groups decode items by `DecodeItem.Position`; each same-position cohort becomes one ONNX Runtime invocation.

Input and result order remain the request order exposed by the generic decoder adapter. Cohorting is an execution detail inside the model binding.

## Physical execution

For one cohort the binding:

1. validates every immutable prior `DecoderOrtState`,
2. packs next-token frontiers into `[B, 1]` `input_ids`,
3. creates `[B, 1]` `position_ids` and `[B, past + 1]` attention masks,
4. packs each sequence's past key/value tensors into `[B, H, past, D]`,
5. performs one caller-owned `InferenceSession.Run`,
6. samples each row's logits independently, and
7. splits batched present-KV back into independently owned immutable sequence states.

`OrtRunCount` is a diagnostic counter used by executable specs to prove that one cohort maps to one physical ORT run.

## Ownership invariant

Prior states are immutable. Batched output buffers are temporary owners only. Before returning, every sequence receives its own `OrtValue` key/value handles; disposing one returned state must not invalidate any sibling state.

The current implementation is correctness-first and copies each row from the batched present-KV buffers into per-sequence managed buffers. This avoids shared-handle lifetime ambiguity and preserves the transactional snapshot/fork/restore model.

A future performance pass may replace the split copy with a ref-counted batch arena and/or ONNX Runtime I/O binding. That optimization must preserve the same independent logical ownership semantics.

## Current scope

This batching path targets the legacy dense Optimum decoder-with-past FP32 binding. It does not imply that paged KV, recurrent/linear-attention state, mixed cache layouts, or arbitrary exporter-specific tensors can share this packing strategy. Those remain separate binding implementations behind the same generic batch protocol.
