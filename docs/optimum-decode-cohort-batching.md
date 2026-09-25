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
7. exposes each present-KV row as independently owned `OrtValue` handles over non-overlapping `Memory<float>` slices of the batched output buffers.

`OrtRunCount` is a diagnostic counter used by executable specs to prove that one cohort maps to one physical ORT run.

## Zero-copy output ownership

Prior states remain immutable. ONNX Runtime writes present-KV into one managed batched array per layer/key/value. After the run, Fission does **not** copy each row into a new managed array. Instead, every sequence receives its own tensor handles created over the corresponding `Memory<float>` slice of the shared backing array.

Each slice `OrtValue` pins its own memory region for its lifetime. The temporary full-batch output handles can therefore be disposed after state construction while the row handles remain valid. Disposing one returned state releases only that state's tensor handles; sibling rows remain readable and independently disposable.

The executable Optimum batch spec validates both properties directly: mutations of a managed backing array are immediately visible through a Memory-backed `OrtValue`, and disposing one slice handle does not invalidate another slice.

## Remaining copy boundary

Present-KV splitting is now zero-copy, but **past-KV packing still copies**. Before each dense cohort run, independently owned prior sequence states are gathered into contiguous `[B, H, past, D]` input buffers.

Eliminating that gather requires a longer-lived cohort/batch arena, stable row placement, paged/indirect KV addressing, or a backend-specific device-memory strategy. That is deliberately separate from logical snapshot/fork/restore ownership semantics.

## Current scope

This batching path targets the legacy dense Optimum decoder-with-past FP32 binding. It does not imply that paged KV, recurrent/linear-attention state, mixed cache layouts, or arbitrary exporter-specific tensors can share this packing strategy. Those remain separate binding implementations behind the same generic batch protocol.
