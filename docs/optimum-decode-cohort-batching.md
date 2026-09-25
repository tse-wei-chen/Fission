# Optimum decode cohort batching

`OptimumLegacyFloatDecoderBinding` implements `IDecoderOrtBatchModelBinding` so a runtime `DecodeBatch` becomes physical model batching rather than a loop of scalar `InferenceSession.Run` calls.

## Cohort rule

The legacy dense KV layout is `[batch, kv_heads, sequence, head_dim]`. A single dense tensor requires every row in one invocation to have the same past sequence length, so the binding groups decode items by `DecodeItem.Position`.

## Stable cohort arena

Every successful batched decode produces one managed key buffer and one value buffer per layer for the complete `[B, H, present, D]` frontier. Returned `DecoderOrtState` objects own independently disposable `OrtValue` slices over non-overlapping rows of those buffers and also retain internal `(arena, row)` placement metadata.

On the next decode step the binding can reuse the full dense past KV without a gather copy when all of these invariants hold:

- every requested prior state comes from the same arena,
- the arena position matches the requested decode position,
- the cohort contains exactly `arena.BatchSize` states,
- every physical row appears exactly once, and
- the arena geometry still matches the binding.

Request order does not need to match physical row order. The binding executes ORT in stable arena-row order and maps results back to request order.

If fork, restore, partial scheduling, or any other operation creates duplicate/missing rows, the optimization is deliberately abandoned for that step and the existing gather/pack path is used. This keeps transactional state semantics independent from the batching optimization.

## Physical execution

For a reusable cohort frontier:

1. next-token frontiers are placed into stable physical row order,
2. `[B, 1]` `input_ids`, position ids, and the attention mask are prepared,
3. prior arena key/value arrays are wrapped directly as batched ORT input tensors,
4. one caller-owned `InferenceSession.Run` executes the cohort,
5. logits are sampled per physical row,
6. a new output arena becomes the next frontier, and
7. row results are mapped back to caller request order.

The first batch formed from independent states still performs one gather/pack in order to establish an arena. Subsequent complete stable cohorts avoid that past-KV copy.

`PastKvPackCount`, `PastKvArenaReuseCount`, and `PastKvCopiedElementCount` are diagnostic counters used by executable specs to keep this physical invariant testable.

## Ownership

Prior states remain immutable. Row states have independent `OrtValue` handles even though their managed backing arrays are shared. Disposing one row does not invalidate sibling rows. Disposing a state also drops its arena reference so dead frontiers do not keep full cohort arrays alive unnecessarily.

A newly produced frontier uses new output buffers, so the complete prior arena may be released immediately after the model step once the runtime commits the new states.

## Current scope

This optimization targets the legacy dense Optimum decoder-with-past FP32 binding. It is not paged KV and it does not make partial cohorts zero-copy. Paged caches, recurrent/linear-attention state, mixed cache layouts, and exporter-specific tensors remain separate binding concerns.
