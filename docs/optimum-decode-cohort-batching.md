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
3. prior arena key/value memory is wrapped directly as batched ORT input tensors,
4. one caller-owned `InferenceSession.Run` executes the cohort,
5. logits are sampled per physical row,
6. a new output arena becomes the next frontier, and
7. row results are mapped back to caller request order.

The first batch formed from independent states still performs one gather/pack in order to establish an arena. Subsequent complete stable cohorts avoid that past-KV copy.

`PastKvPackCount`, `PastKvArenaReuseCount`, and `PastKvCopiedElementCount` are diagnostic counters used by executable specs to keep this physical invariant testable.

## Pooled retained KV buffers

Batched present-KV arrays are not allocated with `new float[...]` on every decode step. `OptimumLegacyFloatDecoderBinding` owns a private `ArrayPool<float>` by default and each new `DecoderOrtCohortArena` rents one key and one value buffer per decoder layer.

`ArrayPool.Rent` may return an array whose capacity is larger than requested. The arena therefore records the logical tensor length separately, and ORT only receives `Memory<float>` covering the logical `[B, H, present, D]` element range. Physical pool capacity never changes tensor shape.

The retained-KV pool is binding-private rather than `ArrayPool<float>.Shared` so model KV is not intentionally mixed into a process-wide shared pool. Buffers are returned with `clearArray: false`; reuse stays inside the binding-owned pool. A caller may inject another `ArrayPool<float>` when it needs a different allocation/security policy or deterministic diagnostics.

Pooling follows a strict lifetime rule:

- a new arena starts with one builder reference,
- every row `DecoderOrtState` retains the arena after its `OrtValue` payload validates,
- a reusable prior arena takes a temporary execution reference while its full-batch ORT input handles exist,
- state disposal destroys row `OrtValue` handles before releasing the arena,
- full-batch ORT handles are destroyed before builder/execution references are released, and
- only the final arena release returns all rented arrays to the pool.

This prevents a pooled buffer from being returned while any `OrtValue` can still read or write it. Error paths follow the same rule: partial row states are disposed first, full ORT handles next, and the builder reference last.

The executable Optimum batch spec injects a deterministic pool and verifies that two simultaneously live frontiers require distinct physical buffers, the last sibling row release returns an arena exactly once, and a later frontier can rent those returned arrays without another physical allocation.

## Pooled transient ORT scratch

Step-local tensors have a different lifetime from retained KV and therefore use separate binding-private scratch pools. The batched decode path rents exact logical views for:

- `input_ids`,
- `position_ids`,
- `attention_mask`,
- logits, and
- packed past-key/value buffers used only when stable arena reuse is unavailable.

`ArrayPoolLease<T>` keeps the physical rental private and exposes only the requested `Memory<T>` / `Span<T>` length. A larger physical pool array therefore cannot leak spare capacity into an ONNX tensor shape.

Scratch lifetime is deliberately shorter than arena lifetime. At the end of one model step the binding disposes full ORT output handles, then input handles, then returns every scratch lease, and only then releases retained arena execution/builder references. Stable cohorts therefore rent three `long` scratch buffers and one FP32 logits buffer per physical model run; fork/partial fallback additionally rents two FP32 packed-past buffers per decoder layer. `ScratchLongRentCount` and `ScratchFloatRentCount` expose this physical behavior for diagnostics.

Scalar prefill/decode remains on the existing allocation path for now. Keeping the first scratch-pooling change limited to the batched hot path makes its lifetime independent from scalar state ownership and keeps rollback/fork behavior unchanged.

## Ownership

Prior states remain immutable. Row states have independent `OrtValue` handles even though their managed backing arrays are shared. Disposing one row does not invalidate sibling rows. `DecoderStateStore` may additionally share a complete `DecoderOrtState` across snapshots or forked branches; in that case the state object itself is disposed only after its final store owner releases it, so its arena reference remains valid for the complete transactional lifetime.

A newly produced frontier uses a different checked-out arena while the prior frontier is still an input. Once the runtime commits the new states and releases the prior states, that prior arena's buffers become eligible for pool reuse by a later decode frontier.

## Current scope

This optimization targets the legacy dense Optimum decoder-with-past FP32 binding. It is not paged KV and it does not make partial/fork fallback gathers zero-copy. Scalar decode outputs and the small managed collections used to assemble ORT name/value lists still allocate normally. Paged caches, recurrent/linear-attention state, mixed cache layouts, exporter-specific tensors, device-native allocator reuse, and explicit scheduler budgeting of transient workspace remain separate concerns.
