# Chunked prefill and token-block KV sizing

Fission schedules prompt prefill as bounded token quanta rather than requiring an entire prompt to fit one iteration.

## Core invariants

- `ReadySequence.TokenDemand` is the remaining token demand for the current scheduling phase.
- `MaxPrefillChunkTokens` caps one prefill quantum; batch token and KV budgets may reduce the grant further.
- KV page demand is derived from `Position`, `TokenGrant`, and `TokensPerKvPage`:

  `pages(position + grant) - pages(position)`

- A decode inside an already materialized page may proceed with `AvailableKvPages = 0`. A decode at a page boundary requires one new page.
- The F# scheduler emits `KvPageGrant`; `ScheduledBatchExecutor` recomputes the grant against the live runtime position and rejects the whole batch before side effects if the values disagree.
- Scheduled prefill bindings retain the full prompt. The runtime slices the current chunk using the sequence position.
- Partial chunks keep `SequenceStatus.Prefilling`. Only a work item with `CompletesPrefill = true` transitions the sequence to `Decoding`.

## Runtime feedback

`ExecutionPlanExecutor.KvCapacity` exposes capacity, allocation, availability, and token block size. The serving/admission layer should use that snapshot when constructing the next scheduler `ResourceBudget` and `ReadySequence` values.

This keeps policy in F# and physical KV ownership in the C# runtime while making the resource contract independently checkable on both sides.
