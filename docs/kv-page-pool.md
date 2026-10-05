# KV page pool

`KvPagePool` is the runtime ownership boundary for metadata-level KV pages.

A newly materialized KV page consumes one pool slot. `Fork()` and `Snapshot()` acquire references to the same `KvPageLease`, so immutable shared history does not consume additional page capacity. The page returns to the pool only when the final sequence/snapshot reference releases its lease.

This gives Fission the lifecycle needed for transactional KV semantics:

- prefill/decode materialization consumes capacity in fixed token blocks
- fork/snapshot share immutable history through refcounts
- a write into a shared, partially filled tail page performs copy-on-write before mutation
- copy-on-write replaces the logical tail page while consuming one additional physical pool slot
- a full shared tail remains shareable; the next boundary write appends a fresh page normally
- rollback releases private pages that are no longer referenced
- runtime disposal returns all owned pages
- `AvailablePages` can feed the F# scheduler's resource budget

The metadata copy-on-write rule is deliberately independent from backend storage. Stateful backends remain responsible for preserving their physical KV/logit/decoder state through the serialized fork/snapshot/restore transaction hooks described in `backend-state-transactions.md`.

The next integration step is to surface partial-tail sharing in scheduling candidates so the F# scheduler can charge the one-page copy-on-write surcharge before dispatch. Until that contract is wired through, direct runtime execution is COW-correct, while scheduler-side page grants still model only token-boundary growth.
