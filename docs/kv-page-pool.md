# KV page pool

`KvPagePool` is the runtime ownership boundary for metadata-level KV pages.

A newly materialized KV page consumes one pool slot. `Fork()` and `Snapshot()` acquire references to the same `KvPageLease`, so shared history does not consume additional page capacity. The page returns to the pool only when the final sequence/snapshot reference releases its lease.

This gives Fission the lifecycle needed for transactional KV semantics:

- prefill/decode materialization consumes capacity
- fork/snapshot share capacity through refcounts
- rollback releases pages that are no longer referenced
- runtime disposal returns all owned pages
- `AvailablePages` can feed the F# scheduler's resource budget

The current metadata model still materializes one page per prefill/decode runtime operation. Token-block sizing and chunked prefill will replace that temporary granularity in a later milestone.
