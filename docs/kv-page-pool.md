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
- `AvailablePages` feeds the F# scheduler's physical page budget

Scheduler admission models the partial-tail copy as a fixed `KvPageWriteOverhead` carried by each runnable sequence. For any positive grant, F# first charges that overhead and then adds normal token-boundary page growth. This preserves the distinction between a physical COW replacement and a new logical KV page: with one free page, a shared half-full tail can be copied and filled to its existing boundary; crossing that boundary requires another free page.

The scheduled runtime does not trust the earlier candidate snapshot blindly. Immediately before backend dispatch it recomputes `AdditionalKvPagesFor(tokenCount)` from the live `SequenceProcess`, so a stale schedule is rejected if fork/snapshot sharing changed after admission.

Backend inference items also carry the corresponding live `InferenceKvWriteIntent`. The intent exposes only physical write geometry (`CopyOnWritePages`, `TokensPerPage`, and `TailTokenCount`), never runtime page ids or reference counts. A default value means ordinary append/write semantics. A non-default intent means an in-place paged backend must make the shared partial tail privately writable before applying that prefill/decode item. Backends that already produce a new immutable physical state version for every inference step satisfy the contract naturally without an extra hot-path dispatch.

Runtime metadata is committed only after the backend inference call succeeds. This keeps the physical materialization boundary aligned with the existing transaction rule: a backend must not advance causal state merely because it prepared a private tail, and runtime `RecordPrefill`/`RecordDecode` remains the logical commit point after successful model work.
